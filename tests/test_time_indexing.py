import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import inference as inference_cli
from arc.models.arc.arc import Arc
from arc.models.arc.dinov2.vision_transformer import DinoVisionTransformer
# Imported from the module rather than the arc.training package to avoid pulling
# in sparse_tracking (and its eval.* dependency) for these tests.
from arc.training.checkpoint import (
    load_temporal_tracking_checkpoint,
    read_temporal_patch_metadata,
    save_temporal_tracking_checkpoint,
)


def _time_token_model(embed_dim=4, max_time_indices=8):
    model = DinoVisionTransformer.__new__(DinoVisionTransformer)
    nn.Module.__init__(model)
    model.max_time_indices = max_time_indices
    model.time_token = nn.Parameter(torch.arange(embed_dim, dtype=torch.float32).view(1, 1, -1))
    model.time_index_embedding = nn.Embedding(max_time_indices, embed_dim)
    nn.init.zeros_(model.time_index_embedding.weight)
    return model


def _configured_time_transformer(max_time_indices=7):
    return DinoVisionTransformer(
        img_size=14,
        patch_size=14,
        embed_dim=8,
        depth=3,
        num_heads=2,
        mlp_ratio=2,
        alt_start=2,
        has_time_token=True,
        cat_token=False,
        max_time_indices=max_time_indices,
    )


def _full_meta_arc(freeze="none", max_time_indices=32):
    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            return Arc(
                freeze=freeze,
                max_time_indices=max_time_indices,
            )
    finally:
        torch.linspace = original_linspace


class _PassThroughTimeTransformer(DinoVisionTransformer):
    """Small transformer shell that exercises the real reorder/token-insertion path."""

    def __init__(self, embed_dim=4, max_time_indices=8):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.max_time_indices = max_time_indices
        self.alt_start = 2
        self.rope_start = -1
        self.rope = None
        self.cat_token = False
        self.has_time_token = True
        self.num_register_tokens = 0
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.camera_token = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_index_embedding = nn.Embedding(max_time_indices, embed_dim)
        nn.init.zeros_(self.time_index_embedding.weight)

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S = x.shape[:2]
        return torch.zeros(B, S, 2, self.embed_dim, device=x.device, dtype=x.dtype)

    def _prepare_rope(self, B, S, H, W, device):
        return None, None

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        return x


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_time_indices = None

    def forward(self, x, ref_view_strategy="first", time_indices=None):
        self.seen_time_indices = None if time_indices is None else time_indices.detach().clone()
        B, S = x.shape[:2]
        feature_dim = 3072
        patch = torch.zeros(B, S, 1, feature_dim, device=x.device)
        camera = torch.zeros(B, S, feature_dim, device=x.device)
        time = torch.zeros(B, S, feature_dim, device=x.device)
        if time_indices is not None:
            time[..., 1536:] = time_indices.to(time.dtype).unsqueeze(-1)
        features = tuple((patch.clone(), camera.clone(), time.clone()) for _ in range(4))
        return features, []


class _FakeReconstructionHead(nn.Module):
    def forward(self, feats, H, W, patch_start_idx):
        return {}


class _FakeCameraDecoder(nn.Module):
    def forward(self, camera_tokens):
        return torch.zeros(camera_tokens.shape[0], camera_tokens.shape[1], 1)


class _FakeMotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_indices = []
        self.seen_time_tokens = []

    def forward(self, tokens, images, patch_start_idx, track_query_idx):
        self.query_indices.append(track_query_idx)
        self.seen_time_tokens.append(tokens[:, :, 1].detach().clone())
        B, S, _, C = tokens.shape
        return torch.full(
            (B, S, 2, C),
            float(track_query_idx),
            device=tokens.device,
            dtype=tokens.dtype,
        )


class _FakeTrackHead(nn.Module):
    def forward(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        frames_chunk_size,
    ):
        query_idx = aggregated_tokens_list[0][0, 0, 0, 0]
        B, S, _, H, W = images.shape
        track = query_idx.expand(B, S, H, W, 3).clone()
        confidence = query_idx.expand(B, S, H, W).clone()
        return track, confidence


class _GradBackbone(nn.Module):
    """Features that carry grad history, so head graph retention is observable."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x, ref_view_strategy="first", time_indices=None):
        B, S = x.shape[:2]
        feature_dim = 3072
        patch = torch.zeros(B, S, 1, feature_dim, device=x.device) + self.scale
        camera = torch.zeros(B, S, feature_dim, device=x.device) + self.scale
        time = torch.zeros(B, S, feature_dim, device=x.device) + self.scale
        return tuple((patch, camera, time) for _ in range(4)), []


class _GradReconstructionHead(nn.Module):
    def forward(self, feats, H, W, patch_start_idx):
        return {"depth": feats[0][0].sum(-1)}


class _GradCameraDecoder(nn.Module):
    def forward(self, camera_tokens):
        return camera_tokens.sum(-1, keepdim=True)


class _GradMotionDecoder(nn.Module):
    def forward(self, tokens, images, patch_start_idx, track_query_idx):
        return tokens[:, :, :2, :] + float(track_query_idx)


class _TinyPretrained(nn.Module):
    def __init__(self, include_time_embedding):
        super().__init__()
        shared = nn.Linear(2, 2)
        self.shared = shared
        self.shared_alias = shared
        if include_time_embedding:
            self.time_index_embedding = nn.Embedding(4, 2)
            nn.init.zeros_(self.time_index_embedding.weight)


class _TinyBackbone(nn.Module):
    def __init__(self, include_time_embedding):
        super().__init__()
        self.pretrained = _TinyPretrained(include_time_embedding)


class _TinyHubArc(Arc):
    def __init__(self, freeze="none"):
        nn.Module.__init__(self)
        self.max_time_indices = 4
        self.backbone = _TinyBackbone(include_time_embedding=True)
        self.head = nn.Linear(2, 2)
        self.cam_dec = nn.Linear(2, 2)
        self.motion_decoder = nn.Linear(2, 2)
        self.track_head = nn.Linear(2, 2)
        self.set_freeze(freeze)


class _LegacyTinyArc(nn.Module):
    def __init__(self, *, include_head=True, include_unexpected=False):
        super().__init__()
        self.backbone = _TinyBackbone(include_time_embedding=False)
        if include_head:
            self.head = nn.Linear(2, 2)
        self.cam_dec = nn.Linear(2, 2)
        self.motion_decoder = nn.Linear(2, 2)
        self.track_head = nn.Linear(2, 2)
        if include_unexpected:
            self.unexpected = nn.Parameter(torch.zeros(1))


class _TinyAliasScratch(nn.Module):
    def __init__(self):
        super().__init__()
        shared_layer_norm = nn.LayerNorm(2)
        self.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Identity(),
                    nn.Identity(),
                    shared_layer_norm,
                )
                for _ in range(4)
            ]
        )


class _TinyAliasHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scratch = _TinyAliasScratch()


class _TinyUnsharedPretrained(nn.Module):
    def __init__(self, include_time_embedding):
        super().__init__()
        self.shared = nn.Linear(2, 2)
        if include_time_embedding:
            self.time_index_embedding = nn.Embedding(4, 2)
            nn.init.zeros_(self.time_index_embedding.weight)


class _TinyUnsharedBackbone(nn.Module):
    def __init__(self, include_time_embedding):
        super().__init__()
        self.pretrained = _TinyUnsharedPretrained(include_time_embedding)


class _AliasTinyHubArc(_TinyHubArc):
    def __init__(self, freeze="none"):
        super().__init__(freeze=freeze)
        self.backbone = _TinyUnsharedBackbone(include_time_embedding=True)
        self.head = _TinyAliasHead()


class _LegacyAliasTinyArc(_LegacyTinyArc):
    def __init__(self):
        super().__init__(include_head=False)
        self.backbone = _TinyUnsharedBackbone(include_time_embedding=False)
        self.head = _TinyAliasHead()


def _arc_shell(max_time_indices=8):
    model = Arc.__new__(Arc)
    nn.Module.__init__(model)
    model.max_time_indices = max_time_indices
    return model


def _run_pass_through(model, images, time_indices=None):
    outputs, _ = model._get_intermediate_layers_not_chunked(
        images,
        n=[2],
        ref_view_strategy="middle",
        time_indices=time_indices,
    )
    return outputs[0][1]


def _save_safetensors_model(model, directory):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_model

    directory.mkdir()
    save_model(model, directory / "model.safetensors")


def _save_raw_safetensors_state(state_dict, directory):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    directory.mkdir()
    save_file(
        {
            name: tensor.detach().clone().contiguous()
            for name, tensor in state_dict.items()
        },
        directory / "model.safetensors",
    )


def test_time_embedding_size_is_configurable_and_zero_initialized():
    model = _configured_time_transformer()

    assert model.time_index_embedding.weight.shape == (7, 8)
    assert torch.count_nonzero(model.time_index_embedding.weight) == 0


def test_legacy_state_leaves_only_the_zero_time_embedding_missing():
    source = _configured_time_transformer()
    legacy_state = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if name != "time_index_embedding.weight"
    }
    restored = _configured_time_transformer()

    incompatibility = restored.load_state_dict(legacy_state, strict=False)

    assert incompatibility.missing_keys == ["time_index_embedding.weight"]
    assert incompatibility.unexpected_keys == []
    assert torch.count_nonzero(restored.time_index_embedding.weight) == 0


def test_inference_parser_preserves_legacy_defaults_and_multi_query_values():
    args = inference_cli.build_arg_parser().parse_args(
        [
            "--input",
            "frames",
            "--save",
            "output.npz",
            "--track_query_idx",
            "0",
            "12",
        ]
    )

    assert args.time_indices is None
    assert args.track_query_idx == [0, 12]
    assert args.checkpoint_dir == "Luo-Yihang/4RC"
    assert args.temporal_patch is None


def test_inference_parser_accepts_a_temporal_patch_path():
    args = inference_cli.build_arg_parser().parse_args(
        [
            "--input",
            "frames",
            "--save",
            "output.npz",
            "--temporal_patch",
            "runs/overfit/temporal_tracking.pt",
            "--track_query_idx",
            "0",
        ]
    )

    assert args.temporal_patch == "runs/overfit/temporal_tracking.pt"
    assert args.track_query_idx == [0]


def test_inference_parser_accepts_repeated_semantic_times():
    semantic_times = list(range(12)) * 2
    args = inference_cli.build_arg_parser().parse_args(
        [
            "--input",
            "frames",
            "--save",
            "output.npz",
            "--track_query_idx",
            "0",
            "12",
            "23",
            "--time_indices",
            *[str(value) for value in semantic_times],
        ]
    )

    assert args.time_indices == semantic_times
    assert args.track_query_idx == [0, 12, 23]


def test_inference_time_validation_errors_are_clear():
    with pytest.raises(TypeError, match="must be an integer"):
        inference_cli.validate_time_indices([0, 1.5])
    with pytest.raises(ValueError, match="non-negative"):
        inference_cli.validate_time_indices([0, -1])
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        inference_cli.validate_time_indices([32], max_time_indices=32)
    with pytest.raises(SystemExit):
        inference_cli.build_arg_parser().parse_args(
            [
                "--input",
                "frames",
                "--save",
                "output.npz",
                "--time_indices",
                "not-an-integer",
            ]
        )


def test_a_deliberate_grid_is_refused_rather_than_subsampled():
    """Lockstep resampling of --time_indices used to be the behaviour here.

    It kept paths and indices consistent with each other, so nothing downstream
    complained -- but the grid the caller asked to be scored on was not the grid
    that got scored, and it was still labelled as the original.
    """

    paths = [f"frame_{index:03d}.png" for index in range(48)]
    semantic_times = list(range(12)) * 4

    with pytest.raises(ValueError, match="--max_frames") as excinfo:
        inference_cli.select_input_frames(paths, semantic_times)

    message = str(excinfo.value)
    assert "48" in message
    assert "30" in message


def test_raising_the_cap_keeps_every_frame_and_its_time_index():
    paths = [f"frame_{index:03d}.png" for index in range(48)]
    semantic_times = list(range(12)) * 4

    selected_paths, selected_times = inference_cli.select_input_frames(
        paths,
        semantic_times,
        max_frames=96,
    )

    assert selected_paths == paths
    assert selected_times == semantic_times


def test_paths_without_time_indices_still_subsample_to_the_cap():
    paths = [f"frame_{index:03d}.png" for index in range(48)]
    expected_positions = np.linspace(0, 47, 30, dtype=int)

    selected_paths, selected_times = inference_cli.select_input_frames(paths)

    assert selected_paths == [paths[index] for index in expected_positions]
    assert selected_times is None


def test_a_short_video_may_not_manufacture_synchronized_observations():
    """The duplicate case is the worse one, and it fires below the cap.

    ``np.linspace(0, 3, 30)`` repeats indices, and repeated time values are
    exactly how --time_indices marks observations as simultaneous.  Carrying
    them along in lockstep would invent synchronization the footage does not
    contain, while leaving every 1:1 check downstream satisfied.
    """

    paths = [f"frame_{index:03d}.png" for index in range(4)]

    with pytest.raises(ValueError, match="drop or duplicate"):
        inference_cli.select_input_frames(
            paths,
            [0, 1, 2, 3],
            is_video=True,
        )


def test_a_video_selection_that_is_the_identity_stays_legal():
    """No spurious refusal when linspace is exactly arange.

    A 30-frame video at the default cap selects every frame in order, so the
    grid survives untouched and there is nothing to refuse.
    """

    paths = [f"frame_{index:03d}.png" for index in range(30)]
    semantic_times = list(range(15)) * 2

    selected_paths, selected_times = inference_cli.select_input_frames(
        paths,
        semantic_times,
        is_video=True,
    )

    assert selected_paths == paths
    assert selected_times == semantic_times


def test_a_video_without_time_indices_is_resampled_as_before():
    paths = [f"frame_{index:03d}.png" for index in range(12)]
    expected_positions = np.linspace(0, 11, 30, dtype=int)

    selected_paths, selected_times = inference_cli.select_input_frames(
        paths,
        is_video=True,
    )

    assert selected_paths == [paths[index] for index in expected_positions]
    assert len(selected_paths) == 30
    assert selected_times is None


def test_inference_parser_exposes_the_frame_cap():
    parser = inference_cli.build_arg_parser()

    default_args = parser.parse_args(["--input", "frames", "--save", "output.npz"])
    assert default_args.max_frames == 30

    raised_args = parser.parse_args(
        ["--input", "frames", "--save", "output.npz", "--max_frames", "96"]
    )
    assert raised_args.max_frames == 96

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--input", "frames", "--save", "output.npz", "--max_frames", "not-an-integer"]
        )


class _FrameSelectionReached(Exception):
    """Stops main() at the call under test, before it imports the model stack."""


def test_main_forwards_the_frame_cap_to_the_selection(monkeypatch):
    """The one seam the tests above cannot see.

    Both the parser default and select_input_frames' own default are 30, so a
    main() that never passed --max_frames through would satisfy every other test
    in this file while silently ignoring the flag at the values it exists for.
    """

    recorded = {}

    def fake_collect_images(input_path):
        return [f"frame_{index:03d}.png" for index in range(48)], False

    def spy_select_input_frames(paths, time_indices=None, *, is_video=False, max_frames=30):
        recorded["max_frames"] = max_frames
        raise _FrameSelectionReached

    monkeypatch.setattr(inference_cli, "collect_images", fake_collect_images)
    monkeypatch.setattr(inference_cli, "select_input_frames", spy_select_input_frames)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--input",
            "frames",
            "--save",
            "output.npz",
            "--max_frames",
            "96",
        ],
    )

    with pytest.raises(_FrameSelectionReached):
        inference_cli.main()

    assert recorded["max_frames"] == 96


def test_main_rejects_a_non_positive_frame_cap(monkeypatch):
    monkeypatch.setattr(
        inference_cli,
        "collect_images",
        lambda input_path: pytest.fail("a rejected cap must not reach frame collection"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--input",
            "frames",
            "--save",
            "output.npz",
            "--max_frames",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        inference_cli.main()


def test_time_count_is_checked_before_subsampling_and_after_loading():
    paths = [f"frame_{index:03d}.png" for index in range(31)]
    with pytest.raises(ValueError, match="expected 31, got 30"):
        inference_cli.select_input_frames(paths, list(range(30)))

    imgs = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(3)]
    with pytest.raises(ValueError, match="loaded 3 images but have 2 indices"):
        inference_cli.attach_frame_metadata(
            imgs,
            track_query_idx=[0],
            time_indices=[0, 1],
        )


def test_24_observations_receive_12_repeated_times_without_camera_ids():
    semantic_times = list(range(12)) * 2
    imgs = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(24)]

    query_indices = inference_cli.attach_frame_metadata(
        imgs,
        track_query_idx=[0, 12, 23],
        time_indices=semantic_times,
    )

    assert query_indices == [0, 12, 23]
    for time_idx in range(12):
        for view_idx in (time_idx, time_idx + 12):
            assert torch.equal(
                imgs[view_idx]["time_index"],
                torch.tensor([time_idx]),
            )
            assert torch.equal(
                imgs[view_idx]["track_query_idx"],
                torch.tensor([0, 12, 23]),
            )

    _, normalized_queries, normalized_times = _arc_shell(
        max_time_indices=32
    )._preprocess_input(imgs)
    assert normalized_queries == [0, 12, 23]
    assert normalized_times.shape == (1, 24)
    assert torch.equal(normalized_times[:, :12], normalized_times[:, 12:])


def test_absent_inference_times_do_not_add_model_metadata():
    imgs = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(3)]

    inference_cli.attach_frame_metadata(
        imgs,
        track_query_idx=[0, 2],
        time_indices=None,
    )

    assert all("time_index" not in view for view in imgs)
    _, query_indices, time_indices = _arc_shell()._preprocess_input(imgs)
    assert query_indices == [0, 2]
    assert time_indices is None


def test_missing_time_metadata_keeps_legacy_path_and_multi_query_input():
    model = _arc_shell()
    views = [
        {
            "img": torch.zeros(1, 3, 2, 2),
            "track_query_idx": torch.tensor([0, 2]),
        }
        for _ in range(3)
    ]

    _, query_indices, time_indices = model._preprocess_input(views)

    assert query_indices == [0, 2]
    assert time_indices is None


def test_zero_initialized_time_indices_match_legacy_tokens_and_outputs():
    token_model = _time_token_model()
    indices = torch.tensor([[0, 1, 0]])
    legacy_tokens = token_model._prepare_time_tokens(1, 3, None)
    indexed_tokens = token_model._prepare_time_tokens(1, 3, indices)
    assert torch.equal(indexed_tokens, legacy_tokens)

    images = torch.zeros(1, 4, 3, 2, 2)
    transformer = _PassThroughTimeTransformer()
    legacy_output = _run_pass_through(transformer, images)
    indexed_output = _run_pass_through(
        transformer,
        images,
        torch.tensor([[0, 1, 0, 1]]),
    )
    assert torch.equal(indexed_output, legacy_output)

    with torch.no_grad():
        token_model.time_index_embedding.weight.fill_(100)
    assert torch.equal(token_model._prepare_time_tokens(1, 3, None), legacy_tokens)


def test_time_embedding_is_trainable_through_the_real_transformer():
    """Guard the premise of the whole temporal-tracking stage.

    A stubbed backbone cannot show this: it must run the real
    DinoVisionTransformer so that a `.detach()` on the embedding lookup, or any
    change that drops the embedding off the autograd path, fails here rather
    than silently producing a flat loss curve on the cluster.
    """

    model = _configured_time_transformer(max_time_indices=7)
    with torch.no_grad():
        for index in range(model.max_time_indices):
            model.time_index_embedding.weight[index].fill_(0.1 * (index + 1))

    images = torch.zeros(1, 4, 3, 14, 14)
    time_indices = torch.tensor([[0, 1, 0, 1]])

    legacy_output = _run_pass_through(model, images)
    indexed_output = _run_pass_through(model, images, time_indices)

    # A non-zero embedding must actually reach the output.
    assert not torch.equal(indexed_output, legacy_output)

    # ...and it must be on the autograd path, not merely read as a constant.
    model.zero_grad(set_to_none=True)
    indexed_output.sum().backward()
    gradient = model.time_index_embedding.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0

    # Only the rows named by time_indices may receive gradient.
    per_row = gradient.abs().sum(dim=1)
    used_rows = sorted(set(time_indices.flatten().tolist()))
    for row in range(model.max_time_indices):
        if row in used_rows:
            assert per_row[row] > 0, f"row {row} is used but got no gradient"
        else:
            assert per_row[row] == 0, f"row {row} is unused but got gradient"


def test_optimizer_step_leaves_frozen_parameter_values_untouched():
    """requires_grad and a missing .grad are not proof that a weight held still."""

    model = _TinyHubArc(freeze="temporal_tracking")
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert frozen_before and trainable_before

    # Built the same way overfit_temporal_tracking.py builds it.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in trainable).backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(
                parameter.detach(),
                frozen_before[name],
                rtol=0,
                atol=0,
                msg=f"frozen parameter {name} moved during an optimizer step",
            )
    for name, parameter in model.named_parameters():
        if name in trainable_before:
            assert not torch.equal(parameter.detach(), trainable_before[name]), (
                f"trainable parameter {name} did not move"
            )


def test_synchronized_views_share_an_embedding_and_different_times_do_not():
    model = _time_token_model()
    with torch.no_grad():
        model.time_index_embedding.weight[2].fill_(2)
        model.time_index_embedding.weight[5].fill_(5)

    tokens = model._prepare_time_tokens(1, 3, torch.tensor([[2, 5, 2]])).squeeze(2)

    assert torch.equal(tokens[:, 0], tokens[:, 2])
    assert not torch.equal(tokens[:, 0], tokens[:, 1])
    assert torch.equal(
        tokens[0, 0] - model.time_token[0, 0],
        torch.full((4,), 2.0),
    )
    assert torch.equal(
        tokens[0, 1] - model.time_token[0, 0],
        torch.full((4,), 5.0),
    )


def test_camera_major_24_observation_layout_reuses_12_embedding_rows():
    model = _time_token_model(max_time_indices=12)
    with torch.no_grad():
        for index in range(model.max_time_indices):
            model.time_index_embedding.weight[index].fill_(index)

    semantic_times = torch.tensor([list(range(12)) * 2])
    tokens = model._prepare_time_tokens(1, 24, semantic_times)

    assert tokens.shape == (1, 24, 1, 4)
    assert torch.equal(tokens[:, :12], tokens[:, 12:])
    assert not torch.equal(tokens[:, 0], tokens[:, 1])


def test_reference_reordering_preserves_original_frame_to_time_mapping():
    model = _PassThroughTimeTransformer()
    with torch.no_grad():
        for index in range(model.max_time_indices):
            model.time_index_embedding.weight[index].fill_(index)

    images = torch.zeros(1, 4, 3, 2, 2)
    time_indices = torch.tensor([[3, 1, 2, 0]])
    output = _run_pass_through(model, images, time_indices)

    restored_time_values = output[0, :, 1, 0]
    assert torch.equal(restored_time_values, time_indices[0].to(output.dtype))


def test_time_metadata_is_all_or_none_and_validated():
    model = _arc_shell(max_time_indices=6)
    image = torch.zeros(2, 3, 2, 2)

    partial_views = [
        {"img": image, "time_index": 0},
        {"img": image},
    ]
    with pytest.raises(ValueError, match="every view"):
        model._preprocess_input(partial_views)

    float_views = [
        {"img": image, "time_index": 0.5},
        {"img": image, "time_index": 1},
    ]
    with pytest.raises(TypeError, match="integer"):
        model._preprocess_input(float_views)

    out_of_bounds_views = [
        {"img": image, "time_index": [0, 1]},
        {"img": image, "time_index": [6, 2]},
    ]
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        model._preprocess_input(out_of_bounds_views)


def test_temporal_tracking_freeze_is_exact_and_reversible():
    model = _full_meta_arc()
    assert all(parameter.requires_grad for parameter in model.parameters())

    model.set_freeze("temporal_tracking")
    report = model.get_trainable_parameter_report()
    expected_names = {
        "backbone.pretrained.time_index_embedding.weight",
        *{
            f"motion_decoder.{name}"
            for name, _ in model.motion_decoder.named_parameters()
        },
        *{
            f"track_head.{name}"
            for name, _ in model.track_head.named_parameters()
        },
    }

    assert {name for name, _ in report["parameters"]} == expected_names
    assert report["tensor_count"] == 231
    assert report["parameter_count"] == 314_600_740
    assert not any(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(parameter.requires_grad for parameter in model.cam_dec.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.backbone.named_parameters()
        if name != "pretrained.time_index_embedding.weight"
    )

    model.set_freeze("none")
    assert all(parameter.requires_grad for parameter in model.parameters())

    model.set_freeze("temporal_tracking")
    assert model.get_trainable_parameter_report() == report

    with pytest.raises(ValueError, match="Unknown freeze mode"):
        model.set_freeze("unknown")
    assert model.get_trainable_parameter_report() == report


def test_from_pretrained_accepts_only_the_legacy_time_embedding_gap(tmp_path):
    legacy_dir = tmp_path / "legacy"
    _save_safetensors_model(_LegacyTinyArc(), legacy_dir)

    restored = _TinyHubArc.from_pretrained(str(legacy_dir))

    assert torch.count_nonzero(
        restored.backbone.pretrained.time_index_embedding.weight
    ) == 0

    with pytest.raises(RuntimeError, match="time_index_embedding"):
        _TinyHubArc.from_pretrained(str(legacy_dir), strict=True)


def test_legacy_load_records_the_consumed_time_embedding_gap(tmp_path):
    """A zero-filled time embedding must be reportable, not silent.

    ``inference.py`` uses this to warn that ``--time_indices`` cannot affect the
    output, which is otherwise indistinguishable from a finetune that did not help.
    """

    legacy_dir = tmp_path / "legacy"
    _save_safetensors_model(_LegacyTinyArc(), legacy_dir)

    restored = _TinyHubArc.from_pretrained(str(legacy_dir))

    assert restored.consumed_legacy_missing_keys == frozenset(
        Arc.LEGACY_CHECKPOINT_MISSING_KEYS
    )
    assert inference_cli.TIME_EMBEDDING_KEY in restored.consumed_legacy_missing_keys

    complete_dir = tmp_path / "complete"
    source = _TinyHubArc()
    with torch.no_grad():
        source.backbone.pretrained.time_index_embedding.weight.fill_(0.5)
    _save_safetensors_model(source, complete_dir)

    reloaded = _TinyHubArc.from_pretrained(str(complete_dir))

    assert reloaded.consumed_legacy_missing_keys == frozenset()


def test_temporal_patch_restores_a_nonzero_time_embedding(tmp_path):
    """The overfit's output must be loadable back onto a base checkpoint.

    This is the mechanism ``inference.py --temporal_patch`` drives.
    """

    legacy_dir = tmp_path / "legacy"
    _save_safetensors_model(_LegacyTinyArc(), legacy_dir)

    trained = _TinyHubArc(freeze="temporal_tracking")
    with torch.no_grad():
        trained.backbone.pretrained.time_index_embedding.weight.fill_(1.75)
        trained.motion_decoder.weight.fill_(0.25)
    patch = save_temporal_tracking_checkpoint(
        trained,
        tmp_path / "temporal_tracking.pt",
    )

    restored = _TinyHubArc.from_pretrained(str(legacy_dir))
    assert torch.count_nonzero(
        restored.backbone.pretrained.time_index_embedding.weight
    ) == 0

    restored.set_freeze("temporal_tracking")
    load_temporal_tracking_checkpoint(restored, patch)

    torch.testing.assert_close(
        restored.backbone.pretrained.time_index_embedding.weight,
        trained.backbone.pretrained.time_index_embedding.weight,
    )
    torch.testing.assert_close(
        restored.motion_decoder.weight,
        trained.motion_decoder.weight,
    )


def test_from_pretrained_accepts_identical_known_safetensor_aliases(tmp_path):
    legacy_dir = tmp_path / "legacy-aliases"
    source = _LegacyAliasTinyArc()
    with torch.no_grad():
        shared_layer_norm = source.head.scratch.output_conv2_aux[0][2]
        shared_layer_norm.weight.copy_(torch.tensor([2.0, 3.0]))
        shared_layer_norm.bias.copy_(torch.tensor([-1.0, 4.0]))
    legacy_state = source.state_dict()
    _save_raw_safetensors_state(legacy_state, legacy_dir)

    assert set(Arc.LEGACY_SAFETENSOR_ALIASES).issubset(legacy_state)
    restored = _AliasTinyHubArc.from_pretrained(str(legacy_dir))

    restored_layer_norm = restored.head.scratch.output_conv2_aux[0][2]
    assert torch.equal(
        restored_layer_norm.weight,
        torch.tensor([2.0, 3.0]),
    )
    assert torch.equal(
        restored_layer_norm.bias,
        torch.tensor([-1.0, 4.0]),
    )
    assert torch.count_nonzero(
        restored.backbone.pretrained.time_index_embedding.weight
    ) == 0

    with pytest.raises(RuntimeError, match="time_index_embedding"):
        _AliasTinyHubArc.from_pretrained(str(legacy_dir), strict=True)


def test_strict_loading_rejects_known_safetensor_aliases(tmp_path):
    checkpoint_dir = tmp_path / "complete-with-aliases"
    _save_raw_safetensors_state(
        _AliasTinyHubArc().state_dict(),
        checkpoint_dir,
    )

    with pytest.raises(
        RuntimeError,
        match=r"output_conv2_aux\.1\.2",
    ):
        _AliasTinyHubArc.from_pretrained(
            str(checkpoint_dir),
            strict=True,
        )


def test_from_pretrained_rejects_conflicting_known_safetensor_alias(tmp_path):
    checkpoint_dir = tmp_path / "conflicting-alias"
    state = {
        name: tensor.detach().clone()
        for name, tensor in _LegacyAliasTinyArc().state_dict().items()
    }
    conflicting_alias = "head.scratch.output_conv2_aux.1.2.weight"
    state[conflicting_alias].add_(1)
    _save_raw_safetensors_state(state, checkpoint_dir)

    with pytest.raises(
        RuntimeError,
        match=rf"{conflicting_alias}.*conflicts",
    ):
        _AliasTinyHubArc.from_pretrained(str(checkpoint_dir))


def test_known_safetensor_aliases_do_not_allow_arbitrary_unexpected_keys(tmp_path):
    checkpoint_dir = tmp_path / "aliases-with-unexpected"
    state = {
        name: tensor.detach().clone()
        for name, tensor in _LegacyAliasTinyArc().state_dict().items()
    }
    state["arbitrary.unexpected"] = torch.zeros(1)
    _save_raw_safetensors_state(state, checkpoint_dir)

    with pytest.raises(RuntimeError, match="arbitrary.unexpected"):
        _AliasTinyHubArc.from_pretrained(str(checkpoint_dir))


def test_from_pretrained_strictly_loads_a_complete_time_indexed_state(tmp_path):
    checkpoint_dir = tmp_path / "complete"
    source = _TinyHubArc()
    with torch.no_grad():
        source.backbone.pretrained.time_index_embedding.weight.fill_(3)
    _save_safetensors_model(source, checkpoint_dir)

    restored = _TinyHubArc.from_pretrained(
        str(checkpoint_dir),
        strict=True,
    )

    assert torch.equal(
        restored.backbone.pretrained.time_index_embedding.weight,
        source.backbone.pretrained.time_index_embedding.weight,
    )


def test_from_pretrained_rejects_time_embedding_shape_mismatch(tmp_path):
    checkpoint_dir = tmp_path / "wrong-shape"
    source = _TinyHubArc()
    source.backbone.pretrained.time_index_embedding = nn.Embedding(3, 2)
    _save_safetensors_model(source, checkpoint_dir)

    with pytest.raises(RuntimeError, match="size mismatch"):
        _TinyHubArc.from_pretrained(str(checkpoint_dir))


def test_from_pretrained_rejects_additional_missing_or_unexpected_keys(tmp_path):
    missing_dir = tmp_path / "missing"
    _save_safetensors_model(
        _LegacyTinyArc(include_head=False),
        missing_dir,
    )
    with pytest.raises(RuntimeError, match="head"):
        _TinyHubArc.from_pretrained(str(missing_dir))

    unexpected_dir = tmp_path / "unexpected"
    _save_safetensors_model(
        _LegacyTinyArc(include_unexpected=True),
        unexpected_dir,
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        _TinyHubArc.from_pretrained(str(unexpected_dir))


def test_pickle_loader_accepts_only_the_legacy_time_embedding_gap(tmp_path):
    checkpoint_path = tmp_path / "legacy.bin"
    torch.save(_LegacyTinyArc().state_dict(), checkpoint_path)

    restored = Arc._load_as_pickle(
        _TinyHubArc(),
        str(checkpoint_path),
        "cpu",
        strict=False,
    )

    assert torch.count_nonzero(
        restored.backbone.pretrained.time_index_embedding.weight
    ) == 0


def test_time_indices_reach_motion_decoder_without_changing_multi_query_semantics():
    model = _arc_shell(max_time_indices=32)
    model.backbone = _FakeBackbone()
    model.head = _FakeReconstructionHead()
    model.cam_dec = _FakeCameraDecoder()
    model.motion_decoder = _FakeMotionDecoder()
    model.track_head = _FakeTrackHead()

    time_indices = torch.tensor([list(range(12)) * 2])
    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(24)]
    inference_cli.attach_frame_metadata(
        views,
        track_query_idx=[0, 12, 23],
        time_indices=time_indices[0].tolist(),
    )
    output = model(views, force_no_output_conversion=True)

    assert torch.equal(model.backbone.seen_time_indices, time_indices)
    assert model.motion_decoder.query_indices == [
        *([0] * 4),
        *([12] * 4),
        *([23] * 4),
    ]
    assert len(model.motion_decoder.seen_time_tokens) == 12
    for seen_time_tokens in model.motion_decoder.seen_time_tokens:
        assert torch.equal(
            seen_time_tokens[:, :, 0],
            time_indices.to(seen_time_tokens.dtype),
        )

    assert output["track"].shape == (1, 24, 2, 2, 3)
    assert output["track_multi"].shape == (1, 3, 24, 2, 2, 3)
    assert output["conf_track_multi"].shape == (1, 3, 24, 2, 2)
    assert torch.equal(output["track_query_idx"], torch.tensor([0, 12, 23]))
    assert torch.count_nonzero(output["track_multi"][:, 0]) == 0
    assert torch.all(output["track_multi"][:, 1] == 12)
    assert torch.all(output["track_multi"][:, 2] == 23)


def test_forward_recomposes_from_its_three_public_pieces():
    """``_forward`` must equal encode + reconstruct + per-query track.

    Supervising several anchors drives those three directly so the encoder is
    paid for once and only one track-head graph is alive at a time.  That is
    only sound if composing them by hand reproduces ``_forward`` exactly, which
    is what keeps ``inference.py`` and ``app.py`` unaffected by the split.
    """

    def build():
        model = _arc_shell(max_time_indices=32)
        model.backbone = _FakeBackbone()
        model.head = _FakeReconstructionHead()
        model.cam_dec = _FakeCameraDecoder()
        model.motion_decoder = _FakeMotionDecoder()
        model.track_head = _FakeTrackHead()
        return model

    queries = [0, 3, 7]
    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(8)]
    inference_cli.attach_frame_metadata(
        views,
        track_query_idx=queries,
        time_indices=list(range(8)),
    )

    whole = build()
    expected = whole(views, force_no_output_conversion=True)

    piecewise = build()
    images, track_query_idx, time_indices = piecewise._preprocess_input(views)
    feats = piecewise.encode_features(images, time_indices=time_indices)
    actual = piecewise.reconstruct(feats, images)
    tracks = [
        piecewise.track_for_query(feats, images, query_idx)
        for query_idx in track_query_idx
    ]
    actual["track"] = tracks[0][0]
    actual["conf_track"] = tracks[0][1]
    actual["track_multi"] = torch.stack([track for track, _ in tracks], dim=1)
    actual["conf_track_multi"] = torch.stack([conf for _, conf in tracks], dim=1)
    actual["track_query_idx"] = torch.tensor(track_query_idx)

    assert set(actual) == set(expected)
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(actual[key], value), key
        elif isinstance(value, list):
            assert len(actual[key]) == len(value), key
            for left, right in zip(actual[key], value):
                assert torch.equal(left, right), key
        else:
            assert actual[key] == value, key


def test_temporal_tracking_drops_the_frozen_reconstruction_graph():
    """Frozen depth/camera heads must not retain a backward graph.

    Their outputs are consumed only through detached paths, so the retained
    dual-pyramid DPT graph is about 1.2 GB per observation of pure waste. Values
    must be unchanged and the tracking branch must stay differentiable.
    """

    def build(freeze):
        model = _arc_shell(max_time_indices=32)
        model.backbone = _GradBackbone()
        model.head = _GradReconstructionHead()
        model.cam_dec = _GradCameraDecoder()
        model.motion_decoder = _GradMotionDecoder()
        model.track_head = _FakeTrackHead()
        model.freeze = freeze
        return model

    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(4)]
    inference_cli.attach_frame_metadata(views, track_query_idx=[0])

    trainable = build("none")
    frozen = build("temporal_tracking")
    unfrozen_output = trainable(views, force_no_output_conversion=True)
    frozen_output = frozen(views, force_no_output_conversion=True)

    # Same numbers either way.
    torch.testing.assert_close(frozen_output["depth"], unfrozen_output["depth"])
    torch.testing.assert_close(
        frozen_output["pose_enc"],
        unfrozen_output["pose_enc"],
    )

    # But no graph is kept for the frozen reconstruction branch.
    assert unfrozen_output["depth"].requires_grad
    assert unfrozen_output["pose_enc"].requires_grad
    assert not frozen_output["depth"].requires_grad
    assert not frozen_output["pose_enc"].requires_grad

    # The tracking branch stays differentiable, and gradient still reaches the
    # backbone through it.
    assert unfrozen_output["track_multi"].requires_grad
    assert frozen_output["track_multi"].requires_grad
    frozen_output["track_multi"].sum().backward()
    assert frozen.backbone.scale.grad is not None
    assert frozen.backbone.scale.grad.abs().item() > 0


def test_npz_keeps_24_observations_without_serializing_time_metadata(tmp_path):
    output_dict = {
        "preds": [
            {"value": torch.tensor([view_idx])}
            for view_idx in range(24)
        ],
        "views": [
            {
                "img": torch.tensor([view_idx]),
                "time_index": torch.tensor([view_idx % 12]),
            }
            for view_idx in range(24)
        ],
        "refine_track_visual": False,
    }
    output_path = tmp_path / "timed-output.npz"

    inference_cli.save_npz(output_dict, output_path)

    with np.load(output_path) as saved:
        assert int(saved["n_frames"]) == 24
        assert all("time_index" not in key for key in saved.files)
        assert sum(key.startswith("pred_") for key in saved.files) == 24
        assert sum(key.startswith("view_") for key in saved.files) == 24


def test_released_checkpoint_header_has_only_the_time_embedding_gap():
    header_path = os.environ.get("FOUR_RC_RELEASED_SAFETENSORS_HEADER")
    if not header_path:
        pytest.skip("Set FOUR_RC_RELEASED_SAFETENSORS_HEADER to the downloaded header file")

    with open(header_path, "rb") as checkpoint_header:
        header_length = int.from_bytes(
            checkpoint_header.read(8),
            byteorder="little",
        )
        released_state = json.loads(
            checkpoint_header.read(header_length)
        )
    released_state.pop("__metadata__", None)

    model = _full_meta_arc()
    current_state = model.state_dict()
    expected_missing = set(Arc.LEGACY_CHECKPOINT_MISSING_KEYS)

    assert set(current_state) - set(released_state) == expected_missing
    assert set(released_state) - set(current_state) == set()
    for name, metadata in released_state.items():
        assert list(current_state[name].shape) == metadata["shape"]
        assert metadata["dtype"] == "F32"
        assert current_state[name].dtype == torch.float32


def test_released_checkpoint_load():
    checkpoint_dir = os.environ.get("FOUR_RC_RELEASED_CHECKPOINT_DIR")
    if not checkpoint_dir:
        pytest.skip(
            "Set FOUR_RC_RELEASED_CHECKPOINT_DIR to a directory containing model.safetensors"
        )

    model = Arc.from_pretrained(
        checkpoint_dir,
        map_location="cpu",
    )

    assert torch.count_nonzero(
        model.backbone.pretrained.time_index_embedding.weight
    ) == 0


# ------------------------------------------------------------------------------
# temporal_tracking_global_attention: the freeze mode that can learn fusion
# ------------------------------------------------------------------------------


class _GlobalAttnTinyPretrained(nn.Module):
    """Minimal encoder shape the new freeze mode needs: blocks plus alt_start."""

    def __init__(self, alt_start=1, block_count=4):
        super().__init__()
        self.alt_start = alt_start
        self.blocks = nn.ModuleList(
            [nn.Linear(2, 2) for _ in range(block_count)]
        )
        self.time_index_embedding = nn.Embedding(4, 2)
        nn.init.zeros_(self.time_index_embedding.weight)


class _GlobalAttnTinyBackbone(nn.Module):
    def __init__(self, alt_start=1):
        super().__init__()
        self.pretrained = _GlobalAttnTinyPretrained(alt_start=alt_start)


class _GlobalAttnTinyArc(Arc):
    def __init__(self, freeze="none", alt_start=1):
        nn.Module.__init__(self)
        self.max_time_indices = 4
        self.backbone = _GlobalAttnTinyBackbone(alt_start=alt_start)
        self.head = nn.Linear(2, 2)
        self.cam_dec = nn.Linear(2, 2)
        self.motion_decoder = nn.Linear(2, 2)
        self.track_head = nn.Linear(2, 2)
        self.set_freeze(freeze)


def test_global_attention_freeze_unfreezes_only_odd_blocks_from_alt_start():
    model = _GlobalAttnTinyArc(freeze="temporal_tracking_global_attention")

    trainable_blocks = {
        index
        for index, block in enumerate(model.backbone.pretrained.blocks)
        if any(parameter.requires_grad for parameter in block.parameters())
    }
    assert trainable_blocks == {1, 3}
    for index in (0, 2):
        assert not any(
            parameter.requires_grad
            for parameter in model.backbone.pretrained.blocks[index].parameters()
        )
    assert model.backbone.pretrained.time_index_embedding.weight.requires_grad
    assert model.motion_decoder.weight.requires_grad
    assert not model.head.weight.requires_grad
    assert not model.cam_dec.weight.requires_grad

    # Reversible in both directions.
    model.set_freeze("temporal_tracking")
    assert not any(
        parameter.requires_grad
        for block in model.backbone.pretrained.blocks
        for parameter in block.parameters()
    )
    model.set_freeze("none")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_global_attention_freeze_requires_alternating_attention():
    with pytest.raises(ValueError, match="alt_start"):
        _GlobalAttnTinyArc(
            freeze="temporal_tracking_global_attention",
            alt_start=-1,
        )


def test_global_attention_freeze_is_exact_on_the_full_model():
    """Pin the exact trainable set of the new mode on the real 1.5B Arc.

    The counts are also asserted at runtime by overfit_temporal_tracking.py's
    EXPECTED_TRAINABLE_SETS; the two must agree.
    """

    model = _full_meta_arc()
    model.set_freeze("temporal_tracking_global_attention")
    report = model.get_trainable_parameter_report()

    encoder = model.backbone.pretrained
    expected_names = {
        "backbone.pretrained.time_index_embedding.weight",
        *{
            f"motion_decoder.{name}"
            for name, _ in model.motion_decoder.named_parameters()
        },
        *{
            f"track_head.{name}"
            for name, _ in model.track_head.named_parameters()
        },
        *{
            f"backbone.pretrained.blocks.{index}.{name}"
            for index in range(13, 40, 2)
            for name, _ in encoder.blocks[index].named_parameters()
        },
    }
    assert {name for name, _ in report["parameters"]} == expected_names
    assert report["tensor_count"] == 483
    assert report["parameter_count"] == 711_317_284

    # Local blocks, everything below alt_start, and the frozen heads stay put.
    for index in (0, 12, 14, 38):
        assert not any(
            parameter.requires_grad
            for parameter in encoder.blocks[index].parameters()
        )
    assert not any(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.cam_dec.parameters()
    )

    model.set_freeze("temporal_tracking")
    narrowed = model.get_trainable_parameter_report()
    assert narrowed["tensor_count"] == 231
    assert narrowed["parameter_count"] == 314_600_740


def test_global_attention_mode_backward_reaches_all_unfrozen_blocks():
    """Training-path guard for the new mode on a real small encoder.

    Runs the actual DinoVisionTransformer in train mode, so the global blocks
    go through torch.utils.checkpoint; every parameter of every unfrozen block
    must come back with a finite gradient, and every frozen block with none.
    """

    torch.manual_seed(0)
    encoder = DinoVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=8,
        depth=6,
        num_heads=2,
        ffn_layer="mlp",
        alt_start=3,
        qknorm_start=3,
        rope_start=3,
        cat_token=True,
        has_time_token=True,
        max_time_indices=4,
    )
    with torch.no_grad():
        encoder.camera_token.normal_()
        encoder.time_token.normal_()
        encoder.pos_embed.normal_(std=0.02)
    encoder.reinitialize_time_index_embedding(
        "orthogonal",
        scale=0.1,
        generator=torch.Generator().manual_seed(0),
    )

    class _EncoderBackbone(nn.Module):
        def __init__(self, pretrained):
            super().__init__()
            self.pretrained = pretrained

    model = _arc_shell(max_time_indices=4)
    model.backbone = _EncoderBackbone(encoder)
    model.head = nn.Linear(2, 2)
    model.cam_dec = nn.Linear(2, 2)
    model.motion_decoder = nn.Linear(2, 2)
    model.track_head = nn.Linear(2, 2)
    model.set_freeze("temporal_tracking_global_attention")

    encoder.train()
    images = torch.randn(1, 4, 3, 28, 28)
    outputs, _ = encoder._get_intermediate_layers_not_chunked(
        images,
        n=[5],
        ref_view_strategy="first",
        time_indices=torch.tensor([[0, 1, 0, 1]]),
    )
    outputs[0][1].sum().backward()

    global_blocks = {3, 5}
    for index, block in enumerate(encoder.blocks):
        for name, parameter in block.named_parameters():
            if index in global_blocks:
                assert parameter.requires_grad, f"blocks.{index}.{name}"
                assert parameter.grad is not None, f"blocks.{index}.{name}"
                assert torch.isfinite(parameter.grad).all(), f"blocks.{index}.{name}"
            else:
                assert not parameter.requires_grad, f"blocks.{index}.{name}"
                assert parameter.grad is None, f"blocks.{index}.{name}"
    assert encoder.time_index_embedding.weight.grad is not None


# ------------------------------------------------------------------------------
# reinitialize_time_index_embedding
# ------------------------------------------------------------------------------


def test_reinitialize_orthogonal_writes_scaled_orthonormal_rows():
    model = _configured_time_transformer()  # 7 rows, embed_dim 8

    model.reinitialize_time_index_embedding(
        "orthogonal",
        scale=0.25,
        generator=torch.Generator().manual_seed(3),
    )

    weight = model.time_index_embedding.weight
    expected_row_norm = 0.25 * model.time_token.detach().float().norm()
    torch.testing.assert_close(
        weight.norm(dim=1),
        expected_row_norm.expand(7),
        atol=1e-5,
        rtol=1e-5,
    )
    gram = weight @ weight.T
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    assert off_diagonal.abs().max() < 1e-4
    assert weight.requires_grad


def test_reinitialize_is_deterministic_under_a_seed_and_leaves_the_token_alone():
    reference = _configured_time_transformer()
    token_before = reference.time_token.detach().clone()

    reference.reinitialize_time_index_embedding(
        "orthogonal", generator=torch.Generator().manual_seed(11)
    )
    first = reference.time_index_embedding.weight.detach().clone()
    assert torch.equal(reference.time_token.detach(), token_before)

    repeat = _configured_time_transformer()
    with torch.no_grad():
        # Same time_token, so the same seed must reproduce the same rows.
        repeat.time_token.copy_(token_before)
    repeat.reinitialize_time_index_embedding(
        "orthogonal", generator=torch.Generator().manual_seed(11)
    )
    assert torch.equal(repeat.time_index_embedding.weight.detach(), first)

    repeat.reinitialize_time_index_embedding(
        "orthogonal", generator=torch.Generator().manual_seed(12)
    )
    assert not torch.equal(repeat.time_index_embedding.weight.detach(), first)


def test_reinitialize_zeros_restores_the_constructor_state():
    model = _configured_time_transformer()
    model.reinitialize_time_index_embedding(
        "orthogonal", generator=torch.Generator().manual_seed(0)
    )
    assert torch.count_nonzero(model.time_index_embedding.weight) > 0

    model.reinitialize_time_index_embedding("zeros")

    assert torch.count_nonzero(model.time_index_embedding.weight) == 0


def test_reinitialize_keeps_the_legacy_no_index_path_bit_identical():
    """The orthogonal rows enter only through time_indices lookups.

    Complements test_zero_initialized_time_indices_match_legacy_tokens_and_outputs:
    that test pins the zero table, this one pins that a *nonzero* table still
    leaves an index-free forward untouched.
    """

    model = _configured_time_transformer()
    legacy_tokens = model._prepare_time_tokens(1, 3, None).detach().clone()

    model.reinitialize_time_index_embedding(
        "orthogonal", generator=torch.Generator().manual_seed(0)
    )

    assert torch.equal(model._prepare_time_tokens(1, 3, None), legacy_tokens)
    assert not torch.equal(
        model._prepare_time_tokens(1, 3, torch.tensor([[0, 1, 2]])),
        legacy_tokens,
    )


def test_reinitialize_validates_mode_scale_and_table_shape():
    model = _configured_time_transformer()

    with pytest.raises(ValueError, match="Unknown time-embedding init mode"):
        model.reinitialize_time_index_embedding("sinusoidal")
    with pytest.raises(ValueError, match="scale"):
        model.reinitialize_time_index_embedding("orthogonal", scale=0.0)
    with pytest.raises(ValueError, match="scale"):
        model.reinitialize_time_index_embedding("orthogonal", scale=float("nan"))

    oversized = _configured_time_transformer(max_time_indices=9)  # 9 rows > dim 8
    with pytest.raises(ValueError, match="at most"):
        oversized.reinitialize_time_index_embedding("orthogonal")

    tokenless = DinoVisionTransformer(
        img_size=14,
        patch_size=14,
        embed_dim=8,
        depth=3,
        num_heads=2,
        mlp_ratio=2,
        alt_start=2,
        has_time_token=False,
        cat_token=False,
    )
    with pytest.raises(ValueError, match="no time token"):
        tokenless.reinitialize_time_index_embedding("orthogonal")


# ------------------------------------------------------------------------------
# patch checkpoint format v2
# ------------------------------------------------------------------------------


def test_patch_records_freeze_mode_and_embedding_rows(tmp_path):
    trained = _TinyHubArc(freeze="temporal_tracking")
    patch = save_temporal_tracking_checkpoint(trained, tmp_path / "patch.pt")

    metadata = read_temporal_patch_metadata(patch)

    assert metadata == {
        "freeze_mode": "temporal_tracking",
        "max_time_indices": 4,
    }


def test_patch_save_requires_a_temporal_freeze_mode(tmp_path):
    unfrozen = _TinyHubArc(freeze="none")

    with pytest.raises(ValueError, match="model.freeze"):
        save_temporal_tracking_checkpoint(unfrozen, tmp_path / "patch.pt")


def test_patches_predating_the_freeze_mode_field_are_rejected(tmp_path):
    """Pre-freeze_mode patches also predate the motion-decoder gradient fix,
    so they carry weights trained on corrupted gradients; refuse them with an
    error that says to re-train rather than pretending they are loadable."""

    trained = _TinyHubArc(freeze="temporal_tracking")
    payload = {
        "format_version": torch.tensor(1, dtype=torch.int64),
        "state_dict": {
            name: parameter.detach().clone()
            for name, parameter in trained.named_parameters()
            if parameter.requires_grad
        },
    }
    path = tmp_path / "legacy_patch.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="re-run the overfit"):
        read_temporal_patch_metadata(path)
    with pytest.raises(RuntimeError, match="re-run the overfit"):
        load_temporal_tracking_checkpoint(
            _TinyHubArc(freeze="temporal_tracking"), path
        )


def test_patch_freeze_mode_mismatch_is_rejected_and_matching_mode_loads(tmp_path):
    trained = _GlobalAttnTinyArc(freeze="temporal_tracking_global_attention")
    with torch.no_grad():
        trained.backbone.pretrained.blocks[1].weight.fill_(1.25)
        trained.backbone.pretrained.time_index_embedding.weight.fill_(0.5)
    patch = save_temporal_tracking_checkpoint(trained, tmp_path / "patch.pt")

    metadata = read_temporal_patch_metadata(patch)
    assert metadata["freeze_mode"] == "temporal_tracking_global_attention"

    mismatched = _GlobalAttnTinyArc(freeze="temporal_tracking")
    with pytest.raises(ValueError, match="temporal_tracking_global_attention"):
        load_temporal_tracking_checkpoint(mismatched, patch)

    restored = _GlobalAttnTinyArc(freeze="temporal_tracking_global_attention")
    load_temporal_tracking_checkpoint(restored, patch)
    torch.testing.assert_close(
        restored.backbone.pretrained.blocks[1].weight,
        trained.backbone.pretrained.blocks[1].weight,
    )
    torch.testing.assert_close(
        restored.backbone.pretrained.time_index_embedding.weight,
        trained.backbone.pretrained.time_index_embedding.weight,
    )


def test_new_mode_patch_covers_the_trained_encoder_blocks(tmp_path):
    """The save path must key off the mode the run trained under.

    Re-asserting the narrower temporal_tracking mode before saving would
    silently drop every trained encoder tensor from the patch; keyed off the
    trained mode, the encoder blocks round-trip.
    """

    trained = _GlobalAttnTinyArc(freeze="temporal_tracking_global_attention")
    patch = save_temporal_tracking_checkpoint(trained, tmp_path / "patch.pt")

    payload = torch.load(patch, map_location="cpu", weights_only=True)
    saved_names = set(payload["state_dict"])
    assert "backbone.pretrained.blocks.1.weight" in saved_names
    assert "backbone.pretrained.blocks.3.weight" in saved_names
    assert "backbone.pretrained.blocks.0.weight" not in saved_names
    assert "backbone.pretrained.blocks.2.weight" not in saved_names


def test_harness_step_helpers_drive_the_split_forward():
    """The multi-anchor step's glue, exercised on the real Arc methods.

    ``_encode_and_reconstruct`` and ``_anchor_tracks`` are what a training step
    calls instead of ``model(views)``. They are thin, but they are the only new
    code between the model and the loss, and a shape or key mistake in them
    would surface as a wasted GPU allocation rather than a test failure.
    """

    import overfit_temporal_tracking as overfit_cli

    model = _arc_shell(max_time_indices=32)
    model.backbone = _FakeBackbone()
    model.head = _FakeReconstructionHead()
    model.cam_dec = _FakeCameraDecoder()
    model.motion_decoder = _FakeMotionDecoder()
    model.track_head = _FakeTrackHead()

    anchor_slots = (0, 3)
    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(4)]
    inference_cli.attach_frame_metadata(
        views,
        track_query_idx=list(anchor_slots),
        time_indices=list(range(4)),
    )
    scene = SimpleNamespace(
        anchor_observation_slots=anchor_slots,
        num_observations=len(views),
    )

    images, feats, recon = overfit_cli._encode_and_reconstruct(model, views)

    assert images.shape == (1, 4, 3, 2, 2)
    # The reconstruction is built once and shared: it is what the Sim(3) fit and
    # the query pointmap anchors read, for every anchor. This stack's fake depth
    # head returns nothing, so only the camera branch is asserted here; the real
    # dict is pinned by test_forward_recomposes_from_its_three_public_pieces.
    assert {"pose_enc", "pose_enc_list"} <= set(recon)
    assert recon["pose_enc"].shape == (1, 4, 1)
    assert len(feats) == 4

    for anchor_index, slot in enumerate(anchor_slots):
        raw = overfit_cli._anchor_tracks(model, feats, images, scene, anchor_index)
        # Shaped as the Q=1 raw dict the loss expects, so sparse_tracking_loss
        # keeps its contract whether it is scoring one anchor or a stacked Q=A.
        assert raw["track_multi"].shape == (1, 1, 4, 2, 2, 3)
        assert raw["conf_track_multi"].shape == (1, 1, 4, 2, 2)
        assert raw["track_query_idx"].tolist() == [slot]
        assert torch.all(raw["track_multi"] == slot)
