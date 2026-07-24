import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import inference as inference_cli
from arc.models.arc.arc import Arc
from arc.models.arc.dinov2.vision_transformer import DinoVisionTransformer


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


def test_input_paths_and_time_metadata_are_subsampled_in_lockstep():
    paths = [f"frame_{index:03d}.png" for index in range(41)]
    semantic_times = list(range(41))
    expected_positions = np.linspace(0, 40, 30, dtype=int)

    selected_paths, selected_times = inference_cli.select_input_frames(
        paths,
        semantic_times,
    )

    assert selected_paths == [paths[index] for index in expected_positions]
    assert selected_times == [semantic_times[index] for index in expected_positions]


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
