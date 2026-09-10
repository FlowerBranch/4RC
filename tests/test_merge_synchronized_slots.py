"""The merged track head: keys pooled by time index behind a default-off flag.

``--merge_synchronized_slots`` makes the motion decoder attend, for each of the
window's T times, over ALL cameras' patch tokens of that instant and emit one
displacement field per time instead of one per observation slot.  Three things
here are load-bearing enough to pin at the head level:

* the flag OFF must be bit-identical to the pre-flag decoder -- every archived
  number was produced on that path and must stay reproducible;
* the pooling is POSITIONAL (slot = camera*T + t), so the derivation of
  views-per-time from the time indices must refuse any layout positional
  pooling would silently mis-merge (a time-major window has uniform
  multiplicity and no shape error anywhere -- only the predicate here stops it);
* the pooled key rows must actually carry every camera's patches, in blocks,
  each prefixed by that slot's camera token -- concatenated, not added, because
  the camera token is an un-normed residual-stream activation whose RMS was
  measured at 4-18x the normed patches' (adding it would suppress patch content
  in every pooled key).
"""

import pytest
import torch

import inference as inference_cli
from arc.models.arc.heads.dpt_head import DPTHead
from arc.models.arc.heads.motiondecoder import MotionDecoder, merge_time_grouped_tokens
from test_time_indexing import (
    _arc_shell,
    _FakeBackbone,
    _FakeCameraDecoder,
    _FakeMotionDecoder,
    _FakeReconstructionHead,
    _FakeTrackHead,
)


def _tiny_decoder(use_adaln, **overrides):
    torch.manual_seed(0)
    settings = dict(patch_size=14, embed_dim=64, depth=4, num_heads=4, use_adaln=use_adaln)
    settings.update(overrides)
    return MotionDecoder(**settings).eval()


def _decoder_inputs(views_per_time, time_count, patch_count=4, embed_dim=64):
    torch.manual_seed(1)
    slots = views_per_time * time_count
    tokens = torch.randn(1, slots, 2 + patch_count, embed_dim)
    images = torch.zeros(1, slots, 3, 28, 28)
    return tokens, images


def _pre_flag_decoder_reference(decoder, tokens, images, patch_start_idx, track_query_idx):
    """The pre-flag forward, transcribed by hand against the decoder's modules.

    Eval mode only, so the checkpointed segments never engage and the block
    loop is a plain sequence.  This is the reference the default branch is
    pinned to: the refactor must leave that branch as today's code, not a
    re-derivation of it.
    """

    B, S, _, C = tokens.shape
    _, _, _, H, W = images.shape
    patches = tokens[:, :, patch_start_idx:, :]
    P = patches.shape[2]
    query_patches = patches[:, track_query_idx : track_query_idx + 1, :, :].expand(B, S, P, C)
    time_emb = tokens[:, :, 1:2, :]
    time_cond = time_emb.flatten(0, 2) if decoder.use_adaln else None
    query = torch.cat([time_emb, query_patches], dim=2)
    kv = patches
    pos_patches = decoder.position_getter(
        B * S, H // decoder.patch_size, W // decoder.patch_size, device=images.device
    )
    pos_patches = pos_patches + 1
    pos_time = torch.zeros(B * S, 1, 2, device=tokens.device, dtype=pos_patches.dtype)
    pos_q = torch.cat([pos_time, pos_patches], dim=1)
    pos_cross = (pos_q, pos_patches)
    query = query.flatten(0, 1)
    kv = kv.flatten(0, 1)
    for index in range(decoder.depth):
        query = decoder.cross_blocks[index](query, kv, pos=pos_cross)
        if decoder.use_adaln:
            query = decoder.self_blocks[index](query, cond=time_cond, pos=pos_q)
        else:
            query = decoder.self_blocks[index](query, pos=pos_q)
    return query.view(B, S, 1 + P, C)


# ------------------------------------------------ the flag off is inert ---


@pytest.mark.parametrize("use_adaln", [True, False])
def test_views_per_time_one_is_bit_identical_to_the_pre_flag_decoder(use_adaln):
    """Flag off, the decoder must BE the pre-flag decoder, bit for bit.

    Would catch the branch restructure quietly re-deriving the per-slot path
    (reordered ops, a shared line changing semantics) instead of keeping
    today's code verbatim -- the failure mode that silently breaks
    reproducibility of every archived number.
    """

    decoder = _tiny_decoder(use_adaln)
    tokens, images = _decoder_inputs(views_per_time=2, time_count=3)

    with torch.no_grad():
        reference = _pre_flag_decoder_reference(
            decoder, tokens, images, patch_start_idx=2, track_query_idx=0
        )
        default = decoder(tokens, images=images, patch_start_idx=2, track_query_idx=0)
        explicit = decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0, views_per_time=1
        )

    assert torch.equal(default, reference)
    assert torch.equal(explicit, reference)


# ------------------------------------------------ the merged branch ---


@pytest.mark.parametrize("use_adaln", [True, False])
def test_the_merged_decoder_emits_one_row_per_time(use_adaln):
    """views_per_time=V collapses the S=V*T axis to T rows of the same width.

    Would catch the merged branch leaving the output on the slot axis (the
    trainer's shape check would then pass S through where every merged
    consumer expects T) and a views_per_time that does not divide S slipping
    through to a garbled view.
    """

    decoder = _tiny_decoder(use_adaln)
    tokens, images = _decoder_inputs(views_per_time=2, time_count=3)

    with torch.no_grad():
        merged = decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0, views_per_time=2
        )

    assert merged.shape == (1, 3, 1 + 4, 64)
    with pytest.raises(ValueError, match="does not divide"):
        decoder(tokens, images=images, patch_start_idx=2, track_query_idx=0, views_per_time=4)


def test_the_merged_branch_runs_without_positions():
    """rope_freq<=0 leaves position_getter None; the merged branch must still bind.

    No static analysis covers arc/ modules, so an unbound pos_q/pos_k on the
    merged-only path would surface only here.  Cross-attention is off because
    both branches deliberately mirror the pre-existing quirk that pos_cross is
    unbound without a position_getter.
    """

    torch.manual_seed(0)
    decoder = MotionDecoder(
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        rope_freq=0,
        has_cross_attention=False,
    ).eval()
    tokens, images = _decoder_inputs(views_per_time=2, time_count=3)

    with torch.no_grad():
        merged = decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0, views_per_time=2
        )
        per_slot = decoder(tokens, images=images, patch_start_idx=2, track_query_idx=0)

    assert merged.shape == (1, 3, 1 + 4, 64)
    assert per_slot.shape == (1, 6, 1 + 4, 64)


def test_the_pooled_keys_keep_every_cameras_patches_in_its_own_block_prefixed_by_that_slots_camera_token():
    """The pooling rule itself, on hand values the eye can check.

    Would catch a wrong permutation (camera- and time-major swapped, the
    silent failure the whole layout machinery guards), the camera token added
    into the patches instead of concatenated, and a query time token that is a
    mean over cameras rather than the anchor camera's own.  Hand values cannot
    prove a "camera identity" -- the token is a per-slot summary -- so this
    claims only what it shows: each camera's patches sit in their own block,
    prefixed by that slot's token, and the blocks are distinguishable.
    """

    views, times, patch_count, width = 3, 4, 2, 5
    tokens = torch.zeros(1, views * times, 2 + patch_count, width)
    for slot in range(views * times):
        tokens[0, slot, 0] = 100.0 + slot
        # Squares, not an arithmetic progression: with linear values the anchor
        # (the middle camera) would EQUAL the camera mean and the not-a-mean
        # assertion below would be vacuous.
        tokens[0, slot, 1] = 200.0 + float(slot**2)
        tokens[0, slot, 2:] = float(slot)
    anchor = 1 * times + 2  # camera 1, time 2

    query, kv = merge_time_grouped_tokens(
        tokens, patch_start_idx=2, track_query_idx=anchor, views_per_time=views
    )

    assert query.shape == (1, times, 1 + patch_count, width)
    assert kv.shape == (1, times, views * (1 + patch_count), width)
    block_width = 1 + patch_count
    for camera in range(views):
        for time in range(times):
            slot = camera * times + time
            block = kv[0, time, camera * block_width : (camera + 1) * block_width]
            assert torch.equal(block[0], tokens[0, slot, 0])
            assert torch.equal(block[1:], tokens[0, slot, 2:])
    for time in range(times):
        blocks = kv[0, time].view(views, block_width, width)
        for first in range(views):
            for second in range(first + 1, views):
                assert not torch.equal(blocks[first], blocks[second])

    for time in range(times):
        # The ANCHOR CAMERA's time token for this instant, and the anchor
        # slot's own patches at every row -- the merged query row is the row
        # slot (anchor_camera, time) produces on the per-slot path.
        assert torch.equal(query[0, time, 0], tokens[0, 1 * times + time, 1])
        assert torch.equal(query[0, time, 1:], tokens[0, anchor, 2:])
    camera_mean = torch.stack(
        [tokens[0, camera * times + 0, 1] for camera in range(views)]
    ).mean(dim=0)
    assert not torch.equal(query[0, 0, 0], camera_mean)


# ------------------------------------------------ Arc derives V from the inputs ---


def _merged_fake_arc():
    model = _arc_shell(max_time_indices=32)
    model.backbone = _FakeBackbone()
    model.head = _FakeReconstructionHead()
    model.cam_dec = _FakeCameraDecoder()
    model.motion_decoder = _FakeMotionDecoder()
    model.track_head = _FakeTrackHead()
    return model


def _metadata_views(time_indices):
    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(8)]
    inference_cli.attach_frame_metadata(
        views, track_query_idx=[0], time_indices=time_indices
    )
    return views


def test_arc_derives_views_per_time_from_the_time_indices():
    """One forward kwarg turns the S axis into a T axis, from the inputs alone.

    Would catch the flag reaching the decoder without the derivation (the
    decoder then divides by a stale constant), and the derivation accepting a
    layout the positional pooling mis-merges.  The accept/refuse table is the
    substance: camera-major and the shuffled arm's per-camera reversal pass,
    a time-major window and a ragged one are refused by name, and a window
    with no time metadata cannot be merged at all.
    """

    model = _merged_fake_arc()
    output = model(
        _metadata_views([0, 1, 2, 3, 0, 1, 2, 3]),
        force_no_output_conversion=True,
        merge_synchronized_slots=True,
    )
    assert output["track_multi"].shape == (1, 1, 4, 2, 2, 3)
    assert output["conf_track_multi"].shape == (1, 1, 4, 2, 2)
    assert model.motion_decoder.seen_views_per_time == [2, 2, 2, 2]

    model = _merged_fake_arc()
    output = model(
        _metadata_views([0, 1, 2, 3, 0, 1, 2, 3]),
        force_no_output_conversion=True,
    )
    assert output["track_multi"].shape == (1, 1, 8, 2, 2, 3)
    assert model.motion_decoder.seen_views_per_time == [1, 1, 1, 1]

    # The shuffled-index eval arm reverses every non-primary camera's indices;
    # positional pooling stays physically correct, so it must be ACCEPTED.
    model = _merged_fake_arc()
    output = model(
        _metadata_views([0, 1, 2, 3, 3, 2, 1, 0]),
        force_no_output_conversion=True,
        merge_synchronized_slots=True,
    )
    assert output["track_multi"].shape == (1, 1, 4, 2, 2, 3)

    # Uniform multiplicity but time-major: positional pooling would merge the
    # wrong slots with no shape error anywhere. The permutation predicate is
    # the only thing that stops it.
    model = _merged_fake_arc()
    with pytest.raises(ValueError, match="camera-major slot group"):
        model(
            _metadata_views([0, 0, 1, 1, 2, 2, 3, 3]),
            force_no_output_conversion=True,
            merge_synchronized_slots=True,
        )

    model = _merged_fake_arc()
    with pytest.raises(ValueError, match="same number of cameras"):
        model(
            _metadata_views([0, 0, 0, 1, 1, 2, 2, 3]),
            force_no_output_conversion=True,
            merge_synchronized_slots=True,
        )

    model = _merged_fake_arc()
    views = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(8)]
    inference_cli.attach_frame_metadata(views, track_query_idx=[0])
    with pytest.raises(ValueError, match="time_index"):
        model(views, force_no_output_conversion=True, merge_synchronized_slots=True)


def test_the_merge_refuses_the_postprocessing_output_path():
    """_postprocess_output unbinds track_multi per input view; refuse, not garble.

    inference.py and app.py convert outputs by default, and a merged T-row
    track_multi would be unbound against S per-view pointmaps -- a shape error
    at best, a silent misalignment at worst. The guard must fire before any
    work is done.
    """

    model = _merged_fake_arc()
    with pytest.raises(ValueError, match="force_no_output_conversion"):
        model(
            _metadata_views([0, 1, 2, 3, 0, 1, 2, 3]),
            merge_synchronized_slots=True,
        )


# ------------------------------------------------ the track head slice is shape-only ---


def test_the_track_head_reads_only_the_shape_of_its_images():
    """arc.py passes x[:, :T] to the head under the merge; the slice is shape-only.

    DPTHead reads `images` for B, S, H, W and its frames-chunk slicing alone,
    never the pixels -- which is the only reason the seemingly wrong-looking
    frame selection in track_for_query cannot matter. Would catch the head
    growing a real pixel read, at which point the merge needs a real answer to
    "which frames".
    """

    torch.manual_seed(0)
    head = DPTHead(
        dim_in=8,
        output_dim=4,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    ).eval()
    torch.manual_seed(1)
    tokens = [torch.randn(1, 3, 1 + 16, 8) for _ in range(4)]

    with torch.no_grad():
        track_a, conf_a = head(
            tokens,
            images=torch.zeros(1, 3, 3, 56, 56),
            patch_start_idx=1,
            frames_chunk_size=8,
        )
        track_b, conf_b = head(
            tokens,
            # A T-length slice of a wider tensor full of garbage, the exact
            # shape track_for_query hands over under the merge.
            images=torch.full((1, 6, 3, 56, 56), 7.0)[:, :3],
            patch_start_idx=1,
            frames_chunk_size=8,
        )

    assert track_a.shape == (1, 3, 56, 56, 3)
    assert conf_a.shape == (1, 3, 56, 56)
    assert torch.equal(track_a, track_b)
    assert torch.equal(conf_a, conf_b)
