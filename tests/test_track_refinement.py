"""Unrolled track refinement: ONE iteration in the model, the loops in the callers.

What is pinned here, and why each pin is the one that would catch wrong code:

* the refiner is zero-init and constructed under fork_rng, so ``--refine_iters
  1`` is today's model to the bit, seeded construction included. The executed
  arm -- filled weights move the output, the zero tensors receive gradient,
  the projections behind them sit at an exactly-zero saddle until read_proj
  is filled -- is what separates "inert by design" from "never wired": a
  ``refinement`` keyword swallowed or never read is byte-identical to a
  working zero-init branch.
* the kNN read is checked by hand at a size the eye can follow, over two
  cameras and two times with distinct clouds per time, because a flipped
  offset sign, a corner-pixel read or a time-major slot pooling is bit-silent
  under zero init and merely "different" under random init.
* ``Arc._forward`` and ``train_step`` loop the single-iteration head themselves
  and detach the carry. A loop baked into the head would hold K head graphs; a
  carry left attached is refused at construction. The retention test measures
  what per-iteration backward buys on CPU, the way test_dpt_head_checkpointing
  measures the chunk checkpoint, and the trainer test counts the backwards
  because the gradient identity alone cannot (backward is linear).
* ``StepOutcome.iteration_losses`` and the eval's ``iteration_position_losses``
  are the instrument that says whether iteration k+1 came in below iteration k
  from step 1 of any trainer run, on real data; at K=1 both are None, like
  ``loss_breakdown`` on a position-only step, and every call keeps today's
  exact spelling, which the CPU fakes of three suites bind.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import textwrap
import weakref

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import inference as inference_cli
import train_temporal_tracking as train_cli
from arc.models.arc.arc import Arc
from arc.models.arc.heads.dpt_head import DPTHead
from arc.models.arc.heads.motiondecoder import (
    REFINER_CORRELATION_DIM,
    REFINER_FIELD_CHANNELS,
    REFINER_NEIGHBOURS,
    MotionDecoder,
    RefinementInput,
    TrackRefiner,
    _pool_slots_by_time,
    merge_time_grouped_tokens,
    patch_centre_points,
)
from arc.models.arc.utils.transform import predicted_pointmaps
from arc.training import refinement_iteration_weights, runtime
from arc.training.checkpoint import (
    load_temporal_tracking_checkpoint,
    read_temporal_patch_metadata,
)
from arc.training.manifest_plan import plan_record
from test_manifest_plan import _record
from test_time_indexing import (
    _arc_shell,
    _FakeBackbone,
    _FakeCameraDecoder,
    _FakeMotionDecoder,
    _FakeReconstructionHead,
    _FakeTrackHead,
    _TinyHubArc,
)
from test_trainer_loop import (
    _EVAL_ALPHA,
    _FakeArc,
    _history,
    _loop_args,
    _plans,
    _step_scene,
)

_PATCH = 14


# ------------------------------------------------------------ fixtures ---


def _refinement_case(*, grid=(3, 6), views=2, times=2, embed_dim=64, seed=0):
    """Tokens, images and a predicted cloud for the REAL tiny decoder.

    3x6 patches rather than the suite's 2x3: the production refiner reads
    REFINER_NEIGHBOURS = 16 keys and on the per-slot path the key set is one
    slot's P patches, so P must be at least 16 or TrackRefiner refuses the
    grid (that refusal has its own test below). Still non-square and
    multi-patch, so a transposed flatten or a permuted slot order has
    somewhere to show; V and T both 2, so camera-major and time-major
    poolings differ.
    """

    torch.manual_seed(seed)
    rows, cols = grid
    height, width = rows * _PATCH, cols * _PATCH
    slots = views * times
    patches = rows * cols
    tokens = torch.randn(1, slots, 2 + patches, embed_dim)
    images = torch.zeros(1, slots, 3, height, width)
    cloud = torch.randn(1, slots, height, width, 3)
    key_xyz = patch_centre_points(cloud, _PATCH)
    return tokens, images, key_xyz


def _tiny_decoder():
    torch.manual_seed(0)
    return MotionDecoder(
        patch_size=_PATCH, embed_dim=64, depth=2, num_heads=4, use_adaln=True
    ).eval()


def _refinement(key_xyz, query_idx, previous_field):
    return RefinementInput(
        previous_field=previous_field,
        anchor_xyz=key_xyz[:, query_idx],
        key_xyz=key_xyz,
    )


def _square_fill(parameter):
    """Fill a weight with the squares 1, 4, 9, ... scaled into (0, 1]: hand
    values that are not an arithmetic progression, so no two rows or columns
    of the weight are a shift of each other."""

    squares = torch.arange(1, parameter.numel() + 1, dtype=torch.float32).square()
    with torch.no_grad():
        parameter.copy_((squares / squares.max()).view_as(parameter))


# ------------------------------------------------------ layout helpers ---


def test_patch_centre_points_reads_the_centre_pixel_in_patch_embed_order():
    """Pixel (7, 7) of every 14x14 block, row-major over the grid -- the order
    PatchEmbed's conv -> flatten(2).transpose(1, 2) produces, checked against
    a real Conv2d rather than assumed. A corner-pixel read, a mean over the
    block, or a column-major flatten each pass on a square 1x1 grid and fail
    on this 2x3 one; the point values encode (row, col, slot) so a swap of
    any two axes changes the numbers. The two refusals are the helper's own:
    a pointmap without its xyz axis, and a grid the patch does not divide,
    which would otherwise reshape into the wrong number of patches silently."""

    views, times = 2, 2
    slots = views * times
    height, width = 2 * _PATCH, 3 * _PATCH
    rows = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    slot = torch.arange(slots, dtype=torch.float32).view(1, slots, 1, 1)
    pointmaps = torch.stack(
        [
            rows.expand(1, slots, height, width),
            cols.expand(1, slots, height, width),
            slot.expand(1, slots, height, width),
        ],
        dim=-1,
    )

    centres = patch_centre_points(pointmaps, _PATCH)

    assert centres.shape == (1, slots, 6, 3)
    expected = torch.tensor(
        [[7.0, 7.0], [7.0, 21.0], [7.0, 35.0], [21.0, 7.0], [21.0, 21.0], [21.0, 35.0]]
    )
    for index in range(slots):
        assert torch.equal(centres[0, index, :, :2], expected)
        assert torch.equal(centres[0, index, :, 2], torch.full((6,), float(index)))

    # The same order a patch conv flattens to: a one-hot kernel at (7, 7)
    # over an image whose block (r, c) carries the marker r*3 + c + 1 at its
    # centre pixel must come out as 1..6 in sequence.
    image = torch.zeros(1, 1, height, width)
    for row in range(2):
        for col in range(3):
            image[0, 0, 7 + _PATCH * row, 7 + _PATCH * col] = float(row * 3 + col + 1)
    conv = nn.Conv2d(1, 1, kernel_size=_PATCH, stride=_PATCH, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 7, 7] = 1.0
    flattened = conv(image).flatten(2).transpose(1, 2)
    assert torch.equal(flattened[0, :, 0], torch.arange(1.0, 7.0))

    with pytest.raises(ValueError, match=r"\(B, S, H, W, 3\)"):
        patch_centre_points(pointmaps[..., :2], _PATCH)
    with pytest.raises(ValueError, match="not a multiple of patch_size"):
        patch_centre_points(pointmaps[:, :, : height - 1], _PATCH)


def test_pool_slots_by_time_is_merge_time_grouped_tokens_key_permutation():
    """The refiner pools the key patch tokens AND key_xyz with this one
    helper, so the two can only disagree if the helper is not the permutation
    merge_time_grouped_tokens applies to kv (motiondecoder.py:63-68). Hand
    values as in test_merge_synchronized_slots (V=3, T=4, squares): a
    time-major view(B, T, V, ...) instead of view(B, V, T, ...) reshuffles
    which slot lands in which pooled row with no shape error anywhere, which
    is why the time-major layout is built and asserted unequal."""

    views, times, patches, width = 3, 4, 6, 5
    tokens = torch.zeros(1, views * times, 2 + patches, width)
    for slot in range(views * times):
        tokens[0, slot, 0] = 100.0 + slot
        tokens[0, slot, 1] = 200.0 + float(slot**2)
        for patch in range(patches):
            tokens[0, slot, 2 + patch] = float(slot**2 + patch)

    _, kv = merge_time_grouped_tokens(
        tokens, patch_start_idx=2, track_query_idx=0, views_per_time=views
    )
    pooled = _pool_slots_by_time(tokens[:, :, 2:], views)

    assert pooled.shape == (1, times, views * patches, width)
    assert torch.equal(
        pooled,
        kv.view(1, times, views, 1 + patches, width)[:, :, :, 1:].reshape(
            1, times, views * patches, width
        ),
    )
    for time in range(times):
        for camera in range(views):
            block = pooled[0, time, camera * patches : (camera + 1) * patches]
            assert torch.equal(block, tokens[0, camera * times + time, 2:])
    time_major = (
        tokens[:, :, 2:]
        .view(1, times, views, patches, width)
        .reshape(1, times, views * patches, width)
    )
    assert not torch.equal(pooled, time_major)
    with pytest.raises(ValueError, match="does not divide"):
        _pool_slots_by_time(tokens[:, :, 2:], 5)


# ------------------------------------------------- the refiner itself ---


def test_the_refiner_is_built_under_fork_rng_with_eight_tensors():
    """Flag-off bit-identity's construction half, which no output comparison
    sees: Linear default init draws from the global RNG, and MotionDecoder
    builds its refiner LAST, right before Arc builds the DPTHead -- an
    unguarded draw would shift every seeded head weight against today's.
    The state check runs the real constructor; the lexical pin is the
    inspected-rather-than-executed precedent from
    test_injection_construction_is_rng_guarded_in_the_real_constructor and
    catches a guard that covers three of the four modules. The zero-init is
    asserted on weight AND bias of exactly the two output-side tensors, and
    the default init on the two projections, because all four at zero would
    make the correlation channel a fixed point (see TrackRefiner)."""

    torch.manual_seed(0)
    state = torch.random.get_rng_state()
    refiner = TrackRefiner(64, _PATCH)
    assert torch.equal(torch.random.get_rng_state(), state)

    assert sorted(name for name, _ in refiner.named_parameters()) == [
        "field_embed.bias",
        "field_embed.weight",
        "key_proj.bias",
        "key_proj.weight",
        "query_proj.bias",
        "query_proj.weight",
        "read_proj.bias",
        "read_proj.weight",
    ]
    for module in (refiner.field_embed, refiner.read_proj):
        assert torch.count_nonzero(module.weight) == 0
        assert torch.count_nonzero(module.bias) == 0
    for module in (refiner.query_proj, refiner.key_proj):
        assert torch.count_nonzero(module.weight) > 0
    assert refiner.field_embed.in_channels == REFINER_FIELD_CHANNELS == 3
    assert refiner.field_embed.kernel_size == (_PATCH, _PATCH)
    assert refiner.field_embed.stride == (_PATCH, _PATCH)
    assert refiner.query_proj.out_features == REFINER_CORRELATION_DIM == 128
    assert refiner.key_proj.out_features == REFINER_CORRELATION_DIM
    assert refiner.read_proj.in_features == REFINER_NEIGHBOURS * 4 == 64
    assert refiner.read_proj.out_features == 64

    source = textwrap.dedent(inspect.getsource(TrackRefiner.__init__))
    guarded = {}

    def visit(node, inside_fork):
        if isinstance(node, ast.With):
            inside_fork = inside_fork or any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "attr", None) == "fork_rng"
                for item in node.items
            )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr
                    in ("field_embed", "query_proj", "key_proj", "read_proj")
                ):
                    guarded[target.attr] = inside_fork
        for child in ast.iter_child_nodes(node):
            visit(child, inside_fork)

    visit(ast.parse(source), False)
    assert guarded == {
        "field_embed": True,
        "query_proj": True,
        "key_proj": True,
        "read_proj": True,
    }

    decoder = _tiny_decoder()
    assert isinstance(decoder.refiner, TrackRefiner)
    assert list(decoder._modules)[-1] == "refiner"
    assert [
        name for name, _ in decoder.named_parameters() if name.startswith("refiner.")
    ] == [
        "refiner.field_embed.weight",
        "refiner.field_embed.bias",
        "refiner.query_proj.weight",
        "refiner.query_proj.bias",
        "refiner.key_proj.weight",
        "refiner.key_proj.bias",
        "refiner.read_proj.weight",
        "refiner.read_proj.bias",
    ]


def test_the_refiner_reads_the_k_nearest_predicted_points_by_hand():
    """V=2, T=2 under the merge (R=2 rows, six pooled keys per row), a 1x3
    patch grid (P=3, H=14, W=42), k=2, embed 8 with correlation_dim 5:
    rectangular-identity projections keep the first five token channels, so
    corr = q . k over those five / sqrt(5). sqrt(5) is irrational and equals
    no other number in this fixture -- not the token width 8 or its root, not
    the neighbour count 2 or its root, not the field channels 3 or 1 -- so a
    normaliser by any of those, or by none, moves every corr digit. The
    width was 4 in an earlier version, where sqrt(4) = 2 = the neighbour
    count and a divisor by `self.neighbours` passed unnoticed.
    read_proj is the identity, so the returned term IS the read vector
    [corr, dx, dy, dz] per neighbour. Camera 0's clouds sit on the x axis at
    squares (0, 4, 16), camera 1's at (1, 9, 25), and the time-1 clouds are
    lifted to z=100, so every number below is one the eye can check and a
    cloud pooled TIME-major (row 0 seeing camera 0's two times instead of
    both cameras at time 0) picks different neighbours for every estimate.
    An offset with the sign flipped, an estimate read at the block corner
    rather than its centre, or corr and offsets concatenated the other way
    round each move a digit; the anchor is a camera-1 slot, so a read of the
    anchor's own cloud instead of the row's keys shows as well. Nothing
    under random weights would say which."""

    embed_dim, neighbours, correlation_dim = 8, 2, 5
    refiner = TrackRefiner(
        embed_dim, _PATCH, neighbours=neighbours, correlation_dim=correlation_dim
    )
    with torch.no_grad():
        # The projections keep the first correlation_dim channels of the
        # eight-wide tokens; read_proj (neighbours * 4 = 8 -> 8) is square.
        for module in (refiner.query_proj, refiner.key_proj):
            module.weight.copy_(torch.eye(correlation_dim, embed_dim))
            module.bias.zero_()
        refiner.read_proj.weight.copy_(torch.eye(embed_dim))
        refiner.read_proj.bias.zero_()
    assert refiner.query_proj.weight.shape == (correlation_dim, embed_dim)
    views, times, patches = 2, 2, 3  # merge: R = T = 2, Nk = V * P = 6

    # slot = camera * T + t. Camera 0 at x = 0, 4, 16; camera 1 at x = 1, 9,
    # 25; time 1 lifted to z = 100 so the two times' clouds differ.
    key_xyz = torch.zeros(1, views * times, patches, 3)
    key_xyz[0, 0, :, 0] = torch.tensor([0.0, 4.0, 16.0])
    key_xyz[0, 1, :, 0] = torch.tensor([0.0, 4.0, 16.0])
    key_xyz[0, 1, :, 2] = 100.0
    key_xyz[0, 2, :, 0] = torch.tensor([1.0, 9.0, 25.0])
    key_xyz[0, 3, :, 0] = torch.tensor([1.0, 9.0, 25.0])
    key_xyz[0, 3, :, 2] = 100.0
    anchor_slot = 2  # camera 1, time 0
    anchor_xyz = key_xyz[:, anchor_slot]
    # The previous field is noise everywhere except the three block CENTRES,
    # (7, 7), (7, 21) and (7, 35): only those may be read.
    torch.manual_seed(3)
    previous = torch.randn(1, times, _PATCH, patches * _PATCH, 3)
    centres = ((7, 7), (7, 21), (7, 35))
    fields = (
        ((0.5, 0.0, 0.0), (2.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
        ((0.0, 3.0, 0.0), (0.0, 0.0, -4.0), (2.0, 0.0, 0.0)),
    )
    for time in range(times):
        for patch, (row, col) in enumerate(centres):
            previous[0, time, row, col] = torch.tensor(fields[time][patch])
    # Row 0 estimates: (1.5,0,0), (11,0,0), (21,0,0) against time-0 keys
    #   pooled [x0, x4, x16 | x1, x9, x25]: nearest (idx 3, idx 0), (4, 2), (5, 2).
    # Row 1 estimates: (1,3,0), (9,0,-4), (27,0,0) against the z=100 keys:
    #   nearest (3, 0), (4, 1), (5, 2), every offset carrying dz = 100 or 104.

    e = torch.eye(embed_dim)
    query_patches = (
        torch.stack([1 * e[0], 4 * e[1], 9 * e[2]])
        .view(1, 1, patches, embed_dim)
        .expand(1, times, patches, embed_dim)
        .contiguous()
    )
    # Key (slot s, patch j) = (3s + j + 1)^2 on the first three axes, so its
    # correlation with query patch p is (p + 1)^2 * (3s + j + 1)^2 / sqrt(5):
    # the dot product runs over the five projected channels, of which only
    # the first three are non-zero.
    key_patches = torch.zeros(1, views * times, patches, embed_dim)
    for slot in range(views * times):
        for patch in range(patches):
            key_patches[0, slot, patch] = float((3 * slot + patch + 1) ** 2) * (
                e[0] + e[1] + e[2]
            )

    term = refiner(
        query_patches,
        key_patches,
        RefinementInput(previous_field=previous, anchor_xyz=anchor_xyz, key_xyz=key_xyz),
        views_per_time=views,
        merge=True,
    )

    scale = 1.0 / math.sqrt(correlation_dim)
    expected = torch.tensor(
        [
            [
                [49 * scale, -0.5, 0.0, 0.0, 1 * scale, -1.5, 0.0, 0.0],
                [256 * scale, -2.0, 0.0, 0.0, 36 * scale, 5.0, 0.0, 0.0],
                [729 * scale, 4.0, 0.0, 0.0, 81 * scale, -5.0, 0.0, 0.0],
            ],
            [
                [100 * scale, 0.0, -3.0, 100.0, 16 * scale, -1.0, -3.0, 100.0],
                [484 * scale, 0.0, 0.0, 104.0, 100 * scale, -5.0, 0.0, 104.0],
                [1296 * scale, -2.0, 0.0, 100.0, 324 * scale, -11.0, 0.0, 100.0],
            ],
        ]
    ).view(1, times, patches, embed_dim)
    assert term.shape == (1, times, patches, embed_dim)
    # rtol for the correlations: 1/sqrt(5) is irrational, so the hand value and
    # the refiner's round to float32 apart by up to an ulp -- 1.5e-5 at 216.
    # Every wrong normaliser below is at least 10% off, far outside it.
    torch.testing.assert_close(term, expected, atol=1e-5, rtol=1e-6)
    # The offsets are a gather and a subtraction, so they are exact.
    assert torch.equal(term[..., 1:4], expected[..., 1:4])
    assert torch.equal(term[..., 5:8], expected[..., 5:8])
    # The wrong normalisers, each excluded by name: the square root of the
    # token width, of the neighbour count and of the field channels, the bare
    # neighbour count and correlation width, and no normaliser at all.
    for column in (0, 4):
        unnormalised = expected[..., column] / scale
        for wrong in (
            math.sqrt(embed_dim),
            math.sqrt(neighbours),
            math.sqrt(REFINER_FIELD_CHANNELS),
            neighbours,
            correlation_dim,
            1,
        ):
            assert not torch.allclose(
                term[..., column], unnormalised / wrong, atol=1e-5, rtol=0
            ), wrong

    # The same answer from torch.cdist + topk done inline per row, with the
    # keys pooled camera-major by hand -- MVTracker's _knn_torch
    # (mvtracker.py:75-79) -- so the hand numbers are not the only witness.
    for time in range(times):
        estimate = anchor_xyz[0] + previous[0, time, 7::_PATCH, 7::_PATCH].reshape(
            patches, 3
        )
        pooled = torch.cat([key_xyz[0, camera * times + time] for camera in range(views)])
        indices = torch.cdist(estimate, pooled).topk(
            neighbours, dim=-1, largest=False
        ).indices
        assert indices.tolist() == [[3, 0], [4, 2 if time == 0 else 1], [5, 2]]
        offsets = pooled[indices] - estimate[:, None]
        assert torch.equal(term[0, time, :, 1:4], offsets[:, 0])
        assert torch.equal(term[0, time, :, 5:8], offsets[:, 1])
    # And the wrong pooling would have chosen differently: time-major row 0
    # is camera 0's two times, and (1.5, 0, 0) then finds x0 and x4.
    time_major_row0 = torch.cat([key_xyz[0, 0], key_xyz[0, 1]])
    estimate_row0 = anchor_xyz[0] + previous[0, 0, 7::_PATCH, 7::_PATCH].reshape(patches, 3)
    assert torch.cdist(estimate_row0, time_major_row0).topk(
        neighbours, dim=-1, largest=False
    ).indices.tolist() != [[3, 0], [4, 2], [5, 2]]


def test_the_refiner_refuses_a_grid_below_k_and_mismatched_rows_and_clouds():
    """Each refusal names its rule instead of surfacing four calls deep as a
    topk or reshape error that names neither: fewer key points than
    neighbours (the suite's 2x3 grid, P=6, against k=16), a previous field
    with S rows under the merge where the decoder emits T, a cloud with more
    points than key patch tokens, and RefinementInput's own construction
    checks -- a field without its xyz axis, a cloud whose patch count
    disagrees with the anchor's, tensors that disagree on B, and a carry that
    still carries a graph, which would chain iteration k-1's head graph into
    iteration k's backward and fail there, far from the cause."""

    decoder = _tiny_decoder()
    tokens, images, key_xyz = _refinement_case(grid=(2, 3))
    slots = tokens.shape[1]
    height, width = images.shape[-2:]
    field = torch.zeros(1, slots, height, width, 3)
    with pytest.raises(ValueError, match="needs at least 16 key points"):
        decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0,
            refinement=_refinement(key_xyz, 0, field),
        )
    # The bound is "at least", not "more than": a per-slot key set of exactly
    # k points is admissible (topk over k keys is defined and the read fits
    # read_proj), and the 56x56 fixture scenes' 4x4 grids sit exactly there.
    tokens, images, key_xyz = _refinement_case(grid=(2, 8))
    assert tokens.shape[2] - 2 == REFINER_NEIGHBOURS
    slots = tokens.shape[1]
    height, width = images.shape[-2:]
    boundary = decoder(
        tokens, images=images, patch_start_idx=2, track_query_idx=0,
        refinement=_refinement(key_xyz, 0, torch.zeros(1, slots, height, width, 3)),
    )
    assert boundary.shape == (1, slots, 1 + REFINER_NEIGHBOURS, 64)

    tokens, images, key_xyz = _refinement_case()
    slots = tokens.shape[1]
    height, width = images.shape[-2:]
    with pytest.raises(ValueError, match="one row per decoder output row"):
        decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0,
            views_per_time=2, merge=True,
            refinement=_refinement(key_xyz, 0, torch.zeros(1, slots, height, width, 3)),
        )
    wide = torch.cat([key_xyz, key_xyz[:, :, :1]], dim=2)
    with pytest.raises(ValueError, match="one point per key patch token"):
        decoder(
            tokens, images=images, patch_start_idx=2, track_query_idx=0,
            refinement=RefinementInput(
                previous_field=torch.zeros(1, slots, height, width, 3),
                anchor_xyz=wide[:, 0],
                key_xyz=wide,
            ),
        )

    good = dict(
        previous_field=torch.zeros(1, slots, height, width, 3),
        anchor_xyz=key_xyz[:, 0],
        key_xyz=key_xyz,
    )
    RefinementInput(**good)
    for key, bad, message in (
        ("previous_field", torch.zeros(1, slots, height, width), r"\(B, R, H, W, 3\)"),
        ("anchor_xyz", key_xyz[:, 0, :1], "patches per slot"),
        ("key_xyz", key_xyz[:, :, :, :2], r"\(B, S, P, 3\)"),
        ("key_xyz", key_xyz.expand(2, *key_xyz.shape[1:]), "disagree on B"),
        ("previous_field", good["previous_field"].clone().requires_grad_(True), "must be detached"),
    ):
        with pytest.raises(ValueError, match=message):
            RefinementInput(**{**good, key: bad})


# ------------------------------------------- inside the MotionDecoder ---


@pytest.mark.parametrize("merge", [False, True])
def test_refinement_zero_init_is_inert_and_the_branch_executes(merge):
    """The **kwargs-trap killer, in the shape of
    test_depth_branch_executes_and_zero_init_is_inert: a `refinement`
    swallowed or never read is indistinguishable from a working zero-init
    branch -- both leave the output byte-identical. So beyond inertness this
    asserts the branch EXECUTED, separately for its two zero-init tensors: a
    filled read_proj must move the output (the kNN read reached the query
    patches), a filled field_embed must move it (the previous field reached
    them), and the converse control -- filled weights, no refinement passed
    -- must still be today's output, or the inertness equality was "never
    read". The gradient arm then pins the zero-init SADDLE rather than a
    not-None check, which a detached correlation would pass: with the output
    side at zero, field_embed and read_proj receive nonzero gradient while
    query_proj and key_proj receive EXACTLY zero, and once read_proj holds
    hand squares the two projections receive gradient too -- which is the
    reason they keep their default init. Both decoder branches, since each
    builds its own query tensor and the hoisted flatten sits after them."""

    views, times = 2, 2
    tokens, images, key_xyz = _refinement_case(views=views, times=times)
    slots = tokens.shape[1]
    rows = times if merge else slots
    height, width = images.shape[-2:]
    torch.manual_seed(1)
    previous = torch.randn(1, rows, height, width, 3)
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=1)
    if merge:
        kwargs.update(views_per_time=views, merge=True)
    decoder = _tiny_decoder()

    base = decoder(tokens, **kwargs)
    assert base.shape == (1, rows, 1 + 18, 64)
    inert = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    assert torch.equal(inert, base)

    with torch.no_grad():
        decoder.refiner.read_proj.weight.fill_(1.0)
    hot_read = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    assert not torch.equal(hot_read, base)
    # The time token is the AdaLN condition and never receives the term.
    assert torch.equal(hot_read[:, :, 0], base[:, :, 0])
    assert torch.equal(decoder(tokens, **kwargs), base)

    with torch.no_grad():
        decoder.refiner.read_proj.weight.zero_()
        decoder.refiner.field_embed.weight.fill_(1.0)
    hot_field = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    assert not torch.equal(hot_field, base)
    assert not torch.equal(hot_field, hot_read)
    assert torch.equal(decoder(tokens, **kwargs), base)

    with torch.no_grad():
        decoder.refiner.field_embed.weight.zero_()
    out = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    out.sum().backward()
    for module in (decoder.refiner.read_proj, decoder.refiner.field_embed):
        assert float(module.weight.grad.abs().sum()) > 0
        assert float(module.bias.grad.abs().sum()) > 0
    # The saddle: the correlation reaches the output only through read_proj's
    # zero columns, so the projections' gradient is exactly zero -- present,
    # which assert_trainable_gradients_finite requires of every trainable
    # tensor, but a witness of nothing.
    for module in (decoder.refiner.query_proj, decoder.refiner.key_proj):
        assert torch.count_nonzero(module.weight.grad) == 0
        assert torch.count_nonzero(module.bias.grad) == 0

    decoder.zero_grad(set_to_none=True)
    _square_fill(decoder.refiner.read_proj.weight)
    out = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    assert not torch.equal(out, base)
    out.sum().backward()
    for module in (decoder.refiner.query_proj, decoder.refiner.key_proj):
        assert float(module.weight.grad.abs().sum()) > 0


def _indexed_term(shape):
    """A term whose value names its own (row, patch, channel): 10000*row + 100*patch + channel."""

    B, R, P, C = shape
    rows = torch.arange(R, dtype=torch.float32).view(1, R, 1, 1)
    cols = torch.arange(P, dtype=torch.float32).view(1, 1, P, 1)
    chans = torch.arange(C, dtype=torch.float32).view(1, 1, 1, C)
    return (10_000.0 * rows + 100.0 * cols + chans).expand(B, R, P, C).contiguous()


class _IndexedTermRefiner(nn.Module):
    """Stands in the refiner seat, records every argument, returns _indexed_term."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, query_patches, patches, refinement, *, views_per_time, merge):
        self.seen.append((query_patches, patches, refinement, views_per_time, merge))
        return _indexed_term(query_patches.shape)


class _QueryRecorder(nn.Module):
    """Wraps a cross block and records the query it is handed."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.seen = []

    def forward(self, query, kv, pos=None):
        self.seen.append(query)
        return self.block(query, kv, pos=pos)


@pytest.mark.parametrize("merge", [False, True])
def test_the_decoder_hands_the_refiner_the_anchor_patches_and_adds_the_term_in_place(merge):
    """Which tensors the decoder hands the refiner, and WHERE its term lands,
    pinned by recorders rather than inferred from a moved output. The
    inertness test above proves the branch executes; nothing there says the
    refiner saw the right tensors or that the term reached the patch it was
    computed for -- a refiner handed each row's OWN patches instead of the
    tracked anchor's, or the pooled keys instead of the slot-major patches,
    or a term added one patch over, all left that test green and K=1
    bit-identical. A stand-in refiner records both tensor arguments and
    returns a term whose value names its own (row, patch, channel); a
    recorder in the first cross-block seat sees the query the block loop
    receives. The anchor is slot 1, not 0, so a refiner fed slot 0's patches
    cannot pass by coincidence. The three rolls refuse a term landed one
    row, one patch or one channel over, which a shape check cannot; the
    time token must be untouched; and the flag-off call must reach the
    block with today's query and never call the refiner. Both branches,
    since each builds its own query."""

    views, times = 2, 2
    tokens, images, key_xyz = _refinement_case(views=views, times=times)
    slots = tokens.shape[1]
    patches = tokens.shape[2] - 2
    width = tokens.shape[-1]
    rows = times if merge else slots
    height, image_width = images.shape[-2:]
    anchor = 1  # Not 0: a refiner handed slot 0's patches would pass at anchor 0.
    decoder = _tiny_decoder()
    refiner = _IndexedTermRefiner()
    decoder.refiner = refiner
    recorder = _QueryRecorder(decoder.cross_blocks[0])
    decoder.cross_blocks[0] = recorder
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=anchor)
    if merge:
        kwargs.update(views_per_time=views, merge=True)
        expected_query, _ = merge_time_grouped_tokens(
            tokens, patch_start_idx=2, track_query_idx=anchor, views_per_time=views
        )
    else:
        expected_query = torch.cat(
            [tokens[:, :, 1:2], tokens[:, anchor : anchor + 1, 2:].expand(1, slots, patches, width)],
            dim=2,
        )
    previous = torch.zeros(1, rows, height, image_width, 3)

    decoder(tokens, **kwargs)
    assert refiner.seen == []
    assert torch.equal(recorder.seen[-1], expected_query.flatten(0, 1))

    refinement = _refinement(key_xyz, anchor, previous)
    decoder(tokens, **kwargs, refinement=refinement)
    ((query_patches, seen_patches, seen_refinement, seen_views, seen_merge),) = refiner.seen
    assert seen_refinement is refinement
    assert seen_views == (views if merge else 1) and seen_merge is merge
    assert query_patches.shape == (1, rows, patches, width)
    assert torch.equal(query_patches, expected_query[:, :, 1:])
    assert torch.equal(seen_patches, tokens[:, :, 2:])
    term = _indexed_term((1, rows, patches, width))
    expected = torch.cat([expected_query[:, :, :1], expected_query[:, :, 1:] + term], dim=2)
    assert torch.equal(recorder.seen[-1], expected.flatten(0, 1))
    for rolled in (term.roll(1, dims=1), term.roll(1, dims=2), term.roll(1, dims=3)):
        wrong = torch.cat([expected_query[:, :, :1], expected_query[:, :, 1:] + rolled], dim=2)
        assert not torch.equal(recorder.seen[-1], wrong.flatten(0, 1))


@pytest.mark.parametrize("merge", [False, True])
def test_the_field_term_lands_on_the_patch_whose_block_it_embeds(merge):
    """The field path by hand: with a one-hot centre kernel on the three
    field channels and read_proj at its zero init, the term at (row, patch)
    must be the previous field's centre pixel of THAT block, in PatchEmbed's
    flatten order, with every other channel exactly zero. The centre values
    are distinct squares per (row, patch), so a column-major flatten, a
    transposed permute before the conv, or a term shifted by one patch or
    one row moves a digit -- a fixture with one patch or with equal rows
    would pass all of them, which is how the Stage B one-patch fixture let
    a landing bug through. Both branches, since the row count differs."""

    views, times = 2, 2
    tokens, images, key_xyz = _refinement_case(views=views, times=times)
    slots = tokens.shape[1]
    patches = tokens.shape[2] - 2
    width = tokens.shape[-1]
    rows = times if merge else slots
    height, image_width = images.shape[-2:]
    columns = image_width // _PATCH
    refiner = TrackRefiner(width, _PATCH)
    with torch.no_grad():
        for channel in range(3):
            refiner.field_embed.weight[channel, channel, _PATCH // 2, _PATCH // 2] = 1.0
    previous = torch.zeros(1, rows, height, image_width, 3)
    for row in range(rows):
        for patch in range(patches):
            pixel_row = _PATCH // 2 + _PATCH * (patch // columns)
            pixel_col = _PATCH // 2 + _PATCH * (patch % columns)
            previous[0, row, pixel_row, pixel_col] = torch.tensor(
                [float((row + 1) ** 2), float((patch + 1) ** 2), float(((row + 1) * (patch + 2)) ** 2)]
            )
    query_patches = tokens[:, 1:2, 2:].expand(1, rows, patches, width).contiguous()

    term = refiner(
        query_patches, tokens[:, :, 2:], _refinement(key_xyz, 1, previous),
        views_per_time=(views if merge else 1), merge=merge,
    )

    expected = previous[0, :, _PATCH // 2 :: _PATCH, _PATCH // 2 :: _PATCH].reshape(rows, patches, 3)
    assert torch.equal(term[0, :, :, :3], expected)
    assert torch.count_nonzero(term[0, :, :, 3:]) == 0
    assert not torch.equal(term[0, :, :, :3], expected.roll(1, dims=1))
    assert not torch.equal(term[0, :, :, :3], expected.roll(1, dims=0))


@pytest.mark.parametrize("merge", [False, True])
def test_a_non_finite_carry_pixel_reads_as_zero_displacement(merge):
    """A non-finite carry pixel is a pixel the loss tolerates today:
    sparse_tracking_loss guards supervised pixels only, so a NaN or inf the
    head emits where no correspondence lands costs a K=1 run nothing. At
    K > 1 that pixel is the next pass's carry, and zero-init alone does not
    make the refiner inert on it -- 0 * NaN is NaN, so field_embed's term
    for that block and, through attention, every token of the pass go
    non-finite, and the next loss raises FloatingPointError at supervised
    pixels for a pixel it never reads; run_training catches no such error,
    so the run dies. The refiner therefore reads every non-finite carry
    pixel as zero displacement, the iteration-0 value. Pinned on both sides:
    the zero-init refiner stays bit-inert on a carry with a NaN at one block
    CENTRE (the pixel the kNN estimate reads), an inf inside another block
    and a -inf in the last row; and with both output layers filled, the
    poisoned carry gives exactly the output of the same carry with those
    three pixels zeroed, finite everywhere, and not the output of the carry
    with them set to one -- so the pixels are read, and read as zero."""

    views, times = 2, 2
    tokens, images, key_xyz = _refinement_case(views=views, times=times)
    slots = tokens.shape[1]
    rows = times if merge else slots
    height, width = images.shape[-2:]
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=1)
    if merge:
        kwargs.update(views_per_time=views, merge=True)
    torch.manual_seed(5)
    clean = torch.randn(1, rows, height, width, 3)
    pixels = (
        (0, _PATCH // 2, _PATCH // 2 + _PATCH),  # the centre of block (0, 1)
        (1, 3, 2 * _PATCH + 5),  # inside block (0, 2), off its centre
        (rows - 1, height - 1, width - 1),
    )
    poisoned, zeroed, ones = clean.clone(), clean.clone(), clean.clone()
    for (row, y, x), value in zip(pixels, (float("nan"), float("inf"), float("-inf"))):
        poisoned[0, row, y, x] = value
        zeroed[0, row, y, x] = 0.0
        ones[0, row, y, x] = 1.0
    assert not torch.isfinite(poisoned).all()

    decoder = _tiny_decoder()
    base = decoder(tokens, **kwargs)
    inert = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, poisoned))
    assert torch.equal(inert, base)

    with torch.no_grad():
        decoder.refiner.read_proj.weight.fill_(1.0)
        decoder.refiner.field_embed.weight.fill_(1.0)
    with_poison = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, poisoned))
    assert torch.isfinite(with_poison).all()
    assert not torch.equal(with_poison, base)
    assert torch.equal(
        with_poison, decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, zeroed))
    )
    assert not torch.equal(
        with_poison, decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, ones))
    )


def test_the_correlation_carries_gradient_into_the_tokens():
    """The correlation is trained THROUGH, not only read. With the field path
    at zero (a zero carry, field_embed at its zero init) and read_proj
    filled, the term's gradient reaches the query patches through query_proj
    and the key patches through key_proj and the gather -- so a `.detach()`
    on either projection's input, on the gathered neighbour features or on
    the correlation itself fails here, while every weight-gradient assertion
    in the inertness test still passes, because a Linear's weight takes its
    gradient from a detached input just the same. With read_proj at zero both
    token gradients are exactly zero: the saddle, seen from the token side.
    The decoder-level arm says the same of the decoder's own input tokens,
    whose gradient under the refined pass must differ from the flag-off
    pass's."""

    tokens, images, key_xyz = _refinement_case(views=2, times=2)
    slots = tokens.shape[1]
    patches = tokens.shape[2] - 2
    width = tokens.shape[-1]
    height, image_width = images.shape[-2:]
    refiner = TrackRefiner(width, _PATCH)
    zeros = torch.zeros(1, slots, height, image_width, 3)

    def token_gradients():
        query_patches = (
            tokens[:, 1:2, 2:].expand(1, slots, patches, width).contiguous().requires_grad_(True)
        )
        key_patches = tokens[:, :, 2:].clone().requires_grad_(True)
        term = refiner(
            query_patches, key_patches, _refinement(key_xyz, 1, zeros),
            views_per_time=1, merge=False,
        )
        term.sum().backward()
        return query_patches.grad, key_patches.grad

    query_grad, key_grad = token_gradients()
    assert query_grad is not None and key_grad is not None
    assert torch.count_nonzero(query_grad) == 0
    assert torch.count_nonzero(key_grad) == 0

    _square_fill(refiner.read_proj.weight)
    query_grad, key_grad = token_gradients()
    assert float(query_grad.abs().sum()) > 0
    assert float(key_grad.abs().sum()) > 0
    # A zero carry feeds field_embed nothing: the token gradient above came
    # through the correlation alone.
    assert torch.count_nonzero(refiner.field_embed.weight.grad) == 0

    decoder = _tiny_decoder()
    decoder_kwargs = dict(images=images, patch_start_idx=2, track_query_idx=1)

    def decoder_token_gradient(refinement):
        leaf = tokens.clone().requires_grad_(True)
        decoder.zero_grad(set_to_none=True)
        if refinement is None:
            out = decoder(leaf, **decoder_kwargs)
        else:
            out = decoder(leaf, **decoder_kwargs, refinement=refinement)
        out.sum().backward()
        return leaf.grad

    flag_off = decoder_token_gradient(None)
    _square_fill(decoder.refiner.read_proj.weight)
    refined = decoder_token_gradient(_refinement(key_xyz, 1, zeros))
    assert not torch.equal(refined, flag_off)


def test_a_moved_slot_cloud_moves_only_that_rows_output():
    """Row r reads row r's own keys and nothing else: the anchor's points
    displaced by row r's field, matched against slot r's cloud on the
    per-slot path and against time r's V pooled clouds under the merge. With
    read_proj filled, moving one slot's cloud far away changes that slot's
    row (its neighbours leave) and leaves every other row bit-identical -- a
    refiner that pooled every slot's points into one cloud, or misindexed
    rows against slots, moves the other rows too, because the moved points
    were in their neighbourhoods. Flipping the slot order instead would not
    separate the two: it also breaks the point-to-token pairing of a pooled
    implementation, so both would change. The anchor stays fixed and is not
    the moved slot."""

    tokens, images, key_xyz = _refinement_case(views=2, times=2)
    slots = tokens.shape[1]
    height, width = images.shape[-2:]
    decoder = _tiny_decoder()
    with torch.no_grad():
        decoder.refiner.read_proj.weight.fill_(1.0)

    torch.manual_seed(2)
    previous = torch.randn(1, slots, height, width, 3)
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=0)
    ordered = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 0, previous))
    moved = key_xyz.clone()
    moved[:, 2] += 1e3
    shifted = decoder(
        tokens,
        **kwargs,
        refinement=RefinementInput(
            previous_field=previous, anchor_xyz=key_xyz[:, 0], key_xyz=moved
        ),
    )
    for row in range(slots):
        assert torch.equal(shifted[0, row], ordered[0, row]) is (row != 2), row

    previous = torch.randn(1, 2, height, width, 3)
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=0, views_per_time=2, merge=True)
    ordered = decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 0, previous))
    moved = key_xyz.clone()
    moved[:, 3] += 1e3  # camera 1, time 1: pooled into row 1 only
    shifted = decoder(
        tokens,
        **kwargs,
        refinement=RefinementInput(
            previous_field=previous, anchor_xyz=key_xyz[:, 0], key_xyz=moved
        ),
    )
    assert torch.equal(shifted[0, 0], ordered[0, 0])
    assert not torch.equal(shifted[0, 1], ordered[0, 1])


def test_the_merged_adaln_condition_views_the_rebuilt_query():
    """Memory, not values: under merge + AdaLN the condition is a view of the
    query, AdaLNBlock's SiLU saves its input, and a condition sliced before
    the refinement insert would keep the whole pre-insert (B, T, 1+P, C)
    storage alive until the iteration's backward for nothing -- about 143 MB
    per tap at the production window. A recorder in the self-block seat sees
    what the block sees: with a filled read_proj the condition's base tensor
    carries the POST-insert patch columns (and the untouched column 0), so it
    views the rebuilt query; without a refinement it still views the
    pre-insert query, exactly as at HEAD. Values are identical either way,
    which is why nothing but a recorder can tell."""

    tokens, images, key_xyz = _refinement_case()
    decoder = _tiny_decoder()
    conditions = []

    class _Recorder(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x, cond, pos=None, attn_mask=None):
            conditions.append(cond)
            return self.block(x, cond, pos=pos, attn_mask=attn_mask)

    decoder.self_blocks[0] = _Recorder(decoder.self_blocks[0])
    kwargs = dict(images=images, patch_start_idx=2, track_query_idx=1, views_per_time=2, merge=True)
    expected_query, _ = merge_time_grouped_tokens(
        tokens, patch_start_idx=2, track_query_idx=1, views_per_time=2
    )

    decoder(tokens, **kwargs)
    flag_off = conditions[-1]
    assert flag_off._base is not None
    assert torch.equal(flag_off._base, expected_query)

    with torch.no_grad():
        decoder.refiner.read_proj.weight.fill_(1.0)
    height, width = images.shape[-2:]
    torch.manual_seed(4)
    previous = torch.randn(1, 2, height, width, 3)
    decoder(tokens, **kwargs, refinement=_refinement(key_xyz, 1, previous))
    with_refinement = conditions[-1]
    assert with_refinement._base is not None
    assert with_refinement._base.shape == expected_query.shape
    assert torch.equal(with_refinement._base[:, :, 0], expected_query[:, :, 0])
    assert not torch.equal(with_refinement._base[:, :, 1:], expected_query[:, :, 1:])
    assert torch.equal(with_refinement, flag_off)


# --------------------------------------------------- Arc owns the loop ---


class _RecordingRefineDecoder(_FakeMotionDecoder):
    """_FakeMotionDecoder plus the refinement keyword, recorded per call, and
    a real parameter so the returned track carries grad history -- without
    one, "the carry is detached" would hold for a tensor that never had a
    graph. The value depends on the carry, so iterations differ from a zero
    start and the next iteration's carry is exactly this iteration's field."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.seen_kwargs = []
        self.seen_refinements = []

    def forward(self, tokens, images, patch_start_idx, track_query_idx, **kwargs):
        self.seen_kwargs.append(sorted(kwargs))
        self.seen_views_per_time.append(kwargs.get("views_per_time", 1))
        self.seen_merge.append(kwargs.get("merge", False))
        refinement = kwargs.get("refinement")
        carry = 0.0
        if refinement is not None:
            field = refinement.previous_field
            self.seen_refinements.append(
                (
                    field.detach().clone(),
                    field.requires_grad,
                    field.dtype,
                    refinement.anchor_xyz.detach().clone(),
                    refinement.key_xyz.detach().clone(),
                )
            )
            carry = float(field.flatten()[0])
        B, S, _, C = tokens.shape
        views_per_time = kwargs.get("views_per_time", 1)
        rows = S // views_per_time if kwargs.get("merge", False) else S
        value = 1.0 + float(track_query_idx) + 0.5 * carry
        return torch.full((B, rows, 2, C), value, dtype=tokens.dtype) * self.gain


class _UnitDepthHead(nn.Module):
    """A depth the real predicted_pointmaps can unproject (all ones), on the
    fake head's (feats, H, W, patch_start_idx) surface."""

    def forward(self, feats, H, W, patch_start_idx):
        B, S = feats[0][0].shape[:2]
        return {"depth": torch.ones(B, S, H, W)}


class _SquareTranslationPoseDecoder(nn.Module):
    """A VALID pose encoding where _FakeCameraDecoder returns a one-wide
    zero: identity rotation (xyzw = 0, 0, 0, 1), unit fov, and slot s
    translated by s^2 along x. A zero quaternion with zero fov unprojects to
    NaN; an identical pose on every slot unprojects to one cloud repeated,
    under which "the anchor point is the anchor SLOT's cloud" holds for any
    slot index at all. Squares, so no two slot clouds are a shift of a third."""

    def forward(self, camera_tokens):
        B, S = camera_tokens.shape[:2]
        pose = torch.zeros(B, S, 9)
        pose[..., 6] = 1.0
        pose[..., 7:] = 1.0
        pose[..., 0] = torch.arange(S, dtype=torch.float32).square()
        return pose


def _refining_shell(*, cloud):
    """An _arc_shell with recording fakes; ``cloud`` picks reconstruction
    fakes predicted_pointmaps can unproject, or the suite's originals, which
    it refuses (no depth, one-wide pose) -- the K=1 witness that the cloud is
    never built."""

    model = _arc_shell(max_time_indices=32)
    model.backbone = _FakeBackbone()
    model.head = _UnitDepthHead() if cloud else _FakeReconstructionHead()
    model.cam_dec = _SquareTranslationPoseDecoder() if cloud else _FakeCameraDecoder()
    model.motion_decoder = _RecordingRefineDecoder()
    model.track_head = _FakeTrackHead()
    calls = []
    real = model.track_for_query

    def recording(feats, x, query_idx, **kwargs):
        calls.append((query_idx, sorted(kwargs)))
        return real(feats, x, query_idx, **kwargs)

    model.track_for_query = recording
    return model, calls


def _views(count=4, *, batch=1, height=28, width=42, anchors=(0, 3), times=None):
    views = [{"img": torch.zeros(batch, 3, height, width)} for _ in range(count)]
    inference_cli.attach_frame_metadata(
        views,
        track_query_idx=list(anchors),
        time_indices=list(range(count)) if times is None else list(times),
    )
    return views


def _expected_cloud(model, slots, height, width):
    return patch_centre_points(
        predicted_pointmaps(
            {
                "depth": torch.ones(1, slots, height, width),
                "pose_enc": model.cam_dec(torch.zeros(1, slots, 1)),
            }
        ),
        Arc.PATCH_SIZE,
    )


def test_arc_forward_loops_one_iteration_and_detaches_the_carry(monkeypatch):
    """Arc._forward owns the loop: K calls of the single-iteration head per
    query, iteration 0 conditioned on an all-zero field, iteration k on
    iteration k-1's own output DETACHED -- torch.equal to the prior track and
    requires_grad False, on a track that does carry grad history. A loop that
    re-used iteration 0's zeros for every iteration passes the count and
    fails the equality. The last iteration is the reported field; every
    iteration is stacked under track_multi_iterations on a K axis, for the
    eval's per-iteration losses, and the stack carries the caller's grad
    context like track_multi does. The cloud reaches the head as patch
    centres of the model's OWN prediction, recomputed here from the same
    fakes through the real predicted_pointmaps, and the anchor point is that
    cloud at the anchor SLOT -- distinguishable from every other slot's, so
    an anchor read at index 0 or at the wrong slot fails. The cloud is built
    ONCE per forward, not once per query or pass: it depends on neither, and
    at production each build is a 378x504 unprojection of every slot."""

    import arc.models.arc.arc as arc_module

    unprojections = []
    real_predicted_pointmaps = arc_module.predicted_pointmaps

    def counting(raw_predictions):
        unprojections.append(True)
        return real_predicted_pointmaps(raw_predictions)

    monkeypatch.setattr(arc_module, "predicted_pointmaps", counting)
    model, calls = _refining_shell(cloud=True)
    output = model(_views(), force_no_output_conversion=True, refine_iters=3)

    assert unprojections == [True]
    slots, height, width = 4, 28, 42
    assert [query for query, _ in calls] == [0, 0, 0, 3, 3, 3]
    assert all(kwargs == ["merge", "refinement", "views_per_time"] for _, kwargs in calls)
    assert output["track_multi"].shape == (1, 2, slots, height, width, 3)
    assert output["conf_track_multi"].shape == (1, 2, slots, height, width)
    iterations = output["track_multi_iterations"]
    assert iterations.shape == (1, 2, 3, slots, height, width, 3)
    assert torch.equal(iterations[:, :, -1], output["track_multi"])
    assert not torch.equal(iterations[:, :, 0], iterations[:, :, 1])
    assert not torch.equal(iterations[:, :, 1], iterations[:, :, 2])
    assert output["track_multi"].requires_grad
    assert iterations.requires_grad

    expected_cloud = _expected_cloud(model, slots, height, width)
    assert expected_cloud.shape == (1, slots, 6, 3)
    assert torch.isfinite(expected_cloud).all()
    for slot in range(1, slots):
        assert not torch.equal(expected_cloud[:, 0], expected_cloud[:, slot])

    decoder = model.motion_decoder
    # 2 queries x 3 iterations x 4 taps, taps innermost.
    assert len(decoder.seen_refinements) == 24
    assert all(
        kwargs == ["merge", "refinement", "views_per_time"]
        for kwargs in decoder.seen_kwargs
    )
    for query_index, query in enumerate((0, 3)):
        for iteration in range(3):
            for tap in range(4):
                field, requires_grad, dtype, anchor_xyz, key_xyz = (
                    decoder.seen_refinements[query_index * 12 + iteration * 4 + tap]
                )
                assert field.shape == (1, slots, height, width, 3)
                assert dtype is torch.float32
                assert requires_grad is False
                if iteration == 0:
                    assert torch.count_nonzero(field) == 0
                else:
                    assert torch.equal(
                        field, iterations[:, query_index, iteration - 1].detach()
                    )
                assert torch.equal(key_xyz, expected_cloud)
                assert torch.equal(anchor_xyz, expected_cloud[:, query])
                assert not torch.equal(anchor_xyz, expected_cloud[:, 1])


@pytest.mark.parametrize(
    "time_indices, views", [((0, 1, 0, 1), 2), ((0, 1, 2, 3), 1)]
)
def test_arc_forward_merges_the_carry_rows_by_time(time_indices, views):
    """The merged arm of the loop: the head emits one row per time, T = S / V,
    so the zero carry and every later carry are (1, T, H, W, 3), the stack's
    row axis is T, and the head is called with merge=True beside the
    refinement -- while the cloud stays slot-major (every slot's points, S of
    them) and the anchor is still the anchor SLOT's cloud, which is what the
    refiner pools by time itself. A carry sized on S here would fail the
    refiner's row check on the real decoder and pass any fake that ignores
    it. Two layouts: two cameras observing two times, and one camera
    observing four -- the merge is defined at V=1 too (the pooled key row is
    the slot's own patches behind its camera token), and a refinement that
    fell back to the per-slot branch there would change the head geometry of
    a single-camera window under a flag that promises the merged one."""

    model, calls = _refining_shell(cloud=True)
    output = model(
        _views(times=time_indices, anchors=(0, 3)),
        force_no_output_conversion=True,
        merge_synchronized_slots=True,
        refine_iters=2,
    )

    slots, height, width = 4, 28, 42
    times = slots // views
    assert [query for query, _ in calls] == [0, 0, 3, 3]
    assert output["track_multi"].shape == (1, 2, times, height, width, 3)
    iterations = output["track_multi_iterations"]
    assert iterations.shape == (1, 2, 2, times, height, width, 3)
    assert torch.equal(iterations[:, :, -1], output["track_multi"])
    decoder = model.motion_decoder
    assert decoder.seen_merge == [True] * 16
    assert decoder.seen_views_per_time == [views] * 16
    expected_cloud = _expected_cloud(model, slots, height, width)
    for query_index, query in enumerate((0, 3)):
        for iteration in range(2):
            for tap in range(4):
                field, _, _, anchor_xyz, key_xyz = decoder.seen_refinements[
                    query_index * 8 + iteration * 4 + tap
                ]
                assert field.shape == (1, times, height, width, 3)
                if iteration == 0:
                    assert torch.count_nonzero(field) == 0
                else:
                    assert torch.equal(field, iterations[:, query_index, 0].detach())
                assert key_xyz.shape == (1, slots, 6, 3)
                assert torch.equal(key_xyz, expected_cloud)
                assert torch.equal(anchor_xyz, expected_cloud[:, query])


def test_arc_forward_at_one_iteration_keeps_todays_spelling_and_never_builds_the_cloud():
    """The flag-off path is today's code, not K=1 of the new loop: the head is
    called with today's exact keywords (no `refinement`), the decoder with
    today's spelling, no track_multi_iterations key is emitted, and the
    cloud is never built -- proven by fakes whose reconstruction has no depth
    and whose pose is one channel wide, which predicted_pointmaps refuses.
    Injected fakes across three suites bind this surface; a K=1 that built
    the cloud would also pay a 378x504 unprojection per step for nothing."""

    model, calls = _refining_shell(cloud=False)
    output = model(_views(), force_no_output_conversion=True)
    assert [query for query, _ in calls] == [0, 3]
    assert all(kwargs == ["views_per_time"] for _, kwargs in calls)
    assert all(kwargs == [] for kwargs in model.motion_decoder.seen_kwargs)
    assert model.motion_decoder.seen_refinements == []
    assert "track_multi_iterations" not in output
    assert output["track_multi"].shape == (1, 2, 4, 28, 42, 3)

    model, calls = _refining_shell(cloud=False)
    model(_views(), force_no_output_conversion=True, refine_iters=1)
    assert all(kwargs == ["views_per_time"] for _, kwargs in calls)
    assert model.motion_decoder.seen_refinements == []


def test_refine_iters_is_validated_before_the_encoder_runs():
    """0 would run no head at all and index an empty list; a bool is an int
    subclass and True would silently mean one iteration; a float count would
    silently truncate in range(). Refused at the top of _forward, before
    encode_features -- the fake backbone records the time indices it is
    handed, and after a refusal it has seen none -- so a bad K costs no
    encoder pass, and so the cloud-less fakes never reach
    predicted_pointmaps' KeyError. K > 1 at B != 1 is refused for the
    B=1 helper's sake, but only when the track heads run: with
    inference_track False nothing reads the cloud and the forward returns."""

    model, _ = _refining_shell(cloud=False)
    with pytest.raises(ValueError, match="refine_iters"):
        model(_views(), force_no_output_conversion=True, refine_iters=0)
    for bad in (True, 2.0):
        with pytest.raises(TypeError, match="refine_iters"):
            model(_views(), force_no_output_conversion=True, refine_iters=bad)
    assert model.backbone.seen_time_indices is None

    with pytest.raises(ValueError, match="batch size 1"):
        model(_views(batch=2), force_no_output_conversion=True, refine_iters=2)
    assert model.backbone.seen_time_indices is None
    output = model(
        _views(batch=2),
        force_no_output_conversion=True,
        inference_track=False,
        refine_iters=2,
    )
    assert "track_multi" not in output
    assert model.backbone.seen_time_indices is not None

    # And at one pass the rule does not bind at all: a batch of two tracked
    # at K=1 is today's forward, which never read the cloud and never refused
    # a batch.
    for extra in ({}, {"refine_iters": 1}):
        model, _ = _refining_shell(cloud=False)
        output = model(_views(batch=2), force_no_output_conversion=True, **extra)
        assert output["track_multi"].shape == (2, 2, 4, 28, 42, 3)


# ----------------------------------------------------------- retention ---


class _Held:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor):
        self.tensor = tensor


def test_per_iteration_backward_bounds_retention_to_one_head_pass():
    """Why the loop is in the caller and backwards per iteration: detaching
    the carry alone does not bound memory -- iteration k's graph stays alive
    until something backwards it. Measured on CPU in the
    test_dpt_head_checkpointing idiom, but with the pack hook handing autograd
    a weakly-tracked wrapper, so the live set shrinks when a graph is freed
    and "bytes retained right now" is what is summed. Per-iteration backward:
    bytes live after iteration 2's forward equal iteration 1's exactly (same
    module, same shapes, iteration 1's graph gone). Summing both losses
    before one backward keeps both graphs: measured 1.88x on this stand-in
    (11,819,392 against 22,226,304 bytes; below 2x because parameter
    storages are shared), asserted above 1.5x. The trainer's modes --
    train(), frames_chunk_size 1 -- so the DPT chunks are checkpointed as in
    a real step, and the real refiner sits in the decoder."""

    torch.manual_seed(0)
    decoder = MotionDecoder(
        patch_size=_PATCH, embed_dim=64, depth=2, num_heads=4, use_adaln=True
    )
    head = DPTHead(
        dim_in=64,
        output_dim=4,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )
    decoder.train()
    head.train()
    tokens, images, key_xyz = _refinement_case()
    tokens.requires_grad_(True)
    slots = tokens.shape[1]
    height, width = images.shape[-2:]

    live = weakref.WeakSet()

    def pack(tensor):
        held = _Held(tensor)
        live.add(held)
        return held

    def unpack(held):
        return held.tensor

    def live_bytes():
        storages = {}
        for held in list(live):
            storage = held.tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())

    def iteration(previous_field):
        refinement = RefinementInput(
            previous_field=previous_field, anchor_xyz=key_xyz[:, 0], key_xyz=key_xyz
        )
        taps = [
            decoder(
                tokens,
                images=images,
                patch_start_idx=2,
                track_query_idx=0,
                refinement=refinement,
            )
            for _ in range(4)
        ]
        preds, _ = head(taps, images=images, patch_start_idx=1, frames_chunk_size=1)
        return preds

    zeros = torch.zeros(1, slots, height, width, 3)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        first = iteration(zeros)
        after_first = live_bytes()
        first.square().mean().backward()
        second = iteration(first.detach())
        after_second = live_bytes()
        second.square().mean().backward()
        del first, second

        first = iteration(zeros)
        second = iteration(first.detach())
        summed = live_bytes()
        (first.square().mean() + second.square().mean()).backward()
        del first, second

    assert after_first > 0
    assert after_second == after_first, (after_first, after_second)
    assert summed > 1.5 * after_first, (after_first, summed)


# --------------------------------------------------------- the trainer ---


class _RefiningFakeArc(_FakeArc):
    """_FakeArc plus the refinement surface train_step and evaluate_held_out
    reach for at --refine_iters > 1, and nothing else. Pass k's field is the
    undisplaced field plus a constant c_k that is DISTINCT per pass and read
    off the carry itself: c_0 = 0 (the zero carry), and c_k = offset + c_{k-1}
    / 2, with c_{k-1} recovered as the mean of carry minus this pass's own
    undisplaced field -- offset, 1.5 offset, 1.75 offset, ... -- so a carry
    from the wrong pass or the wrong anchor is a wrong field and a wrong
    loss, and no two passes coincide: a permutation of the pass axis fails.
    ``force_offset`` makes a K=1 step produce pass k's field for every anchor,
    which is how that pass's loss and gradient are computed by hand;
    ``applied_offsets`` records the constant each head call used, so the hand
    runs reproduce the K > 1 fields bit for bit; ``pass_offsets`` is the same
    sequence in closed form, for the eval fake, which never sees a carry."""

    PATCH_SIZE = 14

    def __init__(self, observations, height, width, views_per_time=1, *, iteration_offset=0.01):
        super().__init__(observations, height, width, views_per_time=views_per_time)
        self.iteration_offset = iteration_offset
        self.force_offset = None
        # (y, x): a pixel every field emitted by `forward` carries a NaN at,
        # for the eval's non-finite-pixel arm; None leaves the fields finite.
        self.poison_pixel = None
        # When set, the confidence channel is +inf on a checkerboard of pixels
        # whenever the pass constant is exactly 0.0 -- pass 0, whose carry is
        # the zero field -- so the confidence term drops samples on pass 0 and
        # on no later pass, and "the reported drop count is the FINAL pass's"
        # is a claim with two different numbers behind it.
        self.drop_first_pass = False
        # None, "short" (the stack omits pass 0) or "stale" (track_multi is
        # pass 0's field rather than the stack's last slice): the two ways a
        # model's stack can disagree with its reported field.
        self.corrupt_stack = None
        # A weak reference to every stack `forward` emits, so a test can ask
        # which ones are still resident without keeping any of them alive.
        self.emitted_stacks = []
        self.applied_offsets = []
        self.seen_track_kwargs = []
        self.seen_track_layout = []
        self.seen_previous_fields = []
        self.seen_forward_kwargs = []

    def pass_offsets(self, refine_iters):
        """c_0 .. c_{K-1} in closed form: 0, then offset * (2 - 2 ** (1 - k))."""

        return [0.0] + [
            self.iteration_offset * (2.0 - 2.0 ** (1 - k)) for k in range(1, refine_iters)
        ]

    def encode_features(
        self,
        images,
        ref_view_strategy="first",
        time_indices=None,
        depth_maps=None,
        camera_vectors=None,
    ):
        # _FakeArc's tap is a parameter sum whose autograd graph saves no
        # tensor, so a second backward through it succeeds silently and the
        # cut would be a spy's claim rather than a behavioural pin. Times a
        # constant one: every value and gradient stays bit-identical (x * 1.0
        # is exact, and d(x * 1)/dx is 1), and the Mul node saves its operand,
        # so a pass that backwards through the UNCUT taps a second time raises
        # "Trying to backward through the graph a second time", as the real
        # encoder graph does.
        (scale,) = super().encode_features(
            images,
            ref_view_strategy=ref_view_strategy,
            time_indices=time_indices,
            depth_maps=depth_maps,
            camera_vectors=camera_vectors,
        )
        return [scale * torch.ones(())]

    def reconstruct(self, feats, images):
        # A VALID pose encoding rather than _FakeArc's zeros: at K > 1 the
        # step unprojects this dict through the real predicted_pointmaps to
        # build the key cloud, and a zero quaternion with zero fov unprojects
        # to NaN. Identity rotation, unit fov, unit depth, slot s translated
        # by s^2 along x so every slot's cloud is its own. The alignment fit
        # and the anchor gather never see it -- _step_scene plants their
        # pointmaps through sparse_tracking's alias -- so K=1 is unaffected,
        # and a step that read the cloud through that alias instead of the
        # public function would receive the planted cloud, not this one.
        pose = torch.zeros(1, self.observations, 9)
        pose[..., 6] = 1.0
        pose[..., 7:] = 1.0
        pose[0, :, 0] = torch.arange(self.observations, dtype=torch.float32).square()
        return {
            "depth": torch.ones(1, self.observations, self.height, self.width),
            "pose_enc": pose,
        }

    def _offset(self, refinement, track):
        if self.force_offset is not None:
            return self.force_offset
        if refinement is None:
            return 0.0
        carry = refinement.previous_field
        if not bool(carry.abs().sum() > 0):
            return 0.0  # pass 0 reads the zero field
        # c_k = offset + c_{k-1} / 2, with c_{k-1} read off the carry ITSELF
        # against this pass's undisplaced field, so a wrong carry is a wrong
        # constant; recorded, so a hand K=1 run can reproduce it exactly.
        return self.iteration_offset + 0.5 * float((carry - track.detach()).mean())

    def track_for_query(self, feats, images, query_idx, views_per_time=1, merge=False, **kwargs):
        self.seen_track_kwargs.append(sorted(kwargs))
        self.seen_track_layout.append((views_per_time, merge))
        refinement = kwargs.get("refinement")
        if refinement is not None:
            field = refinement.previous_field
            self.seen_previous_fields.append(
                (
                    field.detach().clone(),
                    field.requires_grad,
                    refinement.anchor_xyz.detach().clone(),
                    refinement.key_xyz.detach().clone(),
                )
            )
        track, confidence = super().track_for_query(
            feats, images, query_idx, views_per_time=views_per_time, merge=merge
        )
        offset = self._offset(refinement, track)
        self.applied_offsets.append(offset)
        # The confidence channel shifts with the pass too (still > 1, finite,
        # varying), so "the reported confidence summary is the FINAL pass's"
        # is a claim this fake can witness; at K=1 the shift is exactly 0.0.
        confidence = confidence + offset
        if self.drop_first_pass and offset == 0.0:
            rows = torch.arange(self.height).view(-1, 1)
            cols = torch.arange(self.width).view(1, -1)
            confidence = confidence.masked_fill((rows + cols) % 2 == 0, float("inf"))
        return track + offset, confidence

    def forward(self, views, force_no_output_conversion=False, merge_synchronized_slots=False, **kwargs):
        self.seen_forward_kwargs.append(sorted(kwargs))
        refine_iters = kwargs.get("refine_iters", 1)
        output = super().forward(
            views,
            force_no_output_conversion=force_no_output_conversion,
            merge_synchronized_slots=merge_synchronized_slots,
        )
        if refine_iters > 1:
            base = output["track_multi"]
            offsets = self.pass_offsets(refine_iters)
            if self.corrupt_stack == "short":
                offsets = offsets[1:]
            iterations = torch.stack([base + constant for constant in offsets], dim=2)
            if self.poison_pixel is not None:
                iterations[..., self.poison_pixel[0], self.poison_pixel[1], :] = float("nan")
            output["track_multi_iterations"] = iterations
            # A copy, not a view of the stack, exactly as Arc._forward builds
            # track_multi by its own torch.stack: a view would keep the whole
            # stack alive through its base however promptly a caller dropped
            # the stack itself.
            reported = 0 if self.corrupt_stack == "stale" else -1
            output["track_multi"] = iterations[:, :, reported].clone()
            self.emitted_stacks.append(weakref.ref(iterations))
        elif self.poison_pixel is not None:
            track_multi = output["track_multi"].clone()
            track_multi[..., self.poison_pixel[0], self.poison_pixel[1], :] = float("nan")
            output["track_multi"] = track_multi
        return output


class _RecordingScaler(torch.amp.GradScaler):
    """Disabled, so scale() returns its argument, and it records every scalar
    train_step hands to backward: the count and the values are what say a
    step backwarded per iteration rather than once over a discounted sum,
    which the gradient identity cannot tell apart because backward is
    linear."""

    def __init__(self):
        super().__init__("cuda", enabled=False)
        self.scaled = []

    def scale(self, outputs):
        self.scaled.append(float(outputs.detach()))
        return super().scale(outputs)


def _refine_step(model, scene, *, refine_iters, refine_gamma, scaler=None, **overrides):
    """One real train_step. grad_clip is huge so the .grad left on the
    parameters is the accumulated gradient itself: finish_window's clip then
    multiplies by an exactly-1.0 coefficient, and AdamW does not touch .grad,
    which is what the identity below compares. ``overrides`` replace any of
    the keywords below (the loss weights and alpha, the merge flag)."""

    kwargs = dict(
        model=model,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False) if scaler is None else scaler,
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1e9,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
        velocity_weight=0.0,
        learning_rates=[1e-3],
        step=0,
        accum_steps=1,
        window_start=True,
        window_end=True,
        refine_iters=refine_iters,
        refine_gamma=refine_gamma,
    )
    kwargs.update(overrides)
    return train_cli.train_step(**kwargs)


def _gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _spy_on_cut(monkeypatch):
    cuts = []
    real_cut = train_cli.cut_features

    def spying_cut(feats):
        cuts.append(True)
        return real_cut(feats)

    monkeypatch.setattr(train_cli, "cut_features", spying_cut)
    return cuts


def _hand_passes(model, scene, offsets, **overrides):
    """One K=1 step per pass constant, on identically initialised copies of
    ``model``: pass k's field for every anchor, its backwarded scalars (one
    per anchor), its outcome and its gradient -- the by-hand side of every
    K > 1 identity below. ``overrides`` reach train_step (the loss weights,
    an explicit alpha)."""

    references = []
    for constant in offsets:
        reference = copy.deepcopy(model)
        reference.force_offset = constant
        scaler = _RecordingScaler()
        outcome = _refine_step(
            reference, scene, refine_iters=1, refine_gamma=0.5, scaler=scaler, **overrides
        )
        references.append((reference, outcome, scaler.scaled, _gradients(reference)))
    return references


def test_train_step_backwards_each_pass_under_its_discount(tmp_path, monkeypatch):
    """--refine_iters 4 --refine_gamma 0.5 -- the count the live arms are
    recommended to run, and a discount whose four weights (1/32, 1/16, 1/8,
    1/4) are powers of two, so every scale below is bit-exact -- on the fake
    whose passes emit four DISTINCT fields, on a scene anchored at camera 1
    (slot 4, anchor index 0) so slot and index cannot be confused. Above two
    passes on purpose: at K=2 there is one non-final pass and one
    intermediate field, so a carry assigned on pass 0 only, or a per-pass
    record that reads pass 0 twice, collapse onto the correct wiring.

    Four backwards, not one: the recording scaler sees exactly four scalars,
    w_k * t_k with t_k the scalar a K=1 step on an identically initialised
    copy backwards for pass k's field (force_offset = the constant the K=4
    step recorded for that pass), compared EXACTLY. The gradient identity
    sum_k w_k g_k is asserted too, exactly, but on its own it cannot
    separate per-pass backward from one backward over the discounted sum:
    backward is linear, so both leave the same .grad. What it does refuse is
    a step that weighted the anchors and forgot the passes, or discounted in
    the wrong direction. iteration_losses records the per-pass position
    Huber, four distinct values with the final entry equal to `loss`, so
    `loss` keeps today's meaning; a single active anchor at K > 1 goes
    through cut_features, because pass k's backward frees the encoder graph
    pass k+1 would need; the recorded carries are zeros, then pass k-1's OWN
    field at every later pass; and the recorded cloud is the fake's own
    reconstruction through the public predicted_pointmaps, not the planted
    alignment cloud, at the anchor SLOT."""

    scene = _step_scene(tmp_path, monkeypatch, query_anchors=((1, 0),))
    height, width = scene.views[0]["img"].shape[-2:]
    slot = scene.anchor_observation_slots[0]
    assert slot == 4 and len(scene.anchor_observation_slots) == 1
    refine_iters = 4
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    pristine = copy.deepcopy(model)
    cuts = _spy_on_cut(monkeypatch)

    refined_scaler = _RecordingScaler()
    outcome = _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5, scaler=refined_scaler
    )
    assert cuts == [True]
    refined = _gradients(model)
    offsets = model.applied_offsets
    assert offsets == pytest.approx(model.pass_offsets(refine_iters), rel=1e-5)
    assert offsets[0] == 0.0 and len(set(offsets)) == refine_iters

    cuts.clear()
    references = _hand_passes(pristine, scene, offsets)
    assert cuts == []  # one anchor at K=1: today's uncut graph, four times over

    weights = refinement_iteration_weights(refine_iters, 0.5)
    assert weights == (0.03125, 0.0625, 0.125, 0.25)
    scalars = [scaled for _, _, scaled, _ in references]
    assert all(len(scaled) == 1 for scaled in scalars)
    assert len({scaled[0] for scaled in scalars}) == refine_iters
    assert refined_scaler.scaled == [
        weight * scaled[0] for weight, scaled in zip(weights, scalars)
    ]
    gradients = [grads for _, _, _, grads in references]
    assert all(set(grads) == set(refined) for grads in gradients)
    for name in refined:
        for earlier, later in zip(gradients, gradients[1:]):
            assert not torch.equal(earlier[name], later[name]), name
        expected = weights[0] * gradients[0][name]
        for weight, grads in zip(weights[1:], gradients[1:]):
            expected = expected + weight * grads[name]
        assert torch.equal(refined[name], expected), name

    losses = [reference_outcome.loss for _, reference_outcome, _, _ in references]
    assert outcome.iteration_losses == losses
    assert len(set(losses)) == refine_iters
    assert outcome.iteration_losses[-1] == outcome.loss
    assert all(reference_outcome.iteration_losses is None for _, reference_outcome, _, _ in references)
    assert outcome.sample_count == references[0][1].sample_count
    # Every other reported figure is the FINAL pass's and only the final
    # pass's: the metric error and the confidence summary equal the forced
    # K=1 step at pass 3's constant and differ from pass 0's, so a summary
    # read on the first pass, or an error summed over the passes, fails.
    first, final = references[0][1], references[-1][1]
    assert outcome.metric_error_m == final.metric_error_m != first.metric_error_m
    assert outcome.confidence == final.confidence != first.confidence

    # Four head calls, each carrying the refinement keyword and nothing else
    # beyond today's positional surface: pass 0 with zeros, pass k with pass
    # k-1's own field, detached, over the same cloud.
    assert model.seen_track_kwargs == [["refinement"]] * refine_iters
    assert model.seen_track_layout == [(1, False)] * refine_iters
    slots = scene.num_observations
    scale = pristine.encode_features(None)[0].detach()
    expected_cloud = patch_centre_points(
        predicted_pointmaps(pristine.reconstruct(None, None)), pristine.PATCH_SIZE
    )
    assert expected_cloud.shape == (1, slots, (height // 14) * (width // 14), 3)
    assert len(model.seen_previous_fields) == refine_iters
    for iteration, (carry, requires_grad, anchor, cloud) in enumerate(model.seen_previous_fields):
        assert carry.shape == (1, slots, height, width, 3) and carry.dtype is torch.float32
        assert requires_grad is False
        if iteration == 0:
            assert torch.count_nonzero(carry) == 0
        else:
            assert torch.equal(
                carry, torch.ones(1, slots, height, width, 3) * scale + offsets[iteration - 1]
            )
        assert torch.equal(cloud, expected_cloud)
        assert torch.equal(anchor, expected_cloud[:, slot])
        assert not torch.equal(anchor, expected_cloud[:, 0])
    plain = references[0][0]
    assert plain.seen_track_kwargs == [[]] and plain.seen_previous_fields == []


def test_train_step_carries_each_anchor_separately_and_shares_every_pass(tmp_path, monkeypatch):
    """Two active anchors at --refine_iters 4, the pass count the live arms
    are recommended to run, with one occluded track so the anchors' sample
    shares differ (the live arms seat six camera anchors; this fixture has
    two cameras, so two is what it seats). Everything one anchor cannot show:
    the carry is PER ANCHOR -- every anchor's pass 0 reads the zero field and
    its pass k reads its own pass k-1 field, so a carry left over from the
    previous anchor's last non-final pass fails, twice, on the recorded zero
    field and on the constants the fake derives from the carry; every pass of
    every anchor is backwarded on its own, anchor-major, with the anchor's
    share inside the scalar and the pass weight outside; iteration_losses[k]
    is the share-combined position Huber of pass k across the anchors,
    exactly what a K=1 step with every anchor at pass k's field reports, so
    an accumulation that forgot the share or kept the last anchor only fails;
    and each anchor's kNN anchor point is ITS slot's cloud, not the first
    anchor's. The gradient identity across anchors and passes holds to
    rounding only, because the step accumulates anchor-major while the hand
    sum is pass-major, so it is asserted close where the scalars and losses
    are asserted exact."""

    scene = _step_scene(
        tmp_path, monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
    )
    height, width = scene.views[0]["img"].shape[-2:]
    anchor_slots = scene.anchor_observation_slots
    assert anchor_slots == (0, 4)
    anchors, refine_iters = len(anchor_slots), 4
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    pristine = copy.deepcopy(model)
    cuts = _spy_on_cut(monkeypatch)

    refined_scaler = _RecordingScaler()
    outcome = _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5, scaler=refined_scaler
    )
    assert cuts == [True]
    refined = _gradients(model)
    assert len(model.applied_offsets) == anchors * refine_iters
    per_anchor = [
        model.applied_offsets[index * refine_iters : (index + 1) * refine_iters]
        for index in range(anchors)
    ]
    assert per_anchor[0] == per_anchor[1]  # the same recursion from the same zero start
    offsets = per_anchor[0]
    assert offsets == pytest.approx(model.pass_offsets(refine_iters), rel=1e-5)
    assert offsets[0] == 0.0 and len(set(offsets)) == refine_iters

    references = _hand_passes(pristine, scene, offsets)
    assert cuts == [True] * (1 + refine_iters)  # two anchors: every step cuts, K=1 included

    weights = refinement_iteration_weights(refine_iters, 0.5)
    # Anchor-major, pass-minor: the K=4 step's eight scalars are anchor 0's
    # four passes then anchor 1's, and a K=1 step's two are the two anchors
    # at one pass.
    assert all(len(scaled) == anchors for _, _, scaled, _ in references)
    assert refined_scaler.scaled == [
        weights[iteration] * references[iteration][2][index]
        for index in range(anchors)
        for iteration in range(refine_iters)
    ]
    losses = [reference_outcome.loss for _, reference_outcome, _, _ in references]
    assert outcome.iteration_losses == losses
    assert len(set(losses)) == refine_iters
    assert outcome.iteration_losses[-1] == outcome.loss
    first, final = references[0][1], references[-1][1]
    assert outcome.metric_error_m == final.metric_error_m != first.metric_error_m
    assert outcome.confidence == final.confidence != first.confidence
    # ... and that summary is the FIRST active anchor's, not the last one's.
    # The K=4 step and the forced K=1 step run the same code, so the identity
    # above cannot say which anchor either one reads; this computes both
    # anchors' final-pass summaries directly.
    summary_model = copy.deepcopy(pristine)
    summary_model.force_offset = offsets[-1]
    summary_feats = summary_model.encode_features(None)
    summaries = []
    for slot in anchor_slots:
        track, confidence = summary_model.track_for_query(summary_feats, None, slot)
        summaries.append(
            runtime.confidence_stats(
                {
                    "track_multi": track[:, None],
                    "conf_track_multi": confidence[:, None],
                    "track_query_idx": torch.tensor([slot]),
                }
            )
        )
    assert summaries[0] != summaries[1]
    assert outcome.confidence == summaries[0]
    for name in refined:
        expected = weights[0] * references[0][3][name]
        for weight, (_, _, _, grads) in zip(weights[1:], references[1:]):
            expected = expected + weight * grads[name]
        torch.testing.assert_close(refined[name], expected, rtol=1e-5, atol=1e-6)

    assert model.seen_track_kwargs == [["refinement"]] * (anchors * refine_iters)
    slots = scene.num_observations
    scale = pristine.encode_features(None)[0].detach()
    expected_cloud = patch_centre_points(
        predicted_pointmaps(pristine.reconstruct(None, None)), pristine.PATCH_SIZE
    )
    assert len(model.seen_previous_fields) == anchors * refine_iters
    for position, (carry, requires_grad, anchor, cloud) in enumerate(model.seen_previous_fields):
        index, iteration = divmod(position, refine_iters)
        assert requires_grad is False
        if iteration == 0:
            assert torch.count_nonzero(carry) == 0, (index, iteration)
        else:
            assert torch.equal(
                carry, torch.ones(1, slots, height, width, 3) * scale + offsets[iteration - 1]
            ), (index, iteration)
        assert torch.equal(cloud, expected_cloud)
        assert torch.equal(anchor, expected_cloud[:, anchor_slots[index]]), (index, iteration)
        assert not torch.equal(anchor, expected_cloud[:, anchor_slots[1 - index]])


def test_train_step_pins_alpha_on_the_first_pass_and_discounts_every_term(tmp_path, monkeypatch):
    """--confidence_weight 0.5 with --confidence_alpha auto at
    --refine_iters 4. The pin comes from the first anchor's FIRST pass: the
    step's resolved alpha equals what a K=1 auto step on the same weights
    resolves (the same pass-0 field) and differs from what a K=1 auto step
    at pass 3's field resolves, so a pin taken on the final pass, or
    re-resolved on every pass, fails. And every term of a pass is discounted
    alike: with the pinned alpha handed to the hand K=1 runs explicitly, the
    four backwarded scalars are w_k times pass k's position-plus-confidence
    total, exactly, and the gradient identity holds bit for bit (powers of
    two), so a discount applied to the position term alone fails. The
    reported breakdown and drop counts are the final pass's: the fake's
    confidence is non-finite on a checkerboard of pixels on pass 0 only, so
    pass 0 drops samples and the final pass drops none, and a count summed
    over the passes reports the pass-0 drops."""

    scene = _step_scene(tmp_path, monkeypatch, query_anchors=((1, 0),))
    height, width = scene.views[0]["img"].shape[-2:]
    refine_iters = 4
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    model.drop_first_pass = True
    pristine = copy.deepcopy(model)
    term = dict(confidence_weight=0.5)

    refined_scaler = _RecordingScaler()
    outcome = _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5, scaler=refined_scaler,
        confidence_alpha=None, **term,
    )
    assert outcome.confidence_alpha is not None
    offsets = model.applied_offsets
    assert offsets == pytest.approx(model.pass_offsets(refine_iters), rel=1e-5)

    (auto_first,) = _hand_passes(pristine, scene, offsets[:1], confidence_alpha=None, **term)
    (auto_final,) = _hand_passes(pristine, scene, offsets[-1:], confidence_alpha=None, **term)
    assert outcome.confidence_alpha == auto_first[1].confidence_alpha
    assert outcome.confidence_alpha != auto_final[1].confidence_alpha

    pinned = _hand_passes(
        pristine, scene, offsets, confidence_alpha=outcome.confidence_alpha, **term
    )
    weights = refinement_iteration_weights(refine_iters, 0.5)
    assert refined_scaler.scaled == [
        weight * scaled[0] for weight, (_, _, scaled, _) in zip(weights, pinned)
    ]
    # The confidence term is inside every scalar: without it the position-only
    # scalars of the discount test above would be reported instead.
    assert refined_scaler.scaled != [
        weight * reference_outcome.loss for weight, (_, reference_outcome, _, _) in zip(weights, pinned)
    ]
    refined = _gradients(model)
    for name in refined:
        expected = weights[0] * pinned[0][3][name]
        for weight, (_, _, _, grads) in zip(weights[1:], pinned[1:]):
            expected = expected + weight * grads[name]
        assert torch.equal(refined[name], expected), name
    assert outcome.iteration_losses == [
        reference_outcome.loss for _, reference_outcome, _, _ in pinned
    ]
    first, final = pinned[0][1], pinned[-1][1]
    assert outcome.loss == final.loss
    assert outcome.loss_breakdown == final.loss_breakdown is not None
    assert sum(first.confidence_dropped.values()) > 0
    assert sum(final.confidence_dropped.values()) == 0
    assert outcome.confidence_dropped == final.confidence_dropped


def test_train_step_merges_the_carry_rows_by_time(tmp_path, monkeypatch):
    """The merged arm of the step: under --merge_synchronized_slots the head
    emits one row per time, so the zero carry is (1, T, H, W, 3) -- T the
    merged time count, not S -- and the head is called with merge=True and
    the layout beside the refinement; the cloud stays slot-major and the
    anchor is the anchor slot's. A carry sized on S here would fail the real
    refiner's row check and pass any fake that ignores it."""

    scene = _step_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width, views_per_time=2)
    cuts = _spy_on_cut(monkeypatch)

    outcome = _refine_step(
        model, scene, refine_iters=2, refine_gamma=0.5, merge_synchronized_slots=True
    )

    assert cuts == [True]
    assert len(outcome.iteration_losses) == 2
    assert outcome.iteration_losses[-1] == outcome.loss
    assert model.seen_track_layout == [(2, True), (2, True)]
    assert model.seen_track_kwargs == [["refinement"], ["refinement"]]
    times = scene.num_observations // 2
    (zeros, _, anchor_0, cloud_0), (carried, _, _, _) = model.seen_previous_fields
    assert zeros.shape == (1, times, height, width, 3)
    assert torch.count_nonzero(zeros) == 0
    assert carried.shape == (1, times, height, width, 3)
    assert torch.count_nonzero(carried) > 0
    assert cloud_0.shape == (1, scene.num_observations, (height // 14) * (width // 14), 3)
    assert torch.equal(anchor_0, cloud_0[:, scene.anchor_observation_slots[0]])


def test_train_step_at_one_iteration_is_todays_step(tmp_path, monkeypatch):
    """K=1 must be today's step to the keyword: the head is called without a
    refinement keyword (injected fakes bind today's three-positional
    surface), iteration_losses is None like loss_breakdown on a position-only
    step, no cut is taken at one anchor, and the outcome and the stepped
    parameters equal those of a step on the plain _FakeArc through today's
    train_step keywords -- a fake that does not know the refinement surface
    at all, whose reconstruction is the zero pose the cloud path would
    refuse."""

    scene = _step_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    refining = _RefiningFakeArc(scene.num_observations, height, width)
    torch.manual_seed(0)
    plain = _FakeArc(scene.num_observations, height, width)
    cuts = _spy_on_cut(monkeypatch)

    outcome = _refine_step(refining, scene, refine_iters=1, refine_gamma=0.8)
    reference = train_cli.train_step(
        model=plain,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW([{"params": list(plain.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1e9,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
        velocity_weight=0.0,
        learning_rates=[1e-3],
        step=0,
        accum_steps=1,
        window_start=True,
        window_end=True,
    )

    assert cuts == []
    assert outcome.iteration_losses is None
    assert refining.seen_track_kwargs == [[]]
    assert refining.seen_previous_fields == []
    assert outcome.loss == reference.loss
    assert outcome.metric_error_m == reference.metric_error_m
    assert outcome.gradient_norms == reference.gradient_norms
    for (name, stepped), (_, expected) in zip(
        refining.named_parameters(), plain.named_parameters()
    ):
        assert torch.equal(stepped, expected), name


def test_run_training_records_every_iteration_in_the_history_and_the_step_line(
    tmp_path, monkeypatch, capsys
):
    """The loop's own reading of the instrument: run_training at
    --refine_iters 2 writes a history row whose iteration_losses is the
    two-entry list (JSON-native floats, the last equal to the row's loss) and
    prints the ` iters=[...]` suffix -- the branch no K=1 loop test executes,
    since every other run_training test drives the loop at one iteration.
    A row that carried null here at K > 1, or a print that dropped the
    suffix, would leave a K=2 run indistinguishable from a K=1 run in its
    own log."""

    train_cli._STOP_REQUESTED.clear()
    scene = _step_scene(tmp_path / "scene", monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _RefiningFakeArc(scene.num_observations, height, width)

    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(1),
        args=_loop_args(tmp_path, num_steps=1, warmup_steps=0, refine_iters=2, refine_gamma=0.5),
        scene_provider=lambda _plan: scene,
        step_fn=train_cli.train_step,
        output_dir=tmp_path,
    )

    (record,) = _history(tmp_path)
    assert isinstance(record["iteration_losses"], list)
    assert len(record["iteration_losses"]) == 2
    assert all(isinstance(value, float) for value in record["iteration_losses"])
    assert record["iteration_losses"][-1] == record["loss"]
    assert record["iteration_losses"][0] != record["iteration_losses"][1]
    assert result["history"][0].iteration_losses == record["iteration_losses"]
    step_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("step=0/1 ")
    )
    # The exact suffix, in pass order: the last entry printed is the `loss=`
    # on the same line, which a reversed or re-sorted list would contradict.
    assert step_line.endswith(
        f" iters={[round(value, 8) for value in record['iteration_losses']]}"
    )
    assert model.seen_track_kwargs == [["refinement"], ["refinement"]]


@pytest.mark.parametrize("refine_iters", [2, 3])
def test_write_checkpoint_records_the_iteration_count_beside_the_refiner(tmp_path, refine_iters):
    """The trainer's writer is the only sanctioned source of K > 1 patches,
    because the overfit's saver records no count. The payload it writes must
    read back as the arm it trained -- refine from the key set, the count
    from the scalar -- and load onto a model frozen the same way; a writer
    that dropped the scalar would produce a patch the reader refuses as a
    refiner without a count, so it could never be run as trained. At K=2
    and K=3: K=2 is the boundary of the reader's refine-versus-count rule
    and the count of the first cluster run, the memory gate."""

    model = _TinyHubArc(freeze="none")
    model.set_freeze("temporal_tracking", refine=True)
    with torch.no_grad():
        model.motion_decoder.refiner.read_proj.weight.fill_(0.5)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-3}]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = train_cli._write_checkpoint(
        model, optimizer, scaler, [1e-3],
        step=0, output_dir=tmp_path / "refined",
        args=_loop_args(tmp_path, refine_iters=refine_iters, refine_gamma=0.5),
    )

    metadata = read_temporal_patch_metadata(checkpoint)
    assert metadata["refine"] is True
    assert metadata["refine_iters"] == refine_iters
    restored = _TinyHubArc(freeze="none")
    restored.set_freeze("temporal_tracking", refine=True)
    load_temporal_tracking_checkpoint(restored, checkpoint)
    assert torch.equal(
        restored.motion_decoder.refiner.read_proj.weight,
        model.motion_decoder.refiner.read_proj.weight,
    )
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_temporal_tracking_checkpoint(_TinyHubArc(freeze="temporal_tracking"), checkpoint)

    single = _TinyHubArc(freeze="temporal_tracking")
    checkpoint = train_cli._write_checkpoint(
        single,
        torch.optim.AdamW([{"params": [p for p in single.parameters() if p.requires_grad], "lr": 1e-3}]),
        scaler, [1e-3], step=0, output_dir=tmp_path / "single", args=_loop_args(tmp_path),
    )
    metadata = read_temporal_patch_metadata(checkpoint)
    assert metadata["refine"] is False
    assert metadata["refine_iters"] == 1

    # A refiner-bearing model written under K=1 args is the pairing the
    # reader refuses; main() cannot produce it because both calls read the
    # same predicate, which the AST pin in test_trainer_loop holds.
    mismatched = train_cli._write_checkpoint(
        model, optimizer, scaler, [1e-3],
        step=0, output_dir=tmp_path / "mismatched", args=_loop_args(tmp_path),
    )
    with pytest.raises(ValueError, match="refine_iters=1"):
        read_temporal_patch_metadata(mismatched)


_END_TO_END_WIDTH = 64


class _UnfoldBackbone(nn.Module):
    """An encoder for the REAL decoder and head. Arc.track_for_query hands the
    motion decoder every tap channel past 1536 (the production taps are 3072
    wide, so the decoder is 1536 wide); these taps are 1536 + 64 wide, so the
    same code drives a 64-wide decoder at CPU cost. Each tap's patch tokens
    carry the first 64 values of the image's own unfolded 14x14 blocks; the
    time token carries the time-index embedding, so that table takes
    gradient the way it does through the real encoder; the camera token is
    zero. No other parameter, so set_freeze's temporal presets leave exactly
    the embedding trainable here, as on the real backbone."""

    def __init__(self, max_time_indices):
        super().__init__()
        pretrained = nn.Module()
        pretrained.time_index_embedding = nn.Embedding(max_time_indices, 2)
        self.pretrained = pretrained

    def forward(
        self, x, ref_view_strategy="first", time_indices=None, depth_maps=None, camera_vectors=None
    ):
        B, S = x.shape[:2]
        width = 1536 + _END_TO_END_WIDTH
        blocks = F.unfold(x.flatten(0, 1), _PATCH, stride=_PATCH).transpose(1, 2)
        patches = blocks.shape[1]
        patch = torch.zeros(B, S, patches, width, device=x.device)
        patch[..., 1536:] = blocks[..., :_END_TO_END_WIDTH].reshape(B, S, patches, -1)
        camera = torch.zeros(B, S, width, device=x.device)
        time = torch.zeros(B, S, width, device=x.device)
        if time_indices is not None:
            time[..., 1536:1538] = self.pretrained.time_index_embedding(time_indices)
        return tuple((patch, camera, time) for _ in range(4)), []


def _real_refiner_shell(seed=0):
    """An Arc shell whose track path is REAL end to end -- Arc.track_for_query,
    a MotionDecoder with its TrackRefiner, a small real DPTHead -- behind the
    unfold backbone and the cloud-bearing reconstruction fakes, at a 64-wide
    decoder (see _UnfoldBackbone). The decoder's AdaLN modulation is
    zero-initialised by construction (block.py), which pretrained weights
    are not: at zero the time token is only a condition and takes exactly
    zero gradient, so the trainer's embedding-norm guard would refuse the
    very first step; a small normal init stands in for the trained weights.
    The refiner keeps its own zero init: this is step 0 of a refinement arm."""

    torch.manual_seed(seed)
    model = _arc_shell(max_time_indices=32)
    model.backbone = _UnfoldBackbone(32)
    model.head = _UnitDepthHead()
    model.cam_dec = _SquareTranslationPoseDecoder()
    model.motion_decoder = MotionDecoder(
        patch_size=_PATCH, embed_dim=_END_TO_END_WIDTH, depth=1, num_heads=4, use_adaln=True
    )
    for name, module in model.motion_decoder.named_modules():
        if name.endswith("adaLN_modulation.1"):
            nn.init.normal_(module.weight, std=0.02)
    model.track_head = DPTHead(
        dim_in=_END_TO_END_WIDTH,
        output_dim=4,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )
    return model


def test_train_step_and_the_eval_drive_the_real_refiner_end_to_end(tmp_path, monkeypatch):
    """The live configuration's own path, on CPU: the REAL Arc.track_for_query,
    a real MotionDecoder with its TrackRefiner and a real DPTHead (64 wide,
    the production code at CPU cost), frozen by the production
    set_freeze(refine=True), driven by the real
    train_step at --refine_iters 4 with two active anchors and then by the
    real evaluate_held_out at the same K -- the parts every other K > 1 test
    fakes. What only this can show: the gradient guard accepts the eight
    refiner tensors (the two projections carry exactly-zero gradients at the
    zero-init saddle and must still count as present), every pass backwards
    onto the cut, the trainer's fp32 zero field meets the real refiner, and
    the refiner's output layers move after one optimizer step. At step 0 the
    four per-pass losses are bit-identical, because the zero-init term is
    exactly zero and refinement is inert until the first update; after that
    update the eval's per-pass losses are no longer all equal. The 4x4 patch
    grid of the 56x56 fixture sits exactly at k = 16 keys, the admissible
    boundary. The refine=False shell steps and evaluates at K=1 with the
    refiner frozen, today's path."""

    scene = _step_scene(
        tmp_path, monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
    )
    height, width = scene.views[0]["img"].shape[-2:]
    assert (height // _PATCH) * (width // _PATCH) == REFINER_NEIGHBOURS
    refine_iters = 4
    model = _real_refiner_shell()
    model.set_freeze("temporal_tracking", refine=True)
    refiner_names = {name for name, _ in model.named_parameters() if ".refiner." in name}
    assert len(refiner_names) == 8
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name in refiner_names
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in refiner_names
    }
    cuts = _spy_on_cut(monkeypatch)

    outcome = _refine_step(model, scene, refine_iters=refine_iters, refine_gamma=0.5)

    assert cuts == [True]
    assert len(outcome.iteration_losses) == refine_iters
    assert all(math.isfinite(value) for value in outcome.iteration_losses)
    assert len(set(outcome.iteration_losses)) == 1
    assert outcome.iteration_losses[-1] == outcome.loss
    assert outcome.gradient_norms["time_embedding"] > 0
    after = dict(model.named_parameters())
    for name in ("read_proj.weight", "read_proj.bias", "field_embed.weight", "field_embed.bias"):
        full = f"motion_decoder.refiner.{name}"
        assert not torch.equal(before[full], after[full].detach()), name

    plan = plan_record(_record(seq_name="0000"), budget=48, stride=2)

    def evaluate(evaluated, directory, **extra):
        return train_cli.evaluate_held_out(
            model=evaluated,
            plans=[plan],
            scene_provider=lambda _plan: scene,
            precision="32",
            huber_delta_m=0.05,
            step=1,
            output_dir=tmp_path / directory,
            query_anchors=["0:0", "1:0"],
            confidence_alpha=_EVAL_ALPHA,
            **extra,
        )

    metrics = evaluate(model, "refined", refine_iters=refine_iters)
    entry = metrics["per_scene"][0]
    assert len(entry["iteration_position_losses"]) == refine_iters
    assert all(math.isfinite(value) for value in entry["iteration_position_losses"])
    assert entry["iteration_position_losses"][-1] == entry["position_loss"]
    assert len(set(entry["iteration_position_losses"])) > 1
    assert metrics["iteration_position_losses"] == entry["iteration_position_losses"]
    assert math.isfinite(entry["position_loss_shuffled"])

    plain = _real_refiner_shell()
    plain.set_freeze("temporal_tracking")
    assert not any(
        parameter.requires_grad
        for name, parameter in plain.named_parameters()
        if name in refiner_names
    )
    cuts.clear()
    single = _refine_step(plain, scene, refine_iters=1, refine_gamma=0.5)
    assert cuts == [True]  # two anchors cut at K=1 as well
    assert single.iteration_losses is None
    assert torch.count_nonzero(dict(plain.named_parameters())["motion_decoder.refiner.read_proj.weight"]) == 0
    assert evaluate(plain, "single")["per_scene"][0]["iteration_position_losses"] is None


# ------------------------------------------------------------ the eval ---


@pytest.mark.parametrize("merge", [False, True])
def test_evaluate_held_out_scores_every_refinement_pass(tmp_path, monkeypatch, merge):
    """The held-out record of the same instrument, at K=3 -- above two passes
    on purpose, because at K=2 "every intermediate pass" is one pass and a
    loop that scored pass 0 for each intermediate entry, or scored exactly
    one intermediate, collapses onto the correct one. Two scenes with
    different occlusion sets (the same geometry, so the one planted
    alignment pointmap serves both), whose scores differ, so the metrics'
    per-pass entry is a MEAN over scored scenes and not any one scene's.
    Per scene, one position Huber per pass: entry k equal to what a K=1 eval
    of the same weights scores on pass k's field (the fake's closed-form
    constant for that pass, forced), three distinct values, the last equal
    to position_loss, the loss of the reported field; the index-shuffled arm
    scores the FINAL pass, equal to the forced K=1 eval at pass 2 and unequal
    to pass 0's. Every field the refining fake emits carries a NaN at pixel
    (0, 0), a pixel no correspondence supervises -- the K=1 evals prove it by
    scoring a poisoned model identically -- so the stack's wiring check must
    tolerate what the loss tolerates. Both head layouts, per slot and merged
    by time, the live arms' layout: every pass is scored under the same
    reduction as the final one, so a per-pass loss that dropped the merge
    flag scores time rows as observation slots. At K=1 both records are None
    and the model is called with today's exact keywords, which the CPU fakes
    bind."""

    scenes = {
        "0000": _step_scene(
            tmp_path / "a", monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
        ),
        # Track 1 hidden at time 1 in BOTH cameras: the merged head pools the
        # cameras of a time, so hiding it in one would leave the merged
        # supervision, and the merged scores, unchanged.
        "0001": _step_scene(
            tmp_path / "b",
            monkeypatch,
            query_anchors=((0, 0), (1, 0)),
            invisible=((0, 0, 2), (0, 1, 1), (1, 1, 1)),
        ),
    }
    height, width = scenes["0000"].views[0]["img"].shape[-2:]
    plans = [plan_record(_record(seq_name=name), budget=48, stride=2) for name in scenes]
    refine_iters = 3

    def evaluate(model, directory, **extra):
        return train_cli.evaluate_held_out(
            model=model,
            plans=plans,
            scene_provider=lambda plan: scenes[plan.seq_name],
            precision="32",
            huber_delta_m=0.05,
            step=3,
            output_dir=tmp_path / directory,
            query_anchors=["0:0", "1:0"],
            confidence_alpha=_EVAL_ALPHA,
            merge_synchronized_slots=merge,
            **extra,
        )

    def fake(*, poison):
        torch.manual_seed(0)
        model = _RefiningFakeArc(
            scenes["0000"].num_observations, height, width, views_per_time=2 if merge else 1
        )
        model.poison_pixel = (0, 0) if poison else None
        return model

    refining = fake(poison=True)
    refined = evaluate(refining, "refined", refine_iters=refine_iters)
    singles = []
    for iteration, constant in enumerate(refining.pass_offsets(refine_iters)):
        plain = fake(poison=False)
        plain.force_offset = constant
        singles.append((plain, evaluate(plain, f"single-{iteration}")))
    poisoned = fake(poison=True)
    poisoned_single = evaluate(poisoned, "single-poisoned")

    entries = []
    for index, name in enumerate(scenes):
        entry = refined["per_scene"][index]
        assert entry["scene"] == name
        expected = [single["per_scene"][index]["position_loss"] for _, single in singles]
        assert len(set(expected)) == refine_iters
        assert entry["iteration_position_losses"] == expected
        assert entry["iteration_position_losses"][-1] == entry["position_loss"]
        assert entry["position_loss_shuffled"] == singles[-1][1]["per_scene"][index]["position_loss_shuffled"]
        assert entry["position_loss_shuffled"] != singles[0][1]["per_scene"][index]["position_loss_shuffled"]
        # The NaN at (0, 0) is a pixel the loss never reads, on both arms.
        for key in ("position_loss", "position_loss_shuffled"):
            assert poisoned_single["per_scene"][index][key] == singles[0][1]["per_scene"][index][key]
        entries.append(expected)
    assert entries[0] != entries[1]
    means = [sum(column) / len(column) for column in zip(*entries)]
    assert refined["iteration_position_losses"] == means
    assert refined["iteration_position_losses"][-1] == refined["position_loss"]
    assert refined["iteration_position_losses"] != [max(column) for column in zip(*entries)]
    assert refining.seen_forward_kwargs == [["refine_iters"]] * (2 * len(scenes))
    written = json.loads(
        (tmp_path / "refined" / "eval" / "step-3" / "metrics.json").read_text()
    )
    assert written["iteration_position_losses"] == means
    assert [scene["iteration_position_losses"] for scene in written["per_scene"]] == entries

    for plain, single in singles:
        assert all(scene["iteration_position_losses"] is None for scene in single["per_scene"])
        assert single["iteration_position_losses"] is None
        assert plain.seen_forward_kwargs == [[]] * (2 * len(scenes))


@pytest.mark.parametrize("corruption", ["short", "stale"])
def test_the_eval_refuses_a_stack_that_does_not_end_in_the_reported_field(
    tmp_path, monkeypatch, corruption
):
    """The eval's one wiring check on the model's stack, from both sides: a
    stack one pass short (whose last slice still equals the reported field)
    and a full stack whose reported field is not its last slice. Either is a
    model that would score a per-pass curve the reported loss does not end
    in, and each alone must be refused -- so the check needs both of its
    conditions, joined by `or`."""

    scene = _step_scene(
        tmp_path, monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
    )
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    model.corrupt_stack = corruption

    with pytest.raises(RuntimeError, match="must hold every pass and end in track_multi"):
        train_cli.evaluate_held_out(
            model=model,
            plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
            scene_provider=lambda _plan: scene,
            precision="32",
            huber_delta_m=0.05,
            step=3,
            output_dir=tmp_path / "out",
            query_anchors=["0:0", "1:0"],
            confidence_alpha=_EVAL_ALPHA,
            refine_iters=3,
        )


def test_the_eval_releases_each_stack_before_the_shuffled_arm_is_scored(tmp_path, monkeypatch):
    """Memory, measured rather than inferred from the source. Each arm's
    forward returns K x Q dense fields -- 0.44 GB at the held-out window
    under the merge, about 1.6 GiB per slot -- so the plain arm's stack must
    be gone before the shuffled forward allocates its own, and the shuffled
    arm's before its loss allocates. The fake hands out weak references to
    every stack it emits, and a recorder on the eval's loss counts how many
    are still resident at each call: at most one (the plain arm's own, while
    its passes are scored) during the plain arm, and none while the
    shuffled arm is scored. A stack read from the output rather than popped,
    or a shuffled stack left in its dict, is resident there."""

    import arc.training as training_package

    scene = _step_scene(
        tmp_path, monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
    )
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    resident = []
    real_loss = training_package.sparse_tracking_loss

    def recording(*args, **kwargs):
        emitted = len(model.emitted_stacks)
        alive = sum(reference() is not None for reference in model.emitted_stacks)
        resident.append((emitted, alive))
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)
    refine_iters = 3
    train_cli.evaluate_held_out(
        model=model,
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=3,
        output_dir=tmp_path / "out",
        query_anchors=["0:0", "1:0"],
        confidence_alpha=_EVAL_ALPHA,
        refine_iters=refine_iters,
    )

    assert len(model.emitted_stacks) == 2
    plain = [alive for emitted, alive in resident if emitted == 1]
    shuffled = [alive for emitted, alive in resident if emitted == 2]
    assert len(plain) == refine_iters and len(shuffled) == 1
    assert all(alive <= 1 for alive in plain)
    assert shuffled == [0]


@pytest.mark.parametrize("flag", ["ground_truth_query_anchor", "oracle_query_anchor"])
def test_the_eval_scores_every_pass_in_the_anchor_diagnostics_frame(tmp_path, monkeypatch, flag):
    """The eval-only anchor diagnostics hand the loss a WORLD-frame anchor,
    and every pass must be composed in that frame, as the final one is: under
    the ground-truth or oracle anchor the per-pass curve is the instrument
    for how much of the error refinement removes once the anchor is right,
    so an intermediate pass composed as if its anchor were in the model's
    gauge reports a different curve with nothing to flag it. On the scaled
    gauge (scale 2.25, a 25-degree rotation and a translation), where the two
    frames differ, each pass's entry must equal a forced K=1 eval of that
    pass's field under the same diagnostic."""

    from test_trainer_loop import _cpu_eval_scene

    scene = _cpu_eval_scene(tmp_path, monkeypatch, gauge="scaled")
    height, width = scene.views[0]["img"].shape[-2:]
    plan = plan_record(_record(seq_name="0000"), budget=48, stride=2)
    refine_iters = 3

    def evaluate(model, directory, **extra):
        return train_cli.evaluate_held_out(
            model=model,
            plans=[plan],
            scene_provider=lambda _plan: scene,
            precision="32",
            huber_delta_m=0.05,
            step=3,
            output_dir=tmp_path / directory,
            query_anchors=["0:0"],
            confidence_alpha=_EVAL_ALPHA,
            **{flag: True},
            **extra,
        )

    torch.manual_seed(0)
    refining = _RefiningFakeArc(scene.num_observations, height, width)
    refined = evaluate(refining, "refined", refine_iters=refine_iters)
    expected = []
    for iteration, constant in enumerate(refining.pass_offsets(refine_iters)):
        torch.manual_seed(0)
        plain = _RefiningFakeArc(scene.num_observations, height, width)
        plain.force_offset = constant
        expected.append(evaluate(plain, f"single-{iteration}")["per_scene"][0]["position_loss"])

    assert len(set(expected)) == refine_iters
    assert refined["per_scene"][0]["iteration_position_losses"] == expected
    assert refined[flag] is True


# ----------------------------------------------------------- signatures ---


def test_refinement_signatures_are_keyword_only_and_default_off():
    """Flag-off bit-identity's signature half, as
    test_geometry_signatures_are_keyword_only pins it for the geometry
    inputs (that test also covers set_freeze's and
    assert_trainable_parameter_set's `refine`): every new parameter on the
    model, runtime and trainer surfaces is keyword-only with the flag-off
    default, so no positional caller can shift onto it; Arc.forward and the
    two inference_multiview entry points take it as an ordinary defaulted
    parameter, one iteration; and neither constructor takes a refinement
    parameter, so the Hub config can never serialise one."""

    for function, name, default in (
        (Arc._forward, "refine_iters", 1),
        (Arc.track_for_query, "refinement", None),
        (MotionDecoder.forward, "refinement", None),
        (runtime.anchor_tracks, "refinement", None),
        (train_cli.train_step, "refine_iters", 1),
        (train_cli.train_step, "refine_gamma", train_cli.DEFAULT_REFINE_GAMMA),
        (train_cli.evaluate_held_out, "refine_iters", 1),
    ):
        parameter = inspect.signature(function).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (function.__qualname__, name)
        assert parameter.default == default and type(parameter.default) is type(default)
    from arc.dust3r import inference_multiview

    for function in (Arc.forward, inference_multiview.inference, inference_multiview.loss_of_one_batch):
        parameter = inspect.signature(function).parameters["refine_iters"]
        assert parameter.default == 1 and type(parameter.default) is int
    for name in ("refine", "refine_iters", "refinement"):
        assert name not in inspect.signature(Arc.__init__).parameters
        assert name not in inspect.signature(MotionDecoder.__init__).parameters
    assert train_cli.DEFAULT_REFINE_GAMMA == 0.8
