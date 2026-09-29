# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from arc.models.arc.layers.block import Block, CrossBlock, AdaLNBlock
from arc.models.arc.layers.attention import CrossAttention
from arc.models.arc.layers.rope import RotaryPositionEmbedding2D, PositionGetter

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def merge_time_grouped_tokens(
    tokens: torch.Tensor,
    *,
    patch_start_idx: int,
    track_query_idx: int,
    views_per_time: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(query [B,T,1+P,C], kv [B,T,V*(1+P),C]) with keys pooled by time index.

    ``tokens`` is [B, S, N, C] laid out camera-major (slot = camera*T + t) with
    [camera_token, time_token, patches...] along N.  For each of the T times,
    the pooled key row concatenates all V cameras' blocks for that instant, so
    cross-view correspondence happens inside the attention softmax instead of
    being penalised between independently computed fields.

    tokens[:, :, 0] is the encoder's per-slot camera token -- a per-slot
    attention-evolved summary, NOT a camera identity (the encoder seats one
    ref token on slot 0 and one shared src token on all others; under
    --camera_input the injected per-view pose encoding is ADDED into that
    seat, so the summary then also carries true per-camera pose).  Unused by
    the per-slot path; CONCATENATED here as one extra key token per camera
    block, the way the time token enters the query -- adding it into the
    patches would cross a normalization boundary (patch outputs are normed,
    this token is not: measured RMS 4-18x the patches' at the taps) and
    suppress patch content in every pooled key.

    The query row for time t is the ANCHOR CAMERA's time token for t plus the
    anchor slot's own patches: bit-equal in content to the row slot
    (anchor_camera, t) produces on the per-slot path, so only the keys change.
    NOT a mean of the V time tokens -- those are different tap activations
    (processed through many blocks after the raw injection), so a mean would
    be a blend of them, not the shared value it looks like.
    """

    B, S, _, C = tokens.shape
    T = S // views_per_time
    patches = tokens[:, :, patch_start_idx:, :]
    P = patches.shape[2]
    kv = torch.cat([tokens[:, :, 0:1, :], patches], dim=2)            # [B,S,1+P,C]
    kv = (
        kv.view(B, views_per_time, T, 1 + P, C)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, T, views_per_time * (1 + P), C)                   # [B,T,V*(1+P),C]
    )
    query_patches = patches[:, track_query_idx : track_query_idx + 1].expand(B, T, P, C)
    time_emb = tokens[:, :, 1:2, :].view(B, views_per_time, T, 1, C)[
        :, track_query_idx // T
    ]
    return torch.cat([time_emb, query_patches], dim=2), kv


# MVTracker's correlation read, mirrored: k=16 world-space neighbours per
# query point (mvtracker.py:107-111, configs/model/mvtracker.yaml:18-22), one
# scaled dot-product correlation plus the 3-vector offset per neighbour
# (PointcloudCorrBlock.corr_sample, mvtracker.py:832-840). The correlation
# width is ours: MVTracker correlates its 128-wide track latent against a
# 128-wide feature cloud; the decoder tokens are 1536 wide, so both sides are
# projected to 128 before the dot product. arc.training.runtime imports these
# three numbers for its trainable-set arithmetic and pins the resulting
# count against the built module by test.
REFINER_NEIGHBOURS = 16
REFINER_CORRELATION_DIM = 128
REFINER_FIELD_CHANNELS = 3


@dataclass(frozen=True)
class RefinementInput:
    """What one refinement iteration is conditioned on, beyond the tokens.

    ``previous_field`` is the previous iteration's dense model-gauge
    displacement, (B, R, H, W, 3), zeros on iteration 0, DETACHED by the
    caller. ``anchor_xyz`` is the anchor slot's patch-centre points in the
    model's own gauge, (B, P, 3). ``key_xyz`` is every slot's patch-centre
    points, (B, S, P, 3), slot-major exactly like the decoder's tokens
    (slot = camera*T + t), so the refiner can pool it with the very
    permutation it pools the key patches with.

    Shape checks plus one detach check; nothing here reads a value.
    Finiteness is not CHECKED, which would cost a host sync per anchor per
    iteration, and it is not assumed either: the loss tolerates a
    non-finite prediction at any pixel no correspondence supervises
    (sparse_tracking_loss guards supervised pixels only), so such a pixel
    costs a K=1 run nothing, and at K > 1 it arrives here as a carry pixel.
    TrackRefiner reads every non-finite carry pixel as zero displacement,
    the iteration-0 value, elementwise on device (see its forward), because
    zero-init alone does not make the refiner inert on it: 0 * NaN is NaN,
    so the block's field term and, through attention, every token of the
    pass would go non-finite and the next loss would refuse the step for a
    pixel it never reads. The detach check is a flag read, no sync, and it
    names the bug at its source: a carry still holding iteration k-1's
    graph would chain the iterations back together, and the trainer's next
    backward would fail on a freed graph, far from the cause. The cloud
    comes detached from predicted_pointmaps by construction and is only
    ever read under no_grad, so it needs no such check.
    """

    previous_field: torch.Tensor
    anchor_xyz: torch.Tensor
    key_xyz: torch.Tensor

    def __post_init__(self) -> None:
        if self.previous_field.ndim != 5 or self.previous_field.shape[-1] != 3:
            raise ValueError(
                "previous_field must have shape (B, R, H, W, 3), got "
                f"{tuple(self.previous_field.shape)}"
            )
        if self.anchor_xyz.ndim != 3 or self.anchor_xyz.shape[-1] != 3:
            raise ValueError(
                f"anchor_xyz must have shape (B, P, 3), got {tuple(self.anchor_xyz.shape)}"
            )
        if self.key_xyz.ndim != 4 or self.key_xyz.shape[-1] != 3:
            raise ValueError(
                f"key_xyz must have shape (B, S, P, 3), got {tuple(self.key_xyz.shape)}"
            )
        if self.key_xyz.shape[2] != self.anchor_xyz.shape[1]:
            raise ValueError(
                f"key_xyz has {self.key_xyz.shape[2]} patches per slot but "
                f"anchor_xyz has {self.anchor_xyz.shape[1]}"
            )
        if not (
            self.previous_field.shape[0]
            == self.anchor_xyz.shape[0]
            == self.key_xyz.shape[0]
        ):
            raise ValueError("previous_field, anchor_xyz and key_xyz disagree on B")
        if self.previous_field.requires_grad:
            raise ValueError(
                "previous_field must be detached: the caller owns the carry "
                "between iterations, and a carry with history would chain the "
                "per-iteration graphs the trainer backwards one at a time"
            )


def patch_centre_points(pointmaps: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, S, H, W, 3) pointmaps -> (B, S, P, 3), one point per patch.

    Reads the pixel at (patch_size // 2, patch_size // 2) of every patch
    block, row-major over (H // patch_size, W // patch_size): the SAME order
    PatchEmbed flattens its Conv2d output in (conv -> flatten(2).transpose(1,
    2), dinov2/layers/patch_embed.py:65-85) and TrackRefiner.field_embed
    below, so patch token p and point p are the same patch. A centre read,
    not a mean over the block: a mean straddles depth edges and lands in
    free space between two surfaces, where no token's content lives.
    """

    if pointmaps.ndim != 5 or pointmaps.shape[-1] != 3:
        raise ValueError(
            f"pointmaps must have shape (B, S, H, W, 3), got {tuple(pointmaps.shape)}"
        )
    B, S, H, W, _ = pointmaps.shape
    if H % patch_size or W % patch_size:
        raise ValueError(
            f"pointmap grid {H}x{W} is not a multiple of patch_size={patch_size}"
        )
    centre = patch_size // 2
    return pointmaps[:, :, centre::patch_size, centre::patch_size].reshape(B, S, -1, 3)


def _pool_slots_by_time(x: torch.Tensor, views_per_time: int) -> torch.Tensor:
    """(B, S, N, D) slot-major -> (B, T, V*N, D), the V slots of each time side by side.

    The identical view/permute/reshape merge_time_grouped_tokens applies to
    kv above, kept as one helper so the refiner pools its key PATCHES and
    its key POINTS with the same permutation and the two cannot disagree --
    a test asserts the pooled patches equal kv's patch columns.
    """

    B, S, N, D = x.shape
    if S % views_per_time != 0:
        raise ValueError(
            f"views_per_time={views_per_time} does not divide the {S} observation slots"
        )
    T = S // views_per_time
    return (
        x.view(B, views_per_time, T, N, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, T, views_per_time * N, D)
    )


class TrackRefiner(nn.Module):
    """The additive term one refinement iteration adds to the query PATCH tokens.

    Two sources, both keyed off the previous iteration's field. (a) A kNN
    read into the model's OWN predicted world-space cloud, MVTracker's
    actual mechanism (PointcloudCorrBlock.corr_sample, mvtracker.py:798-844;
    kNN via cdist + topk, _knn_torch at :75-79): the anchor's patch centres
    displaced by the previous field are the current estimate, the k nearest
    cloud points of the row's own slots are gathered, and each contributes
    one correlation between the projected query token and the projected key
    token plus the 3-vector offset neighbour - estimate, concatenated and
    projected back to the token width. (b) The previous field itself,
    patch-embedded by a Conv2d(kernel=stride=patch) exactly like the depth
    injection (arc/models/arc/dinov2/vision_transformer.py:191-196), so the
    token also sees the dense displacement it is refining, not only its
    centre pixel.

    Zero-init on field_embed and read_proj makes iteration 0 of a fresh
    refiner bit-equal to today's decoder (a test pins this). query_proj and
    key_proj keep the DEFAULT init on purpose: with all three at zero the
    correlation channel is a fixed point -- read_proj's correlation columns
    see gradient proportional to corr, and corr is zero while either
    projection is zero -- so only the offset columns would ever learn.

    The cloud, the kNN indices and the offsets carry NO gradient: under
    every temporal freeze mode Arc.reconstruct runs the depth head and the
    camera decoder inside its frozen_reconstruction no_grad block,
    predicted_pointmaps detaches on top of that, and the read below is under
    no_grad besides. Gradient reaches the refiner only through the
    displacement -- previous_field is the caller's detached carry, so
    through THIS iteration's field_embed and read_proj weights and the token
    content -- which is what the freeze intends: the frozen geometry heads
    are read, never trained through. No depth-validity filtering: this
    cloud is the model's own prediction, defined at every pixel; MVTracker's
    corr_filter_invalid_depth exists for handed sensor depth with holes, and
    MVTracker itself ships it off (mvtracker.py:112,
    configs/model/mvtracker.yaml:23, hubconf.py:63). No checkpoint()
    anywhere in here: the activations are small next to one decoder block's,
    and a checkpoint would be one more lambda that could late-bind (ce12837).
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        *,
        neighbours: int = REFINER_NEIGHBOURS,
        correlation_dim: int = REFINER_CORRELATION_DIM,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.neighbours = neighbours
        self.correlation_dim = correlation_dim
        # ALWAYS constructed, never gated on a training flag, so the
        # state-dict key set is stable: one checkpoint runs at any
        # --refine_iters, and the released checkpoint loads with these as
        # legacy missing keys (Arc.LEGACY_CHECKPOINT_MISSING_KEYS) -- the
        # time_index_embedding contract. fork_rng because Conv2d/Linear
        # default init draws from the global RNG, and that draw would
        # silently shift every later randn() against the pre-refiner code:
        # K=1 must be bit-identical including the seeded construction path
        # (arc/models/arc/dinov2/vision_transformer.py:178-201 is the
        # precedent, comment and all). All four go in ONE fork so the two
        # default-init projections' draws are contained too, not only the
        # ones the zeros overwrite. devices=[] preserves the CPU generator
        # only, which covers every in-repo construction (CPU or meta device);
        # a CUDA-default-device construction would still advance the CUDA
        # stream.
        with torch.random.fork_rng(devices=[]):
            self.field_embed = nn.Conv2d(
                REFINER_FIELD_CHANNELS, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.query_proj = nn.Linear(embed_dim, correlation_dim)
            self.key_proj = nn.Linear(embed_dim, correlation_dim)
            self.read_proj = nn.Linear(neighbours * (1 + 3), embed_dim)
        nn.init.zeros_(self.field_embed.weight)
        nn.init.zeros_(self.field_embed.bias)
        nn.init.zeros_(self.read_proj.weight)
        nn.init.zeros_(self.read_proj.bias)

    def forward(
        self,
        query_patches: torch.Tensor,
        patches: torch.Tensor,
        refinement: RefinementInput,
        *,
        views_per_time: int,
        merge: bool,
    ) -> torch.Tensor:
        """(B, R, P, C) term for the query patch tokens; never the time token.

        ``query_patches`` is (B, R, P, C), the expanded anchor patches of the
        decoder's query rows; ``patches`` is (B, S, P, C), every slot's
        patches, the kv source. Under ``merge`` a row is a time and its keys
        are the V slots of that time pooled; otherwise a row is a slot and
        its keys are that slot alone -- the same rows the decoder's keys use.
        """

        B, R, P, C = query_patches.shape
        # A non-finite carry pixel reads as ZERO displacement, the iteration-0
        # value. The loss tolerates a non-finite prediction wherever no
        # correspondence supervises (sparse_tracking_loss guards supervised
        # pixels only), so today such a pixel costs nothing; at K > 1 it
        # would enter here, and zero-init does not make this module inert on
        # it -- 0 * NaN is NaN -- so field_embed's term for that block and,
        # through attention, every token of the pass would go non-finite, and
        # the next loss would refuse the whole step for a pixel it never
        # reads. Elementwise on device, no host sync; one (R, H, W, 3) fp32
        # transient per tap, freed on return.
        previous_field = torch.nan_to_num(
            refinement.previous_field, nan=0.0, posinf=0.0, neginf=0.0
        )
        # (1) Keys: pooled by time under the merge, with the ONE permutation
        # helper for both patches and points.
        if merge:
            key_patches = _pool_slots_by_time(patches, views_per_time)
            key_xyz = _pool_slots_by_time(refinement.key_xyz, views_per_time)
        else:
            key_patches = patches
            key_xyz = refinement.key_xyz
        if key_xyz.shape[:3] != key_patches.shape[:3]:
            raise ValueError(
                "key_xyz must carry one point per key patch token, got "
                f"{tuple(key_xyz.shape)} against patches {tuple(key_patches.shape)}"
            )
        key_count = key_patches.shape[2]
        if key_count < self.neighbours:
            # topk cannot take more than there are, and a narrower read would
            # not fit read_proj; MVTracker asserts the same (mvtracker.py:783).
            raise ValueError(
                f"The kNN read needs at least {self.neighbours} key points per "
                f"row and this grid has {key_count}"
            )
        # (2) The read's geometry: float32, autocast off, no grad, the way
        # predicted_pointmaps handles the same geometry. Distances are a
        # selection, not learning, and the selection must not depend on
        # which autocast the caller runs: the trainer, the held-out eval
        # (autocast_context(precision)) and inference (bf16-mixed) all run
        # this decoder under reduced precision.
        with torch.no_grad(), torch.autocast(
            device_type=previous_field.device.type, enabled=False
        ):
            # The ONE centre-pixel rule, so the estimate for patch p is
            # anchor point p moved by field pixel p, and the helper's own
            # check that the grid divides. The rows must be the decoder's
            # output rows -- times under the merge, slots otherwise -- or
            # the reshape below would report a numel mismatch that names
            # neither rule.
            field_at_centres = patch_centre_points(previous_field, self.patch_size).float()
            if field_at_centres.shape[1] != R or field_at_centres.shape[2] != P:
                raise ValueError(
                    "previous_field must have one row per decoder output row "
                    "(times under the merge, slots otherwise) at the image grid, "
                    f"got {tuple(previous_field.shape)} for {R} rows of {P} patches"
                )
            estimate = (
                refinement.anchor_xyz[:, None].float() + field_at_centres
            ).reshape(B * R, P, 3)
            cloud = key_xyz.reshape(B * R, key_count, 3).float()
            distances = torch.cdist(estimate, cloud)                            # (B*R, P, Nk)
            indices = distances.topk(self.neighbours, dim=-1, largest=False).indices
            row = torch.arange(B * R, device=indices.device)[:, None, None]
            neighbour_xyz = cloud[row, indices]                                   # (B*R, P, k, 3)
            offsets = neighbour_xyz - estimate[:, :, None, :]
        # (3) Correlation, with grad: MVTracker's einsum, divided by the square
        # root of the CORRELATION width. MVTracker writes it as
        # sqrt(self.C / self.groups) at groups=1 (mvtracker.py:833), and its
        # self.C is the width its feature projections emit -- the analogue of
        # self.correlation_dim here, NOT this method's local C, the token
        # width, which is twelve times wider at production and would shrink
        # every correlation by 1/sqrt(12) against the intended scale.
        query_features = self.query_proj(query_patches.reshape(B * R, P, C))
        key_features = self.key_proj(key_patches.reshape(B * R, key_count, C))
        neighbour_features = key_features[row, indices]                          # (B*R, P, k, c)
        correlation = torch.einsum(
            "bpc,bpkc->bpk", query_features, neighbour_features
        ) / (self.correlation_dim ** 0.5)
        # (4) [corr, offset] per neighbour, concatenated the way corr_sample
        # concatenates (mvtracker.py:838-840), projected to the token width.
        read = torch.cat(
            [correlation[..., None], offsets.to(correlation.dtype)], dim=-1
        ).reshape(B * R, P, self.neighbours * (1 + 3))
        read_term = self.read_proj(read)
        # (5) The dense field, patch-embedded in PatchEmbed's flatten order.
        _, _, H, W, _ = previous_field.shape
        field_term = (
            self.field_embed(previous_field.reshape(B * R, H, W, 3).permute(0, 3, 1, 2))
            .flatten(2)
            .transpose(1, 2)
        )
        # (6)
        return (read_term + field_term).view(B, R, P, C)


class MotionDecoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_adaln=False,
        has_self_attention=True,
        has_cross_attention=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.use_adaln = use_adaln
        self.has_self_attention = has_self_attention
        self.has_cross_attention = has_cross_attention

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        if self.has_cross_attention:
            self.cross_blocks = nn.ModuleList(
                [
                    CrossBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        cross_attn_init_values=init_values,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                        attn_class=CrossAttention,
                    )
                    for _ in range(depth)
                ]
            )

        if self.has_self_attention:
            self.self_blocks = nn.ModuleList(
                [
                    (AdaLNBlock if use_adaln else block_fn)(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )

        self.depth = depth
        # LAST, after every block: TrackRefiner constructs under fork_rng, so
        # the RNG the blocks above drew from is untouched by it, and anything
        # built after this decoder (Arc's track head) draws what it drew
        # before the refiner existed. State-dict keys: refiner.{field_embed,
        # query_proj,key_proj,read_proj}.{weight,bias}.
        self.refiner = TrackRefiner(embed_dim, patch_size)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        patch_start_idx: int,
        track_query_idx = 0,
        *,
        views_per_time: int = 1,
        merge: bool = False,
        refinement: RefinementInput | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, S, N, C], laid out [camera_token, time_token, patches...]
                at the production call site (patch_start_idx=2)
            patch_start_idx: index where patches start
            views_per_time: how many camera-major slots share each time index
                (S = V*T). Read only under ``merge``; the per-slot path ignores
                it beyond the divisibility check.
            merge: False, the default, is exactly today's per-slot path --
                every slot's row attends over that slot's own patches, whatever
                views_per_time says. True pools the V slots of each time index
                into one key set and emits one row per time; see
                merge_time_grouped_tokens. Well-defined at V=1 too, where the
                pooled key row is [camera_token(t), patches(t)] -- the per-slot
                keys prefixed by the camera token -- so a single-camera window
                under the merge runs the same head geometry as a multi-camera
                one instead of silently falling back to the per-slot branch.
            refinement: None, the default, is exactly today's forward on
                either branch. Otherwise ONE refinement iteration: after the
                branch has built query and kv exactly as today, TrackRefiner's
                term is added to the query PATCH tokens -- never the time
                token, which is also the AdaLN condition -- before the block
                loop. The keys are untouched and so is RoPE: the refiner
                enters as token CONTENT and positions stay the integer grid.
                Runs once per tap, because this method does.
        """
        B, S, _, C = tokens.shape
        _, _, _, H, W = images.shape

        patches = tokens[:, :, patch_start_idx:, :] # [B, S, P, C]
        P = patches.shape[2]

        if S % views_per_time != 0:
            raise ValueError(
                f"views_per_time={views_per_time} does not divide the "
                f"{S} observation slots"
            )

        if not merge:
            out_rows = S

            query_patches = patches[:, track_query_idx:track_query_idx+1, :, :]
            query_patches = query_patches.expand(B, S, P, C)

            time_emb = tokens[:, :, 1:2, :]

            time_cond = None
            if self.use_adaln:
                time_cond = time_emb.flatten(0, 2)

            # Concat time token to query patches
            query = torch.cat([time_emb, query_patches], dim=2) # [B, S, 1+P, C]

            kv = patches

            # 3. Prepare Positional Embeddings
            pos_q = None
            pos_k = None

            if self.position_getter is not None:
                pos_patches = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

                pos_patches = pos_patches + 1

                pos_time = torch.zeros(B * S, 1, 2, device=tokens.device, dtype=pos_patches.dtype)

                pos_q = torch.cat([pos_time, pos_patches], dim=1)

                pos_k = pos_patches

                pos_cross = (pos_q, pos_k)
        else:
            out_rows = S // views_per_time

            query, kv = merge_time_grouped_tokens(
                tokens,
                patch_start_idx=patch_start_idx,
                track_query_idx=track_query_idx,
                views_per_time=views_per_time,
            )

            time_cond = None
            if self.use_adaln:
                time_cond = query[:, :, 0:1, :].flatten(0, 2)

            pos_q = None
            pos_k = None

            if self.position_getter is not None:
                pos_patches = self.position_getter(B * out_rows, H // self.patch_size, W // self.patch_size, device=images.device)

                pos_patches = pos_patches + 1

                pos_time = torch.zeros(B * out_rows, 1, 2, device=tokens.device, dtype=pos_patches.dtype)

                pos_q = torch.cat([pos_time, pos_patches], dim=1)

                # Every camera block of a pooled key row repeats the query's
                # own layout: the block's camera token at the reserved (0,0),
                # then the same 2D patch grid -- matching the camera-major
                # block order of merge_time_grouped_tokens' reshape.
                pos_k = pos_q.repeat(1, views_per_time, 1)

                pos_cross = (pos_q, pos_k)

        if refinement is not None:
            # AFTER the branch built query and kv, BEFORE the flatten: the
            # query patches are one expanded copy of the anchor slot's patches
            # on both branches, so a per-row term has to land here, and adding
            # it to `tokens` upstream would have changed the keys too. Rebuilt
            # by cat, never added in place: the tensor the branch built is
            # left unmutated (on the merge branch time_cond is a view of it,
            # and an in-place add would bump the version of the storage they
            # share), and the contract fixes this spelling.
            term = self.refiner(
                query[:, :, 1:], patches, refinement,
                views_per_time=views_per_time, merge=merge,
            )
            query = torch.cat(
                [query[:, :, :1], query[:, :, 1:] + term.to(query.dtype)], dim=2
            )
            if merge and self.use_adaln:
                # Column 0 is untouched, so the values are identical; the
                # re-slice is about memory. AdaLNBlock's SiLU saves its input,
                # and time_cond as sliced above is a view of the PRE-insert
                # query, so it would keep that whole (B, T, 1+P, C) storage
                # alive until this iteration's backward (about 143 MB per tap
                # in fp32 at T=24). The rebuilt query is saved by cross block
                # 0 regardless, so a view of it costs nothing. The per-slot
                # branch's time_cond is a view of `tokens`, retained anyway.
                time_cond = query[:, :, 0:1, :].flatten(0, 2)

        # Hoisted out of the two branches, which both ended with exactly these
        # two lines: flatten(0, 1) is a reshape (a view where the strides
        # allow, a copy otherwise) and changes no value on either path, so
        # moving it past the insert above is value-neutral.
        query = query.flatten(0, 1)
        kv = kv.flatten(0, 1)

        # The checkpoint lambdas must bind the block index at definition time:
        # non-reentrant recomputation runs during backward, after this loop has
        # finished, so a closure over the loop variable would recompute every
        # checkpointed segment with the last block's weights and silently
        # corrupt the gradients.
        for cur_i in range(self.depth):
            # Cross Attention
            if self.has_cross_attention:
                if cur_i > 1 and self.training:
                    query = checkpoint(
                        lambda q, k, p, i=cur_i: self.cross_blocks[i](q, k, pos=p),
                        query, kv, pos_cross,
                        use_reentrant=False
                    )
                else:
                    query = self.cross_blocks[cur_i](query, kv, pos=pos_cross)

            if self.has_self_attention:
                # Self Attention
                if cur_i > 1 and self.training:
                    if self.use_adaln:
                        query = checkpoint(
                            lambda q, c, p, i=cur_i: self.self_blocks[i](q, cond=c, pos=p),
                            query, time_cond, pos_q,
                            use_reentrant=False
                        )
                    else:
                        query = checkpoint(
                            lambda q, p, i=cur_i: self.self_blocks[i](q, pos=p),
                            query, pos_q,
                            use_reentrant=False
                        )
                else:
                    if self.use_adaln:
                        query = self.self_blocks[cur_i](query, cond=time_cond, pos=pos_q)
                    else:
                        query = self.self_blocks[cur_i](query, pos=pos_q)

        query = query.view(B, out_rows, 1+P, C)

        return query
