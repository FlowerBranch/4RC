# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
    ref token on slot 0 and one shared src token on all others).  Unused by
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

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        patch_start_idx: int,
        track_query_idx = 0,
        *,
        views_per_time: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, S, N, C], laid out [camera_token, time_token, patches...]
                at the production call site (patch_start_idx=2)
            patch_start_idx: index where patches start
            views_per_time: 1, the default, is exactly today's per-slot path --
                every slot's row attends over that slot's own patches. At V > 1
                the S = V*T camera-major slots pool into one key set per time
                index and the decoder emits one row per time; see
                merge_time_grouped_tokens.
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

        if views_per_time == 1:
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

            query = query.flatten(0, 1)
            kv = kv.flatten(0, 1)
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
