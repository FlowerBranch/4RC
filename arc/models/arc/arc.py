import inspect
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from addict import Dict
from huggingface_hub import PyTorchModelHubMixin

from arc.models.arc.utils.transform import (
    pose_encoding_to_extri_intri,
    affine_inverse,
    get_extrinsic_from_camray,
    as_homogeneous,
)
from arc.models.arc.dinov2.dinov2 import DinoV2
from arc.models.arc.heads.dualdpt import DualDPT
from arc.models.arc.heads.cam_dec import CameraDec
from arc.models.arc.heads.motiondecoder import MotionDecoder
from arc.models.arc.heads.dpt_head import DPTHead

from arc.dust3r.utils.image import ImgRenormalize
from arc.models.arc.utils.geometry import unproject_depth_map_to_point_map


class Arc(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="4RC",
    repo_url="https://github.com/Luo-Yihang/4RC",
):
    PATCH_SIZE = 14
    TIME_INDEX_KEY = "time_index"
    MAX_TIME_INDICES = 32
    # Freeze presets that train the temporal-tracking stack; set_freeze is the
    # single authority on mode names.
    TEMPORAL_FREEZE_MODES = (
        "temporal_tracking",
        "temporal_tracking_global_attention",
    )
    LEGACY_CHECKPOINT_MISSING_KEYS = (
        "backbone.pretrained.time_index_embedding.weight",
    )
    LEGACY_SAFETENSOR_ALIASES = {
        "head.scratch.output_conv2_aux.1.2.weight":
            "head.scratch.output_conv2_aux.0.2.weight",
        "head.scratch.output_conv2_aux.1.2.bias":
            "head.scratch.output_conv2_aux.0.2.bias",
        "head.scratch.output_conv2_aux.2.2.weight":
            "head.scratch.output_conv2_aux.0.2.weight",
        "head.scratch.output_conv2_aux.2.2.bias":
            "head.scratch.output_conv2_aux.0.2.bias",
        "head.scratch.output_conv2_aux.3.2.weight":
            "head.scratch.output_conv2_aux.0.2.weight",
        "head.scratch.output_conv2_aux.3.2.bias":
            "head.scratch.output_conv2_aux.0.2.bias",
    }

    def __init__(
        self,
        freeze="none",
        motion_decoder_depth=4,
        motion_decoder_has_self_attention=True,
        motion_decoder_has_cross_attention=True,
        motion_decoder_use_adaln=True,
        track_head_activation="inv_log",
        max_time_indices=MAX_TIME_INDICES,
    ):
        super().__init__()

        # Keys the checkpoint loader accepted as legacy gaps and zero-filled.
        # Callers use this to warn that a feature is silently inactive.
        self.consumed_legacy_missing_keys = frozenset()

        if isinstance(max_time_indices, bool) or not isinstance(max_time_indices, int):
            raise TypeError("max_time_indices must be a positive integer")
        if max_time_indices <= 0:
            raise ValueError("max_time_indices must be a positive integer")
        self.max_time_indices = max_time_indices

        self.backbone = DinoV2(
            name="vitg",
            out_layers=[19, 27, 33, 39],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
            has_time_token=True,
            max_time_indices=max_time_indices,
        )

        self.head = DualDPT(
            dim_in=3072,
            output_dim=2,
            features=256,
        )

        self.cam_dec = CameraDec(dim_in=3072)

        self.motion_decoder = MotionDecoder(
            patch_size=self.PATCH_SIZE, 
            embed_dim=1536,
            use_adaln=motion_decoder_use_adaln, 
            depth=motion_decoder_depth, 
            has_self_attention=motion_decoder_has_self_attention, 
            has_cross_attention=motion_decoder_has_cross_attention,
        )
        
        self.track_head = DPTHead(
            dim_in=1536,
            output_dim=4,
            activation=track_head_activation,
            conf_activation="expp1",
            intermediate_layer_idx=[0, 1, 2, 3],
        )

        self.set_freeze(freeze)

    def _preprocess_input(self, views):
        images = torch.stack([view["img"] for view in views], dim=1)
        images = ImgRenormalize(images)
        track_query_idx = 0 if "track_query_idx" not in views[0] else views[0]["track_query_idx"]
        track_query_idx_list = self._normalize_track_query_idx(track_query_idx, images.shape[1])
        time_indices = self._preprocess_time_indices(
            views,
            batch_size=images.shape[0],
            device=images.device,
        )

        return images, track_query_idx_list, time_indices

    def _preprocess_time_indices(self, views, batch_size, device):
        has_time_index = [self.TIME_INDEX_KEY in view for view in views]
        if not any(has_time_index):
            return None
        if not all(has_time_index):
            raise ValueError(
                f"Either every view must provide '{self.TIME_INDEX_KEY}' or none may provide it"
            )

        per_view_indices = []
        for view_idx, view in enumerate(views):
            value = view[self.TIME_INDEX_KEY]
            try:
                value = torch.as_tensor(value, device=device)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"View {view_idx} '{self.TIME_INDEX_KEY}' must contain integer values"
                ) from exc

            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise TypeError(
                    f"View {view_idx} '{self.TIME_INDEX_KEY}' must contain integer values"
                )

            if value.numel() == 1:
                value = value.reshape(1).expand(batch_size)
            elif value.numel() == batch_size:
                value = value.reshape(batch_size)
            else:
                raise ValueError(
                    f"View {view_idx} '{self.TIME_INDEX_KEY}' must be a scalar or contain "
                    f"one value per batch element ({batch_size}), got {value.numel()}"
                )

            per_view_indices.append(value.to(dtype=torch.long))

        time_indices = torch.stack(per_view_indices, dim=1)
        if torch.any(time_indices < 0) or torch.any(time_indices >= self.max_time_indices):
            min_index = int(time_indices.min().item())
            max_index = int(time_indices.max().item())
            raise ValueError(
                f"'{self.TIME_INDEX_KEY}' values must be in [0, {self.max_time_indices - 1}], "
                f"got range [{min_index}, {max_index}]"
            )
        return time_indices

    def _normalize_track_query_idx(self, track_query_idx, num_views):
        if isinstance(track_query_idx, torch.Tensor):
            track_query_idx = track_query_idx.detach().cpu().flatten().tolist()
        elif isinstance(track_query_idx, (list, tuple)):
            track_query_idx = list(track_query_idx)
        else:
            track_query_idx = [int(track_query_idx)]

        track_query_idx = [int(idx) for idx in track_query_idx]
        track_query_idx = [idx for idx in track_query_idx if 0 <= idx < num_views]
        if not track_query_idx:
            track_query_idx = [0]
        return track_query_idx
    
    def _postprocess_output(self, preds, use_ray_pose=False):
        H, W = preds['depth'].shape[2:4]

        if use_ray_pose:
            self._process_ray_pose_estimation(preds, H, W)
        else:
            self._process_camera_estimation(H, W, preds)

        output_list = []
        
        B, N = preds['depth'].shape[:2]

        depth_conf_list = torch.unbind(preds["depth_conf"], dim=1)
        depth_np = preds["depth"].detach().cpu().numpy()
        extrinsic_np = (preds["extrinsics"] if use_ray_pose else preds["extrinsics_token"]).detach().cpu().numpy()
        intrinsic_np = (preds["intrinsics"] if use_ray_pose else preds["intrinsics_token"]).detach().cpu().numpy()

        if "track" not in preds:
            preds["track"] = torch.ones(B, N, H, W, 3).to(preds["depth"].device)
            preds["conf_track"] = preds["depth_conf"]
            print("Warning: track not found in preds, using world_points instead")

        track_query_idx_list = self._normalize_track_query_idx(
            preds.get("track_query_idx", 0), N
        )
        track_multi = preds.get("track_multi")
        conf_track_multi = preds.get("conf_track_multi")
        if track_multi is None or conf_track_multi is None:
            track_multi = preds["track"].unsqueeze(1)
            conf_track_multi = preds["conf_track"].unsqueeze(1)
            track_query_idx_list = [track_query_idx_list[0]]

        track_multi_list = torch.unbind(track_multi, dim=2)  # list over views
        conf_track_multi_list = torch.unbind(conf_track_multi, dim=2)

        all_world_points = []
        for b in range(B):
            wp, _ = unproject_depth_map_to_point_map(
                depth_np[b][..., None], 
                extrinsic_np[b], 
                intrinsic_np[b]
            )
            wp_tensor = torch.from_numpy(wp).to(device=preds["depth"].device, dtype=preds["depth"].dtype)
            all_world_points.append(wp_tensor)
            
        all_world_points = torch.stack(all_world_points, dim=0) # [B, N, H, W, 3]

        world_points_list = torch.unbind(all_world_points, dim=1)

        track_query_idx_tensor = torch.tensor(track_query_idx_list, device=preds["depth"].device)
        
        for i in range(N):
            pts3d_world = world_points_list[i] # [B, H, W, 3]

            track_list_per_query = []
            conf_list_per_query = []
            for q_i, q_idx in enumerate(track_query_idx_list):
                track_q = track_multi_list[i][:, q_i]
                conf_q = conf_track_multi_list[i][:, q_i]
                track_q = track_q + world_points_list[q_idx]
                track_list_per_query.append(track_q)
                conf_list_per_query.append(conf_q)

            track = track_list_per_query[0]
            conf_track = conf_list_per_query[0]
            track_multi_out = torch.stack(track_list_per_query, dim=1)
            conf_track_multi_out = torch.stack(conf_list_per_query, dim=1)
            
            extrinsic_w2c = torch.from_numpy(extrinsic_np[b][i]).to(
                preds["depth"].device
            )
            extrinsic_c2w = affine_inverse(as_homogeneous(extrinsic_w2c))
            intrinsic_matrix = torch.from_numpy(intrinsic_np[b][i]).to(
                preds["depth"].device
            )

            output_list.append({
                "pts": pts3d_world,
                "conf": depth_conf_list[i],
                "track": track,
                "conf_track": conf_track,
                "track_multi": track_multi_out,
                "conf_track_multi": conf_track_multi_out,
                "track_query_idx": track_query_idx_tensor,
                "extrinsic": extrinsic_c2w,
                "intrinsic": intrinsic_matrix,
            })

        return output_list

    def set_freeze(self, freeze):
        supported_modes = {"none", *self.TEMPORAL_FREEZE_MODES}
        if freeze not in supported_modes:
            raise ValueError(
                f"Unknown freeze mode '{freeze}'. Expected one of {sorted(supported_modes)}"
            )

        self.requires_grad_(True)
        if freeze in self.TEMPORAL_FREEZE_MODES:
            self.requires_grad_(False)
            self.backbone.pretrained.time_index_embedding.requires_grad_(True)
            self.motion_decoder.requires_grad_(True)
            self.track_head.requires_grad_(True)
        if freeze == "temporal_tracking_global_attention":
            # Cross-view fusion can only be learned in the blocks that attend
            # across frames, so this mode adds exactly the global-attention
            # blocks (i >= alt_start, odd i) and leaves the interleaved local
            # blocks frozen. Attribute access is deliberately unguarded: a
            # backbone without `blocks`/`alt_start` must fail loudly here
            # rather than silently train nothing extra.
            encoder = self.backbone.pretrained
            if encoder.alt_start == -1:
                raise ValueError(
                    "Freeze mode 'temporal_tracking_global_attention' requires "
                    "an encoder with alternating attention (alt_start != -1)"
                )
            for index in range(encoder.alt_start, len(encoder.blocks)):
                if index % 2 == 1:
                    encoder.blocks[index].requires_grad_(True)

        self.freeze = freeze

    def get_trainable_parameter_report(self):
        parameters = [
            (name, parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        return {
            "parameters": parameters,
            "tensor_count": len(parameters),
            "parameter_count": sum(count for _, count in parameters),
        }

    @classmethod
    def _validate_checkpoint_incompatibility(
        cls,
        missing_keys,
        unexpected_keys,
        *,
        strict,
    ):
        missing_keys = set(missing_keys)
        unexpected_keys = set(unexpected_keys)
        allowed_missing_keys = (
            set()
            if strict
            else set(cls.LEGACY_CHECKPOINT_MISSING_KEYS)
        )
        invalid_missing_keys = missing_keys - allowed_missing_keys

        if invalid_missing_keys or unexpected_keys:
            details = []
            if invalid_missing_keys:
                details.append(
                    f"missing keys: {sorted(invalid_missing_keys)}"
                )
            if unexpected_keys:
                details.append(
                    f"unexpected keys: {sorted(unexpected_keys)}"
                )
            raise RuntimeError(
                "Checkpoint is incompatible with Arc; " + "; ".join(details)
            )

        return frozenset(missing_keys & allowed_missing_keys)

    @classmethod
    def _load_as_safetensor(
        cls,
        model,
        model_file,
        map_location,
        strict,
    ):
        from safetensors.torch import load_model

        verified_legacy_aliases = (
            cls._verify_legacy_safetensor_aliases(model_file)
            if not strict
            else set()
        )

        load_kwargs = {"strict": False}
        supports_device = "device" in inspect.signature(load_model).parameters
        if supports_device:
            load_kwargs["device"] = map_location

        missing_keys, unexpected_keys = load_model(
            model,
            model_file,
            **load_kwargs,
        )
        if not supports_device and str(map_location) != "cpu":
            model.to(map_location)

        unexpected_keys = set(unexpected_keys) - verified_legacy_aliases
        model.consumed_legacy_missing_keys = cls._validate_checkpoint_incompatibility(
            missing_keys,
            unexpected_keys,
            strict=strict,
        )
        return model

    @classmethod
    def _verify_legacy_safetensor_aliases(cls, model_file):
        from safetensors import safe_open

        verified_aliases = set()
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            for alias, canonical in cls.LEGACY_SAFETENSOR_ALIASES.items():
                if alias not in checkpoint_keys:
                    continue
                if canonical not in checkpoint_keys:
                    raise RuntimeError(
                        "Checkpoint is incompatible with Arc; legacy safetensors "
                        f"alias '{alias}' has no canonical tensor '{canonical}'"
                    )

                alias_tensor = checkpoint.get_tensor(alias)
                canonical_tensor = checkpoint.get_tensor(canonical)
                if (
                    alias_tensor.shape != canonical_tensor.shape
                    or alias_tensor.dtype != canonical_tensor.dtype
                    or not torch.equal(alias_tensor, canonical_tensor)
                ):
                    raise RuntimeError(
                        "Checkpoint is incompatible with Arc; legacy safetensors "
                        f"alias '{alias}' conflicts with canonical tensor "
                        f"'{canonical}' (shared-module checkpoint is ambiguous)"
                    )
                verified_aliases.add(alias)

        return verified_aliases

    @classmethod
    def _load_as_pickle(
        cls,
        model,
        model_file,
        map_location,
        strict,
    ):
        state_dict = torch.load(
            model_file,
            map_location=torch.device(map_location),
            weights_only=True,
        )
        incompatibility = model.load_state_dict(state_dict, strict=False)
        model.consumed_legacy_missing_keys = cls._validate_checkpoint_incompatibility(
            incompatibility.missing_keys,
            incompatibility.unexpected_keys,
            strict=strict,
        )
        model.eval()
        return model
    
    def forward(
        self,
        views,
        use_ray_pose: bool = False,
        profiling=False,
        force_no_output_conversion=False,
        inference_track = True,
        **kwargs
    ):
        if profiling:
            profiling_info = {} if profiling else None
            start_time = time.time()

        images, track_query_idx, time_indices = self._preprocess_input(views)

        predictions = self._forward(
            images,
            track_query_idx,
            inference_track=inference_track,
            time_indices=time_indices,
            **kwargs,
        )
        
        if not self.training and not force_no_output_conversion:
            predictions = self._postprocess_output(predictions, use_ray_pose)

        if profiling:
            profiling_info['total_time'] = time.time() - start_time
            return predictions, profiling_info
        else:
            return predictions

    def _forward(
        self,
        x: torch.Tensor,
        track_query_idx,
        ref_view_strategy: str = "first",
        inference_track: bool = True,
        time_indices=None,
    ) -> Dict[str, torch.Tensor]:
        feats = self.encode_features(
            x,
            ref_view_strategy=ref_view_strategy,
            time_indices=time_indices,
        )

        track_query_idx_list = self._normalize_track_query_idx(track_query_idx, x.shape[1])
        output_track_query_idx = torch.tensor(track_query_idx_list, device=x.device)

        output = self.reconstruct(feats, x)

        if inference_track:
            track_list = []
            conf_list = []
            for query_idx in track_query_idx_list:
                track, track_conf = self.track_for_query(feats, x, query_idx)
                track_list.append(track)
                conf_list.append(track_conf)

            output["track"] = track_list[0]
            output["conf_track"] = conf_list[0]
            output["track_multi"] = torch.stack(track_list, dim=1)
            output["conf_track_multi"] = torch.stack(conf_list, dim=1)

        output['track_query_idx'] = output_track_query_idx

        return output

    def encode_features(
        self,
        x: torch.Tensor,
        ref_view_strategy: str = "first",
        time_indices=None,
    ):
        """Run the backbone once and return its tap list.

        Split out of :meth:`_forward` so a caller that needs several track
        queries can pay for the encoder once and drive the per-query heads
        itself.  ``feats`` does not depend on which frame is the query.
        """

        feats, _ = self.backbone(
            x,
            ref_view_strategy=ref_view_strategy,
            time_indices=time_indices,
        )
        return feats

    def reconstruct(self, feats, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Depth head and camera decoder, keyed off the shared backbone taps."""

        H, W = x.shape[-2], x.shape[-1]
        # Process features through depth head
        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            # Under every temporal freeze mode the depth head and camera decoder
            # are frozen and their outputs are consumed only through detached
            # paths (arc.training.sparse_tracking._predicted_pointmaps is
            # no_grad plus .detach()), so retaining the dual-pyramid DPT graph
            # costs about 1.2 GB per observation for nothing. This holds even
            # when encoder blocks are trainable: no loss reads these outputs
            # undetached, so no gradient is lost by cutting the graph here.
            # Keep the values, drop the graph. Nested inside the autocast
            # above, which is doing real work forcing fp32 for these heads
            # under an outer bf16 autocast.
            frozen_reconstruction = (
                getattr(self, "freeze", "none") in self.TEMPORAL_FREEZE_MODES
            )
            with torch.no_grad() if frozen_reconstruction else nullcontext():
                output = self.head(feats, H, W, patch_start_idx=0)
                pose_enc = self.cam_dec(feats[-1][1])
            output["pose_enc"] = pose_enc
            output["pose_enc_list"] = [pose_enc]
        return output

    def track_for_query(self, feats, x: torch.Tensor, query_idx: int):
        """One query frame's dense displacement field and its confidence.

        This is the body of the Q loop.  Its activations are the bulk of a
        training step's memory, and they are freed once this query's loss has
        been backwarded -- which is why a caller supervising several anchors
        drives this per query rather than asking for a stacked Q axis.
        """

        frames_chunk_size = 1 if self.training else 8
        aggregated_track_tokens_list = []
        for feature in feats:
            feature = torch.cat(
                [feature[1].unsqueeze(2), feature[2].unsqueeze(2), feature[0]],
                dim=2,
            )[..., 1536:] # [cam, time, patch] in global feauture as required by MotionDecoder
            track_tokens = self.motion_decoder(
                feature, images=x, patch_start_idx=2, track_query_idx=query_idx
            )
            aggregated_track_tokens_list.append(track_tokens)
        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            track, track_conf = self.track_head(
                aggregated_track_tokens_list, images=x, patch_start_idx=1, frames_chunk_size=frames_chunk_size
            )
        return track, track_conf

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic) # c2w -> w2c
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            # del output.ray
            # del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_camera_estimation(
        self, H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        # Convert pose encoding to extrinsics and intrinsics
        c2w, ixt = pose_encoding_to_extri_intri(output.pose_enc, (H, W))
        output.extrinsics_token = affine_inverse(c2w)
        output.intrinsics_token = ixt

        return output
