from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from src.depth_anything_3.api import DepthAnything3
from src.depth_anything_3.utils.geometry import affine_inverse
from src.depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


class FrozenDA3FeatureExtractor(nn.Module):
    """Minimal frozen DA3 wrapper used by the open-source inference demo."""

    def __init__(
        self,
        model_id: str,
        export_layers: Sequence[int] | None = None,
        local_files_only: bool = False,
        ref_view_strategy: str = "first",
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.ref_view_strategy = str(ref_view_strategy)
        self.patch_size = int(patch_size)

        self.model = DepthAnything3.from_pretrained(model_id, local_files_only=local_files_only)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.da3 = self.model.model.da3 if hasattr(self.model.model, "da3") else self.model.model
        self.backbone = self.da3.backbone

        self.head_layers = list(getattr(self.backbone, "out_layers", []))
        if len(self.head_layers) != 4:
            raise ValueError(f"Expected 4 DPT input layers from backbone.out_layers, got {self.head_layers}")

        if export_layers is None:
            self.export_layers = list(self.head_layers)
        else:
            self.export_layers = [int(layer) for layer in export_layers]
            if self.export_layers != self.head_layers:
                raise ValueError(
                    f"Configured export_layers={self.export_layers}, but DPT head actually consumes backbone.out_layers={self.head_layers}."
                )

        self.level_dims = [int(project.in_channels) for project in self.da3.head.projects]
        if len(self.level_dims) != 4:
            raise ValueError(f"Expected 4 per-level channel dimensions, got {self.level_dims}")
        self.total_feature_dim = int(sum(self.level_dims))

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def normalize_images(self, images_01: torch.Tensor) -> torch.Tensor:
        return (images_01 - self.imagenet_mean) / self.imagenet_std

    def pack_levels(self, level_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(level_features) != len(self.level_dims):
            raise ValueError(f"Expected {len(self.level_dims)} levels, got {len(level_features)}")
        return torch.cat(list(level_features), dim=2)

    def unpack_levels(self, packed_features: torch.Tensor) -> list[torch.Tensor]:
        if packed_features.ndim != 5:
            raise ValueError(f"Expected packed features [B, V, C, H, W], got {tuple(packed_features.shape)}")
        if packed_features.shape[2] != self.total_feature_dim:
            raise ValueError(
                f"Packed feature channels {packed_features.shape[2]} do not match expected total {self.total_feature_dim}"
            )
        return list(torch.split(packed_features, self.level_dims, dim=2))

    def _check_image_size(self, images_01: torch.Tensor) -> None:
        if images_01.ndim != 5:
            raise ValueError(f"Expected images with shape [B, V, 3, H, W], got {tuple(images_01.shape)}")
        if images_01.shape[-2] % self.patch_size != 0 or images_01.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"Input image size {tuple(images_01.shape[-2:])} must be divisible by patch_size={self.patch_size}"
            )

    @torch.no_grad()
    def extract_level_features(self, images_01: torch.Tensor) -> list[torch.Tensor]:
        self._check_image_size(images_01)
        images_norm = self.normalize_images(images_01)
        head_inputs, _ = self.backbone(
            images_norm,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )

        feat_h = images_01.shape[-2] // self.patch_size
        feat_w = images_01.shape[-1] // self.patch_size
        if len(head_inputs) != len(self.export_layers):
            raise RuntimeError(
                f"Backbone returned {len(head_inputs)} DPT input features, expected {len(self.export_layers)}"
            )

        features: list[torch.Tensor] = []
        for level_idx, feature_tuple in enumerate(head_inputs):
            feature = feature_tuple[0]
            if feature.ndim != 4:
                raise ValueError(f"Expected head feature shape [B, V, N, C], got {tuple(feature.shape)}")
            if feature.shape[2] != feat_h * feat_w:
                raise ValueError(
                    f"Feature token count {feature.shape[2]} does not match expected grid {feat_h}x{feat_w}"
                )
            feature = feature.reshape(feature.shape[0], feature.shape[1], feat_h, feat_w, feature.shape[-1])
            feature = feature.permute(0, 1, 4, 2, 3).contiguous().float()
            if feature.shape[2] != self.level_dims[level_idx]:
                raise ValueError(
                    f"Level {level_idx} channels {feature.shape[2]} do not match expected {self.level_dims[level_idx]}"
                )
            features.append(feature)
        return features

    def _levels_to_head_tokens(self, level_features: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        feats: list[tuple[torch.Tensor, torch.Tensor]] = []
        for level in level_features:
            tokens = rearrange(level, "b v c h w -> b v (h w) c")
            dummy_camera_token = torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                tokens.shape[-1],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            feats.append((tokens, dummy_camera_token))
        return feats

    def _output_get(self, output: Any, key: str) -> Any:
        if isinstance(output, dict):
            return output.get(key)
        return getattr(output, key, None)

    def _process_ray_pose_estimation_safe(self, output: Any, height: int, width: int) -> Any:
        if "ray" not in output or "ray_conf" not in output:
            return output

        try:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
                training=True,
            )
        except ValueError as exc:
            print(
                f"[DeblurNVS] get_extrinsic_from_camray failed: {exc}. "
                "Using identity extrinsics and default intrinsics."
            )
            batch_size, num_views = output.ray.shape[0], output.ray.shape[1]
            device = output.ray.device
            dtype = output.ray.dtype
            pred_extrinsic = torch.eye(4, device=device, dtype=dtype)[None, None].repeat(batch_size, num_views, 1, 1)
            pred_focal_lengths = torch.ones(batch_size, num_views, 2, device=device, dtype=dtype)
            pred_principal_points = torch.full((batch_size, num_views, 2), 0.5, device=device, dtype=dtype)

        pred_extrinsic = affine_inverse(pred_extrinsic)
        pred_extrinsic = pred_extrinsic[:, :, :3, :]
        pred_intrinsic = torch.eye(3, 3, device=pred_extrinsic.device, dtype=pred_extrinsic.dtype)[None, None]
        pred_intrinsic = pred_intrinsic.repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone()
        pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
        del output.ray
        del output.ray_conf
        output.extrinsics = pred_extrinsic
        output.intrinsics = pred_intrinsic
        return output

    def _run_head_from_feature_tuples(
        self,
        feats: Sequence[tuple[torch.Tensor, torch.Tensor | None]],
        images_01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._check_image_size(images_01)
        _, _, _, height, width = images_01.shape

        head_dtype = next(self.da3.head.parameters()).dtype
        casted_feats = []
        for patches, cls_token in feats:
            patches = patches.to(dtype=head_dtype)
            if cls_token is not None:
                cls_token = cls_token.to(dtype=head_dtype)
            casted_feats.append((patches, cls_token))

        with torch.autocast(device_type=images_01.device.type, enabled=False):
            output = self.da3.head(casted_feats, height, width, patch_start_idx=0)
            output = self._process_ray_pose_estimation_safe(output, height, width)

        return {
            "depth": self._output_get(output, "depth"),
            "depth_conf": self._output_get(output, "depth_conf"),
            "sky": self._output_get(output, "sky"),
            "ray": self._output_get(output, "ray"),
            "ray_conf": self._output_get(output, "ray_conf"),
            "extrinsics": self._output_get(output, "extrinsics"),
            "intrinsics": self._output_get(output, "intrinsics"),
        }

    @torch.no_grad()
    def predict_outputs_from_level_features(
        self,
        level_features: Sequence[torch.Tensor],
        images_01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._run_head_from_feature_tuples(self._levels_to_head_tokens(level_features), images_01)

    @torch.no_grad()
    def predict_outputs_from_decoder_features(
        self,
        decoder_features: Sequence[tuple[torch.Tensor, torch.Tensor | None]],
        images_01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._run_head_from_feature_tuples(decoder_features, images_01)

    @torch.no_grad()
    def predict_camera_params_from_level_features(
        self,
        level_features: Sequence[torch.Tensor],
        images_01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.predict_outputs_from_level_features(level_features, images_01)
        return {
            "extrinsics": output["extrinsics"],
            "intrinsics": output["intrinsics"],
        }

    @torch.no_grad()
    def predict_camera_params_from_packed_features(
        self,
        packed_features: torch.Tensor,
        images_01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.predict_camera_params_from_level_features(self.unpack_levels(packed_features), images_01)

    @torch.no_grad()
    def predict_camera_params(self, images_01: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict_camera_params_from_level_features(self.extract_level_features(images_01), images_01)

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        return self.pack_levels(self.extract_level_features(images_01))
