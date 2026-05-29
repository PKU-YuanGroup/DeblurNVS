from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module):
    """Minimal local DA3 loader for inference-only demo usage."""

    def __init__(self, model_name: str = "da3-base", model_config: dict | None = None):
        super().__init__()
        self.model_name = str(model_name)
        if model_config is None:
            if self.model_name not in MODEL_REGISTRY:
                raise KeyError(f"Unknown DA3 preset: {self.model_name}")
            config = load_config(MODEL_REGISTRY[self.model_name])
        else:
            config = OmegaConf.create(model_config)
        self.config = config
        self.model = create_object(config)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, local_files_only: bool = False, **_: object) -> "DepthAnything3":
        path = Path(model_id).expanduser()
        if path.is_dir():
            config_path = path / CONFIG_NAME
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing DA3 config file: {config_path}")
            payload = json.loads(config_path.read_text())
            model = cls(
                model_name=str(payload.get("model_name", "da3-base")),
                model_config=payload.get("config"),
            )
            weights_path = path / SAFETENSORS_NAME
            if weights_path.is_file():
                state_dict = load_file(str(weights_path))
            else:
                bin_path = path / "pytorch_model.bin"
                if not bin_path.is_file():
                    raise FileNotFoundError(f"Missing DA3 weights under {path}")
                state_dict = torch.load(str(bin_path), map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f"[DeblurNVS] loaded DA3 weights from {path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            return model

        if local_files_only:
            raise FileNotFoundError(f"DA3 local checkpoint directory not found: {path}")
        return cls(model_name=str(model_id))

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        export_feat_layers = [] if export_feat_layers is None else list(export_feat_layers)
        autocast_dtype = torch.bfloat16 if image.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=image.device.type, dtype=autocast_dtype, enabled=image.device.type == "cuda"):
            return self.model(
                image,
                extrinsics,
                intrinsics,
                export_feat_layers,
                infer_gs,
                use_ray_pose,
                ref_view_strategy,
            )
