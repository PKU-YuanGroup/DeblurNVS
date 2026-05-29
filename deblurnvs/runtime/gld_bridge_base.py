from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from omegaconf import OmegaConf

from .camera import get_camera_embedding

REPO_ROOT = Path(__file__).resolve().parents[2]
GLD_SRC = Path(os.environ.get("MVDIFF_GLD_SRC", str(REPO_ROOT / "utils" / "gld" / "src")))
from stage1.rae_da3 import RAE_DA3
from stage2.models.DDT import DiTwDDTHead
from stage2.transport.transport import ModelType, PathType, Transport, WeightType


def _cfg_params(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "get"):
        params = cfg.get("params", {})
        return OmegaConf.to_container(params, resolve=True) if params is not None else {}
    return dict(getattr(cfg, "params", {}) or {})


def instantiate_gld_stage1(stage1_cfg: Any) -> torch.nn.Module:
    return RAE_DA3(**_cfg_params(stage1_cfg))


def instantiate_gld_stage2(stage2_cfg: Any) -> torch.nn.Module:
    return DiTwDDTHead(**_cfg_params(stage2_cfg))


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
    time_dist_type: str = "uniform",
    time_dist_shift: float = 1.0,
) -> Transport:
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_enum = path_choice[str(path_type)]

    if path_enum == PathType.VP:
        train_eps = 1.0e-5 if train_eps is None else train_eps
        sample_eps = 1.0e-3 if sample_eps is None else sample_eps
    elif path_enum in {PathType.GVP, PathType.LINEAR} and model_type != ModelType.VELOCITY:
        train_eps = 1.0e-3 if train_eps is None else train_eps
        sample_eps = 1.0e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0.0 if train_eps is None else train_eps
        sample_eps = 0.0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=path_enum,
        loss_type=loss_type,
        time_dist_type=time_dist_type,
        time_dist_shift=float(time_dist_shift),
        train_eps=float(train_eps),
        sample_eps=float(sample_eps),
    )


def load_gld_pretrained(model: torch.nn.Module, checkpoint_path: str, strict: bool = False) -> tuple[list[str], list[str]]:
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(checkpoint_path, mmap=True, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("y_embedder.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def load_normalization_stats(stat_path: str) -> dict[str, torch.Tensor]:
    stats = torch.load(stat_path, map_location="cpu")
    return {
        "mean": stats["mean"].float().reshape(-1),
        "var": stats["var"].float().reshape(-1),
    }


def _normalize_images_for_rae(rae: torch.nn.Module, images_01: torch.Tensor) -> torch.Tensor:
    return (images_01 - rae.encoder_mean[None]) / rae.encoder_std[None]


def encode_gld_latents(
    rae: torch.nn.Module,
    images_01: torch.Tensor,
    level: int | None = None,
    stats: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    images_norm = _normalize_images_for_rae(rae, images_01)

    original_do_norm = getattr(rae, "do_normalization", False)
    original_mean = getattr(rae, "latent_mean", None)
    original_var = getattr(rae, "latent_var", None)
    try:
        if stats is not None:
            rae.latent_mean = stats["mean"].to(device=images_01.device)
            rae.latent_var = stats["var"].to(device=images_01.device)
            rae.do_normalization = True
        return rae.encode(images_norm, mode="single", level=level)
    finally:
        rae.do_normalization = original_do_norm
        rae.latent_mean = original_mean
        rae.latent_var = original_var


def build_reference_condition(
    rae: torch.nn.Module,
    images_01: torch.Tensor,
    cond_num: int,
    layout: str = "prefix_zero_pad",
) -> torch.Tensor:
    layout = str(layout).lower()
    if layout == "all_views":
        return encode_gld_latents(rae, images_01)
    if layout != "prefix_zero_pad":
        raise ValueError(f"Unsupported conditioning_layout={layout!r}")

    batch_size, num_views = int(images_01.shape[0]), int(images_01.shape[1])
    if not 1 <= int(cond_num) < num_views:
        raise ValueError(f"Expected 1 <= cond_num < num_views, got cond_num={cond_num}, num_views={num_views}")

    ref_latents = encode_gld_latents(rae, images_01[:, :cond_num])
    channels, feat_h, feat_w = int(ref_latents.shape[1]), int(ref_latents.shape[2]), int(ref_latents.shape[3])
    x1_cond = torch.zeros(
        batch_size,
        num_views,
        channels,
        feat_h,
        feat_w,
        device=ref_latents.device,
        dtype=ref_latents.dtype,
    )
    x1_cond[:, :cond_num] = rearrange(ref_latents, "(b v) c h w -> b v c h w", b=batch_size, v=cond_num)
    return rearrange(x1_cond, "b v c h w -> (b v) c h w")


def _as_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {tuple(extrinsics.shape)}")
    out = torch.zeros(*extrinsics.shape[:-2], 4, 4, device=extrinsics.device, dtype=extrinsics.dtype)
    out[..., :3, :4] = extrinsics
    out[..., 3, 3] = 1.0
    return out


def build_camera_embedding(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
    cond_num: int,
    mode: str = "plucker",
    reference_view: str = "last",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, num_views = intrinsics.shape[:2]
    reference_view = str(reference_view).lower()
    if reference_view == "last":
        reference_view_index = int(num_views - 1)
        normalize_extrinsic = True
        normalize_t = mode == "plucker"
    elif reference_view == "first":
        reference_view_index = 0
        normalize_extrinsic = True
        normalize_t = mode == "plucker"
    elif reference_view == "native":
        reference_view_index = -1
        normalize_extrinsic = False
        normalize_t = False
    else:
        raise ValueError(
            f"Unsupported camera reference_view={reference_view!r}. "
            "Expected one of: native, first, last."
        )

    camera_embedding = get_camera_embedding(
        intrinsic=rearrange(intrinsics, "b v i j -> (b v) i j"),
        extrinsic=rearrange(extrinsics, "b v i j -> (b v) i j"),
        batch_size=batch_size,
        num_views=num_views,
        image_h=image_h,
        image_w=image_w,
        mode=mode,
        normalize_extrinsic=normalize_extrinsic,
        normalize_t=normalize_t,
        reference_view=reference_view_index,
    )
    camera_embedding = rearrange(camera_embedding, "b v c h w -> (b v) c h w")

    cond_mask = torch.ones(
        batch_size,
        num_views,
        1,
        image_h,
        image_w,
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    cond_mask[:, :cond_num] = 0
    cond_mask = rearrange(cond_mask, "b v c h w -> (b v) c h w")
    camera_embedding = torch.cat([cond_mask, camera_embedding], dim=1)

    c2w = _as_homogeneous(extrinsics)
    if reference_view != "native":
        ref_inv = torch.linalg.inv(c2w[:, reference_view_index])
        c2w = ref_inv.unsqueeze(1) @ c2w
        translation = c2w[:, :, :3, 3]
        farthest = translation.abs().amax(dim=1).amax(dim=1, keepdim=True)
        scale = 1.0 / (farthest + 1e-8)
        c2w = c2w.clone()
        c2w[:, :, :3, 3] = c2w[:, :, :3, 3] * scale.unsqueeze(1)
    viewmats = torch.linalg.inv(c2w)

    return camera_embedding, {
        "viewmats": viewmats,
        "Ks": intrinsics,
    }


def compute_time_dist_shift(channels: int, feat_h: int, feat_w: int, total_view: int, shift_base: float = 4096.0) -> float:
    shift_dim = int(channels) * int(feat_h) * int(feat_w) * int(total_view)
    return float(math.sqrt(max(float(shift_dim), 1.0) / float(shift_base)))


def prepare_gld_teacher_batch(
    extractor: torch.nn.Module,
    rae: torch.nn.Module,
    blur: torch.Tensor,
    clear: torch.Tensor,
    gld_cfg: Any,
    source_condition_stats: dict[str, torch.Tensor] | None = None,
    source_init_stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    cond_num = int(gld_cfg.cond_num)
    x1_all = encode_gld_latents(rae, clear)
    x1_cond = build_reference_condition(
        rae,
        blur,
        cond_num=cond_num,
        layout=gld_cfg.get("conditioning_layout", "prefix_zero_pad"),
    )

    camera_source = str(gld_cfg.get("camera_prediction_source", "blur")).lower()
    if camera_source == "blur":
        camera_images = blur
    elif camera_source == "clear":
        camera_images = clear
    else:
        raise ValueError(f"Unsupported camera_prediction_source={camera_source!r}")

    camera_params = extractor.predict_camera_params(camera_images)
    image_h, image_w = int(blur.shape[-2]), int(blur.shape[-1])
    camera_embedding, prope_inputs = build_camera_embedding(
        intrinsics=camera_params["intrinsics"],
        extrinsics=camera_params["extrinsics"],
        image_h=image_h,
        image_w=image_w,
        cond_num=cond_num,
        mode=gld_cfg.get("camera_mode", "plucker"),
        reference_view=gld_cfg.get("camera_reference_view", "last"),
    )

    batch = {
        "x1_cond": x1_cond,
        "x1_all": x1_all,
        "camera_embedding": camera_embedding,
        "pred_intrinsics": camera_params["intrinsics"],
        "pred_extrinsics": camera_params["extrinsics"],
    }
    batch.update(prope_inputs)

    source_condition_level = gld_cfg.get("source_condition_level")
    if source_condition_level is not None:
        if source_condition_stats is None:
            raise ValueError("source_condition_level is set but source_condition_stats is missing")
        batch["source_condition"] = encode_gld_latents(
            rae,
            blur,
            level=int(source_condition_level),
            stats=source_condition_stats,
        )

    source_init_level = gld_cfg.get("source_init_level")
    if source_init_level is not None:
        if source_init_stats is None:
            raise ValueError("source_init_level is set but source_init_stats is missing")
        x0_init = encode_gld_latents(
            rae,
            blur,
            level=int(source_init_level),
            stats=source_init_stats,
        )
        noise_tau = float(gld_cfg.get("source_init_noise_tau", 0.0))
        if noise_tau > 0:
            noise_std = float(torch.abs(torch.randn(1, device=x0_init.device) * noise_tau).item())
        else:
            noise_std = 0.0
        if noise_std > 0:
            x0_init = (x0_init + torch.randn_like(x0_init) * noise_std) / math.sqrt(1.0 + noise_std ** 2)
        batch["x0_init"] = x0_init

    return batch
