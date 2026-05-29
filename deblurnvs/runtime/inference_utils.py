from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from omegaconf import OmegaConf

from .da3_lora import load_lora_state_dict
from .gld_bridge import (
    _apply_camera_conditioning_override,
    build_camera_embedding,
    create_sampler,
    create_transport,
    instantiate_gld_stage2,
    load_gld_pretrained,
)


@dataclass(frozen=True)
class DecoderStaticContext:
    batch_size: int
    total_view: int
    image_h: int
    image_w: int
    current_level_neg: int
    merged_cls: torch.Tensor
    gt_feats_neg: dict[int, torch.Tensor] | None


def build_autocast_context(device: torch.device, mixed_precision: str):
    mixed_precision = str(mixed_precision).lower()
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def parse_range_spec(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        spec = value.strip()
        if "-" in spec:
            lo_str, hi_str = spec.split("-", maxsplit=1)
            lo, hi = int(lo_str), int(hi_str)
        else:
            lo = hi = int(spec)
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Invalid {name} specification: {value!r}")
        lo, hi = int(value[0]), int(value[1])
    else:
        lo = hi = int(value)
    if lo < 1 or hi < lo:
        raise ValueError(f"Invalid {name} specification: {value!r}")
    return lo, hi


def parse_num_views_spec(value: Any) -> tuple[int, int]:
    lo, hi = parse_range_spec(value, "num_views")
    if hi < 2:
        raise ValueError(f"num_views must be at least 2 for multiview inference, got {value!r}")
    return lo, hi


def create_sampler_from_cfg_section(transport: Any, section_cfg: Any | None) -> Any:
    sampler_params = {
        "sampling_method": "euler",
        "num_steps": 50,
        "atol": 1.0e-6,
        "rtol": 1.0e-3,
        "reverse": False,
    }
    if section_cfg is not None and section_cfg.get("sampler") is not None:
        for key in list(sampler_params.keys()):
            value = section_cfg.sampler.get(key)
            if value is not None:
                sampler_params[key] = value
    return create_sampler(transport).sample_ode(**sampler_params)


def build_sampler(transport: Any, section_cfg: Any | None, override_steps: int | None) -> Any:
    if override_steps is None:
        return create_sampler_from_cfg_section(transport, section_cfg)
    if section_cfg is None:
        merged_cfg = OmegaConf.create({"sampler": {"num_steps": int(override_steps)}})
    else:
        merged_cfg = OmegaConf.create(OmegaConf.to_container(section_cfg, resolve=True))
        if merged_cfg.get("sampler") is None:
            merged_cfg.sampler = {}
        merged_cfg.sampler["num_steps"] = int(override_steps)
    return create_sampler_from_cfg_section(transport, merged_cfg)


def load_checkpoint_step(checkpoint_path: str | Path) -> int | None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        step = checkpoint.get("global_step")
        if step is not None:
            return int(step)
    return None


def ensure_model_finite(model: torch.nn.Module, model_name: str) -> None:
    bad_names: list[str] = []
    for name, param in model.named_parameters():
        if not torch.is_tensor(param):
            continue
        if not torch.isfinite(param).all():
            bad_names.append(name)
            if len(bad_names) >= 8:
                break
    if bad_names:
        raise ValueError(f"{model_name} contains non-finite parameters after loading checkpoint: {bad_names}")


def _normalize_gt_feats_to_neg_levels(gt_feats_dict: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    keys = {int(k) for k in gt_feats_dict.keys()}
    if keys.issubset({0, 1, 2, 3}):
        return {int(k) - 4: v for k, v in gt_feats_dict.items()}
    if keys.issubset({-4, -3, -2, -1}):
        return {int(k): v for k, v in gt_feats_dict.items()}
    raise ValueError(
        f"Unexpected gt_feats_all keys={sorted(list(keys))}. Expected subset of {{0,1,2,3}} or {{-4,-3,-2,-1}}."
    )


def _apply_da3_norm(rae: torch.nn.Module, feat: torch.Tensor) -> torch.Tensor:
    embed_dim = int(getattr(rae.encoder.backbone.pretrained, "embed_dim", feat.shape[-1] // 2))
    local = feat[..., :embed_dim]
    current = feat[..., embed_dim:]
    norm_layer = rae.encoder.backbone.pretrained.norm
    return torch.cat([local, norm_layer(current)], dim=-1)


def maybe_create_context_refiner_from_prepared(
    cfg: Any,
    device: torch.device,
    prepared: dict[str, torch.Tensor],
) -> dict[str, Any] | None:
    refiner_cfg = cfg.get("refiner")
    if refiner_cfg is None or not bool(refiner_cfg.get("enabled", False)):
        return None

    checkpoint_path = str(refiner_cfg.get("checkpoint", "")).strip()
    if not checkpoint_path:
        raise ValueError("refiner.enabled=true but refiner.checkpoint is empty")

    x1_all = prepared.get("x1_all")
    if not torch.is_tensor(x1_all):
        raise ValueError("prepared.x1_all is required to build context refiner")

    target_channels = int(x1_all.shape[1])
    feat_h = int(x1_all.shape[2])
    feat_w = int(x1_all.shape[3])

    refiner_model = instantiate_gld_stage2(cfg.gld.stage2).to(device)
    missing, unexpected = load_gld_pretrained(
        refiner_model,
        checkpoint_path,
        strict=bool(refiner_cfg.get("strict", False)),
    )
    freeze_module(refiner_model)

    _, max_context_views = parse_num_views_spec(refiner_cfg.get("max_context_views", cfg.data.cond_num))
    transport_cfg = refiner_cfg.get("transport", cfg.gld.transport)
    refiner_transport = create_transport(
        path_type=transport_cfg.get("path_type", cfg.gld.transport.get("path_type", "Linear")),
        prediction=transport_cfg.get("prediction", cfg.gld.transport.get("prediction", "velocity")),
        loss_weight=transport_cfg.get("loss_weight", cfg.gld.transport.get("loss_weight")),
        time_dist_type=transport_cfg.get("time_dist_type", cfg.gld.transport.get("time_dist_type", "uniform")),
        time_dist_shift=float(
            ((target_channels * feat_h * feat_w * int(max_context_views)) / float(refiner_cfg.get("time_dist_shift_base", cfg.gld.get("time_dist_shift_base", 4096.0)))) ** 0.5
        ),
    )
    refiner_sampler = create_sampler_from_cfg_section(refiner_transport, refiner_cfg)
    print(
        f"[DeblurNVS] enabled context refiner: {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return {
        "model": refiner_model,
        "sampler": refiner_sampler,
        "camera_mode": str(refiner_cfg.get("camera_mode", cfg.gld.get("camera_mode", "plucker"))),
        "camera_conditioning_mode": str(refiner_cfg.get("camera_conditioning_mode", "identity")),
        "context_blend": float(refiner_cfg.get("context_blend", 1.0)),
    }


@torch.no_grad()
def _build_identity_refiner_prepared_batch(
    prepared: dict[str, torch.Tensor],
    batch_size: int,
    total_view: int,
    cond_num: int,
    image_h: int,
    image_w: int,
    camera_mode: str,
) -> dict[str, torch.Tensor]:
    x1_all_5d = rearrange(prepared["x1_all"], "(b v) c h w -> b v c h w", b=batch_size, v=total_view)
    x_blur_all = prepared.get("x_blur_all")
    if not torch.is_tensor(x_blur_all):
        raise ValueError("prepared.x_blur_all is required for context refiner")
    x_blur_5d = rearrange(x_blur_all, "(b v) c h w -> b v c h w", b=batch_size, v=total_view)

    context_clear = x1_all_5d[:, :cond_num]
    context_blur = x_blur_5d[:, :cond_num]
    flat_context_clear = rearrange(context_clear, "b v c h w -> (b v) c h w")
    flat_context_blur = rearrange(context_blur, "b v c h w -> (b v) c h w")

    eye3 = torch.eye(3, device=flat_context_clear.device, dtype=flat_context_clear.dtype).view(1, 1, 3, 3)
    eye4 = torch.eye(4, device=flat_context_clear.device, dtype=flat_context_clear.dtype).view(1, 1, 4, 4)
    intrinsics = eye3.expand(batch_size, cond_num, 3, 3).clone()
    extrinsics = eye4.expand(batch_size, cond_num, 4, 4).clone()

    camera_embedding, prope_inputs = build_camera_embedding(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_h=image_h,
        image_w=image_w,
        cond_num=int(cond_num),
        mode=camera_mode,
        reference_view="native",
    )
    camera_embedding, prope_inputs = _apply_camera_conditioning_override(
        camera_embedding=camera_embedding,
        prope_inputs=prope_inputs,
        mode="identity",
    )

    return {
        "x1_cond": flat_context_blur,
        "x1_all": flat_context_clear,
        "x0_init": torch.randn_like(flat_context_clear),
        "camera_embedding": camera_embedding,
        **prope_inputs,
    }


@torch.no_grad()
def maybe_apply_context_refiner(
    prepared: dict[str, torch.Tensor],
    clear: torch.Tensor,
    total_view: int,
    cond_num: int,
    refiner_bundle: dict[str, Any] | None,
    mixed_precision: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if refiner_bundle is None or int(cond_num) <= 0:
        return prepared

    batch_size = int(clear.shape[0])
    refiner_prepared = _build_identity_refiner_prepared_batch(
        prepared=prepared,
        batch_size=batch_size,
        total_view=total_view,
        cond_num=cond_num,
        image_h=int(clear.shape[-2]),
        image_w=int(clear.shape[-1]),
        camera_mode=str(refiner_bundle.get("camera_mode", "plucker")),
    )
    context_clear = clear[:, :cond_num]
    refined_context = sample_diffusion_latents(
        model=refiner_bundle["model"],
        sampler=refiner_bundle["sampler"],
        prepared=refiner_prepared,
        clear=context_clear,
        total_view=cond_num,
        cond_num=cond_num,
        mixed_precision=mixed_precision,
        device=device,
    ).detach()

    refined_context_5d = rearrange(refined_context, "(b v) c h w -> b v c h w", b=batch_size, v=cond_num)
    x1_cond_5d = rearrange(prepared["x1_cond"], "(b v) c h w -> b v c h w", b=batch_size, v=total_view)
    original_context = x1_cond_5d[:, :cond_num]

    context_blend = float(refiner_bundle.get("context_blend", 1.0))
    if context_blend < 1.0:
        refined_context_5d = refined_context_5d * context_blend + original_context * (1.0 - context_blend)

    updated_context = torch.zeros_like(x1_cond_5d)
    updated_context[:, :cond_num] = refined_context_5d

    updated = dict(prepared)
    updated["x1_cond"] = rearrange(updated_context, "b v c h w -> (b v) c h w")
    updated["refined_context_cond"] = rearrange(refined_context_5d, "b v c h w -> (b v) c h w")
    return updated


@torch.no_grad()
def build_decoder_static_context(
    rae: torch.nn.Module,
    clear: torch.Tensor,
    total_view: int,
    cond_num: int,
    validation_mode: str,
) -> DecoderStaticContext:
    batch_size = int(clear.shape[0])
    image_h, image_w = int(clear.shape[-2]), int(clear.shape[-1])
    images_norm = (clear - rae.encoder_mean[None]) / rae.encoder_std[None]

    ref_images_norm = images_norm[:, :cond_num].reshape(batch_size * cond_num, 3, image_h, image_w)
    _, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode="single", level=rae.level)
    merged_cls = (
        ref_gt_cls.reshape(batch_size, cond_num, -1)[:, :1]
        .expand(-1, total_view, -1)
        .reshape(batch_size * total_view, -1)
        .detach()
    )

    current_level_neg = int(rae.level)
    if current_level_neg >= 0:
        current_level_neg -= 4

    gt_feats_neg = None
    if validation_mode == "propagation_gt" and current_level_neg != -4:
        gt_feats_neg = _normalize_gt_feats_to_neg_levels(rae.encode(images_norm, mode="all"))

    return DecoderStaticContext(
        batch_size=batch_size,
        total_view=int(total_view),
        image_h=image_h,
        image_w=image_w,
        current_level_neg=int(current_level_neg),
        merged_cls=merged_cls,
        gt_feats_neg=gt_feats_neg,
    )


def build_decoder_tokens_from_samples(
    rae: torch.nn.Module,
    samples: torch.Tensor,
    decoder_static: DecoderStaticContext,
    validation_mode: str,
) -> tuple[torch.Tensor, int, int, int, int]:
    if getattr(rae, "mae_decoder", None) is None:
        raise ValueError("GLD stage1 is missing mae_decoder; cannot build decoder tokens.")
    if samples.ndim != 4:
        raise ValueError(f"Expected sampled latents [B*V, C, H, W], got {tuple(samples.shape)}")

    batch_size = int(decoder_static.batch_size)
    total_view = int(decoder_static.total_view)
    image_h = int(decoder_static.image_h)
    image_w = int(decoder_static.image_w)

    _, feat_channels, feat_h, feat_w = samples.shape
    merged_shallow_flat = samples.reshape(batch_size * total_view, feat_channels, feat_h, feat_w)
    merged_cls = decoder_static.merged_cls.to(device=samples.device, dtype=samples.dtype)

    current_level_neg = int(decoder_static.current_level_neg)
    if validation_mode not in {"propagation", "propagation_0", "propagation_gt"}:
        raise ValueError(f"Unsupported decoder validation_mode={validation_mode!r}")

    use_gt_for_upward = validation_mode == "propagation_gt"
    use_zero_for_upward = validation_mode == "propagation_0"
    if current_level_neg == -4 or (not use_gt_for_upward and not use_zero_for_upward):
        propagated_feats = rae.propagate_features(
            merged_shallow_flat,
            from_level=rae.level,
            total_view=total_view,
            cls_token=merged_cls,
        )
    else:
        downward_feats = rae.propagate_features(
            merged_shallow_flat,
            from_level=rae.level,
            total_view=total_view,
            cls_token=merged_cls,
        )
        if not downward_feats:
            raise RuntimeError("RAE propagation returned no features for decoder testing.")

        propagated_feats = []
        gt_feats_neg = decoder_static.gt_feats_neg
        for level in (-4, -3, -2, -1):
            if level < current_level_neg:
                if gt_feats_neg is not None and level in gt_feats_neg:
                    gt_feat = gt_feats_neg[level].to(device=samples.device)
                    gt_feat_5d = gt_feat.reshape(batch_size, total_view, *gt_feat.shape[1:])
                    cls_3d_raw = gt_feat_5d[:, :, 0, :]
                    patches_4d_raw = gt_feat_5d[:, :, 1:, :]
                    patches_4d_norm = _apply_da3_norm(rae, patches_4d_raw)
                    propagated_feats.append((patches_4d_norm, cls_3d_raw))
                else:
                    ref_p, ref_c = downward_feats[0]
                    zero_cls = torch.zeros_like(ref_c) if ref_c is not None else None
                    propagated_feats.append((torch.zeros_like(ref_p), zero_cls))
            else:
                idx = level - current_level_neg
                if 0 <= idx < len(downward_feats):
                    propagated_feats.append(downward_feats[idx])
                else:
                    ref_p, ref_c = downward_feats[0]
                    zero_cls = torch.zeros_like(ref_c) if ref_c is not None else None
                    propagated_feats.append((torch.zeros_like(ref_p), zero_cls))

        if len(propagated_feats) < 4:
            ref_p, ref_c = propagated_feats[0]
            zero_cls = torch.zeros_like(ref_c) if ref_c is not None else None
            while len(propagated_feats) < 4:
                propagated_feats = [(torch.zeros_like(ref_p), zero_cls)] + propagated_feats

    decoder_dtype = next(rae.mae_decoder.parameters()).dtype
    mae_feats = [patches.to(dtype=decoder_dtype) for patches, _ in propagated_feats]
    z_cat = torch.cat(mae_feats, dim=-1)
    return z_cat.reshape(batch_size * total_view, z_cat.shape[2], z_cat.shape[3]), batch_size, total_view, image_h, image_w


def decode_samples_with_frozen_decoder(
    rae: torch.nn.Module,
    samples: torch.Tensor,
    decoder_static: DecoderStaticContext,
    validation_mode: str,
    view_chunk_size: int | None = None,
) -> torch.Tensor:
    z_cat, batch_size, total_view, image_h, image_w = build_decoder_tokens_from_samples(
        rae=rae,
        samples=samples,
        decoder_static=decoder_static,
        validation_mode=validation_mode,
    )

    if view_chunk_size is None or int(view_chunk_size) <= 0 or int(view_chunk_size) >= int(total_view):
        logits = rae.mae_decoder(z_cat, input_size=(image_h, image_w), drop_cls_token=False).logits
        x_rec = rae.mae_decoder.unpatchify(logits, (image_h, image_w))
    else:
        chunk_bv = max(1, int(batch_size) * int(view_chunk_size))
        recon_chunks = []
        for start in range(0, int(z_cat.shape[0]), chunk_bv):
            end = min(int(z_cat.shape[0]), start + chunk_bv)
            logits_chunk = rae.mae_decoder(
                z_cat[start:end],
                input_size=(image_h, image_w),
                drop_cls_token=False,
            ).logits
            recon_chunks.append(rae.mae_decoder.unpatchify(logits_chunk, (image_h, image_w)))
        x_rec = torch.cat(recon_chunks, dim=0)

    rgb = x_rec.reshape(batch_size, total_view, 3, image_h, image_w)
    return rgb * rae.encoder_std + rae.encoder_mean


def apply_sample_residual(
    samples: torch.Tensor,
    sample_residual: torch.Tensor | None,
    batch_size: int | None = None,
    total_view: int | None = None,
    cond_num: int | None = None,
) -> torch.Tensor:
    if sample_residual is None:
        return samples

    residual = sample_residual.to(device=samples.device, dtype=samples.dtype)
    if residual.ndim != 4:
        raise ValueError(f"Expected sample_residual [N or 1, C, H, W], got {tuple(residual.shape)}")
    if residual.shape[1:] != samples.shape[1:]:
        raise ValueError(f"sample_residual shape mismatch: residual={tuple(residual.shape)} samples={tuple(samples.shape)}")
    if residual.shape[0] == 1 and samples.shape[0] != 1:
        if batch_size is not None and total_view is not None and cond_num is not None:
            residual = residual.expand(max(0, int(batch_size) * max(0, int(total_view) - int(cond_num))), -1, -1, -1)
        else:
            residual = residual.expand(samples.shape[0], -1, -1, -1)
    elif residual.shape[0] != samples.shape[0]:
        if batch_size is None or total_view is None or cond_num is None or residual.shape[0] != int(batch_size) * max(0, int(total_view) - int(cond_num)):
            raise ValueError(f"sample_residual batch mismatch: residual={tuple(residual.shape)} samples={tuple(samples.shape)}")

    if batch_size is None or total_view is None or cond_num is None:
        return samples + residual

    target_count = max(0, int(total_view) - int(cond_num))
    if target_count == 0:
        return samples
    samples_5d = rearrange(samples, "(b v) c h w -> b v c h w", b=int(batch_size), v=int(total_view)).clone()
    residual_5d = rearrange(residual, "(b v) c h w -> b v c h w", b=int(batch_size), v=target_count)
    samples_5d[:, int(cond_num):] = samples_5d[:, int(cond_num):] + residual_5d
    return rearrange(samples_5d, "b v c h w -> (b v) c h w")


def decode_scene_prediction_from_samples(
    rae: torch.nn.Module,
    sampled: torch.Tensor,
    decoder_static: DecoderStaticContext,
    validation_mode: str,
    view_chunk_size: int,
    mixed_precision: str,
    device: torch.device,
    sample_residual: torch.Tensor | None = None,
    cond_num: int | None = None,
) -> torch.Tensor:
    sampled = apply_sample_residual(
        sampled,
        sample_residual,
        batch_size=int(decoder_static.batch_size),
        total_view=int(decoder_static.total_view),
        cond_num=cond_num,
    )
    with build_autocast_context(device, mixed_precision):
        pred_rgb = decode_samples_with_frozen_decoder(
            rae=rae,
            samples=sampled,
            decoder_static=decoder_static,
            validation_mode=validation_mode,
            view_chunk_size=view_chunk_size,
        )
    return pred_rgb


def load_frozen_stage1_lora_if_available(rae: torch.nn.Module, cfg: Any) -> Path | None:
    lora_cfg = cfg.get("da3_lora")
    if lora_cfg is None or not bool(lora_cfg.get("enabled", False)):
        return None

    checkpoint_hint = str(lora_cfg.get("checkpoint", cfg.get("refiner", {}).get("checkpoint", ""))).strip()
    if not checkpoint_hint:
        raise ValueError("da3_lora.enabled=true but no da3_lora.checkpoint or refiner.checkpoint is provided.")

    checkpoint_path = Path(checkpoint_hint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage1 LoRA checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    lora_state = checkpoint.get("gld_rae_lora")
    if not lora_state:
        raise KeyError(f"Checkpoint has no gld_rae_lora state: {checkpoint_path}")

    missing, unexpected = load_lora_state_dict(
        rae.encoder.backbone.pretrained,
        lora_state,
        strict=False,
    )
    print(
        f"[DeblurNVS] stage1 LoRA loaded from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return checkpoint_path


@torch.no_grad()
def sample_diffusion_latents(
    model: torch.nn.Module,
    sampler: Any,
    prepared: dict[str, torch.Tensor],
    clear: torch.Tensor,
    total_view: int,
    cond_num: int,
    mixed_precision: str,
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(clear.shape[0])
    x1_cond_5d = rearrange(prepared["x1_cond"], "(b v) c h w -> b v c h w", b=batch_size, v=total_view)
    x0_init_5d = rearrange(prepared["x0_init"], "(b v) c h w -> b v c h w", b=batch_size, v=total_view)
    latent_dim = int(x0_init_5d.shape[2])
    sample_input_flat = rearrange(torch.cat([x1_cond_5d, x0_init_5d], dim=2), "b v c h w -> (b v) c h w")

    model_kwargs = {
        "camera_embedding": prepared["camera_embedding"],
        "total_view": total_view,
        "cond_num": cond_num,
        "is_concat_mode": True,
        "ref_cond": prepared["x1_cond"],
        "x1_global": prepared["x1_all"],
        "prope_image_size": tuple(int(v) for v in clear.shape[-2:]),
        "viewmats": prepared["viewmats"],
        "Ks": prepared["Ks"],
    }
    with build_autocast_context(device, mixed_precision):
        ode_sampler = getattr(sampler, "__self__", None)
        if (
            ode_sampler is not None
            and hasattr(ode_sampler, "t")
            and hasattr(ode_sampler, "drift")
            and str(getattr(ode_sampler, "sampler_type", "")).lower() in {"euler", "heun"}
        ):
            t_values = ode_sampler.t.to(sample_input_flat.device)
            state = sample_input_flat
            sampler_type = str(getattr(ode_sampler, "sampler_type", "")).lower()
            for t_curr, t_next in zip(t_values[:-1], t_values[1:]):
                t_batch = torch.full((state.shape[0],), float(t_curr), device=state.device, dtype=state.dtype)
                drift_curr = ode_sampler.drift(state, t_batch, model, **model_kwargs)
                dt = t_next - t_curr
                if sampler_type == "heun":
                    predictor = state + dt * drift_curr
                    t_next_batch = torch.full((state.shape[0],), float(t_next), device=state.device, dtype=state.dtype)
                    drift_next = ode_sampler.drift(predictor, t_next_batch, model, **model_kwargs)
                    state = state + dt * 0.5 * (drift_curr + drift_next)
                else:
                    state = state + dt * drift_curr
            sampled = state
        else:
            sampled = sampler(sample_input_flat, model, **model_kwargs)[-1]
    return sampled[:, latent_dim:]
