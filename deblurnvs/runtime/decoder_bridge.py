from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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


def _strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def extract_decoder_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and 'decoder' in checkpoint and isinstance(checkpoint['decoder'], dict):
        state_dict = checkpoint['decoder']
    elif isinstance(checkpoint, dict) and 'mae_decoder' in checkpoint and isinstance(checkpoint['mae_decoder'], dict):
        state_dict = checkpoint['mae_decoder']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
        state_dict = checkpoint['ema']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f'Unsupported decoder checkpoint type: {type(checkpoint)!r}')

    state_dict = _strip_prefix_if_present(state_dict, 'module.')
    state_dict = _strip_prefix_if_present(state_dict, 'mae_decoder.')
    state_dict = _strip_prefix_if_present(state_dict, 'decoder.')
    return state_dict


def load_decoder_checkpoint(
    decoder: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
    map_location: str = 'cpu',
) -> tuple[list[str], list[str], dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    state_dict = extract_decoder_state_dict(checkpoint)
    missing, unexpected = decoder.load_state_dict(state_dict, strict=strict)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return list(missing), list(unexpected), metadata


def build_decoder_tokens_from_samples(
    rae: torch.nn.Module,
    clear: torch.Tensor,
    samples: torch.Tensor,
    total_view: int,
    cond_num: int,
    validation_mode: str = 'propagation_0',
) -> tuple[torch.Tensor, int, int, int, int]:
    if getattr(rae, 'mae_decoder', None) is None:
        raise ValueError('GLD stage1 is missing mae_decoder; cannot build decoder tokens.')
    if samples.ndim != 4:
        raise ValueError(f'Expected sampled latents [B*V, C, H, W], got {tuple(samples.shape)}')

    batch_size = int(clear.shape[0])
    _, _, _, image_h, image_w = clear.shape
    images_norm = (clear - rae.encoder_mean[None]) / rae.encoder_std[None]

    with torch.no_grad():
        ref_images_norm = images_norm[:, :cond_num].reshape(batch_size * cond_num, 3, image_h, image_w)
        _, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode='single', level=rae.level)

        _, feat_channels, feat_h, feat_w = samples.shape
        # Camera canonicalization is handled upstream in prepare_deblur_batch().
        # Decoder-side view order stays in the original sample order here.
        merged_shallow_flat = samples.reshape(batch_size * total_view, feat_channels, feat_h, feat_w)
        merged_cls = (
            ref_gt_cls.reshape(batch_size, cond_num, -1)[:, :1]
            .expand(-1, total_view, -1)
            .reshape(batch_size * total_view, -1)
        )

        current_level_neg = int(rae.level)
        if current_level_neg >= 0:
            current_level_neg -= 4

        if validation_mode not in {'propagation', 'propagation_0', 'propagation_gt'}:
            raise ValueError(f'Unsupported decoder validation_mode={validation_mode!r}')

        use_gt_for_upward = validation_mode == 'propagation_gt'
        use_zero_for_upward = validation_mode == 'propagation_0'

        if current_level_neg == -4 or (not use_gt_for_upward and not use_zero_for_upward):
            propagated_feats = rae.propagate_features(
                merged_shallow_flat,
                from_level=rae.level,
                total_view=total_view,
                cls_token=merged_cls,
            )
        else:
            gt_feats_neg = None
            if use_gt_for_upward:
                gt_feats_neg = _normalize_gt_feats_to_neg_levels(rae.encode(images_norm, mode='all'))

            downward_feats = rae.propagate_features(
                merged_shallow_flat,
                from_level=rae.level,
                total_view=total_view,
                cls_token=merged_cls,
            )
            if not downward_feats:
                raise RuntimeError('RAE propagation returned no features for decoder inference.')

            propagated_feats = []
            for level in (-4, -3, -2, -1):
                if level < current_level_neg:
                    if gt_feats_neg is not None and level in gt_feats_neg:
                        gt_feat = gt_feats_neg[level]
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
    mae_feats = []
    for patches, _ in propagated_feats:
        mae_feats.append(patches.to(dtype=decoder_dtype))
    z_cat = torch.cat(mae_feats, dim=-1)
    return z_cat.reshape(batch_size * total_view, z_cat.shape[2], z_cat.shape[3]), batch_size, total_view, image_h, image_w


def decode_decoder_token_chunk(
    rae: torch.nn.Module,
    z_chunk: torch.Tensor,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    logits = rae.mae_decoder(z_chunk, input_size=(image_h, image_w), drop_cls_token=False).logits
    x_rec = rae.mae_decoder.unpatchify(logits, (image_h, image_w))
    return x_rec * rae.encoder_std + rae.encoder_mean


def decode_samples_with_trainable_decoder(
    rae: torch.nn.Module,
    clear: torch.Tensor,
    samples: torch.Tensor,
    total_view: int,
    cond_num: int,
    validation_mode: str = 'propagation_0',
    view_chunk_size: int | None = None,
) -> torch.Tensor:
    z_cat, batch_size, total_view, image_h, image_w = build_decoder_tokens_from_samples(
        rae=rae,
        clear=clear,
        samples=samples,
        total_view=total_view,
        cond_num=cond_num,
        validation_mode=validation_mode,
    )

    if view_chunk_size is None or int(view_chunk_size) <= 0 or int(view_chunk_size) >= int(total_view):
        x_rec = decode_decoder_token_chunk(
            rae=rae,
            z_chunk=z_cat,
            image_h=image_h,
            image_w=image_w,
        )
    else:
        chunk_bv = max(1, int(batch_size) * int(view_chunk_size))
        recon_chunks = []
        for start in range(0, int(z_cat.shape[0]), chunk_bv):
            end = min(int(z_cat.shape[0]), start + chunk_bv)
            recon_chunks.append(
                decode_decoder_token_chunk(
                    rae=rae,
                    z_chunk=z_cat[start:end],
                    image_h=image_h,
                    image_w=image_w,
                )
            )
        x_rec = torch.cat(recon_chunks, dim=0)

    rgb = x_rec.reshape(batch_size, total_view, 3, image_h, image_w)
    return rgb
