from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

DEFAULT_UTILS_ROOT = Path(__file__).resolve().parents[2] / "utils"
DEFAULT_GLD_SRC = DEFAULT_UTILS_ROOT / 'gld' / 'src'
DEFAULT_DA3_SRC = DEFAULT_UTILS_ROOT / 'da3_runtime' / 'src'
if DEFAULT_GLD_SRC.exists():
    os.environ.setdefault('MVDIFF_GLD_SRC', str(DEFAULT_GLD_SRC))
if DEFAULT_DA3_SRC.exists():
    os.environ.setdefault('MVDIFF_DA3_SRC', str(DEFAULT_DA3_SRC))

from .gld_bridge_base import (
    GLD_SRC,
    build_camera_embedding,
    build_reference_condition,
    compute_time_dist_shift,
    create_transport,
    encode_gld_latents,
    instantiate_gld_stage1,
    instantiate_gld_stage2,
    load_gld_pretrained,
)

_GLD_TRANSPORT = importlib.import_module('stage2.transport.transport')
Sampler = _GLD_TRANSPORT.Sampler


def create_sampler(transport: Any) -> Any:
    return Sampler(transport)


def _as_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f'Unsupported extrinsic shape: {tuple(extrinsics.shape)}')
    out = torch.zeros(*extrinsics.shape[:-2], 4, 4, device=extrinsics.device, dtype=extrinsics.dtype)
    out[..., :3, :4] = extrinsics
    out[..., 3, 3] = 1.0
    return out


def convert_da3_camera_params_to_gld(camera_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert DA3 ray-pose outputs to the camera convention expected by GLD.

    DA3 ray-pose predictions are numerically aligned with OpenCV w2c on the resized
    training images. GLD camera embedding expects OpenCV c2w and later derives w2c
    viewmats internally for PRoPE. Intrinsics stay in pixel units here; ProPE will
    normalize them using prope_image_size.
    """

    raw_w2c = _as_homogeneous(camera_params['extrinsics'])
    c2w = torch.linalg.inv(raw_w2c)
    return {
        'intrinsics': camera_params['intrinsics'],
        'extrinsics': c2w,
        'raw_extrinsics_w2c': raw_w2c,
    }


def canonicalize_camera_params_to_reference_view(
    camera_params: dict[str, torch.Tensor],
    reference_view: str = 'last',
) -> dict[str, torch.Tensor]:
    """Normalize cameras to an explicit reference view while preserving view order.

    Important: this function only changes the camera coordinate frame used by
    the camera embedding / PRoPE path. It does not reorder image, latent, or
    decoder view tokens. Decoder-side view order is expected to remain in the
    original sequence order unless a caller explicitly permutes it elsewhere.
    """

    intrinsics = camera_params['intrinsics']
    raw_c2w = _as_homogeneous(camera_params['extrinsics'])
    reference_view = str(reference_view).lower()
    if reference_view == 'native':
        reference_view_index = -1
        canonical_c2w = raw_c2w.clone()
        canonical_w2c = torch.linalg.inv(canonical_c2w)
        scale = torch.ones(
            canonical_c2w.shape[0],
            1,
            device=canonical_c2w.device,
            dtype=canonical_c2w.dtype,
        )
    elif reference_view == 'last':
        reference_view_index = int(raw_c2w.shape[1] - 1)
        ref_inv = torch.linalg.inv(raw_c2w[:, reference_view_index])
        canonical_c2w = ref_inv.unsqueeze(1) @ raw_c2w

        translation = canonical_c2w[:, :, :3, 3]
        farthest = translation.abs().amax(dim=1).amax(dim=1, keepdim=True)
        scale = 1.0 / (farthest + 1.0e-8)

        canonical_c2w = canonical_c2w.clone()
        canonical_c2w[:, :, :3, 3] = canonical_c2w[:, :, :3, 3] * scale.unsqueeze(1)
        canonical_w2c = torch.linalg.inv(canonical_c2w)
    elif reference_view == 'first':
        reference_view_index = 0
        ref_inv = torch.linalg.inv(raw_c2w[:, reference_view_index])
        canonical_c2w = ref_inv.unsqueeze(1) @ raw_c2w

        translation = canonical_c2w[:, :, :3, 3]
        farthest = translation.abs().amax(dim=1).amax(dim=1, keepdim=True)
        scale = 1.0 / (farthest + 1.0e-8)

        canonical_c2w = canonical_c2w.clone()
        canonical_c2w[:, :, :3, 3] = canonical_c2w[:, :, :3, 3] * scale.unsqueeze(1)
        canonical_w2c = torch.linalg.inv(canonical_c2w)
    else:
        raise ValueError(
            f'Unsupported camera reference_view={reference_view!r}. Expected one of: native, first, last.'
        )

    out = dict(camera_params)
    out.update({
        'intrinsics': intrinsics,
        'extrinsics': canonical_c2w,
        'canonical_viewmats': canonical_w2c,
        'translation_scale': scale,
        'raw_extrinsics_c2w': raw_c2w,
        'reference_view_index': reference_view_index,
        'reference_view_mode': reference_view,
    })
    return out



@torch.no_grad()
def predict_camera_params_with_reference_view(
    extractor: torch.nn.Module,
    images_01: torch.Tensor,
    reference_view: str = 'native',
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run DA3 camera prediction with an explicit outer reference-view policy.

    DA3 does not expose a native "last view" reference policy. To guarantee that
    the final view acts as the internal reference, we reorder views so the final
    view becomes position 0, force the extractor to use its built-in "first"
    strategy for this prediction call, then restore the original order.
    """

    reference_view = str(reference_view).lower()
    if reference_view not in {'native', 'first', 'last'}:
        raise ValueError(
            f'Unsupported camera reference_view={reference_view!r}. Expected one of: native, first, last.'
        )

    num_views = int(images_01.shape[1])
    device = images_01.device
    perm = torch.arange(num_views, device=device, dtype=torch.long)
    if reference_view == 'last':
        perm = torch.tensor([num_views - 1, *range(num_views - 1)], device=device, dtype=torch.long)

    restore_perm = torch.empty_like(perm)
    restore_perm[perm] = torch.arange(num_views, device=device)

    prediction_images = images_01 if reference_view in {'native', 'first'} else images_01.index_select(1, perm)
    original_strategy = getattr(extractor, 'ref_view_strategy', 'first')
    forced_strategy = original_strategy if reference_view == 'native' else 'first'

    setattr(extractor, 'ref_view_strategy', forced_strategy)
    try:
        raw_camera_params = extractor.predict_camera_params(prediction_images)
    finally:
        setattr(extractor, 'ref_view_strategy', original_strategy)

    if reference_view == 'last':
        raw_camera_params = {
            'extrinsics': raw_camera_params['extrinsics'].index_select(1, restore_perm),
            'intrinsics': raw_camera_params['intrinsics'].index_select(1, restore_perm),
        }

    debug = {
        'camera_reference_view_mode': reference_view,
        'camera_prediction_perm': perm.detach().cpu(),
        'camera_restore_perm': restore_perm.detach().cpu(),
        'camera_original_ref_view_strategy': str(original_strategy),
        'camera_forced_ref_view_strategy': str(forced_strategy),
        'camera_internal_reference_view_index': int(perm[0].item()) if num_views > 0 else 0,
    }
    return raw_camera_params, debug


def build_reference_condition_from_source(
    source_latents: torch.Tensor,
    num_views: int,
    cond_num: int,
) -> torch.Tensor:
    if source_latents.ndim != 4:
        raise ValueError(f'Expected source_latents [B*V, C, H, W], got {tuple(source_latents.shape)}')
    if source_latents.shape[0] % int(num_views) != 0:
        raise ValueError(
            f'source_latents batch {source_latents.shape[0]} is not divisible by num_views={num_views}'
        )
    if not 1 <= int(cond_num) <= int(num_views):
        raise ValueError(f'Expected 1 <= cond_num <= num_views, got cond_num={cond_num}, num_views={num_views}')

    batch_size = source_latents.shape[0] // int(num_views)
    latents_5d = rearrange(source_latents, '(b v) c h w -> b v c h w', b=batch_size, v=int(num_views))
    if int(cond_num) == int(num_views):
        return rearrange(latents_5d, 'b v c h w -> (b v) c h w')

    ref_cond = torch.zeros_like(latents_5d)
    ref_cond[:, : int(cond_num)] = latents_5d[:, : int(cond_num)]
    return rearrange(ref_cond, 'b v c h w -> (b v) c h w')


def _apply_camera_conditioning_override(
    camera_embedding: torch.Tensor,
    prope_inputs: dict[str, torch.Tensor],
    mode: str = 'full',
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mode = str(mode).lower()
    if mode in {'full', 'default', 'enabled'}:
        return camera_embedding, prope_inputs

    updated_embedding = camera_embedding.clone()
    updated_prope_inputs = dict(prope_inputs)

    if mode == 'zero':
        updated_embedding.zero_()
    elif mode in {'identity', 'mask_only'}:
        if int(updated_embedding.shape[1]) > 1:
            updated_embedding[:, 1:] = 0
    else:
        raise ValueError(
            f'Unsupported camera_conditioning_mode={mode!r}. '
            'Expected one of: full, zero, identity, mask_only.'
        )

    viewmats = updated_prope_inputs.get('viewmats')
    if torch.is_tensor(viewmats):
        identity_viewmats = torch.eye(4, device=viewmats.device, dtype=viewmats.dtype).unsqueeze(0).unsqueeze(0)
        updated_prope_inputs['viewmats'] = identity_viewmats.expand_as(viewmats).clone()

    intrinsics = updated_prope_inputs.get('Ks')
    if torch.is_tensor(intrinsics):
        identity_intrinsics = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype).unsqueeze(0).unsqueeze(0)
        updated_prope_inputs['Ks'] = identity_intrinsics.expand_as(intrinsics).clone()

    return updated_embedding, updated_prope_inputs


def prepare_deblur_batch(
    extractor: torch.nn.Module,
    rae: torch.nn.Module,
    blur: torch.Tensor,
    clear: torch.Tensor,
    cond_num: int,
    camera_prediction_source: str = 'clear',
    camera_mode: str = 'plucker',
    camera_conditioning_mode: str = 'full',
    camera_reference_view: str = 'native',
    camera_images: torch.Tensor | None = None,
    camera_select_indices: torch.Tensor | None = None,
    clear_target_rae: torch.nn.Module | None = None,
    return_student_clear_latents: bool = False,
    enable_rae_grad: bool = False,
) -> dict[str, torch.Tensor]:
    num_views = int(blur.shape[1])

    # In the default NVS-style setup, diffusion only sees context blur through
    # x1_cond and predicts clean latents for all views. When cond_num == num_views,
    # this becomes context-only multiview deblur: every view is observed blur and
    # every view is supervised by the clean latent.
    student_clear_latents = None
    with torch.set_grad_enabled(bool(enable_rae_grad)):
        if clear_target_rae is None or bool(return_student_clear_latents):
            student_clear_latents = encode_gld_latents(rae, clear)
        if int(cond_num) == int(num_views):
            x1_cond = encode_gld_latents(rae, blur)
        else:
            x1_cond = build_reference_condition(rae, blur, cond_num=cond_num, layout='prefix_zero_pad')

    if clear_target_rae is None:
        if student_clear_latents is None:
            with torch.no_grad():
                student_clear_latents = encode_gld_latents(rae, clear)
        x1_all = student_clear_latents
        teacher_clear_latents = None
    else:
        with torch.no_grad():
            teacher_clear_latents = encode_gld_latents(clear_target_rae, clear)
        x1_all = teacher_clear_latents

    x0_init = torch.randn_like(x1_all)

    if camera_images is None:
        camera_source = str(camera_prediction_source).lower()
        if camera_source == 'clear':
            camera_images = clear
        elif camera_source == 'blur':
            camera_images = blur
        else:
            raise ValueError(f'Unsupported camera_prediction_source={camera_prediction_source!r}')

    with torch.no_grad():
        raw_camera_params, camera_debug = predict_camera_params_with_reference_view(
            extractor=extractor,
            images_01=camera_images,
            reference_view=camera_reference_view,
        )
    if camera_select_indices is not None:
        select_indices = camera_select_indices.to(device=raw_camera_params['extrinsics'].device, dtype=torch.long)
        raw_camera_params = {
            'extrinsics': raw_camera_params['extrinsics'].index_select(1, select_indices),
            'intrinsics': raw_camera_params['intrinsics'].index_select(1, select_indices),
        }
        camera_debug = dict(camera_debug)
        camera_debug['camera_selected_indices'] = select_indices.detach().cpu()
        if int(select_indices.numel()) != num_views:
            raise ValueError(
                f'camera_select_indices length {int(select_indices.numel())} does not match num_views={num_views}'
            )
    else:
        camera_debug = dict(camera_debug)
        camera_debug['camera_selected_indices'] = None

    camera_params = convert_da3_camera_params_to_gld(raw_camera_params)
    camera_params = canonicalize_camera_params_to_reference_view(
        camera_params,
        reference_view=camera_reference_view,
    )
    image_h, image_w = int(clear.shape[-2]), int(clear.shape[-1])
    camera_embedding, prope_inputs = build_camera_embedding(
        intrinsics=camera_params['intrinsics'],
        extrinsics=camera_params['extrinsics'],
        image_h=image_h,
        image_w=image_w,
        cond_num=int(cond_num),
        mode=camera_mode,
        reference_view=camera_reference_view,
    )
    prope_inputs['viewmats'] = camera_params['canonical_viewmats']
    camera_embedding, prope_inputs = _apply_camera_conditioning_override(
        camera_embedding=camera_embedding,
        prope_inputs=prope_inputs,
        mode=camera_conditioning_mode,
    )

    batch = {
        'x1_cond': x1_cond,
        'x1_all': x1_all,
        'x0_init': x0_init,
        'camera_embedding': camera_embedding,
        'pred_intrinsics': camera_params['intrinsics'],
        'pred_extrinsics': camera_params['extrinsics'],
        'pred_extrinsics_raw_c2w': camera_params['raw_extrinsics_c2w'],
        'pred_extrinsics_raw_w2c': camera_params['raw_extrinsics_w2c'],
        'pred_reference_view_index': camera_params['reference_view_index'],
        'pred_reference_view_mode': camera_params['reference_view_mode'],
        'pred_translation_scale': camera_params['translation_scale'],
        'pred_camera_reference_view_mode': camera_debug['camera_reference_view_mode'],
        'pred_camera_prediction_perm': camera_debug['camera_prediction_perm'],
        'pred_camera_restore_perm': camera_debug['camera_restore_perm'],
        'pred_camera_original_ref_view_strategy': camera_debug['camera_original_ref_view_strategy'],
        'pred_camera_forced_ref_view_strategy': camera_debug['camera_forced_ref_view_strategy'],
        'pred_camera_internal_reference_view_index': camera_debug['camera_internal_reference_view_index'],
        'pred_camera_selected_indices': camera_debug['camera_selected_indices'],
        'pred_camera_conditioning_mode': str(camera_conditioning_mode),
        'pred_decoder_view_order': 'native',
        'pred_decoder_reorder_applied': False,
    }
    if student_clear_latents is not None:
        batch['student_clear_latents'] = student_clear_latents
    if teacher_clear_latents is not None:
        batch['teacher_clear_latents'] = teacher_clear_latents
    batch.update(prope_inputs)
    return batch
