from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def _as_homogeneous(extrinsic: torch.Tensor) -> torch.Tensor:
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {tuple(extrinsic.shape)}")
    out = torch.zeros(*extrinsic.shape[:-2], 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    out[..., :3, :4] = extrinsic
    out[..., 3, 3] = 1.0
    return out


def batch_sample_rays(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_h: int,
    image_w: int,
    nframe: int,
    normalize_extrinsic: bool = True,
    normalize_t: bool = True,
    reference_view: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-pixel rays from OpenCV c2w cameras."""
    device = intrinsic.device
    extrinsic = _as_homogeneous(extrinsic)

    if normalize_extrinsic:
        extri_5d = rearrange(extrinsic, "(b v) r c -> b v r c", v=nframe)
        ref_inv = extri_5d[:, reference_view].inverse()
        ref_inv = ref_inv.repeat_interleave(nframe, dim=0)
        extrinsic = ref_inv @ extrinsic

    c2w = extrinsic[:, :3, :4]

    # Match GLD ray sampling so pretrained camera conditioning stays in-distribution.
    xs = torch.arange(image_w, device=device, dtype=intrinsic.dtype) - 0.5
    ys = torch.arange(image_h, device=device, dtype=intrinsic.dtype) + 0.5
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    pixels = rearrange(pixels, "w h c -> 1 (h w) c").expand(intrinsic.shape[0], -1, -1)

    directions = pixels @ intrinsic.inverse().transpose(-1, -2)
    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
    rays_o = c2w[:, None, :3, 3].expand_as(rays_d)

    if normalize_t:
        rays_o_base = rearrange(c2w[:, :3, 3], "(b v) c -> b v c", v=nframe)
        farthest = rays_o_base.abs().amax(dim=1).amax(dim=1, keepdim=True)
        scale = 1.0 / (farthest + 1e-8)
        rays_o_base = rays_o_base * scale
        rays_o = rearrange(rays_o_base, "b v c -> (b v) c")[:, None, :].expand_as(rays_d)

    return rays_o, rays_d


def embed_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    nframe: int,
    mode: str = "plucker",
) -> torch.Tensor:
    if mode not in {"plucker", "camray"}:
        raise ValueError(f"Unknown camera mode: {mode}")

    if mode == "camray":
        emb = rays_d
    else:
        emb = torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], dim=-1)

    return rearrange(emb, "(b v) n c -> b v n c", v=nframe)


def get_camera_embedding(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    batch_size: int,
    num_views: int,
    image_h: int,
    image_w: int,
    mode: str = "plucker",
    normalize_extrinsic: bool = True,
    normalize_t: bool | None = None,
    reference_view: int = -1,
) -> torch.Tensor:
    if normalize_t is None:
        normalize_t = mode == "plucker"
    rays_o, rays_d = batch_sample_rays(
        intrinsic,
        extrinsic,
        image_h=image_h,
        image_w=image_w,
        nframe=num_views,
        normalize_extrinsic=normalize_extrinsic,
        normalize_t=bool(normalize_t),
        reference_view=int(reference_view),
    )
    camera_embedding = embed_rays(rays_o, rays_d, nframe=num_views, mode=mode)
    camera_embedding = rearrange(
        camera_embedding,
        "b v (h w) c -> b v c h w",
        b=batch_size,
        v=num_views,
        h=image_h,
        w=image_w,
    )
    return camera_embedding
