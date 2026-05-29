from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrajectorySpec:
    mode: str
    num_frames: int
    anchor_view: str
    theta: float
    phi: float
    radius_scale: float
    shift_x: float
    shift_y: float
    traj_txt: str | None = None


def choose_anchor_index(num_views: int, anchor_view: str) -> int:
    anchor_view = str(anchor_view).lower()
    if anchor_view == "first":
        return 0
    if anchor_view == "center":
        return num_views // 2
    if anchor_view == "last":
        return max(0, num_views - 1)
    raise ValueError(f"Unsupported anchor_view={anchor_view!r}")


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1.0 - float(t)) + b * float(t)


def _orthonormalize_rotation(rotation: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rotation)
    ortho = u @ vh
    if float(torch.linalg.det(ortho).item()) < 0.0:
        u = u.clone()
        u[:, -1] *= -1.0
        ortho = u @ vh
    return ortho


def estimate_focus_point(c2ws: torch.Tensor) -> torch.Tensor:
    directions = c2ws[:, :3, 2:3]
    origins = c2ws[:, :3, 3:4]
    eye = torch.eye(3, device=c2ws.device, dtype=c2ws.dtype).unsqueeze(0)
    m = eye - directions * directions.transpose(1, 2)
    mt_m = m.transpose(1, 2) @ m
    lhs = mt_m.mean(0)
    rhs = (mt_m @ origins).mean(0)[:, 0]
    focus = torch.linalg.pinv(lhs) @ rhs
    if not torch.isfinite(focus).all():
        focus = origins[:, :, 0].mean(0)
    return focus


def _rotation_x(theta_deg: torch.Tensor) -> torch.Tensor:
    theta = torch.deg2rad(theta_deg)
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    row0 = torch.stack([one, zero, zero], dim=-1)
    row1 = torch.stack([zero, cos_t, -sin_t], dim=-1)
    row2 = torch.stack([zero, sin_t, cos_t], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rotation_y(phi_deg: torch.Tensor) -> torch.Tensor:
    phi = torch.deg2rad(phi_deg)
    one = torch.ones_like(phi)
    zero = torch.zeros_like(phi)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    row0 = torch.stack([cos_p, zero, sin_p], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-sin_p, zero, cos_p], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _interpolate_sequence(values: list[float], num_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(values) == 1:
        return torch.full((int(num_frames),), float(values[0]), device=device, dtype=dtype)
    positions = torch.linspace(0.0, 1.0, steps=len(values), device=device, dtype=dtype)
    query = torch.linspace(0.0, 1.0, steps=int(num_frames), device=device, dtype=dtype)
    source = torch.tensor([float(v) for v in values], device=device, dtype=dtype)
    right_idx = torch.searchsorted(positions, query, right=True).clamp(max=len(values) - 1)
    left_idx = (right_idx - 1).clamp(min=0)
    left_pos = positions[left_idx]
    right_pos = positions[right_idx]
    denom = (right_pos - left_pos).clamp_min(1.0e-8)
    weight = (query - left_pos) / denom
    return source[left_idx] * (1.0 - weight) + source[right_idx] * weight


def load_traj_txt(path: str | Path, num_frames: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lines = Path(path).expanduser().read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Trajectory file must contain at least 3 lines: {path}")
    phi_values = [float(v) for v in lines[0].split()]
    theta_values = [float(v) for v in lines[1].split()]
    radius_values = [float(v) for v in lines[2].split()]
    return (
        _interpolate_sequence(theta_values, num_frames=num_frames, device=device, dtype=dtype),
        _interpolate_sequence(phi_values, num_frames=num_frames, device=device, dtype=dtype),
        _interpolate_sequence(radius_values, num_frames=num_frames, device=device, dtype=dtype),
    )


def build_interpolated_camera_trajectory(
    context_raw_c2w: torch.Tensor,
    context_intrinsics: torch.Tensor,
    spec: TrajectorySpec,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    num_context = int(context_raw_c2w.shape[0])
    if num_context == 1:
        target_c2w = context_raw_c2w[:1].expand(int(spec.num_frames), -1, -1).clone()
        target_intrinsics = context_intrinsics[:1].expand(int(spec.num_frames), -1, -1).clone()
        trajectory_info = {
            "mode": "interp",
            "source_views": num_context,
            "anchor_view": spec.anchor_view,
            "anchor_index": 0,
            "sample_positions": [0.0 for _ in range(int(spec.num_frames))],
        }
        return target_c2w, target_intrinsics, trajectory_info

    query_positions = torch.linspace(
        0.0,
        float(num_context - 1),
        steps=int(spec.num_frames) + 2,
        device=context_raw_c2w.device,
        dtype=context_raw_c2w.dtype,
    )[1:-1]
    target_c2ws = []
    target_intrinsics = []
    for pos in query_positions:
        left_index = int(torch.floor(pos).item())
        right_index = min(left_index + 1, num_context - 1)
        blend = float((pos - left_index).item())

        left_pose = context_raw_c2w[left_index]
        right_pose = context_raw_c2w[right_index]
        left_intr = context_intrinsics[left_index]
        right_intr = context_intrinsics[right_index]

        blended_rotation = _orthonormalize_rotation(_lerp(left_pose[:3, :3], right_pose[:3, :3], blend))
        blended_translation = _lerp(left_pose[:3, 3], right_pose[:3, 3], blend)
        blended_intrinsics = _lerp(left_intr, right_intr, blend)

        pose = torch.eye(4, device=context_raw_c2w.device, dtype=context_raw_c2w.dtype)
        pose[:3, :3] = blended_rotation
        pose[:3, 3] = blended_translation
        target_c2ws.append(pose)
        target_intrinsics.append(blended_intrinsics)

    target_c2w = torch.stack(target_c2ws, dim=0)
    target_intrinsics_tensor = torch.stack(target_intrinsics, dim=0)
    trajectory_info = {
        "mode": "interp",
        "source_views": num_context,
        "anchor_view": spec.anchor_view,
        "anchor_index": choose_anchor_index(num_context, spec.anchor_view),
        "sample_positions": [float(v) for v in query_positions.tolist()],
    }
    return target_c2w, target_intrinsics_tensor, trajectory_info


def build_target_camera_trajectory(
    context_raw_c2w: torch.Tensor,
    context_intrinsics: torch.Tensor,
    spec: TrajectorySpec,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if context_raw_c2w.ndim != 3 or context_raw_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected context_raw_c2w [V,4,4], got {tuple(context_raw_c2w.shape)}")
    if context_intrinsics.ndim != 3 or context_intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected context_intrinsics [V,3,3], got {tuple(context_intrinsics.shape)}")

    if spec.mode == "interp":
        return build_interpolated_camera_trajectory(
            context_raw_c2w=context_raw_c2w,
            context_intrinsics=context_intrinsics,
            spec=spec,
        )

    device = context_raw_c2w.device
    dtype = context_raw_c2w.dtype
    anchor_index = choose_anchor_index(int(context_raw_c2w.shape[0]), spec.anchor_view)
    anchor_pose = context_raw_c2w[anchor_index]
    anchor_intr = context_intrinsics[anchor_index]
    focus = estimate_focus_point(context_raw_c2w)

    anchor_center = anchor_pose[:3, 3]
    anchor_rot = anchor_pose[:3, :3]
    anchor_up = -anchor_rot[:, 1]
    base_offset_world = anchor_center - focus
    orbit_radius = float(base_offset_world.norm().item())
    if orbit_radius < 1.0e-4:
        centers = context_raw_c2w[:, :3, 3]
        scene_center = centers.mean(0)
        orbit_radius = float((centers - scene_center).norm(dim=-1).mean().item())
        if orbit_radius < 1.0e-4:
            orbit_radius = 1.0
        focus = anchor_center - anchor_rot[:, 2] * orbit_radius
        base_offset_world = anchor_center - focus

    base_offset_local = anchor_rot.transpose(0, 1) @ base_offset_world

    if spec.mode == "txt":
        thetas, phis, radius_scales = load_traj_txt(
            path=str(spec.traj_txt),
            num_frames=spec.num_frames,
            device=device,
            dtype=dtype,
        )
        shift_xs = torch.zeros_like(thetas)
        shift_ys = torch.zeros_like(thetas)
    else:
        steps = torch.linspace(1.0 / float(spec.num_frames), 1.0, steps=int(spec.num_frames), device=device, dtype=dtype)
        thetas = steps * float(spec.theta)
        phis = steps * float(spec.phi)
        radius_scales = steps * float(spec.radius_scale)
        shift_xs = steps * float(spec.shift_x)
        shift_ys = steps * float(spec.shift_y)

    target_c2ws = []
    for theta_i, phi_i, radius_i, shift_x_i, shift_y_i in zip(thetas, phis, radius_scales, shift_xs, shift_ys):
        offset_local = _normalize(base_offset_local.unsqueeze(0))[0] * (orbit_radius * (1.0 + float(radius_i)))
        offset_local[0] += float(shift_x_i) * orbit_radius
        offset_local[1] += float(shift_y_i) * orbit_radius
        rot_local = _rotation_y(phi_i.unsqueeze(0))[0] @ _rotation_x(theta_i.unsqueeze(0))[0]
        offset_world = anchor_rot @ (rot_local @ offset_local)
        camera_center = focus + offset_world

        forward = _normalize((focus - camera_center).unsqueeze(0))[0]
        right = torch.cross(forward, anchor_up, dim=0)
        if float(right.norm().item()) < 1.0e-6:
            right = anchor_rot[:, 0]
        right = _normalize(right.unsqueeze(0))[0]
        down = _normalize(torch.cross(forward, right, dim=0).unsqueeze(0))[0]

        pose = torch.eye(4, device=device, dtype=dtype)
        pose[:3, 0] = right
        pose[:3, 1] = down
        pose[:3, 2] = forward
        pose[:3, 3] = camera_center
        target_c2ws.append(pose)

    target_c2w = torch.stack(target_c2ws, dim=0)
    target_intrinsics = anchor_intr.unsqueeze(0).expand(int(spec.num_frames), -1, -1).clone()
    trajectory_info = {
        "mode": spec.mode,
        "anchor_view": spec.anchor_view,
        "anchor_index": int(anchor_index),
        "orbit_radius": orbit_radius,
        "focus_point": [float(v) for v in focus.tolist()],
        "theta": float(spec.theta),
        "phi": float(spec.phi),
        "radius_scale": float(spec.radius_scale),
        "shift_x": float(spec.shift_x),
        "shift_y": float(spec.shift_y),
        "traj_txt": spec.traj_txt,
    }
    return target_c2w, target_intrinsics, trajectory_info
