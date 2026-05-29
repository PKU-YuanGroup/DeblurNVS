from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass(frozen=True)
class DemoScene:
    scene_name: str
    scene_dir: Path
    input_dir_name: str
    input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class DemoSceneBatch:
    scene: DemoScene
    context_paths: tuple[Path, ...]
    target_paths: tuple[Path, ...]
    batch: dict[str, Any]
    context_inputs: torch.Tensor


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return abs(hash(path.stem)) % (10**9)


def list_images(directory: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(directory.glob(pattern))
    return tuple(sorted(paths, key=lambda path: (_numeric_stem(path), path.name)))


def select_evenly_spaced(paths: tuple[Path, ...], count: int) -> tuple[Path, ...]:
    if count >= len(paths):
        return paths
    if count == 1:
        return (paths[len(paths) // 2],)
    indices = [int(round(i * (len(paths) - 1) / (count - 1))) for i in range(count)]
    return tuple(paths[index] for index in indices)


def resize_preserve_aspect_and_center_crop(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    target_h, target_w = int(image_size[0]), int(image_size[1])
    src_w, src_h = image.size
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image size: {(src_h, src_w)}")

    scale = max(target_h / float(src_h), target_w / float(src_w))
    resized_h = max(target_h, int(round(src_h * scale)))
    resized_w = max(target_w, int(round(src_w * scale)))
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    image = image.resize((resized_w, resized_h), resampling)

    top = max(0, (resized_h - target_h) // 2)
    left = max(0, (resized_w - target_w) // 2)
    return TF.crop(image, top, left, target_h, target_w)


def load_image_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = resize_preserve_aspect_and_center_crop(image, image_size)
        return TF.to_tensor(image)


def discover_scene(scene_root: Path) -> DemoScene:
    scene_root = Path(scene_root).expanduser().resolve()
    train_dir = scene_root / "images_train"
    images_dir = scene_root / "images"
    if train_dir.is_dir():
        input_dir = train_dir
        input_dir_name = "images_train"
    elif images_dir.is_dir():
        input_dir = images_dir
        input_dir_name = "images"
    else:
        raise FileNotFoundError(f"Expected {scene_root} to contain images_train/ or images/")

    input_paths = list_images(input_dir)
    if not input_paths:
        raise RuntimeError(f"Scene {scene_root} has no input images under {input_dir}")

    return DemoScene(
        scene_name=scene_root.name,
        scene_dir=scene_root,
        input_dir_name=input_dir_name,
        input_paths=input_paths,
    )


def build_demo_trajectory_batch(
    scene: DemoScene,
    image_size: tuple[int, int],
    context_count: int,
    target_count: int,
) -> DemoSceneBatch:
    if len(scene.input_paths) < int(context_count):
        raise RuntimeError(
            f"Scene {scene.scene_name} has only {len(scene.input_paths)} input views, "
            f"cannot select {context_count} context views."
        )
    if int(target_count) < 1:
        raise ValueError(f"target_count must be positive, got {target_count}")

    context_paths = select_evenly_spaced(scene.input_paths, int(context_count))
    target_paths = tuple(scene.scene_dir / f"traj_{idx:03d}.png" for idx in range(int(target_count)))
    context_inputs = torch.stack([load_image_tensor(path, image_size) for path in context_paths], dim=0)
    target_placeholders = torch.zeros(
        int(target_count),
        3,
        int(image_size[0]),
        int(image_size[1]),
        dtype=context_inputs.dtype,
    )
    model_inputs = torch.cat([context_inputs, target_placeholders], dim=0)

    view_indices = list(range(int(context_count) + int(target_count)))
    batch = {
        "scene_id": f"demo_{scene.scene_name}",
        "scene_name": scene.scene_name,
        "clear": model_inputs.unsqueeze(0),
        "blur": model_inputs.unsqueeze(0),
        "camera_images": context_inputs.unsqueeze(0),
        "camera_select_indices": None,
        "view_indices": torch.tensor(view_indices, dtype=torch.long).unsqueeze(0),
        "camera_view_indices": torch.arange(int(context_count), dtype=torch.long).unsqueeze(0),
        "context_indices": torch.arange(int(context_count), dtype=torch.long).unsqueeze(0),
        "target_indices": torch.arange(int(context_count), int(context_count) + int(target_count), dtype=torch.long).unsqueeze(0),
        "cond_num_override": int(context_count),
    }
    return DemoSceneBatch(
        scene=scene,
        context_paths=context_paths,
        target_paths=target_paths,
        batch=batch,
        context_inputs=context_inputs,
    )
