from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from .config import build_default_config
from .deps import REPO_ROOT, configure_import_paths, resolve_dependency_paths
from .scene import DemoSceneBatch, build_demo_trajectory_batch, discover_scene
from .trajectory import TrajectorySpec, build_target_camera_trajectory


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


def save_rgb_image(image: torch.Tensor, path: Path) -> None:
    image_uint8 = image.detach().float().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    to_pil_image(image_uint8).save(path)


def as_homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {tuple(extrinsics.shape)}")
    out = torch.zeros(*extrinsics.shape[:-2], 4, 4, device=extrinsics.device, dtype=extrinsics.dtype)
    out[..., :3, :4] = extrinsics
    out[..., 3, 3] = 1.0
    return out


def _make_blank_like(example: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(example)


def _concat_with_spacer(images: list[torch.Tensor], spacer_width: int = 8, spacer_value: float = 0.04) -> torch.Tensor:
    if not images:
        raise ValueError("images must be non-empty")
    output = images[0]
    if len(images) == 1:
        return output
    _, h, _ = images[0].shape
    spacer = torch.full((3, h, spacer_width), float(spacer_value), dtype=images[0].dtype)
    for image in images[1:]:
        output = torch.cat([output, spacer, image], dim=2)
    return output


def make_overview(
    context_inputs: torch.Tensor,
    pred_full: torch.Tensor,
    cond_num: int,
) -> torch.Tensor:
    blank = _make_blank_like(context_inputs[0])
    target_count = int(pred_full.shape[0]) - int(cond_num)
    observed_tiles = [context_inputs[idx] for idx in range(cond_num)] + [blank for _ in range(target_count)]
    return torch.cat(
        [
        _concat_with_spacer(observed_tiles),
        _concat_with_spacer([pred_full[idx] for idx in range(int(pred_full.shape[0]))]),
        ],
        dim=1,
    )


@dataclass(frozen=True)
class RuntimeModules:
    FrozenDA3FeatureExtractor: Any
    configure_rae_da3_lora: Any
    load_decoder_checkpoint: Any
    instantiate_gld_stage1: Any
    instantiate_gld_stage2: Any
    load_gld_pretrained: Any
    compute_time_dist_shift: Any
    create_transport: Any
    build_sampler: Any
    load_checkpoint_step: Any
    load_frozen_stage1_lora_if_available: Any
    ensure_model_finite: Any
    build_reference_condition_from_source: Any
    build_camera_embedding: Any
    canonicalize_camera_params_to_reference_view: Any
    _apply_camera_conditioning_override: Any
    _maybe_create_context_refiner_from_prepared: Any
    _maybe_apply_context_refiner: Any
    build_decoder_static_context: Any
    decode_scene_prediction_from_samples: Any
    sample_diffusion_latents: Any


def import_runtime_modules() -> RuntimeModules:
    from .runtime import (
        FrozenDA3FeatureExtractor,
        _apply_camera_conditioning_override,
        build_camera_embedding,
        build_decoder_static_context,
        build_reference_condition_from_source,
        build_sampler,
        canonicalize_camera_params_to_reference_view,
        compute_time_dist_shift,
        configure_rae_da3_lora,
        create_transport,
        decode_scene_prediction_from_samples,
        ensure_model_finite,
        instantiate_gld_stage1,
        instantiate_gld_stage2,
        load_checkpoint_step,
        load_decoder_checkpoint,
        load_frozen_stage1_lora_if_available,
        load_gld_pretrained,
        maybe_apply_context_refiner,
        maybe_create_context_refiner_from_prepared,
        sample_diffusion_latents,
    )

    return RuntimeModules(
        FrozenDA3FeatureExtractor=FrozenDA3FeatureExtractor,
        configure_rae_da3_lora=configure_rae_da3_lora,
        load_decoder_checkpoint=load_decoder_checkpoint,
        instantiate_gld_stage1=instantiate_gld_stage1,
        instantiate_gld_stage2=instantiate_gld_stage2,
        load_gld_pretrained=load_gld_pretrained,
        compute_time_dist_shift=compute_time_dist_shift,
        create_transport=create_transport,
        build_sampler=build_sampler,
        load_checkpoint_step=load_checkpoint_step,
        load_frozen_stage1_lora_if_available=load_frozen_stage1_lora_if_available,
        ensure_model_finite=ensure_model_finite,
        build_reference_condition_from_source=build_reference_condition_from_source,
        build_camera_embedding=build_camera_embedding,
        canonicalize_camera_params_to_reference_view=canonicalize_camera_params_to_reference_view,
        _apply_camera_conditioning_override=_apply_camera_conditioning_override,
        _maybe_create_context_refiner_from_prepared=maybe_create_context_refiner_from_prepared,
        _maybe_apply_context_refiner=maybe_apply_context_refiner,
        build_decoder_static_context=build_decoder_static_context,
        decode_scene_prediction_from_samples=decode_scene_prediction_from_samples,
        sample_diffusion_latents=sample_diffusion_latents,
    )


@dataclass(frozen=True)
class InferenceOutputs:
    output_dir: Path
    pred_full: torch.Tensor
    metadata: dict[str, Any]


class DeblurNVSPipeline:
    def __init__(self, device: str | None = None, sampler_steps: int | None = None) -> None:
        self.repo_root = REPO_ROOT
        self.paths = resolve_dependency_paths(self.repo_root)
        configure_import_paths(self.paths)
        self.runtime = import_runtime_modules()
        self.cfg = build_default_config(self.paths, self.repo_root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sampler_steps = sampler_steps
        self._configure_runtime()
        self._load_models()

    def _configure_runtime(self) -> None:
        if bool(self.cfg.runtime.get("allow_tf32", False)) and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(int(self.cfg.seed))

    def _resolve_checkpoint(self, path: str | Path) -> Path:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    def _build_sampler(self, total_view: int):
        latent_dim = int(self.cfg.gld.stage2.params.in_channels)
        feat_h = int(self.cfg.data.image_size[0]) // int(self.cfg.da3.patch_size)
        feat_w = int(self.cfg.data.image_size[1]) // int(self.cfg.da3.patch_size)
        transport = self.runtime.create_transport(
            path_type=self.cfg.gld.transport.get("path_type", "Linear"),
            prediction=self.cfg.gld.transport.get("prediction", "velocity"),
            loss_weight=self.cfg.gld.transport.get("loss_weight"),
            time_dist_type=self.cfg.gld.transport.get("time_dist_type", "uniform"),
            time_dist_shift=self.runtime.compute_time_dist_shift(
                channels=latent_dim,
                feat_h=feat_h,
                feat_w=feat_w,
                total_view=int(total_view),
                shift_base=float(self.cfg.gld.get("time_dist_shift_base", 4096.0)),
            ),
        )
        return self.runtime.build_sampler(transport, self.cfg.get("sampling"), self.sampler_steps)

    def _resolve_feature_level(self, level: int) -> int:
        level = int(level)
        if level in {-4, -3, -2, -1}:
            return level + 4
        if level in {0, 1, 2, 3}:
            return level
        raise ValueError(f"Unsupported feature level {level!r}")

    def _normalize_with_stage1(self, level_features: torch.Tensor) -> torch.Tensor:
        flattened = level_features.reshape(-1, *level_features.shape[2:])
        if bool(getattr(self.rae, "do_normalization", False)):
            flattened = self.rae._normalize(flattened, dim=int(flattened.shape[1]))
        return flattened

    def _predict_context_cameras(self, context_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_batch = context_inputs.unsqueeze(0).to(self.device, non_blocking=True)
        context_pyramid = self.extractor.extract_level_features(context_batch)
        raw_camera_outputs = self.extractor.predict_outputs_from_level_features(context_pyramid, context_batch)
        raw_w2c = as_homogeneous_extrinsics(raw_camera_outputs["extrinsics"])
        raw_c2w = torch.linalg.inv(raw_w2c)
        intrinsics = raw_camera_outputs["intrinsics"]
        return raw_c2w, intrinsics

    def _build_synthetic_prepared_batch(
        self,
        clear: torch.Tensor,
        blur: torch.Tensor,
        cond_num: int,
        intrinsics: torch.Tensor,
        raw_c2w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        total_view = int(clear.shape[1])
        level_index = self._resolve_feature_level(self.feature_level)

        clear_pyramid = self.extractor.extract_level_features(clear)
        blur_pyramid = self.extractor.extract_level_features(blur)
        clear_level = clear_pyramid[level_index]
        blur_level = blur_pyramid[level_index]

        x1_all = self._normalize_with_stage1(clear_level)
        blur_level_norm = self._normalize_with_stage1(blur_level)
        x1_cond = self.runtime.build_reference_condition_from_source(
            source_latents=blur_level_norm,
            num_views=total_view,
            cond_num=cond_num,
        )
        x0_init = torch.randn_like(x1_all)

        camera_params = self.runtime.canonicalize_camera_params_to_reference_view(
            {
                "intrinsics": intrinsics,
                "extrinsics": raw_c2w,
                "raw_extrinsics_w2c": torch.linalg.inv(raw_c2w),
            },
            reference_view=str(self.cfg.gld.get("camera_reference_view", "native")),
        )
        camera_embedding, prope_inputs = self.runtime.build_camera_embedding(
            intrinsics=camera_params["intrinsics"],
            extrinsics=camera_params["extrinsics"],
            image_h=int(clear.shape[-2]),
            image_w=int(clear.shape[-1]),
            cond_num=int(cond_num),
            mode=str(self.cfg.gld.get("camera_mode", "plucker")),
            reference_view=str(self.cfg.gld.get("camera_reference_view", "native")),
        )
        prope_inputs["viewmats"] = camera_params["canonical_viewmats"]
        camera_embedding, prope_inputs = self.runtime._apply_camera_conditioning_override(
            camera_embedding=camera_embedding,
            prope_inputs=prope_inputs,
            mode=str(self.cfg.gld.get("camera_conditioning_mode", "full")),
        )

        return {
            "x1_cond": x1_cond,
            "x1_all": x1_all,
            "x_blur_all": blur_level_norm,
            "x0_init": x0_init,
            "camera_embedding": camera_embedding,
            "pred_intrinsics": camera_params["intrinsics"],
            "pred_extrinsics": camera_params["extrinsics"],
            "pred_extrinsics_raw_c2w": camera_params["raw_extrinsics_c2w"],
            "pred_extrinsics_raw_w2c": camera_params["raw_extrinsics_w2c"],
            "pred_reference_view_index": camera_params["reference_view_index"],
            "pred_reference_view_mode": camera_params["reference_view_mode"],
            "pred_translation_scale": camera_params["translation_scale"],
            **prope_inputs,
        }

    def _load_models(self) -> None:
        feature_level = int(self.cfg.da3.get("feature_level", self.cfg.gld.stage1.params.get("level", 1)))
        self.feature_level = feature_level
        self.mixed_precision = str(self.cfg.runtime.get("mixed_precision", "bf16"))

        self.stage1_lora_checkpoint = self._resolve_checkpoint(self.cfg.checkpoints.stage1_lora)
        self.stage2_checkpoint = self._resolve_checkpoint(self.cfg.checkpoints.stage2_diffusion)
        self.stage3_checkpoint = self._resolve_checkpoint(self.cfg.checkpoints.stage3_decoder)

        self.extractor = self.runtime.FrozenDA3FeatureExtractor(
            model_id=self.cfg.da3.model_id,
            export_layers=None if self.cfg.da3.export_layers is None else list(self.cfg.da3.export_layers),
            local_files_only=bool(self.cfg.da3.local_files_only),
            ref_view_strategy=self.cfg.da3.ref_view_strategy,
            patch_size=int(self.cfg.da3.patch_size),
        ).to(self.device)
        freeze_module(self.extractor)

        self.rae = self.runtime.instantiate_gld_stage1(self.cfg.gld.stage1).to(self.device)
        self.runtime.configure_rae_da3_lora(self.rae, self.cfg.get("da3_lora"))
        freeze_module(self.rae)
        self.runtime.load_frozen_stage1_lora_if_available(self.rae, self.cfg)
        missing_dec, unexpected_dec, _ = self.runtime.load_decoder_checkpoint(
            self.rae.mae_decoder,
            self.stage3_checkpoint,
            strict=False,
        )
        print(f"[DeblurNVS] decoder loaded: missing={len(missing_dec)} unexpected={len(unexpected_dec)}")

        self.model = self.runtime.instantiate_gld_stage2(self.cfg.gld.stage2).to(self.device)
        missing, unexpected = self.runtime.load_gld_pretrained(
            self.model,
            str(self.stage2_checkpoint),
            strict=bool(self.cfg.gld.get("pretrained_strict", False)),
        )
        freeze_module(self.model)
        self.runtime.ensure_model_finite(self.model, "stage2 model")
        print(f"[DeblurNVS] stage2 loaded: missing={len(missing)} unexpected={len(unexpected)}")

        self.stage2_step = self.runtime.load_checkpoint_step(self.stage2_checkpoint)
        self.stage3_step = self.runtime.load_checkpoint_step(self.stage3_checkpoint)

    @torch.no_grad()
    def _run_scene_with_generated_trajectory(
        self,
        scene_root: str | Path,
        context_views: int,
        output_dir: str | Path,
        trajectory_spec: TrajectorySpec,
    ) -> InferenceOutputs:
        scene = discover_scene(Path(scene_root))
        image_size = tuple(int(v) for v in self.cfg.data.image_size)
        scene_batch = build_demo_trajectory_batch(
            scene=scene,
            image_size=image_size,
            context_count=int(context_views),
            target_count=int(trajectory_spec.num_frames),
        )

        cond_num = int(scene_batch.batch["cond_num_override"])
        clear = scene_batch.batch["clear"].to(self.device, non_blocking=True)
        blur = scene_batch.batch["blur"].to(self.device, non_blocking=True)
        context_raw_c2w, context_intrinsics = self._predict_context_cameras(scene_batch.context_inputs)
        target_raw_c2w, target_intrinsics, trajectory_info = build_target_camera_trajectory(
            context_raw_c2w=context_raw_c2w[0],
            context_intrinsics=context_intrinsics[0],
            spec=trajectory_spec,
        )

        raw_c2w_total = torch.cat([context_raw_c2w, target_raw_c2w.unsqueeze(0)], dim=1)
        intrinsics_total = torch.cat([context_intrinsics, target_intrinsics.unsqueeze(0)], dim=1)
        prepared = self._build_synthetic_prepared_batch(
            clear=clear,
            blur=blur,
            cond_num=cond_num,
            intrinsics=intrinsics_total,
            raw_c2w=raw_c2w_total,
        )

        refiner_bundle = self.runtime._maybe_create_context_refiner_from_prepared(
            cfg=self.cfg,
            device=self.device,
            prepared=prepared,
        )
        prepared = self.runtime._maybe_apply_context_refiner(
            prepared=prepared,
            clear=clear,
            total_view=int(clear.shape[1]),
            cond_num=cond_num,
            refiner_bundle=refiner_bundle,
            mixed_precision=self.mixed_precision,
            device=self.device,
        )
        decoder_static = self.runtime.build_decoder_static_context(
            rae=self.rae,
            clear=clear,
            total_view=int(clear.shape[1]),
            cond_num=cond_num,
            validation_mode=str(self.cfg.decoder.get("propagation_mode", "propagation_0")),
        )
        sampler = self._build_sampler(total_view=int(clear.shape[1]))
        sampled = self.runtime.sample_diffusion_latents(
            model=self.model,
            sampler=sampler,
            prepared=prepared,
            clear=clear,
            total_view=int(clear.shape[1]),
            cond_num=cond_num,
            mixed_precision=self.mixed_precision,
            device=self.device,
        )
        pred_rgb = self.runtime.decode_scene_prediction_from_samples(
            rae=self.rae,
            sampled=sampled,
            decoder_static=decoder_static,
            validation_mode=str(self.cfg.decoder.get("propagation_mode", "propagation_0")),
            view_chunk_size=int(self.cfg.decoder.get("view_chunk_size", 1)),
            mixed_precision=self.mixed_precision,
            device=self.device,
            cond_num=cond_num,
        ).clamp(0.0, 1.0)

        pred_full = pred_rgb[0].detach().cpu().clamp(0.0, 1.0)
        output_dir = Path(output_dir).expanduser().resolve()
        self._save_outputs(
            output_dir=output_dir,
            scene_batch=scene_batch,
            pred_full=pred_full,
            raw_c2w_total=raw_c2w_total[0].detach().cpu(),
            intrinsics_total=intrinsics_total[0].detach().cpu(),
            trajectory_mode="generated",
        )

        metadata = {
            "scene_name": scene_batch.scene.scene_name,
            "scene_root": str(scene_batch.scene.scene_dir),
            "input_dir_name": scene_batch.scene.input_dir_name,
            "context_views": int(context_views),
            "target_views": int(trajectory_spec.num_frames),
            "image_size": list(image_size),
            "device": str(self.device),
            "stage1_lora": str(self.stage1_lora_checkpoint),
            "stage2_diffusion": str(self.stage2_checkpoint),
            "stage3_decoder": str(self.stage3_checkpoint),
            "stage2_step": self.stage2_step,
            "stage3_step": self.stage3_step,
            "context_files": [path.name for path in scene_batch.context_paths],
            "target_files": [path.name for path in scene_batch.target_paths],
            "trajectory": trajectory_info,
            "notes": {
                "camera_reference_view": str(self.cfg.gld.get("camera_reference_view", "native")),
                "trajectory_source": "generated_from_context_cameras",
            },
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return InferenceOutputs(output_dir=output_dir, pred_full=pred_full, metadata=metadata)

    @torch.no_grad()
    def run_scene(
        self,
        scene_root: str | Path,
        context_views: int,
        output_dir: str | Path,
        trajectory_mode: str = "auto",
        num_novel_views: int = 25,
        anchor_view: str = "last",
        traj_theta: float = 0.0,
        traj_phi: float = 20.0,
        traj_radius_scale: float = 0.0,
        traj_shift_x: float = 0.0,
        traj_shift_y: float = 0.0,
        traj_txt: str | None = None,
    ) -> InferenceOutputs:
        trajectory_mode = str(trajectory_mode).lower()
        spec = TrajectorySpec(
            mode="interp" if trajectory_mode == "auto" else ("txt" if trajectory_mode == "txt" else trajectory_mode),
            num_frames=int(num_novel_views),
            anchor_view=str(anchor_view),
            theta=float(traj_theta),
            phi=float(traj_phi),
            radius_scale=float(traj_radius_scale),
            shift_x=float(traj_shift_x),
            shift_y=float(traj_shift_y),
            traj_txt=None if traj_txt is None else str(traj_txt),
        )
        return self._run_scene_with_generated_trajectory(
            scene_root=scene_root,
            context_views=context_views,
            output_dir=output_dir,
            trajectory_spec=spec,
        )

    def _save_outputs(
        self,
        output_dir: Path,
        scene_batch: DemoSceneBatch,
        pred_full: torch.Tensor,
        raw_c2w_total: torch.Tensor | None = None,
        intrinsics_total: torch.Tensor | None = None,
        trajectory_mode: str = "eval",
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        context_views_dir = output_dir / "context_views"
        context_views_pred_dir = output_dir / "context_views_pred"
        pred_dir = output_dir / "pred"
        for directory in (context_views_dir, context_views_pred_dir, pred_dir):
            directory.mkdir(parents=True, exist_ok=True)

        cond_num = int(scene_batch.batch["cond_num_override"])
        context_inputs = scene_batch.context_inputs.detach().cpu().clamp(0.0, 1.0)

        for idx, path in enumerate(scene_batch.context_paths):
            save_rgb_image(context_inputs[idx], context_views_dir / path.name)
            save_rgb_image(pred_full[idx], context_views_pred_dir / path.name)

        for target_idx, path in enumerate(scene_batch.target_paths):
            pred_idx = cond_num + target_idx
            save_rgb_image(pred_full[pred_idx], pred_dir / path.name)

        overview = make_overview(context_inputs, pred_full, cond_num=cond_num)
        save_image(overview, output_dir / "overview.png")

        if raw_c2w_total is not None and intrinsics_total is not None:
            camera_payload = {
                "trajectory_mode": str(trajectory_mode),
                "context_count": int(cond_num),
                "total_view": int(raw_c2w_total.shape[0]),
                "raw_c2w": raw_c2w_total.tolist(),
                "intrinsics": intrinsics_total.tolist(),
            }
            (output_dir / "camera_path.json").write_text(json.dumps(camera_payload, indent=2))
