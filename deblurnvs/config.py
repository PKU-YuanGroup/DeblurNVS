from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .deps import DependencyPaths, REPO_ROOT


def build_default_config(paths: DependencyPaths, repo_root: Path | None = None):
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    pretrained_root = repo_root / "pretrained"
    stage1_ckpt = repo_root / "pretrained" / "stage1_lora.pt"
    stage2_ckpt = repo_root / "pretrained" / "stage2_diffusion.pt"
    stage3_ckpt = repo_root / "pretrained" / "stage3_decoder.pt"
    da3_model_dir = pretrained_root / "da3_base"
    normalization_stats = pretrained_root / "normalization_stats_level1.pt"

    return OmegaConf.create(
        {
            "seed": 42,
            "da3": {
                "model_id": str(da3_model_dir),
                "local_files_only": True,
                "export_layers": None,
                "ref_view_strategy": "first",
                "patch_size": 14,
                "feature_level": 1,
            },
            "da3_lora": {
                "enabled": True,
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
                "target_modules": ["qkv", "proj", "fc1", "fc2"],
                "exclude_name_contains": ["patch_embed"],
                "verbose": False,
                "checkpoint": str(stage1_ckpt),
            },
            "data": {
                "image_size": [280, 504],
                "num_views": 17,
                "cond_num": "3-12",
            },
            "runtime": {
                "mixed_precision": "bf16",
                "allow_tf32": True,
            },
            "decoder": {
                "propagation_mode": "propagation_0",
                "view_chunk_size": 1,
            },
            "refiner": {
                "enabled": True,
                "checkpoint": str(stage1_ckpt),
                "strict": False,
                "max_context_views": 12,
                "camera_mode": "plucker",
                "camera_conditioning_mode": "identity",
                "context_blend": 1.0,
                "sampler": {
                    "sampling_method": "euler",
                    "num_steps": 8,
                    "atol": 1.0e-6,
                    "rtol": 1.0e-3,
                    "reverse": False,
                },
            },
            "sampling": {
                "sampler": {
                    "sampling_method": "euler",
                    "num_steps": 12,
                    "atol": 1.0e-6,
                    "rtol": 1.0e-3,
                    "reverse": False,
                },
            },
            "gld": {
                "camera_prediction_source": "clear",
                "camera_mode": "plucker",
                "camera_conditioning_mode": "full",
                "camera_reference_view": "native",
                "pretrained_stage2": str(stage2_ckpt),
                "pretrained_strict": False,
                "time_dist_shift_base": 4096.0,
                "transport": {
                    "path_type": "Linear",
                    "prediction": "velocity",
                    "loss_weight": None,
                    "time_dist_type": "uniform",
                },
                "stage1": {
                    "target": "stage1.rae_da3.RAE_DA3",
                    "params": {
                        "encoder_pretrained_path": str(da3_model_dir),
                        "encoder_input_size": 504,
                        "encoder_type": "DA3EncoderDirect",
                        "decoder_config_path": str(paths.gld_root / "configs" / "decoder" / "ViTXL"),
                        "da3_weights_path": str(da3_model_dir / "model.safetensors"),
                        "mae_weight": None,
                        "noise_tau": 0.0,
                        "reshape_to_2d": True,
                        "normalization_stat_path": str(normalization_stats),
                        "level": 1,
                    },
                },
                "stage2": {
                    "target": "stage2.models.DDT.DiTwDDTHead",
                    "params": {
                        "patch_size": 1,
                        "in_channels": 1536,
                        "hidden_size": [768, 2048],
                        "depth": [28, 6],
                        "num_heads": [16, 16],
                        "mlp_ratio": 4.0,
                        "use_qknorm": False,
                        "use_swiglu": True,
                        "use_rope": False,
                        "use_rmsnorm": True,
                        "wo_shift": False,
                        "use_pos_embed": False,
                        "use_prope": True,
                        "architecture_mode": "new",
                        "cfg_mode": "new",
                        "cam_input_size": 504,
                        "cam_patch_size": 14,
                        "level": 1,
                        "predict_cls": False,
                        "is_concat_mode": True,
                    },
                },
            },
            "checkpoints": {
                "stage1_lora": str(stage1_ckpt),
                "stage2_diffusion": str(stage2_ckpt),
                "stage3_decoder": str(stage3_ckpt),
            },
        }
    )
