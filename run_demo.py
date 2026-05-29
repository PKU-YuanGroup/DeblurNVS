from __future__ import annotations

import argparse
from pathlib import Path

from deblurnvs import DeblurNVSPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DeblurNVS interpolation demo.")
    parser.add_argument(
        "--scene-root",
        type=str,
        default="example",
        help="Scene folder containing blurred input views under images_train/ or images/.",
    )
    parser.add_argument(
        "--context-views",
        type=int,
        default=9,
        help="Number of evenly spaced input views used as context.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--sampler-steps",
        type=int,
        default=None,
        help="Optional override for the stage2 diffusion sampler step count.",
    )
    parser.add_argument(
        "--num-novel-views",
        type=int,
        default=25,
        help="Number of interpolated novel views generated between the context cameras.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_interp",
        help="Directory where predictions and metadata will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DeblurNVSPipeline(device=args.device, sampler_steps=args.sampler_steps)
    outputs = pipeline.run_scene(
        scene_root=Path(args.scene_root),
        context_views=int(args.context_views),
        output_dir=Path(args.output_dir),
        trajectory_mode="interp",
        num_novel_views=int(args.num_novel_views),
    )
    print(f"[DeblurNVS] output_dir={outputs.output_dir}")
    print("[DeblurNVS] interpolation demo finished")


if __name__ == "__main__":
    main()
