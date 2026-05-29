from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_ROOT = REPO_ROOT / "utils"


@dataclass(frozen=True)
class DependencyPaths:
    da3_root: Path
    da3_src_root: Path
    gld_root: Path
    gld_src_root: Path


def resolve_dependency_paths(repo_root: Path | None = None) -> DependencyPaths:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    utils_root = repo_root / "utils"
    da3_root = (utils_root / "da3_runtime").resolve()
    da3_src_root = (da3_root / "src").resolve()
    gld_root = (utils_root / "gld").resolve()
    gld_src_root = (gld_root / "src").resolve()

    required_dirs = {
        "local DA3 root": da3_root,
        "local DA3 src": da3_src_root,
        "local GLD root": gld_root,
        "local GLD src": gld_src_root,
    }
    for label, path in required_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")

    return DependencyPaths(
        da3_root=da3_root,
        da3_src_root=da3_src_root,
        gld_root=gld_root,
        gld_src_root=gld_src_root,
    )


def configure_import_paths(paths: DependencyPaths) -> None:
    os.environ["MVDIFF_GLD_SRC"] = str(paths.gld_src_root)
    os.environ["MVDIFF_DA3_SRC"] = str(paths.da3_src_root)

    for candidate in (paths.da3_root, paths.da3_src_root, paths.gld_src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
