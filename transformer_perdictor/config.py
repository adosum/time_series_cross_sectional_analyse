from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RunPaths:
    root: Path
    models: Path
    metrics: Path
    plots: Path
    config_snapshot: Path


@dataclass
class RuntimeContext:
    cfg: Dict[str, Any]
    run_paths: RunPaths


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    training_mode = cfg["training_mode"].lower()
    if training_mode not in ("pure", "pbt", "both"):
        raise ValueError(
            f"config: training_mode must be 'pure', 'pbt', or 'both', got '{training_mode}'"
        )
    cfg["training_mode"] = training_mode
    return cfg


def create_run_paths(base_dir: str = "results") -> RunPaths:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base_dir) / stamp
    models = root / "models"
    metrics = root / "metrics"
    plots = root / "plots"
    for p in (root, models, metrics, plots):
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        models=models,
        metrics=metrics,
        plots=plots,
        config_snapshot=root / "config_snapshot.yaml",
    )


def build_runtime_context(config_path: str) -> RuntimeContext:
    cfg = load_config(config_path)
    run_paths = create_run_paths(base_dir="results")
    with open(run_paths.config_snapshot, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return RuntimeContext(cfg=cfg, run_paths=run_paths)


def model_output_path(run_paths: RunPaths, configured_name: str) -> str:
    return str((run_paths.models / Path(configured_name).name).resolve())
