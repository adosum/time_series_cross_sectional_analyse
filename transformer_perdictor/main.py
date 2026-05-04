from __future__ import annotations

import argparse
import copy
import os
import optuna

import torch

from .config import create_run_paths, load_config
from .data import prepare_data
from .helpers import apply_reproducibility, configure_torch_runtime
from .pbt_trainer import run_pbt_training
from .trainer import run_pure_training


def _set_nested(cfg, path, value):
    node = cfg
    parts = path.split(".")
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = value


def _run_single_config(cfg, run_name):
    """
    Runs a single training pipeline and returns the summary metrics.
    """
    run_paths = create_run_paths(base_dir="results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Persist exact config used for this run.
    with open(run_paths.config_snapshot, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\nRun label: {run_name}")
    print(f"Training mode: {cfg['training_mode']}")
    print(f"Run output folder: {run_paths.root}")

    data = prepare_data(cfg, device)
    mode = cfg["training_mode"]
    pure_summary = None
    pbt_summary = None

    if mode in ("pure", "both"):
        pure_summary = run_pure_training(cfg, run_paths, data, device)

    if mode in ("pbt", "both"):
        pbt_summary = run_pbt_training(cfg, run_paths, data, device)

    print("\nRun complete.")
    print(f"All results are in: {run_paths.root}")
    if pure_summary is not None:
        print(f"Pure best model: {pure_summary['model_path']}")
    if pbt_summary is not None:
        print(f"PBT production model: {pbt_summary['production_path']}")
        
    # Return the summary so Optuna can read the metrics!
    # If running "both", PBT summary takes precedence
    return pbt_summary if pbt_summary else pure_summary


def run_optuna_sweep(base_cfg):
    """
    Smart Bayesian Optimization using Optuna.
    Dynamically parses categorical and continuous parameters from the YAML config.
    Runs one isolated sweep per config (per dataset).
    """

    sweep_cfg = base_cfg.get("sweep", {})
    if not bool(sweep_cfg.get("enabled", False)):
        print("Sweep disabled. Running single base config.")
        _run_single_config(base_cfg, "base_run")
        return

    sweep_params = sweep_cfg.get("params", {})
    if not sweep_params:
        print("Sweep enabled but no params found. Running single base config.")
        _run_single_config(base_cfg, "base_run")
        return

    csv_file = str(base_cfg.get("data", {}).get("csv_file", "default")).strip()
    model_name = os.path.splitext(os.path.basename(csv_file))[0] or "default"
    print(f"\nStarting isolated sweep for dataset: {csv_file} (study key: {model_name})")
    _run_single_sweep(base_cfg, sweep_params, sweep_cfg, model_name=model_name)


def _run_single_sweep(base_cfg, sweep_params, sweep_cfg, model_name="default"):
    """
    Run a single Optuna sweep for a specific model configuration.
    """
    import copy
    import optuna

    n_trials = sweep_cfg.get("n_trials", 20)
    print(f"\nStarting Bayesian Sweep for {model_name} with {n_trials} trials...")

    def objective(trial):
        cfg_variant = copy.deepcopy(base_cfg)
        
        # 1. Let Optuna dynamically suggest parameters based on your YAML
        for key, param_config in sweep_params.items():
            
            # Safety check: ensure it's a dictionary before checking keys
            if isinstance(param_config, dict):
                # If the YAML uses "values" (Categorical choices like [32, 64])
                if "values" in param_config:
                    suggested_val = trial.suggest_categorical(key, param_config["values"])
                    
                # If the YAML uses "low" and "high" (Continuous Float ranges like LR)
                elif "low" in param_config and "high" in param_config:
                    is_log = param_config.get("log", False)
                    suggested_val = trial.suggest_float(
                        key, 
                        param_config["low"], 
                        param_config["high"], 
                        log=is_log
                    )
                else:
                    suggested_val = param_config
            else:
                # Fallback: if you just passed a raw list in YAML (e.g., d_model: [32, 64])
                if isinstance(param_config, list):
                    suggested_val = trial.suggest_categorical(key, param_config)
                else:
                    suggested_val = param_config

            # Apply the suggestion to the config variant
            _set_nested(cfg_variant, key, suggested_val)

        run_name = f"{model_name}_trial_{trial.number:03d}"
        print(f"\n{'='*50}\nStarting {run_name}\nParams: {trial.params}\n{'='*50}")

        try:
            # 2. Run the training pipeline
            summary = _run_single_config(cfg_variant, run_name)
            
            if summary is None:
                raise optuna.TrialPruned("Trainer returned no summary.")
                
            # 3. Extract the target metric
            # We want to MAXIMIZE test_accuracy. Optuna MINIMIZES by default.
            # So we return the negative accuracy.
            target_metric = summary.get("test_accuracy", 0.0)
            return -1.0 * target_metric
            
        except Exception as e:
            # If an architecture causes CUDA Out Of Memory, prune the trial and move on
            print(f"Trial {trial.number} failed or was pruned: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned()

    # 4. Create model-specific database to store progress
    # Each model gets its own Optuna study to keep results isolated
    db_name = f"optuna_sweep_{model_name}.db"
    study_name = f"quant_transformer_sweep_{model_name}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # We minimize the negative accuracy
        storage=f"sqlite:///{db_name}",  # Separate DB per model
        load_if_exists=True
    )
    
    # 5. Run the optimization loop
    study.optimize(objective, n_trials=n_trials)

    # 6. Print the ultimate winner for this model
    print("\n" + "="*50)
    print(f"🏆 SWEEP COMPLETE FOR {model_name.upper()} 🏆")
    print(f"Best Test Accuracy: {-study.best_value:.4f}")
    print("Best Architecture Parameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)


def run(config_path: str):
    cfg = load_config(config_path)

    apply_reproducibility(cfg["seed"])
    deterministic = bool(cfg.get("deterministic", False))
    configure_torch_runtime(deterministic=deterministic)
    print(f"Deterministic mode: {deterministic}")

    # Fire the Optuna function
    run_optuna_sweep(cfg)


def main():
    parser = argparse.ArgumentParser(description="USDCNH Transformer Training")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()