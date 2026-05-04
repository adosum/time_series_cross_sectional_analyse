from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
from numpy.random import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from .config import model_output_path
from .helpers import (
    augment_financial_data,
    compute_trend_strength,
    prepare_error_states,
    update_error_history,
)
from .model import USDCNHTransformer
from .losses import UncertaintyLoss, gaussian_nll_loss


def run_pure_training(cfg, run_paths, data, device):
    def _log(msg: str):
        print(msg)

    def _avg_metric(total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return total / count

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    error_history_len = model_cfg["error_history_len"]
    model_input_size = len(data.feature_cols)  # Features only, no error history appended
    _log(f"Model input size (features only): {model_input_size}")
    _log(f"Error history length: {error_history_len}")

    d_model = model_cfg["d_model"]
    nhead = model_cfg["nhead"]
    num_layers = model_cfg["num_layers"]
    dropout = model_cfg["dropout"]

    base_model = USDCNHTransformer(
        input_size=model_input_size,
        seq_len=cfg["data"]["sequence_length"],
        error_history_len=error_history_len,
        nhead=nhead,
        num_layers=num_layers,
        d_model=d_model,
        dropout=dropout,
        patch_len=model_cfg.get("patch_len", 4),
        patch_stride=model_cfg.get("patch_stride", 2),
    ).to(device)

    model = torch.compile(base_model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val_loss = float("inf")
    model_save_path = model_output_path(run_paths, model_cfg["model_save_path"])

    uncertainty_loss_function = UncertaintyLoss(num_tasks=3).to(device)
    grad_clip_max_norm = train_cfg["grad_clip_max_norm"]
    val_freq = train_cfg["val_freq"]
    stage1_epochs = train_cfg["stage1_epochs"]
    total_epochs = train_cfg["epochs"]
    stage2_cfg = train_cfg["stage2"]
    stage2_grad_accum_steps = max(1, int(stage2_cfg.get("grad_accum_steps", 1)))
    parallel_streams = max(1, int(stage2_cfg.get("parallel_streams", stage2_grad_accum_steps)))

    empirical_error_sigma = float(torch.std(data.y_train_tensor[:, 0]).item())
    default_stage1_error_sigma = max(empirical_error_sigma, 1e-3)
    stage1_error_sigma = float(
        stage2_cfg.get("stage1_error_state_sigma", default_stage1_error_sigma)
    )

    training_losses = []
    training_nll_15m = []
    training_nll_1h = []
    training_nll_4h = []
    validation_losses = []
    validation_nll_15m = []
    validation_nll_1h = []
    validation_nll_4h = []
    validation_epochs = []

    test_accuracy = None
    test_long_accuracy = None
    test_short_accuracy = None
    test_confidence_mean = None
    test_accuracy_conf_55 = None
    test_coverage_conf_55 = None
    test_accuracy_conf_60 = None
    test_coverage_conf_60 = None
    test_accuracy_conf_70 = None
    test_coverage_conf_70 = None

    _log("\n" + "=" * 60)
    _log("STAGE 1: BATCH TRAINING WITH RANDOM ERROR CHANNELS (Pre-training)")
    _log("=" * 60)

    for epoch in tqdm(
        range(1, stage1_epochs + 1),
        desc="Stage 1: Base Training",
        unit="epoch",
    ):
        model.train()
        epoch_loss = 0.0
        epoch_nll_15m = 0.0
        epoch_nll_1h = 0.0
        epoch_nll_4h = 0.0
        epoch_count = 0

        for batch_X, batch_y in data.train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_size_current = batch_X.shape[0]
            y_15m = batch_y[:, 0]
            y_1h  = batch_y[:, 1]
            y_4h  = batch_y[:, 2]

            optimizer.zero_grad()

            batch_X_aug = augment_financial_data(
                batch_X,
                noise_std=data.feature_noise_std,
                mask_prob=data.feature_mask_prob,
            )

            error_states = (
                torch.randn(batch_size_current, error_history_len, device=device, dtype=batch_X_aug.dtype)
                * stage1_error_sigma
            )

            pred_15m, pred_1h, pred_4h = model(batch_X_aug, error_state=error_states)
            mu_15m, sig_15m = pred_15m
            mu_1h, sig_1h = pred_1h
            mu_4h, sig_4h = pred_4h

            # Compute NLL for each horizon against its own target
            loss_15m = gaussian_nll_loss(mu_15m, sig_15m, y_15m)
            loss_1h  = gaussian_nll_loss(mu_1h,  sig_1h,  y_1h)
            loss_4h  = gaussian_nll_loss(mu_4h,  sig_4h,  y_4h)

            # Combine dynamically using uncertainty weighting
            loss = uncertainty_loss_function([loss_15m, loss_1h, loss_4h])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            epoch_loss += loss.item() * batch_size_current
            epoch_nll_15m += loss_15m.item() * batch_size_current
            epoch_nll_1h += loss_1h.item() * batch_size_current
            epoch_nll_4h += loss_4h.item() * batch_size_current
            epoch_count += batch_size_current

        avg_epoch_loss = _avg_metric(epoch_loss, epoch_count)
        avg_nll_15m = _avg_metric(epoch_nll_15m, epoch_count)
        avg_nll_1h = _avg_metric(epoch_nll_1h, epoch_count)
        avg_nll_4h = _avg_metric(epoch_nll_4h, epoch_count)
        training_losses.append(avg_epoch_loss)
        training_nll_15m.append(avg_nll_15m)
        training_nll_1h.append(avg_nll_1h)
        training_nll_4h.append(avg_nll_4h)
        tqdm.write(
            f"Epoch {epoch}/{stage1_epochs} - Total: {avg_epoch_loss:.4f} "
            f"(nll_15m={avg_nll_15m:.4f}, nll_1h={avg_nll_1h:.4f}, nll_4h={avg_nll_4h:.4f})"
        )

        if epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                total_val_nll_15m = 0.0
                total_val_nll_1h = 0.0
                total_val_nll_4h = 0.0
                total_val_loss = 0.0
                total_val_count = 0

                for batch_X_val, batch_y_val in data.val_loader:
                    batch_X_val = batch_X_val.to(device, non_blocking=True)
                    batch_y_val = batch_y_val.to(device, non_blocking=True)
                    batch_size_val = batch_X_val.shape[0]
                    y_val_15m = batch_y_val[:, 0]
                    y_val_1h  = batch_y_val[:, 1]
                    y_val_4h  = batch_y_val[:, 2]
                    error_states_val = (
                        torch.randn(batch_size_val, error_history_len, device=device, dtype=batch_X_val.dtype)
                        * stage1_error_sigma
                    )
                    val_pred_15m, val_pred_1h, val_pred_4h = model(
                        batch_X_val, error_state=error_states_val
                    )
                    val_mu_15m, val_sig_15m = val_pred_15m
                    val_mu_1h, val_sig_1h = val_pred_1h
                    val_mu_4h, val_sig_4h = val_pred_4h

                    v_loss_15m = gaussian_nll_loss(val_mu_15m, val_sig_15m, y_val_15m)
                    v_loss_1h  = gaussian_nll_loss(val_mu_1h,  val_sig_1h,  y_val_1h)
                    v_loss_4h  = gaussian_nll_loss(val_mu_4h,  val_sig_4h,  y_val_4h)
                    v_loss = uncertainty_loss_function([v_loss_15m, v_loss_1h, v_loss_4h])

                    bs = batch_X_val.shape[0]
                    total_val_nll_15m += v_loss_15m.item() * bs
                    total_val_nll_1h += v_loss_1h.item() * bs
                    total_val_nll_4h += v_loss_4h.item() * bs
                    total_val_loss += v_loss.item() * bs
                    total_val_count += bs

                val_nll_15m = _avg_metric(total_val_nll_15m, total_val_count)
                val_nll_1h = _avg_metric(total_val_nll_1h, total_val_count)
                val_nll_4h = _avg_metric(total_val_nll_4h, total_val_count)
                val_loss = _avg_metric(total_val_loss, total_val_count)
                validation_losses.append(val_loss)
                validation_nll_15m.append(val_nll_15m)
                validation_nll_1h.append(val_nll_1h)
                validation_nll_4h.append(val_nll_4h)
                validation_epochs.append(epoch)

                tqdm.write(
                    f"[Val] Epoch {epoch}/{stage1_epochs} - Total: {val_loss:.4f} "
                    f"(nll_15m={val_nll_15m:.4f}, nll_1h={val_nll_1h:.4f}, nll_4h={val_nll_4h:.4f})"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    _log(
                        f"Saved best pure model at epoch {epoch}, val_loss={best_val_loss:.4f}"
                    )

    _log("\n" + "=" * 60)
    _log("STAGE 2: SEQUENTIAL FINE-TUNING WITH REAL ERROR HISTORIES")
    _log("=" * 60)

    # Load best model from Stage 1
    _log(f"Loading best model from Stage 1: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    _log(f"Loaded checkpoint with validation loss: {best_val_loss:.4f}")

    _log("Configuring Layer-Wise Learning Rates...")
    
    base_lr = float(train_cfg["stage2"]["lr"])
    model_for_params = model
    
    # 1. Isolate the components that need to learn the new Error State rapidly
    error_params = []
    error_params += list(model_for_params.error_projection.parameters())
    error_params += list(model_for_params.pos_encoder.parameters())

    # 2. Isolate the core "Physics" engine that we want to protect
    core_params = []
    for name, param in model_for_params.named_parameters():
        if "error_projection" not in name and "pos_encoder" not in name:
            core_params.append(param)
            
    # 3. Assign a 10x smaller learning rate to the core physics engine!
    optimizer = optim.AdamW([
        {'params': core_params, 'lr': base_lr * 0.5},  # Protect the core
        {'params': error_params, 'lr': base_lr}        # Train the error state aggressively
    ], weight_decay=float(train_cfg["stage2"]["weight_decay"]))

    scheduler = None
    scheduler_cfg = train_cfg["stage2"].get("scheduler", {})
    scheduler_enabled = bool(scheduler_cfg.get("enabled", False))
    if scheduler_enabled:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        _log(
            "Stage 2 LR scheduler enabled: "
            f"factor={scheduler_cfg.get('factor', 0.5)}, "
            f"patience={scheduler_cfg.get('patience', 3)}, "
            f"min_lr={scheduler_cfg.get('min_lr', 1e-6)}"
        )
    
    stage2_epochs = total_epochs - stage1_epochs
    early_stop_enabled = bool(stage2_cfg.get("early_stop_enabled", True))
    early_stop_patience = int(stage2_cfg.get("early_stop_patience", 10))
    early_stop_min_delta = float(stage2_cfg.get("early_stop_min_delta", 0.0))
    stage2_best_val_loss = float("inf")
    stage2_no_improve_count = 0

    if early_stop_enabled:
        _log(
            "Stage 2 early stop enabled: "
            f"patience={early_stop_patience}, min_delta={early_stop_min_delta}"
        )
    _log(
        f"Stage 1 error_state sigma={stage1_error_sigma:.6f} "
        f"(empirical={empirical_error_sigma:.6f})"
    )
    _log(
        f"Stage 2 streams={parallel_streams}, grad_accum_steps={stage2_grad_accum_steps}"
    )

    X_train_stage2 = data.X_train_tensor
    y_train_stage2 = data.y_train_tensor

    N = len(X_train_stage2)

    for epoch in tqdm(
        range(1, stage2_epochs + 1),
        desc="Stage 2: Parallel Fine-tune",
        unit="epoch",
    ):
        model.train()
        epoch_loss = 0.0
        epoch_nll_15m = 0.0
        epoch_nll_1h = 0.0
        epoch_nll_4h = 0.0
        processed_train_count = 0

        # ⭐️ TEMPORAL JITTER (Random Start Point)
        # Calculate a safe maximum offset (e.g., up to 25% of a chunk's length)
        # This shifts the boundaries of all parallel chunks every epoch.
        base_chunk_len = N // parallel_streams
        start_offset = int(random() * (base_chunk_len // 4))
        
        # Recalculate how much data we have left after the random offset
        N_active = N - start_offset
        chunk_len = N_active // parallel_streams
        _log(f"Epoch {epoch}: start_offset={start_offset}, chunk_len={chunk_len}")

        # ---------------------------------------------------------
        # 2. VECTORIZED ERROR HISTORY (WITH WARM-UP PRE-ROLL)
        # ---------------------------------------------------------
        train_error_histories = torch.zeros((parallel_streams, error_history_len), device=device)
        
        # ⭐️ PRE-ROLL: Fill the buffer with actual historical errors before training
        model.eval() # Turn off dropout for true baseline errors
        with torch.no_grad():
            for w in range(error_history_len):
                # Step backwards in time. Do not wrap around with modulo,
                # because that leaks future samples into early history.
                warmup_t = -error_history_len + w
                raw_indices = [
                    start_offset + b * chunk_len + warmup_t
                    for b in range(parallel_streams)
                ]

                valid_lanes = [lane for lane, idx in enumerate(raw_indices) if idx >= 0]
                if not valid_lanes:
                    continue

                valid_indices = [raw_indices[lane] for lane in valid_lanes]
                X_seq_w = X_train_stage2[valid_indices].to(device, non_blocking=True)
                y_seq_w = y_train_stage2[valid_indices].to(device, non_blocking=True)

                # No data augmentation during warmup! We want real history.
                lane_tensor = torch.tensor(valid_lanes, device=device, dtype=torch.long)
                lane_histories = train_error_histories[lane_tensor]
                error_state_w = prepare_error_states(
                    lane_histories,
                    len(valid_lanes),
                    device,
                    X_seq_w.dtype,
                )

                (mu_w, _), _, _ = model(X_seq_w, error_state=error_state_w)
                newest_errors_w = (y_seq_w[:, 0] - mu_w).detach().unsqueeze(1)

                updated_lane_histories = torch.cat(
                    [lane_histories[:, 1:], newest_errors_w], dim=1
                )
                train_error_histories[lane_tensor] = updated_lane_histories

        # Buffer is now full of reality. Switch back to train mode!
        model.train()
        optimizer.zero_grad()

        # ---------------------------------------------------------
        # 3. MAIN TRAINING LOOP
        # ---------------------------------------------------------
        for t in range(chunk_len):
            # ⭐️ Apply the random offset to our index gathering!
            indices = [start_offset + b * chunk_len + t for b in range(parallel_streams)]

            X_seq = X_train_stage2[indices].to(device, non_blocking=True)
            y_val = y_train_stage2[indices].to(device, non_blocking=True)

            X_aug = augment_financial_data(
                X_seq, noise_std=data.feature_noise_std, mask_prob=data.feature_mask_prob
            )
            
            (mu_15m, sig_15m), (mu_1h, sig_1h), (mu_4h, sig_4h) = model(
                X_aug, error_state=train_error_histories
            )
            y_15m = y_val[:, 0]
            y_1h  = y_val[:, 1]
            y_4h  = y_val[:, 2]

            loss_15m = gaussian_nll_loss(mu_15m, sig_15m, y_15m)
            loss_1h  = gaussian_nll_loss(mu_1h,  sig_1h,  y_1h)
            loss_4h  = gaussian_nll_loss(mu_4h,  sig_4h,  y_4h)
            loss = uncertainty_loss_function([loss_15m, loss_1h, loss_4h])

            (loss / stage2_grad_accum_steps).backward()

            with torch.no_grad():
                newest_errors = (y_15m - mu_15m.detach())
                train_error_histories = torch.cat(
                    [train_error_histories[:, 1:], newest_errors.unsqueeze(1)], dim=1
                )

            bs_curr = y_15m.shape[0]
            epoch_loss += loss.item() * bs_curr
            epoch_nll_15m += loss_15m.item() * bs_curr
            epoch_nll_1h += loss_1h.item() * bs_curr
            epoch_nll_4h += loss_4h.item() * bs_curr
            processed_train_count += bs_curr

            should_step = ((t + 1) % stage2_grad_accum_steps == 0) or (t == chunk_len - 1)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        avg_epoch_loss = _avg_metric(epoch_loss, processed_train_count)
        avg_nll_15m = _avg_metric(epoch_nll_15m, processed_train_count)
        avg_nll_1h = _avg_metric(epoch_nll_1h, processed_train_count)
        avg_nll_4h = _avg_metric(epoch_nll_4h, processed_train_count)
        training_losses.append(avg_epoch_loss)
        training_nll_15m.append(avg_nll_15m)
        training_nll_1h.append(avg_nll_1h)
        training_nll_4h.append(avg_nll_4h)
        tqdm.write(
            f"Epoch {epoch}/{stage2_epochs} - Total: {avg_epoch_loss:.4f} "
            f"(nll_15m={avg_nll_15m:.4f}, nll_1h={avg_nll_1h:.4f}, nll_4h={avg_nll_4h:.4f})"
        )

        if epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                total_val_nll_15m = 0.0
                total_val_nll_1h = 0.0
                total_val_nll_4h = 0.0
                total_val_loss = 0.0
                total_val_count = 0
                val_error_history = [0.0] * error_history_len

                # -----------------------------------------------------------
                # ⭐️ PHASE 1: THE VALIDATION WARM-UP (The Sacrifice)
                # We use the first N bars ONLY to prime the autoregressive memory.
                # We DO NOT record the loss.
                # -----------------------------------------------------------
                warmup_steps = min(error_history_len, len(data.X_val_tensor) - 1)

                for i in range(warmup_steps):
                    X_seq = data.X_val_tensor[i : i + 1].to(device, non_blocking=True)
                    y_15m_seq = data.y_val_tensor[i, 0].to(device, non_blocking=True).item()

                    error_state_val = torch.tensor([val_error_history], dtype=X_seq.dtype, device=device)
                    (mu_val, _), _, _ = model(X_seq, error_state=error_state_val)

                    newest_error = y_15m_seq - mu_val[0].item()
                    val_error_history = update_error_history(val_error_history, newest_error)

                for i in range(warmup_steps, len(data.X_val_tensor)):
                    X_seq = data.X_val_tensor[i : i + 1].to(device, non_blocking=True)
                    y_row = data.y_val_tensor[i].to(device, non_blocking=True)  # shape (3,)
                    y_15m_v = y_row[0:1]          # shape (1,)
                    y_1h_v  = y_row[1:2]
                    y_4h_v  = y_row[2:3]

                    error_state_val = torch.tensor([val_error_history], dtype=X_seq.dtype, device=device)
                    (mu_15m_v, sig_15m_v), (mu_1h_v, sig_1h_v), (mu_4h_v, sig_4h_v) = model(
                        X_seq, error_state=error_state_val
                    )

                    v_loss_15m = gaussian_nll_loss(mu_15m_v, sig_15m_v, y_15m_v)
                    v_loss_1h  = gaussian_nll_loss(mu_1h_v,  sig_1h_v,  y_1h_v)
                    v_loss_4h  = gaussian_nll_loss(mu_4h_v,  sig_4h_v,  y_4h_v)
                    v_loss = uncertainty_loss_function([v_loss_15m, v_loss_1h, v_loss_4h])

                    bs = X_seq.shape[0]
                    total_val_nll_15m += v_loss_15m.item() * bs
                    total_val_nll_1h += v_loss_1h.item() * bs
                    total_val_nll_4h += v_loss_4h.item() * bs
                    total_val_loss += v_loss.item() * bs
                    total_val_count += bs

                    # Update history for the next step!
                    newest_error = y_row[0].item() - mu_15m_v[0].item()
                    val_error_history = update_error_history(val_error_history, newest_error)
                # Calculate final averages
                val_nll_15m = _avg_metric(total_val_nll_15m, total_val_count)
                val_nll_1h = _avg_metric(total_val_nll_1h, total_val_count)
                val_nll_4h = _avg_metric(total_val_nll_4h, total_val_count)
                val_loss = _avg_metric(total_val_loss, total_val_count)
                validation_losses.append(val_loss)
                validation_nll_15m.append(val_nll_15m)
                validation_nll_1h.append(val_nll_1h)
                validation_nll_4h.append(val_nll_4h)
                validation_epochs.append(stage1_epochs + epoch)

                tqdm.write(
                    f"[Val] Epoch {stage1_epochs + epoch} - Total: {val_loss:.4f} "
                    f"(nll_15m={val_nll_15m:.4f}, nll_1h={val_nll_1h:.4f}, nll_4h={val_nll_4h:.4f})"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    _log(
                        f"Saved best pure model at epoch {stage1_epochs + epoch}, "
                        f"val_loss={best_val_loss:.4f}"
                    )

                if scheduler is not None:
                    lr_before = optimizer.param_groups[0]["lr"]
                    scheduler.step(val_loss)
                    lr_after = optimizer.param_groups[0]["lr"]
                    if lr_after < lr_before:
                        _log(
                            f"Stage 2 LR reduced: {lr_before:.2e} -> {lr_after:.2e} "
                            f"at epoch {stage1_epochs + epoch}"
                        )

                if early_stop_enabled:
                    improved_stage2 = (stage2_best_val_loss - val_loss) > early_stop_min_delta
                    if improved_stage2:
                        stage2_best_val_loss = val_loss
                        stage2_no_improve_count = 0
                    else:
                        stage2_no_improve_count += 1
                        _log(
                            "Stage 2 early-stop wait "
                            f"{stage2_no_improve_count}/{early_stop_patience} "
                            f"(best={stage2_best_val_loss:.4f}, current={val_loss:.4f})"
                        )

                    if stage2_no_improve_count >= early_stop_patience:
                        _log(
                            "Stage 2 early stopping triggered at "
                            f"epoch {stage1_epochs + epoch}."
                        )
                        break

    best_model = USDCNHTransformer(
        input_size=model_input_size,
        seq_len=cfg["data"]["sequence_length"],
        error_history_len=error_history_len,
        nhead=nhead,
        num_layers=num_layers,
        d_model=d_model,
        dropout=dropout,
        patch_len=model_cfg.get("patch_len", 4),
        patch_stride=model_cfg.get("patch_stride", 2),
    ).to(device)
    best_model.load_state_dict(torch.load(model_save_path, map_location=device))
    best_model.eval()

    with torch.no_grad():
        # Collect predictions from all three heads
        test_mu_15m_chunks = []
        test_sig_15m_chunks = []
        test_mu_1h_chunks = []
        test_sig_1h_chunks = []
        test_mu_4h_chunks = []
        test_sig_4h_chunks = []

        total_test_nll_15m = 0.0
        total_test_nll_1h = 0.0
        total_test_nll_4h = 0.0
        total_test_loss = 0.0
        total_test_count = 0

        test_error_history = [0.0] * error_history_len
        warmup_steps_test = min(error_history_len, len(data.X_test_tensor) - 1)

        _log("\nWarming up Test Set autoregressive memory...")
        # -----------------------------------------------------------
        # ⭐️ TEST PHASE 1: WARM-UP (Do not record metrics)
        # -----------------------------------------------------------
        for i in range(warmup_steps_test):
            X_seq = data.X_test_tensor[i : i + 1].to(device, non_blocking=True)
            y_15m_val = data.y_test_tensor[i, 0].to(device, non_blocking=True).item()

            error_state_test = torch.tensor([test_error_history], dtype=X_seq.dtype, device=device)
            (mu_t, _), _, _ = best_model(X_seq, error_state=error_state_test)

            newest_error = y_15m_val - mu_t[0].item()
            test_error_history = update_error_history(test_error_history, newest_error)

        _log("Evaluating true Test Set performance...")
        # -----------------------------------------------------------
        # ⭐️ TEST PHASE 2: TRUE EVALUATION
        # -----------------------------------------------------------
        for i in range(warmup_steps_test, len(data.X_test_tensor)):
            X_seq = data.X_test_tensor[i : i + 1].to(device, non_blocking=True)
            y_row_t = data.y_test_tensor[i].to(device, non_blocking=True)   # shape (3,)
            y_15m_t = y_row_t[0:1]
            y_1h_t  = y_row_t[1:2]
            y_4h_t  = y_row_t[2:3]

            error_state_test = torch.tensor([test_error_history], dtype=X_seq.dtype, device=device)
            (mu_15m_t, sig_15m_t), (mu_1h_t, sig_1h_t), (mu_4h_t, sig_4h_t) = best_model(
                X_seq, error_state=error_state_test
            )

            # Collect predictions from all three heads
            test_mu_15m_chunks.append(mu_15m_t)
            test_sig_15m_chunks.append(sig_15m_t)
            test_mu_1h_chunks.append(mu_1h_t)
            test_sig_1h_chunks.append(sig_1h_t)
            test_mu_4h_chunks.append(mu_4h_t)
            test_sig_4h_chunks.append(sig_4h_t)

            t_loss_15m = gaussian_nll_loss(mu_15m_t, sig_15m_t, y_15m_t)
            t_loss_1h  = gaussian_nll_loss(mu_1h_t,  sig_1h_t,  y_1h_t)
            t_loss_4h  = gaussian_nll_loss(mu_4h_t,  sig_4h_t,  y_4h_t)
            t_loss = uncertainty_loss_function([t_loss_15m, t_loss_1h, t_loss_4h])

            bs = X_seq.shape[0]
            total_test_nll_15m += t_loss_15m.item() * bs
            total_test_nll_1h += t_loss_1h.item() * bs
            total_test_nll_4h += t_loss_4h.item() * bs
            total_test_loss += t_loss.item() * bs
            total_test_count += bs

            newest_error = y_row_t[0].item() - mu_15m_t[0].item()
            test_error_history = update_error_history(test_error_history, newest_error)

        # Concatenate predictions from all three heads
        test_mu_15m = torch.cat(test_mu_15m_chunks, dim=0)
        test_sig_15m = torch.cat(test_sig_15m_chunks, dim=0)
        test_mu_1h = torch.cat(test_mu_1h_chunks, dim=0)
        test_sig_1h = torch.cat(test_sig_1h_chunks, dim=0)
        test_mu_4h = torch.cat(test_mu_4h_chunks, dim=0)
        test_sig_4h = torch.cat(test_sig_4h_chunks, dim=0)

        test_nll_15m = _avg_metric(total_test_nll_15m, total_test_count)
        test_nll_1h = _avg_metric(total_test_nll_1h, total_test_count)
        test_nll_4h = _avg_metric(total_test_nll_4h, total_test_count)
        test_loss = _avg_metric(total_test_loss, total_test_count)

        # Get binary targets for all three horizons
        target_binary_15m = (data.y_test_tensor[warmup_steps_test:, 0] > 0.5).float().to(device)
        target_binary_1h = (data.y_test_tensor[warmup_steps_test:, 1] > 0.5).float().to(device)
        target_binary_4h = (data.y_test_tensor[warmup_steps_test:, 2] > 0.5).float().to(device)

        # Helper function to compute accuracy metrics for a given horizon
        def compute_horizon_metrics(mu, sig, target_binary, horizon_name):
            binary_predictions = (mu > 0.5).float()
            test_confidence = torch.exp(-sig)
            
            correct = (binary_predictions == target_binary).sum().item()
            accuracy = correct / len(target_binary)
            
            long_mask = target_binary == 1.0
            short_mask = target_binary == 0.0
            long_accuracy = (
                (binary_predictions[long_mask] == target_binary[long_mask]).float().mean().item()
                if long_mask.any()
                else float("nan")
            )
            short_accuracy = (
                (binary_predictions[short_mask] == target_binary[short_mask]).float().mean().item()
                if short_mask.any()
                else float("nan")
            )
            confidence_mean = test_confidence.mean().item()
            
            # Confidence-threshold metrics
            mask_55 = test_confidence >= 0.55
            mask_60 = test_confidence >= 0.60
            mask_70 = test_confidence >= 0.70
            
            cov_55 = mask_55.float().mean().item()
            cov_60 = mask_60.float().mean().item()
            cov_70 = mask_70.float().mean().item()
            
            acc_55 = (
                (binary_predictions[mask_55] == target_binary[mask_55]).float().mean().item()
                if mask_55.any()
                else float("nan")
            )
            acc_60 = (
                (binary_predictions[mask_60] == target_binary[mask_60]).float().mean().item()
                if mask_60.any()
                else float("nan")
            )
            acc_70 = (
                (binary_predictions[mask_70] == target_binary[mask_70]).float().mean().item()
                if mask_70.any()
                else float("nan")
            )
            
            return {
                "accuracy": accuracy,
                "long_accuracy": long_accuracy,
                "short_accuracy": short_accuracy,
                "confidence_mean": confidence_mean,
                "accuracy_conf_55": acc_55,
                "coverage_conf_55": cov_55,
                "accuracy_conf_60": acc_60,
                "coverage_conf_60": cov_60,
                "accuracy_conf_70": acc_70,
                "coverage_conf_70": cov_70,
            }
        
        # Compute metrics for all three horizons
        metrics_15m = compute_horizon_metrics(test_mu_15m, test_sig_15m, target_binary_15m, "15m")
        metrics_1h = compute_horizon_metrics(test_mu_1h, test_sig_1h, target_binary_1h, "1h")
        metrics_4h = compute_horizon_metrics(test_mu_4h, test_sig_4h, target_binary_4h, "4h")
        
        _log(f"\nTest Loss: {test_loss:.4f} (nll_15m={test_nll_15m:.4f}, nll_1h={test_nll_1h:.4f}, nll_4h={test_nll_4h:.4f})")
        _log(f"\n15m Accuracy: {metrics_15m['accuracy']:.4f}, Long: {metrics_15m['long_accuracy']:.4f}, Short: {metrics_15m['short_accuracy']:.4f}")
        _log(f"1h  Accuracy: {metrics_1h['accuracy']:.4f}, Long: {metrics_1h['long_accuracy']:.4f}, Short: {metrics_1h['short_accuracy']:.4f}")
        _log(f"4h  Accuracy: {metrics_4h['accuracy']:.4f}, Long: {metrics_4h['long_accuracy']:.4f}, Short: {metrics_4h['short_accuracy']:.4f}")
        
        # Store for summary and CSV
        test_accuracy = metrics_15m["accuracy"]
        test_long_accuracy = metrics_15m["long_accuracy"]
        test_short_accuracy = metrics_15m["short_accuracy"]
        test_confidence_mean = metrics_15m["confidence_mean"]
        test_accuracy_conf_55 = metrics_15m["accuracy_conf_55"]
        test_coverage_conf_55 = metrics_15m["coverage_conf_55"]
        test_accuracy_conf_60 = metrics_15m["accuracy_conf_60"]
        test_coverage_conf_60 = metrics_15m["coverage_conf_60"]
        test_accuracy_conf_70 = metrics_15m["accuracy_conf_70"]
        test_coverage_conf_70 = metrics_15m["coverage_conf_70"]

    metrics_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(training_losses) + 1)),
            "total_loss": training_losses,
            "nll_15m": training_nll_15m,
            "nll_1h": training_nll_1h,
            "nll_4h": training_nll_4h,
        }
    )
    training_metrics_path = run_paths.metrics / "training_metrics.csv"
    metrics_df.to_csv(training_metrics_path, index=False)

    val_metrics_df = pd.DataFrame(
        {
            "epoch": validation_epochs,
            "total_loss": validation_losses,
            "nll_15m": validation_nll_15m,
            "nll_1h": validation_nll_1h,
            "nll_4h": validation_nll_4h,
        }
    )
    val_metrics_path = run_paths.metrics / "validation_metrics.csv"
    val_metrics_df.to_csv(val_metrics_path, index=False)

    # Flatten all metrics into a single row with horizon prefix
    test_metrics_dict = {
        "total_loss": [test_loss],
        "nll_15m": [test_nll_15m],
        "nll_1h": [test_nll_1h],
        "nll_4h": [test_nll_4h],
    }
    
    for horizon, metrics in [("15m", metrics_15m), ("1h", metrics_1h), ("4h", metrics_4h)]:
        test_metrics_dict[f"accuracy_{horizon}"] = [metrics["accuracy"]]
        test_metrics_dict[f"long_accuracy_{horizon}"] = [metrics["long_accuracy"]]
        test_metrics_dict[f"short_accuracy_{horizon}"] = [metrics["short_accuracy"]]
        test_metrics_dict[f"confidence_mean_{horizon}"] = [metrics["confidence_mean"]]
        test_metrics_dict[f"accuracy_conf_55_{horizon}"] = [metrics["accuracy_conf_55"]]
        test_metrics_dict[f"coverage_conf_55_{horizon}"] = [metrics["coverage_conf_55"]]
        test_metrics_dict[f"accuracy_conf_60_{horizon}"] = [metrics["accuracy_conf_60"]]
        test_metrics_dict[f"coverage_conf_60_{horizon}"] = [metrics["coverage_conf_60"]]
        test_metrics_dict[f"accuracy_conf_70_{horizon}"] = [metrics["accuracy_conf_70"]]
        test_metrics_dict[f"coverage_conf_70_{horizon}"] = [metrics["coverage_conf_70"]]
    
    test_metrics_df = pd.DataFrame(test_metrics_dict)
    test_metrics_path = run_paths.metrics / "test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False)

    # Create a 2x2 figure: Loss curves + accuracy metrics for 3 horizons
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training and validation loss
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(
        range(1, len(training_losses) + 1),
        training_losses,
        label="Training Loss",
        linewidth=2,
        alpha=0.8,
    )
    ax0.plot(
        validation_epochs,
        validation_losses,
        label="Validation Loss",
        marker="o",
        linewidth=2,
        markersize=5,
        alpha=0.8,
    )
    ax0.set_xlabel("Epoch", fontsize=11)
    ax0.set_ylabel("Loss", fontsize=11)
    ax0.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy across three horizons
    ax1 = fig.add_subplot(gs[0, 1])
    horizons = ["15m", "1h", "4h"]
    accuracies = [metrics_15m["accuracy"], metrics_1h["accuracy"], metrics_4h["accuracy"]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax1.bar(horizons, accuracies, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Test Accuracy by Horizon", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f"{acc:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Plot 3: Long vs Short accuracy
    ax2 = fig.add_subplot(gs[1, 0])
    x = range(len(horizons))
    width = 0.35
    long_accs = [metrics_15m["long_accuracy"], metrics_1h["long_accuracy"], metrics_4h["long_accuracy"]]
    short_accs = [metrics_15m["short_accuracy"], metrics_1h["short_accuracy"], metrics_4h["short_accuracy"]]
    bars1 = ax2.bar([i - width/2 for i in x], long_accs, width, label="Long", alpha=0.7, edgecolor="black")
    bars2 = ax2.bar([i + width/2 for i in x], short_accs, width, label="Short", alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Long vs Short Accuracy by Horizon", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Summary text
    ax3 = fig.add_subplot(gs[1, 1])
    best_val_text = f"{min(validation_losses):.4f}" if validation_losses else "N/A"
    summary_text = f"""Test Metrics Summary

Loss Metrics:
  Total Loss: {test_loss:.4f}
  NLL-15m: {test_nll_15m:.4f}
  NLL-1h: {test_nll_1h:.4f}
  NLL-4h: {test_nll_4h:.4f}

15m Metrics:
  Accuracy: {metrics_15m['accuracy']:.4f}
  Mean Confidence: {metrics_15m['confidence_mean']:.4f}

1h Metrics:
  Accuracy: {metrics_1h['accuracy']:.4f}
  Mean Confidence: {metrics_1h['confidence_mean']:.4f}

4h Metrics:
  Accuracy: {metrics_4h['accuracy']:.4f}
  Mean Confidence: {metrics_4h['confidence_mean']:.4f}

Best Validation Loss: {best_val_text}"""
    
    ax3.text(
        0.1, 0.5,
        summary_text,
        ha="left",
        va="center",
        fontsize=10,
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7, "pad": 1},
    )
    ax3.axis("off")
    
    plt.suptitle("Model Training and Test Results", fontsize=14, fontweight="bold", y=0.995)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = run_paths.plots / f"training_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model_path": model_save_path,
        "training_metrics_path": str(training_metrics_path),
        "validation_metrics_path": str(val_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "plot_path": str(plot_path),
        "test_accuracy": test_accuracy,
        "test_long_accuracy": test_long_accuracy,
        "test_short_accuracy": test_short_accuracy,
        "test_confidence_mean": test_confidence_mean,
        "test_accuracy_conf_55": test_accuracy_conf_55,
        "test_coverage_conf_55": test_coverage_conf_55,
        "test_accuracy_conf_60": test_accuracy_conf_60,
        "test_coverage_conf_60": test_coverage_conf_60,
        "test_accuracy_conf_70": test_accuracy_conf_70,
        "test_coverage_conf_70": test_coverage_conf_70,
    }

    _log(f"Saved training metrics to: {training_metrics_path}")
    _log(f"Saved validation metrics to: {val_metrics_path}")
    _log(f"Saved test metrics to: {test_metrics_path}")
    _log(f"Saved plot to: {plot_path}")
    return summary
