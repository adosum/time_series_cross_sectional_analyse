from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import model_output_path
from .helpers import (
    augment_financial_data,
    compute_combined_confidence,
    compute_trend_strength,
    generate_random_error_history,
    prepare_error_states,
    update_error_history,
)
from .model import PBTAgent


def _validate_pbt_config(cfg):
    pbt_cfg = cfg["pbt"]
    if not (0.0 <= pbt_cfg["dropout_min"] <= pbt_cfg["dropout_max"] <= 1.0):
        raise ValueError("Dropout range must satisfy 0 <= min <= max <= 1")
    if not (0.0 < pbt_cfg["lr_min"] <= pbt_cfg["lr_max"]):
        raise ValueError("Learning rate range must satisfy 0 < min <= max")
    if not (0.0 <= pbt_cfg["wd_min"] <= pbt_cfg["wd_max"]):
        raise ValueError("Weight decay range must satisfy 0 <= min <= max")
    if not (0.0 <= pbt_cfg["weight_copy_ratio"] <= 1.0):
        raise ValueError("PBT_WEIGHT_COPY_RATIO must satisfy 0 <= ratio <= 1")
    sample_ratio = float(pbt_cfg.get("stage2_sample_ratio", 1.0))
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("pbt.stage2_sample_ratio must satisfy 0 < ratio <= 1")


def run_pbt_training(cfg, run_paths, data, device):
    _validate_pbt_config(cfg)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    pbt_cfg = cfg["pbt"]

    error_history_len = model_cfg["error_history_len"]
    model_input_size = len(data.feature_cols)  # Features only, no error history appended

    d_model = model_cfg["d_model"]
    nhead = model_cfg["nhead"]
    num_layers = model_cfg["num_layers"]

    abs_w = train_cfg["abs_loss_weight"]
    vol_w = train_cfg["volatility_loss_weight"]
    trend_w = train_cfg["trend_loss_weight"]
    conf_w = train_cfg["confidence_loss_weight"]

    grad_clip_max_norm = train_cfg["grad_clip_max_norm"]
    stage1_epochs = train_cfg["stage1_epochs"]
    total_epochs = train_cfg["epochs"]
    stage2_grad_accum_steps = train_cfg["stage2"]["grad_accum_steps"]

    loss_function = nn.BCEWithLogitsLoss()
    abs_loss_function = nn.MSELoss()
    volatility_loss_function = nn.MSELoss()
    trend_loss_function = nn.MSELoss()

    pop_size = pbt_cfg["population_size"]
    generations = pbt_cfg["generations"]
    epochs_per_gen = pbt_cfg["epochs_per_gen"]

    print(
        f"Configured PBT: pop={pop_size}, gens={generations}, epochs/gen={epochs_per_gen}"
    )

    population = []
    lr_values = np.linspace(pbt_cfg["lr_min"], pbt_cfg["lr_max"], pop_size).tolist()
    dropout_values = np.linspace(
        pbt_cfg["dropout_min"], pbt_cfg["dropout_max"], pop_size
    ).tolist()
    wd_values = np.linspace(pbt_cfg["wd_min"], pbt_cfg["wd_max"], pop_size).tolist()

    random.shuffle(lr_values)
    random.shuffle(dropout_values)
    random.shuffle(wd_values)

    for i in range(pop_size):
        agent = PBTAgent(
            input_size=model_input_size,
            seq_len=cfg["data"]["sequence_length"],
            error_history_len=error_history_len,
            lr=float(lr_values[i]),
            dropout=float(dropout_values[i]),
            weight_decay=float(wd_values[i]),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            device=device,
        )
        population.append(agent)

    train_batches = list(data.train_loader)

    pbt_stage1_ep = max(1, int(round(epochs_per_gen * stage1_epochs / total_epochs)))
    pbt_stage2_ep = epochs_per_gen - pbt_stage1_ep

    latest_path = model_output_path(
        run_paths, pbt_cfg.get("latest_model_save_path", "USDCNH_PBT_EMA_latest.pth")
    )
    best_path = model_output_path(
        run_paths, pbt_cfg.get("best_model_save_path", "USDCNH_PBT_EMA_best.pth")
    )
    production_path = str((run_paths.models / "USDCNH_PBT_EMA_Production.pth").resolve())

    print("\nStarting Continuous PBT Evolution...")
    best_pbt_fitness = float("-inf")

    pbar = tqdm(range(1, generations + 1), desc="PBT Evolution", unit="gen")

    for gen in pbar:
        total_train_samples = len(data.X_train_tensor)
        stage2_window = max(
            1, int(round(total_train_samples * float(pbt_cfg.get("stage2_sample_ratio", 1.0))))
        )
        stage2_start = random.randint(0, total_train_samples - 1)

        agent_pbar = tqdm(population, desc=f"Gen {gen} Train", unit="agent", leave=False)
        for agent in agent_pbar:
            agent.active_model.train()

            epoch_pbar = tqdm(range(1, pbt_stage1_ep + 1), desc="S1:Batch", unit="ep", leave=False)
            for _ in epoch_pbar:
                for batch_X, batch_y in train_batches:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    batch_size_current = batch_X.shape[0]
                    train_error_histories = [
                        generate_random_error_history(error_history_len)
                        for _ in range(batch_size_current)
                    ]

                    agent.optimizer.zero_grad(set_to_none=True)
                    batch_X_aug = augment_financial_data(
                        batch_X,
                        noise_std=data.feature_noise_std,
                        mask_prob=data.feature_mask_prob,
                    )
                    error_states = prepare_error_states(
                        train_error_histories,
                        batch_size_current,
                        device,
                        batch_X_aug.dtype
                    )

                    abs_logits, direction_logits, volatility_pred, trend_pred, confidence_logits = agent.active_model(
                        batch_X_aug, error_state=error_states
                    )

                    abs_target = torch.abs(batch_y)
                    abs_loss = abs_loss_function(torch.relu(abs_logits), abs_target)
                    direction_target = (batch_y > 0.5).float()
                    direction_loss = loss_function(direction_logits, direction_target)
                    direction_prob = torch.sigmoid(direction_logits)
                    realized_error = batch_y - direction_prob
                    volatility_target = torch.abs(realized_error)
                    vol_loss = volatility_loss_function(
                        torch.abs(volatility_pred), volatility_target
                    )
                    trend_target = compute_trend_strength(batch_X, data.trend_source_feature_idx)
                    trend_loss = trend_loss_function(torch.relu(trend_pred), trend_target)
                    conf_target = (
                        (direction_prob.detach() >= 0.5).float() == direction_target
                    ).float()
                    conf_loss = loss_function(confidence_logits, conf_target)
                    loss = (
                        direction_loss
                        + abs_w * abs_loss
                        + vol_w * vol_loss
                        + trend_w * trend_loss
                        + conf_w * conf_loss
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.active_model.parameters(), max_norm=grad_clip_max_norm
                    )
                    agent.optimizer.step()

                agent.update_ema(decay=0.80)
            epoch_pbar.close()

            if pbt_stage2_ep > 0:
                pbt_s2_lr = agent.lr * pbt_cfg["stage2_lr_factor"]
                stage2_optimizer = optim.AdamW(
                    agent.active_model.parameters(),
                    lr=pbt_s2_lr,
                    weight_decay=agent.weight_decay,
                )
                epoch_pbar = tqdm(range(1, pbt_stage2_ep + 1), desc="S2:Seq", unit="ep", leave=False)
                for _ in epoch_pbar:
                    B = max(1, min(stage2_grad_accum_steps, stage2_window))
                    chunk_len = max(1, stage2_window // B)
                    train_error_histories = torch.zeros((B, error_history_len), device=device)
                    stage2_optimizer.zero_grad()

                    for t in range(chunk_len):
                        indices = [
                            (stage2_start + b * chunk_len + t) % total_train_samples
                            for b in range(B)
                        ]
                        X_seq = data.X_train_tensor[indices].to(device, non_blocking=True)
                        y_seq = data.y_train_tensor[indices].to(device, non_blocking=True)

                        X_aug = augment_financial_data(
                            X_seq,
                            noise_std=data.feature_noise_std,
                            mask_prob=data.feature_mask_prob,
                        )

                        abs_logits, direction_logits, volatility_pred, trend_pred, confidence_logits = agent.active_model(
                            X_aug, error_state=train_error_histories
                        )

                        abs_target = torch.abs(y_seq)
                        abs_loss = abs_loss_function(torch.relu(abs_logits), abs_target)
                        direction_loss = loss_function(
                            direction_logits, (y_seq > 0.5).float()
                        )
                        direction_prob = torch.sigmoid(direction_logits)
                        realized_error = y_seq - direction_prob
                        vol_loss = volatility_loss_function(
                            torch.abs(volatility_pred), torch.abs(realized_error)
                        )
                        trend_target = compute_trend_strength(X_seq, data.trend_source_feature_idx)
                        trend_loss = trend_loss_function(torch.relu(trend_pred), trend_target)
                        conf_target = (
                            (direction_prob.detach() >= 0.5).float()
                            == (y_seq > 0.5).float()
                        ).float()
                        conf_loss = loss_function(confidence_logits, conf_target)
                        loss = (
                            direction_loss
                            + abs_w * abs_loss
                            + vol_w * vol_loss
                            + trend_w * trend_loss
                            + conf_w * conf_loss
                        )

                        loss.backward()

                        with torch.no_grad():
                            prob = torch.sigmoid(direction_logits)
                            binary_target = (y_seq > 0.5).float()
                            newest_errors = (binary_target - prob).squeeze(-1)
                            train_error_histories = torch.cat(
                                [train_error_histories[:, 1:], newest_errors.unsqueeze(1)],
                                dim=1,
                            )

                        torch.nn.utils.clip_grad_norm_(
                            agent.active_model.parameters(), max_norm=grad_clip_max_norm
                        )
                        stage2_optimizer.step()
                        stage2_optimizer.zero_grad()

                    agent.update_ema(decay=0.80)
                epoch_pbar.close()

        agent_base_losses = []
        agent_raw_predictions = []

        for agent in tqdm(population, desc=f"Gen {gen} Eval", unit="agent", leave=False):
            agent.ema_model.eval()

            total_val_loss = 0.0
            total_val_count = 0
            agent_probs = []

            with torch.no_grad():
                val_total = len(data.X_val_tensor)
                B_val = max(1, min(stage2_grad_accum_steps, val_total))
                val_chunk_len = max(1, val_total // B_val)
                val_error_histories = torch.zeros((B_val, error_history_len), device=device)

                for t in range(val_chunk_len):
                    indices = [b * val_chunk_len + t for b in range(B_val)]
                    X_seq = data.X_val_tensor[indices].to(device, non_blocking=True)
                    y_seq = data.y_val_tensor[indices].to(device, non_blocking=True)

                    val_abs, val_direction, val_vol, val_trend, conf_logits = agent.ema_model(
                        X_seq, error_state=val_error_histories
                    )

                    val_dir_prob = torch.sigmoid(val_direction)
                    agent_probs.extend(val_dir_prob.squeeze(-1).detach().cpu().tolist())

                    batch_abs_target = torch.abs(y_seq)
                    batch_abs = abs_loss_function(torch.relu(val_abs), batch_abs_target)
                    batch_direction = loss_function(val_direction, (y_seq > 0.5).float())

                    batch_realized_err = y_seq - val_dir_prob
                    batch_vol = volatility_loss_function(
                        torch.abs(val_vol), torch.abs(batch_realized_err)
                    )

                    batch_trend_target = compute_trend_strength(X_seq, data.trend_source_feature_idx)
                    batch_trend = trend_loss_function(
                        torch.relu(val_trend), batch_trend_target
                    )
                    conf_target = (
                        (val_dir_prob.detach() >= 0.5).float() == (y_seq > 0.5).float()
                    ).float()
                    batch_conf = loss_function(conf_logits, conf_target)

                    combined_loss = (
                        batch_direction
                        + abs_w * batch_abs
                        + vol_w * batch_vol
                        + trend_w * batch_trend
                        + conf_w * batch_conf
                    )

                    bs = X_seq.shape[0]
                    total_val_loss += combined_loss.item() * bs
                    total_val_count += bs

                    newest_errors = ((y_seq > 0.5).float() - val_dir_prob).squeeze(-1)
                    val_error_histories = torch.cat(
                        [val_error_histories[:, 1:], newest_errors.unsqueeze(1)], dim=1
                    )

            avg_val_loss = total_val_loss / total_val_count
            agent_base_losses.append(avg_val_loss)
            agent_raw_predictions.append(agent_probs)

        all_preds_tensor = torch.tensor(agent_raw_predictions, dtype=torch.float32, device=device)
        centered_preds = all_preds_tensor - 0.5
        consensus_preds = centered_preds.mean(dim=0)

        diversity_penalty_weight = 0.05

        for idx, agent in enumerate(population):
            similarity = torch.nn.functional.cosine_similarity(
                centered_preds[idx].unsqueeze(0), consensus_preds.unsqueeze(0)
            ).item()
            correlation_penalty = max(0.0, similarity) * diversity_penalty_weight
            adjusted_loss = agent_base_losses[idx] + correlation_penalty
            agent.fitness = 1.0 / (adjusted_loss + 1e-8)

        population.sort(key=lambda x: x.fitness, reverse=True)
        best_agent = population[0]

        torch.save(
            {
                "generation": gen,
                "fitness": best_agent.fitness,
                "model_state_dict": best_agent.ema_model.state_dict(),
                "lr": best_agent.lr,
                "dropout": best_agent.dropout,
                "weight_decay": best_agent.weight_decay,
            },
            latest_path,
        )

        if best_agent.fitness > best_pbt_fitness:
            best_pbt_fitness = best_agent.fitness
            torch.save(
                {
                    "generation": gen,
                    "fitness": best_agent.fitness,
                    "model_state_dict": best_agent.ema_model.state_dict(),
                    "lr": best_agent.lr,
                    "dropout": best_agent.dropout,
                    "weight_decay": best_agent.weight_decay,
                },
                best_path,
            )

        pbar.set_postfix(
            {
                "Val_Fit": f"{best_agent.fitness:.4f}",
                "LR": f"{best_agent.lr:.5f}",
                "Drop": f"{best_agent.dropout:.2f}",
            }
        )

        if gen < generations:
            half_pop = pop_size // 2
            for i in range(half_pop, pop_size):
                winner = population[i - half_pop]
                loser = population[i]

                loser.active_model.load_state_dict(winner.ema_model.state_dict())
                loser.ema_model.load_state_dict(winner.ema_model.state_dict())

                loser.lr = winner.lr * random.choice([0.8, 1.2])
                loser.dropout = max(
                    pbt_cfg["dropout_min"],
                    min(
                        pbt_cfg["dropout_max"],
                        winner.dropout * random.choice([0.8, 1.2]),
                    ),
                )
                loser.weight_decay = max(
                    pbt_cfg["wd_min"],
                    min(
                        pbt_cfg["wd_max"],
                        winner.weight_decay * random.choice([0.8, 1.2]),
                    ),
                )

                loser.optimizer = optim.AdamW(
                    loser.active_model.parameters(),
                    lr=loser.lr,
                    weight_decay=loser.weight_decay,
                )

    print("\n========================================")
    print("PBT Training Complete. Evaluating Ultimate Master Model...")
    print("========================================")

    ultimate_model = population[0].ema_model

    ultimate_model.eval()
    with torch.no_grad():
        test_error_history = [0.0] * error_history_len
        test_abs_chunks = []
        test_direction_chunks = []
        test_vol_chunks = []

        total_test_abs = 0.0
        total_test_direction = 0.0
        total_test_vol = 0.0
        total_test_trend = 0.0
        total_test_count = 0

        # Sequential test pass preserves true temporal error-history continuity.
        for i in range(len(data.X_test_tensor)):
            X_seq = data.X_test_tensor[i : i + 1].to(device, non_blocking=True)
            y_seq = data.y_test_tensor[i : i + 1].to(device, non_blocking=True)

            error_state_test = torch.tensor([test_error_history], dtype=X_seq.dtype, device=device)
            test_abs, test_direction, test_vol, test_trend, _ = ultimate_model(
                X_seq, error_state=error_state_test
            )

            test_abs_chunks.append(test_abs)
            test_direction_chunks.append(test_direction)
            test_vol_chunks.append(test_vol)

            batch_abs_target = torch.abs(y_seq)
            batch_abs_loss = abs_loss_function(torch.relu(test_abs), batch_abs_target)

            batch_direction_target = (y_seq > 0.5).float()
            batch_direction_loss = loss_function(test_direction, batch_direction_target)

            batch_direction_prob = torch.sigmoid(test_direction)
            batch_realized_error = y_seq - batch_direction_prob
            batch_vol_target = torch.abs(batch_realized_error)
            batch_vol_loss = volatility_loss_function(torch.abs(test_vol), batch_vol_target)

            batch_trend_target = compute_trend_strength(X_seq, data.trend_source_feature_idx)
            batch_trend_loss = trend_loss_function(torch.relu(test_trend), batch_trend_target)

            total_test_abs += batch_abs_loss.item()
            total_test_direction += batch_direction_loss.item()
            total_test_vol += batch_vol_loss.item()
            total_test_trend += batch_trend_loss.item()
            total_test_count += 1

            prob = torch.sigmoid(test_direction[0]).item()
            binary_target = (y_seq[0] > 0.5).float().item()
            newest_error = binary_target - prob
            test_error_history = update_error_history(test_error_history, newest_error)

        test_abs_predictions = torch.cat(test_abs_chunks, dim=0)
        test_direction_predictions = torch.cat(test_direction_chunks, dim=0)
        test_vol_predictions = torch.cat(test_vol_chunks, dim=0)

        test_direction_prob = torch.sigmoid(test_direction_predictions)
        test_confidence = compute_combined_confidence(
            torch.relu(test_abs_predictions), test_direction_prob, test_vol_predictions
        )
        binary_predictions = (test_direction_prob >= 0.5).float()
        target_binary = (data.y_test_tensor > 0.5).float().to(device)

        correct = (binary_predictions == target_binary).sum().item()
        accuracy = correct / len(data.y_test_tensor)
        long_mask = target_binary == 1.0
        short_mask = target_binary == 0.0
        accuracy_long = (
            (binary_predictions[long_mask] == target_binary[long_mask]).float().mean().item()
            if long_mask.any()
            else float("nan")
        )
        accuracy_short = (
            (binary_predictions[short_mask] == target_binary[short_mask]).float().mean().item()
            if short_mask.any()
            else float("nan")
        )

        test_abs_loss = total_test_abs / total_test_count
        test_direction_loss = total_test_direction / total_test_count
        test_vol_loss = total_test_vol / total_test_count
        test_trend_loss = total_test_trend / total_test_count
        test_loss = (
            test_direction_loss
            + abs_w * test_abs_loss
            + vol_w * test_vol_loss
            + trend_w * test_trend_loss
        )

        mask_60 = test_confidence >= 0.60
        cov_60 = mask_60.float().mean().item()
        if mask_60.any():
            acc_60 = (binary_predictions[mask_60] == target_binary[mask_60]).float().mean().item()
        else:
            acc_60 = float("nan")

    print(
        f"Ultimate Test Loss: {test_loss:.4f} (abs={test_abs_loss:.4f}, "
        f"dir={test_direction_loss:.4f}, vol={test_vol_loss:.4f}, trend={test_trend_loss:.4f})"
    )
    print(f"Ultimate Test Accuracy:   {accuracy * 100:.2f}%")
    print(f"Ultimate Long Accuracy:   {accuracy_long * 100:.2f}%")
    print(f"Ultimate Short Accuracy:  {accuracy_short * 100:.2f}%")
    print(f"Ultimate Mean Confidence: {test_confidence.mean().item():.4f}")
    print(f"Ultimate conf>=0.60: acc={acc_60:.4f}, cov={cov_60:.4f}")

    torch.save(ultimate_model.state_dict(), production_path)
    print(f"Master Model saved to '{production_path}'.")
    print(f"Latest PBT checkpoint saved to '{latest_path}'.")
    print(f"Best PBT checkpoint saved to '{best_path}'.")

    return {
        "latest_path": latest_path,
        "best_path": best_path,
        "production_path": production_path,
        "test_accuracy": accuracy,
        "test_accuracy_long": accuracy_long,
        "test_accuracy_short": accuracy_short,
        "test_confidence_mean": test_confidence.mean().item(),
    }
