import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import trange


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def normalize_target_to_return(y: np.ndarray) -> np.ndarray:
    y = y.reshape(-1).astype(np.float32)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))

    # Binary-like labels are mapped to [-1, 1] signed returns.
    if y_min >= 0.0 and y_max <= 1.0:
        return (y - 0.5) * 2.0

    # If already return-like, keep as is and softly bound outliers.
    if y_min >= -3.0 and y_max <= 3.0:
        return np.tanh(y)

    # Fallback for unbounded targets.
    return np.tanh(y)


def build_rl_reward_series(y_seq: np.ndarray, rl_cfg: dict) -> np.ndarray:
    mode = str(rl_cfg.get("reward_mode", "target_scaled")).strip().lower()

    if mode == "target_scaled":
        return normalize_target_to_return(y_seq)

    if mode == "next_log_return":
        rewards = y_seq.reshape(-1).astype(np.float32)
        clip_abs = rl_cfg.get("reward_clip_abs", None)
        if clip_abs is not None:
            clip_abs = float(clip_abs)
            if clip_abs > 0:
                rewards = np.clip(rewards, -clip_abs, clip_abs)
        reward_scale = float(rl_cfg.get("reward_scale", 1.0))
        rewards = rewards * reward_scale
        return rewards

    raise ValueError(
        f"Unsupported rl.reward_mode='{mode}'. Supported: target_scaled, next_log_return"
    )


class TradingSequenceEnv:
    def __init__(
        self,
        base_sequences: np.ndarray,
        returns: np.ndarray,
        error_history_len: int,
        trading_cost: float,
        spread_cost: float,
        slippage_cost: float,
        drawdown_penalty_coef: float,
        position_values: list,
        initial_capital: float,
        max_episode_steps: int,
        random_start: bool,
    ):
        if len(base_sequences) != len(returns):
            raise ValueError("base_sequences and returns must have the same length")
        if len(base_sequences) < 2:
            raise ValueError("Need at least 2 sequence samples for RL environment")

        self.base_sequences = base_sequences
        self.returns = returns
        self.error_history_len = error_history_len
        self.trading_cost = float(trading_cost)
        self.spread_cost = float(spread_cost)
        self.slippage_cost = float(slippage_cost)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)

        self.position_values = [float(p) for p in position_values]
        if len(self.position_values) < 2:
            raise ValueError("position_values must contain at least 2 actions")

        self.initial_capital = float(initial_capital)
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be > 0")

        self.max_episode_steps = int(max_episode_steps)
        self.random_start = bool(random_start)

        self.idx = 0
        self.steps = 0
        self.position = 0.0
        self.error_history = [0.0] * self.error_history_len
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.drawdown = 0.0

    @property
    def action_dim(self):
        return len(self.position_values)

    @property
    def observation_shape(self):
        seq_len, n_features = self.base_sequences.shape[1], self.base_sequences.shape[2]
        return seq_len, n_features + self.error_history_len

    def _get_observation(self) -> np.ndarray:
        seq = self.base_sequences[self.idx]
        err = np.repeat(np.asarray(self.error_history, dtype=np.float32)[None, :], seq.shape[0], axis=0)
        return np.concatenate([seq, err], axis=1).astype(np.float32)

    def reset(self):
        self.position = 0.0
        self.error_history = [0.0] * self.error_history_len
        self.steps = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.drawdown = 0.0

        if self.random_start:
            max_start = max(0, len(self.base_sequences) - self.max_episode_steps - 1)
            self.idx = np.random.randint(0, max_start + 1)
        else:
            self.idx = 0

        return self._get_observation(), {}

    def step(self, action: int):
        if action < 0 or action >= len(self.position_values):
            raise ValueError(f"Invalid action {action}")

        new_position = self.position_values[int(action)]
        period_return = float(self.returns[self.idx])
        turnover = abs(new_position - self.position)

        step_notional = self.equity
        gross_pnl = step_notional * new_position * period_return

        base_cost = step_notional * self.trading_cost * turnover
        spread_component = step_notional * self.spread_cost * turnover
        slippage_component = step_notional * self.slippage_cost * turnover * (1.0 + abs(period_return))
        execution_cost = base_cost + spread_component + slippage_component

        net_step_pnl = gross_pnl - execution_cost
        self.equity = max(1e-8, self.equity + net_step_pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        new_drawdown = max(0.0, 1.0 - (self.equity / self.peak_equity))
        drawdown_increase = max(0.0, new_drawdown - self.drawdown)
        # Drawdown penalty: fraction of capital (same scale as normalized reward)
        drawdown_penalty = self.drawdown_penalty_coef * drawdown_increase

        # Normalize reward by initial capital so PPO sees small-magnitude returns,
        # while equity tracking above uses real dollar values.
        reward = (net_step_pnl / self.initial_capital) - drawdown_penalty
        self.drawdown = new_drawdown

        target_dir = 1.0 if period_return > 0.0 else 0.0
        action_prob_proxy = 1.0 if new_position > 0.0 else (0.0 if new_position < 0.0 else 0.5)
        prediction_error = target_dir - action_prob_proxy
        self.error_history = self.error_history[1:] + [float(prediction_error)]

        self.position = new_position
        self.steps += 1

        terminated = self.idx >= (len(self.base_sequences) - 2)
        truncated = self.steps >= self.max_episode_steps

        if not (terminated or truncated):
            self.idx += 1

        obs = self._get_observation()
        info = {
            "period_return": period_return,
            "position": new_position,
            "turnover": turnover,
            "gross_pnl": gross_pnl,
            "execution_cost": execution_cost,
            "net_step_pnl": net_step_pnl,
            "equity": self.equity,
            "drawdown": self.drawdown,
            "drawdown_penalty": drawdown_penalty,
        }
        return obs, reward, terminated, truncated, info


class TransformerActorCritic(nn.Module):
    def __init__(self, input_size, seq_len, d_model, nhead, num_layers, dropout, action_dim):
        super().__init__()
        self.feature_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 16)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.policy_head = nn.Linear(16, action_dim)
        self.value_head = nn.Linear(16, 1)

    def forward(self, x):
        x = self.feature_projection(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        last_step_output = x[:, -1, :]

        shared = self.relu(self.fc1(last_step_output))
        shared = self.dropout_layer(shared)

        policy_logits = self.policy_head(shared)
        state_value = self.value_head(shared).squeeze(-1)
        return policy_logits, state_value


class PPOTrainer:
    def __init__(self, model: TransformerActorCritic, device: torch.device, cfg: dict):
        self.model = model
        self.device = device

        self.gamma = float(cfg["gamma"])
        self.gae_lambda = float(cfg["gae_lambda"])
        self.clip_range = float(cfg["clip_range"])
        self.entropy_coef = float(cfg["entropy_coef"])
        self.value_coef = float(cfg["value_coef"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.rollout_steps = int(cfg["rollout_steps"])
        self.update_epochs = int(cfg["update_epochs"])
        self.minibatch_size = int(cfg["minibatch_size"])

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def collect_rollout(self, env: TradingSequenceEnv, initial_obs: np.ndarray):
        obs = initial_obs
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        self.model.eval()
        for _ in range(self.rollout_steps):
            obs_t = self._obs_to_tensor(obs)
            with torch.no_grad():
                logits, value = self.model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(int(action.item()))
            logp_buf.append(float(log_prob.item()))
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(float(value.item()))

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            next_value = float(self.model(self._obs_to_tensor(obs))[1].item())

        return {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(act_buf, dtype=np.int64),
            "old_log_probs": np.asarray(logp_buf, dtype=np.float32),
            "rewards": np.asarray(rew_buf, dtype=np.float32),
            "dones": np.asarray(done_buf, dtype=np.float32),
            "values": np.asarray(val_buf, dtype=np.float32),
            "next_value": next_value,
            "next_obs": obs,
            "episode_return_mean": float(np.mean(rew_buf)),
        }

    def _compute_gae(self, rewards, dones, values, next_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: dict):
        advantages, returns = self._compute_gae(
            rollout["rewards"], rollout["dones"], rollout["values"], rollout["next_value"]
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = RolloutBatch(
            observations=torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device),
            old_log_probs=torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=self.device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            values=torch.as_tensor(rollout["values"], dtype=torch.float32, device=self.device),
        )

        n_steps = batch.actions.shape[0]
        indices = np.arange(n_steps)

        self.model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_obs = batch.observations[mb_idx]
                mb_actions = batch.actions[mb_idx]
                mb_old_log_probs = batch.old_log_probs[mb_idx]
                mb_returns = batch.returns[mb_idx]
                mb_advantages = batch.advantages[mb_idx]

                logits, values = self.model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

                policy_loss_1 = ratio * mb_advantages
                policy_loss_2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_policy_loss += float(policy_loss.item())
                epoch_value_loss += float(value_loss.item())
                epoch_entropy += float(entropy.item())

        updates = max(1, self.update_epochs * int(np.ceil(n_steps / self.minibatch_size)))
        return {
            "policy_loss": epoch_policy_loss / updates,
            "value_loss": epoch_value_loss / updates,
            "entropy": epoch_entropy / updates,
        }


def evaluate_policy(model, env: TradingSequenceEnv, device: torch.device, episodes: int = 1):
    model.eval()
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }


def evaluate_policy_with_trace(model, env: TradingSequenceEnv, device: torch.device):
    """Run a single deterministic episode and collect step-wise reward trace."""
    model.eval()
    obs, _ = env.reset()

    rewards = []
    positions = []
    returns = []
    turnovers = []
    done = False

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        positions.append(float(info["position"]))
        returns.append(float(info["period_return"]))
        turnovers.append(float(info["turnover"]))

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    trace = {
        "rewards": rewards_arr,
        "cum_rewards": np.cumsum(rewards_arr),
        "positions": np.asarray(positions, dtype=np.float32),
        "period_returns": np.asarray(returns, dtype=np.float32),
        "turnovers": np.asarray(turnovers, dtype=np.float32),
    }
    return trace


def rewards_to_step_pct(rewards: np.ndarray, initial_capital: float) -> np.ndarray:
    # Rewards are already normalized by initial_capital (returns). Convert to %.
    return rewards.astype(np.float32) * 100.0


def rewards_to_cum_pct(rewards: np.ndarray, initial_capital: float) -> np.ndarray:
    return np.cumsum(rewards.astype(np.float32)) * 100.0


def save_test_reward_plot(trace: dict, output_path: str, initial_capital: float):
    x = np.arange(len(trace["rewards"]))
    step_reward_pct = rewards_to_step_pct(trace["rewards"], initial_capital)
    cum_reward_pct = rewards_to_cum_pct(trace["rewards"], initial_capital)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(x, step_reward_pct, color="#1f77b4", linewidth=1.5, label="Step Reward (%)")
    axes[0].axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Reward (%)")
    axes[0].set_title("Test Set Step Reward (%)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, cum_reward_pct, color="#2ca02c", linewidth=2.0, label="Cumulative Reward (%)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative Reward (%)")
    axes[1].set_title("Test Set Cumulative Reward (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_validation_reward_plot(log_df: pd.DataFrame, output_path: str):
    val_df = log_df.dropna(subset=["val_reward_mean"]).copy()
    if val_df.empty:
        return

    if "val_reward_step_pct" in val_df.columns:
        y = val_df["val_reward_step_pct"].values
    else:
        y = val_df["val_reward_mean"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        val_df["update"].values,
        y,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#d62728",
        label="Validation Step Reward (%)",
    )

    best_idx = int(val_df["val_reward_mean"].idxmax())
    best_row = val_df.loc[best_idx]
    best_y = float(best_row["val_reward_step_pct"]) if "val_reward_step_pct" in val_df.columns else float(best_row["val_reward_mean"])
    ax.scatter([best_row["update"]], [best_y], color="#2ca02c", s=60, zorder=3)
    ax.annotate(
        f"best={best_y:.4f}% @ {int(best_row['update'])}",
        xy=(best_row["update"], best_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_title("Validation Reward vs Update (%)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Validation Mean Step Reward (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maybe_load_supervised_backbone(model: TransformerActorCritic, ckpt_path: str, device: torch.device):
    if not ckpt_path:
        return 0
    if not os.path.exists(ckpt_path):
        print(f"[warmstart] checkpoint not found, skipping: {ckpt_path}")
        return 0

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        print("[warmstart] unsupported checkpoint format, skipping")
        return 0

    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v

    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"[warmstart] loaded {len(filtered)} matching tensors from {ckpt_path}")
    return len(filtered)


def load_data_and_build_envs(cfg: dict):
    data_cfg = cfg["data"]
    rl_cfg = cfg["rl"]
    reward_mode = str(rl_cfg.get("reward_mode", "target_scaled")).strip().lower()
    seq_len = int(data_cfg["sequence_length"])

    df = pd.read_csv(data_cfg["csv_file"])
    feature_cols = data_cfg["feature_cols"]

    required_cols = set(feature_cols)
    if reward_mode == "target_scaled":
        required_cols.add("Target")
    elif reward_mode == "next_log_return":
        required_cols.add("Log_Returns")
    else:
        raise ValueError(
            f"Unsupported rl.reward_mode='{reward_mode}'. Supported: target_scaled, next_log_return"
        )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in csv: {missing}")

    ml_df = df.dropna(subset=list(required_cols)).copy()

    X = ml_df[feature_cols].values

    if reward_mode == "target_scaled":
        reward_source = ml_df["Target"].values.reshape(-1, 1)
    else:
        # Use next bar log return as tradable reward proxy for realistic RL objective.
        reward_source = ml_df["Log_Returns"].shift(-1).values.reshape(-1, 1)
        reward_valid_mask = np.isfinite(reward_source.reshape(-1))
        X = X[reward_valid_mask]
        reward_source = reward_source[reward_valid_mask]

    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg["val_ratio"])

    train_idx = int(len(X) * train_ratio)
    val_idx = int(len(X) * val_ratio)

    X_train_raw = X[:train_idx]
    X_val_raw = X[train_idx:val_idx]
    X_test_raw = X[val_idx:]

    y_train_raw = reward_source[:train_idx]
    y_val_raw = reward_source[train_idx:val_idx]
    y_test_raw = reward_source[val_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train_raw, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val_raw, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test_raw, seq_len)

    train_returns = build_rl_reward_series(y_train_seq, rl_cfg)
    val_returns = build_rl_reward_series(y_val_seq, rl_cfg)
    test_returns = build_rl_reward_series(y_test_seq, rl_cfg)

    err_len = int(cfg["model"]["error_history_len"])
    position_values = rl_cfg.get("position_values", [-1.0, 0.0, 1.0])
    initial_capital = float(rl_cfg.get("initial_capital", 10000.0))
    spread_cost = float(rl_cfg.get("spread_cost", 0.0))
    slippage_cost = float(rl_cfg.get("slippage_cost", 0.0))
    drawdown_penalty_coef = float(rl_cfg.get("drawdown_penalty_coef", 0.0))

    train_random_start = bool(rl_cfg.get("train_random_start", True))
    train_max_episode_steps_cfg = int(rl_cfg.get("train_max_episode_steps", rl_cfg["max_episode_steps"]))
    if train_max_episode_steps_cfg <= 0:
        train_max_episode_steps = max(1, len(X_train_seq) - 1)
    else:
        train_max_episode_steps = train_max_episode_steps_cfg

    train_env = TradingSequenceEnv(
        base_sequences=X_train_seq,
        returns=train_returns,
        error_history_len=err_len,
        trading_cost=float(rl_cfg["trading_cost"]),
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        drawdown_penalty_coef=drawdown_penalty_coef,
        position_values=position_values,
        initial_capital=initial_capital,
        max_episode_steps=train_max_episode_steps,
        random_start=train_random_start,
    )

    val_env = TradingSequenceEnv(
        base_sequences=X_val_seq,
        returns=val_returns,
        error_history_len=err_len,
        trading_cost=float(rl_cfg["trading_cost"]),
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        drawdown_penalty_coef=drawdown_penalty_coef,
        position_values=position_values,
        initial_capital=initial_capital,
        max_episode_steps=max(1, len(X_val_seq) - 1),
        random_start=False,
    )

    test_env = TradingSequenceEnv(
        base_sequences=X_test_seq,
        returns=test_returns,
        error_history_len=err_len,
        trading_cost=float(rl_cfg["trading_cost"]),
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        drawdown_penalty_coef=drawdown_penalty_coef,
        position_values=position_values,
        initial_capital=initial_capital,
        max_episode_steps=max(1, len(X_test_seq) - 1),
        random_start=False,
    )

    return train_env, val_env, test_env, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(description="PPO RL trainer using GA-mlp Transformer backbone")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_rl_cnh.yaml"),
        help="Path to RL YAML config",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate/plot using saved checkpoint",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_env, val_env, test_env, scaler, feature_cols = load_data_and_build_envs(cfg)

    model_cfg = cfg["model"]
    seq_len, input_size = train_env.observation_shape

    model = TransformerActorCritic(
        input_size=input_size,
        seq_len=seq_len,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        action_dim=train_env.action_dim,
    ).to(device)

    maybe_load_supervised_backbone(
        model,
        ckpt_path=str(cfg["rl"].get("init_from_supervised_path", "")).strip(),
        device=device,
    )

    if args.eval_only:
        save_path = str(cfg["rl"]["model_save_path"])
        if not os.path.exists(save_path):
            raise RuntimeError(f"Checkpoint not found for eval-only mode: {save_path}")

        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_stats = evaluate_policy(model, test_env, device, episodes=1)
        test_step_reward_pct = (test_stats["mean_reward"] / max(1, test_env.max_episode_steps)) * 100.0
        trace = evaluate_policy_with_trace(model, test_env, device)
        plot_path = str(cfg["rl"].get("test_reward_plot_png", "rl_test_reward_plot.png"))
        save_test_reward_plot(trace, plot_path, initial_capital=test_env.initial_capital)

        print(
            "Final Test Reward: "
            f"{test_stats['mean_reward']:.6f} +/- {test_stats['std_reward']:.6f} "
            f"| Mean Step Reward: {test_step_reward_pct:.6f}%"
        )
        print(f"Saved test reward plot: {plot_path}")
        return

    trainer = PPOTrainer(model=model, device=device, cfg=cfg["rl"])

    total_timesteps = int(cfg["rl"]["total_timesteps"])
    rollout_steps = int(cfg["rl"]["rollout_steps"])
    eval_every = int(cfg["rl"]["eval_every_updates"])
    early_stop_patience_evals = int(cfg["rl"].get("early_stop_patience_evals", 0))
    early_stop_min_delta = float(cfg["rl"].get("early_stop_min_delta", 1e-8))

    updates = max(1, total_timesteps // rollout_steps)
    obs, _ = train_env.reset()

    log_rows = []
    best_val_reward = -np.inf
    best_update = -1
    evals_without_improvement = 0
    save_path = str(cfg["rl"]["model_save_path"])

    pbar = trange(1, updates + 1, desc="PPO updates", unit="update")
    for update_idx in pbar:
        rollout = trainer.collect_rollout(train_env, obs)
        obs = rollout["next_obs"]
        train_stats = trainer.update(rollout)

        row = {
            "update": update_idx,
            "rollout_reward_mean": rollout["episode_return_mean"],
            "policy_loss": train_stats["policy_loss"],
            "value_loss": train_stats["value_loss"],
            "entropy": train_stats["entropy"],
        }

        if update_idx % eval_every == 0 or update_idx == updates:
            val_stats = evaluate_policy(model, val_env, device, episodes=1)
            row["val_reward_mean"] = val_stats["mean_reward"]
            row["val_reward_std"] = val_stats["std_reward"]
            row["val_steps"] = int(val_env.max_episode_steps)
            row["val_reward_step_pct"] = (val_stats["mean_reward"] / max(1, val_env.max_episode_steps)) * 100.0

            improved = (val_stats["mean_reward"] - best_val_reward) > early_stop_min_delta
            if improved:
                best_val_reward = val_stats["mean_reward"]
                best_update = update_idx
                evals_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "feature_cols": feature_cols,
                        # Store scaler stats as plain lists to stay compatible with
                        # torch.load default weights_only=True behavior in newer PyTorch.
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                        "config": cfg,
                    },
                    save_path,
                )
            else:
                evals_without_improvement += 1

            pbar.set_postfix(
                {
                    "train_r": f"{rollout['episode_return_mean']:.4f}",
                    "val_r": f"{val_stats['mean_reward']:.4f}",
                    "best_val": f"{best_val_reward:.4f}",
                }
            )

            if early_stop_patience_evals > 0 and evals_without_improvement >= early_stop_patience_evals:
                print(
                    "Early stopping triggered: "
                    f"no val_reward_mean improvement for {evals_without_improvement} eval points "
                    f"(best={best_val_reward:.6f} at update={best_update})."
                )
                log_rows.append(row)
                break
        else:
            pbar.set_postfix(
                {
                    "train_r": f"{rollout['episode_return_mean']:.4f}",
                    "policy": f"{train_stats['policy_loss']:.4f}",
                }
            )

        log_rows.append(row)

    if not os.path.exists(save_path):
        raise RuntimeError("No checkpoint was saved. Check your eval settings.")

    # This checkpoint is generated by this script and includes metadata beyond
    # bare tensors; load it in trusted mode for PyTorch >= 2.6.
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_stats = evaluate_policy(model, test_env, device, episodes=1)
    test_step_reward_pct = (test_stats["mean_reward"] / max(1, test_env.max_episode_steps)) * 100.0
    print(
        "Final Test Reward: "
        f"{test_stats['mean_reward']:.6f} +/- {test_stats['std_reward']:.6f} "
        f"| Mean Step Reward: {test_step_reward_pct:.6f}%"
    )

    trace = evaluate_policy_with_trace(model, test_env, device)
    plot_path = str(cfg["rl"].get("test_reward_plot_png", "rl_test_reward_plot.png"))
    save_test_reward_plot(trace, plot_path, initial_capital=test_env.initial_capital)
    print(f"Saved test reward plot: {plot_path}")

    log_df = pd.DataFrame(log_rows)
    metrics_path = str(cfg["rl"].get("metrics_csv", "rl_training_metrics.csv"))
    log_df.to_csv(metrics_path, index=False)

    val_plot_path = str(cfg["rl"].get("val_reward_plot_png", "rl_val_reward_plot.png"))
    save_validation_reward_plot(log_df, val_plot_path)

    if best_update >= 0:
        best_val_step_pct = (best_val_reward / max(1, val_env.max_episode_steps)) * 100.0
        print(
            "Best validation reward: "
            f"{best_val_reward:.6f} (mean step {best_val_step_pct:.6f}%) at update {best_update}"
        )
    print(f"Saved RL metrics: {metrics_path}")
    print(f"Saved validation reward plot: {val_plot_path}")
    print(f"Saved best RL model: {save_path}")


if __name__ == "__main__":
    main()
