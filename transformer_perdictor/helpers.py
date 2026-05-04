import os
import random
from typing import Iterable, List

import numpy as np
import torch


def get_state_dict_for_save(model: torch.nn.Module) -> dict:
    """Return a checkpoint-safe state_dict, unwrapping torch.compile when needed."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Normalize known wrapper prefixes (e.g., torch.compile) for robust loading."""
    normalized = state_dict
    prefixes = ("_orig_mod.", "module.")
    for prefix in prefixes:
        if all(isinstance(k, str) and k.startswith(prefix) for k in normalized.keys()):
            normalized = {k[len(prefix):]: v for k, v in normalized.items()}
    return normalized


def load_state_dict_compat(
    model: torch.nn.Module,
    state_dict: dict,
    strict: bool = True,
) -> None:
    """Load checkpoints that may come from wrapped/compiled modules."""
    normalized = normalize_state_dict_keys(state_dict)

    # torch.compile wraps the real module in OptimizedModule; load into the
    # underlying original model to avoid key namespace mismatches.
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    try:
        target_model.load_state_dict(normalized, strict=strict)
        return
    except RuntimeError:
        # Fallback for legacy checkpoints that may already match target namespace.
        pass

    target_model.load_state_dict(state_dict, strict=strict)


def apply_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_torch_runtime(deterministic: bool = False) -> None:
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def augment_financial_data(X_tensor, noise_std=0.05, mask_prob=0.10):
    X_aug = X_tensor.clone()
    num_features = X_aug.shape[2]

    noise_std_tensor = torch.as_tensor(noise_std, dtype=X_aug.dtype, device=X_aug.device)
    if noise_std_tensor.ndim == 0:
        noise_std_tensor = noise_std_tensor.view(1, 1, 1)
    elif noise_std_tensor.ndim == 1 and noise_std_tensor.numel() == num_features:
        noise_std_tensor = noise_std_tensor.view(1, 1, num_features)
    else:
        raise ValueError(f"noise_std must be a scalar or length-{num_features} vector")

    mask_prob_tensor = torch.as_tensor(mask_prob, dtype=X_aug.dtype, device=X_aug.device)
    if mask_prob_tensor.ndim == 0:
        mask_prob_tensor = mask_prob_tensor.view(1, 1, 1)
    elif mask_prob_tensor.ndim == 1 and mask_prob_tensor.numel() == num_features:
        mask_prob_tensor = mask_prob_tensor.view(1, 1, num_features)
    else:
        raise ValueError(f"mask_prob must be a scalar or length-{num_features} vector")
    mask_prob_tensor = mask_prob_tensor.clamp(0.0, 1.0)

    noise = torch.randn_like(X_aug) * noise_std_tensor
    X_aug = X_aug + noise

    mask = (torch.rand(X_aug.shape[0], 1, num_features, device=X_aug.device) > mask_prob_tensor).float()
    X_aug = X_aug * mask

    scale = torch.empty(X_aug.shape[0], 1, 1, device=X_aug.device).uniform_(0.95, 1.05)
    X_aug = X_aug * scale
    return X_aug


def prepare_error_states(error_histories, batch_size, device, dtype):
    """
    Convert error histories to [Batch, error_history_len] tensor for ERROR_STATE token projection.
    
    Args:
        error_histories: List of error histories or tensor
        batch_size: batch size
        device: torch device
        dtype: torch dtype
    
    Returns:
        [Batch, error_history_len] tensor
    """
    if isinstance(error_histories, torch.Tensor):
        err_stack = error_histories.to(device=device, dtype=dtype)
        if err_stack.ndim == 1:
            err_stack = err_stack.unsqueeze(0).expand(batch_size, -1)
        elif err_stack.ndim == 2:
            if err_stack.shape[0] == 1 and batch_size > 1:
                err_stack = err_stack.expand(batch_size, -1)
            elif err_stack.shape[0] != batch_size:
                raise ValueError(
                    f"Tensor error_histories first dim must be 1 or batch_size={batch_size}, got {err_stack.shape[0]}"
                )
        else:
            raise ValueError("Tensor error_histories must be 1D or 2D")
    else:
        if not isinstance(error_histories[0], (list, tuple, torch.Tensor)):
            error_histories = [error_histories] * batch_size

        err_tensors = []
        for err_hist in error_histories:
            if isinstance(err_hist, torch.Tensor):
                err_vec = err_hist.to(device=device, dtype=dtype).flatten()
            else:
                err_vec = torch.tensor(err_hist, dtype=dtype, device=device).flatten()
            err_tensors.append(err_vec)

        err_stack = torch.stack(err_tensors, dim=0)

    return err_stack


def append_error_feature(X_batch, error_histories):
    """
    DEPRECATED: Kept for backward compatibility. Use prepare_error_states instead.
    Prepends error history as a separate token in the model forward pass.
    """
    batch_size = X_batch.shape[0]

    if isinstance(error_histories, torch.Tensor):
        err_stack = error_histories.to(device=X_batch.device, dtype=X_batch.dtype)
        if err_stack.ndim == 1:
            err_stack = err_stack.unsqueeze(0).expand(batch_size, -1)
        elif err_stack.ndim == 2:
            if err_stack.shape[0] == 1 and batch_size > 1:
                err_stack = err_stack.expand(batch_size, -1)
            elif err_stack.shape[0] != batch_size:
                raise ValueError(
                    f"Tensor error_histories first dim must be 1 or batch_size={batch_size}, got {err_stack.shape[0]}"
                )
        else:
            raise ValueError("Tensor error_histories must be 1D or 2D")
    else:
        if not isinstance(error_histories[0], (list, tuple, torch.Tensor)):
            error_histories = [error_histories] * batch_size

        err_tensors = []
        for err_hist in error_histories:
            if isinstance(err_hist, torch.Tensor):
                err_vec = err_hist.to(device=X_batch.device, dtype=X_batch.dtype).flatten()
            else:
                err_vec = torch.tensor(err_hist, dtype=X_batch.dtype, device=X_batch.device).flatten()
            err_tensors.append(err_vec)

        err_stack = torch.stack(err_tensors, dim=0)

    k = err_stack.shape[1]
    err_map = err_stack.unsqueeze(1).expand(batch_size, X_batch.shape[1], k)
    return torch.cat([X_batch, err_map], dim=2)


def update_error_history(error_history: List[float], new_error: float) -> List[float]:
    return error_history[1:] + [float(new_error)]


def volatility_to_confidence(volatility_pred):
    return torch.clamp(1.0 - torch.abs(volatility_pred), 0.0, 1.0)


def compute_combined_confidence(abs_pred, direction_prob, volatility_pred):
    abs_confidence = torch.clamp(abs_pred, 0.0, 1.0)
    direction_confidence = torch.clamp(2.0 * torch.abs(direction_prob - 0.5), 0.0, 1.0)
    vol_confidence = volatility_to_confidence(volatility_pred)
    combined = (abs_confidence + direction_confidence + vol_confidence) / 3.0
    return torch.clamp(combined, 0.0, 1.0)


def compute_trend_strength(X_batch, trend_source_feature_idx: int):
    log_returns = X_batch[:, :, trend_source_feature_idx]
    trend = torch.mean(torch.abs(log_returns), dim=1, keepdim=True)
    return trend


def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)


def generate_random_error_history(k: int):
    return np.random.randn(k).tolist()


def pick_feature_level_values(feature_cols: Iterable[str], by_tf: dict):
    values = []
    # Features that represent absolute reality and should never be mutated
    deterministic_features = {"Time_Sin", "Time_Cos", "Session_Asia"}
    
    for col in feature_cols:
        if col in deterministic_features:
            values.append(0.0) # 0.0 noise / 0.0 dropout
        elif col.endswith("_1D"):
            values.append(by_tf.get("1D", 0.0))
        elif col.endswith("_4H"):
            values.append(by_tf.get("4H", 0.0))
        else:
            values.append(by_tf.get("15m", 0.0))
            
    return values
