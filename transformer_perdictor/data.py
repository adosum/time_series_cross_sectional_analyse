from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .helpers import create_sequences, pick_feature_level_values


@dataclass
class DataBundle:
    feature_cols: List[str]
    trend_source_feature_idx: int
    X_train_tensor: torch.Tensor
    y_train_tensor: torch.Tensor
    X_val_tensor: torch.Tensor
    y_val_tensor: torch.Tensor
    X_test_tensor: torch.Tensor
    y_test_tensor: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_noise_std: List[float]
    feature_mask_prob: List[float]


def prepare_data(
    cfg: dict,
    device: torch.device,
) -> DataBundle:
    data_cfg = cfg["data"]
    aug_cfg = cfg["augmentation"]
    train_cfg = cfg["training"]

    print("Loading MTF ML Data...")
    df = pd.read_csv(data_cfg["csv_file"], index_col="Datetime", parse_dates=True)

    # Cyclical intraday encoding so the model can learn that 23:45 and 00:15
    # are temporally close on a 24-hour liquidity cycle.
    hours = df.index.hour + (df.index.minute / 60.0)
    df["Time_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["Time_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)

    feature_cols = data_cfg["feature_cols"]
    missing_feature_cols = [c for c in feature_cols if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing features in training table: {missing_feature_cols}")

    aug_symbol = os.getenv("AUG_SYMBOL", "").strip().upper()
    default_noise = aug_cfg["default_noise_std_by_tf"]
    noise_by_symbol = aug_cfg["noise_std_by_symbol"]
    noise_std_by_tf = noise_by_symbol.get(aug_symbol, default_noise).copy()

    noise_mult = float(os.getenv("AUG_NOISE_MULT", "1.0"))
    noise_std_by_tf = {k: v * noise_mult for k, v in noise_std_by_tf.items()}

    mask_prob_by_tf = aug_cfg["mask_prob_by_tf"]

    print(
        f"Augmentation profile: symbol={aug_symbol or 'DEFAULT'}, "
        f"noise_mult={noise_mult:.3f}, noise_std_by_tf={noise_std_by_tf}, mask_prob_by_tf={mask_prob_by_tf}"
    )

    feature_noise_std = pick_feature_level_values(feature_cols, noise_std_by_tf)
    feature_mask_prob = pick_feature_level_values(feature_cols, mask_prob_by_tf)

    ml_df = df.dropna(subset=feature_cols + ["Target_15m", "Target_1H", "Target_4H"]).copy()

    XX = ml_df[feature_cols].copy()
    y = ml_df[["Target_15m", "Target_1H", "Target_4H"]].values  # shape (N, 3)

    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    seq_len = data_cfg["sequence_length"]

    train_idx = int(len(XX) * train_ratio)
    val_idx = int(len(XX) * val_ratio)

    XX_train = XX[:train_idx]
    XX_val = XX[train_idx:val_idx]
    XX_test = XX[val_idx:]

    y_train_raw = y[:train_idx]
    y_val_raw = y[train_idx:val_idx]
    y_test_raw = y[val_idx:]

    # Mixed feature scaling strategy: some features are outlier-prone,
    # some bounded oscillators, and some binary/already normalized.
    robust_features = ["Vol_Ratio", "ADX_4H", "ADX_1D", "VWAP_Dist","Volume"]
    standard_features = [
        "Log_Returns",
        "AO_15m",
        "AO_4H",
        "AO_1D",
        "AO15_x_VolRatio",
        "AO_4H_Diff4",
        "AO_1D_Diff1",
        "AO_15m_Diff4",
        "Ret_Sum_8",
        "Ret_Sum_16",
        "Donchian_Pos_20",
        "OBV_Diff4",
        "MACD_Hist",
        "trend_macd",
        "momentum_ao",
        "trend_cci",
        "momentum_tsi",
        "trend_vortex_ind_neg",
        "trend_vortex_ind_pos",
    ]
    minmax_features = ["MFI_20", "RSI_14", "MFI_20_Diff1", "momentum_rsi", "volatility_bbp"]
    passthrough_features = ["Session_Asia", "BB_PctB", "Time_Sin", "Time_Cos", "volatility_dcp"]
    # OHLC price columns: always passthrough (RevIN handles their normalization at model level)
    ohlc_features = ["Open", "High", "Low", "Close"]

    def _present(cols):
        return [c for c in cols if c in feature_cols]

    robust_features = _present(robust_features)
    standard_features = _present(standard_features)
    minmax_features = _present(minmax_features)
    passthrough_features = _present(passthrough_features)
    ohlc_features = _present(ohlc_features)

    # ⭐️ FIX: Redefine feature_cols to match the exact output order of ColumnTransformer
    feature_cols = robust_features + standard_features + minmax_features + passthrough_features + ohlc_features

    forgotten_features = [c for c in data_cfg["feature_cols"] if c not in feature_cols]
    if forgotten_features:
        raise ValueError(f"CRITICAL: Features {forgotten_features} are in config.yaml but missing from the scaler lists! Add them to robust, standard, minmax, or passthrough.")

    # NOW generate the noise and mask lists using the newly ordered feature_cols!
    feature_noise_std = pick_feature_level_values(feature_cols, noise_std_by_tf)
    feature_mask_prob = pick_feature_level_values(feature_cols, mask_prob_by_tf)
    multi_scaler = ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), robust_features),
            ("standard", StandardScaler(), standard_features),
            ("minmax", MinMaxScaler(feature_range=(0, 1)), minmax_features),
            ("pass", "passthrough", passthrough_features),
            ("ohlc", "passthrough", ohlc_features),
        ],
        remainder="drop",
    )

    # We must slice XX using the NEW feature_cols order so the ColumnTransformer 
    # receives exactly what we expect
    XX_train_scaled = multi_scaler.fit_transform(XX_train[feature_cols])
    XX_val_scaled = multi_scaler.transform(XX_val[feature_cols])
    XX_test_scaled = multi_scaler.transform(XX_test[feature_cols])

    output_feature_names = list(multi_scaler.get_feature_names_out())
    log_returns_feature_name = "standard__Log_Returns"
    if log_returns_feature_name in output_feature_names:
        trend_source_feature_idx = output_feature_names.index(log_returns_feature_name)
    elif "Log_Returns" in output_feature_names:
        # Fallback for environments with non-prefixed output names.
        trend_source_feature_idx = output_feature_names.index("Log_Returns")
    else:
        raise ValueError(
            f"Could not locate Log_Returns in transformed features. "
            f"Expected '{log_returns_feature_name}'. Available names: {output_feature_names}"
        )

    X_train_3d, y_train_aligned = create_sequences(XX_train_scaled, y_train_raw, seq_len=seq_len)
    X_val_3d, y_val_aligned = create_sequences(XX_val_scaled, y_val_raw, seq_len=seq_len)
    X_test_3d, y_test_aligned = create_sequences(XX_test_scaled, y_test_raw, seq_len=seq_len)

    X_train_tensor = torch.tensor(X_train_3d, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_aligned, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_3d, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_aligned, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_aligned, dtype=torch.float32).to(device)

    batch_size = train_cfg["batch_size"]

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    stage1_shuffle = bool(train_cfg.get("stage1_shuffle", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=stage1_shuffle,
        drop_last=True,  # Drop last batch if it's smaller than batch_size
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)

    print(f"Batches per Epoch: {len(train_loader)}")
    print(f"Stage 1 train shuffle: {stage1_shuffle}")

    return DataBundle(
        feature_cols=feature_cols,
        trend_source_feature_idx=trend_source_feature_idx,
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        X_test_tensor=X_test_tensor,
        y_test_tensor=y_test_tensor,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_noise_std=feature_noise_std,
        feature_mask_prob=feature_mask_prob,
    )
