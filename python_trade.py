import ctypes  # Needed for high-res DPI awareness
import os
import tkinter as tk
import tkinter.font as tkfont
import traceback
from datetime import datetime
import yaml

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from ta.momentum import AwesomeOscillatorIndicator
from transformer_perdictor.model import USDCNHTransformer

# 引入纯血 ta 库 (无需 C++ 编译)
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator

# Import unified feature engineering module
from feature_engineering import build_all_features, get_feature_cols

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

class FloatingHUD:
    def __init__(self):
        # --- 1. Fix Blurry Text on 2K/4K Screens ---
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDpiAware(1)

        self.root = tk.Tk()
        self.root.title("Python Quant HUD")
        self.root.attributes("-topmost", True)

        # --- 2. Adjust Geometry for 2K Screen ---
        # Wider layout for side-by-side multi-asset panels
        self.root.geometry("1250x850")
        self.root.configure(bg="#1e1e1e")  # Darker background for pro look
        self.base_font_size = 13
        self.min_font_size = 8

        self.left_font = tkfont.Font(
            family="Consolas", size=self.base_font_size, weight="bold"
        )
        self.right_font = tkfont.Font(
            family="Consolas", size=self.base_font_size, weight="bold"
        )

        # --- 3. Two-column HUD panel ---
        self.container = tk.Frame(self.root, bg="#1e1e1e")
        self.container.pack(expand=True, fill="both", padx=10, pady=10)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        self.left_label = tk.Label(
            self.container,
            text="USDCNH\nWaiting for data...",
            font=self.left_font,
            bg="#1e1e1e",
            fg="#00FF00",
            justify=tk.LEFT,
            anchor="nw",
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            wraplength=560,
        )
        self.left_label.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.right_label = tk.Label(
            self.container,
            text="CORN\nWaiting for data...",
            font=self.right_font,
            bg="#1e1e1e",
            fg="#00FF00",
            justify=tk.LEFT,
            anchor="nw",
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            wraplength=560,
        )
        self.right_label.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # 响应窗口大小变化，动态调整字体和换行宽度
        self.root.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        try:
            if not self.root.winfo_exists():
                return

            width = max(self.root.winfo_width(), 400)
            height = max(self.root.winfo_height(), 300)

            # 根据可视面积估算字体，窗口缩小时自动减小字体
            est_size_w = int(width / 95)
            est_size_h = int(height / 65)
            dynamic_size = max(
                self.min_font_size, min(self.base_font_size, est_size_w, est_size_h)
            )

            self.left_font.configure(size=dynamic_size)
            self.right_font.configure(size=dynamic_size)

            panel_wrap = max(int((width - 80) / 2), 220)
            if self.left_label.winfo_exists():
                self.left_label.config(wraplength=panel_wrap)
            if self.right_label.winfo_exists():
                self.right_label.config(wraplength=panel_wrap)
        except tk.TclError:
            pass

    def refresh(self, content=None):
        try:
            if not self.root.winfo_exists():
                return

            if content:
                if isinstance(content, dict):
                    left_text = content.get("left", "USDCNH\nWaiting for data...")
                    right_text = content.get("right", "CORN\nWaiting for data...")

                    if self.left_label.winfo_exists():
                        self.left_label.config(text=left_text.strip())
                    if self.right_label.winfo_exists():
                        self.right_label.config(text=right_text.strip())
                else:
                    # Backward-compatible fallback if plain text is passed
                    if self.left_label.winfo_exists():
                        self.left_label.config(text=str(content).strip())

            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # Handles the case where you close the window manually
            pass


# ==========================================
# [驾驶舱] 策略超参配置面板
# ==========================================
CONFIG = {
    # 1. 宏观环境阈值
    "macro_trend_adx": 25.0,  # ADX 大于此值认定为趋势
    "macro_range_adx": 20.0,  # ADX 小于此值认定为震荡
    # 2. 震荡微观参数
    "mfi_base_oversold": 25.0,  # MFI 基础超卖线
    "mfi_base_overbought": 75.0,  # MFI 基础超买线
    "min_expected_rr": 1.5,  # 最小允许盈亏比
    # 3. 趋势微观参数
    "trend_mfi_confirm": 50.0,  # 顺势突破时，MFI 必须大于此值证明资金流入
    # 4. 冷静期机制（防频繁交易）
    "cooldown_after_long_bars": 3,  # 做多后，至少冷静N根K线再做空
    "cooldown_after_short_bars": 3,  # 做空后，至少冷静N根K线再做多
    "cooldown_after_filtered_bars": 2,  # 信号被过滤后，也要冷静一段时间
    # 5. 风控熔断参数 (Circuit Breaker)
    "volatility_shock_threshold": 2.5,  # 当快慢 ATR 比值超过此数，触发熔断禁止开新仓
    # 6. 趋势加仓参数 (Pyramiding)
    "max_trend_positions": 3,  # 趋势中最大允许同向持仓数 (底仓 + 2次加仓)
    "trend_add_pos_atr_step": 0.5,
}


SYMBOL_PREFERENCES = {
    "FX": ["USDCNH"],
    "CORN": ["CORN.c"],
}

# Keep enough 15m bars for 1D AO(34) + ADX warmup while avoiding oversized fetches.
M15_HISTORY_BARS = 4200
M15_INCREMENT_BARS = 300
ENGINE_TICK_MS = 150

# 默认映射: CNH checkpoint -> USDCNH, 通用 checkpoint -> CORN
# Now includes training config files for proper model initialization
MODEL_RUNTIME_CONFIG = {
    "USDCNH": {
        "model_path": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\cnh_best_transformer_model.pth",
        "scaler_csv": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\USDCNH_MTF_ML_ready.csv",
        "config_file": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\config_cnh.yaml",
    },
    "CORN": {
        "model_path": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\corn_best_transformer_model.pth",
        "scaler_csv": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\CORN_MTF_ML_ready.csv",
        "config_file": r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\config_corn.yaml",
    },
}


def validate_config(config):
    """验证配置参数的合理性，防止运行时错误"""
    if config["macro_trend_adx"] <= config["macro_range_adx"]:
        raise ValueError("配置致命错误: 趋势ADX阈值 必须大于 震荡ADX阈值！")
    if config["mfi_base_oversold"] >= config["mfi_base_overbought"]:
        raise ValueError("配置致命错误: MFI超卖线 必须小于 超买线！")
    if (
        config["cooldown_after_long_bars"] < 1
        or config["cooldown_after_short_bars"] < 1
    ):
        raise ValueError("配置致命错误: 冷静期参数 必须 >= 1 根K线！")
    if config["min_expected_rr"] < 0.5:
        raise ValueError("配置致命错误: 最小风险回报比应该 >= 0.5！")
    if config["trend_mfi_confirm"] < 0 or config["trend_mfi_confirm"] > 100:
        raise ValueError("配置致命错误: trend_mfi_confirm 应该在 [0, 100] 范围内！")
    print("✅ 配置参数校验通过！")


def infer_asset_key(symbol):
    upper = symbol.upper()
    if "USDCNH" in upper:
        return "USDCNH"
    if "CORN" in upper:
        return "CORN"
    return upper


def update_error_history_live(error_history, new_error):
    return error_history[1:] + [float(new_error)]


def _ensure_time_cyc_features(df, feature_cols):
    """Ensure Time_Sin/Time_Cos exist when requested by the model feature list."""
    needs_time_sin = "Time_Sin" in feature_cols and "Time_Sin" not in df.columns
    needs_time_cos = "Time_Cos" in feature_cols and "Time_Cos" not in df.columns
    if not (needs_time_sin or needs_time_cos):
        return df

    dt_series = None
    if isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series(index=df.index)
    else:
        for candidate in ["Datetime", "datetime", "time", "Time"]:
            if candidate in df.columns:
                dt_series = pd.to_datetime(df[candidate], errors="coerce")
                break

    if dt_series is not None and dt_series.notna().any():
        hours = dt_series.dt.hour + (dt_series.dt.minute / 60.0)
        theta = 2.0 * np.pi * (hours / 24.0)
        if needs_time_sin:
            df["Time_Sin"] = np.sin(theta)
        if needs_time_cos:
            df["Time_Cos"] = np.cos(theta)
    else:
        # Fallback for non-datetime sources.
        if needs_time_sin:
            df["Time_Sin"] = 0.0
        if needs_time_cos:
            df["Time_Cos"] = 0.0
    return df


def build_model_feature_frame(strategy_df, feature_cols, asset='default'):
    """
    Build model feature frame for live inference.
    Uses unified feature_engineering module to ensure consistency with training.
    
    Args:
        strategy_df: DataFrame with OHLCV (expects 'close', 'volume', OHLC indicators, etc.)
        feature_cols: Ordered list of features expected by the model (from config YAML, reordered)
        asset: 'USDCNH', 'CORN', or 'default'
    
    Returns:
        DataFrame with all feature columns present, NaN rows dropped.
    """
    feat_df = strategy_df.copy()
    feat_df['Log_Returns'] = np.log(feat_df['close']).diff()

    atr_slow = feat_df["ATR_Slow"].replace(0, np.nan)
    feat_df["Vol_Ratio"] = (feat_df["ATR_Fast"] / atr_slow).replace(
        [np.inf, -np.inf], np.nan
    )

    bb_range = (feat_df["BB_Upper"] - feat_df["BB_Lower"]).replace(0, np.nan)
    feat_df["BB_PctB"] = ((feat_df["close"] - feat_df["BB_Lower"]) / bb_range).replace(
        [np.inf, -np.inf], np.nan
    )

    donch_range = (feat_df["Donchian_Upper_20"] - feat_df["Donchian_Lower_20"]).replace(
        0, np.nan
    )
    feat_df["Donchian_Pos_20"] = (
        (feat_df["close"] - feat_df["Donchian_Lower_20"]) / donch_range
    ).replace([np.inf, -np.inf], np.nan)

    # Use shared feature engineering module to build full unified feature set.
    feat_df = build_all_features(feat_df, asset=asset)

    # Time_Sin/Time_Cos are passthrough features in training; ensure they exist here too.
    feat_df = _ensure_time_cyc_features(feat_df, feature_cols)

    # Fill NaN in model features with 0 (keeps all rows for warm-up and inference)
    feat_df[feature_cols] = feat_df[feature_cols].fillna(0)
    return feat_df


def build_scaler_feature_frame(scaler_df, feature_cols):
    """
    Build scaler feature frame from prepared training CSV.
    
    Args:
        scaler_df: DataFrame from prepared CSV (USDCNH_MTF_ML_ready.csv or CORN_MTF_ML_ready.csv)
        feature_cols: List of feature column names to validate and prepare
    
    Returns:
        DataFrame with features for scaler fitting (ColumnTransformer)
    """
    feat_df = scaler_df.copy()

    # Backward compatibility for older prepared CSVs missing passthrough time features.
    feat_df = _ensure_time_cyc_features(feat_df, feature_cols)

    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Scaler CSV missing feature columns: {missing}")

    for col in feature_cols:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    feat_df = feat_df.dropna(subset=feature_cols)
    return feat_df


def build_multi_scaler(feature_cols):
    """
    Build a ColumnTransformer with mixed scaling strategies, matching training pipeline.
    
    Different feature types require different scaling approaches:
    - Robust: Outlier-prone features (volume, ATR-based metrics)
    - Standard: General features (returns, indicators)
    - MinMax: Bounded oscillators (MFI, RSI)
    - Passthrough: Already normalized/binary features
    
    Args:
        feature_cols: List of feature column names (from config YAML)
    
    Returns:
        tuple: (ColumnTransformer, ordered_feature_cols)
        ordered_feature_cols is the definitive column order that ColumnTransformer
        will output — robust + standard + minmax + passthrough.
    """
    # Master lists define the canonical order within each scaler group.
    # This MUST match transformer_perdictor/data.py exactly.
    robust_features_all = ["Vol_Ratio", "ADX_4H", "ADX_1D", "VWAP_Dist", "Volume"]
    standard_features_all = [
        "Log_Returns", "AO_15m", "AO_4H", "AO_1D", "AO15_x_VolRatio", "AO_4H_Diff4",
        "AO_1D_Diff1", "AO_15m_Diff4", "Ret_Sum_8", "Ret_Sum_16", "Donchian_Pos_20",
        "OBV_Diff4", "MACD_Hist", "trend_macd", "momentum_ao", "trend_cci", "momentum_tsi",
        "trend_vortex_ind_neg", "trend_vortex_ind_pos",
    ]
    minmax_features_all = ["MFI_20", "RSI_14", "MFI_20_Diff1", "momentum_rsi", "volatility_bbp"]
    passthrough_features_all = ["Session_Asia", "BB_PctB", "Time_Sin", "Time_Cos", "volatility_dcp"]
    # OHLC columns are passthrough in training (RevIN handles normalization in-model).
    ohlc_features_all = ["Open", "High", "Low", "Close"]

    # Filter to only features present in feature_cols.
    # Iterating through the master list preserves master-list order, not YAML order.
    feature_cols_set = set(feature_cols)  # O(1) lookup
    robust_features = [c for c in robust_features_all if c in feature_cols_set]
    standard_features = [c for c in standard_features_all if c in feature_cols_set]
    minmax_features = [c for c in minmax_features_all if c in feature_cols_set]
    passthrough_features = [c for c in passthrough_features_all if c in feature_cols_set]
    ohlc_features = [c for c in ohlc_features_all if c in feature_cols_set]

    # Canonical output order: must match ColumnTransformer output order exactly
    ordered_feature_cols = robust_features + standard_features + minmax_features + passthrough_features + ohlc_features

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

    return multi_scaler, ordered_feature_cols

class LiveModelRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contexts = {}
        self.error_history = {}
        self.status = {}
        self.model_feature_cols = {}  # Track per-asset feature columns
        self.model_configs = {}  # Store config params per asset
        self._load_contexts()

    def _load_config(self, asset, config_path):
        """Load YAML config and extract model/data parameters."""
        import yaml
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            return cfg
        except Exception as e:
            print(f"[WARNING] Failed to load config for {asset}: {e}")
            # Return defaults if config load fails
            return {
                'data': {'sequence_length': 96, 'feature_cols': []},
                'model': {
                    'd_model': 32,
                    'nhead': 2,
                    'num_layers': 2,
                    'dropout': 0.4,
                    'error_history_len': 96,
                    'patch_len': 8,
                    'patch_stride': 4,
                }
            }

    def _warm_up_error_history(self, asset, strategy_df, model, scaler, feature_cols, cfg):
        """Warm up the error history with predictions on initial data.
        
        This prevents using all-zero error states which can bias early predictions.
        """
        print(f"  [WARMUP] Initializing error history for {asset}...")
        seq_len = cfg['data']['sequence_length']
        error_history_len = cfg['model']['error_history_len']
        error_history = [0.0] * error_history_len
        
        # Use tail of strategy_df for warm-up predictions
        # We need at least seq_len rows to make a prediction
        available_rows = len(strategy_df)
        if available_rows < seq_len:
            print(f"  [WARMUP] Insufficient data for warm-up (need {seq_len}, got {available_rows})")
            return error_history
        
        # Start warm-up from seq_len rows and iterate through remaining data
        num_warmup_steps = min(available_rows - seq_len, error_history_len)
        print(f"  [WARMUP] Running {num_warmup_steps} warm-up steps...")
        
        for i in range(num_warmup_steps):
            start_idx = available_rows - seq_len - num_warmup_steps + i
            end_idx = start_idx + seq_len
            
            # Extract window
            window_df = strategy_df[feature_cols].iloc[start_idx:end_idx]
            window_scaled = scaler.transform(window_df)
            
            x = torch.tensor(window_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
            error_state = torch.tensor(error_history, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                (mu_15m, sigma_15m), (mu_1h, sigma_1h), (mu_4h, sigma_4h) = model(x, error_state)
            
            direction_prob = float(mu_15m.squeeze().item())
            
            # --- THE FIX: TRUE HISTORICAL ERROR ---
            # The model just predicted the bar at 'end_idx' based on data up to 'end_idx - 1'
            # Because this is historical warm-up, we actually HAVE the bar at 'end_idx'!
            last_close = strategy_df['close'].iloc[end_idx - 1]
            future_close = strategy_df['close'].iloc[end_idx]
            
            actual_target = 1.0 if future_close > last_close else 0.0
            real_error = actual_target - direction_prob
            
            # Update error history with reality
            error_history = error_history[1:] + [float(real_error)]
        
        print(f"  [WARMUP] Complete. Error history range: [{min(error_history):.4f}, {max(error_history):.4f}]")
        self.contexts[asset]["last_predicted_prob"] = direction_prob
        return error_history

    def _load_contexts(self):
        for asset, cfg_dict in MODEL_RUNTIME_CONFIG.items():
            try:
                model_path = cfg_dict["model_path"]
                scaler_csv = cfg_dict["scaler_csv"]
                config_file = cfg_dict.get("config_file")

                if not os.path.exists(model_path):
                    self.status[asset] = f"Model missing: {model_path}"
                    continue
                if not os.path.exists(scaler_csv):
                    self.status[asset] = f"Scaler csv missing: {scaler_csv}"
                    continue

                # Load training config to get model architecture params
                if config_file and os.path.exists(config_file):
                    cfg = self._load_config(asset, config_file)
                    self.model_configs[asset] = cfg
                else:
                    # Use defaults
                    cfg = {
                        'data': {'sequence_length': 96, 'feature_cols': []},
                        'model': {
                            'd_model': 32,
                            'nhead': 2,
                            'num_layers': 2,
                            'dropout': 0.4,
                            'error_history_len': 96,
                            'patch_len': 8,
                            'patch_stride': 4,
                        }
                    }
                    self.model_configs[asset] = cfg

                # Get feature columns directly from config YAML (same as training)
                feature_cols_from_config = cfg['data']['feature_cols']
                if not feature_cols_from_config:
                    self.status[asset] = "No feature_cols in config file"
                    continue

                # Load and prepare scaler
                scaler_df = pd.read_csv(scaler_csv)
                scaler_df = build_scaler_feature_frame(scaler_df, feature_cols=feature_cols_from_config)
                
                # Build multi-scaler: returns (ColumnTransformer, ordered_feature_cols)
                # ordered_feature_cols is the canonical group order: robust+standard+minmax+passthrough
                multi_scaler, feature_cols_ordered = build_multi_scaler(feature_cols_from_config)

                # Verify no features from config are lost (not assigned to any scaler group)
                forgotten_features = [c for c in feature_cols_from_config if c not in feature_cols_ordered]
                if forgotten_features:
                    raise ValueError(f"CRITICAL: Features {forgotten_features} not in any scaler group!")

                # Store the ordered feature_cols (this is what we'll use for inference)
                self.model_feature_cols[asset] = feature_cols_ordered

                # Fit the multi-scaler with ordered features
                multi_scaler.fit(scaler_df[feature_cols_ordered])
                scaler = multi_scaler

                print(f"[SCALER:{asset}] Feature order: {len(feature_cols_ordered)} features: {feature_cols_ordered}")

                # Get model architecture params from config
                seq_len = cfg['data']['sequence_length']
                model_cfg = cfg['model']
                d_model = model_cfg.get('d_model', 32)
                nhead = model_cfg.get('nhead', 2)
                num_layers = model_cfg.get('num_layers', 2)
                dropout = model_cfg.get('dropout', 0.4)
                error_history_len = model_cfg.get('error_history_len', 96)
                patch_len = model_cfg.get('patch_len', 8)
                patch_stride = model_cfg.get('patch_stride', 4)

                # Create model with proper architecture
                model = USDCNHTransformer(
                    input_size=len(feature_cols_ordered),
                    seq_len=seq_len,
                    error_history_len=error_history_len,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=dropout,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                ).to(self.device)

                state_dict = torch.load(model_path, map_location=self.device)
                load_state_dict_compat(model, state_dict, strict=True)
                model.eval()

                # Load calibration temperature (T=1.0 fallback if file absent)
                T_path = model_path + ".T.pt"
                if os.path.exists(T_path):
                    T_val = torch.load(T_path, map_location="cpu")["T"]
                else:
                    T_val = 1.0

                self.contexts[asset] = {
                    "model": model,
                    "scaler": scaler,
                    "T": float(T_val),
                    "seq_len": seq_len,
                    "error_history_len": error_history_len,
                }
                
                # Initialize error history (will be warm-up'd on first real data)
                self.error_history[asset] = [0.0] * error_history_len
                
                self.status[asset] = (
                    f"Loaded ({model_path}) | features: {len(feature_cols_ordered)} "
                    f"| error_len: {error_history_len} | patch_len: {patch_len} | patch_stride: {patch_stride}"
                )
            except Exception as e:
                self.status[asset] = f"Load failed: {e}"

    def predict(self, asset, strategy_df):
        if asset not in self.contexts:
            return {"error": self.status.get(asset, "Model not configured")}

        try:
            # Use feature_cols that were set during model loading (already reordered to match training)
            feature_cols = self.model_feature_cols.get(asset)
            if not feature_cols:
                return {"error": f"Feature columns not configured for {asset}"}
            
            feat_df = build_model_feature_frame(strategy_df, feature_cols=feature_cols, asset=asset)
            
            seq_len = self.contexts[asset]["seq_len"]
            
            # 1. Check if we have enough data
            if len(feat_df) < seq_len + 1: # +1 because we need previous bars to check real errors!
                return {"error": f"Need {seq_len + 1} feature rows, got {len(feat_df)}"}
                
            # 2. Trigger Warm-up ONCE
            warmup_key = f"{asset}_warmed_up"
            just_warmed_up = False  # <--- NEW FLAG
            
            if not self.status.get(warmup_key, False):
                self.error_history[asset] = self._warm_up_error_history(
                    asset, feat_df, self.contexts[asset]["model"],
                    self.contexts[asset]["scaler"], feature_cols, self.model_configs[asset]
                )
                self.status[warmup_key] = True
                just_warmed_up = True  # <--- SET FLAG

            # Extract sequence window as DataFrame to preserve column names for ColumnTransformer
            window_df = feat_df[feature_cols].tail(seq_len)
            window_scaled = self.contexts[asset]["scaler"].transform(window_df)
            x = (
                torch.tensor(window_scaled, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            
            # Use current error history
            error_state = torch.tensor(self.error_history[asset], dtype=torch.float32, device=self.device).unsqueeze(0)

            # --- THE FIX: ONLY UPDATE IF NOT JUST WARMED UP ---
            if not just_warmed_up:
                # We look at the last two rows to see if the price actually went up or down
                last_close = feat_df['close'].iloc[-2]
                current_close = feat_df['close'].iloc[-1]
                actual_direction_target = 1.0 if current_close > last_close else 0.0
                
                # Retrieve the probability the model PREDICTED one step ago
                last_predicted_prob = self.contexts[asset].get("last_predicted_prob", 0.5)
                
                # Calculate the TRUE autoregressive error!
                realized_error = actual_direction_target - last_predicted_prob
                
                # Update the history BEFORE making the new prediction
                self.error_history[asset] = update_error_history_live(
                    self.error_history[asset], realized_error
                )
                
                # Re-create the error_state tensor because we just updated the history array!
                error_state = torch.tensor(self.error_history[asset], dtype=torch.float32, device=self.device).unsqueeze(0)
            # -----------------------------------------------------------

            with torch.no_grad():
                (mu_15m, sigma_15m), (mu_1h, sigma_1h), (mu_4h, sigma_4h) = (
                    self.contexts[asset]["model"](x, error_state)
                )

            direction_prob = float(mu_15m.squeeze().item())
            # mu_15m is a raw return prediction; positive => LONG (threshold = 0).
            direction = "LONG" if direction_prob >= 0 else "SHORT"
            
            # STORE THIS PREDICTION so we can check it against reality on the NEXT bar!
            self.contexts[asset]["last_predicted_prob"] = direction_prob

            return {
                "direction": direction,
                "direction_prob": direction_prob,
                "mu_15m": float(mu_15m.squeeze().item()),
                "sigma_15m": float(sigma_15m.squeeze().item()),
                "mu_1h": float(mu_1h.squeeze().item()),
                "sigma_1h": float(sigma_1h.squeeze().item()),
                "mu_4h": float(mu_4h.squeeze().item()),
                "sigma_4h": float(sigma_4h.squeeze().item()),
            }
        except Exception as e:
            return {"error": f"Inference failed: {e}"}


# ==========================================
# 模块 0.5：交易信号历史追踪器 (已修复状态覆写漏洞)
# ==========================================
class SignalTracker:
    def __init__(self):
        self.last_executed_signal = None  # 只记录真正通过的 LONG / SHORT
        self.signal_bar_count = 0  # 自上次真实信号以来的 K线根数

    def update(self, candidate_signal):
        """只有当信号是 LONG 或 SHORT 时，才重置计数器和状态"""
        if candidate_signal in ["LONG", "SHORT"]:
            self.last_executed_signal = candidate_signal
            self.signal_bar_count = 0
        else:
            self.signal_bar_count += (
                1  # 如果没有信号，或者是过滤/冷静状态，默默增加计数
            )

    def is_cooldown_active(self, candidate_signal, config):
        # 如果当前根本没产生想做的信号，直接返回不拦截
        if candidate_signal not in ["LONG", "SHORT"]:
            return False, 0

        if self.last_executed_signal is None:
            return False, 0

        # 检查冷静期规则
        if self.last_executed_signal == "LONG":
            required_cooldown = config["cooldown_after_long_bars"]
        elif self.last_executed_signal == "SHORT":
            required_cooldown = config["cooldown_after_short_bars"]
        else:
            return False, 0

        remaining_bars = required_cooldown - self.signal_bar_count
        if remaining_bars > 0:
            return True, remaining_bars
        else:
            return False, 0


# ==========================================
# 模块 1：获取 MT5 底层数据
# ==========================================
def get_mt5_data(symbol, timeframe, num_bars):
    """从 MT5 获取 K 线数据并转换为 DataFrame"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    # 优先使用 tick_volume；若全为0则回退到 real_volume；仍不可用则给一个常数体积
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    elif "real_volume" in df.columns:
        df["volume"] = df["real_volume"]
    else:
        df["volume"] = 1.0

    if (df["volume"] <= 0).all() and "real_volume" in df.columns:
        if (df["real_volume"] > 0).any():
            df["volume"] = df["real_volume"]

    if (df["volume"] <= 0).all():
        df["volume"] = 1.0

    # 过滤无效价格行（部分合约可能出现0价占位bar，导致指标全失真）
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=price_cols)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

    df.sort_index(inplace=True)
    return df


def resolve_available_symbol(symbol_candidates):
    """Return the first available symbol name from candidates."""
    for name in symbol_candidates:
        info = mt5.symbol_info(name)
        if info is not None:
            mt5.symbol_select(name, True)
            return name
    return None


# ==========================================
# 模块 2：多周期特征工程 (原生 MTF + AsOf 对齐)
# ==========================================
def build_multi_timeframe_features(df_15m, df_4h, df_1d):
    df = df_15m.copy()

    # --- 1. 计算 15m 微观指标 (不变) ---
    df["AO_15m"] = AwesomeOscillatorIndicator(
        window1=5, window2=34, high=df["high"], low=df["low"]
    ).awesome_oscillator()
    df["MFI_10"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=10,
    ).money_flow_index()
    df["MFI_14"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=14,
    ).money_flow_index()
    df["MFI_20"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=20,
    ).money_flow_index()

    for mfi_col in ["MFI_10", "MFI_14", "MFI_20"]:
        if df[mfi_col].isna().all():
            df[mfi_col] = 50.0
        else:
            df[mfi_col] = df[mfi_col].fillna(50.0)

    df["ATR_Fast"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["ATR_Slow"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=50
    ).average_true_range()

    bb = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Upper"] = bb.bollinger_hband()

    df["Donchian_Upper_20"] = df["high"].rolling(window=20).max().shift(1)
    df["Donchian_Lower_10"] = df["low"].rolling(window=10).min().shift(1)
    df["Donchian_Lower_20"] = df["low"].rolling(window=20).min().shift(1)
    df["Donchian_Upper_10"] = df["high"].rolling(window=10).max().shift(1)

    # --- 2. 在原生的 4H 和 1D 数据上直接计算指标 ---
    df_4h["ADX_4H"] = ADXIndicator(
        high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], window=14
    ).adx()
    df_4h["AO_4H"] = AwesomeOscillatorIndicator(
        window1=5, window2=34, high=df_4h["high"], low=df_4h["low"]
    ).awesome_oscillator()

    df_1d["ADX_1D"] = ADXIndicator(
        high=df_1d["high"], low=df_1d["low"], close=df_1d["close"], window=14
    ).adx()
    df_1d["AO_1D"] = AwesomeOscillatorIndicator(
        window1=5, window2=34, high=df_1d["high"], low=df_1d["low"]
    ).awesome_oscillator()

    # --- 3. Shift 防止未来函数，并提取所需列 ---
    df_4h_shifted = df_4h[["ADX_4H", "AO_4H"]].shift(1).dropna()
    df_1d_shifted = df_1d[["ADX_1D", "AO_1D"]].shift(1).dropna()

    # --- 4. 核武器：使用 merge_asof 进行完美的时间轴对齐 ---
    # 这解决了玉米 15m 的 03:00 和日线的 00:00 无法 join 的问题
    # direction='backward' 保证了在任何时刻，只使用过去最新固化的数据
    df = pd.merge_asof(
        df, df_4h_shifted, left_index=True, right_index=True, direction="backward"
    )
    df = pd.merge_asof(
        df, df_1d_shifted, left_index=True, right_index=True, direction="backward"
    )

    # 回退机制：极少数情况下回填
    df["ADX_4H"] = df["ADX_4H"].ffill().bfill().fillna(20.0)
    df["ADX_1D"] = df["ADX_1D"].ffill().bfill().fillna(20.0)
    df["AO_4H"] = df["AO_4H"].ffill().bfill().fillna(0.0)
    df["AO_1D"] = df["AO_1D"].ffill().bfill().fillna(0.0)

    # 清理 NaN
    required_cols = [
        "ADX_1D",
        "ADX_4H",
        "AO_1D",
        "AO_4H",
        "AO_15m",
        "MFI_10",
        "ATR_Fast",
        "BB_Lower",
        "Donchian_Upper_20",
    ]
    clean_df = df.replace([float("inf"), float("-inf")], pd.NA)
    clean_df = clean_df.dropna(subset=required_cols)
    return clean_df


# ==========================================
# 模块 3：信号侦测与路由 (做多/做空全闭环版 + 冷静期)
# ==========================================
def check_signals(
    latest_state, config, current_price, signal_tracker, current_positions
):
    """侦测交易信号，返回 市场状态 与 信号文本

    参数:
        signal_tracker: SignalTracker 对象，用于追踪上一次信号并施加冷静期
    """
    signal_msg = ""
    candidate_signal_type = "NONE"

    # 【安全检查】验证关键指标是否有 NaN 值
    critical_indicators = [
        "ADX_1D",
        "ADX_4H",
        "AO_1D",
        "AO_4H",
        "MFI_10",
        "MFI_14",
        "MFI_20",
        "ATR_Fast",
        "ATR_Slow",
    ]
    for indicator in critical_indicators:
        if pd.isna(latest_state[indicator]):
            return "ERROR", f"[ERROR] 指标 {indicator} 计算异常 (NaN)"

    # 获取当前多单和空单的数量
    num_long_pos = (
        len([p for p in current_positions if p.type == mt5.ORDER_TYPE_BUY])
        if current_positions
        else 0
    )
    num_short_pos = (
        len([p for p in current_positions if p.type == mt5.ORDER_TYPE_SELL])
        if current_positions
        else 0
    )
    total_pos = num_long_pos + num_short_pos

    # 提取宏观状态
    adx_1d = latest_state["ADX_1D"]
    adx_4h = latest_state["ADX_4H"]
    ao_1d = latest_state["AO_1D"]
    ao_4h = latest_state["AO_4H"]

    market_regime = "MIXED"
    if adx_1d > config["macro_trend_adx"] and adx_4h > config["macro_trend_adx"]:
        market_regime = "TREND"
    elif adx_1d < config["macro_range_adx"] and adx_4h < config["macro_range_adx"]:
        market_regime = "RANGE"

    # 提取微观数据
    atr_fast = latest_state["ATR_Fast"]
    atr_slow = latest_state["ATR_Slow"]
    mfi_10, mfi_14, mfi_20 = (
        latest_state["MFI_10"],
        latest_state["MFI_14"],
        latest_state["MFI_20"],
    )
    # 波动率自适应计算
    vol_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0
    if vol_ratio > config["volatility_shock_threshold"]:
        # 强制覆盖市场状态为 SHOCK
        return (
            "SHOCK",
            f"⚠️ [CIRCUIT BREAKER] 波动率异常放大 ({vol_ratio:.2f}x)！暂停一切新开仓动作！",
        )

    # ==========================================
    # 仓位风控铁门：震荡期如果有仓位，直接熔断不处理任何新信号！
    # ==========================================
    if market_regime == "RANGE" and total_pos > 0:
        return (
            market_regime,
            f"[HOLD] 震荡环境已持有 {total_pos} 个仓位，禁止加仓等待获利/止损。",
        )
    vol_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0

    # ==========================================
    # 策略 A：震荡均值回归 (Range)
    # ==========================================
    if market_regime == "RANGE":
        # 【安全检查】验证 BB 和 ATR 指标
        bb_indicators = ["BB_Lower", "BB_Upper"]
        for ind in bb_indicators:
            if pd.isna(latest_state[ind]):
                return market_regime, f"[WARNING] 布林带指标 {ind} 暂未准备好"

        dyn_oversold = config["mfi_base_oversold"] - (vol_ratio - 1) * 10
        dyn_overbought = config["mfi_base_overbought"] + (vol_ratio - 1) * 10

        long_votes = sum(
            [mfi_10 < dyn_oversold, mfi_14 < dyn_oversold, mfi_20 < dyn_oversold]
        )
        short_votes = sum(
            [mfi_10 > dyn_overbought, mfi_14 > dyn_overbought, mfi_20 > dyn_overbought]
        )

        # --- 震荡做多逻辑 (跌破下轨抄底) ---
        if long_votes > 0 and current_price <= latest_state["BB_Lower"] * 1.001:
            stop_loss = latest_state["BB_Lower"] - 0.5 * atr_fast
            target_price = latest_state["BB_Upper"]
            risk = current_price - stop_loss
            reward = target_price - current_price
            expected_rr = reward / risk if risk > 0 else 0

            if expected_rr >= config["min_expected_rr"]:
                candidate_signal_type = "LONG"
                confidence = (long_votes / 3.0) * 100
                signal_msg = f"[LONG - RANGE] Conf:{confidence:.0f}% | RR:{expected_rr:.2f} | SL:{stop_loss:.5f} | TP:{target_price:.5f}"
            else:
                candidate_signal_type = "FILTERED"
                signal_msg = (
                    f"[FILTERED] Range Long matched, poor RR ({expected_rr:.2f})"
                )

        # --- 震荡做空逻辑 (突破上轨摸顶) ---
        elif short_votes > 0 and current_price >= latest_state["BB_Upper"] * 0.999:
            # 止损放在上轨之上，目标看向下轨
            stop_loss = latest_state["BB_Upper"] + 0.5 * atr_fast
            target_price = latest_state["BB_Lower"]
            # 做空的风险是 止损价 - 进场价，利润是 进场价 - 目标价
            risk = stop_loss - current_price
            reward = current_price - target_price
            expected_rr = reward / risk if risk > 0 else 0

            if expected_rr >= config["min_expected_rr"]:
                candidate_signal_type = "SHORT"
                confidence = (short_votes / 3.0) * 100
                signal_msg = f"[SHORT - RANGE] Conf:{confidence:.0f}% | RR:{expected_rr:.2f} | SL:{stop_loss:.5f} | TP:{target_price:.5f}"
            else:
                candidate_signal_type = "FILTERED"
                signal_msg = (
                    f"[FILTERED] Range Short matched, poor RR ({expected_rr:.2f})"
                )

    # ==========================================
    # 策略 B：趋势突破跟随 (Trend + 阶梯加仓)
    # ==========================================
    elif market_regime == "TREND":
        # 【安全检查】验证 Donchian 指标
        donchian_indicators = [
            "Donchian_Upper_20",
            "Donchian_Lower_10",
            "Donchian_Lower_20",
            "Donchian_Upper_10",
        ]
        for ind in donchian_indicators:
            if pd.isna(latest_state[ind]):
                return market_regime, f"[WARNING] Donchian指标 {ind} 暂未准备好"

        trend_direction = (
            1 if (ao_1d > 0 and ao_4h > 0) else (-1 if (ao_1d < 0 and ao_4h < 0) else 0)
        )

        # --- 顺大势做多 ---
        if trend_direction == 1:
            # 1. 首次开仓 (底仓)
            if num_long_pos == 0 and num_short_pos == 0:
                if (
                    current_price > latest_state["Donchian_Upper_20"]
                    and mfi_14 > config["trend_mfi_confirm"]
                ):
                    candidate_signal_type = "LONG"
                    stop_loss = latest_state["Donchian_Lower_10"]
                    signal_msg = f"[LONG ENTRY] 趋势突破首仓 | SL:{stop_loss:.5f}"

            # 2. 盈利加仓 (Pyramiding)
            elif num_long_pos > 0 and num_long_pos < config["max_trend_positions"]:
                # 提取当前所有多单的最高开仓价 (寻找最后一次加仓的位置)
                highest_entry = max(
                    [
                        p.price_open
                        for p in current_positions
                        if p.type == mt5.ORDER_TYPE_BUY
                    ]
                )

                # 核心加仓数学条件：当前价格 > 上次入场价 + (0.5 * ATR)
                distance_required = atr_fast * config["trend_add_pos_atr_step"]
                if current_price > highest_entry + distance_required:
                    # 再次确认动能没有衰竭
                    if mfi_14 > 50:
                        candidate_signal_type = "LONG"
                        stop_loss = latest_state["Donchian_Lower_10"]
                        signal_msg = f"🔥 [LONG ADD] 趋势顺势加仓 (#{num_long_pos + 1}) | 步长满足 | SL:{stop_loss:.5f}"
                else:
                    # 价格还在盘整，未达到加仓距离
                    pass

        # --- 顺大势做空 (逻辑是对称镜像的) ---
        elif trend_direction == -1:
            short_mfi_confirm = 100 - config["trend_mfi_confirm"]

            if num_short_pos == 0 and num_long_pos == 0:
                if (
                    current_price < latest_state["Donchian_Lower_20"]
                    and mfi_14 < short_mfi_confirm
                ):
                    candidate_signal_type = "SHORT"
                    stop_loss = latest_state["Donchian_Upper_10"]
                    signal_msg = f"[SHORT ENTRY] 趋势跌破首仓 | SL:{stop_loss:.5f}"

            elif num_short_pos > 0 and num_short_pos < config["max_trend_positions"]:
                # 空单加仓：寻找最低的开仓价，要求当前价格跌得更深
                lowest_entry = min(
                    [
                        p.price_open
                        for p in current_positions
                        if p.type == mt5.ORDER_TYPE_SELL
                    ]
                )
                distance_required = atr_fast * config["trend_add_pos_atr_step"]

                if current_price < lowest_entry - distance_required:
                    if mfi_14 < 50:
                        candidate_signal_type = "SHORT"
                        stop_loss = latest_state["Donchian_Upper_10"]
                        signal_msg = f"🔥 [SHORT ADD] 趋势顺势加仓 (#{num_short_pos + 1}) | 步长满足 | SL:{stop_loss:.5f}"

    # ==========================================
    # 冷静期检查逻辑 (使用修复后的 Tracker)
    # ==========================================
    if signal_tracker is not None and candidate_signal_type in ["LONG", "SHORT"]:
        is_cooling, cooldown_remaining = signal_tracker.is_cooldown_active(
            candidate_signal_type, config
        )

        if is_cooling:
            signal_msg = f"[COOLDOWN] {candidate_signal_type} 信号被冷静期拦截，还需等待 {cooldown_remaining} 根K线"
            candidate_signal_type = (
                "COOLDOWN"  # 临时改变状态用于打印，但不影响 tracker 的底层记忆
            )

    # 【修复重点】无论是否被拦截，把最终决定的 candidate (NONE, LONG, SHORT, COOLDOWN) 喂给 tracker
    # Tracker 内部会聪明地只记住真正的 LONG/SHORT
    if signal_tracker is not None:
        signal_tracker.update(candidate_signal_type)

    return market_regime, signal_msg


# ==========================================
# 模块 3.5：趋势舰队移动止损同步器 (Trailing Stop)
# ==========================================
def update_trailing_stops(latest_state, current_positions, market_regime):
    """
    检查并更新当前持仓的移动止损线
    """
    if not current_positions or market_regime != "TREND":
        return None

    # 提取最新的防守线
    long_defense_line = latest_state["Donchian_Lower_10"]  # 多头防守线
    short_defense_line = latest_state["Donchian_Upper_10"]  # 空头防守线

    # 新品种/历史不足时 Donchian 可能尚未就绪，避免 NaN 比较导致异常逻辑。
    if pd.isna(long_defense_line) or pd.isna(short_defense_line):
        return "[TRAILING STOP SKIPPED] Donchian defense line not ready (NaN)."

    updates_msg = []

    for pos in current_positions:
        ticket = pos.ticket
        current_sl = pos.sl

        # 处理多单 (BUY)
        if pos.type == mt5.ORDER_TYPE_BUY:
            # 只有当新的防守线 高于 当前的止损线时，才允许上移止损 (止损只能顺势移动，不能倒退)
            if not pd.isna(current_sl) and long_defense_line > current_sl:
                # [TODO: 调用 mt5.order_send 发送修改 SL 的指令]
                updates_msg.append(
                    f"多单 #{ticket} SL 上移: {current_sl:.5f} -> {long_defense_line:.5f}"
                )

        # 处理空单 (SELL)
        elif pos.type == mt5.ORDER_TYPE_SELL:
            # 只有当新的防守线 低于 当前的止损线时，才允许下移止损
            # 注意：如果原本没有止损 (sl==0)，也必须挂上
            if pd.isna(current_sl):
                current_sl = 0.0
            if current_sl == 0.0 or short_defense_line < current_sl:
                # [TODO: 调用 mt5.order_send 发送修改 SL 的指令]
                updates_msg.append(
                    f"空单 #{ticket} SL 下移: {current_sl:.5f} -> {short_defense_line:.5f}"
                )

    if updates_msg:
        return "[TRAILING STOP UPDATED]\n" + "\n".join(updates_msg)

    return None


# ==========================================
# 模块 4：HUD 仪表盘文件桥接
# ==========================================
def update_mt5_dashboard(symbol_snapshots):
    """
    Returns a formatted string for the FloatingHUD and updates the MT5 file.
    """
    terminal_info = mt5.terminal_info()

    def explain_regime(latest_state, config, market_regime):
        adx_1d = latest_state["ADX_1D"]
        adx_4h = latest_state["ADX_4H"]
        trend_th = config["macro_trend_adx"]
        range_th = config["macro_range_adx"]

        if market_regime == "TREND":
            return (
                f"TREND because ADX_1D={adx_1d:.2f}>{trend_th:.2f} and "
                f"ADX_4H={adx_4h:.2f}>{trend_th:.2f}"
            )
        if market_regime == "RANGE":
            return (
                f"RANGE because ADX_1D={adx_1d:.2f}<{range_th:.2f} and "
                f"ADX_4H={adx_4h:.2f}<{range_th:.2f}"
            )
        return (
            f"MIXED because ADXs not aligned: ADX_1D={adx_1d:.2f}, "
            f"ADX_4H={adx_4h:.2f}, trend>{trend_th:.2f}, range<{range_th:.2f}"
        )

    def explain_ao(latest_state):
        ao_1d = latest_state["AO_1D"]
        ao_4h = latest_state["AO_4H"]
        ao_15m = latest_state["AO_15m"]

        if ao_1d > 0 and ao_4h > 0:
            macro_bias = "Macro bullish bias (1D & 4H both > 0)"
        elif ao_1d < 0 and ao_4h < 0:
            macro_bias = "Macro bearish bias (1D & 4H both < 0)"
        else:
            macro_bias = "Macro mixed bias (1D and 4H disagree)"

        if ao_15m > 0:
            micro_momentum = "15m momentum currently up"
        elif ao_15m < 0:
            micro_momentum = "15m momentum currently down"
        else:
            micro_momentum = "15m momentum neutral"

        return f"{macro_bias}; {micro_momentum}"

    local_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. Build the Dashboard String (for MT5 file) ---
    text = "========================================\n"
    text += "      QUANT ENGINE: MULTI-ASSET         \n"
    text += "========================================\n"
    text += f"Local Time: {local_time_str}\n"
    text += "========================================\n"

    hud_panels = {
        "left": "USDCNH\nWaiting for data...",
        "right": "CORN\nWaiting for data...",
    }

    for symbol, snapshot in symbol_snapshots.items():
        text += f"[{symbol}]\n"

        panel_lines = [f"[{symbol}]"]

        if "error" in snapshot:
            text += f"Local Time: {local_time_str}\n"
            text += f"Status    : {snapshot['error']}\n"
            text += "----------------------------------------\n"
            panel_lines.append(f"Local Time: {local_time_str}")
            panel_lines.append(f"Status   : {snapshot['error']}")
            panel_text = "\n".join(panel_lines)
            if "USDCNH" in symbol.upper():
                hud_panels["left"] = panel_text
            else:
                hud_panels["right"] = panel_text
            continue

        latest_state = snapshot["latest_state"]
        market_regime = snapshot["market_regime"]
        signal_msg = snapshot["signal_msg"]
        current_price = snapshot["current_price"]
        signal_tracker = snapshot["signal_tracker"]
        vol_ratio = (
            latest_state["ATR_Fast"] / latest_state["ATR_Slow"]
            if latest_state["ATR_Slow"] > 0
            else 1.0
        )
        regime_reason = explain_regime(latest_state, CONFIG, market_regime)
        ao_reason = explain_ao(latest_state)

        text += f"Local Time: {local_time_str}\n"
        text += f"Bar Time  : {snapshot['bar_time']}\n"
        text += f"Regime    : [{market_regime}]\n"
        text += f"Reason    : {regime_reason}\n"
        text += f"Last Px   : {current_price:.5f}\n"
        text += f"AO 1D/4H/15m: {latest_state['AO_1D']:.5f} / {latest_state['AO_4H']:.5f} / {latest_state['AO_15m']:.5f}\n"
        text += f"AO Explain: {ao_reason}\n"
        text += "[Macro 1D/4H]\n"
        text += f"ADX 1D: {latest_state['ADX_1D']:>6.2f} | AO 1D: {latest_state['AO_1D']:>7.5f}\n"
        text += f"ADX 4H: {latest_state['ADX_4H']:>6.2f} | AO 4H: {latest_state['AO_4H']:>7.5f}\n"
        text += f"ADX Thresholds -> TREND>{CONFIG['macro_trend_adx']:.2f}, RANGE<{CONFIG['macro_range_adx']:.2f}\n"
        text += "[Micro 15m]\n"
        text += f"MFI 10/14/20: {latest_state['MFI_10']:.2f} / {latest_state['MFI_14']:.2f} / {latest_state['MFI_20']:.2f}\n"
        text += f"ATR Fast/Slow: {latest_state['ATR_Fast']:.5f} / {latest_state['ATR_Slow']:.5f} | Ratio: {vol_ratio:.2f}x\n"
        text += f"B-Bands : {latest_state['BB_Lower']:.5f} - {latest_state['BB_Upper']:.5f}\n"
        text += (
            f"Donchian U20/L10/L20/U10: {latest_state['Donchian_Upper_20']:.5f} / {latest_state['Donchian_Lower_10']:.5f} / "
            f"{latest_state['Donchian_Lower_20']:.5f} / {latest_state['Donchian_Upper_10']:.5f}\n"
        )

        cooldown_info = f"Last Signal: {signal_tracker.last_executed_signal}"
        if signal_tracker.last_executed_signal in ["LONG", "SHORT"]:
            cooldown_info += f" (Bar #{signal_tracker.signal_bar_count})"
        text += f"Cooldown  : {cooldown_info}\n"

        model_inference = snapshot.get("model_inference")
        if model_inference:
            if "error" in model_inference:
                text += f"ML Model  : {model_inference['error']}\n"
            else:
                def _sig_tag(s):
                    return "sure" if s < 0.20 else ("ok" if s < 0.40 else "unsure")
                text += (
                    f"ML 15m    : mu={model_inference['mu_15m']:+.4f}, "
                    f"sigma={model_inference['sigma_15m']:.4f} [{_sig_tag(model_inference['sigma_15m'])}]\n"
                )
                text += (
                    f"ML 1H     : mu={model_inference['mu_1h']:+.4f}, "
                    f"sigma={model_inference['sigma_1h']:.4f} [{_sig_tag(model_inference['sigma_1h'])}]\n"
                )
                text += (
                    f"ML 4H     : mu={model_inference['mu_4h']:+.4f}, "
                    f"sigma={model_inference['sigma_4h']:.4f} [{_sig_tag(model_inference['sigma_4h'])}]\n"
                )
        text += "----------------------------------------\n"

        panel_lines.append(f"Local Time: {local_time_str}")
        panel_lines.append(f"Bar Time : {snapshot['bar_time']}")
        panel_lines.append(f"Regime   : [{market_regime}]")
        panel_lines.append(f"Reason   : {regime_reason}")
        panel_lines.append(f"Last Px  : {current_price:.5f}")
        panel_lines.append(
            f"AO 1D/4H/15m: {latest_state['AO_1D']:.5f} / {latest_state['AO_4H']:.5f} / {latest_state['AO_15m']:.5f}"
        )
        panel_lines.append(f"AO Explain: {ao_reason}")
        panel_lines.append(
            f"ADX 1D/4H : {latest_state['ADX_1D']:.2f} / {latest_state['ADX_4H']:.2f}"
        )
        panel_lines.append(
            f"MFI 10/14/20: {latest_state['MFI_10']:.2f} / {latest_state['MFI_14']:.2f} / {latest_state['MFI_20']:.2f}"
        )
        panel_lines.append(
            f"ATR F/S/R : {latest_state['ATR_Fast']:.5f} / {latest_state['ATR_Slow']:.5f} / {vol_ratio:.2f}x"
        )
        panel_lines.append(
            f"B-Bands  : {latest_state['BB_Lower']:.5f} - {latest_state['BB_Upper']:.5f}"
        )
        panel_lines.append(
            f"Donch U20/L10: {latest_state['Donchian_Upper_20']:.5f} / {latest_state['Donchian_Lower_10']:.5f}"
        )
        panel_lines.append(
            f"Donch L20/U10: {latest_state['Donchian_Lower_20']:.5f} / {latest_state['Donchian_Upper_10']:.5f}"
        )
        panel_lines.append(f"Cooldown : {cooldown_info}")

        if model_inference:
            if "error" in model_inference:
                panel_lines.append(f"ML       : {model_inference['error']}")
            else:
                def _sig_tag(s):
                    return "sure" if s < 0.20 else ("ok" if s < 0.40 else "unsure")
                panel_lines.append(
                    f"ML Sig   : {model_inference['direction']} (p={model_inference['direction_prob']:.3f})"
                )
                panel_lines.append(
                    f"ML 15m   : mu={model_inference['mu_15m']:+.4f} σ={model_inference['sigma_15m']:.4f} [{_sig_tag(model_inference['sigma_15m'])}]"
                )
                panel_lines.append(
                    f"ML 1H    : mu={model_inference['mu_1h']:+.4f} σ={model_inference['sigma_1h']:.4f} [{_sig_tag(model_inference['sigma_1h'])}]"
                )
                panel_lines.append(
                    f"ML 4H    : mu={model_inference['mu_4h']:+.4f} σ={model_inference['sigma_4h']:.4f} [{_sig_tag(model_inference['sigma_4h'])}]"
                )

        panel_text = "\n".join(panel_lines)
        if "USDCNH" in symbol.upper():
            hud_panels["left"] = panel_text
        else:
            hud_panels["right"] = panel_text

    # --- 2. Keep the File Update (Optional) ---
    if terminal_info:
        file_path = os.path.join(
            terminal_info.data_path, "MQL5", "Files", "python_dashboard.txt"
        )
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass  # Silent fail for file if folder isn't ready

    # --- 3. Return both file text and panel payload for FloatingHUD ---
    return text, hud_panels


# ==========================================
# 模块 6：自动交易执行接口 (TODO / Placeholder)
# ==========================================
def execute_trade(
    symbol, order_type, lot_size, sl_price, tp_price=0.0, comment="Quant_Engine"
):
    """
    [TODO] 发送交易指令到 MT5 服务器

    参数说明:
    :param symbol: 交易品种 (如 "USDCNH")
    :param order_type: 订单类型 (mt5.ORDER_TYPE_BUY 或 mt5.ORDER_TYPE_SELL)
    :param lot_size: 交易手数 (需根据 ATR 和总资金 2% 动态计算)
    :param sl_price: 绝对止损价格 (必须填写，风控底线)
    :param tp_price: 绝对止盈价格 (震荡市填入目标价，趋势市填 0 走移动止盈)
    :param comment: 订单注释，方便在历史记录里复盘
    """

    # 1. 获取当前最新 Tick 数据以确定买卖价格
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ [发单失败] 无法获取 {symbol} 的最新 Tick 数据")
        return False

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    # 2. 构建 MT5 标准下单请求字典 (Request Dictionary)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,  # 市价单执行
        "symbol": symbol,
        "volume": float(lot_size),  # 必须是浮点数，且符合 Broker 的步长设定
        "type": order_type,
        "price": price,
        "sl": float(sl_price),  # 硬止损
        "tp": float(tp_price),  # 止盈 (如果为 0 则不设止盈)
        "deviation": 20,  # 允许的最大滑点 (单位: points)
        "magic": 88889999,  # [重要] 魔术码，用于让 EA 识别哪些单子是自己开的
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,  # 订单有效期: 取消前有效
        # [TODO 警报] 不同的券商支持的填充模式不同！
        # 如果发单报错 "Unsupported filling mode"，请将下方修改为:
        # mt5.ORDER_FILLING_IOC 或 mt5.ORDER_FILLING_RETURN
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    # ==========================================
    # ⚠️ 安全锁 (Safety Switch) ⚠️
    # 在你确认信号准确率达到预期之前，请保持下面的 return 处于激活状态。
    # 这样程序只会打印拟下单信息，不会真的扣动扳机。
    # ==========================================

    print(
        f"\n🛑 [安全锁拦截] 模拟下单报文 -> {request['type']} | 手数: {request['volume']} | SL: {request['sl']} | TP: {request['tp']}"
    )
    return True

    # --- 以下为真实下单代码，准备好实盘时解除下方注释 ---

    """
    # 3. 发送订单并接收结果
    result = mt5.order_send(request)
    
    # 4. 检查结果并打印日志
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ [实盘发单报错] 错误码: {result.retcode}, 描述: {result.comment}")
        return False
        
    print(f"✅ [实盘发单成功] 订单号: {result.order}, 成交价: {result.price}, 手数: {result.volume}")
    return True
    """


# ==========================================
# 模块 5：主循环 (心脏起搏器)
# ==========================================
def main():
    validate_config(CONFIG)
    hud = FloatingHUD()
    dashboard_panels = {
        "left": "USDCNH\nInitializing Engine...",
        "right": "CORN\nInitializing Engine...",
    }
    if not mt5.initialize():
        print(f"MT5 Initialization Failed: {mt5.last_error()}")
        return

    fx_symbol = resolve_available_symbol(SYMBOL_PREFERENCES["FX"])
    corn_symbol = resolve_available_symbol(SYMBOL_PREFERENCES["CORN"])

    active_symbols = [s for s in [fx_symbol, corn_symbol] if s is not None]
    if not active_symbols:
        print("❌ No valid symbols found for USDCNH/CORN in this MT5 terminal.")
        mt5.shutdown()
        return

    print(
        f"[{datetime.now()}] === Quant Engine Started -> Targets: {active_symbols} ==="
    )

    model_runner = LiveModelRunner()
    for asset, status in model_runner.status.items():
        print(f"[ML:{asset}] {status}")

    last_processed_time = {symbol: None for symbol in active_symbols}
    signal_trackers = {symbol: SignalTracker() for symbol in active_symbols}
    raw_cache = {symbol: None for symbol in active_symbols}
    symbol_snapshots = {
        symbol: {"error": "Waiting for first bar..."} for symbol in active_symbols
    }
    is_running = True

    def upsert_raw_cache(symbol):
        """Maintain a rolling M15 cache, fetching only bars newer than the last cached bar."""
        if raw_cache[symbol] is None:
            df_seed = get_mt5_data(symbol, mt5.TIMEFRAME_M15, M15_HISTORY_BARS)
            raw_cache[symbol] = df_seed
            return raw_cache[symbol]

        # Anchor fetch to the last bar's timestamp so there are no gaps or overlaps.
        last_ts = raw_cache[symbol].index[-1]
        # copy_rates_range is inclusive on both ends; add 1 second to exclude the
        # already-cached last bar, then fetch up to now + a small buffer.
        from_dt = last_ts + pd.Timedelta(seconds=1)
        to_dt = pd.Timestamp.utcnow() + pd.Timedelta(minutes=20)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, from_dt, to_dt)
        if rates is None or len(rates) == 0:
            return raw_cache[symbol]

        df_new = pd.DataFrame(rates)
        df_new["time"] = pd.to_datetime(df_new["time"], unit="s")
        df_new.set_index("time", inplace=True)
        if "tick_volume" in df_new.columns:
            df_new.rename(columns={"tick_volume": "volume"}, inplace=True)
        elif "real_volume" in df_new.columns:
            df_new["volume"] = df_new["real_volume"]
        else:
            df_new["volume"] = 1.0
        df_new.sort_index(inplace=True)

        merged = pd.concat([raw_cache[symbol], df_new])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        raw_cache[symbol] = merged.tail(M15_HISTORY_BARS)
        return raw_cache[symbol]

    def on_close():
        nonlocal is_running
        is_running = False
        try:
            if hud.root.winfo_exists():
                hud.root.destroy()
        except tk.TclError:
            pass

    hud.root.protocol("WM_DELETE_WINDOW", on_close)

    def process_cycle():
        nonlocal dashboard_panels
        if not is_running:
            return

        try:
            hud.refresh(dashboard_panels)
            dashboard_needs_refresh = False

            for symbol in active_symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1)
                if rates is None or len(rates) == 0:
                    symbol_snapshots[symbol] = {
                        "error": "No M15 data returned from MT5"
                    }
                    dashboard_needs_refresh = True
                    continue

                current_bar_time = rates[0]["time"]

                # 探测到 15m 新 K 线产生
                if last_processed_time[symbol] != current_bar_time:
                    # 维护滚动历史，避免每次都全量拉取超长区间。
                    df_raw = upsert_raw_cache(symbol)

                    # [新增] 并行获取原生的 4H 和 1D 数据 (只需拉取足够算指标的根数，如100根)
                    # 这彻底解决了长周期计算需要拉取几万根 15m K线的问题
                    df_4h = get_mt5_data(symbol, mt5.TIMEFRAME_H4, 100)
                    df_1d = get_mt5_data(symbol, mt5.TIMEFRAME_D1, 100)

                    if (
                        df_raw is None
                        or len(df_raw) < 200
                        or df_4h is None
                        or df_1d is None
                    ):
                        symbol_snapshots[symbol] = {
                            "error": "Insufficient native history (M15, H4, or D1)"
                        }
                        last_processed_time[symbol] = current_bar_time
                        dashboard_needs_refresh = True
                        continue

                    # 0. 查账户底牌 (获取当前品种的所有持仓)
                    current_positions = mt5.positions_get(symbol=symbol)
                    if current_positions is None:  # MT5 可能返回 None，防错处理
                        current_positions = ()

                    # 1. 计算所有指标
                    strategy_data = build_multi_timeframe_features(df_raw, df_4h, df_1d)
                    if strategy_data is None or len(strategy_data) < 2:
                        symbol_snapshots[symbol] = {
                            "error": f"Feature pipeline not ready (rows={0 if strategy_data is None else len(strategy_data)}, raw={len(df_raw)})"
                        }
                        last_processed_time[symbol] = current_bar_time
                        dashboard_needs_refresh = True
                        continue

                    # 2. 提取最新固化状态 (倒数第二根K线) 和 最新Tick价格
                    latest_state = strategy_data.iloc[-2]
                    current_price = df_raw.iloc[-1]["close"]

                    # 3. 策略路由，产生信号（包括冷静期检查）
                    market_regime, signal_msg = check_signals(
                        latest_state,
                        CONFIG,
                        current_price,
                        signal_trackers[symbol],
                        current_positions,
                    )
                    ts_msg = update_trailing_stops(
                        latest_state, current_positions, market_regime
                    )
                    if ts_msg:
                        print(f"\n[🛡️ {symbol} 风控系统] {ts_msg}")

                    # 4. 打印到控制台
                    print(f"\n[{symbol}] [{latest_state.name}] Regime: {market_regime}")
                    if signal_msg:
                        print(f">>> {signal_msg}")

                    asset_key = infer_asset_key(symbol)
                    model_inference = model_runner.predict(asset_key, strategy_data)
                    if "error" in model_inference:
                        print(f"[ML:{symbol}] {model_inference['error']}")
                    else:
                        print(
                            f"[ML:{symbol}] {model_inference['direction']} "
                            f"mu_15m={model_inference['direction_prob']:.4f} | "
                            f"15m mu={model_inference['mu_15m']:+.4f} σ={model_inference['sigma_15m']:.4f} | "
                            f"1H mu={model_inference['mu_1h']:+.4f} σ={model_inference['sigma_1h']:.4f} | "
                            f"4H mu={model_inference['mu_4h']:+.4f} σ={model_inference['sigma_4h']:.4f}"
                        )

                    symbol_snapshots[symbol] = {
                        "bar_time": latest_state.name,
                        "market_regime": market_regime,
                        "signal_msg": signal_msg,
                        "latest_state": latest_state,
                        "current_price": current_price,
                        "signal_tracker": signal_trackers[symbol],
                        "model_inference": model_inference,
                    }

                    last_processed_time[symbol] = current_bar_time
                    dashboard_needs_refresh = True

            if dashboard_needs_refresh:
                _, dashboard_panels = update_mt5_dashboard(symbol_snapshots)

        except KeyboardInterrupt:
            print("\nShutting down MT5 connection...")
            on_close()
        except Exception as e:
            print(f"Runtime Error: {e}")
            traceback.print_exc()
        finally:
            if is_running:
                hud.root.after(ENGINE_TICK_MS, process_cycle)

    process_cycle()
    hud.root.mainloop()
    mt5.shutdown()


if __name__ == "__main__":
    main()
