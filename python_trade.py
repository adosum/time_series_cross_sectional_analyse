import ctypes  # Needed for high-res DPI awareness
import os
import tkinter as tk
import tkinter.font as tkfont
import traceback
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
from ta.momentum import AwesomeOscillatorIndicator

# 引入纯血 ta 库 (无需 C++ 编译)
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator


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
# 模块 2：多周期特征工程 (核心数据工厂)
# ==========================================
def build_multi_timeframe_features(df_15m):
    df = df_15m.copy()

    # --- 1. 计算 15m 微观指标 ---
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

    # 部分期货/差价合约在MT5里 volume 质量较差，MFI可能全NaN；回退为中性值
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

    df["Donchian_Lower_20"] = df["low"].rolling(window=20).min().shift(1)  # 做空跌破线
    df["Donchian_Upper_10"] = df["high"].rolling(window=10).max().shift(1)  # 做空防守线

    # --- 2. 向上合成 4H 数据 ---
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_4h = df.resample("4h").agg(agg_dict).dropna()
    df_4h["ADX_4H"] = ADXIndicator(
        high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], window=14
    ).adx()
    df_4h["AO_4H"] = AwesomeOscillatorIndicator(
        window1=5, window2=34, high=df_4h["high"], low=df_4h["low"]
    ).awesome_oscillator()

    # --- 3. 向上合成 1D 数据 ---
    df_1d = df.resample("D").agg(agg_dict).dropna()
    df_1d["ADX_1D"] = ADXIndicator(
        high=df_1d["high"], low=df_1d["low"], close=df_1d["close"], window=14
    ).adx()
    df_1d["AO_1D"] = AwesomeOscillatorIndicator(
        window1=5, window2=34, high=df_1d["high"], low=df_1d["low"]
    ).awesome_oscillator()

    # --- 4. 跨周期安全对齐 (防未来函数) ---
    df_4h_shifted = df_4h[["ADX_4H", "AO_4H"]].shift(1)
    df_1d_shifted = df_1d[["ADX_1D", "AO_1D"]].shift(1)

    df = df.join(df_4h_shifted).ffill()
    df = df.join(df_1d_shifted).ffill()

    # 某些合约高周期指标可能长期缺失，回退到中性值避免全量过滤
    df["ADX_4H"] = df["ADX_4H"].ffill().bfill().fillna(20.0)
    df["ADX_1D"] = df["ADX_1D"].ffill().bfill().fillna(20.0)
    df["AO_4H"] = df["AO_4H"].ffill().bfill().fillna(0.0)
    df["AO_1D"] = df["AO_1D"].ffill().bfill().fillna(0.0)

    # 仅按策略关键列做过滤，避免因为无关列NaN导致整表被清空
    required_cols = [
        "ADX_1D",
        "ADX_4H",
        "AO_1D",
        "AO_4H",
        "MFI_10",
        "MFI_14",
        "MFI_20",
        "ATR_Fast",
        "ATR_Slow",
        "BB_Lower",
        "BB_Upper",
        "Donchian_Upper_20",
        "Donchian_Lower_10",
        "Donchian_Lower_20",
        "Donchian_Upper_10",
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

        action_text = signal_msg if signal_msg else "SCANNING..."
        text += f"Action    : {action_text}\n"
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
        panel_lines.append(f"Action   : {action_text}")

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

    last_processed_time = {symbol: None for symbol in active_symbols}
    signal_trackers = {symbol: SignalTracker() for symbol in active_symbols}
    raw_cache = {symbol: None for symbol in active_symbols}
    symbol_snapshots = {
        symbol: {"error": "Waiting for first bar..."} for symbol in active_symbols
    }
    is_running = True

    def upsert_raw_cache(symbol):
        """Maintain a rolling M15 cache instead of re-fetching huge history every cycle."""
        if raw_cache[symbol] is None:
            df_seed = get_mt5_data(symbol, mt5.TIMEFRAME_M15, M15_HISTORY_BARS)
            raw_cache[symbol] = df_seed
            return raw_cache[symbol]

        df_new = get_mt5_data(symbol, mt5.TIMEFRAME_M15, M15_INCREMENT_BARS)
        if df_new is None or len(df_new) == 0:
            return raw_cache[symbol]

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
                    if df_raw is None or len(df_raw) < 200:
                        symbol_snapshots[symbol] = {"error": "Insufficient M15 history"}
                        last_processed_time[symbol] = current_bar_time
                        dashboard_needs_refresh = True
                        continue

                    # 0. 查账户底牌 (获取当前品种的所有持仓)
                    current_positions = mt5.positions_get(symbol=symbol)
                    if current_positions is None:  # MT5 可能返回 None，防错处理
                        current_positions = ()

                    # 1. 计算所有指标
                    strategy_data = build_multi_timeframe_features(df_raw)
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

                    symbol_snapshots[symbol] = {
                        "bar_time": latest_state.name,
                        "market_regime": market_regime,
                        "signal_msg": signal_msg,
                        "latest_state": latest_state,
                        "current_price": current_price,
                        "signal_tracker": signal_trackers[symbol],
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
