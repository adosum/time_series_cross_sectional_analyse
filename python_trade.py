import ctypes  # Needed for high-res DPI awareness
import os
import time
import tkinter as tk
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
        # Widened to 650 to prevent text clipping at larger font sizes
        self.root.geometry("650x850")
        self.root.configure(bg="#1e1e1e")  # Darker background for pro look

        # --- 3. Larger, Sharper Font ---
        # Size 14 is the "sweet spot" for Consolas on a 2K monitor
        self.label = tk.Label(
            self.root,
            text="Waiting for data...",
            font=("Consolas", 14, "bold"),  # Added Bold for extra clarity
            bg="#1e1e1e",
            fg="#00FF00",  # "Matrix Green" text looks great on dark
            justify=tk.LEFT,
            anchor="nw",
            padx=20,
            pady=20,
        )
        self.label.pack(expand=True, fill="both")

    def refresh(self, content=None):
        if content:
            # We strip extra whitespace to keep the layout tight
            self.label.config(text=content.strip())

        try:
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
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


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

    return df.dropna()


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

    updates_msg = []

    for pos in current_positions:
        ticket = pos.ticket
        current_sl = pos.sl

        # 处理多单 (BUY)
        if pos.type == mt5.ORDER_TYPE_BUY:
            # 只有当新的防守线 高于 当前的止损线时，才允许上移止损 (止损只能顺势移动，不能倒退)
            if long_defense_line > current_sl:
                # [TODO: 调用 mt5.order_send 发送修改 SL 的指令]
                updates_msg.append(
                    f"多单 #{ticket} SL 上移: {current_sl:.5f} -> {long_defense_line:.5f}"
                )

        # 处理空单 (SELL)
        elif pos.type == mt5.ORDER_TYPE_SELL:
            # 只有当新的防守线 低于 当前的止损线时，才允许下移止损
            # 注意：如果原本没有止损 (sl==0)，也必须挂上
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
def update_mt5_dashboard(latest_state, market_regime, signal_msg, signal_tracker=None):
    """
    Returns a formatted string for the FloatingHUD and updates the MT5 file.
    """
    terminal_info = mt5.terminal_info()

    # --- 1. Build the Dashboard String ---
    text = "========================================\n"
    text += "           QUANT ENGINE: ACTIVE         \n"
    text += "========================================\n"
    text += f"Local Time: {datetime.now().strftime('%H:%M:%S')}\n"
    text += f"Bar Time  : {latest_state.name}\n"
    text += f"Regime    : [{market_regime}]\n"
    text += "----------------------------------------\n"
    text += "[Macro (1D & 4H)]\n"
    text += f"ADX 1D: {latest_state['ADX_1D']:>6.2f} | AO 1D: {latest_state['AO_1D']:>7.5f}\n"
    text += f"ADX 4H: {latest_state['ADX_4H']:>6.2f} | AO 4H: {latest_state['AO_4H']:>7.5f}\n"
    text += "----------------------------------------\n"
    text += "[Micro (15m)]\n"
    text += f"MFI(14): {latest_state['MFI_14']:>6.2f}\n"
    text += f"ATR(14): {latest_state['ATR_Fast']:>7.5f}\n"
    text += (
        f"B-Bands: {latest_state['BB_Lower']:.5f} - {latest_state['BB_Upper']:.5f}\n"
    )
    text += "========================================\n"

    if signal_tracker is not None:
        cooldown_info = f"Last Signal: {signal_tracker.last_executed_signal}"
        if signal_tracker.last_executed_signal in ["LONG", "SHORT"]:
            cooldown_info += f" (Bar #{signal_tracker.signal_bar_count})"
        text += f"[Cooldown Status]\n{cooldown_info}\n"
        text += "----------------------------------------\n"

    action_text = signal_msg if signal_msg else "SCANNING..."
    text += f"ACTION: {action_text}\n"

    # --- 2. Keep the File Update (Optional) ---
    if terminal_info:
        file_path = os.path.join(
            terminal_info.data_path, "MQL5", "Files", "python_dashboard.txt"
        )
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            pass  # Silent fail for file if folder isn't ready

    # --- 3. Return the string for the FloatingHUD ---
    return text


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
    dashboard_string = "Initializing Engine..."
    if not mt5.initialize():
        print(f"MT5 Initialization Failed: {mt5.last_error()}")
        return

    symbol = "USDCNH"
    print(f"[{datetime.now()}] === Quant Engine Started -> Target: {symbol} ===")
    mt5.symbol_select(symbol, True)

    last_processed_time = None
    signal_tracker = SignalTracker()  # 初始化冷静期追踪器

    while True:
        try:
            hud.refresh(dashboard_string)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1)
            if rates is None or len(rates) == 0:
                continue

            current_bar_time = rates[0]["time"]

            # 探测到 15m 新 K 线产生
            if last_processed_time != current_bar_time:
                # 提取过去 1500 根 15m K线 (约15个交易日，确保1D指标有足够数据计算)
                df_raw = get_mt5_data(symbol, mt5.TIMEFRAME_M15, 8000)
                if df_raw is not None:
                    # 0. 查账户底牌 (获取当前品种的所有持仓)
                    current_positions = mt5.positions_get(symbol=symbol)
                    if current_positions is None:  # MT5 可能返回 None，防错处理
                        current_positions = ()

                    # 1. 计算所有指标
                    strategy_data = build_multi_timeframe_features(df_raw)

                    # 2. 提取最新固化状态 (倒数第二根K线) 和 最新Tick价格
                    latest_state = strategy_data.iloc[-2]
                    current_price = df_raw.iloc[-1]["close"]

                    # 3. 策略路由，产生信号（包括冷静期检查）
                    market_regime, signal_msg = check_signals(
                        latest_state,
                        CONFIG,
                        current_price,
                        signal_tracker,
                        current_positions,
                    )
                    ts_msg = update_trailing_stops(
                        latest_state, current_positions, market_regime
                    )
                    if ts_msg:
                        print(f"\n[🛡️ 风控系统] {ts_msg}")
                    # 4. 打印到控制台
                    print(f"\n[{latest_state.name}] Regime: {market_regime}")
                    if signal_msg:
                        print(f">>> {signal_msg}")

                    # 5. 发送数据给 MT5 图表显示器（并传递冷静期状态）
                    dashboard_string = update_mt5_dashboard(
                        latest_state, market_regime, signal_msg, signal_tracker
                    )

                last_processed_time = current_bar_time

            for _ in range(10):
                hud.refresh()  # Just process window events (clicks/drags)
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nShutting down MT5 connection...")
            break
        except Exception as e:
            print(f"Runtime Error: {e}")
            traceback.print_exc()
            time.sleep(5)

    mt5.shutdown()


if __name__ == "__main__":
    main()
