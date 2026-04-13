from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


CSV_COLUMNS = [
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
    "<TICKVOL>",
    "<VOL>",
    "<SPREAD>",
]


@dataclass
class UpdateTarget:
    file_path: Path
    symbol_candidates: list[str]


TARGETS = [
    UpdateTarget(
        file_path=Path(
            r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\USDCNH_M15_202401020000_202604131415.csv"
        ),
        symbol_candidates=["USDCNH", "USD.CNH", "USDCNH."],
    ),
    UpdateTarget(
        file_path=Path(
            r"C:\Users\a6744\OneDrive - Terranet AB\Dokument\CORN.c_M15_202303301000_202604131415.csv"
        ),
        symbol_candidates=["CORN.c", "CORN"],
    ),
]


def read_mt5_style_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    missing = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {file_path}: {missing}")

    df["_dt"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"], format="%Y.%m.%d %H:%M:%S")
    return df


def resolve_symbol(symbol_candidates: list[str]) -> str | None:
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return None

    names = [s.name for s in all_symbols]
    upper_map = {name.upper(): name for name in names}

    for candidate in symbol_candidates:
        if candidate.upper() in upper_map:
            return upper_map[candidate.upper()]

    for candidate in symbol_candidates:
        c_up = candidate.upper()
        starts = [name for name in names if name.upper().startswith(c_up)]
        if starts:
            return starts[0]

    return None


def mt5_rates_to_df(rates) -> pd.DataFrame:
    raw = pd.DataFrame(rates)
    if raw.empty:
        return pd.DataFrame(columns=CSV_COLUMNS + ["_dt"])

    dt_index = pd.to_datetime(raw["time"], unit="s")
    out = pd.DataFrame(
        {
            "_dt": dt_index,
            "<DATE>": dt_index.dt.strftime("%Y.%m.%d"),
            "<TIME>": dt_index.dt.strftime("%H:%M:%S"),
            "<OPEN>": raw["open"],
            "<HIGH>": raw["high"],
            "<LOW>": raw["low"],
            "<CLOSE>": raw["close"],
            "<TICKVOL>": raw["tick_volume"].astype("int64"),
            "<VOL>": raw["real_volume"].astype("int64"),
            "<SPREAD>": raw["spread"].astype("int64"),
        }
    )
    return out


def update_one(target: UpdateTarget) -> None:
    if not target.file_path.exists():
        raise FileNotFoundError(f"CSV not found: {target.file_path}")

    existing = read_mt5_style_csv(target.file_path)
    if existing.empty:
        raise ValueError(f"CSV is empty: {target.file_path}")

    symbol = resolve_symbol(target.symbol_candidates)
    if not symbol:
        print(f"[WARN] Could not resolve symbol for {target.file_path.name}; skipped")
        return

    # Fetch last 300 bars (~5 hours of M15 data) to ensure we capture any new bars
    # This avoids datetime/timezone issues with copy_rates_range
    print(f"[DEBUG] {target.file_path.name}: fetching last 300 bars for {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 3000)
    if rates is None or len(rates) == 0:
        print(f"[WARN] MT5 returned None or empty for {symbol}; skipped")
        return

    print(f"[DEBUG] MT5 returned {len(rates)} bars")
    incoming = mt5_rates_to_df(rates)
    if not incoming.empty:
        print(f"[DEBUG] Incoming data range: {incoming['_dt'].min()} to {incoming['_dt'].max()}")
    
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["_dt"], keep="last")
    merged = merged.sort_values("_dt").reset_index(drop=True)

    before_rows = len(existing)
    after_rows = len(merged)
    new_rows = after_rows - before_rows

    merged[CSV_COLUMNS].to_csv(target.file_path, sep="\t", index=False)
    print(
        f"[OK] {target.file_path.name} | symbol={symbol} | rows: {before_rows} -> {after_rows} (delta={new_rows}) | last={merged['_dt'].max()}"
    )


def main() -> None:
    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {code} {msg}")

    try:
        for target in TARGETS:
            update_one(target)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()