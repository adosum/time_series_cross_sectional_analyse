import datetime as dt
import os

import akshare as ak
import pandas as pd
import yfinance as yf

pd.options.mode.copy_on_write = True

DATA_DIR = "data"

YF_SERIES = {
    "china_a50": "XIN9.FGI",
    "cbot_corn": "ZC=F",
    "gold_spot": "GC=F",
    "crude_oil_wti": "CL=F",
    "usd_cnh": "CNH=X",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_yf_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )
    drop_cols = [
        c
        for c in ["Date", "Dividends", "Stock Splits", "Capital Gains"]
        if c in df.columns
    ]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    df["date"] = df["date"].astype(str)
    df["symbol"] = symbol
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    else:
        df["volume"] = 0
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]]


def update_yfinance_series(
    symbol: str, csv_path: str, start_default: str = "2000-01-01"
) -> None:
    today = dt.date.today()
    start_date = start_default

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if not existing.empty:
            last_date = pd.to_datetime(existing["date"]).max().date()
            start_date = (last_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if pd.to_datetime(start_date).date() > today:
        return

    new_data = yf.Ticker(symbol).history(
        start=start_date,
        end=(today + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
    )
    new_df = _normalize_yf_frame(new_data, symbol)

    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")

    combined["pct_change"] = combined["close"].pct_change(fill_method=None)
    combined.sort_values("date", inplace=True)
    combined.to_csv(csv_path, index=False)


def update_bond_zh_us_rates(csv_path: str, start_default: str = "2015-01-01") -> None:
    today = dt.date.today()
    start_date = start_default

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if not existing.empty:
            last_date = pd.to_datetime(existing["date"]).max().date()
            start_date = (last_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if pd.to_datetime(start_date).date() > today:
        return

    new_data = ak.bond_zh_us_rate(start_date=start_date)
    if new_data.empty:
        return

    new_data["date"] = pd.to_datetime(new_data["日期"]).dt.date
    new_data.drop(["日期"], axis=1, inplace=True)
    new_data["date"] = new_data["date"].astype(str)

    if existing.empty:
        combined = new_data
    else:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")

    base_cols = [
        c
        for c in combined.columns
        if c != "date" and not c.endswith("_change") and not c.endswith("_pct_change")
    ]
    combined = combined[["date"] + base_cols]
    combined.sort_values("date", inplace=True)

    for col in base_cols:
        combined[f"{col}_change"] = combined[col].diff()
        combined[f"{col}_pct_change"] = combined[col].pct_change(fill_method=None)

    combined.to_csv(csv_path, index=False)


def generate_chatbot_txt(csv_paths: dict, output_txt: str) -> None:
    """Generate a single structured txt file with last 30 days from all datasets for chatbot."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"MARKET DATA SUMMARY - {dt.date.today()}")
    lines.append("=" * 80)
    lines.append("")

    for dataset_name, csv_path in csv_paths.items():
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Get last 30 days of data (or all if less than 30 days)
        df["date"] = pd.to_datetime(df["date"])
        last_30_days = df[
            df["date"] >= (df["date"].max() - pd.Timedelta(days=30))
        ].sort_values("date")

        lines.append(f"\n{'=' * 80}")
        lines.append(f"DATASET: {dataset_name}")
        lines.append(f"{'=' * 80}")
        lines.append(
            f"PERIOD: {last_30_days['date'].min().date()} to {last_30_days['date'].max().date()}"
        )
        lines.append(f"TOTAL RECORDS: {len(last_30_days)}")
        lines.append("")

        for idx, row in last_30_days.iterrows():
            lines.append(f"DATE: {row['date'].date()}")
            for col in last_30_days.columns:
                if col != "date":
                    value = row[col]
                    if pd.isna(value):
                        lines.append(f"  {col.upper()}: N/A")
                    elif isinstance(value, float):
                        lines.append(f"  {col.upper()}: {value:.6g}")
                    else:
                        lines.append(f"  {col.upper()}: {value}")
            lines.append("")

    with open(output_txt, "w") as f:
        f.write("\n".join(lines))


def generate_today_txt(csv_paths: dict, output_txt: str) -> None:
    """Generate a structured txt file with today's or latest available data from all datasets."""
    today = dt.date.today()
    lines = []
    lines.append("=" * 80)
    lines.append(f"MARKET DATA - TODAY ({today})")
    lines.append("=" * 80)
    lines.append("")

    for dataset_name, csv_path in csv_paths.items():
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Get today's data, or latest available if today's data doesn't exist
        df["date"] = pd.to_datetime(df["date"])
        today_data = df[df["date"].dt.date == today]

        if today_data.empty:
            # Fall back to latest available data
            latest_date = df["date"].max()
            today_data = df[df["date"] == latest_date]

        if today_data.empty:
            continue

        lines.append(f"\n{'=' * 80}")
        lines.append(f"DATASET: {dataset_name}")
        lines.append(f"{'=' * 80}")
        lines.append("")

        for idx, row in today_data.iterrows():
            lines.append(f"DATE: {row['date'].date()}")
            for col in today_data.columns:
                if col != "date":
                    value = row[col]
                    if pd.isna(value):
                        lines.append(f"  {col.upper()}: N/A")
                    elif isinstance(value, float):
                        lines.append(f"  {col.upper()}: {value:.6g}")
                    else:
                        lines.append(f"  {col.upper()}: {value}")
            lines.append("")

    with open(output_txt, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dir(DATA_DIR)

    for name, symbol in YF_SERIES.items():
        csv_path = os.path.join(DATA_DIR, f"{name}.csv")
        update_yfinance_series(symbol, csv_path)

    bond_csv = os.path.join(DATA_DIR, "bond_zh_us_rate.csv")
    update_bond_zh_us_rates(bond_csv)

    # Generate single consolidated chatbot txt file
    csv_paths = {}
    for name in YF_SERIES.keys():
        csv_paths[name] = os.path.join(DATA_DIR, f"{name}.csv")
    csv_paths["bond_zh_us_rate"] = bond_csv

    output_txt = os.path.join(DATA_DIR, "market_data_chatbot.txt")
    generate_chatbot_txt(csv_paths, output_txt)

    # Generate today's data txt file
    today_txt = os.path.join(DATA_DIR, "market_data_today.txt")
    generate_today_txt(csv_paths, today_txt)


if __name__ == "__main__":
    main()
