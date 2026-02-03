from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


def _filter_by_index_range(price_df, items, key="index", low_i=None, high_i=None):
    """Helper function to filter objects by index range"""
    if low_i is None or high_i is None:
        return items
    out = []
    for it in items:
        idx_val = it.get(key) if isinstance(it, dict) else None
        if idx_val is not None and low_i <= idx_val <= high_i:
            out.append(it)
    return out


def plot_chan_analysis(
    chan,
    fractals,
    strokes,
    segments,
    zhongshu,
    points_df,
    boduan_results=None,
    last_n=None,
    start_date=None,
    end_date=None,
    text_offset_ratio=0.01,
    stack_step_ratio=0.005,
    figsize=None,
    title=None,
):
    """
    Plot Chan Analysis with price, fractals, strokes, segments, zhongshu and buy/sell points.
    Optionally includes boduan technical indicators in additional panels.

    Parameters:
    -----------
    chan : ChanAnalysis object
        The initialized Chan analysis object
    fractals : list
        List of fractal dictionaries
    strokes : list
        List of stroke dictionaries
    segments : list
        List of segment dictionaries
    zhongshu : list
        List of zhongshu dictionaries
    points_df : pandas.DataFrame
        DataFrame containing buy/sell points
    boduan_results : dict, optional
        Dictionary containing boduan technical indicators. If provided, creates multi-panel plot.
    last_n : int, optional
        Number of recent bars to plot
    start_date : str, optional
        Start date for date range filtering (e.g., '2022-01-01')
    end_date : str, optional
        End date for date range filtering (e.g., '2023-12-31')
    text_offset_ratio : float, default 0.01
        Vertical offset for text labels as ratio of price range
    stack_step_ratio : float, default 0.005
        Vertical step between stacked labels as ratio of price range
    figsize : tuple, optional
        Figure size. Defaults to (14, 6) for basic plot or (16, 12) for boduan plot
    title : str, optional
        Custom title for the plot

    Returns:
    --------
    fig, ax(es) : matplotlib figure and axis objects
        Returns (fig, ax) for basic plot or (fig, axes) for multi-panel boduan plot
    """

    # Prepare price data
    price_df = chan.comp_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Determine subset
    if last_n is not None:
        subset_df = price_df.tail(last_n)
    elif start_date or end_date:
        sd = pd.to_datetime(start_date) if start_date else price_df["date"].min()
        ed = pd.to_datetime(end_date) if end_date else price_df["date"].max()
        subset_df = price_df[(price_df["date"] >= sd) & (price_df["date"] <= ed)]
    else:
        subset_df = price_df

    low_i = subset_df["index"].min()
    high_i = subset_df["index"].max()

    # Filter data by index range
    fractals_sub = _filter_by_index_range(
        price_df, fractals, low_i=low_i, high_i=high_i
    )
    strokes_sub = _filter_by_index_range(
        price_df, strokes, key="start_index", low_i=low_i, high_i=high_i
    )
    segments_sub = _filter_by_index_range(
        price_df, segments, key="start_index", low_i=low_i, high_i=high_i
    )
    zhongshu_sub = _filter_by_index_range(
        price_df, zhongshu, key="start_index", low_i=low_i, high_i=high_i
    )
    points_sub = _filter_by_index_range(
        price_df, points_df.to_dict("records"), key="index", low_i=low_i, high_i=high_i
    )

    # Determine if we need multi-panel layout for boduan indicators
    has_boduan = boduan_results is not None and len(boduan_results) > 0

    # Set default figure size based on plot type
    if figsize is None:
        figsize = (16, 12) if has_boduan else (14, 6)

    # Create figure - single panel or multi-panel based on boduan presence
    if has_boduan:
        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[3, 1, 1, 1])
        ax_price, ax_c1c2, ax_rsi_adx, ax_signals = axes
        ax = ax_price  # Main price chart

        # Prepare boduan indicators for the subset - FIXED INDEXING
        boduan_subset = {}

        # Create mapping from original dataframe indices to boduan array positions
        df_original_indices = chan.comp_df["index"].values
        subset_original_indices = subset_df["index"].values

        for key, values in boduan_results.items():
            if hasattr(values, "__len__") and len(values) > 0:
                # Map subset original indices to boduan array positions
                subset_values = []
                for orig_idx in subset_original_indices:
                    # Find position in original dataframe
                    pos_in_df = np.where(df_original_indices == orig_idx)[0]
                    if len(pos_in_df) > 0 and pos_in_df[0] < len(values):
                        subset_values.append(values[pos_in_df[0]])
                    else:
                        subset_values.append(np.nan)
                boduan_subset[key] = np.array(subset_values)
            else:
                boduan_subset[key] = values
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # === Main Price Chart with Chan Analysis ===
    ax.plot(
        subset_df["date"], subset_df["close"], color="black", linewidth=1, label="Close"
    )

    # Style definitions
    fractals_style = {
        "top": {"color": "#d62728", "marker": "^"},
        "bottom": {"color": "#2ca02c", "marker": "v"},
    }
    stroke_colors = {"up": "#1f77b4", "down": "#ff7f0e"}
    point_styles = {
        "buy": {"color": "#006400", "marker": "o", "edgecolor": "white", "size": 90},
        "sell": {"color": "#8B0000", "marker": "s", "edgecolor": "white", "size": 90},
        "warn": {"color": "#F53404", "marker": "X", "edgecolor": "black", "size": 80},
    }
    segment_alpha = 0.06
    zhongshu_face = "#6a5acd"
    zhongshu_edge = "#483d8b"

    # Plot fractals markers
    for f in fractals_sub:
        date_val = subset_df.loc[subset_df["index"] == f["index"], "date"]
        if date_val.empty:
            continue
        d = date_val.values[0]
        sty = fractals_style[f["type"]]
        ax.scatter(
            d,
            f["price"],
            color=sty["color"],
            marker=sty["marker"],
            s=70,
            zorder=5,
            linewidths=0.5,
            edgecolors="black",
        )

    # Plot strokes as lines
    for s in strokes_sub:
        if not (
            low_i <= s["start_index"] <= high_i or low_i <= s["end_index"] <= high_i
        ):
            continue
        sd_row = subset_df.loc[subset_df["index"] == s["start_index"], "date"]
        ed_row = subset_df.loc[subset_df["index"] == s["end_index"], "date"]
        if sd_row.empty or ed_row.empty:
            continue
        sd = sd_row.values[0]
        ed = ed_row.values[0]
        ax.plot(
            [sd, ed],
            [s["start_price"], s["end_price"]],
            color=stroke_colors[s["direction"]],
            linewidth=1.8,
            alpha=0.9,
        )

    # Plot segments shading
    for seg in segments_sub:
        sd_row = subset_df.loc[subset_df["index"] == seg["start_index"], "date"]
        ed_row = subset_df.loc[subset_df["index"] == seg["end_index"], "date"]
        if sd_row.empty or ed_row.empty:
            continue
        sd = sd_row.values[0]
        ed = ed_row.values[0]
        ax.axvspan(sd, ed, color=stroke_colors[seg["direction"]], alpha=segment_alpha)

    # Plot zhongshu rectangles
    for zs in zhongshu_sub:
        sd_row = subset_df.loc[subset_df["index"] == zs["start_index"], "date"]
        ed_row = subset_df.loc[subset_df["index"] == zs["end_index"], "date"]
        if sd_row.empty or ed_row.empty:
            continue
        sd = sd_row.values[0]
        ed = ed_row.values[0]
        width = ed - sd
        ax.add_patch(
            Rectangle(
                (sd, zs["lower"]),
                width,
                zs["upper"] - zs["lower"],
                facecolor=zhongshu_face,
                alpha=0.18,
                edgecolor=zhongshu_edge,
                linewidth=1,
            )
        )
        ax.text(
            sd,
            zs["upper"],
            f"Z({zs['center']:.2f})",
            color=zhongshu_edge,
            fontsize=8,
            va="bottom",
        )

    # Plot buy/sell points
    points_by_index = defaultdict(list)
    for row in points_sub:
        points_by_index[row["index"]].append(row)

    for idx, pts in points_by_index.items():
        date_val = subset_df.loc[subset_df["index"] == idx, "date"]
        if date_val.empty:
            continue
        d = date_val.values[0]

        # Separate by type: buys vs sells vs warn
        buys = [p for p in pts if p["type"] == "buy"]
        sells = [p for p in pts if p["type"] == "sell"]
        warns = [p for p in pts if p["type"] == "warn"]

        # Sort each list by price to make offsets progressive
        buys.sort(key=lambda x: x["price"], reverse=True)
        sells.sort(key=lambda x: x["price"])
        warns.sort(key=lambda x: x["price"], reverse=True)

        def plot_stack(stack, direction):
            for i, row in enumerate(stack):
                sty = point_styles.get(row["type"], point_styles["warn"])
                ax.scatter(
                    d,
                    row["price"],
                    color=sty["color"],
                    marker=sty["marker"],
                    s=sty["size"],
                    edgecolors=sty["edgecolor"],
                    linewidths=1.0,
                    zorder=6,
                )
                # Compute text offset
                base = row["price"]
                offset = text_offset_ratio + i * stack_step_ratio
                y = base * (1 + offset) if direction == "up" else base * (1 - offset)
                va = "bottom" if direction == "up" else "top"
                ax.text(
                    d,
                    y,
                    row["pattern"],
                    color=sty["color"],
                    fontsize=7,
                    ha="center",
                    va=va,
                )

        plot_stack(buys + warns, "up")  # place warns with buys above
        plot_stack(sells, "down")

    # Build custom legend (only for single-panel plots to avoid crowding)
    if not has_boduan:
        legend_elements = [
            Line2D([0], [0], color="black", lw=1, label="Close"),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Fractal Top",
                markerfacecolor=fractals_style["top"]["color"],
                markeredgecolor="black",
                markersize=9,
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="w",
                label="Fractal Bottom",
                markerfacecolor=fractals_style["bottom"]["color"],
                markeredgecolor="black",
                markersize=9,
            ),
            Line2D([0], [0], color=stroke_colors["up"], lw=2, label="Stroke Up"),
            Line2D([0], [0], color=stroke_colors["down"], lw=2, label="Stroke Down"),
            Patch(
                facecolor=zhongshu_face,
                edgecolor=zhongshu_edge,
                alpha=0.4,
                label="Zhongshu Zone",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Buy Signal",
                markerfacecolor=point_styles["buy"]["color"],
                markeredgecolor="white",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Sell Signal",
                markerfacecolor=point_styles["sell"]["color"],
                markeredgecolor="white",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label="Warn (Divergence)",
                markerfacecolor=point_styles["warn"]["color"],
                markeredgecolor="black",
                markersize=10,
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper left", ncol=2, frameon=True, fontsize=9
        )
    else:
        ax.legend()

    # Set title and formatting for main chart
    if title is None:
        title_suffix = " with Boduan Indicators" if has_boduan else ""
        title = f"Chan Analysis{title_suffix} (Subset {low_i}-{high_i})"
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # === Additional Boduan Indicator Panels ===
    if has_boduan:
        # Panel 2: C1/C2 Williams Indicators
        c1_values = boduan_subset.get("C1", [])
        c2_values = boduan_subset.get("C2", [])

        if len(c1_values) == len(subset_df):
            # Remove NaN values for plotting
            valid_mask = ~(pd.isna(c1_values) | pd.isna(c2_values))
            if valid_mask.any():
                dates_valid = subset_df["date"][valid_mask]
                c1_valid = c1_values[valid_mask]
                c2_valid = c2_values[valid_mask]

                ax_c1c2.plot(
                    dates_valid,
                    c1_valid,
                    label="C1 (Williams)",
                    color="blue",
                    linewidth=1.5,
                )
                ax_c1c2.plot(
                    dates_valid,
                    c2_valid,
                    label="C2 (Smooth)",
                    color="red",
                    linewidth=1.5,
                )
                ax_c1c2.axhline(
                    y=20, color="gray", linestyle="--", alpha=0.7, label="Oversold (20)"
                )
                ax_c1c2.axhline(
                    y=80,
                    color="gray",
                    linestyle="--",
                    alpha=0.7,
                    label="Overbought (80)",
                )

                # Highlight C1/C2 crossovers
                tiao_signals = boduan_subset.get("tiao", [])
                if len(tiao_signals) == len(subset_df):
                    for i, signal in enumerate(tiao_signals):
                        if (
                            signal > 0
                            and i < len(subset_df)
                            and not pd.isna(c1_values[i])
                        ):
                            ax_c1c2.scatter(
                                subset_df.iloc[i]["date"],
                                c1_values[i],
                                color="green",
                                marker="^",
                                s=80,
                                zorder=5,
                            )

        ax_c1c2.set_ylabel("C1/C2 Williams")
        ax_c1c2.legend(fontsize=8)
        ax_c1c2.grid(alpha=0.3)

        # Panel 3: RSI/ADX/DXR Indicators
        ax2 = ax_rsi_adx
        rsi_values = boduan_subset.get("RSI5", [])
        if len(rsi_values) == len(subset_df):
            valid_mask = ~pd.isna(rsi_values)
            if valid_mask.any():
                ax2.plot(
                    subset_df["date"][valid_mask],
                    rsi_values[valid_mask],
                    label="RSI5",
                    color="purple",
                    linewidth=1.5,
                )
                ax2.axhline(y=30, color="red", linestyle="--", alpha=0.7)
                ax2.axhline(y=70, color="red", linestyle="--", alpha=0.7)

        ax3 = ax2.twinx()
        adx_values = boduan_subset.get("ADX", [])
        dxr_values = boduan_subset.get("DXR", [])
        if len(adx_values) == len(subset_df):
            valid_mask = ~pd.isna(adx_values)
            if valid_mask.any():
                ax3.plot(
                    subset_df["date"][valid_mask],
                    adx_values[valid_mask],
                    label="ADX",
                    color="orange",
                    linewidth=1.5,
                )
        if len(dxr_values) == len(subset_df):
            valid_mask = ~pd.isna(dxr_values)
            if valid_mask.any():
                ax3.plot(
                    subset_df["date"][valid_mask],
                    dxr_values[valid_mask],
                    label="DXR",
                    color="brown",
                    linewidth=1.5,
                )

        ax2.set_ylabel("RSI5", color="purple")
        ax3.set_ylabel("ADX/DXR", color="orange")
        ax2.legend(loc="upper left", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)

        # Panel 4: Trading Signals
        ax_sig = ax_signals

        # Create signal bars
        bar_width = 1.0

        ready_signals = boduan_subset.get("ready", [])
        go_signals = boduan_subset.get("go", [])
        qiang_signals = boduan_subset.get("qiang_la_sheng", [])
        kai_signals = boduan_subset.get("kai_shi_la_sheng", [])

        if len(ready_signals) == len(subset_df):
            ready_bars = [1 if x > 0 else 0 for x in ready_signals]
            ax_sig.bar(
                subset_df["date"],
                ready_bars,
                width=bar_width,
                alpha=0.7,
                color="blue",
                label="Ready",
            )

        if len(go_signals) == len(subset_df):
            go_bars = [1.5 if x > 0 else 0 for x in go_signals]
            ax_sig.bar(
                subset_df["date"],
                go_bars,
                width=bar_width,
                alpha=0.7,
                color="green",
                label="Go",
            )

        if len(qiang_signals) == len(subset_df):
            qiang_bars = [2 if x > 0 else 0 for x in qiang_signals]
            ax_sig.bar(
                subset_df["date"],
                qiang_bars,
                width=bar_width,
                alpha=0.7,
                color="red",
                label="Strong Pull",
            )

        if len(kai_signals) == len(subset_df):
            kai_bars = [2.5 if x else 0 for x in kai_signals]
            ax_sig.bar(
                subset_df["date"],
                kai_bars,
                width=bar_width,
                alpha=0.7,
                color="purple",
                label="Start Pull",
            )

        ax_sig.set_ylabel("Signals")
        ax_sig.legend(fontsize=8)
        ax_sig.grid(alpha=0.3)
        ax_sig.set_ylim(0, 3)

    plt.tight_layout()

    # Print summary
    summary_msg = f"Subset fractals: {len(fractals_sub)}, strokes: {len(strokes_sub)}, segments: {len(segments_sub)}, zhongshu: {len(zhongshu_sub)}, points: {len(points_sub)}"
    if has_boduan:
        c1_valid_count = np.sum(~pd.isna(boduan_subset.get("C1", [])))
        summary_msg += f"\nBoduan data mapped: C1 valid values: {c1_valid_count}"
    print(summary_msg)

    # Return appropriate objects based on plot type
    if has_boduan:
        return fig, axes
    else:
        return fig, ax
