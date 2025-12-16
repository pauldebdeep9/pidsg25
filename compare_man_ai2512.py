#!/usr/bin/env python3
"""
Plot prices, order quantities, procurement costs, and storage metrics from pidsg26Evaluation12.xlsx (sheet: Eval12Clean).

X-axis: Date
Lines:  Hedged price (solid) and Unhedged price (dotted)
Bars:   Manual Hedged, Manual Unhedged, AI Hedged, AI Unhedged

Run with the Sai2501 environment, e.g.:
  /Users/debdeeppaul/opt/anaconda3/envs/Sai2501/bin/python compare_man_ai2512.py
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCEL_PATH = Path("pidsg26Evaluation12.xlsx")
SHEET_NAME = "Eval12Clean"
MAIN12_SOURCE = Path("pidsg25-07.xlsx")  # align prices with main12.py

# Columns to plot as bars: (column name in sheet, display label)
BAR_SERIES: Sequence[tuple[str, str]] = (
    ("Order-Manual-Hedged", "Manual Hedged"),
    ("Order-Manual-Unhedged", "Manual Unhedged"),
    ("Order-AI-3-Hed", "AI Hedged"),
    ("Order-AI-3-Unhed", "AI Unhedged"),
)

AI_COLOR = "tab:green"
MANUAL_COLOR = "tab:purple"
LINE_AI_COLOR = "tab:blue"
LINE_MANUAL_COLOR = "tab:orange"


def _first_numeric_col(df: pd.DataFrame) -> pd.Series:
    """Return the first numeric column as a Series (raises if none)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for price data.")
    return df[num_cols[0]].copy()


def load_main12_prices(path: Path = MAIN12_SOURCE) -> tuple[pd.Series, pd.Series]:
    """
    Load price series used by main12.py:
    - Unhedged: first numeric column from p1normal sheet
    - Hedged:   first numeric column from p2normal sheet (flat replicated in main12)
    """
    xls = pd.ExcelFile(path)
    p1 = pd.read_excel(xls, sheet_name="p1normal", index_col=0)
    p2 = pd.read_excel(xls, sheet_name="p2normal", index_col=0)

    unhedged = _first_numeric_col(p1)
    hedged = _first_numeric_col(p2)
    unhedged.index = pd.to_datetime(unhedged.index)
    hedged.index = pd.to_datetime(hedged.index)
    return unhedged, hedged


def load_data(path: Path = EXCEL_PATH, sheet: str = SHEET_NAME, use_main12_prices: bool = False) -> pd.DataFrame:
    """
    Load and lightly clean the sheet; compute procurement costs and cumulative sums.

    By default uses the hedged/unhedged prices already in the evaluation sheet so
    procurement costs match the spreadsheet formulas. Set `use_main12_prices=True`
    to override with the price series from main12.py.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure numeric types for the relevant columns
    storage_cols = ["Manual-storage-cost", "AI-storage-cost"]
    numeric_cols = ["Hedged price", "Unhedged price"] + [c for c, _ in BAR_SERIES] + storage_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise KeyError(f"Expected column '{col}' not found in sheet '{sheet}'.")

    df = df.set_index("Date")

    if use_main12_prices:
        # Optionally align prices with the series used by main12.py
        unhedged_series, hedged_series = load_main12_prices()
        df["Unhedged price"] = unhedged_series.reindex(df.index)
        df["Hedged price"] = hedged_series.reindex(df.index)

    # Trim to periods where manual orders are available (stop at last non-NaN manual entry)
    manual_mask = df["Order-Manual-Hedged"].notna() | df["Order-Manual-Unhedged"].notna()
    if manual_mask.any():
        last_idx = manual_mask[manual_mask].index[-1]
        df = df.loc[:last_idx].copy()

    # For cost math, treat missing orders and storage costs as zero beyond the manual horizon
    order_cols = [c for c, _ in BAR_SERIES]
    df[order_cols] = df[order_cols].fillna(0)
    df[storage_cols] = df[storage_cols].fillna(0)

    # Procurement cost per period
    df["AI procurement cost"] = (
        df["Order-AI-3-Hed"] * df["Hedged price"] + df["Order-AI-3-Unhed"] * df["Unhedged price"]
    )
    df["Manual procurement cost"] = (
        df["Order-Manual-Hedged"] * df["Hedged price"] + df["Order-Manual-Unhedged"] * df["Unhedged price"]
    )

    # Cumulative costs
    df["AI cumulative procurement cost"] = df["AI procurement cost"].cumsum()
    df["Manual cumulative procurement cost"] = df["Manual procurement cost"].cumsum()
    df["AI cumulative storage cost"] = df["AI-storage-cost"].cumsum()
    df["Manual cumulative storage cost"] = df["Manual-storage-cost"].cumsum()

    # Restore Date as a column for downstream plotting
    df = df.reset_index().rename(columns={"index": "Date"})

    return df


def plot_prices_and_orders(df: pd.DataFrame) -> None:
    """Plot line prices and bar orders on dual axes."""
    dates = df["Date"]
    x = np.arange(len(dates))
    width = 0.18
    bar_colors = [MANUAL_COLOR, MANUAL_COLOR, AI_COLOR, AI_COLOR]
    hatches = ["", "//", "", "//"]

    fig, ax_orders = plt.subplots(figsize=(12, 6))

    # Bars for orders
    offsets = [-1.5, -0.5, 0.5, 1.5]
    handles = []
    for (col, label), offset, color, hatch in zip(BAR_SERIES, offsets, bar_colors, hatches):
        bar = ax_orders.bar(
            x + offset * width,
            df[col],
            width=width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
        )
        handles.append(bar)

    ax_orders.set_ylabel("Order quantity")

    # Secondary axis for prices
    ax_price = ax_orders.twinx()
    h_price, = ax_price.plot(
        x,
        df["Hedged price"],
        color="tab:blue",
        lw=2,
        ls="--",
        marker="o",
        markersize=5,
        label="Hedged price",
    )
    # Dotted line with diamond markers so points are visible
    u_price, = ax_price.plot(
        x,
        df["Unhedged price"],
        color="tab:orange",
        lw=2,
        ls="--",
        marker="D",
        markersize=6,
        label="Unhedged price",
    )
    ax_price.set_ylabel("Price")

    # X-axis formatting
    ax_orders.set_xticks(x)
    ax_orders.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")

    # Legend combining both axes
    price_handles = [h_price, u_price]
    ax_orders.legend(handles + price_handles, [h.get_label() for h in handles + price_handles], loc="upper left")

    ax_orders.set_title("Prices vs Orders")
    fig.tight_layout()
    plt.show()


def plot_cumulative_procurement_costs(df: pd.DataFrame) -> None:
    """Plot cumulative procurement costs for AI and Manual."""
    dates = df["Date"]
    x = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ai_line, = ax.plot(
        x,
        df["AI cumulative procurement cost"],
        lw=2,
        marker="o",
        label="AI cumulative cost",
        color=LINE_AI_COLOR,
    )
    man_line, = ax.plot(
        x,
        df["Manual cumulative procurement cost"],
        lw=2,
        marker="s",
        label="Manual cumulative cost",
        color=LINE_MANUAL_COLOR,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("Cumulative procurement cost")
    ax.set_title("Cumulative Procurement Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_storage(df: pd.DataFrame) -> None:
    """Plot cumulative storage cost time series derived from the sheet columns."""
    dates = df["Date"]
    x = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    man_line, = ax.plot(
        x,
        df["Manual cumulative storage cost"],
        lw=2,
        marker="o",
        label="Manual cumulative storage",
        color=LINE_MANUAL_COLOR,
    )
    ai_line, = ax.plot(
        x,
        df["AI cumulative storage cost"],
        lw=2,
        marker="s",
        label="AI cumulative storage",
        color=LINE_AI_COLOR,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("Cumulative storage cost")
    ax.set_title("Cumulative Storage Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def main() -> None:
    df = load_data()
    plot_prices_and_orders(df)
    plot_cumulative_procurement_costs(df)
    plot_storage(df)


if __name__ == "__main__":
    main()
