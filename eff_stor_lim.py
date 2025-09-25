#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly plots from Date:
i)  Order-AI-Unhedged — bars (With/Without) + Unhedged price (monthly avg) on 2nd axis
ii) AI-storage — bars (With/Without) + Manual-storage (3 bars)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

WITH_F = Path("WithCapacityConst.xlsx")
WITHOUT_F = Path("WithoutCapacityConst.xlsx")
SHEET = "05"  # change if needed

# --------------- helpers ---------------
def pick_col(df: pd.DataFrame, target: str):
    tl = target.lower()
    for c in df.columns:
        if str(c).lower() == tl: return c
    for c in df.columns:
        if tl in str(c).lower(): return c
    return None

def need(df: pd.DataFrame, target: str) -> str:
    col = pick_col(df, target)
    if col is None:
        raise KeyError(f"Column like '{target}' not found. First columns: {list(df.columns)[:20]} ...")
    return col

def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c) for c in df.columns]
    dcol = need(df, "Date")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df, dcol

def to_numeric(df: pd.DataFrame, exclude=("Date",)) -> pd.DataFrame:
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def monthify(df: pd.DataFrame, date_col: str) -> pd.Series:
    # Normalize to first day of month for clean x-axis
    return df[date_col].dt.to_period("M").dt.to_timestamp()

# --------------- load ---------------
df_w,  date_w  = read_sheet(WITH_F, SHEET)
df_wo, date_wo = read_sheet(WITHOUT_F, SHEET)

col_order_w  = need(df_w,  "Order-AI-Unhedged")
col_order_wo = need(df_wo, "Order-AI-Unhedged")
col_ai_w     = need(df_w,  "AI-storage")
col_ai_wo    = need(df_wo, "AI-storage")
col_man_w    = pick_col(df_w, "Manual-storage") or need(df_wo, "Manual-storage")
col_price    = pick_col(df_w, "Unhedged price") or pick_col(df_wo, "Unhedged price")

# Build slim frames
left = df_w[[date_w, col_order_w, col_ai_w, col_man_w]].rename(
    columns={date_w: "Date", col_order_w: "Order_With", col_ai_w: "AI_With", col_man_w: "Manual"})
right = df_wo[[date_wo, col_order_wo, col_ai_wo]].rename(
    columns={date_wo: "Date", col_order_wo: "Order_Without", col_ai_wo: "AI_Without"})

if col_price:
    if col_price in df_w.columns:
        price_df = df_w[[date_w, col_price]].rename(columns={date_w: "Date", col_price: "Price"})
    else:
        price_df = df_wo[[date_wo, col_price]].rename(columns={date_wo: "Date", col_price: "Price"})
else:
    price_df = None

# Align by Date (inner join to keep common dates)
df = pd.merge(left, right, on="Date", how="inner")
if price_df is not None:
    df = pd.merge(df, price_df, on="Date", how="left")

# Numeric + month key
df = to_numeric(df, exclude=("Date",))
df["Month"] = monthify(df, "Date")

# Monthly aggregation: sum for quantities, mean for price
agg_dict = {
    "Order_With": "sum",
    "Order_Without": "sum",
    "AI_With": "sum",
    "AI_Without": "sum",
    "Manual": "sum",
}
if "Price" in df.columns:
    agg_dict["Price"] = "mean"

m = df.groupby("Month", as_index=False).agg(agg_dict).sort_values("Month")


# --------------- Plot i) Orders + Price (monthly, hatched bars) ---------------
fig, ax1 = plt.subplots(figsize=(11.5, 5))
x = np.arange(len(m))
bar_w = 0.45

# With hatched fill
ax1.bar(x - bar_w/2, m["Order_With"], 
        width=bar_w, label="With: Order-AI-Unhedged",
        color="tab:blue", hatch="//", edgecolor="black")

ax1.bar(x + bar_w/2, m["Order_Without"], 
        width=bar_w, label="Without: Order-AI-Unhedged",
        color="tab:orange", hatch="\\\\", edgecolor="black")

ax1.set_ylabel("Order qty")

# Monthly date axis
ax1.set_xticks(x)
ax1.set_xticklabels(m["Month"].dt.strftime("%Y-%m"), rotation=45, ha="right")
ax1.set_xlabel("Month")

# Price on secondary axis
if "Price" in m.columns:
    ax2 = ax1.twinx()
    ax2.plot(x, m["Price"], linestyle="--", linewidth=1.8, color="black", label="Unhedged price")
    ax2.set_ylabel("Price")
    # combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
else:
    ax1.legend(loc="best")

plt.title(f"Order-AI-Unhedged — With vs Without (Monthly, Sheet {SHEET})")
plt.tight_layout()
plt.show()


# --------------- Plot ii) Storages (3 bars, monthly, hatched + storage limit=12000) ---------------
fig, ax = plt.subplots(figsize=(11.5, 5))
x = np.arange(len(m))
bar_w = 0.28

ax.bar(x - bar_w, m["AI_With"],    width=bar_w, label="AI-storage (With)",
       color="tab:blue", hatch="//", edgecolor="black")
ax.bar(x,         m["AI_Without"], width=bar_w, label="AI-storage (Without)",
       color="tab:orange", hatch="\\", edgecolor="black")
ax.bar(x + bar_w, m["Manual"],     width=bar_w, label="Manual-storage",
       color="tab:green", hatch="xx", edgecolor="black")

# Fixed storage limit line at 12000
ax.axhline(12000, color="red", linestyle=":", linewidth=2, label="Storage limit")

ax.set_ylabel("Storage level")
ax.set_xticks(x)
ax.set_xticklabels(m["Month"].dt.strftime("%Y-%m"), rotation=45, ha="right")
ax.set_xlabel("Month")
ax.legend(loc="best")
plt.title(f"Storage — AI(With/Without) + Manual (Monthly, Sheet {SHEET})")
plt.tight_layout()
plt.show()


