#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
difference_eda.py

Extended EDA for plan deltas (Sep vs Aug) for BOTH Consumption and Purchase.

Enhancements in this version:
- More robust CSV loading:
  * Prints columns from both snapshots so we can see actual headers.
  * Tries to auto-detect Month / Consumption / Purchase columns via heuristics.
- Generates:
  1. Cumulative deviation & 3-month rolling %Δ
  2. Distribution / bias histogram
  3. Waterfall contribution plot
  4. %Δ heatmap
- Saves:
    plan_diff_Consumption_dashboard.png
    plan_diff_Purchase_dashboard.png
- Prints headline stats in console.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
from typing import Tuple


# ---------------------------------------------------------------------
# CONFIG: snapshot files
# Update these filenames if yours are different
# ---------------------------------------------------------------------
OLDER_FILE = "7.0 FY2025 Material Control.csv"  # "as of Aug"
NEWER_FILE = "8.0 FY2025 Material Control.csv"  # "as of Sep"

OLDER_LABEL = "Sep"  # Label for older snapshot (e.g., "Aug", "Sep", "Oct")
NEWER_LABEL = "Nov"  # Label for newer snapshot (e.g., "Sep", "Oct", "Nov")

OUT_PREFIX = "plan_diff"  # prefix for output figs


# ---------------------------------------------------------------------
# Parsing functions (adapted from working difference_plot.py)
# ---------------------------------------------------------------------

def _coerce_number(x):
    """Convert to float, handling comma separators and NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def load_multiheader_csv(path):
    """Load CSV with multi-level headers like in the working difference_plot.py."""
    df = pd.read_csv(path, header=None)
    months = df.iloc[6].copy().ffill()  # Row 6 has month headers
    attrs  = df.iloc[7].copy()         # Row 7 has attributes (Pur, Consumption, etc.)
    subs   = df.iloc[8].copy()         # Row 8 has status indicators (L/F, T/F, etc.)

    # forward-fill attribute within month blocks
    attrs_ffill = attrs.copy()
    last_attr = None
    for j in range(len(attrs_ffill)):
        if pd.notna(attrs_ffill.iloc[j]):
            last_attr = attrs_ffill.iloc[j]
        elif pd.notna(months.iloc[j]) and last_attr is not None:
            attrs_ffill.iloc[j] = last_attr

    def status_from_sub(x):
        x = str(x).strip() if pd.notna(x) else ""
        return "planned" if x in {"L/F", "T/F"} else "actual"

    status = subs.apply(status_from_sub)

    header = pd.DataFrame({
        "month": months,
        "attribute": attrs_ffill,
        "sub": subs,
        "status": status
    })
    return df, header

def extract_material_row(df, material_name):
    """Find the row for a specific material."""
    # Try column 1 first (for 7.0 format)
    mask = df.iloc[:, 1].astype(str).str.strip().str.lower() == material_name.strip().lower()
    if mask.any():
        idx = mask.idxmax()
        return df.iloc[idx]
    
    # Fall back to column 2 (for 8.0 format)
    mask = df.iloc[:, 2].astype(str).str.strip().str.lower() == material_name.strip().lower()
    if mask.any():
        idx = mask.idxmax()
        return df.iloc[idx]
    
    available_materials = df.iloc[8:, 1].dropna().astype(str).str.strip().tolist()
    raise ValueError(f"Material '{material_name}' not found in column 1 or column 2. Available: {available_materials[:10]}")

def build_timeseries_for_material(csv_path, material, months_keep=None):
    """Build time series for a material from CSV."""
    if months_keep is None:
        months_keep = ("Apr-25","May-25","Jun-25","Jul-25","Aug-25","Sep-25",
                      "Oct-25","Nov-25","Dec-25","Jan-26","Feb-26","Mar-26")
    
    df, header = load_multiheader_csv(csv_path)
    row = extract_material_row(df, material)

    tidy = header.copy()
    tidy["value"] = row.values
    tidy = tidy[tidy["month"].isin(months_keep)].copy()

    def norm_attr(a):
        a = str(a).strip().lower()
        if a in {"pur","purchase","purch"}: return "purchase"
        if a in {"consumption","cons"}:     return "consumption"
        if a in {"c/b","cb","closing balance","closing_bal","c_b"}: return "cb"
        if a in {"inv days","invdays","inventory days","inventory_days"}: return "inv_days"
        return a

    tidy["attribute"] = tidy["attribute"].apply(norm_attr)
    tidy["value"] = tidy["value"].apply(_coerce_number)

    wanted = []
    for attr in ["purchase","consumption","cb"]:
        for stat in ["planned","actual"]:
            wanted.append((attr, stat))
    wanted.append(("inv_days","actual"))

    records = []
    for m in months_keep:
        rec = {"month": m}
        for attr, stat in wanted:
            val = tidy.loc[
                (tidy["month"] == m) &
                (tidy["attribute"] == attr) &
                (tidy["status"] == stat),
                "value"
            ]
            value = float(val.iloc[0]) if len(val) else np.nan
            
            # Convert consumption to absolute values (they are negative in the data)
            if attr == "consumption" and not np.isnan(value):
                value = abs(value)
            
            rec[f"{attr}_{stat}"] = value
        records.append(rec)

    out = pd.DataFrame.from_records(records).set_index("month")
    return out

# ---------------------------------------------------------------------
# DEPRECATED functions - replaced with working logic above
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
MANUAL_COLUMN_MAP = {
    "month": None,         # e.g. "Month"
    "consumption": None,   # e.g. "Consumption (Planned)"
    "purchase": None,      # e.g. "Purchase (Planned)"
}


# ---------------------------------------------------------------------
# Merge old/new snapshot for a given metric (Consumption or Purchase)
# -> returns DataFrame with Month, Plan_{OLDER_LABEL}, Plan_{NEWER_LABEL}
# ---------------------------------------------------------------------
def build_metric_df(df_old, df_new, month_col, value_col,
                    label_old=None, label_new=None):
    if label_old is None:
        label_old = f"Plan_{OLDER_LABEL}"
    if label_new is None:
        label_new = f"Plan_{NEWER_LABEL}"
    a = df_old[[month_col, value_col]].copy()
    b = df_new[[month_col, value_col]].copy()

    a.columns = ["Month", label_old]
    b.columns = ["Month", label_new]

    merged = pd.merge(a, b, on="Month", how="inner")

    return merged


# ---------------------------------------------------------------------
# Preprocess for downstream analysis
# Adds:
#   Month_dt, Month_str
#   {OLDER_LABEL}, {NEWER_LABEL}
#   AbsDiff, PctDiff
#   CumAbsDiff, Roll3mPctDiff
# ---------------------------------------------------------------------
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # parse Month like "Apr-25" => datetime first of month
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")

    # If parsing fails because format not %b-%y, try a fallback:
    if df["Month_dt"].isna().all():
        # Attempt YYYY-MM or similar generic parse
        df["Month_dt"] = pd.to_datetime(df["Month"], errors="coerce")

    df = df.sort_values("Month_dt").reset_index(drop=True)
    df["Month_str"] = df["Month"]

    df[OLDER_LABEL] = pd.to_numeric(df[f"Plan_{OLDER_LABEL}"], errors="coerce")
    df[NEWER_LABEL] = pd.to_numeric(df[f"Plan_{NEWER_LABEL}"], errors="coerce")

    df["AbsDiff"] = df[NEWER_LABEL] - df[OLDER_LABEL]

    denom = df[OLDER_LABEL].abs().replace(0, np.nan)
    df["PctDiff"] = (df[NEWER_LABEL] - df[OLDER_LABEL]) / denom * 100.0
    df["PctDiff"] = df["PctDiff"].fillna(0.0)

    df["CumAbsDiff"] = df["AbsDiff"].cumsum()
    df["Roll3mPctDiff"] = df["PctDiff"].rolling(window=3, min_periods=1).mean()

    return df


# ---------------------------------------------------------------------
# Plot dashboard for one metric (Consumption or Purchase)
# 2x2 subplots:
#   (0,0): cumulative deviation + rolling 3mo avg %Δ
#   (0,1): histogram + bias box
#   (1,0): waterfall contribution
#   (1,1): %Δ heatmap
# ---------------------------------------------------------------------
def plot_dashboard(df: pd.DataFrame, label: str, out_prefix: str):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{label} Plan Change: {NEWER_LABEL} vs {OLDER_LABEL}", fontsize=14, fontweight="bold")

    x = np.arange(len(df))
    month_labels = df["Month_str"].tolist()

    # --- (0,0) cumulative deviation + rolling %Δ
    ax_main = axs[0, 0]
    ax_right = ax_main.twinx()

    ax_main.plot(
        x,
        df["CumAbsDiff"],
        marker="o",
        linewidth=2,
        label=f"Cumulative ({NEWER_LABEL} - {OLDER_LABEL})",
    )

    ax_right.plot(
        x,
        df["Roll3mPctDiff"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="3-mo avg %Δ",
    )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(month_labels, rotation=45, ha="right")

    ax_main.set_ylabel("Cumulative deviation (units)")
    ax_right.set_ylabel("Rolling %Δ vs Sep (%)")
    ax_main.set_title("Cumulative Deviation & Smoothed % Change")
    ax_main.grid(alpha=0.3)

    lines_main, labels_main = ax_main.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_main.legend(lines_main + lines_right, labels_main + labels_right, loc="best")

    # --- (0,1) histogram + bias summary
    ax_hist = axs[0, 1]
    diffs = df["AbsDiff"].values

    ax_hist.hist(diffs, bins=8, edgecolor="black", alpha=0.7)
    ax_hist.set_title(f"Distribution of Monthly Change ({NEWER_LABEL} - {OLDER_LABEL})")
    ax_hist.set_xlabel("Monthly change (units)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(alpha=0.3)

    num_up = np.sum(diffs > 0)
    num_down = np.sum(diffs < 0)
    mean_abs = np.mean(np.abs(diffs))
    total_shift = np.sum(diffs)

    summary_txt = (
        f"↑ months : {num_up}\n"
        f"↓ months : {num_down}\n"
        f"Mean |Δ| : {mean_abs:.1f}\n"
        f"Total Δ : {total_shift:.1f}"
    )

    ax_hist.text(
        0.95,
        0.95,
        summary_txt,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        transform=ax_hist.transAxes,
    )

    # --- (1,0) waterfall of AbsDiff
    ax_wf = axs[1, 0]
    running_total = 0.0
    cum_points = [0.0]
    for d in diffs:
        running_total += d
        cum_points.append(running_total)

    for i, d in enumerate(diffs):
        y0 = cum_points[i]
        color = "tab:green" if d >= 0 else "tab:red"

        ax_wf.bar(
            i,
            height=d,
            bottom=y0,
            color=color,
            edgecolor="black",
            alpha=0.7,
        )

        ax_wf.text(
            i,
            y0 + d / 2.0,
            f"{d:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(d) > 5 else "black",
        )

    ax_wf.plot(
        np.arange(len(cum_points)),
        cum_points,
        marker="o",
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    ax_wf.set_xticks(x)
    ax_wf.set_xticklabels(month_labels, rotation=45, ha="right")
    ax_wf.set_ylabel("Cumulative deviation (units)")
    ax_wf.set_title("Waterfall: Month-by-Month Contribution to Total Shift")
    ax_wf.grid(alpha=0.3)

    # --- (1,1) heatmap of %Δ
    ax_hm = axs[1, 1]

    pct_vals = df["PctDiff"].values
    pct_matrix = np.expand_dims(pct_vals, axis=0)

    max_abs = max(1e-6, np.max(np.abs(pct_vals)))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    hm = ax_hm.imshow(
        pct_matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
    )

    for j, val in enumerate(pct_vals):
        ax_hm.text(
            j,
            0,
            f"{val:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax_hm.set_yticks([0])
    ax_hm.set_yticklabels(["%Δ Nov vs Sep"])
    ax_hm.set_xticks(x)
    ax_hm.set_xticklabels(month_labels, rotation=45, ha="right")
    ax_hm.set_title(f"Heatmap of % Change vs {OLDER_LABEL}")
    ax_hm.set_xlabel("Month")

    cbar = plt.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("% change")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot instead of saving
    plt.show()
    print(f"[INFO] Displayed {label} dashboard")


# ---------------------------------------------------------------------
# headline print
# ---------------------------------------------------------------------
def headline(df, name):
    total_abs_dev = df["AbsDiff"].sum()
    max_up_idx = df["AbsDiff"].idxmax()
    max_dn_idx = df["AbsDiff"].idxmin()

    print(f"\n=== {name} summary ===")
    print(f"Total Δ (Nov-Sep): {total_abs_dev:.1f} units")

    print(
        "Max upward revision: "
        f"{df.loc[max_up_idx, 'Month_str']} "
        f"({df.loc[max_up_idx, 'AbsDiff']:.1f} units, "
        f"{df.loc[max_up_idx, 'PctDiff']:.1f}%)"
    )

    print(
        "Max downward revision: "
        f"{df.loc[max_dn_idx, 'Month_str']} "
        f"({df.loc[max_dn_idx, 'AbsDiff']:.1f} units, "
        f"{df.loc[max_dn_idx, 'PctDiff']:.1f}%)"
    )

    print(
        "End-of-horizon cumulative deviation: "
        f"{df['CumAbsDiff'].iloc[-1]:.1f} units"
    )


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    """
    Updated main function using the working parsing logic from difference_plot.py
    """
    # 1. Check files exist
    if not os.path.exists(OLDER_FILE):
        print(f"[FATAL] Cannot find {OLDER_FILE}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(NEWER_FILE):
        print(f"[FATAL] Cannot find {NEWER_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading older snapshot: {OLDER_FILE}")
    print(f"Loading newer snapshot: {NEWER_FILE}")

    # List all materials available in the older file for analysis
    try:
        df_temp, _ = load_multiheader_csv(OLDER_FILE)
        available_materials = df_temp.iloc[8:, 1].dropna().astype(str).str.strip().tolist()
        available_materials = [m for m in available_materials if m not in ['', 'Total (Without GIT)', 'GIT', 'Total (With GIT)']]
        print(f"\nAvailable materials: {available_materials}")
        
        # Use the first valid material or default to "Silver Paste"
        target_material = "Silver Paste"
        if target_material not in available_materials and available_materials:
            target_material = available_materials[0]
            print(f"Using '{target_material}' as target material")
        
    except Exception as e:
        print(f"Error reading materials: {e}")
        target_material = "Silver Paste"  # fallback

    # 2. Build time series for both snapshots
    months_keep = ("Apr-25","May-25","Jun-25","Jul-25","Aug-25","Sep-25",
                   "Oct-25","Nov-25","Dec-25","Jan-26","Feb-26","Mar-26")
    
    try:
        df_old = build_timeseries_for_material(OLDER_FILE, target_material, months_keep)
        df_new = build_timeseries_for_material(NEWER_FILE, target_material, months_keep)
        
        print(f"\nOlder data for {target_material}:")
        print(df_old.head())
        print(f"\nNewer data for {target_material}:")
        print(df_new.head())

        # 3. Build comparison dataframes for planned consumption and purchase
        cons_df = pd.DataFrame({
            'Month': df_old.index,
            f'Plan_{OLDER_LABEL}': df_old['consumption_planned'],
            f'Plan_{NEWER_LABEL}': df_new['consumption_planned']
        }).dropna()
        
        purch_df = pd.DataFrame({
            'Month': df_old.index,
            f'Plan_{OLDER_LABEL}': df_old['purchase_planned'], 
            f'Plan_{NEWER_LABEL}': df_new['purchase_planned']
        }).dropna()

        # Check if we have data to analyze
        if len(cons_df) == 0:
            print(f"\nWarning: No consumption_planned data available in both files for comparison.")
            print("Skipping consumption analysis.")
            cons_df = None
        
        if len(purch_df) == 0:
            print(f"\nWarning: No purchase_planned data available in both files for comparison.")
            print("Skipping purchase analysis.")
            purch_df = None

        if cons_df is None and purch_df is None:
            print("\nError: No comparable planned data found in either consumption or purchase.")
            print("Please check that both CSV files have planned data (L/F or T/F indicators).")
            sys.exit(1)

        # 4. Add difference calculations
        if cons_df is not None:
            cons_df = preprocess(cons_df)
        if purch_df is not None:
            purch_df = preprocess(purch_df)

        # 5. Generate dashboards
        if cons_df is not None:
            plot_dashboard(cons_df, label="Consumption", out_prefix=OUT_PREFIX)
        if purch_df is not None:
            plot_dashboard(purch_df, label="Purchase", out_prefix=OUT_PREFIX)

        # 6. Print headline stats
        if cons_df is not None:
            headline(cons_df, "Consumption")
        if purch_df is not None:
            headline(purch_df, "Purchase")

        print(f"\nAnalysis complete! Dashboards saved with prefix '{OUT_PREFIX}'")

    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please check that the CSV files have the expected structure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
