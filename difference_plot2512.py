#!/usr/bin/env python3
"""
Self-contained: run directly (no CLI args).
- Reads '4.0 FY2025 Material Control.csv' and '5.0 FY2025 Material Control.csv' from CWD
- Extracts monthly series (Apr-25 .. Mar-26) for a material (default: Silver Paste)
- Plots 4 comparisons (actual/planned, purchase/consumption) sequentially
- Adds % difference overlay: 100 * (Series2 - Series1) / Series1 on a secondary Y axis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# -------------------- USER SETTINGS --------------------
CSV1 = "7.0 FY2025 Material Control.csv"  # v4.0
CSV2 = "8.0 FY2025 Material Control.csv"  # v5.0
MATERIAL = "Silver Paste"
MONTHS = ("Apr-25","May-25","Jun-25","Jul-25","Aug-25","Sep-25",
          "Oct-25","Nov-25","Dec-25","Jan-26","Feb-26","Mar-26")
LABEL1 = "Plan as of Sep-25"
LABEL2 = "Plan as of Nov-25"
# ------------------------------------------------------

# ADDED: minimal exporter — saves the two planned series shown on the current axes
import matplotlib.pyplot as plt
import pandas as pd

# --- REPLACE the old helper with this one ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

def _export_planned_series_from_current_axes(out_csv: str = "planned_series.csv"):
    """
    Scrape the active matplotlib Axes for planned series and write a CSV with 4 columns:
      ['consumption_planned', 'Consumption (Planned)',
       'purchase_planned',    'Purchase (Planned)']
    Index is the x-axis of the plot. Works for Line2D and bar plots.
    """
    ax = plt.gca()

    # Accept either human-readable or snake_case labels
    label_map = {
        "Consumption (Planned)": "consumption_planned",
        "consumption_planned": "consumption_planned",
        "Purchase (Planned)": "purchase_planned",
        "purchase_planned": "purchase_planned",
    }

    # 1) Collect candidate series from lines
    series_dict = {}  # raw_name -> (x, y)
    for ln in ax.get_lines():
        if not isinstance(ln, Line2D):  # defensive
            continue
        lab = (ln.get_label() or "").strip()
        if lab in label_map:
            raw = label_map[lab]
            x = np.asarray(ln.get_xdata())
            y = np.asarray(ln.get_ydata())
            series_dict[raw] = (x, y, lab)  # store both raw & original

    # 2) Also look for bar plots (BarContainer)
    for child in ax.containers:
        if isinstance(child, BarContainer):
            lab = (child.get_label() or "").strip()
            if lab in label_map:
                raw = label_map[lab]
                # bar x are rect centers; y are heights
                rects = list(child)
                x = np.array([r.get_x() + r.get_width()/2.0 for r in rects])
                y = np.array([r.get_height() for r in rects])
                series_dict[raw] = (x, y, lab)

    if not series_dict:
        print("[export] Skipped: no matching labels among lines/bars on current axes.")
        return

    # Determine index (x-axis): prefer numeric xdata; if categorical, use tick labels
    # Choose the first found series as reference for x
    ref_raw = next(iter(series_dict.keys()))
    x_ref = series_dict[ref_raw][0]

    if x_ref.dtype.kind in ("U", "S", "O"):  # already string-like
        index = pd.Index(x_ref, name="index")
    else:
        # If x are positions (0..N-1) and categorical ticks exist, use tick labels
        ticks = ax.get_xticks()
        ticklabs = [t.get_text() for t in ax.get_xticklabels()]
        if len(ticklabs) == len(x_ref) and any(ticklabs):
            index = pd.Index(ticklabs, name="index")
        else:
            index = pd.Index(x_ref, name="index")

    df = pd.DataFrame(index=index)

    # Fill both the raw-name and the human label columns
    for raw, (x, y, lab) in series_dict.items():
        # Align by order; assume plotted x aligns with index order
        # (If index came from tick labels, we preserved order)
        df[raw] = y
        df[lab] = y

    # Reorder/limit to requested columns if present
    desired = ["consumption_planned", "Consumption (Planned)",
               "purchase_planned",    "Purchase (Planned)"]
    cols = [c for c in desired if c in df.columns]
    df = df[cols]

    df.to_csv(out_csv)
    print(f"[export] Saved {out_csv} with columns {list(df.columns)}; index from plot x-axis.")


def _coerce_number(x):
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
    df = pd.read_csv(path, header=None)
    months = df.iloc[6].copy().ffill()
    attrs  = df.iloc[7].copy()
    subs   = df.iloc[8].copy()

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
    
    raise ValueError(f"Material '{material_name}' not found in column 1 or column 2.")

def build_timeseries_for_material(csv_path, material=MATERIAL, months_keep=MONTHS):
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
    wanted.append(("inv_days","planned"))  # Add planned inv_days for T/F values

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
            rec[f"{attr}_{stat}"] = float(val.iloc[0]) if len(val) else np.nan
        records.append(rec)

    out = pd.DataFrame.from_records(records).set_index("month")
    return out

def plot_slow(df1, df2, label1=LABEL1, label2=LABEL2):
    """
    Overlay two lines (df1 vs df2) for each metric and add % difference:
    pct = 100 * (df2[col] - df1[col]) / df1[col].
    """
    plots = [
        ("consumption_actual", "Consumption (Actual)"),
        ("consumption_planned", "Consumption (Planned)"),
        ("purchase_actual", "Purchase (Actual)"),
        ("purchase_planned", "Purchase (Planned)")
    ]
    for col, title in plots:
        plt.figure(figsize=(9,5))
        ax = plt.gca()

        # Base lines
        ax.plot(df1.index, df1[col], marker="o", label=label1)
        ax.plot(df2.index, df2[col], marker="o", label=label2)

        # % difference on secondary axis: (df2 - df1)/df1 * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100.0 * (df2[col].to_numpy(dtype=float) - df1[col].to_numpy(dtype=float)) / df1[col].to_numpy(dtype=float)
        pct[~np.isfinite(pct)] = np.nan

        ax2 = ax.twinx()
        # ax2.plot(df1.index, pct, linestyle="--", marker="o", alpha=0.9, label=f"% change vs {label1}")
        ax2.plot(df1.index, pct, linestyle="--", marker="o", color="red", alpha=0.9,
                 label=f"% change vs {label1}")
        ax2.axhline(0.0, linewidth=1, alpha=0.25)
        ax2.set_ylabel("Percent difference (%)")

        # Optional annotations for readability (12 points -> annotate all)
        for x, y in zip(df1.index, pct):
            if np.isfinite(y):
                ax2.annotate(f"{y:+.1f}%", xy=(x, y),
                             xytext=(0, 6), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8)

        # Axes cosmetics
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.25)
        plt.xticks(rotation=45)

        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

        plt.tight_layout()
        plt.show()
        time.sleep(2)  # pause before next plot

def plot_inv_days(df1, df2, label1=LABEL1, label2=LABEL2):
    """
    Plot Inventory Days: use actual values when available, otherwise use planned values.
    This creates a hybrid series showing Act values first, then T/F (planned) values.
    Labels show Plan Aug and Plan Sep (not actual/planned).
    """
    plt.figure(figsize=(9,5))
    ax = plt.gca()

    # Create hybrid series: prefer actual, fall back to planned
    inv_days1 = df1['inv_days_actual'].fillna(df1['inv_days_planned'])
    inv_days2 = df2['inv_days_actual'].fillna(df2['inv_days_planned'])

    # Base lines - using label1 and label2 which are "Plan as of Aug-25" and "Plan as of Sep-25"
    ax.plot(df1.index, inv_days1, marker="o", label=label1)
    ax.plot(df2.index, inv_days2, marker="o", label=label2)

    # % difference on secondary axis: (df2 - df1)/df1 * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (inv_days2.to_numpy(dtype=float) - inv_days1.to_numpy(dtype=float)) / inv_days1.to_numpy(dtype=float)
    pct[~np.isfinite(pct)] = np.nan

    ax2 = ax.twinx()
    ax2.plot(df1.index, pct, linestyle="--", marker="o", color="red", alpha=0.9,
             label=f"% change vs {label1}")
    ax2.axhline(0.0, linewidth=1, alpha=0.25)
    ax2.set_ylabel("Percent difference (%)")

    # Optional annotations for readability (12 points -> annotate all)
    for x, y in zip(df1.index, pct):
        if np.isfinite(y):
            ax2.annotate(f"{y:+.1f}%", xy=(x, y),
                         xytext=(0, 6), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    # Axes cosmetics
    ax.set_title("Inventory Days (Actual/Planned)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Inventory Days")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=45)

    # Combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.show()
    time.sleep(2)  # pause before next plot

def main():
    print(f"Reading:\n  1) {CSV1}\n  2) {CSV2}\nMaterial: {MATERIAL}\n")
    df1 = build_timeseries_for_material(CSV1)
    df2 = build_timeseries_for_material(CSV2)

    print("Preview — 4.0:\n", df1.head(), "\n")
    print("Preview — 5.0:\n", df2.head(), "\n")

    # Export consumption data to CSV
    consumption_export = pd.DataFrame({
        'Month': df1.index,
        f'{LABEL1}': df1['consumption_planned'],
        f'{LABEL2}': df2['consumption_planned']
    })
    
    # Calculate percentage change for consumption
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = 100.0 * (df2['consumption_planned'].to_numpy(dtype=float) - 
                               df1['consumption_planned'].to_numpy(dtype=float)) / \
                               df1['consumption_planned'].to_numpy(dtype=float)
    pct_change[~np.isfinite(pct_change)] = np.nan
    consumption_export['% change'] = pct_change
    
    csv_filename = f"consumption_planned_{MATERIAL.replace(' ', '_')}.csv"
    consumption_export.to_csv(csv_filename, index=False)
    print(f"Exported consumption data to: {csv_filename}")

    # Export purchase data to CSV
    purchase_export = pd.DataFrame({
        'Month': df1.index,
        f'{LABEL1}': df1['purchase_planned'],
        f'{LABEL2}': df2['purchase_planned']
    })
    
    # Calculate percentage change for purchase
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change_purch = 100.0 * (df2['purchase_planned'].to_numpy(dtype=float) - 
                                      df1['purchase_planned'].to_numpy(dtype=float)) / \
                                      df1['purchase_planned'].to_numpy(dtype=float)
    pct_change_purch[~np.isfinite(pct_change_purch)] = np.nan
    purchase_export['% change'] = pct_change_purch
    
    csv_filename_purch = f"purchase_planned_{MATERIAL.replace(' ', '_')}.csv"
    purchase_export.to_csv(csv_filename_purch, index=False)
    print(f"Exported purchase data to: {csv_filename_purch}\n")

    plot_slow(df1, df2)
    
    # Add inventory days plot
    plot_inv_days(df1, df2)
 

if __name__ == "__main__":
    main()
