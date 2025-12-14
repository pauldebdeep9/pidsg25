# plot_pidsg25_eval.py
# Show 6 figures (no saving) from the LAST sheet of pidsg25Evaluation.xlsx:
# 1) Manual Orders (Hedged/Unhedged) as bars + Hedged/Unhedged price (dotted, secondary y)
# 2) AI Orders (Hedged/Unhedged) as bars + Hedged/Unhedged price (dotted, secondary y)
# 3) Storage costs: Manual vs AI
# 4) Cumulative Unhedged Cost: Manual vs AI
# 5) Cumulative Storage Cost: Manual vs AI
# 6) Cumulative Total Cost (Unhedged + Storage): Manual vs AI
#
# Run:
#   python plot_pidsg25_eval.py --file pidsg25Evaluation.xlsx

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt


COLUMN_GROUPS = {
    "date": ["date"],
    "price_h": ["hedged price", "price hedged"],
    "price_u": ["unhedged price", "price unhedged"],
    "man_h": ["order-manual-hedged", "manual-hedged", "order man hedged"],
    "man_u": ["order-manual-unhedged", "manual-unhedged", "order man unhedged"],
    "ai_h": ["Order-AI-1-Hed", "order ml hedged", "ai-hedged"],
    "ai_u": ["Order-AI-1-Unhed", "order ml unhedged", "ai-unhedged"],
    "cost_m": ["manual-storage-cost", "manual storage cost"],
    "cost_ai": ["ai-storage-cost", "al-storage-cost", "ai storage cost"],
    "unhedged_m": ["man-unhedged cost", "manual-unhedged-cost", "manual unhedged cost", "man unhedged cost"],
    "unhedged_ai": ["ai-unhedged cost", "ai unhedged cost"],
}


def _norm(col: str) -> str:
    s = (col or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).replace(" ", "-").replace("_", "-")
    s = re.sub(r"\bal-", "ai-", s)  # fix 'Al-' -> 'AI-'
    s = s.replace("un-hedged", "unhedged")
    return s


def _resolve(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # partial fallback
    for cand in candidates:
        key = _norm(cand)
        for k, v in norm_map.items():
            if key in k:
                return v
    raise KeyError(f"Missing any of columns: {candidates}\nAvailable: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to pidsg25Evaluation.xlsx")
    args = ap.parse_args()

    # Load the most recent sheet that has all required columns
    xl = pd.ExcelFile(args.file)
    df = None
    sheet = None
    resolved_cols = None
    for candidate in reversed(xl.sheet_names):
        df_candidate = pd.read_excel(xl, sheet_name=candidate)
        try:
            resolved = {name: _resolve(df_candidate, options) for name, options in COLUMN_GROUPS.items()}
        except KeyError:
            continue
        date_series = pd.to_datetime(df_candidate[resolved["date"]], dayfirst=True, errors="coerce")
        if date_series.dropna().empty:
            continue
        df = df_candidate
        sheet = candidate
        resolved_cols = resolved
        df[resolved_cols["date"]] = date_series
        break

    if df is None or resolved_cols is None:
        raise ValueError(
            "Could not locate a sheet with the required columns. "
            f"Checked sheets: {xl.sheet_names}"
        )

    print(f"Using sheet: {sheet}")

    # Date
    date_col = resolved_cols["date"]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Columns (robust to small variations)
    col_price_h = resolved_cols["price_h"]
    col_price_u = resolved_cols["price_u"]
    col_man_h   = resolved_cols["man_h"]
    col_man_u   = resolved_cols["man_u"]
    col_ai_h    = resolved_cols["ai_h"]
    col_ai_u    = resolved_cols["ai_u"]
    col_cost_m  = resolved_cols["cost_m"]
    col_cost_ai = resolved_cols["cost_ai"]

    # NEW: cost columns for unhedged (manual vs AI)
    col_unhedged_m  = resolved_cols["unhedged_m"]
    col_unhedged_ai = resolved_cols["unhedged_ai"]

    xdates = df[date_col]
    idx = range(len(xdates))
    width = 0.4

    # -------- Fig 1: Manual Orders + Prices --------
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar([i - width/2 for i in idx], df[col_man_h], width=width, label="Manual-Hedged")
    ax1.bar([i + width/2 for i in idx], df[col_man_u], width=width, label="Manual-Unhedged")
    ax1.set_xticks(list(idx))
    ax1.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")  # MMM-YY
    ax1.set_ylabel("Orders (Manual)")
    ax1.set_xlabel("Date")
    ax1b = ax1.twinx()
    ax1b.plot(list(idx), df[col_price_h], linestyle=":", label="Hedged price")
    ax1b.plot(list(idx), df[col_price_u], linestyle=":", label="Unhedged price")
    ax1b.set_ylabel("Price")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("Manual Orders vs Prices")
    plt.tight_layout()
    plt.show()

    # -------- Fig 2: AI Orders + Prices --------
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar([i - width/2 for i in idx], df[col_ai_h], width=width, label="AI-Hedged")
    ax2.bar([i + width/2 for i in idx], df[col_ai_u], width=width, label="AI-Unhedged")
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")  # MMM-YY
    ax2.set_ylabel("Orders (AI)")
    ax2.set_xlabel("Date")
    ax2b = ax2.twinx()
    ax2b.plot(list(idx), df[col_price_h], linestyle=":", label="Hedged price")
    ax2b.plot(list(idx), df[col_price_u], linestyle=":", label="Unhedged price")
    ax2b.set_ylabel("Price")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("AI Orders vs Prices")
    plt.tight_layout()
    plt.show()

    # -------- Fig 3: Storage Costs --------
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar([i - width/2 for i in idx], df[col_cost_m], width=width, label="Manual-storage-cost")
    ax3.bar([i + width/2 for i in idx], df[col_cost_ai], width=width, label="AI-storage-cost")
    ax3.set_xticks(list(idx))
    ax3.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")  # MMM-YY
    ax3.set_ylabel("Storage cost")
    ax3.set_xlabel("Date")
    ax3.legend(loc="best")
    plt.title("Storage Costs")
    plt.tight_layout()
    plt.show()

    # =========================
    # NEW PLOTS (4, 5, 6)
    # =========================

    # Prepare cumulative series (fill NaNs with 0 before cumsum)
    cum_unhedged_m  = df[col_unhedged_m].fillna(0).cumsum()
    cum_unhedged_ai = df[col_unhedged_ai].fillna(0).cumsum()
    cum_storage_m   = df[col_cost_m].fillna(0).cumsum()
    cum_storage_ai  = df[col_cost_ai].fillna(0).cumsum()

    # -------- Fig 4: Cumulative Unhedged Cost --------
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(list(idx), cum_unhedged_m,  marker="o", label="Cumulative Man-unhedged cost")
    ax4.plot(list(idx), cum_unhedged_ai, marker="o", label="Cumulative AI-unhedged cost")
    ax4.set_xticks(list(idx))
    ax4.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")
    ax4.set_ylabel("Cumulative cost")
    ax4.set_xlabel("Date")
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(loc="best")
    plt.title("Cumulative Unhedged Cost — Manual vs AI")
    plt.tight_layout()
    plt.show()

    # -------- Fig 5: Cumulative Storage Cost --------
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(list(idx), cum_storage_m,  marker="o", label="Cumulative Manual-storage-cost")
    ax5.plot(list(idx), cum_storage_ai, marker="o", label="Cumulative AI-storage-cost")
    ax5.set_xticks(list(idx))
    ax5.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")
    ax5.set_ylabel("Cumulative cost")
    ax5.set_xlabel("Date")
    ax5.grid(True, axis="y", alpha=0.3)
    ax5.legend(loc="best")
    plt.title("Cumulative Storage Cost — Manual vs AI")
    plt.tight_layout()
    plt.show()

    # -------- Fig 6: Cumulative Total Cost (Unhedged + Storage) --------
    cum_total_m  = cum_unhedged_m + cum_storage_m
    cum_total_ai = cum_unhedged_ai + cum_storage_ai

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.plot(list(idx), cum_total_m,  marker="o", label="Cumulative Total Cost — Manual")
    ax6.plot(list(idx), cum_total_ai, marker="o", label="Cumulative Total Cost — AI")
    ax6.set_xticks(list(idx))
    ax6.set_xticklabels([d.strftime("%b-%y") for d in xdates], rotation=45, ha="right")
    ax6.set_ylabel("Cumulative total cost")
    ax6.set_xlabel("Date")
    ax6.grid(True, axis="y", alpha=0.3)
    ax6.legend(loc="best")
    plt.title("Cumulative Total Cost (Unhedged + Storage) — Manual vs AI")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    sys.argv = ["", "--file", "pidsg26Evaluation09.xlsx"]  # default argument
    main()
