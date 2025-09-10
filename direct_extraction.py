import pandas as pd
import numpy as np

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
    """
    Assumes a multi-row header:
      - Row 6: month labels (e.g., 'Apr-25'), sparse; forward-filled.
      - Row 7: attribute names (e.g., 'Pur','Consumption','C/B','Inv Days'); blank cells inherit previous attr within month block.
      - Row 8: sublabels (e.g., 'L/F','T/F','Act' or blank) -> planned vs actual.
    """
    df = pd.read_csv(path, header=None)
    months = df.iloc[6].copy().ffill()
    attrs  = df.iloc[7].copy()
    subs   = df.iloc[8].copy()

    # Forward-fill attribute names only when month present (blank = continuation of previous attr)
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
    """
    Finds the row where column 1 equals material_name (case-insensitive).
    """
    mask = df.iloc[:, 1].astype(str).str.strip().str.lower() == material_name.strip().lower()
    if not mask.any():
        raise ValueError(f"Material '{material_name}' not found in column 1.")
    idx = mask.idxmax()
    return df.iloc[idx]

def build_timeseries_for_material(
    csv_path,
    material="Silver Paste",
    months_keep=("Apr-25","May-25","Jun-25","Jul-25","Aug-25","Sep-25",
                 "Oct-25","Nov-25","Dec-25","Jan-26","Feb-26","Mar-26")
):
    df, header = load_multiheader_csv(csv_path)
    row = extract_material_row(df, material)

    tidy = header.copy()
    tidy["value"] = row.values  # align with all columns
    tidy = tidy[tidy["month"].isin(months_keep)].copy()

    def norm_attr(a):
        a = str(a).strip().lower()
        if a in {"pur", "purchase", "purch"}:
            return "purchase"
        if a in {"consumption", "cons"}:
            return "consumption"
        if a in {"c/b", "cb", "closing balance", "closing_bal", "c_b"}:
            return "cb"
        if a in {"inv days", "invdays", "inventory days", "inventory_days"}:
            return "inv_days"
        return a

    tidy["attribute"] = tidy["attribute"].apply(norm_attr)
    tidy["value"] = tidy["value"].apply(_coerce_number)

    # Columns to produce
    wanted = []
    for attr in ["purchase", "consumption", "cb"]:
        for stat in ["planned", "actual"]:
            wanted.append((attr, stat))
    wanted.append(("inv_days", "actual"))

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

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Extract monthly timeseries (planned/actual) for a material from a multi-row header CSV."
    )
    p.add_argument("--csv", type=str, required=True, help="Path to the CSV (e.g., '5.0 FY2025 Material Control.csv').")
    p.add_argument("--material", type=str, default="Silver Paste", help="Material name to extract.")
    p.add_argument("--out", type=str, default="silver_paste_timeseries.csv", help="Output CSV path.")
    args = p.parse_args()

    df = build_timeseries_for_material(args.csv, args.material)
    df.to_csv(args.out, index=True)
    print("Saved:", args.out)
    print(df)

if __name__ == "__main__":
    main()
