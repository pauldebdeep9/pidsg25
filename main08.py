# main06.py  — aligned & robust plotting version

import pandas as pd
import numpy as np
import yaml

from ClassData import ProcurementConfig, ModelData
from model import solve_price_saa
from postprocess_order import extract_order_matrices
from plots import (
    plot_order_placement_bar,
    plot_price_distribution_band,
    plot_price_and_orders,
    plot_price_and_orders_deterministic,  # imported here
)
from price_distributions import PriceDistributionGenerator
from cost import Cost

# -------------------- Load Excel & config --------------------
file_path = "pidsg25-06.xlsx"
xls = pd.ExcelFile(file_path)

all_distribution_params = {
    "lognormal":   {"mean1": 3.8, "sigma1": 0.25, "mean2": 4.0, "sigma2": 0.3},
    "gamma":       {"shape1": 2.0, "scale1": 22.0, "shape2": 2.5, "scale2": 25.0},
    "normal":      {"mean1": 45, "std1": 5, "mean2": 50, "std2": 6},
    "pareto":      {"alpha1": 3.0, "scale1": 40.0, "alpha2": 2.5, "scale2": 45.0},
    "triangular":  {"left1": 40, "mode1": 45, "right1": 50, "left2": 42, "mode2": 48, "right2": 55},
    "weibull":     {"a1": 1.5, "scale1": 50.0, "a2": 1.2, "scale2": 55.0},
    "beta":        {"a1": 2.0, "b1": 5.0, "scale1": 100, "a2": 2.5, "b2": 4.5, "scale2": 110},
}

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

dist_name = config["distribution_name"]
distribution_params = all_distribution_params[dist_name]

T = config["problem"]["T"]
N = config["problem"]["N"]
seed = config["problem"]["seed"]
h = config["problem"]["h"]
b = config["problem"]["b"]
I_0 = config["problem"]["I_0"]
B_0 = config["problem"]["B_0"]
enforce_fixed_orders = config["problem"]["enforce_fixed_orders"]

# -------------------- Load sheets --------------------
demand_df   = pd.read_excel(xls, sheet_name="demand",   index_col=0)
supplier_df = pd.read_excel(xls, sheet_name="supplier")          # has columns: supplier, lead_time, order_cost
capacity_df = pd.read_excel(xls, sheet_name="capacity", index_col=0)

# -------------------- Synthetic price samples --------------------
generator = PriceDistributionGenerator(T=T, N=N, seed=42)
price_df_s1, price_df_s2 = generator.generate_by_name(dist=dist_name, params=distribution_params)

# -------------------- Fixed demand & sets --------------------
fixed_demand = demand_df["Actual"].dropna().values
T = len(fixed_demand)               # authoritative horizon
S = supplier_df["supplier"].tolist()
lead_time = dict(zip(supplier_df["supplier"], supplier_df["lead_time"]))
lead_time_s2 = int(lead_time["s2"])

# -------------------- Optional raw orders for s2 (by placement t) --------------------
# NOTE: raw orders may be shorter than T; we will align/zero-fill to T later.
raw_orders_s2 = {
    **{i: 4886.83127572017 for i in range(3)},
    **{i: 2764.92 if i == 3 else 2767.36 for i in range(3, 21)},
}

fixed_orders_s2 = (
    {(t, t + lead_time_s2): q for t, q in raw_orders_s2.items() if t + lead_time_s2 < T}
    if enforce_fixed_orders else None
)

print("Fixed orders with arrival time:", fixed_orders_s2)

# -------------------- Price samples for SAA --------------------
price_samples = []
for i in range(N):
    sample_prices = {}
    for t in range(T):
        sample_prices[(t, "s1")] = float(price_df_s1.iloc[t, i])
        sample_prices[(t, "s2")] = float(price_df_s2.iloc[t, i])
    price_samples.append(sample_prices)

# -------------------- Costs & capacities --------------------
order_cost = dict(zip(supplier_df["supplier"], supplier_df["order_cost"]))

# Reset capacity index to 0..T-1 (robust to input indexing) and build dict
capacity_df = capacity_df.copy()
capacity_df.index = range(len(capacity_df))
time_index_for_model = list(capacity_df.index)
capacity_dict = {(t, s): float(capacity_df.loc[t, s]) for t in capacity_df.index for s in capacity_df.columns}

# -------------------- Solve SAA (price uncertainty) --------------------
obj_val, df_result = solve_price_saa(
    fixed_demand=fixed_demand,
    price_samples=price_samples[:10],
    order_cost=order_cost,
    lead_time=lead_time,
    capacity_dict=capacity_dict,
    h=h,
    b=b,
    I_0=I_0,
    B_0=B_0,
    fixed_orders_s2=fixed_orders_s2,
    time_index=time_index_for_model,
)

# -------------------- Postprocess --------------------
print("Objective Value:", obj_val)
print(df_result)

# Extract orders placed/arrived from the solver result
order_placed, order_arr = extract_order_matrices(df_result)

# Ensure order_placed is T×|S| with columns ['s1','s2'] and zero-fill any gaps.
# (If your helper already returns that, this is a harmless no-op.)
expected_cols = ["s1", "s2"]
# If columns are unnamed positions, coerce to expected names
if list(order_placed.columns) != expected_cols and len(order_placed.columns) == 2:
    order_placed.columns = expected_cols

order_placed = (
    order_placed.reindex(index=range(T), fill_value=0.0)  # align to horizon T
                 .reindex(columns=expected_cols, fill_value=0.0)
)

# Overwrite s2 column with aligned raw orders (placement-time t), zero elsewhere
s2_aligned = pd.Series(0.0, index=range(T))
for t, q in raw_orders_s2.items():
    if 0 <= t < T:
        s2_aligned.iloc[t] = float(q)

order_placed["s2"] = s2_aligned.values  # safe: length T

# -------------------- Plots (robust alignment) --------------------
START_DATE = "2025-08-01"

plot_order_placement_bar(order_placed, start_date=START_DATE)
plot_price_distribution_band(price_df_s1, price_df_s2, start_date=START_DATE)
plot_price_and_orders(price_df_s1, order_placed, supplier="s1", start_date=START_DATE)
plot_price_and_orders(price_df_s2, order_placed, supplier="s2", start_date=START_DATE)

# Deterministic plot: pass a DatetimeIndex Series so the plotting helper has dates + length T
idx = pd.date_range(START_DATE, periods=T, freq="MS")
mean_price_s2_series = pd.Series(price_df_s2.iloc[:T, 0].to_numpy(), index=idx)
plot_price_and_orders_deterministic(mean_price_s2_series, order_placed, supplier="s2")

# -------------------- Save & cost breakdown --------------------
print("Raw orders (s2):", raw_orders_s2)
print("Enforced fixed_orders_s2 (with arrival):", fixed_orders_s2)

output_filename = f"order_placed_{dist_name}.csv"
order_placed.to_csv(output_filename, index=True)
print(f"Saved order_placed to {output_filename}")

cost = Cost(df_result, order_placed, initial_inventory=I_0, demand=fixed_demand)
inv_cost, backlog_cost = cost.compute_inventory_backlog_cost(h, b)
print("Storage cost", inv_cost)
print("Backlog cost", backlog_cost)
