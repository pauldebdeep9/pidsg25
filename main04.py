
import pandas as pd
import numpy as np
import yaml
from dataclass import ProcurementConfig, ModelData
from model import solve_price_saa
from postprocess_order import extract_order_matrices
from plots import (plot_order_placement_bar, 
                   plot_price_distribution_band, 
                   plot_price_and_orders, 
                   plot_price_and_orders_deterministic)
from price_distributions import PriceDistributionGenerator
from cost import Cost


# # Load Excel file
file_path = "pidsg25-02_historical.xlsx"
xls = pd.ExcelFile(file_path)

# Define parameters for all supported distributions
all_distribution_params = {
    "lognormal": {"mean1": 3.8, "sigma1": 0.25, "mean2": 4.0, "sigma2": 0.3},
    "gamma": {"shape1": 2.0, "scale1": 22.0, "shape2": 2.5, "scale2": 25.0},
    "normal": {"mean1": 45, "std1": 5, "mean2": 50, "std2": 6},
    "pareto": {"alpha1": 3.0, "scale1": 40.0, "alpha2": 2.5, "scale2": 45.0},
    "triangular": {"left1": 40, "mode1": 45, "right1": 50, "left2": 42, "mode2": 48, "right2": 55},
    "weibull": {"a1": 1.5, "scale1": 50.0, "a2": 1.2, "scale2": 55.0},
    "beta": {"a1": 2.0, "b1": 5.0, "scale1": 100, "a2": 2.5, "b2": 4.5, "scale2": 110}
}

# --- Load config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

dist_name = config["distribution_name"]

# dist_params = config["distribution"]["params"]
distribution_params = all_distribution_params[dist_name]

T = config["problem"]["T"]
N = config["problem"]["N"]
seed = config["problem"]["seed"]
h = config["problem"]["h"]
b = config["problem"]["b"]
I_0 = config["problem"]["I_0"]
B_0 = config["problem"]["B_0"]
enforce_fixed_orders = config["problem"]["enforce_fixed_orders"]

# Load data sheets
demand_df = pd.read_excel(xls, sheet_name="demand", index_col=0)
supplier_df = pd.read_excel(xls, sheet_name="supplier")
capacity_df = pd.read_excel(xls, sheet_name="capacity", index_col=0)



# Generate synthetic price data
generator = PriceDistributionGenerator(T=T, N= N, seed=42)
price_df_s1, price_df_s2 = generator.generate_by_name(dist=dist_name, params=distribution_params)


# --- 1. Fixed deterministic demand
fixed_demand = demand_df["Actual"].dropna().values
T = len(fixed_demand)
S = supplier_df["supplier"].tolist()
# N = price_df_s1.shape[1]

# --- 2. Supplier lead times
lead_time = dict(zip(supplier_df["supplier"], supplier_df["lead_time"]))
lead_time_s2 = int(lead_time["s2"])

# --- 3. Optional raw orders for s2 (order_time: quantity)
raw_orders_s2 = {
    **{i: 4886.83127572017 for i in range(6)},
    **{i: 2764.92 if i == 6 else 2767.36 for i in range(6, 12)}
}


enforce_fixed_orders = True  # Toggle
fixed_orders_s2 = {
    (t, t + lead_time_s2): q
    for t, q in raw_orders_s2.items()
    if t + lead_time_s2 < T
} if enforce_fixed_orders else None



print("Fixed orders with arrival time:", fixed_orders_s2)

# --- 4. Construct price samples [(t,s) -> price] for each sample
price_samples = []
for i in range(N):
    sample_prices = {}
    for t in range(T):
        sample_prices[(t, 's1')] = price_df_s1.iloc[t, i]
        sample_prices[(t, 's2')] = price_df_s2.iloc[t, i]
    price_samples.append(sample_prices)

# --- 5. Supplier order costs
order_cost = dict(zip(supplier_df["supplier"], supplier_df["order_cost"]))

# --- 6. Time-supplier capacities
capacity_dict = {(t, s): capacity_df.loc[t + 1, s] for t in range(T) for s in S}

# --- 7. Solve the price uncertainty SAA problem
obj_val, df_result = solve_price_saa(
    fixed_demand=fixed_demand,
    price_samples=price_samples[:5],
    order_cost=order_cost,
    lead_time=lead_time,
    capacity_dict=capacity_dict,
    h=h,
    b=b,
    I_0=I_0,
    B_0=B_0,
    fixed_orders_s2=fixed_orders_s2
)

# --- 8. Postprocess and plot
print("Objective Value:", obj_val)
print(df_result)

order_placed, order_arr = extract_order_matrices(df_result)
# Replace second column of order_placed
# Convert dict to array
raw_orders_s2_array = np.array([raw_orders_s2[k] for k in sorted(raw_orders_s2.keys())])

# Then assign
order_placed.iloc[:, 1] = raw_orders_s2_array



plot_order_placement_bar(order_placed, start_date="2025-07-01")
plot_price_distribution_band(price_df_s1, price_df_s2, start_date="2025-07-01")
plot_price_and_orders(price_df_s1, order_placed, supplier='s1', start_date="2025-07-01")
plot_price_and_orders(price_df_s2, order_placed, supplier='s2', start_date="2025-07-01")

mean_price_s2 = price_df_s2.iloc[:, 0].values
plot_price_and_orders_deterministic(mean_price_s2, order_placed, supplier='s2', start_date="2025-07-01")

print("Raw orders:", raw_orders_s2)
print("Enforced fixed_orders_s2 (with arrival):", fixed_orders_s2)

# --- 9. Save order_placed to CSV ---
output_filename = f"order_placed_{dist_name}.csv"
order_placed.to_csv(output_filename, index=True)
print(f"Saved order_placed to {output_filename}")


cost= Cost(df_result, order_placed, initial_inventory=I_0, demand=fixed_demand)
inv_cost, backlog_cost = cost.compute_inventory_backlog_cost(h, b)

print('Storage cost', inv_cost)
print('Backlog cost', backlog_cost)
