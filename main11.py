# main06.py  — aligned & robust plotting version

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

# from ClassData import ProcurementConfig, ModelData  # Commented out as not used
from model import solve_price_saa
from postprocess_order import extract_order_matrices
from plots import (
    plot_price_distribution_band,
    plot_price_and_orders,
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

# Load p1normal and p2normal sheets for prices
p1normal_df = pd.read_excel(xls, sheet_name="p1normal", index_col=0)
p2normal_df = pd.read_excel(xls, sheet_name="p2normal", index_col=0)

# -------------------- Fixed demand & sets --------------------
fixed_demand = demand_df["Actual"].dropna().values
T = len(fixed_demand)               # authoritative horizon - use actual demand length

# -------------------- Use prices from Excel sheets --------------------
# Use p1normal for s1 prices (already has samples)
price_df_s1 = p1normal_df.iloc[:T, :N].copy()
price_df_s1.index = range(T)
price_df_s1.columns = range(N)

# Use p2normal for s2 prices (flat prices repeated for all samples)
p2_flat_prices = p2normal_df.iloc[:T, 0].values
price_df_s2_flat = pd.DataFrame(
    np.tile(p2_flat_prices.reshape(-1, 1), (1, N)),
    index=range(T),
    columns=range(N)
)
S = supplier_df["supplier"].tolist()
lead_time = dict(zip(supplier_df["supplier"], supplier_df["lead_time"]))
lead_time_s2 = int(lead_time["s2"])

# -------------------- Optional raw orders for s2 (by placement t) --------------------
# Load orders from Excel file 'order-1' sheet
orders_df = pd.read_excel(xls, sheet_name="order-1")
# Convert to dictionary: index (time period) -> order quantity
# The index represents the time period starting from 0
raw_orders_s2 = {i: float(orders_df.loc[i, 'Order']) for i in range(len(orders_df))}

print(f"Loaded {len(raw_orders_s2)} orders from 'order-1' sheet")
print("First few orders:", {k: raw_orders_s2[k] for k in list(raw_orders_s2.keys())[:5]})

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
        sample_prices[(t, "s2")] = float(price_df_s2_flat.iloc[t, i])  # Use flat prices for s2
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
START_DATE = "2025-10-01"

plot_price_distribution_band(price_df_s1, price_df_s2_flat, start_date=START_DATE)
plot_price_and_orders(price_df_s1, order_placed, supplier="s1", start_date=START_DATE)
plot_price_and_orders(price_df_s2_flat, order_placed, supplier="s2", start_date=START_DATE)  # Use flat prices

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

# -------------------- Plot inventory cost over time --------------------
# Note: This will be updated after Step 2 optimization with storage constraint comparison

# -------------------- Plot orders from both suppliers with prices --------------------
# Create figure with primary and secondary y-axes
fig, ax1 = plt.subplots(figsize=(14, 7))

# Create date index for orders
order_date_index = pd.date_range(START_DATE, periods=T, freq="MS")

# Calculate bar width and positions for side-by-side bars
bar_width = 10  # days
offset = pd.Timedelta(days=bar_width)

# Plot orders (placement time) on primary axis - side by side
ax1.bar(order_date_index - offset, order_placed["s1"], width=bar_width, alpha=0.7, 
        label="Reoptimized Unhedged Order (Placement)", color='tab:blue')
ax1.bar(order_date_index + offset, order_placed["s2"], width=bar_width, alpha=0.7, 
        label="Reoptimized Hedged Order (Placement)", color='tab:orange')

ax1.set_xlabel("Time Period", fontsize=12)
ax1.set_ylabel("Order Quantity (units)", fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(order_date_index)
ax1.set_xticklabels(order_date_index.strftime('%Y-%m'), rotation=45, ha='right')

# Create secondary y-axis for prices
ax2 = ax1.twinx()

# Plot mean prices for both suppliers on secondary axis
# These use the exact same price calculation as the individual plots
mean_price_s1 = price_df_s1.mean(axis=1)
mean_price_s2 = price_df_s2_flat.mean(axis=1)

ax2.plot(order_date_index, mean_price_s1, marker='o', linewidth=2, markersize=5, 
         label="Unhedged Mean Price", color='darkblue', linestyle='--')
ax2.plot(order_date_index, mean_price_s2, marker='s', linewidth=2, markersize=5, 
         label="Hedged Mean Price", color='darkorange', linestyle='--')

ax2.set_ylabel("Price ($/unit)", fontsize=12)
ax2.legend(loc='upper right', fontsize=10)

plt.title(f"Step 1: Orders and Prices (Without Storage Constraint, Lead Time = {lead_time_s2})", fontsize=14)
plt.tight_layout()
plt.show()

print("\nℹ Note: Step 2 combined plot (with storage constraint) will be shown after optimization completes.")

# -------------------- Create detailed results CSV --------------------
from datetime import datetime

# Create date range for the results
result_dates = pd.date_range(START_DATE, periods=T, freq="MS")

# Get mean prices for each supplier
price_supplier_1 = price_df_s1.mean(axis=1).values
price_supplier_2 = price_df_s2_flat.mean(axis=1).values

# Get orders for each supplier
order_supplier_1 = order_placed["s1"].values
order_supplier_2 = order_placed["s2"].values

# Calculate total orders
order_supplier_total = order_supplier_1 + order_supplier_2

# Calculate storage (amount) = previous stock - demand + total order
storage = np.zeros(T)
storage[0] = I_0 - fixed_demand[0] + order_supplier_total[0]
for t in range(1, T):
    storage[t] = storage[t-1] - fixed_demand[t] + order_supplier_total[t]

# Calculate costs
cost_supplier_1 = price_supplier_1 * order_supplier_1
cost_supplier_2 = price_supplier_2 * order_supplier_2
cost_supplier_total = cost_supplier_1 + cost_supplier_2

# Calculate storage cost (storage * 0.05)
storage_cost = storage * 0.05

# Create the results dataframe
results_df = pd.DataFrame({
    'date': result_dates,
    'demand': fixed_demand,
    'price_supplier_1': price_supplier_1,
    'price_supplier_2': price_supplier_2,
    'order_supplier_1': order_supplier_1,
    'order_supplier_2': order_supplier_2,
    'order_supplier_total': order_supplier_total,
    'storage_amount': storage,
    'cost_supplier_1': cost_supplier_1,
    'cost_supplier_2': cost_supplier_2,
    'cost_supplier_total': cost_supplier_total,
    'storage_cost': storage_cost
})

# Generate timestamp and filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f"result_step1_{timestamp}.csv"

# Save to CSV
results_df.to_csv(results_filename, index=False)
print(f"\nDetailed results saved to: {results_filename}")
print(f"\nFirst few rows of results:")
print(results_df.head())

# ================================================================================
# STEP 2: STORAGE-CONSTRAINED OPTIMIZATION
# ================================================================================
print("\n" + "="*80)
print("STEP 2: Re-optimizing orders with storage capacity constraint")
print("="*80)

# -------------------- Step 2 Functions --------------------
def deterministic_reschedule_unhedged_L0_lock_prefix(
    Q_star_df: pd.DataFrame,    # T x 2 with columns ["Hedged","Unhedged"] (AI plan)
    price_df: pd.DataFrame,     # T x 2 with same columns
    capacity_df: pd.DataFrame,  # T x 2 with same columns
    demand: np.ndarray,         # length T
    k: int = 4,                 # lock the first k months of Unhedged to the original plan
    s_max: float = 18000.0,     # per-period on-hand storage cap
    h: float = 5.0,             # holding cost per unit per period
    I0: float = 0.0,            # initial on-hand inventory
    no_backlog: bool = True,    # enforce no backorders (recommended for L=0)
    solver=None,
    verbose: bool = False,
):
    """
    Lead time = 0 model with prefix lock:
      - Hedged schedule is fixed to Q_star_df['Hedged'].
      - Unhedged is fixed for the first k periods to Q_star_df['Unhedged'].
      - Optimize only Unhedged for periods k..T-1.
      - Objective: minimize procurement cost (Unhedged) + holding cost (storage).
      - Enforce per-period storage cap s_max.
      - Default: no backlog allowed.
    """
    import cvxpy as cp
    
    if solver is None:
        solver = cp.CLARABEL
    
    # Check columns
    required_cols = ["Hedged", "Unhedged"]
    for df_ in (Q_star_df, price_df, capacity_df):
        if list(df_.columns) != required_cols:
            raise ValueError(f"Columns must be exactly {required_cols}.")
        for c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")

    d = np.asarray(demand, dtype=float).reshape(-1)
    T = len(Q_star_df)
    if d.size != T:
        raise ValueError("demand length must equal the planning horizon T.")
    if not (0 <= k <= T):
        raise ValueError("k must be between 0 and T inclusive.")

    # Constants
    QH = Q_star_df["Hedged"].to_numpy(float)       # fixed hedged orders
    QU_fix_prefix = Q_star_df["Unhedged"].to_numpy(float)  # used to lock first k periods
    PU = price_df["Unhedged"].to_numpy(float)      # price for unhedged
    PH = price_df["Hedged"].to_numpy(float)        # price for hedged (constant part)
    CU = capacity_df["Unhedged"].to_numpy(float)   # capacity for unhedged

    # Variables
    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")
    S  = cp.Variable(T, name="S")
    P  = cp.Variable(T, nonneg=True, name="P")
    N  = cp.Variable(T, nonneg=True, name="N")

    cons = []

    # Prefix lock for Unhedged
    if k > 0:
        cons += [QU[:k] == QU_fix_prefix[:k]]
        cons += [QU[:k] <= CU[:k]]
    if k < T:
        cons += [QU[k:] <= CU[k:]]

    # Inventory dynamics (L=0)
    cons += [S[0] == float(I0) + (QH[0] + QU[0]) - d[0]]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (QH[t] + QU[t]) - d[t]]

    if no_backlog:
        cons += [S == P]
        cons += [N == 0]
        cons += [P >= 0]
    else:
        cons += [S == P - N]

    # Storage cap
    cons += [P <= s_max]

    # Preserve total Unhedged order quantity (critical constraint!)
    cons += [cp.sum(QU) == float(np.sum(QU_fix_prefix))]

    # Objective
    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_cost  = h * cp.sum(P)
    obj = cp.Minimize(purchase_cost + holding_cost)

    # Solve
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"L0 unhedged reschedule (prefix lock) failed: status={prob.status}")

    # Package results
    ThetaH = QH.copy()
    ThetaU = np.asarray(QU.value).reshape(-1)

    out = {
        "objective": float(prob.value),
        "Qhat_unhedged": pd.Series(ThetaU, name="Unhedged"),
        "Q_fixed_hedged": pd.Series(QH, name="Hedged"),
        "Theta_unhedged": pd.Series(ThetaU, name="Theta_Unhedged"),
        "Theta_hedged": pd.Series(ThetaH, name="Theta_Hedged"),
        "S": pd.Series(np.asarray(S.value).reshape(-1), name="S"),
        "P": pd.Series(np.asarray(P.value).reshape(-1), name="P"),
        "N": pd.Series(np.asarray(N.value).reshape(-1), name="N"),
        "max_storage": float(np.max(P.value)),
        "purchase_cost": float(np.sum(PU * ThetaU) + np.dot(PH, QH)),
        "holding_cost": float(h * np.sum(np.asarray(P.value).reshape(-1))),
        "k_locked": int(k),
    }
    return out

def compute_cost_breakdown(sol, price_df, h=5.0, b=None):
    """
    Compute total costs:
      - Hedged purchase cost
      - Unhedged purchase cost
      - Storage (holding) cost
      - (optional) Backlog cost
    """
    # Pull orders
    if "Qhat" in sol:
        hedged_orders   = sol["Qhat"]["Hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat"]["Unhedged"].to_numpy(float)
    else:
        hedged_orders   = sol["Q_fixed_hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat_unhedged"].to_numpy(float)

    # Prices
    pH = price_df["Hedged"].to_numpy(float)
    pU = price_df["Unhedged"].to_numpy(float)

    # Storage
    P = sol["P"].to_numpy(float)

    # Costs
    hedged_cost    = float(np.sum(pH * hedged_orders))
    unhedged_cost  = float(np.sum(pU * unhedged_orders))
    storage_cost   = float(h * np.sum(P))

    out = {
        "hedged_cost": hedged_cost,
        "unhedged_cost": unhedged_cost,
        "storage_cost": storage_cost,
        "total_without_backlog": hedged_cost + unhedged_cost + storage_cost,
    }

    # Optional backlog cost
    if b is not None and "N" in sol:
        N = sol["N"].to_numpy(float)
        backlog_cost = float(b * np.sum(N))
        out["backlog_cost"] = backlog_cost
        out["total_with_backlog"] = out["total_without_backlog"] + backlog_cost

    return out

def plot_cost_comparison(sol_step2, price_df_step2, order_step1_s1, order_step1_s2, 
                         price_s1, price_s2, start_date="2025-10-01"):
    """
    Makes 2 figures comparing cumulative costs with and without storage constraint:
      1) Supplier 1 (Unhedged) cumulative cost comparison
      2) Supplier 2 (Hedged) cumulative cost comparison
    """
    # Extract optimized orders from Step 2
    order_step2_s1 = sol_step2["Qhat_unhedged"].to_numpy(float)
    order_step2_s2 = sol_step2["Q_fixed_hedged"].to_numpy(float)
    
    T = len(order_step2_s1)
    t = np.arange(T)
    
    # Date labels
    months = pd.date_range(start_date, periods=T, freq="MS")
    xlabels = months.strftime("%b-%y")
    
    # Calculate costs for Step 1 (no storage limit)
    cost_step1_s1 = price_s1 * order_step1_s1
    cost_step1_s2 = price_s2 * order_step1_s2
    
    # Calculate costs for Step 2 (with storage limit)
    cost_step2_s1 = price_s1 * order_step2_s1
    cost_step2_s2 = price_s2 * order_step2_s2
    
    # Cumulative costs
    cum_step1_s1 = np.cumsum(cost_step1_s1)
    cum_step1_s2 = np.cumsum(cost_step1_s2)
    cum_step2_s1 = np.cumsum(cost_step2_s1)
    cum_step2_s2 = np.cumsum(cost_step2_s2)
    
    # Figure 1: Supplier 1 (Unhedged) Cumulative Cost
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t, cum_step1_s1, linewidth=2.5, marker='o', markersize=6, 
             label="Without Storage Limit", color='tab:blue')
    ax1.plot(t, cum_step2_s1, linewidth=2.5, marker='s', markersize=6,
             label="With Storage Limit (≤18000)", color='tab:orange')
    
    ax1.set_xlabel("Time Period", fontsize=12)
    ax1.set_ylabel("Cumulative Cost ($)", fontsize=12)
    ax1.set_title("Supplier 1 (Unhedged) - Cumulative Cost Comparison", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xticks(t)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")
    
    # Add final values as text
    final_s1_step1 = cum_step1_s1[-1]
    final_s1_step2 = cum_step2_s1[-1]
    savings_s1 = final_s1_step1 - final_s1_step2
    ax1.text(0.02, 0.98, f"Without limit: ${final_s1_step1:,.0f}\nWith limit: ${final_s1_step2:,.0f}\nSavings: ${savings_s1:,.0f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Supplier 2 (Hedged) Cumulative Cost
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t, cum_step1_s2, linewidth=2.5, marker='o', markersize=6,
             label="Without Storage Limit", color='tab:blue')
    ax2.plot(t, cum_step2_s2, linewidth=2.5, marker='s', markersize=6,
             label="With Storage Limit (≤18000)", color='tab:orange')
    
    ax2.set_xlabel("Time Period", fontsize=12)
    ax2.set_ylabel("Cumulative Cost ($)", fontsize=12)
    ax2.set_title("Supplier 2 (Hedged) - Cumulative Cost Comparison", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_xticks(t)
    ax2.set_xticklabels(xlabels, rotation=45, ha="right")
    
    # Add final values as text
    final_s2_step1 = cum_step1_s2[-1]
    final_s2_step2 = cum_step2_s2[-1]
    savings_s2 = final_s2_step1 - final_s2_step2
    ax2.text(0.02, 0.98, f"Without limit: ${final_s2_step1:,.0f}\nWith limit: ${final_s2_step2:,.0f}\nSavings: ${savings_s2:,.0f}", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# -------------------- Prepare Data for Step 2 --------------------
# Create DataFrame matching Step 2 requirements
Q_star_df_step2 = pd.DataFrame({
    "Hedged": order_supplier_2,      # s2 is Hedged
    "Unhedged": order_supplier_1     # s1 is Unhedged
})

price_df_step2 = pd.DataFrame({
    "Hedged": price_supplier_2,
    "Unhedged": price_supplier_1
})

# Use capacity from Excel (assuming same for all periods for now)
# You can modify this to read from capacity sheet if needed
capacity_df_step2 = pd.DataFrame({
    "Hedged": [20000.0] * T,
    "Unhedged": [20000.0] * T
})

print(f"\nInput data prepared for Step 2:")
print(f"  Time horizon: {T} periods")
print(f"  Initial inventory: {I_0}")
print(f"  Storage capacity constraint: 18000 units")
print(f"  Locking: 0 months (fully optimized)")

# -------------------- Run Step 2 Optimization --------------------
try:
    sol_step2 = deterministic_reschedule_unhedged_L0_lock_prefix(
        Q_star_df=Q_star_df_step2,
        price_df=price_df_step2,
        capacity_df=capacity_df_step2,
        demand=fixed_demand,
        k=0,                    # No locking - fully optimize
        s_max=18000.0,         # Storage cap
        h=0.05,                # Holding cost (same as storage cost rate)
        I0=I_0,
        no_backlog=True,
        solver=None,           # Will use CLARABEL by default
        verbose=False
    )
    
    print(f"\n✓ Step 2 optimization completed successfully!")
    print(f"  Objective value: ${sol_step2['objective']:,.2f}")
    print(f"  Max storage used: {sol_step2['max_storage']:.2f} units (cap: 18000)")
    print(f"  Purchase cost: ${sol_step2['purchase_cost']:,.2f}")
    print(f"  Holding cost: ${sol_step2['holding_cost']:,.2f}")
    print(f"  First {sol_step2['k_locked']} months locked to original plan")
    
    # -------------------- Create Step 2 Results DataFrame --------------------
    # Optimized orders
    order_supplier_1_opt = sol_step2["Qhat_unhedged"].values
    order_supplier_2_opt = sol_step2["Q_fixed_hedged"].values
    order_supplier_total_opt = order_supplier_1_opt + order_supplier_2_opt
    
    # Storage from optimization
    storage_opt = sol_step2["P"].values
    
    # Calculate costs with optimized orders
    cost_supplier_1_opt = price_supplier_1 * order_supplier_1_opt
    cost_supplier_2_opt = price_supplier_2 * order_supplier_2_opt
    cost_supplier_total_opt = cost_supplier_1_opt + cost_supplier_2_opt
    storage_cost_opt = storage_opt * 0.05
    
    results_df_step2 = pd.DataFrame({
        'date': result_dates,
        'demand': fixed_demand,
        'price_supplier_1': price_supplier_1,
        'price_supplier_2': price_supplier_2,
        'order_supplier_1_optimized': order_supplier_1_opt,
        'order_supplier_2_optimized': order_supplier_2_opt,
        'order_supplier_total_optimized': order_supplier_total_opt,
        'storage_amount_optimized': storage_opt,
        'cost_supplier_1_optimized': cost_supplier_1_opt,
        'cost_supplier_2_optimized': cost_supplier_2_opt,
        'cost_supplier_total_optimized': cost_supplier_total_opt,
        'storage_cost_optimized': storage_cost_opt,
        # Include original values for comparison
        'order_supplier_1_original': order_supplier_1,
        'order_supplier_2_original': order_supplier_2,
        'storage_amount_original': storage,
        'cost_supplier_total_original': cost_supplier_total
    })
    
    # Save Step 2 results (separate file)
    results_filename_step2 = f"result_step2_{timestamp}.csv"
    results_df_step2.to_csv(results_filename_step2, index=False)
    print(f"\n✓ Step 2 results saved to: {results_filename_step2}")
    
    # Also add storage constraint columns to the original Step 1 file
    results_df['order_supplier_1_with_storage_limit'] = order_supplier_1_opt
    results_df['order_supplier_2_with_storage_limit'] = order_supplier_2_opt
    results_df['order_supplier_total_with_storage_limit'] = order_supplier_total_opt
    results_df['storage_amount_with_storage_limit'] = storage_opt
    results_df['cost_supplier_1_with_storage_limit'] = cost_supplier_1_opt
    results_df['cost_supplier_2_with_storage_limit'] = cost_supplier_2_opt
    results_df['cost_supplier_total_with_storage_limit'] = cost_supplier_total_opt
    results_df['storage_cost_with_storage_limit'] = storage_cost_opt
    
    # Resave the Step 1 file with additional columns
    results_df.to_csv(results_filename, index=False)
    print(f"✓ Updated {results_filename} with storage constraint results")
    
    # -------------------- Cost Comparison --------------------
    print("\n" + "="*80)
    print("COST COMPARISON: Step 1 vs Step 2")
    print("="*80)
    
    # Step 1 totals
    total_cost_step1 = cost_supplier_total.sum() + storage_cost.sum()
    purchase_cost_step1 = cost_supplier_total.sum()
    storage_cost_step1 = storage_cost.sum()
    
    # Step 2 totals
    total_cost_step2 = sol_step2['objective']
    purchase_cost_step2 = sol_step2['purchase_cost']
    storage_cost_step2 = sol_step2['holding_cost']
    
    print(f"\nStep 1 (No storage constraint):")
    print(f"  Purchase cost: ${purchase_cost_step1:,.2f}")
    print(f"  Storage cost:  ${storage_cost_step1:,.2f}")
    print(f"  Total cost:    ${total_cost_step1:,.2f}")
    
    print(f"\nStep 2 (With storage constraint ≤ 18000):")
    print(f"  Purchase cost: ${purchase_cost_step2:,.2f}")
    print(f"  Storage cost:  ${storage_cost_step2:,.2f}")
    print(f"  Total cost:    ${total_cost_step2:,.2f}")
    
    savings = total_cost_step1 - total_cost_step2
    savings_pct = (savings / total_cost_step1) * 100 if total_cost_step1 > 0 else 0
    print(f"\nSavings: ${savings:,.2f} ({savings_pct:.2f}%)")
    
    # -------------------- Generate Plots --------------------
    print("\n" + "="*80)
    print("Generating cost comparison plots...")
    print("="*80)
    
    # Plot 1: Step 2 Orders and Prices (With Storage Constraint)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Create date index for orders
    order_date_index = pd.date_range(START_DATE, periods=T, freq="MS")
    
    # Calculate bar width and positions for side-by-side bars
    bar_width = 10  # days
    offset = pd.Timedelta(days=bar_width)
    
    # Plot orders from Step 2 (with storage constraint) on primary axis
    ax1.bar(order_date_index - offset, order_supplier_1_opt, width=bar_width, alpha=0.7, 
            label="Unhedged Order (With Storage Constraint)", color='tab:blue')
    ax1.bar(order_date_index + offset, order_supplier_2_opt, width=bar_width, alpha=0.7, 
            label="Hedged Order (With Storage Constraint)", color='tab:orange')
    
    ax1.set_xlabel("Time Period", fontsize=12)
    ax1.set_ylabel("Order Quantity (units)", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(order_date_index)
    ax1.set_xticklabels(order_date_index.strftime('%Y-%m'), rotation=45, ha='right')
    
    # Create secondary y-axis for prices
    ax2 = ax1.twinx()
    
    # Plot mean prices for both suppliers on secondary axis
    # Use the same price data as Step 1 (prices don't change, only orders do)
    mean_price_s1 = price_df_s1.mean(axis=1)
    mean_price_s2 = price_df_s2_flat.mean(axis=1)
    
    ax2.plot(order_date_index, mean_price_s1, marker='o', linewidth=2, markersize=5, 
             label="Unhedged Mean Price", color='darkblue', linestyle='--')
    ax2.plot(order_date_index, mean_price_s2, marker='s', linewidth=2, markersize=5, 
             label="Hedged Mean Price", color='darkorange', linestyle='--')
    
    ax2.set_ylabel("Price ($/unit)", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title(f"Step 2: Orders and Prices (With Storage Constraint ≤18000, Lead Time = 0)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Inventory/Storage comparison
    date_index = pd.date_range(START_DATE, periods=len(inv_cost), freq="MS")
    
    plt.figure(figsize=(12, 6))
    plt.plot(date_index, inv_cost/50000, marker='o', linewidth=2, markersize=6, 
             color='tab:green', label='Without Storage Constraint')
    plt.plot(date_index, storage_opt/12000, marker='s', linewidth=2, markersize=6,
             color='tab:red', label='With Storage Constraint (≤18000)')
    plt.title("Inventory/Storage Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Storage (units/12K)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 3 & 4: Cost comparison by supplier
    plot_cost_comparison(
        sol_step2=sol_step2,
        price_df_step2=price_df_step2,
        order_step1_s1=order_supplier_1,
        order_step1_s2=order_supplier_2,
        price_s1=price_supplier_1,
        price_s2=price_supplier_2,
        start_date=START_DATE
    )
    
    # Plot 5: Cost breakdown comparison - grouped bars with percentage difference
    # Calculate costs for Step 1
    cost_unhedged_step1 = cost_supplier_1.sum()
    cost_hedged_step1 = cost_supplier_2.sum()
    cost_procurement_step1 = cost_unhedged_step1 + cost_hedged_step1
    cost_storage_step1 = storage_cost.sum()
    cost_backlog_step1 = 0  # No backlog in Step 1
    cost_total_step1 = cost_procurement_step1 + cost_storage_step1 + cost_backlog_step1
    
    # Calculate costs for Step 2
    cost_unhedged_step2 = cost_supplier_1_opt.sum()
    cost_hedged_step2 = cost_supplier_2_opt.sum()
    cost_procurement_step2 = cost_unhedged_step2 + cost_hedged_step2
    cost_storage_step2 = storage_cost_opt.sum()
    cost_backlog_step2 = 0  # No backlog allowed in Step 2
    cost_total_step2 = cost_procurement_step2 + cost_storage_step2 + cost_backlog_step2
    
    # Grouped bar chart with percentage differences
    categories = ['Unhedged\nSupplier', 'Hedged\nSupplier', 'Total\nProcurement', 
                  'Storage\nCost', 'Backlog\nCost', 'Grand\nTotal']
    
    values_without = np.array([cost_unhedged_step1, cost_hedged_step1, cost_procurement_step1,
                               cost_storage_step1, cost_backlog_step1, cost_total_step1])
    values_with = np.array([cost_unhedged_step2, cost_hedged_step2, cost_procurement_step2,
                            cost_storage_step2, cost_backlog_step2, cost_total_step2])
    
    # Calculate percentage differences
    pct_diff = ((values_with - values_without) / values_without) * 100
    
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, values_without, width, label='Without Storage Limit',
                   color='tab:blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, values_with, width, label='With Storage Limit (≤18000)',
                   color='tab:orange', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Cost Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost ($) - Log Scale', fontsize=13, fontweight='bold')
    ax.set_title('Cost Comparison: With vs Without Storage Constraint', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Use logarithmic scale for better visibility of small values
    ax.set_yscale('log')
    
    # Add value labels on bars - positioned carefully for log scale
    for bar, val in zip(bars1, values_without):
        height = bar.get_height()
        if val > 0:  # Only label non-zero values
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                    f'${val:,.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, values_with):
        height = bar.get_height()
        if val > 0:  # Only label non-zero values
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                    f'${val:,.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add percentage difference labels above both bars - moved higher
    for i, (bar1, bar2, diff) in enumerate(zip(bars1, bars2, pct_diff)):
        max_height = max(bar1.get_height(), bar2.get_height())
        if max_height > 0:  # Only add label if there's actual value
            color = 'green' if diff < 0 else 'red'
            sign = '' if diff < 0 else '+'
            # For log scale, move percentage label much higher
            label_height = max_height * 2.5
            ax.text(x[i], label_height,
                    f'{sign}{diff:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ All comparison plots displayed!")
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE")
    print("="*80)
    print(f"Generated files:")
    print(f"  1. {results_filename} (Step 1 results)")
    print(f"  2. {results_filename_step2} (Step 2 optimized results)")
    
except Exception as e:
    print(f"\n✗ Step 2 optimization failed: {str(e)}")
    print("  Step 1 results are still saved and available.")
    import traceback
    traceback.print_exc()
