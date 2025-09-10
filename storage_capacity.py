# storage_capacity_reschedule.py
import cvxpy as cp
import numpy as np
import pandas as pd

# -----------------------------
# 1) Load & align input frames
# -----------------------------
df = pd.read_csv("capacity_opt2509.csv")

# canonical supplier labels
SUPPLIERS = ["Hedged", "Unhedged"]

# align columns across all inputs
Q_star_df = (df[["Order-AI-Hedged", "Order-AI-Unhedged"]]
             .rename(columns={"Order-AI-Hedged": "Hedged",
                              "Order-AI-Unhedged": "Unhedged"}))[SUPPLIERS]

price_df = (df[["Hedged price", "Unhedged price"]]
            .rename(columns={"Hedged price": "Hedged",
                             "Unhedged price": "Unhedged"}))[SUPPLIERS]

capacity_df = (df[["Capacity-Hedged", "Capacity-Unhedged"]]
               .rename(columns={"Capacity-Hedged": "Hedged",
                                "Capacity-Unhedged": "Unhedged"}))[SUPPLIERS]

# demand vector
demand = df["Demand"].to_numpy(dtype=float)

# lead times (integers)
lead_time = {
    "Hedged": int(df["Lead-Time-Hedged"].iloc[0]),
    "Unhedged": int(df["Lead-Time-Unhedged"].iloc[0]),
}

# numeric cleanup
for _x in (Q_star_df, price_df, capacity_df):
    for c in _x.columns:
        _x[c] = pd.to_numeric(_x[c], errors="coerce")

# -----------------------------
# 2) Deterministic reschedule
# -----------------------------
def deterministic_reschedule_with_capacity(
    Q_star_df: pd.DataFrame,    # T x S (exogenous totals per supplier from McC)
    price_df: pd.DataFrame,     # T x S
    capacity_df: pd.DataFrame,  # T x S
    demand: np.ndarray,         # length T
    lead_time: dict,            # {'Hedged': Lh, 'Unhedged': Lu, ...}
    s_max: float,               # max on-hand inventory per period
    h: float, b: float,         # holding & backlog unit costs
    I0: float, B0: float,       # initial inventory & backlog
    force_arrival_within_horizon: bool = True,
    solver=cp.ECOS,
    verbose: bool = False,
):
    # shape checks
    for df_ in (price_df, capacity_df):
        if list(df_.columns) != list(Q_star_df.columns) or len(df_) != len(Q_star_df):
            raise ValueError("Q_star_df, price_df, capacity_df must have identical shape/columns.")
    suppliers = list(Q_star_df.columns)
    T, S = Q_star_df.shape
    demand = np.asarray(demand, dtype=float).reshape(-1)
    if len(demand) != T:
        raise ValueError("demand length must equal number of rows (T).")

    # totals per supplier (preserve these)
    Q_tot = Q_star_df.sum(axis=0).to_dict()

    # numpy payloads
    Pmat = price_df.to_numpy(dtype=float)     # T x S
    Cmat = capacity_df.to_numpy(dtype=float)  # T x S

    # -------------------- variables --------------------
    Qhat  = cp.Variable((T, S), nonneg=True, name="Qhat")   # placed orders
    Theta = cp.Variable((T, S), nonneg=True, name="Theta")  # arrivals (shifted by lead time)

    # net stock S, with positive P (on-hand) and negative N (backlog)
    Svar = cp.Variable(T, name="S")                 # can be +/- (net)
    Ppos = cp.Variable(T, nonneg=True, name="P")    # on-hand (storage)
    Nneg = cp.Variable(T, nonneg=True, name="N")    # backlog

    cons = []

    # -------------------- arrivals via lead-time --------------------
    for j, s in enumerate(suppliers):
        Ls = int(lead_time[s])
        A = np.zeros((T, T))
        for t in range(T):
            tp = t + Ls
            if 0 <= tp < T:
                A[tp, t] = 1.0
        cons.append(Theta[:, j] == A @ Qhat[:, j])

        if force_arrival_within_horizon:
            for t in range(T):
                if t + Ls >= T:
                    cons.append(Qhat[t, j] == 0)

    # -------------------- inventory balance --------------------
    PI = float(I0 - B0)  # initial net position
    cons.append(Svar[0] == PI + cp.sum(Theta[0, :]) - float(demand[0]))
    for t in range(1, T):
        cons.append(Svar[t] == Svar[t-1] + cp.sum(Theta[t, :]) - float(demand[t]))

    # -------------------- decomposition & storage cap --------------------
    cons += [Svar == Ppos - Nneg]
    cons += [Ppos <= s_max]  # hard per-period bound on storage

    # -------------------- per-period supplier capacity --------------------
    cons.append(Qhat <= Cmat)  # elementwise

    # -------------------- preserve per-supplier totals --------------------
    for j, s in enumerate(suppliers):
        cons.append(cp.sum(Qhat[:, j]) == float(Q_tot[s]))

    # -------------------- objective --------------------
    purchase_cost = cp.sum(cp.multiply(Pmat, Qhat))  # sum_t,s price[t,s] * Qhat[t,s]
    holding_backlog = h * cp.sum(Ppos) + b * cp.sum(Nneg)
    obj = cp.Minimize(purchase_cost + holding_backlog)

    # -------------------- solve --------------------
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Reschedule failed: status={prob.status}")

    # -------------------- package outputs --------------------
    sol = {
        "objective": float(prob.value),
        "Qhat":      pd.DataFrame(Qhat.value,  columns=suppliers),
        "Theta":     pd.DataFrame(Theta.value, columns=suppliers),
        "S":         pd.Series(Svar.value, name="S"),
        "P":         pd.Series(Ppos.value, name="P"),
        "N":         pd.Series(Nneg.value, name="N"),
        "max_storage": float(np.max(Ppos.value)),
    }
    return sol

# -----------------------------
# 3) Run + save results
# -----------------------------
sol = deterministic_reschedule_with_capacity(
    Q_star_df=Q_star_df,
    price_df=price_df,
    capacity_df=capacity_df,
    demand=demand,
    lead_time=lead_time,
    s_max= 12000,      # <-- set your storage cap here
    h=5.0, b=20.0,
    I0=12000.0, B0=0.0,
    force_arrival_within_horizon=True,
    solver=cp.ECOS, verbose=False
)

print("Objective:", sol["objective"])
print("Max per-period storage (should be <= s_max):", sol["max_storage"])
print("First 5 rows of Qhat:\n", sol["Qhat"].head())

# save orders to csv (as requested)
sol["Qhat"].to_csv("Qhat_orders.csv", index=False)
