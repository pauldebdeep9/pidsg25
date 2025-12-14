# storage_capacity_reschedule.py
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------------
# 1) Load & align input frames
# -----------------------------
df = pd.read_csv("capacity_opt2510.csv")

# Optional storage cap to draw (set to your actual cap)
s_max = 12000.0

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
    solver=cp.CLARABEL,
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
    for t in range(T-1):
        cons.append(Ppos[t] <= s_max)
  

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
    I0=1000.0, B0=0.0,
    force_arrival_within_horizon=True,
    solver=cp.ECOS, verbose=False
)

print("Objective:", sol["objective"])
print("Max per-period storage (should be <= s_max):", sol["max_storage"])
print("First 5 rows of Qhat:\n", sol["Qhat"].head())

# save orders to csv (as requested)
sol["Qhat"].to_csv("Qhat_orders.csv", index=False)



def deterministic_reschedule_unhedged_only_with_capacity(
    Q_star_df: pd.DataFrame,    # T x 2 with columns ["Hedged","Unhedged"] (AI plan)
    price_df: pd.DataFrame,     # T x 2 with same columns
    capacity_df: pd.DataFrame,  # T x 2 with same columns
    demand: np.ndarray,         # length T
    lead_time: dict,            # {'Hedged': Lh, 'Unhedged': Lu}
    s_max: float = 12000.0,     # max on-hand inventory per period
    h: float = 5.0,             # holding cost per unit/period
    b: float = 20.0,            # backlog penalty per unit/period
    I0: float = 0.0,            # initial inventory
    B0: float = 0.0,            # initial backlog
    force_arrival_within_horizon: bool = True,
    solver=cp.CLARABEL,
    verbose: bool = False,
):
    """
    Optimize only the 'Unhedged' schedule over time given prices, capacities, and lead times,
    while keeping 'Hedged' orders fixed to the provided AI schedule (Q_star_df['Hedged']).
    Enforces a hard per-period storage cap s_max on on-hand inventory.

    Returns:
        {
          'objective': float,
          'Qhat_unhedged': pd.Series (length T),
          'Q_fixed_hedged': pd.Series (length T, equal to input),
          'Theta_unhedged': pd.Series (arrivals),
          'Theta_hedged': pd.Series (arrivals from fixed hedged),
          'S': pd.Series (net stock),
          'P': pd.Series (on-hand),
          'N': pd.Series (backlog),
          'max_storage': float
        }
    """
    # ---------- shape checks & alignment ----------
    required_cols = ["Hedged", "Unhedged"]
    for df_ in (Q_star_df, price_df, capacity_df):
        if list(df_.columns) != required_cols:
            raise ValueError(f"Columns must be exactly {required_cols} in the same order.")
        if len(df_) != len(Q_star_df):
            raise ValueError("All input dataframes must have identical number of rows.")
        # numeric coercion
        for c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")

    demand = np.asarray(demand, dtype=float).reshape(-1)
    T = len(Q_star_df)
    if len(demand) != T:
        raise ValueError("demand length must equal the planning horizon T.")

    # constants (fixed Hedged schedule from AI plan)
    QH = Q_star_df["Hedged"].to_numpy(dtype=float)        # fixed orders (Hedged)
    QU_star = Q_star_df["Unhedged"].to_numpy(dtype=float) # AI plan total for Unhedged (for total-preservation)
    PU = price_df["Unhedged"].to_numpy(dtype=float)
    PH = price_df["Hedged"].to_numpy(dtype=float)
    CU = capacity_df["Unhedged"].to_numpy(dtype=float)

    Lh = int(lead_time["Hedged"])
    Lu = int(lead_time["Unhedged"])

    # ---------- lead-time shift matrices ----------
    def _arrival_matrix(T, L):
        A = np.zeros((T, T))
        for t in range(T):
            tp = t + L
            if 0 <= tp < T:
                A[tp, t] = 1.0
        return A

    A_H = _arrival_matrix(T, Lh)   # arrivals = A_H @ QH
    A_U = _arrival_matrix(T, Lu)   # arrivals = A_U @ QU

    # Soft-check: warn but do not fail. Arrivals beyond horizon are ignored by A_H @ QH anyway.
    if force_arrival_within_horizon:
        tail_idx = np.nonzero(np.arange(T) + Lh >= T)[0]
        if tail_idx.size and np.any(QH[tail_idx] > 1e-9) and verbose:
            total_tail = float(np.sum(QH[tail_idx]))
            print(f"[warn] Hedged has {total_tail:.3f} units ordered in the last {len(tail_idx)} "
                  f"periods that would arrive after T; ignoring in arrivals.")

    # Precompute fixed hedged arrivals (orders that arrive after T are naturally ignored)
    ThetaH = A_H @ QH  # length T

    # ---------- decision variables (Unhedged only) ----------
    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")      # placed Unhedged orders
    ThU = cp.Variable(T, nonneg=True, name="Theta_unhedged") # arrivals (Unhedged)

    # Inventory state variables
    S = cp.Variable(T, name="S")               # net (can be +/-)
    P = cp.Variable(T, nonneg=True, name="P")  # on-hand (storage)
    N = cp.Variable(T, nonneg=True, name="N")  # backlog (shortage)

    cons = []

    # Unhedged arrivals via lead-time shift
    cons += [ThU == A_U @ QU]

    # For Unhedged, forbid orders that would arrive outside horizon when flag is on
    if force_arrival_within_horizon:
        tail_idx_U = np.nonzero(np.arange(T) + Lu >= T)[0]
        if tail_idx_U.size:
            cons += [QU[tail_idx_U] == 0]

    # Inventory balance (include fixed Hedged arrivals)
    PI = float(I0 - B0)
    cons += [S[0] == PI + (ThetaH[0] + ThU[0]) - float(demand[0])]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (ThetaH[t] + ThU[t]) - float(demand[t])]

    # Storage decomposition and cap
    cons += [S == P - N]
    cons += [P <= s_max]  # hard cap per period

    # Per-period capacity on Unhedged
    cons += [QU <= CU]

    # Preserve total Unhedged volume (time-rescheduling only)
    cons += [cp.sum(QU) == float(np.sum(QU_star))]

    # ---------- objective ----------
    # Purchase cost (Hedged part is constant, included for clarity)
    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_backlog = h * cp.sum(P) + b * cp.sum(N)
    obj = cp.Minimize(purchase_cost + holding_backlog)

    # ---------- solve ----------
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Unhedged-only reschedule failed: status={prob.status}")

    # ---------- package ----------
    sol = {
        "objective": float(prob.value),
        "Qhat_unhedged": pd.Series(np.asarray(QU.value).reshape(-1), name="Unhedged"),
        "Q_fixed_hedged": pd.Series(QH, name="Hedged"),
        "Theta_unhedged": pd.Series(np.asarray(ThU.value).reshape(-1), name="Theta_Unhedged"),
        "Theta_hedged": pd.Series(ThetaH, name="Theta_Hedged"),
        "S": pd.Series(np.asarray(S.value).reshape(-1), name="S"),
        "P": pd.Series(np.asarray(P.value).reshape(-1), name="P"),
        "N": pd.Series(np.asarray(N.value).reshape(-1), name="N"),
        "max_storage": float(np.max(P.value)),
    }
    return sol


# assuming you already built the aligned frames like in your existing script
solU = deterministic_reschedule_unhedged_only_with_capacity(
    Q_star_df=Q_star_df, price_df=price_df, capacity_df=capacity_df,
    demand=demand, lead_time=lead_time,
    s_max=12000, h=5.0, b=50.0, I0=1000.0, B0=0.0,
    force_arrival_within_horizon=True, solver=cp.ECOS, verbose=False
)
print("Objective:", solU["objective"])
print("Max per-period storage (<=12000):", solU["max_storage"])
# Save only the optimized Unhedged schedule if you like:


# Show stock per period
# print("\nStock (on-hand P):")
# print(solU["P"].to_string(index=True))

# # Optional: also show net stock S (P - N)
# print("\nNet stock (S = P - N):")
# print(solU["S"].to_string(index=True))

def deterministic_reschedule_unhedged_L0(
    Q_star_df: pd.DataFrame,    # T x 2 with columns ["Hedged","Unhedged"] (AI plan)
    price_df: pd.DataFrame,     # T x 2 with same columns
    capacity_df: pd.DataFrame,  # T x 2 with same columns
    demand: np.ndarray,         # length T
    s_max: float = 12000.0,     # per-period on-hand storage cap
    h: float = 5.0,             # holding cost per unit per period
    I0: float = 0.0,            # initial on-hand inventory
    no_backlog: bool = True,    # enforce no backorders (recommended for L=0)
    solver=cp.CLARABEL,
    verbose: bool = False,
):
    """
    Lead time = 0 model:
      - Hedged schedule is fixed to Q_star_df['Hedged'].
      - Optimize only Unhedged (one decision variable series).
      - Objective: minimize procurement cost (Unhedged) + holding cost (storage).
      - Enforce per-period storage cap s_max.
      - Default: no backlog allowed (P[t] >= 0 and fully meets demand each period cumulatively).

    Returns dict with objective, optimized Unhedged, stock P, etc.
    """
    # ---------- checks ----------
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

    # constants
    QH = Q_star_df["Hedged"].to_numpy(float)   # fixed hedged orders
    PU = price_df["Unhedged"].to_numpy(float)  # price for unhedged
    PH = price_df["Hedged"].to_numpy(float)    # price for hedged (constant part)
    CU = capacity_df["Unhedged"].to_numpy(float)  # capacity for unhedged

    # ---------- variables ----------
    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")  # decision: unhedged orders
    S  = cp.Variable(T, name="S")                        # net inventory
    P  = cp.Variable(T, nonneg=True, name="P")           # on-hand
    N  = cp.Variable(T, nonneg=True, name="N")           # backlog (used if no_backlog=False)

    cons = []

    # Inventory dynamics with L=0: orders impact immediately
    # S[t] = S[t-1] + (QH[t] + QU[t]) - d[t], with S[0] anchored by I0.
    cons += [S[0] == float(I0) + (QH[0] + QU[0]) - d[0]]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (QH[t] + QU[t]) - d[t]]

    if no_backlog:
        # No backorders: S[t] == P[t] >= 0, and N[t] forced to 0
        cons += [S == P]
        cons += [P >= 0]
        cons += [N == 0]
    else:
        # Allow backlog but keep S = P - N
        cons += [S == P - N]

    # Storage cap
    cons += [P <= s_max]

    # Per-period capacity on Unhedged
    cons += [QU <= CU]

    # (Optional) If you want to preserve total Unhedged volume, uncomment:
    # cons += [cp.sum(QU) == float(Q_star_df["Unhedged"].sum())]

    # ---------- objective ----------
    # Procurement (Hedged part is constant; included for completeness)
    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_cost  = h * cp.sum(P)
    obj = cp.Minimize(purchase_cost + holding_cost)

    # ---------- solve ----------
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"L0 unhedged reschedule failed: status={prob.status}")

    # ---------- package ----------
    # With L=0, "arrivals" are just the orders themselves.
    ThetaH = QH.copy()
    ThetaU = np.asarray(QU.value).reshape(-1)

    out = {
        "objective": float(prob.value),
        "Qhat_unhedged": pd.Series(np.asarray(QU.value).reshape(-1), name="Unhedged"),
        "Q_fixed_hedged": pd.Series(QH, name="Hedged"),
        "Theta_unhedged": pd.Series(ThetaU, name="Theta_Unhedged"),
        "Theta_hedged": pd.Series(ThetaH, name="Theta_Hedged"),
        "S": pd.Series(np.asarray(S.value).reshape(-1), name="S"),
        "P": pd.Series(np.asarray(P.value).reshape(-1), name="P"),
        "N": pd.Series(np.asarray(N.value).reshape(-1), name="N"),
        "max_storage": float(np.max(P.value)),
        "purchase_cost": float(np.sum(PU * np.asarray(QU.value).reshape(-1)) + np.dot(PH, QH)),
        "holding_cost": float(h * np.sum(np.asarray(P.value).reshape(-1))),
    }
    return out

sol0 = deterministic_reschedule_unhedged_L0(
    Q_star_df=Q_star_df,
    price_df=price_df,
    capacity_df=capacity_df,
    demand=demand,
    s_max=12000,
    h=5.0,
    I0=1000.0,
    no_backlog=True,      # ensures demand is met cumulatively with zero backorders
    solver=cp.CLARABEL,
    verbose=False
)
print("Objective:", sol0["objective"])
print("Max storage:", sol0["max_storage"])
print(sol0["P"].to_string(index=True))  # Stock per period (capped)
# Combine fixed Hedged + optimized Unhedged into one DataFrame
orders_df = pd.concat(
    [sol0["Q_fixed_hedged"], sol0["Qhat_unhedged"]],
    axis=1
)
# orders_df.to_csv("optimized_orders.csv", index=False)

def deterministic_reschedule_unhedged_L0_lock_prefix(
    Q_star_df: pd.DataFrame,    # T x 2 with columns ["Hedged","Unhedged"] (AI plan)
    price_df: pd.DataFrame,     # T x 2 with same columns
    capacity_df: pd.DataFrame,  # T x 2 with same columns
    demand: np.ndarray,         # length T
    k: int = 4,                 # lock the first k months of Unhedged to the original plan
    s_max: float = 12000.0,     # per-period on-hand storage cap
    h: float = 5.0,             # holding cost per unit per period
    I0: float = 0.0,            # initial on-hand inventory
    no_backlog: bool = True,    # enforce no backorders (recommended for L=0)
    solver=cp.CLARABEL,
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

    Returns dict with objective, optimized Unhedged, stock P, etc.
    """
    # ---------- checks ----------
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

    # constants
    QH = Q_star_df["Hedged"].to_numpy(float)       # fixed hedged orders
    QU_fix_prefix = Q_star_df["Unhedged"].to_numpy(float)  # used to lock first k periods
    PU = price_df["Unhedged"].to_numpy(float)      # price for unhedged
    PH = price_df["Hedged"].to_numpy(float)        # price for hedged (constant part)
    CU = capacity_df["Unhedged"].to_numpy(float)   # capacity for unhedged

    # ---------- variables ----------
    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")  # decision: unhedged orders
    S  = cp.Variable(T, name="S")                        # net inventory
    P  = cp.Variable(T, nonneg=True, name="P")           # on-hand
    N  = cp.Variable(T, nonneg=True, name="N")           # backlog (used if no_backlog=False)

    cons = []

    # ---------- prefix lock for Unhedged ----------
    if k > 0:
        # exactly match original Unhedged for first k periods
        cons += [QU[:k] == QU_fix_prefix[:k]]
        # sanity: ensure locked values do not exceed capacity in those periods
        # (if they do, the problem becomes infeasible)
        cons += [QU[:k] <= CU[:k]]
    # remaining periods are free (subject to capacity)
    if k < T:
        cons += [QU[k:] <= CU[k:]]

    # ---------- inventory dynamics (L=0): orders impact immediately ----------
    cons += [S[0] == float(I0) + (QH[0] + QU[0]) - d[0]]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (QH[t] + QU[t]) - d[t]]

    if no_backlog:
        cons += [S == P]     # S=P, N=0
        cons += [N == 0]
        cons += [P >= 0]
    else:
        cons += [S == P - N]

    # ---------- storage cap ----------
    cons += [P <= s_max]

    # ---------- objective ----------
    # Hedged purchase is constant; Unhedged is variable
    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_cost  = h * cp.sum(P)
    obj = cp.Minimize(purchase_cost + holding_cost)

    # ---------- solve ----------
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"L0 unhedged reschedule (prefix lock) failed: status={prob.status}")

    # ---------- package ----------
    ThetaH = QH.copy()  # L=0 → arrivals == orders
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
      - Hedged purchase cost  = sum_t price_H[t] * hedged_orders[t]
      - Unhedged purchase cost = sum_t price_U[t] * unhedged_orders[t]
      - Storage (holding) cost = h * sum_t P[t]
      - (optional) Backlog cost = b * sum_t N[t]   if b is provided and N exists

    Works with:
      - sol from deterministic_reschedule_with_capacity (has sol["Qhat"] with both cols)
      - sol0 / solU (have sol["Q_fixed_hedged"], sol["Qhat_unhedged"])
    """
    # Pull orders depending on which solver produced 'sol'
    if "Qhat" in sol:  # full matrix (both suppliers)
        hedged_orders   = sol["Qhat"]["Hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat"]["Unhedged"].to_numpy(float)
    else:  # unhedged-only variants
        hedged_orders   = sol["Q_fixed_hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat_unhedged"].to_numpy(float)

    # Prices
    pH = price_df["Hedged"].to_numpy(float)
    pU = price_df["Unhedged"].to_numpy(float)

    # Storage (on-hand P)
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

    # Optional backlog cost if requested and available
    if b is not None and "N" in sol:
        N = sol["N"].to_numpy(float)
        backlog_cost = float(b * np.sum(N))
        out["backlog_cost"] = backlog_cost
        out["total_with_backlog"] = out["total_without_backlog"] + backlog_cost

    return out


# Choose your solution dict here (sol0 for L=0 model, or solU for the lead-time model)
  # or: sol = solU

# Existing solution (all months free)
sol0 = deterministic_reschedule_unhedged_L0(
    Q_star_df=Q_star_df,
    price_df=price_df,
    capacity_df=capacity_df,
    demand=demand,
    s_max=12000,
    h=5.0,
    I0=1000.0,
    no_backlog=True,
    solver=cp.CLARABEL,
    verbose=False
)

# New solution: first k months of Unhedged are fixed to the original plan
sol0_lock = deterministic_reschedule_unhedged_L0_lock_prefix(
    Q_star_df=Q_star_df,
    price_df=price_df,
    capacity_df=capacity_df,
    demand=demand,
    k=4,                   # change to compare different lock lengths
    s_max=12000,
    h=5.0,
    I0=1000.0,
    no_backlog=True,
    solver=cp.CLARABEL,
    verbose=False
)

print("Objective (all free):", sol0["objective"])
print("Objective (first 4 locked):", sol0_lock["objective"])

# Save to compare
pd.concat([sol0["Q_fixed_hedged"], sol0["Qhat_unhedged"]], axis=1)\
  .to_csv("optimized_orders_all_free.csv", index=False)
pd.concat([sol0_lock["Q_fixed_hedged"], sol0_lock["Qhat_unhedged"]], axis=1)\
  .to_csv("optimized_orders_first_k_locked.csv", index=False)

sol = sol0_lock
# Extract series (they are pandas.Series)
hedged   = sol["Q_fixed_hedged"]      # Hedged orders (fixed)
unhedged = sol["Qhat_unhedged"]       # Unhedged orders (optimized)
stock    = sol["P"]                   # On-hand stock
T = len(stock)
t = np.arange(T)



def plot_procurement_figs(sol, price_df, s_max=12000.0, prefix="fig"):
    """
    Makes:
      i)  fig-1  : Orders (side-by-side bars Hedged/Unhedged) + prices (dotted, secondary axis)
      ii) fig-22 : Storage (on-hand P) vs cap
      iii) fig-3 : Cumulative procurement cost (Hedged vs Unhedged)
      iv)  fig-4 : Storage (P and Net S)
    """
  
    # Extract series from solution dict
    QH = sol["Q_fixed_hedged"].to_numpy(float)
    QU = sol["Qhat_unhedged"].to_numpy(float)
    P  = sol["P"].to_numpy(float)
    S  = sol["S"].to_numpy(float)
    T  = len(QH)
    t  = np.arange(T)

    # Prices (period-wise)
    PH = price_df["Hedged"].to_numpy(float)
    PU = price_df["Unhedged"].to_numpy(float)

    # --------- NEW: x-axis date labels Apr-25 ... Mar-26 ----------
    start = pd.Timestamp("2025-04-01")         # Apr-25
    months = pd.date_range(start, periods=T, freq="MS")
    xlabels = months.strftime("%b-%y")         # e.g., Apr-25, May-25, ...
    # ---------------------------------------------------------------

    # =========================
    # i) Orders bars side-by-side + prices dotted
    # =========================
    width = 0.35
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(t - width/2, QH, width=width, label="Hedged Order")
    ax1.bar(t + width/2, QU, width=width, label="Unhedged Order")
    ax1.set_ylabel("Orders")
    ax1.grid(True, alpha=0.3)

    # Price lines on secondary axis
    axp = ax1.twinx()
    axp.plot(t, PH, linestyle=":", linewidth=2, label="Hedged Price")
    axp.plot(t, PU, linestyle=":", linewidth=2, label="Unhedged Price")
    axp.set_ylabel("Price")

    # X ticks as months
    ax1.set_xticks(t)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")

    # combine legends
    lines, labels = [], []
    for a in (ax1, axp):
        L = a.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax1.legend(lines, labels, loc="best")

    plt.title("Orders from Suppliers (side-by-side bars) + Price (dotted, secondary axis)")
    plt.tight_layout()
    # plt.savefig(f"{prefix}-1_orders_prices.png", dpi=150)
    plt.show()

    # =========================
    # ii) Storage (on-hand P)
    # =========================
   # =========================
    # ii) Storage (on-hand P)  — BAR
    # =========================
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(t, P, color="tab:blue", alpha=0.7, label="Stock (on-hand P)")
    ax2.axhline(s_max, linestyle="--", linewidth=1.5, color="red", label=f"Cap = {s_max:g}")
    ax2.set_ylabel("Units")
    ax2.grid(True, alpha=0.3); ax2.legend()

    ax2.set_xticks(t)
    ax2.set_xticklabels(xlabels, rotation=45, ha="right")

    plt.title("Storage (On-hand P) vs Cap")
    plt.tight_layout()
    # plt.savefig(f"{prefix}-22_storage.png", dpi=150)
    plt.show()
    # ========================================
    # iii) Cumulative cost (Hedged, Unhedged)
    # ========================================
    cost_H = PH * QH
    cost_U = PU * QU
    cum_H = np.cumsum(cost_H)
    cum_U = np.cumsum(cost_U)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t, cum_H, linewidth=2, label="Cumulative Hedged Cost")
    ax3.plot(t, cum_U, linewidth=2, label="Cumulative Unhedged Cost")
    ax3.set_ylabel("Cost")
    ax3.grid(True, alpha=0.3); ax3.legend()

    ax3.set_xticks(t)
    ax3.set_xticklabels(xlabels, rotation=45, ha="right")

    plt.title("Cumulative Procurement Cost (Hedged vs Unhedged)")
    plt.tight_layout()
    # plt.savefig(f"{prefix}-3_cumulative_cost.png", dpi=150)
    plt.show()

    # =========================
    # iv) Storage (alt view)
    # =========================
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t, P, linewidth=2, label="On-hand P")
    ax4.plot(t, S, linewidth=1.8, label="Net S (=P-N)")
    ax4.axhline(s_max, linestyle="--", linewidth=1.5, label=f"Cap = {s_max:g}")
    ax4.set_ylabel("Units")
    ax4.grid(True, alpha=0.3); ax4.legend()

    ax4.set_xticks(t)
    ax4.set_xticklabels(xlabels, rotation=45, ha="right")

    plt.title("Storage (On-hand P) and Net Inventory S")
    plt.tight_layout()
    # plt.savefig(f"{prefix}-4_storage_alt.png", dpi=150)
    plt.show()

    print("Saved:")
    print(f"  {prefix}-1_orders_prices.png")
    print(f"  {prefix}-22_storage.png")
    print(f"  {prefix}-3_cumulative_cost.png")
    print(f"  {prefix}-4_storage_alt.png")

# EXAMPLE USAGE:
# plot_procurement_figs(sol0, price_df, s_max=12000.0, prefix="fig")

if __name__ == "__main__":
    plot_procurement_figs(sol, price_df, s_max=12000.0, prefix="fig")
    # For the lead-time-0 solution
    costs0 = compute_cost_breakdown(sol0, price_df, h=5.0, b=None)
    print(costs0)

    # For the unhedged-only (lead-time) solution with backlog penalty b=50
    costsU = compute_cost_breakdown(solU, price_df, h=5.0, b=50.0)
    print(costsU)

    # For the full matrix solution 'sol' (both suppliers optimized)
    costs_full = compute_cost_breakdown(sol, price_df, h=5.0, b=20.0)
    print(costs_full)
