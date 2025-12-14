# storage_capacity_reschedule_with_bounds_autofallback.py
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load & align input frames
# -----------------------------
df = pd.read_csv("capacity_opt2510_dum.csv")

# Storage bounds
s_max = 12000.0   # upper bound on on-hand inventory per period
s_min = 1000    # lower bound (safety stock). Can be scalar or length-T vector.

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

# -------------- helpers --------------
def _as_vec(x, T):
    """Return a length-T numpy vector from scalar or sequence."""
    if np.isscalar(x):
        return np.full(T, float(x))
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != T:
        raise ValueError("Expected s_min length == T.")
    return arr


# -----------------------------
# 2) Deterministic reschedule (both suppliers) with bounds
# -----------------------------
def deterministic_reschedule_with_capacity(
    Q_star_df: pd.DataFrame,    # T x S (exogenous totals per supplier from AI)
    price_df: pd.DataFrame,     # T x S
    capacity_df: pd.DataFrame,  # T x S
    demand: np.ndarray,         # length T
    lead_time: dict,            # {'Hedged': Lh, 'Unhedged': Lu, ...}
    s_max: float,               # max on-hand inventory per period
    h: float, b: float,         # holding & backlog unit costs
    I0: float, B0: float,       # initial inventory & backlog
    s_min=0.0,                  # lower bound on on-hand (scalar or length-T)
    enforce_lb: bool = True,    # hard lower bound if True; soft if False
    lb_penalty: float = 1e4,    # penalty on violations if soft
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

    smin_vec = _as_vec(s_min, T)

    # -------------------- variables --------------------
    Qhat  = cp.Variable((T, S), nonneg=True, name="Qhat")   # placed orders
    Theta = cp.Variable((T, S), nonneg=True, name="Theta")  # arrivals (shifted by lead time)

    # net stock S, with positive P (on-hand) and negative N (backlog)
    Svar = cp.Variable(T, name="S")                 # can be +/- (net)
    Ppos = cp.Variable(T, nonneg=True, name="P")    # on-hand (storage)
    Nneg = cp.Variable(T, nonneg=True, name="N")    # backlog

    # Optional slack for soft lower bound
    Ylb  = cp.Variable(T, nonneg=True, name="Y_lb") if not enforce_lb else None

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

    # -------------------- decomposition & storage bounds --------------------
    cons += [Svar == Ppos - Nneg]
    cons += [Ppos <= s_max]  # upper bound
    if enforce_lb:
        cons += [Ppos >= smin_vec]  # hard lower bound
    else:
        cons += [Ppos + Ylb >= smin_vec]  # soft LB with slack

    # -------------------- per-period supplier capacity --------------------
    cons.append(Qhat <= Cmat)  # elementwise

    # -------------------- preserve per-supplier totals --------------------
    for j, s in enumerate(suppliers):
        cons.append(cp.sum(Qhat[:, j]) == float(Q_tot[s]))

    # -------------------- objective --------------------
    purchase_cost = cp.sum(cp.multiply(Pmat, Qhat))  # sum_t,s price[t,s] * Qhat[t,s]
    holding_backlog = h * cp.sum(Ppos) + b * cp.sum(Nneg)
    penalty = 0
    if not enforce_lb:
        penalty = lb_penalty * cp.sum(Ylb)
    obj = cp.Minimize(purchase_cost + holding_backlog + penalty)

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
        "min_storage": float(np.min(Ppos.value)),
    }
    if not enforce_lb:
        sol["lb_slack_sum"] = float(np.sum(Ylb.value))
        sol["lb_slack"] = pd.Series(Ylb.value, name="Y_lb")
    return sol


# -----------------------------
# 2b) Unhedged-only with bounds (lead time ≥ 0)
# -----------------------------
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
    s_min: float | np.ndarray = 0.0,   # lower bound
    enforce_lb: bool = True,           # hard lower bound if True; soft if False
    lb_penalty: float = 1e4,           # penalty if soft
    force_arrival_within_horizon: bool = True,
    solver=cp.CLARABEL,
    verbose: bool = False,
):
    required_cols = ["Hedged", "Unhedged"]
    for df_ in (Q_star_df, price_df, capacity_df):
        if list(df_.columns) != required_cols:
            raise ValueError(f"Columns must be exactly {required_cols} in the same order.")
        if len(df_) != len(Q_star_df):
            raise ValueError("All input dataframes must have identical number of rows.")
        for c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")

    demand = np.asarray(demand, dtype=float).reshape(-1)
    T = len(Q_star_df)
    if len(demand) != T:
        raise ValueError("demand length must equal the planning horizon T.")

    smin_vec = _as_vec(s_min, T)

    # constants (fixed Hedged schedule)
    QH = Q_star_df["Hedged"].to_numpy(dtype=float)
    QU_star = Q_star_df["Unhedged"].to_numpy(dtype=float)
    PU = price_df["Unhedged"].to_numpy(dtype=float)
    PH = price_df["Hedged"].to_numpy(dtype=float)
    CU = capacity_df["Unhedged"].to_numpy(dtype=float)

    Lh = int(lead_time["Hedged"])
    Lu = int(lead_time["Unhedged"])

    def _arrival_matrix(T, L):
        A = np.zeros((T, T))
        for t in range(T):
            tp = t + L
            if 0 <= tp < T:
                A[tp, t] = 1.0
        return A

    A_H = _arrival_matrix(T, Lh)
    A_U = _arrival_matrix(T, Lu)

    ThetaH = A_H @ QH  # fixed hedged arrivals

    # decision variables (Unhedged only)
    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")
    ThU = cp.Variable(T, nonneg=True, name="Theta_unhedged")

    S = cp.Variable(T, name="S")
    P = cp.Variable(T, nonneg=True, name="P")
    N = cp.Variable(T, nonneg=True, name="N")

    Ylb = cp.Variable(T, nonneg=True, name="Y_lb") if not enforce_lb else None

    cons = []
    cons += [ThU == A_U @ QU]

    if force_arrival_within_horizon:
        tail_idx_U = np.nonzero(np.arange(T) + Lu >= T)[0]
        if tail_idx_U.size:
            cons += [QU[tail_idx_U] == 0]

    PI = float(I0 - B0)
    cons += [S[0] == PI + (ThetaH[0] + ThU[0]) - float(demand[0])]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (ThetaH[t] + ThU[t]) - float(demand[t])]

    cons += [S == P - N]
    cons += [P <= s_max]
    if enforce_lb:
        cons += [P >= smin_vec]
    else:
        cons += [P + Ylb >= smin_vec]

    cons += [QU <= CU]
    cons += [cp.sum(QU) == float(np.sum(QU_star))]

    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_backlog = h * cp.sum(P) + b * cp.sum(N)
    penalty = 0
    if not enforce_lb:
        penalty = lb_penalty * cp.sum(Ylb)
    obj = cp.Minimize(purchase_cost + holding_backlog + penalty)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Unhedged-only reschedule failed: status={prob.status}")

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
        "min_storage": float(np.min(P.value)),
    }
    if not enforce_lb:
        sol["lb_slack_sum"] = float(np.sum(Ylb.value))
        sol["lb_slack"] = pd.Series(Ylb.value, name="Y_lb")
    return sol


# -----------------------------
# 2c) Lead-time 0 (free) with bounds
# -----------------------------
def deterministic_reschedule_unhedged_L0(
    Q_star_df: pd.DataFrame,
    price_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    demand: np.ndarray,
    s_max: float = 12000.0,
    h: float = 5.0,
    I0: float = 0.0,
    no_backlog: bool = True,
    s_min: float | np.ndarray = 0.0,
    enforce_lb: bool = True,
    lb_penalty: float = 1e4,
    solver=cp.CLARABEL,
    verbose: bool = False,
):
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

    smin_vec = _as_vec(s_min, T)

    QH = Q_star_df["Hedged"].to_numpy(float)
    PU = price_df["Unhedged"].to_numpy(float)
    PH = price_df["Hedged"].to_numpy(float)
    CU = capacity_df["Unhedged"].to_numpy(float)

    QU = cp.Variable(T, nonneg=True, name="Q_unhedged")
    S  = cp.Variable(T, name="S")
    P  = cp.Variable(T, nonneg=True, name="P")
    N  = cp.Variable(T, nonneg=True, name="N")

    Ylb = cp.Variable(T, nonneg=True, name="Y_lb") if not enforce_lb else None

    cons = []
    cons += [S[0] == float(I0) + (QH[0] + QU[0]) - d[0]]
    for t in range(1, T):
        cons += [S[t] == S[t-1] + (QH[t] + QU[t]) - d[t]]

    if no_backlog:
        cons += [S == P]
        cons += [P >= 0]
        cons += [N == 0]
    else:
        cons += [S == P - N]

    cons += [P <= s_max]
    if enforce_lb:
        cons += [P >= smin_vec]
    else:
        cons += [P + Ylb >= smin_vec]

    cons += [QU <= CU]

    purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
    holding_cost  = h * cp.sum(P)
    penalty = 0
    if not enforce_lb:
        penalty = lb_penalty * cp.sum(Ylb)
    obj = cp.Minimize(purchase_cost + holding_cost + penalty)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"L0 unhedged reschedule failed: status={prob.status}")

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
        "min_storage": float(np.min(P.value)),
        "purchase_cost": float(np.sum(PU * ThetaU) + np.dot(PH, QH)),
        "holding_cost": float(h * np.sum(np.asarray(P.value).reshape(-1))),
    }
    if not enforce_lb:
        out["lb_slack_sum"] = float(np.sum(Ylb.value))
        out["lb_slack"] = pd.Series(Ylb.value, name="Y_lb")
    return out


# -----------------------------
# 2d) Lead-time 0 with prefix lock (auto-fallback hard→soft LB)
# -----------------------------
def deterministic_reschedule_unhedged_L0_lock_prefix(
    Q_star_df: pd.DataFrame,
    price_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    demand: np.ndarray,
    k: int = 4,
    s_max: float = 12000.0,
    h: float = 5.0,
    I0: float = 0.0,
    no_backlog: bool = True,
    s_min: float | np.ndarray = 0.0,   # safety stock (scalar or length-T)
    solver=cp.CLARABEL,
    verbose: bool = False,

    # Auto behavior:
    auto_soft_lb_if_infeasible: bool = True,   # try hard LB first, then soft LB if infeasible
    soft_lb_penalty: float = 1e5,              # penalty λ · sum(Y) for soft LB
    auto_relax_lb_prefix: bool = True          # optionally shrink LB on locked months to feasible envelope
):
    """L=0, lock first k months of Unhedged; auto-fallback to soft LB if hard LB is infeasible."""

    # ---------- basic checks ----------
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

    # Data vectors
    QH = Q_star_df["Hedged"].to_numpy(float)
    QU_fix_prefix = Q_star_df["Unhedged"].to_numpy(float)
    PU = price_df["Unhedged"].to_numpy(float)
    PH = price_df["Hedged"].to_numpy(float)
    CU = capacity_df["Unhedged"].to_numpy(float)

    # s_min as vector
    smin_vec = _as_vec(s_min, T)

    # Feasible envelope for locked prefix when no_backlog=True
    def _feas_env_prefix():
        return I0 + np.cumsum(QH[:k] + QU_fix_prefix[:k] - d[:k]) if k > 0 else np.array([])

    # Optionally shrink LB on locked months to maintain feasibility envelope
    if no_backlog and k > 0 and auto_relax_lb_prefix:
        feas_env = _feas_env_prefix()  # length k
        if feas_env.size:
            too_high = smin_vec[:k] > feas_env + 1e-9
            if np.any(too_high) and verbose:
                idx = np.where(too_high)[0].tolist()
                print(f"[info] Shrinking s_min on locked prefix months {idx} to feasible envelope.")
            smin_vec[:k] = np.minimum(smin_vec[:k], feas_env)

    # Problem builder so we can attempt hard → soft
    def _build_problem(use_soft_lb: bool):
        QU = cp.Variable(T, nonneg=True, name="Q_unhedged")
        S  = cp.Variable(T, name="S")
        P  = cp.Variable(T, nonneg=True, name="P")
        N  = cp.Variable(T, nonneg=True, name="N")
        Y  = cp.Variable(T, nonneg=True, name="Y_lb") if use_soft_lb else None

        cons = []
        # prefix lock & capacity
        if k > 0:
            cons += [QU[:k] == QU_fix_prefix[:k]]
            cons += [QU[:k] <= CU[:k]]
        if k < T:
            cons += [QU[k:] <= CU[k:]]

        # L=0 dynamics
        cons += [S[0] == float(I0) + (QH[0] + QU[0]) - d[0]]
        for t in range(1, T):
            cons += [S[t] == S[t-1] + (QH[t] + QU[t]) - d[t]]

        if no_backlog:
            cons += [S == P]   # no shortages
            cons += [P >= 0]
            cons += [N == 0]
        else:
            cons += [S == P - N]

        # storage band
        cons += [P <= s_max]
        if use_soft_lb:
            cons += [P + Y >= smin_vec]
        else:
            cons += [P >= smin_vec]

        # objective
        purchase_cost = cp.sum(cp.multiply(PU, QU)) + float(np.dot(PH, QH))
        holding_cost  = h * cp.sum(P)
        penalty       = soft_lb_penalty * cp.sum(Y) if use_soft_lb else 0
        obj = cp.Minimize(purchase_cost + holding_cost + penalty)

        prob = cp.Problem(obj, cons)
        return prob, (QU, S, P, N, Y)

    # Try HARD LB first
    prob_hard, vars_hard = _build_problem(use_soft_lb=False)
    prob_hard.solve(solver=solver, verbose=verbose)
    status_hard = prob_hard.status

    if status_hard in ("optimal", "optimal_inaccurate"):
        QU, S, P, N, _ = vars_hard
        ThetaH = QH.copy()  # L=0
        ThetaU = np.asarray(QU.value).reshape(-1)
        out = {
            "objective": float(prob_hard.value),
            "Qhat_unhedged": pd.Series(ThetaU, name="Unhedged"),
            "Q_fixed_hedged": pd.Series(QH, name="Hedged"),
            "Theta_unhedged": pd.Series(ThetaU, name="Theta_Unhedged"),
            "Theta_hedged": pd.Series(ThetaH, name="Theta_Hedged"),
            "S": pd.Series(np.asarray(S.value).reshape(-1), name="S"),
            "P": pd.Series(np.asarray(P.value).reshape(-1), name="P"),
            "N": pd.Series(np.asarray(N.value).reshape(-1), name="N"),
            "max_storage": float(np.max(P.value)),
            "min_storage": float(np.min(P.value)),
            "k_locked": int(k),
            "lb_softened": False,
            "solve_status_primary": status_hard,
        }
        return out

    # Fallback to SOFT LB if requested
    if not auto_soft_lb_if_infeasible:
        raise RuntimeError(f"L0 unhedged reschedule (prefix lock) failed: status={status_hard}")

    if verbose:
        print(f"[info] Hard-LB solve returned '{status_hard}'. Retrying with soft lower bound (penalty={soft_lb_penalty:g}).")

    prob_soft, vars_soft = _build_problem(use_soft_lb=True)
    prob_soft.solve(solver=solver, verbose=verbose)
    status_soft = prob_soft.status

    if status_soft not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"L0 unhedged reschedule (prefix lock) still failed after fallback: "
            f"hard='{status_hard}', soft='{status_soft}'"
        )

    QU, S, P, N, Y = vars_soft
    ThetaH = QH.copy()
    ThetaU = np.asarray(QU.value).reshape(-1)

    out = {
        "objective": float(prob_soft.value),
        "Qhat_unhedged": pd.Series(ThetaU, name="Unhedged"),
        "Q_fixed_hedged": pd.Series(QH, name="Hedged"),
        "Theta_unhedged": pd.Series(ThetaU, name="Theta_Unhedged"),
        "Theta_hedged": pd.Series(ThetaH, name="Theta_Hedged"),
        "S": pd.Series(np.asarray(S.value).reshape(-1), name="S"),
        "P": pd.Series(np.asarray(P.value).reshape(-1), name="P"),
        "N": pd.Series(np.asarray(N.value).reshape(-1), name="N"),
        "max_storage": float(np.max(P.value)),
        "min_storage": float(np.min(P.value)),
        "k_locked": int(k),
        "lb_softened": True,
        "lb_slack_sum": float(np.sum(Y.value)),
        "lb_slack": pd.Series(np.asarray(Y.value).reshape(-1), name="Y_lb"),
        "solve_status_primary": status_hard,
        "solve_status_fallback": status_soft,
    }
    return out


# -----------------------------
# Cost breakdown / plotting
# -----------------------------
def compute_cost_breakdown(sol, price_df, h=5.0, b=None):
    if "Qhat" in sol:
        hedged_orders   = sol["Qhat"]["Hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat"]["Unhedged"].to_numpy(float)
    else:
        hedged_orders   = sol["Q_fixed_hedged"].to_numpy(float)
        unhedged_orders = sol["Qhat_unhedged"].to_numpy(float)

    pH = price_df["Hedged"].to_numpy(float)
    pU = price_df["Unhedged"].to_numpy(float)

    P = sol["P"].to_numpy(float)

    hedged_cost    = float(np.sum(pH * hedged_orders))
    unhedged_cost  = float(np.sum(pU * unhedged_orders))
    storage_cost   = float(h * np.sum(P))

    out = {
        "hedged_cost": hedged_cost,
        "unhedged_cost": unhedged_cost,
        "storage_cost": storage_cost,
        "total_without_backlog": hedged_cost + unhedged_cost + storage_cost,
    }

    if b is not None and "N" in sol:
        N = sol["N"].to_numpy(float)
        backlog_cost = float(b * np.sum(N))
        out["backlog_cost"] = backlog_cost
        out["total_with_backlog"] = out["total_without_backlog"] + backlog_cost

    if "lb_slack_sum" in sol:
        out["lb_slack_sum"] = sol["lb_slack_sum"]

    return out


def plot_procurement_figs(sol, price_df, s_max=12000.0, prefix="fig"):
    # expects unhedged-only style solution dict
    QH = sol["Q_fixed_hedged"].to_numpy(float)
    QU = sol["Qhat_unhedged"].to_numpy(float)
    P  = sol["P"].to_numpy(float)
    S  = sol["S"].to_numpy(float)
    T  = len(QH)
    t  = np.arange(T)
    PH = price_df["Hedged"].to_numpy(float)
    PU = price_df["Unhedged"].to_numpy(float)

    start = pd.Timestamp("2025-04-01")
    months = pd.date_range(start, periods=T, freq="MS")
    xlabels = months.strftime("%b-%y")

    width = 0.35
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(t - width/2, QH, width=width, label="Hedged Order")
    ax1.bar(t + width/2, QU, width=width, label="Unhedged Order")
    ax1.set_ylabel("Orders")
    ax1.grid(True, alpha=0.3)

    axp = ax1.twinx()
    axp.plot(t, PH, linestyle=":", linewidth=2, label="Hedged Price")
    axp.plot(t, PU, linestyle=":", linewidth=2, label="Unhedged Price")
    axp.set_ylabel("Price")

    ax1.set_xticks(t)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")
    lines, labels = [], []
    for a in (ax1, axp):
        L = a.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax1.legend(lines, labels, loc="best")
    plt.title("Orders + Prices")
    plt.tight_layout(); plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(t, P, color="tab:blue", alpha=0.7, label="Stock (on-hand P)")
    ax2.axhline(s_max, linestyle="--", linewidth=1.5, color="red", label=f"Cap = {s_max:g}")
    ax2.set_ylabel("Units")
    ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_xticks(t); ax2.set_xticklabels(xlabels, rotation=45, ha="right")
    plt.title("Storage (On-hand P) vs Cap")
    plt.tight_layout(); plt.show()

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t, P, linewidth=2, label="On-hand P")
    ax4.plot(t, S, linewidth=1.8, label="Net S (=P-N)")
    ax4.axhline(s_max, linestyle="--", linewidth=1.5, label=f"Cap = {s_max:g}")
    ax4.set_ylabel("Units")
    ax4.grid(True, alpha=0.3); ax4.legend()
    ax4.set_xticks(t); ax4.set_xticklabels(xlabels, rotation=45, ha="right")
    plt.title("Storage (On-hand P) and Net Inventory S")
    plt.tight_layout(); plt.show()


# -----------------------------
# 3) Run + save results (examples)
# -----------------------------
if __name__ == "__main__":
    # Full matrix (both suppliers optimized)
    sol = deterministic_reschedule_with_capacity(
        Q_star_df=Q_star_df,
        price_df=price_df,
        capacity_df=capacity_df,
        demand=demand,
        lead_time=lead_time,
        s_max=s_max,
        h=5.0, b=20.0,
        I0=1000.0, B0=0.0,
        s_min=s_min,
        enforce_lb=True,            # set False + lb_penalty to soften
        lb_penalty=1e4,
        force_arrival_within_horizon=True,
        solver=cp.CLARABEL, verbose=False
    )
    print("Objective (full):", sol["objective"])
    print("Min/Max per-period storage:", sol["min_storage"], sol["max_storage"])
    if "lb_slack_sum" in sol:
        print("Lower-bound slack (sum):", sol["lb_slack_sum"])
    sol["Qhat"].to_csv("Qhat_orders.csv", index=False)

    # Unhedged-only (with lead times)
    solU = deterministic_reschedule_unhedged_only_with_capacity(
        Q_star_df=Q_star_df, price_df=price_df, capacity_df=capacity_df,
        demand=demand, lead_time=lead_time,
        s_max=s_max, h=5.0, b=50.0, I0=1000.0, B0=0.0,
        s_min=s_min, enforce_lb=True, lb_penalty=1e4,
        force_arrival_within_horizon=True, solver=cp.CLARABEL, verbose=False
    )
    print("Objective (Unhedged-only):", solU["objective"])

    # Lead-time 0 (no prefix lock)
    sol0 = deterministic_reschedule_unhedged_L0(
        Q_star_df=Q_star_df,
        price_df=price_df,
        capacity_df=capacity_df,
        demand=demand,
        s_max=s_max,
        h=5.0,
        I0=1000.0,
        no_backlog=True,
        s_min=s_min, enforce_lb=True, lb_penalty=1e4,
        solver=cp.CLARABEL,
        verbose=False
    )
    print("Objective (L=0):", sol0["objective"])

    # Lead-time 0 with prefix lock (auto-fallback if hard LB infeasible)
    sol0_lock = deterministic_reschedule_unhedged_L0_lock_prefix(
        Q_star_df=Q_star_df,
        price_df=price_df,
        capacity_df=capacity_df,
        demand=demand,
        k=4,                       # lock first k months of Unhedged to the plan
        s_max=s_max,
        h=5.0,
        I0=1000.0,
        no_backlog=True,
        s_min=s_min,               # scalar or vector
        solver=cp.CLARABEL,
        verbose=False,
        auto_soft_lb_if_infeasible=True,
        soft_lb_penalty=1e5,
        auto_relax_lb_prefix=True
    )
    print("Objective (L=0, first 4 locked):", sol0_lock["objective"])
    print("Used soft LB fallback?", sol0_lock.get("lb_softened", False))
    if sol0_lock.get("lb_softened"):
        print("LB slack sum:", sol0_lock.get("lb_slack_sum"))

    # Choose which solution to visualize (unhedged-style)
    sol_to_plot = sol0_lock
    plot_procurement_figs(sol_to_plot, price_df, s_max=s_max, prefix="fig")

    # Cost breakdowns
    print("Costs L=0 free:",  compute_cost_breakdown(sol0, price_df, h=5.0))
    print("Costs Unhedged:",  compute_cost_breakdown(solU, price_df, h=5.0, b=50.0))
    print("Costs full:",      compute_cost_breakdown(sol,  price_df, h=5.0, b=20.0))
