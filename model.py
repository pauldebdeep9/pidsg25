
import cvxpy as cp
import numpy as np
import pandas as pd

def solve_price_saa(fixed_demand, 
                    price_samples, 
                    order_cost, 
                    lead_time, 
                    capacity_dict, 
                    h, 
                    b, 
                    I_0, 
                    B_0,
                    fixed_orders_s2=None,
                    time_index=None):
    
    T = len(fixed_demand)
    if time_index is None:
        time = list(range(len(fixed_demand)))
    else:
        time = time_index

    S = list(order_cost.keys())
    N = len(price_samples)

    # time = list(range(T))
    t_supplier_tprime = [(t, s, t_prime) for t in time for s in S for t_prime in time if t_prime >= t]
    t_supplier = [(t, s) for t in time for s in S]

    # Decision variables
    Q = {(t, s, t_prime): cp.Variable(nonneg=True, name=f"order_quantity[{t},{s},{t_prime}]")
         for t, s, t_prime in t_supplier_tprime}
    theta = {(t, s): cp.Variable(nonneg=True, name=f"arrive_quantity[{t},{s}]") for t, s in t_supplier}
    Y = {(t, s, t_prime): cp.Variable(nonneg=True, name=f"if_make_order_arrive[{t},{s},{t_prime}]")  # relaxed binary
         for t, s, t_prime in t_supplier_tprime}
    I = {t: cp.Variable(nonneg=True, name=f"inventory[{t}]") for t in time}
    B = {t: cp.Variable(nonneg=True, name=f"backlog[{t}]") for t in time}

    # Objective: expected procurement + inventory/backlog cost
    sample_costs = []
    for n in range(N):
        sample_cost = 0
        for t, s, t_prime in t_supplier_tprime:
            sample_cost += price_samples[n][(t, s)] * Q[t, s, t_prime]
        for t, s, t_prime in t_supplier_tprime:
            sample_cost += order_cost[s] * Y[t, s, t_prime]
        sample_costs.append(sample_cost)
    obj1 = sum(sample_costs) / N
    obj2 = sum(h * I[t] + b * B[t] for t in time)
    objective = cp.Minimize(obj1 + obj2)

    constraints = []

    # Inventory balance
    for t in time:
        inflow = sum(theta[t, s] for s in S)
        if t == 0:
            constraints.append(I[t] - B[t] == I_0 - B_0 + inflow - fixed_demand[t])
        else:
            constraints.append(I[t] - B[t] == I[t - 1] - B[t - 1] + inflow - fixed_demand[t])

    # Order-arrival logic with lead time
    for s in S:
        for t_prime in time:
            lhs = sum((t + 1) * Y[t, s, t_prime] for t in time if t <= t_prime)
            rhs = sum(Y[t, s, t_prime] for t in time if t <= t_prime) * (t_prime + 1 - int(lead_time[s]))
            constraints.append(lhs == rhs)
            constraints.append(sum(Y[t, s, t_prime] for t in time if t <= t_prime) <= 1)

            valid_orders = [Q[t, s, t_prime] for t in time if bool(t <= t_prime and t_prime >= t + int(lead_time[s]))]
            if len(valid_orders) > 0:
                constraints.append(theta[t_prime, s] == sum(valid_orders))
            else:
                constraints.append(theta[t_prime, s] == 0)

    # Lead time constraint: orders can't arrive early
    total_demand = sum(fixed_demand)
    for t, s, t_prime in t_supplier_tprime:
        if bool(t_prime < t + int(lead_time[s])):
            constraints.append(Q[t, s, t_prime] == 0)
        else:
            constraints.append(Q[t, s, t_prime] <= Y[t, s, t_prime] * total_demand)

    # Capacity constraint per time-slot per supplier
    for t in time:
        for s in S:
            valid_Qs = [Q[t, s, t_prime] for t_prime in time if t_prime >= t and t_prime >= t + int(lead_time[s])]
            if len(valid_Qs) > 0:
                constraints.append(sum(valid_Qs) <= capacity_dict[t, s])

    # Order frequency limit
    for s in S:
        for t in time:
            constraints.append(sum(Y[t, s, t_prime] for t_prime in time if t_prime >= t) <= 1)

    # Enforce fixed order values for supplier 's2' if specified
    # if fixed_orders_s2:
    #     for (t, t_prime), val in fixed_orders_s2.items():
    #         constraints.append(Q[t, 's2', t_prime] == val)

    if fixed_orders_s2:
        for (t, t_prime), val in fixed_orders_s2.items():
            constraints.append(Q[t, 's2', t_prime] == val)
            constraints.append(Y[t, 's2', t_prime] == 1)  # ✅ ensure this order is marked active


    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIPY, verbose=True)

    # Collect results
    df_result = pd.DataFrame(
        [(var.name(), var.value) for var in Q.values()] +
        [(var.name(), var.value) for var in theta.values()] +
        [(var.name(), var.value) for var in Y.values()] +
        [(var.name(), var.value) for var in I.values()] +
        [(var.name(), var.value) for var in B.values()],
        columns=["variable_name", "value"]
    )

    return prob.value, df_result
