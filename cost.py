import numpy as np

class Cost:
    def __init__(self, df_result, order_placed, initial_inventory=0.0, demand=None):
        self.df_result = df_result
        self.order_placed = order_placed  # DataFrame of shape (T, S)
        self.demand = np.array(demand)    # 1D array of shape (T,)
        self.initial_inventory = initial_inventory  # scalar

    def compute_inventory_backlog_cost(self, h, b):
        """
        Computes inventory and backlog cost per time period (returns vectors).

        - h: inventory holding cost per unit
        - b: backlog cost per unit
        """
        T = self.order_placed.shape[0]
        inventory = self.initial_inventory
        backlog = 0.0

        inv_cost = np.zeros(T)
        backlog_cost = np.zeros(T)

        for t in range(T):
            total_order = self.order_placed.iloc[t].sum()
            supply = inventory + total_order - backlog
            fulfilled = min(supply, self.demand[t])

            inventory = max(supply - self.demand[t], 0)
            backlog = max(self.demand[t] - supply, 0)

            inv_cost[t] = h * inventory
            backlog_cost[t] = b * backlog

        return inv_cost, backlog_cost
