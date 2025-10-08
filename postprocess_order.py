
import pandas as pd

def extract_order_matrices(df_result):
    """
    Extracts order placement and arrival matrices from optimization results.

    Parameters:
    -----------
    df_result : pd.DataFrame
        A dataframe containing variable_name and value columns from CVXPY/Gurobi results.

    Returns:
    --------
    order_placement : pd.DataFrame
        Matrix [T × S] where entry (t,s) is the total order placed at time t for supplier s.

    order_arrival : pd.DataFrame
        Matrix [T × S] where entry (t,s) is the total order arriving at time t for supplier s.
    """

    # Filter order_quantity variables
    df_order = df_result[df_result['variable_name'].str.contains("order_quantity")].copy()

    # Parse placement time (t), supplier (s), and arrival time (t')
    df_order[['t', 's', 't_prime']] = df_order['variable_name'] \
        .str.extract(r"order_quantity\[(\d+),([a-zA-Z0-9_]+),(\d+)\]") \
        .astype({0: int, 2: int, 1: str}) \
        .rename(columns={0: 't', 1: 's', 2: 't_prime'})

    # Determine time horizon and supplier list
    T = max(df_order['t'].max(), df_order['t_prime'].max()) + 1
    suppliers = sorted(df_order['s'].unique())

    # Initialize empty matrices
    order_placement = pd.DataFrame(0.0, index=range(T), columns=suppliers)
    order_arrival = pd.DataFrame(0.0, index=range(T), columns=suppliers)

    # Fill matrices
    for _, row in df_order.iterrows():
        t, s, t_prime, val = row['t'], row['s'], row['t_prime'], row['value']
        if val is None:
            continue  # Skip undefined variables
        order_placement.at[t, s] += val
        order_arrival.at[t_prime, s] += val


    return order_placement, order_arrival
