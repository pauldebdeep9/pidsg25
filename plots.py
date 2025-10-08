import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_order_placement_bar(order_placement, start_date="2024-01-01"):
    """
    Plots order placement over time as grouped bars (one per supplier per time point).

    Parameters:
    -----------
    order_placement : pd.DataFrame
        Matrix [T x S] with order quantities placed at each time by supplier.

    start_date : str
        Start date in 'YYYY-MM-DD' format.
    """
    order_placement= 31.104*order_placement
    T = order_placement.shape[0]
    time_index = pd.date_range(start=start_date, periods=T, freq='MS')  # month start
    order_placement.index = time_index.strftime('%b-%y')

    ax = order_placement.plot(kind='bar', stacked=False, figsize=(14, 6), colormap='Set2', width=0.8)
    plt.title("Order Placement Over Time by Supplier")
    plt.xlabel("Month")
    plt.ylabel("Order Quantity")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Supplier")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_price_distribution_band(price_df_s1, price_df_s2, start_date="2024-01-01", quantiles=(0.1, 0.9)):
    """
    Plots shaded band of price distribution over time for each supplier.

    Parameters:
    -----------
    price_df_s1 : pd.DataFrame
        Price samples for supplier 1 (index: time, columns: samples)

    price_df_s2 : pd.DataFrame
        Price samples for supplier 2

    start_date : str
        Date string for start of time index

    quantiles : tuple
        Lower and upper quantile for the uncertainty band (e.g., (0.1, 0.9) for 10th–90th percentile)
    """
    T = price_df_s1.shape[0]
    time_index = pd.date_range(start=start_date, periods=T, freq='MS')

    def plot_band(ax, df, label, color):
        mean = df.mean(axis=1)
        q_low = df.quantile(quantiles[0], axis=1)
        q_high = df.quantile(quantiles[1], axis=1)
        ax.plot(time_index, mean, label=f"{label} Mean", color=color)
        ax.fill_between(time_index, q_low, q_high, alpha=0.3, color=color, label=f"{label} {int(quantiles[0]*100)}–{int(quantiles[1]*100)}%")

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_band(ax, price_df_s1, "Supplier 1", "#1f77b4")  # blue
    plot_band(ax, price_df_s2, "Supplier 2", "#ff7f0e")  # orange

    ax.set_title("Price Distribution Bands Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_price_and_orders(price_df, order_placement, supplier, start_date="2024-01-01", quantiles=(0.1, 0.9)):
    """
    Plots price distribution band and order placement bars for a single supplier.

    Parameters:
    -----------
    price_df : pd.DataFrame
        Price samples for the supplier (index: time, columns: samples)

    order_placement : pd.DataFrame
        Order quantity per supplier per time (index: time, columns: suppliers)

    supplier : str
        Supplier name (e.g., 's1' or 's2')

    start_date : str
        Start date for x-axis (e.g., '2024-01-01')

    quantiles : tuple
        Lower and upper quantile for uncertainty band
    """
    T = price_df.shape[0]
    time_index = pd.date_range(start=start_date, periods=T, freq='MS')
    
    mean = price_df.mean(axis=1)
    q_low = price_df.quantile(quantiles[0], axis=1)
    q_high = price_df.quantile(quantiles[1], axis=1)
    orders = order_placement[supplier].values

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Price band
    ax1.plot(time_index, mean, label='Mean Price', color='tab:blue')
    ax1.fill_between(time_index, q_low, q_high, alpha=0.3, color='tab:blue', label=f"{int(quantiles[0]*100)}–{int(quantiles[1]*100)}% Band")
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Orders on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(time_index, orders, width=20, alpha=0.6, label='Order Quantity', color='tab:orange')
    ax2.set_ylabel('Order Quantity', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Final styling
    plt.title(f"Price Band and Orders Over Time - Supplier {supplier.upper()}")
    fig.autofmt_xdate()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_and_orders_deterministic(price_series, order_placement, supplier, start_date="2024-01-01"):
    """
    Plots deterministic price (line) and order quantities (bar) for a supplier.

    Parameters:
    -----------
    price_series : pd.Series
        Deterministic price over time for the supplier (length T)

    order_placement : pd.DataFrame
        Order matrix [T x S], with order quantities per supplier

    supplier : str
        Supplier name (e.g., 's2')

    start_date : str
        Start date for the x-axis
    """
    T = len(price_series)
    time_index = pd.date_range(start=start_date, periods=T, freq='MS')
    orders = order_placement[supplier].values

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Price line
    ax1.plot(time_index, price_series, label='Price', color='tab:blue', linewidth=2)
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Orders on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(time_index, orders, width=20, alpha=0.6, label='Order Quantity', color='tab:orange')
    ax2.set_ylabel('Order Quantity', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Final styling
    plt.title(f"Deterministic Price and Orders Over Time - Supplier {supplier.upper()}")
    fig.autofmt_xdate()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def plot_data_25()

def _bar_width_for_x(xindex):
    """Choose a width compatible with x-axis type."""
    if isinstance(xindex, pd.DatetimeIndex):
        return pd.Timedelta(days=20)  # good for monthly frequency
    return 0.8  # numeric/factor axis

def plot_price_and_orders_aligned(price_df, order_placed, supplier='s1', start_date=None, freq='MS'):
    """
    Plot price (line) and order quantity (bars) with safe alignment:
    - Builds a DatetimeIndex from `start_date` if needed.
    - Pads/truncates orders to the price horizon so lengths always match.
    """
    # 1) Extract a 1D price series
    if isinstance(price_df, pd.DataFrame):
        num_cols = price_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("price_df has no numeric columns to plot.")
        price_series = price_df[num_cols[0]].copy()
    else:
        price_series = pd.Series(np.asarray(price_df).ravel())

    # 2) Ensure a time index
    P = len(price_series)
    if isinstance(price_series.index, pd.DatetimeIndex):
        time_index = price_series.index
    else:
        if start_date is None:
            raise ValueError("start_date is required when price_df index is not DatetimeIndex.")
        time_index = pd.date_range(start=start_date, periods=P, freq=freq)
        price_series.index = time_index

    # 3) Orders: pad/truncate to match P
    orders_vec = np.zeros(P, dtype=float)
    if isinstance(order_placed, pd.DataFrame) and (supplier in order_placed.columns):
        T = len(order_placed)
        copy_upto = min(P, T)
        if copy_upto > 0:
            orders_vec[:copy_upto] = order_placed[supplier].iloc[:copy_upto].to_numpy(dtype=float)

    # 4) Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time_index, price_series.to_numpy(), linestyle='--', marker='o', label='Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(time_index, orders_vec, width=_bar_width_for_x(time_index), alpha=0.6, label='Order Quantity')
    ax2.set_ylabel('Order Quantity')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()

# ---- SHIM: keep old public name working everywhere ----
def plot_price_and_orders(price_df, order_placed, supplier='s1', start_date=None, freq='MS'):
    """Backward-compatible wrapper that calls the aligned implementation."""
    return plot_price_and_orders_aligned(price_df, order_placed, supplier=supplier, start_date=start_date, freq=freq)