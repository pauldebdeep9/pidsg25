import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

excel_path = 'pidsg25Evaluation.xlsx'
# Get all sheet names
xls = pd.ExcelFile(excel_path)
all_sheets = xls.sheet_names

# Select the right-most (last) sheet
last_sheet = all_sheets[-1]
print(f"Loading last sheet: {last_sheet}")

# Load the last sheet
df = pd.read_excel(excel_path, sheet_name=last_sheet)
print("Original columns:", df.columns.tolist())

# Ensure 'Date' column is datetime
df['Date'] = pd.to_datetime(df['Date'])   # <-- MODIFIED

# Prepare x-axis ticks: Apr-25 to Mar-26
xticks = pd.date_range(start='2025-04-01', end='2026-03-01', freq='MS')  # <-- MODIFIED
xticklabels = xticks.strftime('%b-%y')  # <-- MODIFIED

# Get X axis (time or appropriate column)
x = df['Date']
x_pos = np.arange(len(x))
bar_width = 0.35

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# --- Top: Manual Hedged vs Unhedged
ax[0].bar(x_pos - bar_width/2, df['Order-Manual-Hedged'], bar_width, label='Manual Hedged')
ax[0].bar(x_pos + bar_width/2, df['Order-Manual-Unhedged'], bar_width, label='Manual Unhedged')
ax[0].set_ylabel('Order Qty')
ax[0].set_title('Order - Manual Hedged vs Unhedged')
ax[0].legend()
ax[0].grid(True, axis='y', linestyle='--', alpha=0.4)

# --- Bottom: AI Hedged vs Unhedged
ax[1].bar(x_pos - bar_width/2, df['Order-AI-Hedged'], bar_width, label='AI Hedged')
ax[1].bar(x_pos + bar_width/2, df['Order-AI-Unhedged'], bar_width, label='AI Unhedged')
ax[1].set_ylabel('Order Qty')
ax[1].set_title('Order - AI Hedged vs Unhedged')
ax[1].legend()
ax[1].set_xlabel('Month')
ax[1].set_xticks(np.arange(len(xticks)))  # <-- MODIFIED
ax[1].set_xticklabels(xticklabels, rotation=45)  # <-- MODIFIED
ax[1].grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# Compute cumulative cost for MANUAL (hedged + unhedged)
manual_cum_cost = (
    df['Order-Manual-Hedged'] * df['Hedged price'] +
    df['Order-Manual-Unhedged'] * df['Unhedged price']
).cumsum()

# Compute cumulative cost for AI (hedged + unhedged)
ai_cum_cost = (
    df['Order-AI-Hedged'] * df['Hedged price'] +
    df['Order-AI-Unhedged'] * df['Unhedged price']
).cumsum()


manual_storage_cum = df['Manual-storage-cost'].cumsum()
ai_storage_cum = df['AI-storage-cost'].cumsum()

# Add them for total cost
manual_total_cum = manual_cum_cost + manual_storage_cum
ai_total_cum = ai_cum_cost + ai_storage_cum

def set_month_xticks(ax):  # <-- ADDED
    ax.set_xticks(xticks)  # <-- MODIFIED
    ax.set_xticklabels(xticklabels, rotation=45)  # <-- MODIFIED

plt.figure(figsize=(10,6))
plt.plot(df['Date'], manual_cum_cost, label='Manual (Hedged + Unhedged)', marker='o')
plt.plot(df['Date'], ai_cum_cost, label='AI (Hedged + Unhedged)', marker='s')
plt.xlabel('Month')
plt.ylabel('Cumulative Procurement Cost')
plt.title('Cumulative Procurement Cost: Manual vs AI')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
ax = plt.gca()
set_month_xticks(ax)  # <-- MODIFIED
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df['Date'], manual_storage_cum, label='Manual Storage Cost', marker='o')
plt.plot(df['Date'], ai_storage_cum, label='AI Storage Cost', marker='s')
plt.xlabel('Month')
plt.ylabel('Cumulative Storage Cost')
plt.title('Cumulative Storage Cost: Manual vs AI')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
ax = plt.gca()
set_month_xticks(ax)  # <-- MODIFIED
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df['Date'], manual_total_cum, label='Manual: Total Cumulative Cost', marker='o')
plt.plot(df['Date'], ai_total_cum, label='AI: Total Cumulative Cost', marker='s')
plt.xlabel('Month')
plt.ylabel('Total Cumulative Cost (Procurement + Storage)')
plt.title('Total Cumulative Cost: Manual vs AI')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
ax = plt.gca()
set_month_xticks(ax)  # <-- MODIFIED
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Manual-stock'], label='Manual Stock', marker='o')
plt.plot(df['Date'], df['AI-Stock'], label='AI Stock', marker='s')
plt.xlabel('Month')
plt.ylabel('Stock Level')
plt.title('Manual vs AI Stock Level Over Time')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Use the same month ticks and labels as before
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=45)

plt.tight_layout()
plt.show()
