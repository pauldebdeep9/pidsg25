
import pandas as pd
import numpy as np

# Load the Excel file
# file_path = "price_pidsg.xlsx"  # update the path as needed
file_path= 'pidsg25-02.xlsx'
file_path= 'pidsg25-02_historical.xlsx'
xls = pd.ExcelFile(file_path)

# Read both sheets
sheet1 = xls.parse(xls.sheet_names[0])
sheet2 = xls.parse(xls.sheet_names[1])

# Extract the first column of each sheet
col1 = sheet1.iloc[:, 1].values
col2 = sheet2.iloc[:, 1].values

def generate_matrix(original_col):
    """
    Generate a (T x 100) matrix such that:
    - Each row has mean equal to the original entry
    - Each row has variance = 0.2 × original entry
    - Each column has variance matching original column variance
    """
    T = len(original_col)
    generated = np.zeros((T, 100))

    for t in range(T):
        mean_t = original_col[t]
        var_t = 0.3 * mean_t
        std_t = np.sqrt(var_t)
        generated[t, :] = np.random.normal(loc=mean_t, scale=std_t, size=100)

    # Adjust column-wise variance
    original_col_variance = np.var(original_col, ddof=1)
    for j in range(100):
        col = generated[:, j]
        col_std = np.std(col, ddof=1)
        if col_std > 0:
            generated[:, j] = (col - np.mean(col)) / col_std * np.sqrt(original_col_variance) + np.mean(col)

    return generated

# Generate matrices for both sheets
generated1 = generate_matrix(col1)
generated2 = generate_matrix(col2)

# Optionally convert to DataFrame and save
df1 = pd.DataFrame(generated1)
df2 = pd.DataFrame(generated2)

# Save to Excel
with pd.ExcelWriter("generated_price_vectors_hist.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Generated_Sheet1", index=False)
    df2.to_excel(writer, sheet_name="Generated_Sheet2", index=False)

