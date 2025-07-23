import pandas as pd

def load_and_prepare_data(file_path):
    """
    Loads the CSV file and prepares the DataFrame by parsing dates
    in day/month/year format and extracting the year-month period.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')  # Correct date format
    df['month'] = df['date'].dt.to_period('M')
    return df

def generate_monthly_prediction_matrix(df, output_path=None):
    """
    Generates a matrix with Part Numbers as rows, months as columns,
    and total predictions as entries. Optionally saves to a CSV.
    Also returns monthly totals summed across all parts.
    """
    monthly_pred_matrix = df.groupby(['Part Number', 'month'])['pred'].sum().unstack(fill_value=0)
    monthly_totals = monthly_pred_matrix.sum(axis=0)  # sum across parts for each month
    if output_path:
        monthly_pred_matrix.to_csv(output_path)
    return monthly_pred_matrix, monthly_totals

def preprocess_forecast_excel(file_path, sheet_name='Production qty (forecast)'):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
    df = df.rename(columns={df.columns[0]: 'Part Number'})
    df = df.set_index('Part Number')

    # Clean column names and convert to Period
    df.columns = [pd.Period(c.split()[0] + ' ' + c.split()[1], freq='M') for c in df.columns]

    return df

def reconcile_monthly_forecast(df1, df2):
    """
    Reconciles two DataFrames with Part Number x Month matrices.
    Returns the difference: df1 - df2 (matching by Part Number and month).
    """
    # Ensure columns are of Period type
    df1.columns = pd.PeriodIndex(df1.columns, freq='M')
    df2.columns = pd.PeriodIndex(df2.columns, freq='M')

    # Align and subtract
    diff = df1.sub(df2, fill_value=0)

    return diff

def compare_part_numbers(df1, df2):
    """
    Compare Part Numbers between two DataFrames.
    Returns:
        - parts_only_in_df1
        - parts_only_in_df2
        - common_parts
    """
    parts_df1 = set(df1.index)
    parts_df2 = set(df2.index)

    only_in_df1 = parts_df1 - parts_df2
    only_in_df2 = parts_df2 - parts_df1
    common_parts = parts_df1 & parts_df2

    return only_in_df1, only_in_df2, common_parts

def fuzzy_match_part_numbers(df1, df2):
    """
    Match part numbers between df1 and df2 by substring containment.
    Returns a mapping from df1 parts to df2 parts if one is contained in the other.
    """
    mapping = {}
    unmatched_df1 = set()
    unmatched_df2 = set(df2.index)

    for part1 in df1.index:
        match_found = False
        for part2 in df2.index:
            if part1.replace(" ", "") in part2.replace(" ", "") or part2.replace(" ", "") in part1.replace(" ", ""):
                mapping[part1] = part2
                unmatched_df2.discard(part2)
                match_found = True
                break
        if not match_found:
            unmatched_df1.add(part1)

    return mapping, unmatched_df1, unmatched_df2




# Example usage
if __name__ == "__main__":
    file_path = 'AI-forecast.csv'  # Adjust path if needed
    df = load_and_prepare_data(file_path)
    monthly_matrix, monthly_sum = generate_monthly_prediction_matrix(df, output_path='monthly_prediction_matrix.csv')

    # Optional: print results
    print("Monthly prediction matrix (head):")
    print(monthly_matrix.head())
    
    print("\nMonthly totals across all parts:")
    print(monthly_sum)

    # monthly_matrix.to_csv('distribution.csv')
    # monthly_sum.to_csv('order_sum_pdic')

    # Assume you already have df_pred (from CSV) and want to compare with Excel data
    df_excel = preprocess_forecast_excel("Production Qty (Forecast).xlsx")
    # monthly_matrix = your prediction matrix (from CSV)
# df_excel = production qty matrix (from Excel)

    part_mapping, missing_in_pred, missing_in_excel = fuzzy_match_part_numbers(monthly_matrix, df_excel)

    print("Mapped Part Numbers (Prediction → Excel):")
    for k, v in part_mapping.items():
        print(f"{k} → {v}")

    print("\nUnmatched in sales forecast (not found in APP):", missing_in_pred)
    print("Unmatched in APP (not matched from sales forecast):", missing_in_excel)

    # diff_matrix = reconcile_monthly_forecast(monthly_matrix, df_excel)
    # # diff_matrix.to_csv('reconcile.csv')
    # print(diff_matrix.head())

    # only_in_pred, only_in_excel, common_parts = compare_part_numbers(monthly_matrix, df_excel)

    # print("Parts missing in Excel (but present in prediction):", only_in_pred)
    # print("Parts missing in Prediction (but present in Excel):", only_in_excel)
    # print("Total common parts:", len(common_parts))



