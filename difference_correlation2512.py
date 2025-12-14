#!/usr/bin/env python3
"""
Correlation and Lead-Lag Analysis
Analyzes connections between changes in Consumption, Purchase, and Inventory Days
across two planning snapshots (Aug vs Sep)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns

# Reuse the CSV loading functions from difference_plot.py
from difference_plot import (
    load_multiheader_csv,
    extract_material_row,
    build_timeseries_for_material,
    _coerce_number,
    MONTHS
)

# -------------------- CONFIGURATION --------------------
CSV1 = "5.0 FY2025 Material Control.csv"  # Aug plan
CSV2 = "7.0 FY2025 Material Control.csv"  # Sep plan
MATERIAL = "Silver Paste"
LABEL1 = "Plan Aug"
LABEL2 = "Plan Sep"
# ------------------------------------------------------


def compute_percentage_changes(df1, df2, metrics):
    """
    Compute percentage changes between two snapshots for given metrics.
    Returns a DataFrame with % changes for each metric.
    """
    changes = {}
    
    for metric in metrics:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_change = 100.0 * (df2[metric] - df1[metric]) / df1[metric].abs()
        pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
        changes[metric] = pct_change
    
    return pd.DataFrame(changes)


def compute_cross_correlation(series1, series2, max_lag=5):
    """
    Compute cross-correlation between two time series at different lags.
    Positive lag means series2 leads series1.
    
    Returns:
        lags: array of lag values
        correlations: cross-correlation at each lag
    """
    # Remove NaN values
    mask = ~(np.isnan(series1) | np.isnan(series2))
    s1 = series1[mask].values
    s2 = series2[mask].values
    
    if len(s1) < 3:  # Need at least 3 points
        return np.array([]), np.array([])
    
    # Normalize series
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
    
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag < 0:
            # series1 leads series2
            corr, _ = pearsonr(s1_norm[:lag], s2_norm[-lag:]) if len(s1_norm[:lag]) > 1 else (np.nan, np.nan)
        elif lag > 0:
            # series2 leads series1
            corr, _ = pearsonr(s1_norm[lag:], s2_norm[:-lag]) if len(s1_norm[lag:]) > 1 else (np.nan, np.nan)
        else:
            # No lag
            corr, _ = pearsonr(s1_norm, s2_norm)
        
        correlations.append(corr)
    
    return lags, np.array(correlations)


def plot_change_comparison(pct_changes, title="Percentage Changes Comparison"):
    """
    Plot the percentage changes for all three metrics on the same chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = pct_changes.index
    x = np.arange(len(months))
    
    # Plot each metric
    for col in pct_changes.columns:
        ax.plot(x, pct_changes[col], marker='o', label=col, linewidth=2)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_ylabel('% Change (Sep vs Aug)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(pct_changes):
    """
    Plot correlation matrix between the three metrics' changes.
    """
    # Compute correlation matrix
    corr_matrix = pct_changes.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: % Changes Between Metrics')
    plt.tight_layout()
    plt.show()


def plot_cross_correlation_analysis(pct_changes):
    """
    Plot cross-correlation analysis showing lead-lag relationships.
    """
    metrics = pct_changes.columns.tolist()
    pairs = [
        (metrics[0], metrics[1]),  # Consumption vs Purchase
        (metrics[0], metrics[2]),  # Consumption vs Inv Days
        (metrics[1], metrics[2]),  # Purchase vs Inv Days
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (m1, m2) in enumerate(pairs):
        ax = axes[idx]
        lags, corr = compute_cross_correlation(pct_changes[m1], pct_changes[m2], max_lag=3)
        
        if len(lags) > 0:
            ax.bar(lags, corr, width=0.6, alpha=0.7)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel('Lag (months)')
            ax.set_ylabel('Cross-correlation')
            ax.set_title(f'{m1} vs {m2}')
            ax.grid(True, alpha=0.3)
            
            # Mark maximum correlation
            if not np.all(np.isnan(corr)):
                max_idx = np.nanargmax(np.abs(corr))
                max_lag = lags[max_idx]
                max_corr = corr[max_idx]
                ax.axvline(max_lag, color='red', linestyle=':', linewidth=2, alpha=0.5)
                
                # Add interpretation text
                if max_lag < 0:
                    lead_text = f'{m1} leads by {abs(max_lag)}mo'
                elif max_lag > 0:
                    lead_text = f'{m2} leads by {max_lag}mo'
                else:
                    lead_text = 'Synchronous'
                
                ax.text(0.05, 0.95, f'{lead_text}\nCorr: {max_corr:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_stacked_changes(pct_changes):
    """
    Plot stacked view of changes to see cumulative patterns.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    months = pct_changes.index
    x = np.arange(len(months))
    
    metrics = pct_changes.columns.tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (ax, metric, color) in enumerate(zip(axes, metrics, colors)):
        ax.bar(x, pct_changes[metric], alpha=0.7, color=color, label=metric)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('% Change')
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on bars
        for i, v in enumerate(pct_changes[metric]):
            if not np.isnan(v):
                va = 'bottom' if v >= 0 else 'top'
                ax.text(i, v, f'{v:.1f}%', ha='center', va=va, fontsize=8)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(months, rotation=45, ha='right')
    axes[-1].set_xlabel('Month')
    
    plt.suptitle('Stacked View: Individual Metric Changes (Sep vs Aug)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def analyze_lead_lag_summary(pct_changes):
    """
    Print summary statistics and lead-lag relationships.
    """
    print("\n" + "="*70)
    print("LEAD-LAG ANALYSIS SUMMARY")
    print("="*70)
    
    metrics = pct_changes.columns.tolist()
    
    # Basic statistics
    print("\n1. BASIC STATISTICS (% Changes)")
    print("-" * 70)
    print(pct_changes.describe())
    
    # Correlation summary
    print("\n2. CORRELATION MATRIX")
    print("-" * 70)
    corr_matrix = pct_changes.corr()
    print(corr_matrix)
    
    # Lead-lag relationships
    print("\n3. LEAD-LAG RELATIONSHIPS")
    print("-" * 70)
    
    pairs = [
        (metrics[0], metrics[1], 'Consumption-Purchase'),
        (metrics[0], metrics[2], 'Consumption-Inventory Days'),
        (metrics[1], metrics[2], 'Purchase-Inventory Days'),
    ]
    
    for m1, m2, name in pairs:
        lags, corr = compute_cross_correlation(pct_changes[m1], pct_changes[m2], max_lag=3)
        
        if len(lags) > 0 and not np.all(np.isnan(corr)):
            max_idx = np.nanargmax(np.abs(corr))
            max_lag = lags[max_idx]
            max_corr = corr[max_idx]
            
            print(f"\n{name}:")
            print(f"  Max correlation: {max_corr:.3f} at lag {max_lag}")
            
            if max_lag < 0:
                print(f"  → {m1} changes LEAD {m2} changes by {abs(max_lag)} month(s)")
            elif max_lag > 0:
                print(f"  → {m2} changes LEAD {m1} changes by {max_lag} month(s)")
            else:
                print(f"  → Changes are SYNCHRONOUS (no lag)")
            
            # Interpretation
            if abs(max_corr) > 0.7:
                strength = "Strong"
            elif abs(max_corr) > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "positive" if max_corr > 0 else "negative"
            print(f"  → {strength} {direction} relationship")


def main():
    print(f"Loading data from:")
    print(f"  Aug Plan: {CSV1}")
    print(f"  Sep Plan: {CSV2}")
    print(f"  Material: {MATERIAL}\n")
    
    # Load data
    df_aug = build_timeseries_for_material(CSV1, MATERIAL)
    df_sep = build_timeseries_for_material(CSV2, MATERIAL)
    
    # Convert consumption values to absolute (they are negative in the data)
    df_aug['consumption_planned'] = df_aug['consumption_planned'].abs()
    df_aug['consumption_actual'] = df_aug['consumption_actual'].abs()
    df_sep['consumption_planned'] = df_sep['consumption_planned'].abs()
    df_sep['consumption_actual'] = df_sep['consumption_actual'].abs()
    
    # Define metrics to analyze
    # Use planned values for forward-looking analysis
    # For inventory days, use actual when available, planned otherwise
    df_aug['inv_days'] = df_aug['inv_days_actual'].fillna(df_aug['inv_days_planned'])
    df_sep['inv_days'] = df_sep['inv_days_actual'].fillna(df_sep['inv_days_planned'])
    
    metrics = {
        'Consumption (Planned)': 'consumption_planned',
        'Purchase (Planned)': 'purchase_planned',
        'Inventory Days': 'inv_days'
    }
    
    # Compute percentage changes
    pct_changes = compute_percentage_changes(
        df_aug, 
        df_sep, 
        list(metrics.values())
    )
    
    # Rename columns for better readability
    pct_changes.columns = list(metrics.keys())
    
    print("Percentage Changes (Sep vs Aug):")
    print(pct_changes)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Overlay comparison
    plot_change_comparison(pct_changes, 
                          "Percentage Changes: Consumption vs Purchase vs Inventory Days")
    
    # 2. Correlation matrix
    plot_correlation_matrix(pct_changes)
    
    # 3. Cross-correlation analysis (lead-lag)
    plot_cross_correlation_analysis(pct_changes)
    
    # 4. Stacked view
    plot_stacked_changes(pct_changes)
    
    # 5. Summary statistics and interpretation
    analyze_lead_lag_summary(pct_changes)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
