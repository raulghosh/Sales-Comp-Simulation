import pandas as pd
from pathlib import Path
from typing import Tuple
import re

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    try:
        # Load main data
        file_path = Path(__file__).parent.parent / "data" / "Model 2025 Baseline Simulation.xlsx"
        df = pd.read_excel(file_path, sheet_name="MASTER", header=6)
        
        # Create df_comp
        df_comp = pd.DataFrame({
            'Base rate': [0.05, 0.07, 0.12],
            'Growth Rate': [0.07, 0.11, 0.18],
            'Growth Objective': [0.05, 0.08, 0.07]
        }, index=['SR I', 'SR II', 'SR III'])
        
        # Create df_mm_bands
        df_mm_bands = pd.DataFrame({
            'Multiplier Bands': [0.21, 0.24, 0.29],
            'MM': [0.75, 1.0, 1.2]
        })
        
        return df, df_comp, df_mm_bands
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

def validate_inputs(df_comp: pd.DataFrame, df_mm_bands: pd.DataFrame) -> None:
    """Validate input parameters."""
    # Validate df_comp
    if not all(col in df_comp.columns for col in ['Base rate', 'Growth Rate', 'Growth Objective']):
        raise ValueError("df_comp missing required columns")
    if not all(df_comp[col].between(0, 1).all() for col in df_comp.columns):
        raise ValueError("All rates in df_comp must be between 0 and 1")
        
    # Validate df_mm_bands
    if not all(col in df_mm_bands.columns for col in ['Multiplier Bands', 'MM']):
        raise ValueError("df_mm_bands missing required columns")
    if not df_mm_bands['Multiplier Bands'].is_monotonic_increasing:
        raise ValueError("Multiplier Bands must be in ascending order")

# Define a function to map CGP% to MM
def map_to_mm(df:pd.DataFrame, cgp:float) -> pd.DataFrame:
    # Filter rows where 'Multiplier Bands' <= cgp
    valid_rows = df[df['Multiplier Bands'] <= cgp]
    if not valid_rows.empty:
        # Return the MM corresponding to the largest valid 'Multiplier Bands'
        return valid_rows.iloc[-1]['MM']
    return df['MM'].iloc[0]  # Default to the lowest MM if no match

# Convert Title Description to a standard format
def convert_title_description(title: str) -> str:
    """Convert Title Description to a standard format."""
    if re.search(r'\bSR I\b', title):
        return 'SR I'
    elif re.search(r'\bSR II\b', title):
        return 'SR II'
    elif re.search(r'\bSR III\b', title):
        return 'SR III'
    return title  # Return original if no match

def commission_summary(results: pd.DataFrame, years, groupby_cols):
    """
    Returns a summary DataFrame with sum of the mean of commissions across simulations,
    grouped by the specified columns.
    
    For each sales rep, it first computes the mean of each year's total commission across all simulations,
    then sums across sales reps within each group.
    """
    # List to store per-rep mean commissions
    per_rep_means = []

    # Get unique reps
    reps = results['Full Name'].unique()

    for rep in reps:
        rep_data = results[results['Full Name'] == rep]
        rep_group = rep_data.iloc[0][groupby_cols].to_dict()
        row = {col: rep_group[col] for col in groupby_cols}
        row['Full Name'] = rep

        for year in years:
            year_col = f"{year} Total Commission"
            row[year_col] = rep_data[year_col].mean()

        per_rep_means.append(row)

    # Create DataFrame of per-rep mean commissions
    df_rep_means = pd.DataFrame(per_rep_means)

    # Now group by the desired columns and sum
    summary_cols = [f"{year} Total Commission" for year in years]
    grouped_summary = df_rep_means.groupby(groupby_cols)[summary_cols].sum().reset_index()

    # Add 2024 actual commissions
    actual_2024 = results.drop_duplicates(subset=['Full Name'])[['Full Name', '2024 Commission US'] + groupby_cols]
    actual_2024_summary = actual_2024.groupby(groupby_cols)['2024 Commission US'].sum().reset_index()

    # Merge 2024 actual with simulated summary
    final_summary = pd.merge(actual_2024_summary, grouped_summary, on=groupby_cols, how='outer')

    return final_summary

def add_total_row(summary: pd.DataFrame, groupby_cols, years):
    """
    Adds a 'Total' row to the summary DataFrame.
    """
    summary_cols = [f"{year} Total Commission" for year in years]
    total_row = {col: '' for col in summary.columns}
    total_row[groupby_cols[0]] = 'Total'
    for col in summary_cols:
        total_row[col] = summary[col].sum()
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    return summary
