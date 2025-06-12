import pandas as pd
from pathlib import Path
from typing import Tuple

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