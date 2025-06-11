import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from utils import map_to_mm

class Simulation:
    def __init__(self, df: pd.DataFrame, df_comp: pd.DataFrame, df_mm_bands: pd.DataFrame):
        """Initialize Simulation class with dataframes."""
        if not all(isinstance(df, pd.DataFrame) for df in [df, df_comp, df_mm_bands]):
            raise TypeError("All inputs must be pandas DataFrames")
            
        self.df = df.copy()
        self.df_comp = df_comp.copy()
        self.df_mm_bands = df_mm_bands.sort_values(by='Multiplier Bands')
        self.cgp_dist = self.df['CGP%'].dropna().values
        
        # Validate required columns
        required_cols = ['Full Name', '2024 Sales', '2024 Commission US', 'Title Description']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

    def simulate_total_commission(
        self, 
        num_simulations: int = 100,
        years: range = range(2025, 2032),
        growth_std: float = 0.005,
        cgp_sample_std: float = None
    ) -> pd.DataFrame:
        """Run simulation with given parameters."""
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if growth_std <= 0:
            raise ValueError("growth_std must be positive")
            
        try:
            base_sales = self.df['2024 Sales'].copy()
            base_commission = self.df['2024 Commission US'].copy()
            simulation_results = []

            for sim in range(num_simulations):
                sim_df = self.df.copy()
                prev_sales = base_sales.copy()
                prev_commission = base_commission.copy()

                for year in years:
                    # 1. Sample Growth Objective for each rep
                    growth_objective = sim_df['Title Description'].apply(
                        lambda title: self.df_comp.loc[title, 'Growth Objective'] if title in self.df_comp.index else 0
                    ).values
                    growth_objective_noisy = np.random.normal(growth_objective, growth_std)

                    # 2. Calculate Sales Growth for this year
                    sales_growth = prev_sales * growth_objective_noisy
                    sim_df[f'{year} Sales Growth'] = sales_growth

                    # 3. Calculate current year Sales
                    curr_sales = prev_sales + sales_growth
                    sim_df[f'{year} Sales'] = curr_sales

                    # 4. Sample CGP% for each rep
                    sampled_cgp_pct = np.random.choice(self.cgp_dist, size=len(sim_df), replace=True)
                    sim_df[f'{year} CGP%'] = sampled_cgp_pct

                    # 5. Calculate Growth GP$ (assume flat margin %)
                    growth_gp = sampled_cgp_pct * sales_growth
                    sim_df[f'{year} Growth GP$- asssume flat margin %'] = growth_gp

                    # 6. Calculate Growth Commission
                    growth_rate = sim_df['Title Description'].apply(
                        lambda title: self.df_comp.loc[title, 'Growth Rate'] if title in self.df_comp.index else 0
                    ).values
                    mm = sim_df['MM'].values if 'MM' in sim_df.columns else 1.0  # Use 1.0 if MM not present
                    growth_commission = growth_rate * growth_gp * mm
                    sim_df[f'{year} Growth Commission'] = growth_commission

                    # 7. Calculate Total Commission
                    total_commission = prev_commission + growth_commission
                    sim_df[f'{year} Total Commission'] = total_commission

                    # Prepare for next year
                    prev_sales = curr_sales
                    prev_commission = total_commission

                sim_df['Simulation'] = sim
                simulation_results.append(sim_df)

            all_simulations = pd.concat(simulation_results, ignore_index=True)
            return all_simulations
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}")
    
    def plot_total_commission(self, results, rep_name, years=range(2025, 2032)):
        """Plot total commission for a selected rep across years."""
        import matplotlib.pyplot as plt
        rep_data = results[results['Full Name'] == rep_name]
        fig, ax = plt.subplots(figsize=(10, 6))
        for year in years:
            ax.hist(rep_data[f'{year} Total Commission'], bins=30, alpha=0.5, label=str(year))
        ax.set_title(f'Total Commission Distribution for {rep_name}')
        ax.set_xlabel('Total Commission')
        ax.set_ylabel('Frequency')
        ax.legend()
        # if return_fig:
        #     return fig
        plt.show()

    def plot_projection_with_bands(self, results: pd.DataFrame, rep_name: str, 
                             years: range = range(2025, 2032)) -> plt.Figure:
        """
        Plot projection with confidence bands.
        Returns the figure object for Streamlit.
        """
        rep_data = results[results['Full Name'] == rep_name]
        
        # Calculate statistics for each year
        yearly_stats = {}
        for year in years:
            col = f'{year} Total Commission'
            yearly_stats[year] = {
                'mean': rep_data[col].mean(),
                'std': rep_data[col].std()
            }
        
        # Create arrays for plotting
        x = list(years)
        y_mean = [yearly_stats[year]['mean'] for year in years]
        y_std = [yearly_stats[year]['std'] for year in years]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean line
        ax.plot(x, y_mean, 'b-', label='Mean Projection')
        
        # Plot confidence bands
        ax.fill_between(x, 
                    [m - 2*s for m, s in zip(y_mean, y_std)],
                    [m + 2*s for m, s in zip(y_mean, y_std)],
                    color='lightblue', alpha=0.2, label='95% Confidence')
        
        ax.fill_between(x, 
                    [m - s for m, s in zip(y_mean, y_std)],
                    [m + s for m, s in zip(y_mean, y_std)],
                    color='blue', alpha=0.2, label='68% Confidence')
        
        # Add 2024 actual point
        ax.scatter([2024], [rep_data['2024 Commission US'].iloc[0]], 
                color='red', s=100, label='2024 Actual')
        
        # Customize plot
        ax.set_title(f'Commission Projection for {rep_name}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Commission ($)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        return fig