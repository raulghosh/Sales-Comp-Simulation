import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from simulation import Simulation
from utils import load_data, validate_inputs

def format_pct(value):
    """Format float as percentage"""
    return f"{value * 100:.1f}%"

def format_comma(x):
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return x

def main():
    st.title("Sales Compensation Simulation")
    
    # File uploader for Excel file
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name="MASTER", header=6)
        # define df_comp and df_mm_bands here, or let user edit as before
    else:
        st.warning("Please upload your Excel file to proceed.")
        st.stop()
    
    try:
        # Load data
        df, df_comp, df_mm_bands = load_data()
        
        # Store original values for reset
        if 'original_df_comp' not in st.session_state:
            st.session_state.original_df_comp = df_comp.copy()
        if 'original_mm_bands' not in st.session_state:
            st.session_state.original_mm_bands = df_mm_bands.copy()
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None

        # Sidebar for simulation parameters
        with st.sidebar:
            st.header("Simulation Parameters")
            seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
            num_simulations = st.number_input("Number of Simulations", min_value=1, max_value=10000, value=100)
            growth_std = st.number_input("Growth Standard Deviation (%)", min_value=0.1, max_value=10.0, value=0.5, step=0.1) / 100
            cgp_std = st.number_input("CGP% Standard Deviation (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100  # Default 0%

        # Streamlined interface for SR rates
        st.subheader("Commission Rates (%)")
        cols = st.columns(len(df_comp.columns) + 1)
        cols[0].write("SR Level")
        for i, col_name in enumerate(df_comp.columns, 1):
            cols[i].write(f"**{col_name}**")
        modified_df_comp = df_comp.copy()
        for title in df_comp.index:
            cols = st.columns(len(df_comp.columns) + 1)
            cols[0].write(f"**{title}**")
            for i, col in enumerate(df_comp.columns, 1):
                modified_df_comp.loc[title, col] = cols[i].number_input(
                    f"{col} {title}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(df_comp.loc[title, col] * 100),
                    step=0.1,
                    key=f"{title}_{col}",
                    format="%.1f"
                ) / 100

        # Multiplier Bands as %
        st.subheader("Multiplier Bands (%)")
        modified_mm_bands = df_mm_bands.copy()
        cols = st.columns(len(df_mm_bands.columns))
        for idx in df_mm_bands.index:
            for i, col in enumerate(df_mm_bands.columns):
                val = float(df_mm_bands.loc[idx, col])
                if col == "Multiplier Bands":
                    val = val * 100  # Convert to percent for UI
                    modified_mm_bands.loc[idx, col] = cols[i].number_input(
                        f"{col} Band {idx+1}",
                        min_value=0.0,
                        max_value=100.0,
                        value=val,
                        step=0.1,
                        key=f"mm_{idx}_{col}",
                        format="%.1f"
                    ) / 100  # Convert back to fraction
                else:
                    modified_mm_bands.loc[idx, col] = cols[i].number_input(
                        f"{col} Band {idx+1}",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(df_mm_bands.loc[idx, col]),
                        step=0.1,
                        key=f"mm_{idx}_{col}_mm",
                        format="%.1f"
                    )

        # Reset buttons in a single line
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reset Commission Rates"):
                for title in df_comp.index:
                    for col in df_comp.columns:
                        key = f"{title}_{col}"
                        if key in st.session_state:
                            del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("Reset Multiplier Bands"):
                for idx in df_mm_bands.index:
                    for i, col in enumerate(df_mm_bands.columns):
                        if col == "Multiplier Bands":
                            key = f"mm_{idx}_{col}"
                        else:
                            key = f"mm_{idx}_{col}_mm"
                        if key in st.session_state:
                            del st.session_state[key]
                st.rerun()

        # Run Simulation button
        with col3:
            run_simulation = st.button("Run Simulation")

        if run_simulation:
            try:
                with st.spinner("Running simulation..."):
                    np.random.seed(seed)
                    validate_inputs(modified_df_comp, modified_mm_bands)
                    simulation = Simulation(df, modified_df_comp, modified_mm_bands)
                    st.session_state.simulation_results = simulation.simulate_total_commission(
                        num_simulations=num_simulations,
                        growth_std=growth_std,
                        cgp_sample_std=cgp_std
                    )
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")

        # Sales Rep Selection and Results Display
        st.header("Results")
        rep_name = st.selectbox("Select Sales Rep", df['Full Name'].unique())

        # Show rep details
        rep_row = df[df['Full Name'] == rep_name].iloc[0]
        st.markdown(f"""
        **Title Description:** {rep_row['Title Description']}  
        **Manager Full Name:** {rep_row['Manager Full Name']}  
        **2024 Sales:** {format_comma(rep_row['2024 Sales'])}  
        **2024 Commission US:** {format_comma(rep_row['2024 Commission US'])}
        """)

        if st.session_state.simulation_results is not None:
            results = st.session_state.simulation_results
            simulation = Simulation(df, modified_df_comp, modified_mm_bands)
            
            # Plot results
            st.subheader("Commission Projection with Confidence Bands")
            fig = simulation.plot_projection_with_bands(results, rep_name)
            st.pyplot(fig)
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            years = range(2025, 2032)
            stats_data = {}
            for year in years:
                col = f'{year} Total Commission'
                year_data = results[results['Full Name'] == rep_name][col]
                stats_data[year] = {
                    'Mean': year_data.mean(),
                    # 'Std Dev': year_data.std(),
                    'Min': year_data.min(),
                    'Max': year_data.max(),
                    '25%': year_data.quantile(0.25),
                    '50%': year_data.quantile(0.50),
                    '75%': year_data.quantile(0.75)
                }
            stats_df = pd.DataFrame(stats_data).T
            stats_df = stats_df.applymap(format_comma)
            st.dataframe(stats_df)
            
            st.subheader("Distribution by Year")
            dist_fig = simulation.plot_total_commission(results, rep_name)
            st.pyplot(dist_fig)
            
            # After simulation is run and results are available

            if st.session_state.simulation_results is not None:
                results = st.session_state.simulation_results

                # Prepare mean summary DataFrame
                reps = results['Full Name'].unique()
                years = range(2025, 2032)
                sales_cols = [f"{year} Sales" for year in years]
                comm_cols = [f"{year} Total Commission" for year in years]

                summary_rows = []
                for rep in reps:
                    rep_data = results[results['Full Name'] == rep]
                    row = {
                        "Sales Rep Name": rep,
                        "Title Description": rep_data['Title Description'].iloc[0],
                        "2024 Sales": rep_data['2024 Sales'].iloc[0],
                        "2024 Commission": rep_data['2024 Commission US'].iloc[0],
                    }
                    # Add mean simulated sales and commission for each year
                    for year in years:
                        row[f"{year} Simulated Sales"] = rep_data[f"{year} Sales"].mean()
                        row[f"{year} Simulated Commission"] = rep_data[f"{year} Total Commission"].mean()
                    summary_rows.append(row)

                summary_df = pd.DataFrame(summary_rows)

                # Reorder columns for download
                ordered_cols = (
                    ["Sales Rep Name", "Title Description", "2024 Sales"] +
                    [f"{year} Simulated Sales" for year in years] +
                    ["2024 Commission"] +
                    [f"{year} Simulated Commission" for year in years]
                )
                summary_df = summary_df[ordered_cols]

                # Download button
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Simulated Means as CSV",
                    data=csv,
                    file_name="simulated_commissions.csv",
                    mime="text/csv"
                )


    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()