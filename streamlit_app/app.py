import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
plt.switch_backend('agg')

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from simulation import Simulation
from utils import convert_title_description, load_data, validate_inputs

def format_pct(value):
    return f"{value * 100:.1f}%"

def format_comma(x):
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return x

def main():
    st.title("Sales Compensation Simulation")
    try:
        # Load data from disk
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, sheet_name="MASTER", header=6)
        else:
            st.warning("Please upload your Excel file to proceed.")
            st.stop()

        df, df_comp, df_mm_bands = load_data()

        # Store original values for reset
        if 'original_df_comp' not in st.session_state:
            st.session_state.original_df_comp = df_comp.copy()
        if 'original_mm_bands' not in st.session_state:
            st.session_state.original_mm_bands = df_mm_bands.copy()
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None

        # --- Sidebar: Base Rate, Growth Rate, Multiplier Bands ---
        with st.sidebar:
            st.header("Commission Parameters")
            st.markdown("**Base Rate (%)**")
            base_rate = {}
            for title in df_comp.index:
                base_rate[title] = st.number_input(
                    f"Base Rate - {title}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(df_comp.loc[title, 'Base rate'] * 100),
                    step=0.1,
                    key=f"sidebar_base_{title}"
                ) / 100

            st.markdown("**Growth Rate (%)**")
            growth_rate = {}
            for title in df_comp.index:
                growth_rate[title] = st.number_input(
                    f"Growth Rate - {title}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(df_comp.loc[title, 'Growth Rate'] * 100),
                    step=0.1,
                    key=f"sidebar_growth_{title}"
                ) / 100

            st.markdown("**Multiplier Bands (%)**")
            modified_mm_bands = df_mm_bands.copy()
            cols = st.columns(len(df_mm_bands.columns))
            for idx in df_mm_bands.index:
                for i, col in enumerate(df_mm_bands.columns):
                    val = float(df_mm_bands.loc[idx, col])
                    if col == "Multiplier Bands":
                        val = val * 100  # percent for UI
                        modified_mm_bands.loc[idx, col] = cols[i].number_input(
                            f"{col} Band {idx+1}",
                            min_value=0.0,
                            max_value=100.0,
                            value=val,
                            step=0.1,
                            key=f"sidebar_mm_{idx}_{col}",
                            format="%.1f"
                        ) / 100
                    else:
                        modified_mm_bands.loc[idx, col] = cols[i].number_input(
                            f"{col} Band {idx+1}",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(df_mm_bands.loc[idx, col]),
                            step=0.1,
                            key=f"sidebar_mm_{idx}_{col}_mm",
                            format="%.1f"
                        )

        # --- Main Page: Growth Objectives ---
        st.subheader("Growth Objectives (%)")
        modified_df_comp = df_comp.copy()
        growth_obj_cols = st.columns(len(df_comp.index))
        for idx, title in enumerate(df_comp.index):
            modified_df_comp.loc[title, 'Growth Objective'] = growth_obj_cols[idx].number_input(
                f"{title}",
                min_value=0.0,
                max_value=100.0,
                value=float(df_comp.loc[title, 'Growth Objective'] * 100),
                step=0.1,
                key=f"growth_obj_{title}"
            ) / 100

        # Update base and growth rates in modified_df_comp
        for title in df_comp.index:
            modified_df_comp.loc[title, 'Base rate'] = base_rate[title]
            modified_df_comp.loc[title, 'Growth Rate'] = growth_rate[title]

        # --- Number of Simulations ---
        st.subheader("Number of Simulations")
        num_simulations = st.number_input("Number of Simulations", min_value=1, max_value=10000, value=100, step=10, key="custom_sim")


        # --- Run Simulation Button ---
        run_simulation = st.button("Run Simulation")

        if run_simulation:
            try:
                with st.spinner("Running simulation..."):
                    np.random.seed(42)
                    validate_inputs(modified_df_comp, modified_mm_bands)
                    simulation = Simulation(df, modified_df_comp, modified_mm_bands)
                    st.session_state.simulation_results = simulation.simulate_total_commission(
                        num_simulations=int(num_simulations),
                        growth_std=0.005,
                        cgp_sample_std=0.0
                    )
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")

        # --- Results ---
        st.header("Results")
        
        df['Title Description'] = df['Title Description'].apply(convert_title_description)
        rep_name = st.selectbox("Select Sales Rep", df['Full Name'].unique())

        # Show rep details
        rep_row = df[df['Full Name'] == rep_name].iloc[0]
        st.markdown(f"""
        **Title Description:** {rep_row['Title Description']}  
        **Manager Full Name:** {rep_row['Manager Full Name']}  
        **2024 Sales:** {format_comma(rep_row['2024 Sales'])}  
        **2024 Commission US:** {format_comma(rep_row['2024 Commission US'])}
        """)

        # --- Summary Table ---
        if st.session_state.simulation_results is not None:
            results = st.session_state.simulation_results

            # --- Line Chart ---
            st.subheader("Commission Projection with Confidence Bands")
            simulation = Simulation(df, modified_df_comp, modified_mm_bands)
            fig = simulation.plot_projection_with_bands(results, rep_name)
            st.pyplot(fig)

            # Group by Segment, Title Description, and get sum of means for each year per rep
            group_cols = []
            if 'Segment' in results.columns:
                group_cols.append('Segment')
            group_cols.append('Title Description')

            years = range(2025, 2031)
            summary_cols = [f"{year} Total Commission" for year in years]

            # Compute mean per rep per year, then sum across reps per group
            temp = results.groupby(group_cols + ['Full Name'])[summary_cols].mean().reset_index()
            summary = temp.groupby(group_cols)[summary_cols].sum().reset_index()

            # Add a 'Total' row
            total_row = {col: '' for col in summary.columns}
            total_row[group_cols[0]] = 'Total'
            for col in summary_cols:
                total_row[col] = summary[col].sum()
            summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

            # Add 2024 Actual Commission column to summary
            if 'Segment' in df.columns:
                group_keys = ['Segment', 'Title Description']
            else:
                group_keys = ['Title Description']

            actual_comm = df.groupby(group_keys)['2024 Commission US'].sum().reset_index()
            actual_comm.rename(columns={'2024 Commission US': '2024 Actual Commission'}, inplace=True)

            summary = pd.merge(actual_comm, summary, on=group_keys, how='right')

            # For the 'Total' row, sum the actual commissions
            total_actual = df['2024 Commission US'].sum()
            summary.loc[summary[group_keys[0]] == 'Total', '2024 Actual Commission'] = int(round(total_actual))

            # Format the new column
            summary['2024 Actual Commission'] = summary['2024 Actual Commission'].apply(
                lambda x: f"{int(round(x)):,}" if pd.notnull(x) and x != '' else x
            )

            for col in summary_cols:
                summary[col] = summary[col].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) and x != '' else x)

            st.subheader("Summary Table (Sum of Mean Simulated Commissions)")
            st.dataframe(summary)
            
            
            # --- Download Button ---
            reps = results['Full Name'].unique()
            download_rows = []
            for rep in reps:
                rep_data = results[results['Full Name'] == rep]
                row = {
                    "Sales Rep Name": rep,
                    "Title Description": rep_data['Title Description'].iloc[0],
                    "Segment": rep_data['Segment'].iloc[0] if 'Segment' in rep_data.columns else '',
                    "2024 Sales": rep_data['2024 Sales'].iloc[0],
                    "2024 Commission": rep_data['2024 Commission US'].iloc[0],
                }
                for year in years:
                    row[f"{year} Simulated Sales"] = round(rep_data[f"{year} Sales"].mean(), 2)
                    row[f"{year} Simulated Commission"] = round(rep_data[f"{year} Total Commission"].mean(), 2)
                download_rows.append(row)
            download_df = pd.DataFrame(download_rows)
            ordered_cols = (
                ["Sales Rep Name", "Title Description", "2024 Sales"] +
                [f"{year} Simulated Sales" for year in years] +
                ["2024 Commission"] +
                [f"{year} Simulated Commission" for year in years]
            )
            download_df = download_df[ordered_cols]
            csv = download_df.to_csv(index=False)
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