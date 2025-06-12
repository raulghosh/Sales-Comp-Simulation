# Sales Comp Simulation

A Streamlit app to **simulate and visualize sales representatives' compensation** for the next 5 years, based on user-defined assumptions and historical data.

---

## **Objective**

This tool allows users to:
- Simulate future sales reps' compensation (commissions) for the next 5 years.
- Adjust key parameters such as number of simulations, sales growth rate assumptions (and noise), commission rates, growth objectives, and multiplier bands.
- Visualize the distribution and uncertainty of future compensation using Monte Carlo simulation, where future commission margins are sampled from the current year's commission margin distribution.

---

## **Features**

- **Flexible Simulation:**  
  Set the number of simulations, growth rate, and standard deviation (noise) for sales growth and commission margin.
- **Customizable Parameters:**  
  Edit base rate, growth rate, growth objective, multiplier, and multiplier bands for each sales rep level.
- **Data-Driven Sampling:**  
  All future commission margin sampling is based on the distribution of the current commission margin (CGP%) in your data.
- **Interactive Visualizations:**  
  View projections, confidence intervals, and summary statistics for each sales rep.
- **Download Results:**  
  Export the mean simulated sales and commissions for each rep and year as a CSV file.

---

## **How to Use**

### **1. Prerequisites**

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)

### **2. Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sales-comp-simulation.git
cd sales-comp-simulation
pip install -r requirements.txt
```

### **3. Prepare Your Data**

- Place your Excel file (e.g., `Model 2025 Baseline Simulation.xlsx`) in the `data/` directory.
- The file should have a sheet named `MASTER` and the data should start from row 7 (header=6 in pandas).

### **4. Run the App**

```bash
streamlit run streamlit_app/app.py
```

### **5. Using the App**

1. **Simulation Parameters:**  
   Set the random seed, number of simulations, growth standard deviation, and CGP% standard deviation in the sidebar.

2. **Commission Rates & Multiplier Bands:**  
   Adjust the base rate, growth rate, growth objective, and multiplier bands for each sales rep level. Use the reset buttons to revert to defaults.

3. **Run Simulation:**  
   Click "Run Simulation" to generate projections.

4. **View Results:**  
   - Select a sales rep from the dropdown to view their details, projected commissions, and summary statistics.
   - Visualizations include confidence bands and distribution histograms.

5. **Download Results:**  
   - After running a simulation, use the "Download Simulated Means as CSV" button to export the mean simulated sales and commissions for each rep and year.

---

## **Best Practices**

- Ensure your data file is named and structured correctly.
- Do **not** upload sensitive or proprietary data to public repositories.
- Use the reset buttons to quickly revert any parameter changes.
- For reproducibility, set a random seed before running simulations.

---

## **Contact**

For questions or contributions, please open an issue or pull request on GitHub.

---

**Enjoy simulating your sales compensation plans!**