import pandas as pd
import numpy as np
import json
import os
import glob



def calculate_r_squared(filepath, debug=False):
    def log(message):
        if debug:
            print(message)

    log("📂 Loading CSV file...")
    df = pd.read_csv(f"{filepath}/balance_and_equity.csv")
    
    log("🧹 Cleaning data...")
    df = df.dropna()  # Remove rows with NaN values

    # Extract relevant columns
    log("🔍 Extracting columns...")
    x = df['Unnamed: 0'].values
    balance = df['balance'].values
    equity = df['equity'].values

    # Safety check for positive values
    if (balance <= 0).any() or (equity <= 0).any():
        raise ValueError("🚨 Balance and equity must be positive values for logarithmic computation.")

    # Apply logarithmic transform
    log("🧮 Applying logarithmic transformation...")
    log_balance = np.log(balance)
    log_equity = np.log(equity)

    # Function to create a straight line between first and last points
    def get_line_values(y_values, x_values):
        y0, yn = y_values[0], y_values[-1]
        x0, xn = x_values[0], x_values[-1]
        slope = (yn - y0) / (xn - x0)
        line = y0 + slope * (x_values - x0)
        return line

    log("📈 Calculating balance line...")
    balance_line = get_line_values(log_balance, x)

    log("📉 Calculating equity line...")
    equity_line = get_line_values(log_equity, x)

    # Function to calculate R²
    def calculate_r2(actual, predicted):
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    log("📊 Calculating R² values...")
    balance_r2 = calculate_r2(log_balance, balance_line)
    equity_r2 = calculate_r2(log_equity, equity_line)

    log(f"✅ Balance R²: {balance_r2:.6f}")
    log(f"✅ Equity R²: {equity_r2:.6f}")

    # Load or create analysis.json
    analysis_path = os.path.join(filepath, "analysis.json")
    if os.path.exists(analysis_path):
        log("📖 Loading existing analysis.json...")
        with open(analysis_path, "r") as f:
            analysis_data = json.load(f)
    else:
        log("📝 Creating new analysis.json...")
        analysis_data = {}

    # Update the values
    analysis_data["balance_r_squared"] = balance_r2
    analysis_data["equity_r_squared"] = equity_r2

    # Save back to JSON
    log("💾 Saving results to analysis.json...")
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=4)

    log("🎉 Done!")



# Example usage:
# Path pattern for all folders inside ./backtests/optimizer/combined/
folders = glob.glob("./backtests/optimizer/combined/*/")

# Run the existing function for each folder
for folder in folders:
    print(f"🚀 Processing: {folder}")
    calculate_r_squared(folder, debug=True)