#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import numpy as np
import os
import sys

def calculate_r2(actual, predicted):
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

def colorful_log(msg, emoji="✨"):
    print(f"{emoji} {msg}")

def plot_balance_equity(csv_file,run_number,start_time):
    colorful_log("📖 Reading CSV file...")
    df = pd.read_csv(csv_file)

    df.rename(columns={df.columns[0]: 'minutes'}, inplace=True)

    df['datetime'] = df['minutes'].apply(lambda x: start_time + timedelta(minutes=x))

    colorful_log("🖌️ Preparing the plots...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    STARTING_BALANCE = 1000
    df['balance'] = df['balance'] / STARTING_BALANCE
    df['equity'] = df['equity'] / STARTING_BALANCE

    # Linear scale plot
    axes[0].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[0].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{run_number} Balance & Equity (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True)

    # Log scale plot
    axes[1].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[1].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Value (Log Scale)")
    axes[1].set_title("Balance & Equity (Logarithmic Scale)")

    # Draw trend line for balance (straight line from (x0, y0) to (xn, yn))
    x_vals = df['minutes'].values
    balance_vals = df['balance'].values
    equity_vals = df['equity'].values

    # Trend line for log values
    log_balance = np.log(balance_vals)
    log_equity = np.log(equity_vals)

    # Generate straight line between first and last log values
    def get_trend_line(y_values, x_values):
        y0, yn = y_values[0], y_values[-1]
        x0, xn = x_values[0], x_values[-1]
        slope = (yn - y0) / (xn - x0)
        return y0 + slope * (x_values - x0)

    balance_trend = get_trend_line(log_balance, x_vals)
    equity_trend = get_trend_line(log_equity, x_vals)

    # Plot log-based trend lines on the linear scale chart
    axes[0].plot(df['datetime'], np.exp(equity_trend), color='red', linestyle='--', linewidth=1, label='Log Trend')

    # Plot trend line for balance on log scale (exponentiate back)
    axes[1].plot(df['datetime'], np.exp(equity_trend), color='red', linestyle='--', linewidth=1, label='Trend Line')

    axes[1].legend()
    axes[1].grid(True, which="both", ls="--")

    # Date formatting
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xlabel("Date")

    plt.tight_layout()
    colorful_log("📊 Displaying the plots!")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Balance and Equity from a CSV file.")
    parser.add_argument("csv_file_or_number", help="Path to the CSV file or run number (e.g. 48)")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date in YYYY-MM-DD format (default: 2023-01-01)")
    args = parser.parse_args()

    # Check if input is a number, then resolve to path
    input_arg = args.csv_file_or_number
    if input_arg.isdigit():
        colorful_log(f"🔍 Interpreting '{input_arg}' as run number...")
        run_number = str(input_arg).zfill(6)
        csv_file = f"/home/myusuf/Projects/passivbot/backtests/optimizer/live/{run_number}/balance_and_equity.csv"
        colorful_log(f"📂 Resolved path: {csv_file}")
    else:
        csv_file = input_arg
        run_number = "N/A"
        colorful_log(f"📂 Using direct CSV path: {csv_file}")

    # Check if file exists before proceeding
    if not os.path.isfile(csv_file):
        colorful_log(f"❌ File not found: {csv_file}", emoji="🚨")
        sys.exit(1)

    # Parse start date argument
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        colorful_log(f"❌ Invalid start date format: {args.start_date}. Use YYYY-MM-DD.", emoji="🚨")
        sys.exit(1)

    plot_balance_equity(csv_file, run_number, start_date)
