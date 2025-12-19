#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import numpy as np
import os
import sys
import json

def calculate_r2(actual, predicted):
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

def colorful_log(msg, emoji="✨"):
    print(f"{emoji} {msg}")

def get_start_date_from_config(csv_file_path):
    """Extract start date from config.json in the same directory as the CSV file."""
    try:
        # Get the directory containing the CSV file
        csv_dir = os.path.dirname(csv_file_path)
        config_path = os.path.join(csv_dir, "config.json")
        
        if not os.path.isfile(config_path):
            colorful_log(f"⚠️  Config file not found: {config_path}", emoji="🚨")
            return None
            
        colorful_log(f"📖 Reading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract start_date from backtest section
        start_date_str = config.get('backtest', {}).get('start_date')
        
        if not start_date_str:
            colorful_log("⚠️  start_date not found in config.json", emoji="🚨")
            return None
            
        # Parse the date string
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        colorful_log(f"✅ Found start date in config: {start_date_str}")
        return start_date
        
    except Exception as e:
        colorful_log(f"❌ Error reading config file: {e}", emoji="🚨")
        return None

def find_milestone_crossings(df, start_time):
    """Find when equity crosses 10^x milestones (2x, 5x, 10x, 20x, 50x, 100x, etc.)"""
    colorful_log("🎯 Analyzing milestone crossings...")
    
    # Define milestones to check for
    milestones = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    crossings = []
    
    for milestone in milestones:
        # Find first crossing of this milestone using pandas filtering
        crossing_rows = df[df['equity'] >= milestone]
        
        if not crossing_rows.empty:
            first_crossing = crossing_rows.iloc[0]
            crossing_date = first_crossing['datetime']
            crossing_value = first_crossing['equity']
            
            # Convert pandas Timestamp to datetime if needed
            if hasattr(crossing_date, 'to_pydatetime'):
                crossing_date = crossing_date.to_pydatetime()
            
            # Calculate days from start
            days_from_start = (crossing_date - start_time).days
            
            crossings.append({
                'milestone': milestone,
                'date': crossing_date,
                'value': crossing_value,
                'days': days_from_start
            })
    
    # Print the crossings
    if crossings:
        colorful_log("🚀 Milestone Crossings:")
        for crossing in crossings:
            milestone_str = f"{crossing['milestone']}x" if crossing['milestone'] >= 10 else f"{crossing['milestone']}x"
            colorful_log(f"   {milestone_str:>6} reached on {crossing['date'].strftime('%Y-%m-%d')} "
                        f"(day {crossing['days']:>4}) at {crossing['value']:.2f}x", emoji="📈")
    else:
        colorful_log("📊 No major milestones crossed in this timeframe")
    
    return crossings

def plot_balance_equity(csv_file, run_number, start_time, output_file=None, save_only=False):
    colorful_log("📖 Reading CSV file...")
    df = pd.read_csv(csv_file)

    df.rename(columns={df.columns[0]: 'minutes'}, inplace=True)

    df['datetime'] = df['minutes'].apply(lambda x: start_time + timedelta(minutes=x))
    
    # Calculate and print the end date
    end_time = df['datetime'].iloc[-1]
    colorful_log(f"📅 Start date: {start_time.strftime('%Y-%m-%d')}")
    colorful_log(f"📅 End date: {end_time.strftime('%Y-%m-%d')}")
    colorful_log(f"⏱️  Duration: {(end_time - start_time).days} days")

    colorful_log("🖌️ Preparing the plots...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    STARTING_BALANCE = 10000
    df['balance'] = df['balance'] / STARTING_BALANCE
    df['equity'] = df['equity'] / STARTING_BALANCE

    # Find and print milestone crossings
    find_milestone_crossings(df, start_time)

    colorful_log("🖌️ Preparing the plots...")

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

    # Set x-axis limits to match the actual data range
    axes[0].set_xlim(start_time, end_time)
    axes[1].set_xlim(start_time, end_time)
    
    # Date formatting
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xlabel("Date")

    plt.tight_layout()
    
    if save_only or output_file:
        # Save mode
        if not output_file:
            # Generate default filename
            if run_number != "N/A":
                output_file = f"balance_equity_plot_{run_number}.png"
            else:
                output_file = "balance_equity_plot.png"
        
        colorful_log(f"💾 Saving plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        colorful_log(f"✅ Plot saved as {output_file}")
        plt.close()
    else:
        # Display mode (default)
        try:
            colorful_log("📊 Displaying the plot!")
            plt.show()
        except Exception as e:
            colorful_log(f"⚠️  Cannot display plot (no GUI available): {e}")
            # Fallback to saving
            if run_number != "N/A":
                output_file = f"balance_equity_plot_{run_number}.png"
            else:
                output_file = "balance_equity_plot.png"
            colorful_log(f"💾 Falling back to saving plot as {output_file}")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            colorful_log(f"✅ Plot saved as {output_file}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Balance and Equity from a CSV file.")
    parser.add_argument("csv_file_or_number", help="Path to the CSV file or run number (e.g. 48)")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format (overrides config.json if provided)")
    parser.add_argument("--output", "-o", help="Output filename for the plot (default: auto-generated)")
    parser.add_argument("--save", action="store_true", help="Save plot to file instead of displaying")
    args = parser.parse_args()

    # Check if input is a number, then resolve to path
    input_arg = args.csv_file_or_number
    if input_arg.isdigit():
        colorful_log(f"🔍 Interpreting '{input_arg}' as run number...")
        run_number = str(input_arg).zfill(6)
        csv_file = f"/home/myusuf/Projects/passivbot/backtests/optimizer/live/{run_number}/balance_and_equity.csv"
        if not os.path.isfile(csv_file):
            csv_file = f"/home/myusuf/Projects/passivbot/backtests/optimizer/extremes/{run_number}/balance_and_equity.csv"
        colorful_log(f"📂 Resolved path: {csv_file}")
    else:
        csv_file = input_arg
        run_number = "N/A"
        colorful_log(f"📂 Using direct CSV path: {csv_file}")

    # Check if file exists before proceeding
    if not os.path.isfile(csv_file):
        colorful_log(f"❌ File not found: {csv_file}", emoji="🚨")
        sys.exit(1)

    # Determine start date: use command line arg if provided, otherwise read from config.json
    start_date = None
    
    if args.start_date:
        # Use command line argument if provided
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            colorful_log(f"📅 Using start date from command line: {args.start_date}")
        except ValueError:
            colorful_log(f"❌ Invalid start date format: {args.start_date}. Use YYYY-MM-DD.", emoji="🚨")
            sys.exit(1)
    else:
        # Try to read from config.json
        start_date = get_start_date_from_config(csv_file)
        
        if start_date is None:
            # Fallback to default
            start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
            colorful_log(f"⚠️  Using fallback start date: 2023-01-01", emoji="🚨")

    plot_balance_equity(csv_file, run_number, start_date, args.output, args.save)
