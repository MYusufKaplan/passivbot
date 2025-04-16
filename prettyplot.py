#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import timedelta, datetime
import matplotlib.dates as mdates

def colorful_log(msg, emoji="✨"):
    print(f"{emoji} {msg}")

def plot_balance_equity(csv_file):
    colorful_log("Reading CSV file...", "📖")
    df = pd.read_csv(csv_file)

    df.rename(columns={df.columns[1]: 'minutes'}, inplace=True)

    start_time = datetime(2023, 1, 1)
    df['datetime'] = df['minutes'].apply(lambda x: start_time + timedelta(minutes=x))

    colorful_log("Preparing the plots...", "🖌️")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[0].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[0].set_ylabel("Value")
    axes[0].set_title("Balance & Equity (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[1].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Value (Log Scale)")
    axes[1].set_title("Balance & Equity (Logarithmic Scale)")

    # Draw straight line from (x0, y0) to (xn, yn)
    x0, y0 = df['datetime'].iloc[0], df['balance'].iloc[0]
    xn, yn = df['datetime'].iloc[-1], df['balance'].iloc[-1]
    axes[1].plot([x0, xn], [y0, yn], color='red', linestyle='--', linewidth=1, label='Trend Line')

    axes[1].legend()
    axes[1].grid(True, which="both", ls="--")

    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xlabel("Date")
    plt.tight_layout()
    colorful_log("Displaying the plots!", "📊")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Balance and Equity from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()

    plot_balance_equity(args.csv_file)
