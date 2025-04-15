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

    # Rename the unnamed column to 'minutes'
    df.rename(columns={df.columns[0]: 'minutes'}, inplace=True)

    # Convert 'minutes' to datetime starting from 2020-01-01 00:00 UTC
    start_time = datetime(2020, 1, 1)
    df['datetime'] = df['minutes'].apply(lambda x: start_time + timedelta(minutes=x))

    colorful_log("Preparing the plots...", "🖌️")

    # Set up the plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Linear scale plot
    axes[0].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[0].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[0].set_ylabel("Value")
    axes[0].set_title("Balance & Equity (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True)

    # Logarithmic scale plot
    axes[1].plot(df['datetime'], df['balance'], label='Balance', color='blue')
    axes[1].plot(df['datetime'], df['equity'], label='Equity', color='green')
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Value (Log Scale)")
    axes[1].set_title("Balance & Equity (Logarithmic Scale)")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--")

    # Format date axis
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
