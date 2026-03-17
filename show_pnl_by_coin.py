#!/usr/bin/env python3
"""
Script to analyze backtest fills and show total PnL by coin.
Usage: python show_pnl_by_coin.py <backtest_id>
"""

import sys
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
import requests


def load_fills(backtest_id: str) -> pd.DataFrame:
    """Load fills.csv for the given backtest ID."""
    fills_path = Path(f"backtests/optimizer/live/{backtest_id}/fills.csv")
    
    if not fills_path.exists():
        raise FileNotFoundError(f"Fills file not found: {fills_path}")
    
    return pd.read_csv(fills_path)


def get_backtest_duration(backtest_id: str) -> int:
    """Get total backtest duration in minutes from balance_and_equity.csv."""
    balance_path = Path(f"backtests/optimizer/live/{backtest_id}/balance_and_equity.csv")
    
    if not balance_path.exists():
        raise FileNotFoundError(f"Balance file not found: {balance_path}")
    
    df = pd.read_csv(balance_path)
    # Get the first column value of the last row (total minutes)
    total_minutes = df.iloc[-1, 0]
    return int(total_minutes)


def format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable format."""
    years = minutes // (365 * 24 * 60)
    minutes %= (365 * 24 * 60)
    
    months = minutes // (30 * 24 * 60)
    minutes %= (30 * 24 * 60)
    
    days = minutes // (24 * 60)
    minutes %= (24 * 60)
    
    hours = minutes // 60
    mins = minutes % 60
    
    parts = []
    if years > 0:
        parts.append(f"{years}y")
    if months > 0:
        parts.append(f"{months}mo")
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0:
        parts.append(f"{mins}m")
    
    return " ".join(parts) if parts else "0m"


def get_usdt_try_rate() -> float:
    """Fetch current USDT/TRY rate from Binance."""
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=USDTTRY"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Warning: Could not fetch USDT/TRY rate: {e}")
        return None


def calculate_pnl_by_coin(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total PnL grouped by coin, sorted descending."""
    pnl_by_coin = df.groupby('coin')['pnl'].sum().reset_index()
    pnl_by_coin.columns = ['Coin', 'Total PnL']
    pnl_by_coin = pnl_by_coin.sort_values('Total PnL', ascending=False)
    return pnl_by_coin


def display_pnl_table(pnl_df: pd.DataFrame, backtest_id: str, duration_minutes: int, usdt_try_rate: float = None):
    """Display PnL data in a rich table."""
    console = Console()
    
    # Calculate time-based PnL rates
    total_pnl = pnl_df['Total PnL'].sum()
    
    pnl_per_second = total_pnl / (duration_minutes * 60)
    pnl_per_minute = total_pnl / duration_minutes
    pnl_per_hour = total_pnl / (duration_minutes / 60)
    pnl_per_day = total_pnl / (duration_minutes / (24 * 60))
    pnl_per_month = total_pnl / (duration_minutes / (30 * 24 * 60))
    pnl_per_year = total_pnl / (duration_minutes / (365 * 24 * 60))
    
    duration_str = format_duration(duration_minutes)
    
    # Build title with exchange rate if available
    title = f"PnL by Coin - Backtest {backtest_id}\nDuration: {duration_str}"
    if usdt_try_rate:
        title += f"\nUSDT/TRY: {usdt_try_rate:.2f}"
    
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Coin", style="cyan", justify="left")
    table.add_column("Total PnL (USDT)", style="green", justify="right")
    if usdt_try_rate:
        table.add_column("Total PnL (TRY)", style="yellow", justify="right")
    
    for _, row in pnl_df.iterrows():
        coin = row['Coin']
        pnl = row['Total PnL']
        
        # Color code based on positive/negative PnL
        pnl_style = "green" if pnl >= 0 else "red"
        
        if usdt_try_rate:
            pnl_try = pnl * usdt_try_rate
            table.add_row(
                coin,
                f"[{pnl_style}]{pnl:.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pnl_try:,.2f}[/{pnl_style}]"
            )
        else:
            table.add_row(coin, f"[{pnl_style}]{pnl:.2f}[/{pnl_style}]")
    
    # Add total row
    table.add_section()
    total_style = "bold green" if total_pnl >= 0 else "bold red"
    if usdt_try_rate:
        total_pnl_try = total_pnl * usdt_try_rate
        table.add_row(
            "TOTAL",
            f"[{total_style}]{total_pnl:.2f}[/{total_style}]",
            f"[{total_style}]{total_pnl_try:,.2f}[/{total_style}]"
        )
    else:
        table.add_row("TOTAL", f"[{total_style}]{total_pnl:.2f}[/{total_style}]")
    
    console.print(table)
    
    # Display PnL rates
    console.print("\n[bold cyan]PnL Rates:[/bold cyan]")
    rate_table = Table(show_header=False, box=None)
    rate_table.add_column("Period", style="cyan")
    rate_table.add_column("USDT", style="green", justify="right")
    if usdt_try_rate:
        rate_table.add_column("TRY", style="yellow", justify="right")
    
    if usdt_try_rate:
        rate_table.add_row("Per Second:", f"{pnl_per_second:.6f}", f"{pnl_per_second * usdt_try_rate:,.6f}")
        rate_table.add_row("Per Minute:", f"{pnl_per_minute:.4f}", f"{pnl_per_minute * usdt_try_rate:,.4f}")
        rate_table.add_row("Per Hour:", f"{pnl_per_hour:.2f}", f"{pnl_per_hour * usdt_try_rate:,.2f}")
        rate_table.add_row("Per Day:", f"{pnl_per_day:.2f}", f"{pnl_per_day * usdt_try_rate:,.2f}")
        rate_table.add_row("Per Month:", f"{pnl_per_month:.2f}", f"{pnl_per_month * usdt_try_rate:,.2f}")
        rate_table.add_row("Per Year:", f"{pnl_per_year:.2f}", f"{pnl_per_year * usdt_try_rate:,.2f}")
    else:
        rate_table.add_row("Per Second:", f"{pnl_per_second:.6f}")
        rate_table.add_row("Per Minute:", f"{pnl_per_minute:.4f}")
        rate_table.add_row("Per Hour:", f"{pnl_per_hour:.2f}")
        rate_table.add_row("Per Day:", f"{pnl_per_day:.2f}")
        rate_table.add_row("Per Month:", f"{pnl_per_month:.2f}")
        rate_table.add_row("Per Year:", f"{pnl_per_year:.2f}")
    
    console.print(rate_table)


def main():
    if len(sys.argv) != 2:
        print("Usage: python show_pnl_by_coin.py <backtest_id>")
        print("Example: python show_pnl_by_coin.py 2512")
        sys.exit(1)
    
    # Pad backtest_id to 6 digits with leading zeros
    backtest_id = sys.argv[1].zfill(6)
    
    try:
        # Load fills data
        df = load_fills(backtest_id)
        
        # Get backtest duration
        duration_minutes = get_backtest_duration(backtest_id)
        
        # Get USDT/TRY exchange rate
        usdt_try_rate = get_usdt_try_rate()
        
        # Calculate PnL by coin
        pnl_df = calculate_pnl_by_coin(df)
        
        # Display results
        display_pnl_table(pnl_df, backtest_id, duration_minutes, usdt_try_rate)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
