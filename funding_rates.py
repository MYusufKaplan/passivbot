#!/usr/bin/env python3
import ccxt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from datetime import datetime, timedelta
import time

def load_api_keys():
    with open('api.key', 'r') as file:
        lines = file.readlines()
        api_key = lines[0].strip().split('=')[1]
        api_secret = lines[1].strip().split('=')[1]
    return api_key, api_secret

API_KEY, API_SECRET = load_api_keys()

gate = ccxt.gateio({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

console = Console()

def fetch_all_funding_rates():
    """Fetch funding rates for all perpetual contracts in the last month"""
    console.print("[bold cyan]🔍 Fetching all perpetual contracts...[/]")
    
    # Get all swap markets
    markets = gate.load_markets()
    swap_symbols = [symbol for symbol, market in markets.items() if market['swap']]
    
    console.print(f"[green]✓[/] Found {len(swap_symbols)} perpetual contracts")
    
    # Filter by launch time (at least 3 years old)
    console.print("[bold cyan]🔍 Filtering coins with at least 3 years of trading history...[/]")
    
    three_years_ago_ts = int((datetime.now() - timedelta(days=365*3)).timestamp())
    eligible_symbols = []
    
    for symbol in swap_symbols:
        market = markets[symbol]
        # Check launch_time or create_time from market info
        launch_time = market.get('info', {}).get('launch_time') or market.get('info', {}).get('create_time')
        
        if launch_time:
            try:
                launch_ts = int(launch_time)
                # If launched more than 3 years ago, include it
                if launch_ts <= three_years_ago_ts:
                    eligible_symbols.append(symbol)
            except:
                pass
    
    filtered_count = len(swap_symbols) - len(eligible_symbols)
    console.print(f"[green]✓[/] Found {len(eligible_symbols)} contracts with 3+ years history")
    console.print(f"[yellow]⚠️[/] Filtered out {filtered_count} contracts with less than 3 years of trading history")
    
    # Fetch funding rates for eligible symbols
    funding_data = {}
    
    from rich.progress import BarColumn, TaskProgressColumn, TimeRemainingColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Fetching funding rates...", total=len(eligible_symbols))
        
        for symbol in eligible_symbols:
            try:
                # Fetch funding rate history for last month
                since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
                funding_history = gate.fetch_funding_rate_history(symbol, since=since, limit=1000)
                
                if funding_history and len(funding_history) > 1:
                    # Sum all funding rates in the period
                    total_funding = sum(float(f['fundingRate']) * 100 for f in funding_history)  # Convert to percentage
                    
                    # Calculate funding frequency (time between samples in hours)
                    timestamps = [f['timestamp'] for f in funding_history]
                    time_diffs = [(timestamps[i+1] - timestamps[i]) / (1000 * 3600) for i in range(len(timestamps)-1)]
                    avg_frequency = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                    
                    funding_data[symbol] = {
                        'total_rate': total_funding,
                        'count': len(funding_history),
                        'avg_rate': total_funding / len(funding_history) if funding_history else 0,
                        'frequency_hours': avg_frequency
                    }
                
                progress.advance(task)
                
            except Exception as e:
                # Skip symbols that error out
                progress.advance(task)
                continue
    
    return funding_data

def display_funding_rates(funding_data):
    """Display funding rates in a sorted table"""
    if not funding_data:
        console.print("[red]❌ No funding rate data available[/]")
        return
    
    # Sort by total funding rate (least to most)
    sorted_data = sorted(funding_data.items(), key=lambda x: x[1]['total_rate'])
    
    # Create table
    table = Table(title="📊 30-Day Funding Rates (Sorted: Least → Most)", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6, justify="right")
    table.add_column("Symbol", style="cyan", width=20)
    table.add_column("Total 30d Rate", justify="right", width=18)
    table.add_column("Avg Rate", justify="right", width=15)
    table.add_column("Frequency", justify="right", width=12)
    table.add_column("Samples", justify="right", width=10)
    
    for idx, (symbol, data) in enumerate(sorted_data, 1):
        total_rate = data['total_rate']
        avg_rate = data['avg_rate']
        count = data['count']
        freq_hours = data['frequency_hours']
        
        # Color coding based on rate
        if total_rate < -0.5:
            rate_color = "bright_green"
            emoji = "💰"
        elif total_rate < 0:
            rate_color = "green"
            emoji = "✅"
        elif total_rate < 0.5:
            rate_color = "yellow"
            emoji = "⚠️"
        else:
            rate_color = "red"
            emoji = "🔥"
        
        # Clean symbol name
        clean_symbol = symbol.split(':')[0]
        
        # Format frequency
        freq_str = f"{freq_hours:.1f}h"
        
        table.add_row(
            f"{idx}",
            f"{emoji} {clean_symbol}",
            f"[{rate_color}]{total_rate:+.4f}%[/{rate_color}]",
            f"[{rate_color}]{avg_rate:+.4f}%[/{rate_color}]",
            freq_str,
            f"{count}"
        )
    
    console.print(table)
    
    # Summary statistics
    console.print("\n[bold]📈 Summary Statistics:[/]")
    total_rates = [d['total_rate'] for d in funding_data.values()]
    console.print(f"  • Most Negative (Best for Long): [bright_green]{min(total_rates):+.4f}%[/]")
    console.print(f"  • Most Positive (Best for Short): [red]{max(total_rates):+.4f}%[/]")
    console.print(f"  • Average: [yellow]{sum(total_rates)/len(total_rates):+.4f}%[/]")
    console.print(f"  • Total Contracts Analyzed: [cyan]{len(funding_data)}[/]")

def main():
    console.print("[bold]🚀 Gate.io 30-Day Funding Rate Analyzer[/]\n")
    
    start_time = time.time()
    funding_data = fetch_all_funding_rates()
    elapsed = time.time() - start_time
    
    console.print(f"\n[green]✓[/] Data fetched in {elapsed:.1f} seconds\n")
    
    display_funding_rates(funding_data)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Interrupted by user[/]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/]")
