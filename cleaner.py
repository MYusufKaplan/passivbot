import os
import json
import shutil
import argparse
from rich.console import Console
from rich.prompt import Prompt

console = Console()

# Argument parser setup
parser = argparse.ArgumentParser(description="🧹 Clean up backtest directories based on 'gain' value.")
parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without actually deleting")
args = parser.parse_args()

# Base path for backtest results
base_path = "backtests/optimizer/combined"

# Prompt user for minimum gain threshold
min_gain = Prompt.ask("[bold green]Enter minimum acceptable gain[/] (as a percentage)", default="90")
try:
    min_gain = float(min_gain)
except ValueError:
    console.print("[bold red]Invalid input. Please enter a numeric value.[/bold red]")
    exit(1)

# Start cleaning process
deleted_count = 0
kept_count = 0

for root, dirs, files in os.walk(base_path):
    if "analysis.json" in files:
        file_path = os.path.join(root, "analysis.json")
        with open(file_path) as f:
            data = json.load(f)

        gain = data.get("gain", None)

        if gain is not None:
            if gain < min_gain:
                if args.dry_run:
                    console.print(f"📝 [bold cyan]Would delete:[/] {root} (Gain: {gain:.2f})")
                else:
                    shutil.rmtree(root)
                    console.print(f"🗑️ [bold green]Deleted:[/] {root} (Gain: {gain:.2f})")
                    deleted_count += 1
            else:
                console.print(f"✅ [bold yellow]Kept:[/] {root} (Gain: {gain:.2f})")
                kept_count += 1
        else:
            console.print(f"[bold red]⚠️ No 'gain' key found in:[/] {file_path}")

# Summary
console.print("\n[bold blue]Cleanup complete![/]")
if args.dry_run:
    console.print(f"[bold cyan]This was a dry run — no directories were actually deleted.[/]")
else:
    console.print(f"Deleted: [bold red]{deleted_count}[/] directories")
console.print(f"Kept: [bold green]{kept_count}[/] directories")
