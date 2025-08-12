import os
import json
import re
import subprocess
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import shutil
from tabulate import tabulate
from time import sleep
import math

console = Console()

elite_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}
primary_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}
secondary_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}

# Define elite metrics
elite_metrics = ["adg", "gadg", "gain"]
# Define priority metrics
priority_metrics = ["mdg", "loss_profit_ratio", "position_held_hours_mean", "positions_held_per_day", "time_in_market_percent", "sharpe_ratio", "rsquared"]
# Define lower-is-better metrics
lower_better_metrics = [
    "drawdown_worst", "drawdown_worst_mean_1pct", "equity_balance_diff_neg_max",
    "equity_balance_diff_neg_mean", "expected_shortfall_1pct",
    "loss_profit_ratio", "position_held_hours_mean", "position_unchanged_hours_max"
]
others = ["n_positions", "total_wallet_exposure_limit"]

header_aliases = {
    "adg": "ADG", "adg_w": "ADG(w)", "gadg": "GADG", "gadg_w": "GADG(w)", "mdg": "MDG", "mdg_w": "MDG(w)",
    "gain": "Gain", "positions_held_per_day": "Pos/Day", "sharpe_ratio": "Sharpe", "sharpe_ratio_w": "Sharpe(w)",
    "drawdown_worst": "DD Worst", "drawdown_worst_mean_1pct": "DD 1%", "expected_shortfall_1pct": "ES 1%",
    "position_held_hours_mean": "Hrs/Pos", "position_unchanged_hours_max": "Unchg Max", "loss_profit_ratio": "LPR",
    "loss_profit_ratio_w": "LPR(w)", "calmar_ratio": "Calmar", "calmar_ratio_w": "Calmar(w)", "omega_ratio": "Omega",
    "omega_ratio_w": "Omega(w)", "sortino_ratio": "Sortino", "sortino_ratio_w": "Sortino(w)", "sterling_ratio": "Sterling",
    "sterling_ratio_w": "Sterling(w)", "rsquared": "R²", "time_in_market_percent": "TiM %", "n_positions": "NPos", "total_wallet_exposure_limit": "W Exp"
}

def get_current_running_config():
    try:
        result = subprocess.run(
            ['ssh', 'passivbot', 'sudo cat /etc/systemd/system/passivbot.service | grep configs'],
            capture_output=True, text=True, check=True
        )
        match = re.search(r'configs/(\d+)\.json', result.stdout)
        if match:
            return int(match.group(1))
        return None
    except Exception as e:
        console.print(f"[red]Error getting running config: {e}[/red]")
        return None

current_config = get_current_running_config()

base_path = "backtests/optimizer/live"
rows, column_stats = [], {}

def assign_points_by_rank(values, reverse=False):
    if not values:
        return
    # Filter out None values before sorting
    filtered_values = [(name, val) for name, val in values if val is not None]
    if not filtered_values:
        return
    sorted_vals = sorted(filtered_values, key=lambda x: x[1], reverse=not reverse)
    n = len(sorted_vals)
    for i, (name, _) in enumerate(sorted_vals):
        if name in ignore_names:
            continue
        points = n - i
        total_points[name] += points

def load_analysis_data():
    global rows, column_stats, col_minmax, total_points, total_points_sorted, reverse_total_points_sorted, ignore_names
    rows, column_stats = [], {}
    ignore_names = set()

    for root, dirs, files in os.walk(base_path):
        if "analysis.json" in files:
            file_path = os.path.join(root, "analysis.json")
            with open(file_path) as f:
                data = json.load(f)
            raw_name = os.path.basename(root).replace("_long", "")
            try:
                name = int(float(raw_name) * 1e8) if float(raw_name) < 1 else int(raw_name)
            except ValueError:
                name = raw_name
            row = {"name": name}
            for key, val in data.items():
                row[key] = val
                column_stats.setdefault(key, []).append((name, val))
            rows.append(row)
        if "config.json" in files:
            file_path = os.path.join(root, "config.json")
            with open(file_path) as f:
                data = json.load(f)
            row["n_positions"] = math.floor(data["bot"]["long"]["n_positions"])
            row["total_wallet_exposure_limit"] = data["bot"]["long"]["total_wallet_exposure_limit"]

    rows.sort(key=lambda x: x["name"])
    col_minmax = {}
    for k, v in column_stats.items():
        vv = [x[1] for x in v if x[1] is not None]
        if vv:
            col_minmax[k] = (min(vv), max(vv))
            # Calculate min/max for 'others' metrics which are added from config.json, not in column_stats
    for other_key in others:
        vals = [row[other_key] for row in rows if other_key in row]
        if vals:
            col_minmax[other_key] = (min(vals), max(vals))

    total_points = {row["name"]: 0 for row in rows}

    for name in total_points:
        if all(metric not in column_stats or not any(n == name for n, _ in column_stats[metric]) for metric in elite_metrics + priority_metrics):
            ignore_names.add(name)

    for key, values in column_stats.items():
        if "_w" in key:
            continue
        reverse = key in lower_better_metrics
        assign_points_by_rank(values, reverse)

    total_points_sorted = sorted(total_points.items(), key=lambda x: x[1], reverse=True)
    reverse_total_points_sorted = sorted(total_points.items(), key=lambda x: x[1])

def interpolate_color(val, min_val, max_val, reverse=False):
    if max_val == min_val:
        return "#FFFFFF"
    ratio = (val - min_val) / (max_val - min_val)
    if reverse:
        ratio = 1 - ratio
    r = int(255 * (1 - ratio) + 62 * ratio)
    g = int(255 * (1 - ratio) + 95 * ratio)
    b = int(255 * (1 - ratio) + 151 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"

def build_colored_table(selected_rows, headers, title):
    table = Table(show_header=True, header_style="bold green", title=title)
    table.add_column("Name")
    for h in headers:
        table.add_column(header_aliases.get(h, h))
    for row in selected_rows:
        name = row["name"]
        name_str = f"[bold yellow on blue]{name}[/]" if name == current_config else str(name)
        cells = [name_str]
        for key in headers:
            if key in row:
                val = row[key]
                min_val, max_val = col_minmax[key]
                reverse = key in lower_better_metrics
                color = "#FF0000" if val == (max_val if not reverse else min_val) else interpolate_color(val, min_val, max_val, reverse)
                cells.append(f"[{color}]{val:.6f}[/{color}]")
            else:
                cells.append("[#999999]N/A[/#999999]")
        table.add_row(*cells)
    return table
load_analysis_data()

for name, points in total_points.items():
    if points == 0:
        dir_name = str(name).replace("e+08", "")
        target_dir = None
        for root, dirs, _ in os.walk(base_path):
            for d in dirs:
                raw_name = d.replace("_long", "")
                try:
                    dir_number = int(float(raw_name) * 1e8) if float(raw_name) < 1 else int(raw_name)
                except ValueError:
                    dir_number = raw_name
                if dir_number == name:
                    target_dir = os.path.join(root, d)
                    break
            if target_dir:
                break
        if target_dir and os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            console.print(f"🗑️ [bold green]Deleted:[/] {target_dir}")
        else:
            console.print(f"❌ [bold red]Directory not found for:[/] {name}")

def build_thematic_tables():
    tables = []
    performance_metrics = ["adg", "gadg", "mdg", "gain"]
    risk_metrics = ["drawdown_worst", "drawdown_worst_mean_1pct", "expected_shortfall_1pct", "loss_profit_ratio", "rsquared"]
    position_metrics = ["position_held_hours_mean", "positions_held_per_day", "position_unchanged_hours_max", "time_in_market_percent"]
    ratio_metrics = ["sharpe_ratio", "calmar_ratio", "omega_ratio", "sortino_ratio", "sterling_ratio"]
    priority_table_metrics = elite_metrics + priority_metrics + others
    tables.append(build_colored_table(rows, performance_metrics, "📊 Performance Metrics"))
    tables.append(build_colored_table(rows, risk_metrics, "⚠️ Risk Metrics"))
    tables.append(build_colored_table(rows, position_metrics, "⏳ Position Holding Metrics"))
    tables.append(build_colored_table(rows, ratio_metrics, "📈 Ratio Metrics"))
    tables.append(build_colored_table(rows, priority_table_metrics, "⭐ Priority Metrics"))
    return tables

from rich.panel import Panel
from rich.prompt import Prompt
from rich.align import Align

def interactive_display():
    tables = build_thematic_tables()
    while True:
        menu_panel = Panel.fit(
            Align.left(
                "[bold cyan]Choose a table to display:[/bold cyan]\n\n"
                "📊 [bold yellow]1 -[/bold yellow] [green]Performance Metrics[/green]\n"
                "⚠️  [bold yellow]2 -[/bold yellow] [red]Risk Metrics[/red]\n"
                "⏳ [bold yellow]3 -[/bold yellow] [magenta]Position Holding Metrics[/magenta]\n"
                "📈 [bold yellow]4 -[/bold yellow] [blue]Ratio Metrics[/blue]\n"
                "⭐ [bold yellow]5 -[/bold yellow] [bold green]Priority Metrics[/bold green]\n"
                "🏆 [bold yellow]6 -[/bold yellow] [bold cyan]Final Points-Based Ranking[/bold cyan]\n"
                "🌟 [bold yellow]7 -[/bold yellow] [bold magenta]Top 3 Strategies (Priority Metrics)[/bold magenta]\n"
                "🔄 [bold yellow]8 -[/bold yellow] [bold white]Reload analysis.json files[/bold white]\n"
                "❌ [bold yellow]q -[/bold yellow] [red]Quit[/red]"
            ),
            title="📋 [bold blue]Menu[/bold blue]",
            border_style="blue"
        )

        console.print(menu_panel)
        choice = Prompt.ask("Enter choice", default="1")
        console.clear()

        if choice in {"1", "2", "3", "4", "5"}:
            table = tables[int(choice) - 1]
            console.print(table)
        elif choice == "6":
            ranking_table = Table(show_header=True, header_style="bold green", title="🏆 Final Points-Based Ranking")
            ranking_table.add_column("Name", style="bold yellow")
            ranking_table.add_column("Total Points", style="bold magenta")
            for name, points in reverse_total_points_sorted:
                name_str = f"[bold yellow on blue]{name}[/]" if current_config == name else str(name)
                ranking_table.add_row(name_str, str(points))
            console.print(Panel(ranking_table, title="📊 Rankings", border_style="magenta"))
        elif choice == "7":
            top3_names = [name for name, _ in total_points_sorted[:3]]
            top3_rows = [row for row in rows if row["name"] in top3_names]
            top3_table = build_colored_table(top3_rows, elite_metrics + priority_metrics, "🌟 Top 3 Strategies (Priority Metrics)")
            console.print(Panel(top3_table, title="🌟 Top 3 Strategies", border_style="green"))
        elif choice == "8":
            console.print("[yellow]🔄 Reloading analysis.json files...[/yellow]")
            load_analysis_data()
            tables = build_thematic_tables()
            console.print("[green]✅ Reloaded successfully.[/green]")
        elif choice.lower() == "q":
            console.print(Panel("👋 Exiting... Bye!", style="bold red", border_style="red"))
            break
        else:
            console.print(Panel("[bold red]Invalid choice! Please enter a valid option.[/bold red]", border_style="red"))

console.clear()
interactive_display()
