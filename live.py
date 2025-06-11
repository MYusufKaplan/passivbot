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


console = Console()

# Configurable point scheme
# elite_points_per_rank = {1: 20, 2: 12, 3: 8, 4: 4}
elite_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}
primary_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}
# primary_points_per_rank = {1: 10, 2: 6, 3: 4, 4: 2}
secondary_points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}

top_ranking_points = 100

# Define elite metrics
elite_metrics = [
    "adg", "gadg","gain"
]

# Define priority metrics
priority_metrics = [
    "mdg",  "loss_profit_ratio","position_held_hours_mean", "positions_held_per_day", "time_in_market_percent", "sharpe_ratio", "rsquared"
]

# Define lower-is-better metrics
lower_better_metrics = [
    "drawdown_worst", "drawdown_worst_mean_1pct", "equity_balance_diff_neg_max",
    "equity_balance_diff_neg_mean", "expected_shortfall_1pct",
    "loss_profit_ratio",
    "position_held_hours_mean", "position_unchanged_hours_max"
]

# Friendly names for columns
header_aliases = {
    "adg": "ADG",
    "adg_w": "ADG(w)",
    "gadg": "GADG",
    "gadg_w": "GADG(w)",
    "mdg": "MDG",
    "mdg_w": "MDG(w)",
    "gain": "Gain",
    "positions_held_per_day": "Pos/Day",
    "sharpe_ratio": "Sharpe",
    "sharpe_ratio_w": "Sharpe(w)",
    "drawdown_worst": "DD Worst",
    "drawdown_worst_mean_1pct": "DD 1%",
    "expected_shortfall_1pct": "ES 1%",
    "position_held_hours_mean": "Hrs/Pos",
    "position_unchanged_hours_max": "Unchg Max",
    "loss_profit_ratio": "LPR",
    "loss_profit_ratio_w": "LPR(w)",
    "calmar_ratio": "Calmar",
    "calmar_ratio_w": "Calmar(w)",
    "omega_ratio": "Omega",
    "omega_ratio_w": "Omega(w)",
    "sortino_ratio": "Sortino",
    "sortino_ratio_w": "Sortino(w)",
    "sterling_ratio": "Sterling",
    "sterling_ratio_w": "Sterling(w)",
    "rsquared": "R²",
    "time_in_market_percent": "TiM %"
}

def get_current_running_config():
    try:
        # Run the SSH command to get the service info
        result = subprocess.run(
            ['ssh', 'passivbot', 'sudo cat /etc/systemd/system/passivbot.service | grep configs'],
            capture_output=True, text=True, check=True
        )
        # Extract the config number
        match = re.search(r'configs/(\d+)\.json', result.stdout)
        if match:
            return int(match.group(1))
        return None
    except Exception as e:
        console.print(f"[red]Error getting running config: {e}[/red]")
        return None

current_config = get_current_running_config()

# Read JSON files
base_path = "backtests/optimizer/live"
rows = []
column_stats = {}

for root, dirs, files in os.walk(base_path):
    if "analysis.json" in files:
        file_path = os.path.join(root, "analysis.json")
        with open(file_path) as f:
            data = json.load(f)

        raw_name = os.path.basename(root).replace("_long", "")
        try:
            if float(raw_name) < 1:
                name = int(float(raw_name) * 1e8)
            else:
                name = int(raw_name)
        except ValueError:
            name = raw_name

        row = {"name": name}
        for key, val in data.items():
            row[key] = val
            if key not in column_stats:
                column_stats[key] = []
            column_stats[key].append((name, val))

        rows.append(row)

# Sort rows by ascending name
rows.sort(key=lambda x: x["name"])

# Compute min, max per column for color scaling
col_minmax = {}
for key, values in column_stats.items():
    vals = [v[1] for v in values]
    try:
        col_minmax[key] = (min(vals), max(vals))
    except Exception as e:
        for v in values:
            if v[1] is None:
                print(v[0], " is having issues")
        col_minmax[key] = (min(vals), max(vals))
        print(e)

# Function for color interpolation
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
        table.add_column(header_aliases.get(h, h))  # Use short alias for headers

    for row in selected_rows:
        name = row["name"]
        # Highlight if this is the current running config
        name_str = str(name)
        if current_config is not None and name == current_config:
            name_str = f"[bold yellow on blue]{name_str}[/]"
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


# Function to assign points based on relative distance to first place
def assign_points_based_on_distance_to_first_place(values, max_points=100, reverse=False):
    if not values:
        return

    values = sorted(values, key=lambda x: x[1], reverse=not reverse)

    best_value = values[0][1]
    worst_value = values[-1][1]
    range_of_values = best_value - worst_value

    for name, val in values:
        if range_of_values == 0:
            points = max_points
        else:
            if not reverse:
                distance = (best_value - val) / range_of_values
            else:
                distance = (val - best_value) / range_of_values
            points = max(0, max_points * (1 - distance))
        
        total_points[name] += points


# Track points
total_points = {row["name"]: 0 for row in rows}

# Compute points based on relative distance to first place for each metric
for key, values in column_stats.items():
    if "_w" in key:
        continue
    reverse = key in lower_better_metrics
    points_per_rank = top_ranking_points
    
    assign_points_based_on_distance_to_first_place(values, points_per_rank, reverse)
    # total_points_sorted = sorted(total_points.items(), key=lambda x: x[1], reverse=False)

    # # Print as table
    # print(tabulate(total_points_sorted, headers=["Key", "Value"], tablefmt="fancy_grid"))

# Sort final ranking by total points
total_points_sorted = sorted(total_points.items(), key=lambda x: x[1], reverse=True)
reverse_total_points_sorted = sorted(total_points.items(), key=lambda x: x[1], reverse=False)


# ✅ Delete directories with 0 score
for name, points in total_points_sorted:
    if points == 0:
        dir_name = str(name).replace("e+08", "")  # handle scientific notation case if needed
        target_dir = None

        for root, dirs, files in os.walk(base_path):
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

# Thematic metric groups
def build_thematic_tables():
    tables = []
    performance_metrics = ["adg", "gadg" ,"mdg", "gain"]
    risk_metrics = ["drawdown_worst", "drawdown_worst_mean_1pct", "expected_shortfall_1pct", "loss_profit_ratio", "rsquared"]
    position_metrics = ["position_held_hours_mean", "positions_held_per_day", "position_unchanged_hours_max", "time_in_market_percent"]
    ratio_metrics = ["sharpe_ratio", "calmar_ratio","omega_ratio", "sortino_ratio","sterling_ratio"]
    priority_table_metrics = elite_metrics + priority_metrics

    tables.append(build_colored_table(rows, performance_metrics, "📊 Performance Metrics"))
    tables.append(build_colored_table(rows, risk_metrics, "⚠️ Risk Metrics"))
    tables.append(build_colored_table(rows, position_metrics, "⏳ Position Holding Metrics"))
    tables.append(build_colored_table(rows, ratio_metrics, "📈 Ratio Metrics"))
    tables.append(build_colored_table(rows, priority_table_metrics, "⭐ Priority Metrics"))

    return tables

# Loop for interactivity
from rich.panel import Panel
from rich.live import Live
from rich.align import Align
from rich.layout import Layout

def interactive_display():
    tables = build_thematic_tables()
    
    while True:
        menu_panel = Panel.fit(
            Align.left(
                "[bold green]Choose a table to display:[/bold green]\n\n"
                "1 - Performance Metrics\n"
                "2 - Risk Metrics\n"
                "3 - Position Holding Metrics\n"
                "4 - Ratio Metrics\n"
                "5 - Priority Metrics\n"
                "6 - Final Points-Based Ranking\n"
                "7 - Top 3 Strategies (Priority Metrics)\n"
                "q - Quit"
            ),
            title="📋 Menu",
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
                name_str = str(name)
                if current_config is not None and name == current_config:
                    name_str = f"[bold yellow on blue]{name_str}[/]"
                ranking_table.add_row(name_str, str(points))

            panel = Panel(ranking_table, title="📊 Rankings", border_style="magenta")
            console.print(panel)

        elif choice == "7":
            top3_names = [name for name, _ in total_points_sorted[:3]]
            top3_rows = [row for row in rows if row["name"] in top3_names]
            top3_table = build_colored_table(top3_rows, elite_metrics + priority_metrics, "🌟 Top 3 Strategies (Priority Metrics)")
            panel = Panel(top3_table, title="🌟 Top 3 Strategies", border_style="green")
            console.print(panel)

        elif choice.lower() == "q":
            console.print(Panel("👋 Exiting... Bye!", style="bold red", border_style="red"))
            break
        else:
            console.print(Panel("[bold red]Invalid choice! Please enter a valid option.[/bold red]", border_style="red"))


console.clear()
interactive_display()
