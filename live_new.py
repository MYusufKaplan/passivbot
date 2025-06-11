import os
import json
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.align import Align

# 🎨 Color and console setup
console = Console()

def interpolate_color(val, min_val, max_val, reverse=False):
    if max_val == min_val:
        return "#cccccc"
    ratio = (val - min_val) / (max_val - min_val)
    if reverse:
        ratio = 1 - ratio
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    return f"#{r:02x}{g:02x}00"

# 🏁 Config
top_ranking_points = 1.0
metrics = [
    "gain",
    "drawdown_worst",
    "calmar_ratio",
    "rsquared",
    "time_in_market_percent"
]
lower_better_metrics = ["drawdown_worst"]

header_aliases = {
    "gain": "Gain",
    "drawdown_worst": "DD Worst",
    "calmar_ratio": "Calmar",
    "rsquared": "R²",
    "time_in_market_percent": "TiM %"
}

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

# 🎯 Assign points
def assign_points_based_on_distance_to_first_place(values, points_for_top, reverse=False):
    if not values:
        return
    sorted_vals = sorted(values, key=lambda x: x[1], reverse=not reverse)
    best_val = sorted_vals[0][1]
    worst_val = sorted_vals[-1][1]
    for name, val in values:
        if best_val == worst_val:
            score = points_for_top
        else:
            score = points_for_top * (1 - abs(val - best_val) / abs(worst_val - best_val))
        total_points[name] += score

# 📊 Score each row
total_points = {row["name"]: 0 for row in rows}
col_minmax = {}

for key in metrics:
    values = column_stats.get(key, [])
    if not values:
        continue
    vals = [v[1] for v in values]
    col_minmax[key] = (min(vals), max(vals))
    reverse = key in lower_better_metrics
    assign_points_based_on_distance_to_first_place(values, top_ranking_points, reverse)

# 🧱 Table builder
def build_colored_table(selected_rows, headers, title):
    table = Table(show_header=True, header_style="bold green", title=title)
    table.add_column("Name")
    for h in headers:
        table.add_column(header_aliases.get(h, h))
    for row in selected_rows:
        name_str = f"[bold cyan]{row['name']}[/bold cyan]"
        cells = [name_str]
        for key in headers:
            val = row.get(key)
            if val is None:
                cells.append("[#999999]N/A[/#999999]")
                continue
            min_val, max_val = col_minmax.get(key, (val, val))
            reverse = key in lower_better_metrics
            color = interpolate_color(val, min_val, max_val, reverse)
            cells.append(f"[{color}]{val:.6f}[/{color}]")
        table.add_row(*cells)
    return table

# 📋 Display menu
def interactive_display():
    while True:
        menu_panel = Panel.fit(
            Align.left(
                "[bold green]Choose a table to display:[/bold green]\n\n"
                "1 - 📈 Metric Table\n"
                "2 - 🏆 Points Ranking\n"
                "q - ❌ Quit"
            ),
            title="📋 Menu",
            border_style="blue"
        )
        console.print(menu_panel)
        choice = Prompt.ask("Enter choice", default="1")
        console.clear()

        if choice == "1":
            table = build_colored_table(rows, metrics, "📊 Strategy Metrics")
            console.print(table)
        elif choice == "2":
            ranking_table = Table(show_header=True, header_style="bold green", title="🏆 Final Points Ranking")
            ranking_table.add_column("Name", style="bold yellow")
            ranking_table.add_column("Total Points", style="bold magenta")

            for name, points in sorted(total_points.items(), key=lambda x: x[1], reverse=True):
                ranking_table.add_row(f"{name}", f"{points:.2f}")
            console.print(Panel(ranking_table, title="📊 Final Rankings", border_style="magenta"))
        elif choice.lower() == "q":
            console.print(Panel("👋 Exiting... Bye!", style="bold red", border_style="red"))
            break
        else:
            console.print(Panel("[bold red]Invalid choice! Try again.[/bold red]", border_style="red"))

# 🟢 Run
if __name__ == "__main__":
    interactive_display()
