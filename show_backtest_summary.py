import os
import json
from rich.console import Console
from rich.table import Table

console = Console()

# Configurable point scheme
points_per_rank = {1: 5, 2: 3, 3: 2, 4: 1}

# Read JSON files
base_path = "backtests/optimizer/combined"
rows = []
column_stats = {
    "adg": [],
    "adg_w": [],
    "mdg": [],
    "mdg_w": [],
    "gain": [],
    "lpr": [],
    "lpr_w": [],
    "pos_hrs": [],
    "pos_day": [],
    "sharpe": [],
    "sharpe_w": []
}

for root, dirs, files in os.walk(base_path):
    if "analysis.json" in files:
        file_path = os.path.join(root, "analysis.json")
        with open(file_path) as f:
            data = json.load(f)

        raw_name = os.path.basename(root).replace("_long", "")
        try:
            if name < 1:
                name = int(float(raw_name) * 1e8)
        except ValueError:
            name = raw_name

        row = (
            name,
            data["adg"],
            data["adg_w"],
            data["mdg"],
            data["mdg_w"],
            data["gain"],
            data["loss_profit_ratio"],
            data["loss_profit_ratio_w"],
            data["position_held_hours_mean"],
            data["positions_held_per_day"],
            data["sharpe_ratio"],
            data["sharpe_ratio_w"]
        )

        rows.append(row)
        for idx, key in enumerate(column_stats.keys(), start=1):
            column_stats[key].append((name, row[idx]))

# Sort rows by ascending name
rows.sort(key=lambda x: x[0])

# Compute min, max per column for color scaling
col_minmax = {}
for key, values in column_stats.items():
    vals = [v[1] for v in values]
    col_minmax[key] = (min(vals), max(vals))

# Function for color interpolation
def interpolate_color(val, min_val, max_val, reverse=False):
    if max_val == min_val:
        return "#FFFFFF"  # fallback to white
    ratio = (val - min_val) / (max_val - min_val)
    if reverse:
        ratio = 1 - ratio
    r = int(255 * (1 - ratio) + 62 * ratio)
    g = int(255 * (1 - ratio) + 95 * ratio)
    b = int(255 * (1 - ratio) + 151 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"

# Build a table with the given rows
def build_colored_table(selected_rows, title):
    table = Table(show_header=True, header_style="bold green", title=title)
    headers = ["Name", "adg", "adg_w", "mdg", "mdg_w", "gain",
               "lpr", "lpr_w", "pos_hrs", "pos_day", "sharpe", "sharpe_w"]
    for h in headers:
        table.add_column(h)

    for row in selected_rows:
        name = row[0]
        cells = [str(name)]
        for idx, key in enumerate(column_stats.keys(), start=1):
            val = row[idx]
            min_val, max_val = col_minmax[key]
            reverse = key in ["lpr", "lpr_w", "pos_hrs"]

            color = "#FF0000" if val == (max_val if not reverse else min_val) else interpolate_color(val, min_val, max_val, reverse)
            cells.append(f"[{color}]{val:.6f}[/{color}]")
        table.add_row(*cells)
    return table

# Track points
total_points = {name: 0 for name, *_ in rows}

# Compute rankings and points per metric
for col_idx, key in enumerate(column_stats.keys(), start=1):
    reverse = key in ["lpr", "lpr_w", "pos_hrs"]
    sorted_vals = sorted(column_stats[key], key=lambda x: x[1], reverse=not reverse)

    for rank, (name, val) in enumerate(sorted_vals, start=1):
        if rank in points_per_rank:
            total_points[name] += points_per_rank[rank]

# Sort final ranking by total points
sorted_total_points = sorted(total_points.items(), key=lambda x: x[1], reverse=True)

# Build and print main full table
main_table = build_colored_table(rows, "📊 Strategy Performance Table")
console.print(main_table)

# Build final ranking table
ranking_table = Table(show_header=True, header_style="bold green", title="🏆 Final Points-Based Ranking")
ranking_table.add_column("Name", style="bold yellow")
ranking_table.add_column("Total Points", style="bold magenta")

for name, points in sorted_total_points:
    ranking_table.add_row(str(name), str(points))

console.print(ranking_table)

# Build top 3 table
top3_names = [name for name, _ in sorted_total_points[:3]]
top3_rows = [row for row in rows if row[0] in top3_names]
top3_table = build_colored_table(top3_rows, "🌟 Top 3 Strategies")

console.print(top3_table)
