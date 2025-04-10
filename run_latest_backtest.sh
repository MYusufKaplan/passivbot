#!/bin/bash

# Define BASEDIR for easier path handling
BASEDIR="/home/myusuf/Projects/passivbot"

# Step 1: Find the latest file in optimize_results/
latest_optimize_result=$(ls -t "$BASEDIR/optimize_results/" | head -n 1)
echo "Latest optimize result: $latest_optimize_result"

# Run extract_best_config.py
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/tools/extract_best_config.py" "$BASEDIR/optimize_results/$latest_optimize_result"

# Step 2: Find the latest .json file in optimize_results_analysis/
latest_analysis_json=$(ls -t "$BASEDIR/optimize_results_analysis/"*.json | head -n 1)
echo "Latest analysis JSON: $latest_analysis_json"

# Run backtest.py
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$latest_analysis_json"

# Step 3: Find the latest directory in ./backtests/combined/
latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/combined/"*/ | head -n 1)
echo "Latest backtest directory: $latest_backtest_dir"

# Step 4: cd into the latest backtest directory
cd "$latest_backtest_dir" || { echo "Failed to cd into $latest_backtest_dir"; exit 1; }

# Step 5: Open the PNG from within the directory
png_file="balance_and_equity.png"
echo "Opening: $png_file"

# Open with default image viewer (Linux/macOS support)
eog "$png_file" 2>/dev/null &

