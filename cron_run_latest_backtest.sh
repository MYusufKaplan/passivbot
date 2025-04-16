#!/bin/bash

# Define base directory
BASEDIR="/home/myusuf/Projects/passivbot"

# Switch to project directory
cd "$BASEDIR" || { echo "❌ Failed to cd into $BASEDIR"; exit 1; }

# Activate virtual environment
source "$BASEDIR/.venv/bin/activate"

# Run the backtest script
./run_latest_backtest.sh >> "$BASEDIR/backtest_cron.log" 2>&1

# Optionally, print status
echo "✅ Backtest script executed at $(date)" >> "$BASEDIR/backtest_cron.log"

