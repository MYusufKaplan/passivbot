#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

spinner() {
  local pid=$1
  local delay=0.15
  local spinstr=('🌑' '🌒' '🌓' '🌔' '🌕' '🌖' '🌗' '🌘')
  local i=0
  local start_time=$(date +%s)

  tput civis
  while kill -0 "$pid" 2>/dev/null; do
    local now=$(date +%s)
    local elapsed=$((now - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    printf "\r${CYAN}⏳ %s %02d:%02d${NC}" "${spinstr[$i]}" $mins $secs
    i=$(( (i+1) % ${#spinstr[@]} ))
    sleep $delay
  done
  printf "\r${GREEN}✅  Process completed in %02d:%02d! 🎉 ${NC}\n" $mins $secs
  tput cnorm
}

START_DATE="${1:-2023-01-01}"

BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"

config_dir="$BASEDIR/configs/optimization"
last_id=$(find "$config_dir" -type f -name "*.json" | sed 's/.*\///;s/\.json//' | sort | tail -n 1)
if [ -z "$last_id" ]; then
  FITNESS="000001"
else
  next_id=$((10#$last_id + 1))
  FITNESS=$(printf "%06d" "$next_id")
fi

echo -e "${CYAN}🏋️‍♂️  FITNESS ID assigned: ${YELLOW}$FITNESS${NC} ✨"
echo -e "${CYAN}📅 Start date: ${YELLOW}$START_DATE${NC} 🗓️"

latest_optimize_result=$(ls -t "$BASEDIR/optimize_results/" | head -n 1)
echo -e "${CYAN}📄 Latest optimize result file: ${YELLOW}$latest_optimize_result${NC} 📄"

trimmed_optimize_result="$BASEDIR/optimize_results/finalists.txt"
first_line=$(head -n 1 "$BASEDIR/optimize_results/$latest_optimize_result")
tail -n 5000 "$BASEDIR/optimize_results/$latest_optimize_result" > "$trimmed_optimize_result"
echo "$first_line" | cat - "$trimmed_optimize_result" > temp && mv temp "$trimmed_optimize_result"
echo -e "${CYAN}✂️ Trimmed optimize result created: ${YELLOW}$trimmed_optimize_result${NC} ✂️"

echo -e "${GREEN}⚙️ Running extract_best_config.py... 🚀 ${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/tools/extract_best_config.py" "$trimmed_optimize_result" &
spinner $!

latest_analysis_json=$(ls -t "$BASEDIR/optimize_results_analysis/"*.json | head -n 1)
echo -e "${CYAN}📝 Latest analysis JSON file: ${YELLOW}$latest_analysis_json${NC} 📝"

latest_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$BASEDIR/optimize_results_analysis/finalists.txt.json")

for config_file in "$BASEDIR/configs/optimization"/*.json; do
  config_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$config_file")
  if [ "$latest_values" == "$config_values" ]; then
    echo -e "${RED}🚫 Duplicate values found! 🚫 Aborting... 💥 ${NC}"
    exit 1
  fi
done

echo -e "${GREEN}✅ No duplicates! Continuing... 👍 ${NC}"

echo -e "${GREEN}🛠️ Setting backtest start_date to $START_DATE... 🛠️ ${NC}"
tmp_json="${latest_analysis_json}.tmp"
jq --arg date "$START_DATE" '.backtest.start_date = $date' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ backtest start_date updated ✅ ${NC}"

echo -e "${GREEN}🛠️ Setting backtest end_date to 2025-04-14... ⏱️ ${NC}"
tmp_json="${latest_analysis_json}.tmp"
jq --arg date "2025-04-14" '.backtest.end_date = $date' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ backtest end_date updated ✅ ${NC}"

echo -e "${GREEN}🚀 Running backtest.py... 🚀 ${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$latest_analysis_json" &
spinner $!

latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/optimizer/combined/"*/ | head -n 1)
echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC} 📊"

config_dest="$config_dir/${FITNESS}.json"
cp "$latest_backtest_dir/config.json" "$config_dest"
echo -e "${GREEN}📦 config.json copied to ${YELLOW}$config_dest${NC} 📦"

long_dir="$BASEDIR/backtests/optimizer/combined/${FITNESS}"
mv "$latest_backtest_dir" "$long_dir"
echo -e "${GREEN}📁 Backtest directory renamed to ${YELLOW}$long_dir${NC} 📁"

echo -e "${GREEN}📝 Updating CSV files with datetime column... ⏳ ${NC}"

"$BASEDIR/.venv/bin/python3" - <<EOF
import pandas as pd
from datetime import datetime, timedelta
import json

long_dir = "$long_dir"

with open(f"{long_dir}/config.json") as f:
    start_time_str = json.load(f)["backtest"]["start_date"]

start_time = datetime.fromisoformat(start_time_str)

def process_balance_and_equity(file_path):
    df = pd.read_csv(file_path)
    minutes_col = df.columns[0]
    df['datetime'] = df[minutes_col].apply(lambda x: start_time + timedelta(minutes=x))
    cols = ['datetime'] + df.columns[:-1].tolist()
    df = df[cols]
    df.to_csv(file_path, index=False)

def process_fills(file_path):
    df = pd.read_csv(file_path)
    if 'minute' in df.columns:
        df['datetime'] = df['minute'].apply(lambda x: start_time + timedelta(minutes=x))
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        df.to_csv(file_path, index=False)

process_balance_and_equity(f"{long_dir}/balance_and_equity.csv")
process_fills(f"{long_dir}/fills.csv")

print("✅ CSV files updated successfully.")

# import pandas as pd
# import numpy as np
# import json
# import os

# def calculate_r_squared(filepath, debug=False):
#     def log(message):
#         if debug:
#             print(message)

#     log("📂 Loading CSV file...")
#     df = pd.read_csv(f"{filepath}/balance_and_equity.csv")
    
#     log("🧹 Cleaning data...")
#     df = df.dropna()  # Remove rows with NaN values

#     # Extract relevant columns
#     log("🔍 Extracting columns...")
#     x = df['Unnamed: 0'].values
#     balance = df['balance'].values
#     equity = df['equity'].values

#     # Safety check for positive values
#     if (balance <= 0).any() or (equity <= 0).any():
#         raise ValueError("🚨 Balance and equity must be positive values for logarithmic computation.")

#     # Apply logarithmic transform
#     log("🧮 Applying logarithmic transformation...")
#     log_balance = np.log(balance)
#     log_equity = np.log(equity)

#     # Function to create a straight line between first and last points
#     def get_line_values(y_values, x_values):
#         y0, yn = y_values[0], y_values[-1]
#         x0, xn = x_values[0], x_values[-1]
#         slope = (yn - y0) / (xn - x0)
#         line = y0 + slope * (x_values - x0)
#         return line

#     log("📈 Calculating balance line...")
#     balance_line = get_line_values(log_balance, x)

#     log("📉 Calculating equity line...")
#     equity_line = get_line_values(log_equity, x)

#     # Function to calculate R²
#     def calculate_r2(actual, predicted):
#         ss_res = np.sum((actual - predicted) ** 2)
#         ss_tot = np.sum((actual - np.mean(actual)) ** 2)
#         r2 = 1 - (ss_res / ss_tot)
#         return r2

#     log("📊 Calculating R² values...")
#     balance_r2 = calculate_r2(log_balance, balance_line)
#     equity_r2 = calculate_r2(log_equity, equity_line)

#     log(f"✅ Balance R²: {balance_r2:.6f}")
#     log(f"✅ Equity R²: {equity_r2:.6f}")

#     # Load or create analysis.json
#     analysis_path = os.path.join(filepath, "analysis.json")
#     if os.path.exists(analysis_path):
#         log("📖 Loading existing analysis.json...")
#         with open(analysis_path, "r") as f:
#             analysis_data = json.load(f)
#     else:
#         log("📝 Creating new analysis.json...")
#         analysis_data = {}

#     # Update the values
#     analysis_data["balance_r_squared"] = balance_r2
#     analysis_data["equity_r_squared"] = equity_r2

#     # Save back to JSON
#     log("💾 Saving results to analysis.json...")
#     with open(analysis_path, "w") as f:
#         json.dump(analysis_data, f, indent=4)

#     log("🎉 Done!")

# # Run it only for the long_dir
# print(f"🚀 Calculating R² for: {long_dir}")
# calculate_r_squared(long_dir, debug=True)

EOF

echo -e "${GREEN}📊 CSV datetime columns added. 📅 ${NC}"

echo -e "${GREEN}🎉 All done! 🎉 ${NC}"
