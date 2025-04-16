#!/bin/bash

# Define some color codes 🎨
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Spinner with elapsed time ⏳
spinner() {
  local pid=$1
  local delay=0.15
  local spinstr=('🌑' '🌒' '🌓' '🌔' '🌕' '🌖' '🌗' '🌘')
  local i=0
  local start_time=$(date +%s)

  tput civis  # Hide cursor
  while kill -0 "$pid" 2>/dev/null; do
    local now=$(date +%s)
    local elapsed=$((now - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    printf "\r${CYAN}⏳ %s %02d:%02d${NC}" "${spinstr[$i]}" $mins $secs
    i=$(( (i+1) % ${#spinstr[@]} ))
    sleep $delay
  done
  printf "\r${GREEN}✅ Done in %02d:%02d!         ${NC}\n" $mins $secs
  tput cnorm  # Show cursor again
}

# Set start date (optional argument)
START_DATE="${1:-2023-01-01}"

# Define BASEDIR and fixed SUBDIR
BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"

# Automatically determine next FITNESS ID based on existing config files 🆔
config_dir="$BASEDIR/configs/optimization"
last_id=$(find "$config_dir" -type f -name "*.json" | sed 's/.*\///;s/\.json//' | sort | tail -n 1)
if [ -z "$last_id" ]; then
  FITNESS="000001"
else
  next_id=$((10#$last_id + 1))
  FITNESS=$(printf "%06d" "$next_id")
fi

echo -e "${CYAN}🏋️ Assigned FITNESS ID: ${YELLOW}$FITNESS${NC}"
echo -e "${CYAN}📅 Using start date: ${YELLOW}$START_DATE${NC}"

# Step 1: Find the latest file in optimize_results/
latest_optimize_result=$(ls -t "$BASEDIR/optimize_results/" | head -n 1)
echo -e "${CYAN}📄 Latest optimize result: ${YELLOW}$latest_optimize_result${NC}"

# Create a trimmed optimize result file with the last 5000 lines and add the first line at the top
trimmed_optimize_result="$BASEDIR/optimize_results/finalists.txt"
first_line=$(head -n 1 "$BASEDIR/optimize_results/$latest_optimize_result")
tail -n 5000 "$BASEDIR/optimize_results/$latest_optimize_result" > "$trimmed_optimize_result"
echo "$first_line" | cat - "$trimmed_optimize_result" > temp && mv temp "$trimmed_optimize_result"
echo -e "${CYAN}✂️  Created trimmed optimize result: ${YELLOW}$trimmed_optimize_result${NC}"

# Run extract_best_config.py using the trimmed file
echo -e "${GREEN}⚙️ Running extract_best_config.py...${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/tools/extract_best_config.py" "$trimmed_optimize_result" &
spinner $!

# Step 2: Find the latest .json file in optimize_results_analysis/
latest_analysis_json=$(ls -t "$BASEDIR/optimize_results_analysis/"*.json | head -n 1)
echo -e "${CYAN}📝 Latest analysis JSON: ${YELLOW}$latest_analysis_json${NC}"

# Extract relevant values from the latest analysis JSON
latest_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$BASEDIR/optimize_results_analysis/finalists.txt.json")

# Debugging output

# Step 3: Compare with each config file
for config_file in "$BASEDIR/configs/optimization"/*.json; do
  config_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$config_file")
  
  # Debugging output

  # Compare values
  if [ "$latest_values" == "$config_values" ]; then
    echo -e "${RED}🚫 Aborting due to duplicate values."
    exit 1
  fi
done

# Step 4: Continue with the rest of the script if no duplicates were found
echo -e "${GREEN}✅ No duplicate values found. Continuing with the script...${NC}"

# Modify backtest.start_date to $START_DATE using jq
echo -e "${GREEN}🛠️ Setting backtest start_date to $START_DATE...${NC}"
tmp_json="${latest_analysis_json}.tmp"
jq --arg date "$START_DATE" '.backtest.start_date = $date' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ backtest start_date updated${NC}"

# Modify backtest.end_date to now using jq
echo -e "${GREEN}🛠️ Setting backtest end_date to now...${NC}"
tmp_json="${latest_analysis_json}.tmp"
jq --arg date "now" '.backtest.end_date = $date' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ backtest end_date updated${NC}"

# Step 8: Run backtest.py
echo -e "${GREEN}🚀 Running backtest.py...${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$latest_analysis_json" &
spinner $!

# Step 9: Find latest backtest directory
latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/optimizer/combined/"*/ | head -n 1)
echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC}"

# Step 10: Copy config.json to configs/FITNESS.json
config_dest="$config_dir/${FITNESS}.json"
cp "$latest_backtest_dir/config.json" "$config_dest"
echo -e "${GREEN}📦 Copied config to ${YELLOW}$config_dest${NC}"

# Step 11: Rename backtest dir to FITNESS
long_dir="$BASEDIR/backtests/optimizer/combined/${FITNESS}"
mv "$latest_backtest_dir" "$long_dir"
echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}$long_dir${NC}"

echo -e "${GREEN}📝 Updating CSV files with datetime column...${NC}"

"$BASEDIR/.venv/bin/python3" - <<EOF
import pandas as pd
from datetime import datetime, timedelta
import json

long_dir = "$long_dir"

# Load start_time from config.json
with open(f"{long_dir}/config.json") as f:
    start_time_str = json.load(f)["backtest"]["start_date"]

start_time = datetime.fromisoformat(start_time_str)

def process_balance_and_equity(file_path):
    df = pd.read_csv(file_path)
    minutes_col = df.columns[0]  # first unnamed column
    df['datetime'] = df[minutes_col].apply(lambda x: start_time + timedelta(minutes=x))
    # Move 'datetime' to the first column
    cols = ['datetime'] + df.columns[:-1].tolist()
    df = df[cols]
    df.to_csv(file_path, index=False)

def process_fills(file_path):
    df = pd.read_csv(file_path)
    if 'minute' in df.columns:
        df['datetime'] = df['minute'].apply(lambda x: start_time + timedelta(minutes=x))
        # Move 'datetime' to the first column
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        df.to_csv(file_path, index=False)

process_balance_and_equity(f"{long_dir}/balance_and_equity.csv")
process_fills(f"{long_dir}/fills.csv")

print("✅ CSV files updated successfully.")

EOF

echo -e "${GREEN}📊 CSV datetime columns added.${NC}"


echo -e "${GREEN}🎉 All done!${NC}"
