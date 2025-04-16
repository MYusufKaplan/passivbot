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
  printf "\r${GREEN}✅ Done in %02d:%02d!         ${NC}\n" $mins $secs
  tput cnorm
}

# Define BASEDIR and fixed SUBDIR
BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"

echo -e "${CYAN}📂 Starting batch backtests in ${YELLOW}$BASEDIR/configs/optimization${NC}"

# Loop over each config file in configs/optimization/
for config_path in "$BASEDIR/configs/optimization/"*.json; do
  id=$(basename "$config_path")  # e.g. 000001.json
  FITNESS="${id%.json}"          # remove .json extension
  
  echo -e "${GREEN}🚀 Running backtest.py for ${YELLOW}$id${NC}"

  # Modify backtest.end_date to now using jq
  echo -e "${GREEN}🛠️ Setting backtest end_date to now...${NC}"
  tmp_json="${config_path}.tmp"
  jq --arg date "now" '.backtest.end_date = $date' "$config_path" > "$tmp_json" && mv "$tmp_json" "$config_path"
  echo -e "${CYAN}✅ backtest end_date updated${NC}"
  
  "$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$config_path" &
  spinner $!

  # Find latest backtest directory
  latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/optimizer/combined/"*/ | head -n 1)
  echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC}"

  # Rename backtest directory to FITNESS ID
  long_dir="$BASEDIR/backtests/optimizer/combined/${FITNESS}"
  mv "$latest_backtest_dir" "$long_dir"
  echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}$long_dir${NC}"

  # Post-process CSV files
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

done

echo -e "${GREEN}🎉 All batch backtests complete!${NC}"
