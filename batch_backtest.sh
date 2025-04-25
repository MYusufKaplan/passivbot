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

# Check if --all is passed as an argument
ARCHIVE_MODE=""
if [[ "$1" == "--all" ]]; then
  ARCHIVE_MODE="archive"
  echo -e "${YELLOW}📦 Archive mode enabled. Using 'backtests/optimizer/archive' as base_dir.${NC}"
else
  ARCHIVE_MODE="live"
  echo -e "${CYAN}📂 Standard mode. Using 'backtests/optimizer' as base_dir.${NC}"
fi

# Define BASEDIR and fixed SUBDIR
BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"

echo -e "${CYAN}📂 Starting batch backtests in ${YELLOW}$BASEDIR/configs/optimization${NC}"

# Loop over each config file in configs/optimization/
for config_path in "$BASEDIR/configs/optimization/"*.json; do
  id=$(basename "$config_path")  # e.g. 000001.json
  FITNESS="${id%.json}"          # remove .json extension
  
  echo -e "${GREEN}🚀 Running backtest.py for ${YELLOW}$id${NC}"

  # Modify backtest.end_date and base_dir
  echo -e "${GREEN}🛠️ Updating backtest config...${NC}"
  tmp_json="${config_path}.tmp"
  jq --arg date "2025-04-14" --arg base_dir "backtests/optimizer${ARCHIVE_MODE:+/$ARCHIVE_MODE}" \
    '.backtest.end_date = $date | .backtest.base_dir = $base_dir' \
    "$config_path" > "$tmp_json" && mv "$tmp_json" "$config_path"
  echo -e "${CYAN}✅ backtest end_date and base_dir updated${NC}"

  "$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$config_path" &
  spinner $!

  # Find latest backtest directory
  latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/optimizer${ARCHIVE_MODE:+/$ARCHIVE_MODE}"/combined/* | head -n 1)
  echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC}"

  # Rename backtest directory to FITNESS ID
  long_dir="$BASEDIR/backtests/optimizer${ARCHIVE_MODE:+/$ARCHIVE_MODE}/${FITNESS}"
  mv "$latest_backtest_dir" "$long_dir"
  echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}$long_dir${NC}"

done

echo -e "${GREEN}🎉 All batch backtests complete!${NC}"
