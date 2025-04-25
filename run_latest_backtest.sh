#!/bin/bash

# 🎨 Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 🌕 Spinner animation
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

# 🏁 Parameters
START_DATE="${1:-2023-01-01}"

# 📂 Directories
BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"
config_dir="$BASEDIR/configs/optimization"

# 🎯 Find next FITNESS ID
last_id=$(find "$config_dir" -type f -name "*.json" | sed 's/.*\///;s/\.json//' | sort | tail -n 1)
if [ -z "$last_id" ]; then
  FITNESS="000001"
else
  next_id=$((10#$last_id + 1))
  FITNESS=$(printf "%06d" "$next_id")
fi

echo -e "${CYAN}🏋️‍♂️  FITNESS ID starting from: ${YELLOW}$FITNESS${NC} ✨"
echo -e "${CYAN}📅 Start date: ${YELLOW}$START_DATE${NC} 🗓️"

# 📄 Latest optimize result
latest_optimize_result=$(ls -t "$BASEDIR/optimize_results/" | head -n 1)
echo -e "${CYAN}📄 Latest optimize result file: ${YELLOW}$latest_optimize_result${NC} 📄"

# ✂️ Trim it
trimmed_optimize_result="$BASEDIR/optimize_results/finalists.txt"
first_line=$(head -n 1 "$BASEDIR/optimize_results/$latest_optimize_result")
tail -n 5000 "$BASEDIR/optimize_results/$latest_optimize_result" > "$trimmed_optimize_result"
echo "$first_line" | cat - "$trimmed_optimize_result" > temp && mv temp "$trimmed_optimize_result"
echo -e "${CYAN}✂️ Trimmed optimize result created: ${YELLOW}$trimmed_optimize_result${NC} ✂️"

# ⚙️ Run extract_best_config
echo -e "${GREEN}⚙️ Running extract_best_config.py... 🚀 ${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/tools/extract_best_config.py" "$trimmed_optimize_result" &
spinner $!

# 📑 Find generated configs
analysis_files=($(ls -t "$BASEDIR/optimize_results_analysis/"finalists_*.json | head -n 10))
echo -e "${CYAN}📄 Found ${#analysis_files[@]} finalist config files. 📝${NC}"

# 🔁 Process each config
for index in "${!analysis_files[@]}"; do
  analysis_json="${analysis_files[$index]}"
  echo -e "${YELLOW}⭐ Processing rank $((index + 1)) / ${#analysis_files[@]}: ${CYAN}$analysis_json${NC}"

  # 🛠️ Update start_date
  tmp_json="${analysis_json}.tmp"
  jq --arg date "$START_DATE" '.backtest.start_date = $date' "$analysis_json" > "$tmp_json" && mv "$tmp_json" "$analysis_json"
  echo -e "${CYAN}✅ backtest start_date updated ✅${NC}"

  # 🛠️ Update end_date
  jq --arg date "2025-04-14" '.backtest.end_date = $date' "$analysis_json" > "$tmp_json" && mv "$tmp_json" "$analysis_json"
  echo -e "${CYAN}✅ backtest end_date updated ✅${NC}"

  latest_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$analysis_json")

  for config_file in "$BASEDIR/configs/optimization"/*.json; do
    config_values=$(jq -r '.analyses.combined | .adg, .adg_w, .mdg, .mdg_w, .gain, .loss_profit_ratio, .loss_profit_ratio_w, .position_held_hours_mean, .positions_held_per_day, .sharpe_ratio, .sharpe_ratio_w' "$config_file")
    if [ "$latest_values" == "$config_values" ]; then
      echo -e "${RED}🚫 Duplicate values found! 🚫 Aborting... 💥 ${NC}"
      continue 2
    fi
  done


  echo -e "${GREEN}✅ No duplicates! Continuing... 👍 ${NC}"

  # 🚀 Run backtest
  echo -e "${GREEN}🚀 Running backtest.py... 🚀${NC}"
  "$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$analysis_json" &
  spinner $!

  # 📊 Move results
  latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/optimizer/combined/"* | head -n 1)
  echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC} 📊"

  config_dest="$config_dir/${FITNESS}.json"
  cp "$latest_backtest_dir/config.json" "$config_dest"
  echo -e "${GREEN}📦 config.json copied to ${YELLOW}$config_dest${NC} 📦"

  # Rename backtest directory to FITNESS ID
  long_dir="$BASEDIR/backtests/optimizer/live/${FITNESS}"
  mv "$latest_backtest_dir" "$long_dir"
  echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}$long_dir${NC}"

  # Increment FITNESS ID
  next_id=$((10#$FITNESS + 1))
  FITNESS=$(printf "%06d" "$next_id")

done

echo -e "${GREEN}🎉 All done processing ${#analysis_files[@]} configs! 🎉${NC}"
