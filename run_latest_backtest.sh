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

# Prompt for FITNESS interactively if not provided
if [ -z "$1" ]; then
  read -rp "$(echo -e "${YELLOW}🏋️ Enter FITNESS label: ${NC}")" FITNESS
else
  FITNESS="$1"
fi

# Set start date (optional argument)
START_DATE="${2:-2023-01-01}"

# Define BASEDIR and fixed SUBDIR
BASEDIR="/home/myusuf/Projects/passivbot"
SUBDIR="optimizer"

echo -e "${CYAN}🏋️ Using FITNESS: ${YELLOW}$FITNESS${NC}"
echo -e "${CYAN}📅 Using start date: ${YELLOW}$START_DATE${NC}"

# Step 1: Find the latest file in optimize_results/
latest_optimize_result=$(ls -t "$BASEDIR/optimize_results/" | head -n 1)
echo -e "${CYAN}📄 Latest optimize result: ${YELLOW}$latest_optimize_result${NC}"

# Run extract_best_config.py
echo -e "${GREEN}⚙️ Running extract_best_config.py...${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/tools/extract_best_config.py" "$BASEDIR/optimize_results/$latest_optimize_result" &
spinner $!

# Step 2: Find the latest .json file in optimize_results_analysis/
latest_analysis_json=$(ls -t "$BASEDIR/optimize_results_analysis/"*.json | head -n 1)
echo -e "${CYAN}📝 Latest analysis JSON: ${YELLOW}$latest_analysis_json${NC}"

# Modify backtest.start_date to $START_DATE using jq
echo -e "${GREEN}🛠️ Setting backtest start_date to $START_DATE...${NC}"
tmp_json="${latest_analysis_json}.tmp"
jq --arg date "$START_DATE" '.backtest.start_date = $date' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ backtest start_date updated${NC}"

# Step 3: Run backtest.py (mixed mode)
# echo -e "${GREEN}🚀 Running backtest.py (Mixed)...${NC}"
# "$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$latest_analysis_json" &
# spinner $!

# # Step 4: Find the latest backtest directory
# latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/$SUBDIR/combined/"*/ | head -n 1)
# echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC}"

# # Step 5: Copy config.json to configs/FITNESS_mixed.json
# config_dest="$BASEDIR/configs/${FITNESS}_mixed.json"
# cp "$latest_backtest_dir/config.json" "$config_dest"
# echo -e "${GREEN}📦 Copied config to ${YELLOW}$config_dest${NC}"

# # Step 6: Rename backtest dir to FITNESS_mixed
# mixed_dir="$BASEDIR/backtests/$SUBDIR/combined/${FITNESS}_mixed"
# mv "$latest_backtest_dir" "$mixed_dir"
# echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}${mixed_dir}${NC}"

# Step 7: Prepare for Long-only backtest
echo -e "${GREEN}🛠️ Modifying config for Long-only...${NC}"
jq '.bot.short.n_positions = 0 | .bot.short.total_wallet_exposure_limit = 0' "$latest_analysis_json" > "$tmp_json" && mv "$tmp_json" "$latest_analysis_json"
echo -e "${CYAN}✅ Config modified for Long-only${NC}"

# Step 8: Run backtest.py (Long)
echo -e "${GREEN}🚀 Running backtest.py (Long)...${NC}"
"$BASEDIR/.venv/bin/python3" "$BASEDIR/src/backtest.py" "$latest_analysis_json" &
spinner $!

# Step 9: Find latest backtest directory (Long)
latest_backtest_dir=$(ls -dt "$BASEDIR/backtests/$SUBDIR/combined/"*/ | head -n 1)
echo -e "${CYAN}📊 Latest backtest directory: ${YELLOW}$latest_backtest_dir${NC}"

# Step 10: Copy config.json to configs/FITNESS.json
config_dest="$BASEDIR/configs/${FITNESS}.json"
cp "$latest_backtest_dir/config.json" "$config_dest"
echo -e "${GREEN}📦 Copied config to ${YELLOW}$config_dest${NC}"

# Step 11: Rename backtest dir to FITNESS
long_dir="$BASEDIR/backtests/$SUBDIR/combined/${FITNESS}"
mv "$latest_backtest_dir" "$long_dir"
echo -e "${GREEN}📁 Renamed backtest directory to ${YELLOW}${long_dir}${NC}"

echo -e "${GREEN}🎉 All done!${NC}"

