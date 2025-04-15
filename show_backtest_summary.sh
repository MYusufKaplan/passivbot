#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# Header
printf "${BLUE}%-20s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n${RESET}" \
"Name" "adg" "adg_w" "mdg" "mdg_w" "gain" "lpr" "lpr_w" "pos_hrs" "pos_day" "sharpe" "sharpe_w"

# Loop through JSON files
for file in backtests/optimizer/combined/*/analysis.json; do
    if [ -f "$file" ]; then
        name=$(basename $(dirname "$file") | sed 's/_long//g')

        adg=$(jq -r '.adg' "$file")
        adg_w=$(jq -r '.adg_w' "$file")
        mdg=$(jq -r '.mdg' "$file")
        mdg_w=$(jq -r '.mdg_w' "$file")
        gain=$(jq -r '.gain' "$file")
        lpr=$(jq -r '.loss_profit_ratio' "$file")
        lpr_w=$(jq -r '.loss_profit_ratio_w' "$file")
        pos_hrs=$(jq -r '.position_held_hours_mean' "$file")
        pos_day=$(jq -r '.positions_held_per_day' "$file")
        sharpe=$(jq -r '.sharpe_ratio' "$file")
        sharpe_w=$(jq -r '.sharpe_ratio_w' "$file")

        # Format floats to 6 decimal places
        printf "%-20s ${CYAN}%-9.6f${RESET} %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f\n" \
        "$name" "$adg" "$adg_w" "$mdg" "$mdg_w" "$gain" "$lpr" "$lpr_w" "$pos_hrs" "$pos_day" "$sharpe" "$sharpe_w"
    fi
done

