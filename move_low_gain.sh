#!/usr/bin/env bash
set -euo pipefail

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

LIVE_DIR="backtests/optimizer/live"
ARCHIVE_DIR="backtests/optimizer/archive"
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo -e "${RED}❌ Unknown option: $arg${RESET}" && exit 1 ;;
    esac
done

# Ask user for threshold
read -rp "⚙️ Enter gain threshold (default 1000): " THRESHOLD
THRESHOLD=${THRESHOLD:-1000}

echo -e "${CYAN}🔍 Scanning $LIVE_DIR for folders with gain < $THRESHOLD...${RESET}"
$DRY_RUN && echo -e "${YELLOW}🟨 Dry-run mode enabled. No files will be moved.${RESET}"

# Process folders
for d in "$LIVE_DIR"/*; do
    json_file="$d/analysis.json"
    if [[ -f "$json_file" ]]; then
        gain=$(jq -r '.gain // empty' "$json_file" 2>/dev/null || echo "")
        if [[ -n "$gain" && "$gain" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            if (( $(echo "$gain < $THRESHOLD" | bc -l) )); then
                if $DRY_RUN; then
                    echo -e "${YELLOW}➡️ Would move: $d (gain=$gain)${RESET}"
                else
                    echo -e "${YELLOW}➡️ Moving: $d (gain=$gain)${RESET}"
                    mv "$d" "$ARCHIVE_DIR"/
                    echo -e "${GREEN}✅ Moved to archive${RESET}"
                fi
            else
                echo -e "${GREEN}✔️ Skipped: $d (gain=$gain)${RESET}"
            fi
        else
            echo -e "${RED}❌ Invalid or missing gain in $json_file${RESET}"
        fi
    else
        echo -e "${RED}❌ Missing analysis.json in $d${RESET}"
    fi
done

echo -e "${CYAN}🏁 Done.${RESET}"

