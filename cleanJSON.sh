#!/bin/bash

# Colors
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
RED='\033[1;31m'
RESET='\033[0m'

# Dry run flag
DRY_RUN=false

# Check for --dry-run
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}🧪 Dry run enabled — no files will be moved or deleted.${RESET}"
fi

# Ask the user for the threshold N
read -p "🔢 Enter the gain threshold (N): " N

# Create the destination folder if it doesn't exist
mkdir -p configs/old

# Loop over all JSON files
for file in configs/optimization/*.json; do
    # Get the filename without extension
    filename=$(basename "$file" .json)

    # Extract gain using jq
    gain=$(jq -r '.analyses.combined.gain' "$file")

    # Check if gain is a valid number and less than N
    if [[ "$gain" =~ ^[0-9.eE+-]+$ ]] && (( $(echo "$gain < $N" | bc -l) )); then
        echo -e "${CYAN}🔍 $filename: gain = $gain < $N${RESET}"

        if $DRY_RUN; then
            echo -e "${YELLOW}🚫 Would move: $file → configs/old/${filename}.json${RESET}"
            echo -e "${YELLOW}🚫 Would delete: backtests/optimizer/live/$filename/${RESET}"
        else
            mv "$file" "configs/old/$filename.json"
            rm -rf "backtests/optimizer/live/$filename"
            echo -e "${GREEN}✅ Moved & deleted for $filename${RESET}"
        fi
    fi
done

echo -e "${GREEN}✨ Cleanup complete.${RESET}"
