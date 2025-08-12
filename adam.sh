#!/bin/bash

echo -e "📝 Enter config number (e.g. 245): "
read CONFIG_NUM

echo -e "📝 Enter output text (e.g. test1): "
read OUTPUT_TEXT

SOURCE_FILE="/home/myusuf/Projects/passivbot/configs/optimization/00${CONFIG_NUM}.json"
TEMPLATE_FILE="/home/myusuf/Projects/passivbot/configs/optimize.json"
OUTPUT_FILE="/home/myusuf/Projects/passivbot/configs/optimize.json.${OUTPUT_TEXT}"

if [ ! -f "$SOURCE_FILE" ]; then
  echo "❌ Source file not found: $SOURCE_FILE"
  exit 1
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "❌ Template file not found: $TEMPLATE_FILE"
  exit 1
fi

echo "📂 Reading from: $SOURCE_FILE"
echo "📂 Using template: $TEMPLATE_FILE"
echo "💾 Output will be: $OUTPUT_FILE"

# Use jq to build the new bounds, skipping non-numeric values
NEW_BOUNDS=$(jq '{
  optimize: {
    bounds: (
      (
        .bot.long | to_entries 
        | map(
            select(.value | type == "number")
            | {("long_" + .key): [ .value, .value ]}
          )
      ) + (
        .bot.short | to_entries 
        | map(
            select(.value | type == "number")
            | {("short_" + .key): [ .value, .value ]}
          )
      )
      | add
    )
  }
}' "$SOURCE_FILE")

# Merge NEW_BOUNDS into the template, replacing "optimize.bounds" and setting iters to 1
FINAL_JSON=$(jq --argjson new_bounds "$NEW_BOUNDS" '
  .optimize.bounds = $new_bounds.optimize.bounds |
  .optimize.iters = 1 |
  .optimize.population_size = 1
' "$TEMPLATE_FILE")

# Save the final JSON
echo "$FINAL_JSON" > "$OUTPUT_FILE"

echo -e "✅🎉 Done! Saved to $OUTPUT_FILE"

