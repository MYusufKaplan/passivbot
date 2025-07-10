#!/bin/bash

# Check input
if [ -z "$1" ]; then
    echo "Usage: $0 <id_number>"
    exit 1
fi

# Pad to 6 digits
id=$(printf "%06d" "$1")
input_csv="backtests/optimizer/live/$id/fills.csv"
output_csv="backtests/optimizer/live/$id/fills_with_date.csv"

# Check file existence
if [ ! -f "$input_csv" ]; then
    echo "❌ CSV not found: $input_csv"
    exit 1
fi

# Header processing
header=$(head -n1 "$input_csv")
echo "date,$header" > "$output_csv"

# Epoch start time: Jan 1st 2023 00:00:00 UTC
start_epoch=$(date -u -d '2023-01-01 00:00:00' +%s)

# Process CSV body
tail -n +2 "$input_csv" | while IFS=, read -r idx minute rest; do
    # skip empty lines or invalid minute
    if [[ -z "$minute" || "$minute" == "minute" ]]; then
        continue
    fi

    # calculate final date (UTC+3 = 10800s offset)
    minute_ts=$((start_epoch + (minute * 60) + 10800))
    date_str=$(date -u -d "@$minute_ts" '+%Y-%m-%d %H:%M:%S')

    # append to output
    echo "$date_str,$idx,$minute,$rest" >> "$output_csv"
done

echo "✅ Output saved to: $output_csv"
