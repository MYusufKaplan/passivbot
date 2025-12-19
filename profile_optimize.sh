#!/bin/bash

cd /home/myusuf/Projects/passivbot
source .venv/bin/activate

echo "🔥 Starting flamegraph profiling of optimize.py for 10 minutes..."
echo "📊 This will create a flamegraph.svg file showing where time is spent"

# Get the full path to the python executable
PYTHON_PATH=$(which python3)
echo "Using Python: $PYTHON_PATH"

# Set PYTHONPATH to include the current directory
export PYTHONPATH="/home/myusuf/Projects/passivbot:$PYTHONPATH"

# Run with perf and generate flamegraph
# The -F 99 means sample at 99Hz (99 times per second)
# timeout 10m limits to 10 minutes
sudo -E perf record -F 99 -g --call-graph dwarf -o perf.data -- timeout 5m $PYTHON_PATH src/optimize.py configs/optimize.json

echo "📈 Generating flamegraph..."
sudo perf script -i perf.data | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg && ls -lh flamegraph.svg

echo "✅ Done! Open flamegraph.svg in a browser to see the results"
echo "🔍 Look for the widest bars - those are the functions taking the most time"
