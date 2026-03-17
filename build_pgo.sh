#!/bin/bash
echo "🦀 Building passivbot-rust with PGO and shared memory support..."

cd passivbot-rust

# Step 1: Build with instrumentation
echo "📊 Step 1: Building with instrumentation..."
mkdir -p /tmp/pgo-data
RUSTFLAGS="-C target-cpu=native -C profile-generate=/tmp/pgo-data" maturin develop --release

# Step 2: Run typical workload to generate profile data (10 minutes)
echo "🏃 Step 2: Running workload to generate profile data (10 minutes)..."
cd ..
source .venv/bin/activate
timeout 600 python3 src/optimize.py configs/optimize.json || true

# Step 3: Rebuild with profile data
echo "🚀 Step 3: Rebuilding with profile data and optimizations..."
cd passivbot-rust
RUSTFLAGS="-C target-cpu=native -C profile-use=/tmp/pgo-data" maturin develop --release

echo "✅ PGO build complete with shared memory support!"
echo ""
echo "📝 New features:"
echo "  - RAM-based shared memory (no disk I/O during backtests)"
echo "  - Linux shared memory via /dev/shm/ for optimal performance"
echo "  - Cross-platform compatibility maintained"
echo ""
echo "🚀 Ready for much faster parallel optimization!"
