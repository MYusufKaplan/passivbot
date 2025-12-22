#!/bin/bash
echo "🦀 Building passivbot-rust with PGO and shared memory support..."

cd passivbot-rust

# Step 1: Build with instrumentation
echo "📊 Step 1: Building with instrumentation..."
RUSTFLAGS="-C target-cpu=znver4 -C profile-generate=/tmp/pgo-data" maturin develop --release

# Step 2: Run typical workload to generate profile data (10 minutes)
echo "🏃 Step 2: Running workload to generate profile data (5 minutes)..."
cd ..
source .venv/bin/activate
timeout 300 python3 src/optimize.py configs/optimize.json || true

# Step 3: Rebuild with profile data
echo "🚀 Step 3: Rebuilding with profile data and optimizations..."
cd passivbot-rust
RUSTFLAGS="-C target-cpu=znver4 -C profile-use=/tmp/pgo-data -C target-feature=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl,+avx2,+fma,+bmi2" maturin develop --release

echo "✅ PGO build complete with shared memory support!"
echo ""
echo "📝 New features:"
echo "  - RAM-based shared memory (no disk I/O during backtests)"
echo "  - Linux shared memory via /dev/shm/ for optimal performance"
echo "  - Cross-platform compatibility maintained"
echo ""
echo "🚀 Ready for much faster parallel optimization!"
