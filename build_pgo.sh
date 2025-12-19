#!/bin/bash
cd passivbot-rust

# Step 1: Build with instrumentation
RUSTFLAGS="-C target-cpu=znver4 -C profile-generate=/tmp/pgo-data" maturin develop --release

# Step 2: Run typical workload to generate profile data (10 minutes)
cd ..
source .venv/bin/activate
timeout 600 python3 src/optimize.py configs/optimize.json || true

# Step 3: Rebuild with profile data
cd passivbot-rust
RUSTFLAGS="-C target-cpu=znver4 -C profile-use=/tmp/pgo-data -C target-feature=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl,+avx2,+fma,+bmi2" maturin develop --release

echo "PGO build complete!"
