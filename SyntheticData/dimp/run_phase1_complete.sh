#!/bin/bash
set -e
cd /root/workspace/SyntheticData/dimp

echo "=== Phase 1: DIMP Fair Comparison ==="
echo ""

# Step 1: Smoke Test
echo "[1/3] Smoke test..."
python smoke_test_phase1.py

# Step 2: Run experiments
echo "[2/3] Running experiments..."
python run_phase1.py --protocol-dir outputs/phase1 --seeds 42 43 44

# Step 3: Analysis
echo "[3/3] Generating analysis..."
python analyze_phase1.py --protocol-dir outputs/phase1

echo ""
echo "=== Complete! ==="
echo "Results: outputs/phase1/phase1_results.xlsx"
