# Phase 1: DIMP Fair Comparison Experiments

## Overview

Phase 1 implements rigorous fair comparison for DIMP on AG News classification.

**Key Features:**
- Fixed data splits (no re-sampling between experiments)
- Quality Checker consistency verification  
- Three loss types: CE, Quality-only, DIMP
- Comprehensive diagnostics collection
- Multi-seed support
- Automated Excel report generation

## Quick Start

### 1. Smoke Test (2-3 min)

```bash
cd /root/workspace/SyntheticData/dimp
python smoke_test_phase1.py
```

### 2. Run Phase 1 (3 seeds, ~30-40 min)

```bash
python run_phase1.py --protocol-dir outputs/phase1 --seeds 42 43 44
```

### 3. Generate Excel Report

```bash
python analyze_phase1.py --protocol-dir outputs/phase1
```

## Experiments

- **A1**: Real200 CE (baseline)
- **A2**: QC-only (160 real train)
- **B1-B3**: Qwen320 (CE / Quality-only / DIMP)
- **C1-C3**: Llama320 (CE / Quality-only / DIMP)

## Data Splits (Fixed)

```
Real Train: 160 QC train + 40 QC eval
Qwen: 320 train + 80 test (80/class)
Llama: 320 train + 80 test (80/class)
Real Test: 400 (never used for training)
```

## Output

```
outputs/phase1/
├── splits/                    # Fixed split IDs
├── quality_checker/           # QC checkpoint
├── qc_reload_check.json       # Consistency check
├── {experiment}_seed{N}/      # Per-experiment results
├── all_results.json           # Aggregated results
├── phase1_results.xlsx        # Excel report
└── phase1_analysis.txt        # Summary statistics
```

## Key Questions Answered

1. DIMP improvement over CE (same data)?
2. Quality-only vs CE vs DIMP?
3. Does DIMP beat QC-only (teacher)?
4. Which generator benefits most?
5. Consistent across seeds?
