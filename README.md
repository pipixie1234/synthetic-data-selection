# Synthetic Data Selection for AG News

This repository studies how to select and combine synthetic text for downstream classification when the training budget is fixed. The experiments use four-class AG News and compare data weighting, real-anchored sample selection, source allocation, distribution diagnostics, and model-based verification.

The central question is practical: **when only a fixed number of examples can be used, which synthetic examples help a classifier generalize to real data?** The project treats synthetic-to-synthetic scores as diagnostics and evaluates selection utility on a separate real-data development split.

## Main findings

- In the current Dev2000 protocol, Real-Anchored TopK reached **75.26% accuracy / 73.98% macro-F1**, compared with **72.59% / 70.17%** for the 25-run random-selection mean. The paired TopK-minus-random mean difference was **+2.67 and +3.81 percentage points**, respectively; all five paired model-seed deltas were positive. These are development-set results, not a claim of final generalization.
- A 4×4 train-source/test-source matrix exposed a large domain asymmetry: synthetic-trained classifiers scored roughly **97–100%** on synthetic holdouts, while their accuracy on the real test set was about **68–70%**. This is why synthetic holdout performance is not used as a substitute for real-data evaluation.
- Cross-verifier selection improved over random selection in one fixed candidate pool, and exceeded self-verification in four of five paired model seeds. It still did **not** beat Real-Anchored TopK on mean macro-F1 (**72.92% vs. 73.57%**). Under the project's pre-defined interpretation rule, this is not useful evidence for replacing TopK or adding more verifier complexity.
- Several alternatives produced useful mechanism diagnostics without showing a downstream advantage: length-normalized DSIR largely relieved weight concentration, while DSIR, clustering-based CaR, adaptive TopK, and coarse source quotas did not consistently exceed TopK.

The detailed Chinese project summary and a short discussion outline for faculty meetings are in [`成果/`](成果/).

Presentation materials are available in [`成果/图表使用说明.md`](成果/图表使用说明.md): six figures in PNG/SVG, a workbook with editable charts, six source workbooks, and a seven-slide PowerPoint draft. Rebuild them with `python tools/build_research_presentation.py` after installing `requirements-presentation.txt`. Upload scope is documented in [`成果/GitHub上传范围.md`](成果/GitHub上传范围.md).

## Repository map

| Path | Contents |
| --- | --- |
| `SyntheticData/dimp/` | Quality estimation and training-time weighting experiments |
| `SyntheticData/MMD/` | Embedding, MMD, and lexical-overlap diagnostics |
| `SyntheticData/exp/` | Real-Anchored TopK multi-seed validation |
| `SyntheticData/DisrTry/`, `CaRTry/`, `APT/` | DSIR, clustering/ranking, and adaptive-selector comparisons |
| `SyntheticData/allocation/`, `candidate_pool/` | Source quota and candidate-pool experiments |
| `SyntheticData/SmaExp/` | Initial generation protocol and source compatibility matrix |
| `SyntheticData/cross_verifier/` | Verifier selection and downstream validation |
| `SyntheticData/redundancy/`, `redundancy_repair/` | Verifier agreement, label reliability, and prompt-repair pilots |
| `SyntheticData/diagnostics/`, `real_reference_uncertainty/` | Feedback, reference, and candidate-pool diagnostics |
| `dsir-main/` | DSIR upstream implementation used by the DSIR experiment; not original project code |
| `CaR-main/` | Local CaR reference snapshot, excluded from the public-upload default; the project experiment is under `SyntheticData/CaRTry/` |

Most experiment outputs, local datasets, caches, and model checkpoints are intentionally excluded from version control. The analyses depend on local AG News splits and generated candidate pools that are not redistributed here.

## Setup

Use Python 3.10–3.12. Install a PyTorch build appropriate for your CUDA or CPU environment first, then install the shared analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts use Hugging Face model IDs by default in some experiments and local checkpoint paths in others. The verifier experiments require local Qwen3-4B and Llama-3.2-3B checkpoints; embedding and classifier experiments also use RoBERTa, MiniLM, and `bert-base-uncased`. **Do not commit model weights.** Set model paths through the relevant CLI options or adapt the defaults for your machine. Model availability, license, memory requirements, and Transformers compatibility depend on the selected checkpoint.

The repository is an accumulated research workspace rather than a packaged Python library. Some historical scripts and run notes still contain paths from the original Linux workstation (for example `/root/workspace` and Hugging Face cache paths). Run commands from the repository root, inspect each experiment README, and update local paths before running on another machine.

## Reproducing the main selector comparison

The central multi-seed script is documented in [`SyntheticData/exp/README_e1_e2_validation.md`](SyntheticData/exp/README_e1_e2_validation.md). It expects the project data split and cached model checkpoints to be prepared. Outputs are written beneath `SyntheticData/exp/outputs/` and remain local.

```bash
python SyntheticData/exp/run_e1_e2_validation.py \
  --dev-size 2000 \
  --dev-seed 2026 \
  --selection-seeds 41 42 43 44 45 \
  --model-seeds 42 43 44 45 46 \
  --top-k 5 \
  --real-reference qc_train \
  --no-final-holdout
```

The final holdout was not used to develop the reported selectors. Keep it reserved until the method and evaluation protocol are frozen.

## Reproducibility and interpretation

- The main selector results are from a fixed candidate pool and Dev2000; they are exploratory and should be confirmed on an independent candidate pool before making broad claims.
- Seeds, candidate pools, data splits, and budgets differ across experiment families. Do not compare values across tables unless the protocol matches.
- A high score on a synthetic holdout measures within-source compatibility; it does not establish real-world generalization.
- Per-experiment protocols, limitations, and output manifests are documented in the corresponding subdirectory README and analysis reports.

## Attribution and release notes

`dsir-main/` and `CaR-main/` contain upstream/reference projects. Cite their papers and preserve their licenses and notices if redistributing their code. Before making a public release, review the license and redistribution terms for every dataset, model checkpoint, and upstream artifact. This repository does not include model weights.

## Dependencies

The shared environment is listed in [`requirements.txt`](requirements.txt). PyTorch installation is separated because the correct wheel depends on the target hardware and CUDA runtime. Individual experiment notes may list additional or historical environment requirements.
