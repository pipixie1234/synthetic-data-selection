"""Compare Real, candidate, TopK, and RA-CaR text-length distributions."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "SyntheticData" / "exp" / "outputs" / "curation_validation"
CAR = ROOT / "SyntheticData" / "CaRTry" / "outputs" / "car_selector_validation"
sys.path.insert(0, str(ROOT / "SyntheticData" / "exp"))
import run_distribution_curation as curation  # noqa: E402

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}


def lengths(records: list[dict]) -> np.ndarray:
    return np.asarray([len(str(row["text"]).split()) for row in records], dtype=float)


def main() -> None:
    data = curation.load_frozen_data()
    scores = pd.read_csv(EXP / "sample_proximity_scores.csv")
    topk = pd.read_csv(EXP / "e2_uniform_topk_fixed.csv").to_dict("records")
    car_sets = {}
    for path in sorted(CAR.glob("car_sel*.csv")):
        match = re.fullmatch(r"car_sel(\d+)", path.stem)
        if match:
            car_sets[int(match.group(1))] = pd.read_csv(path).to_dict("records")
    if set(car_sets) != {41, 42, 43, 44, 45}:
        raise FileNotFoundError("Expected car_sel41.csv through car_sel45.csv")

    rows = []
    for source in ("qwen", "llama"):
        for label, cls in LABEL_NAMES.items():
            real = [x for x in data["real_reference"] if int(x["label"]) == label]
            candidate = [x for x in data[f"{source}_train"] if int(x["label"]) == label]
            top = [x for x in topk if x["source"] == source and int(x["label"]) == label]
            # Pool the five fixed-budget CaR selections. This reports the
            # distribution actually explored across cluster seeds.
            car = [x for seed in sorted(car_sets) for x in car_sets[seed] if x["source"] == source and int(x["label"]) == label]
            candidate_lengths = lengths(candidate)
            proximity = scores[(scores["source"] == source) & (scores["label"] == label)]
            rho = float(spearmanr(proximity["num_tokens"] if "num_tokens" in proximity else proximity["text"].map(lambda x: len(str(x).split())), proximity["top5_real_cosine_mean"]).statistic)
            real_lengths = lengths(real); top_lengths = lengths(top); car_lengths = lengths(car)
            real_candidate_w1 = float(wasserstein_distance(real_lengths, candidate_lengths))
            real_topk_w1 = float(wasserstein_distance(real_lengths, top_lengths))
            rows.append({
                "Source": source.capitalize(),
                "Class": cls,
                "Real_Median": float(np.median(real_lengths)),
                "Candidate_Median": float(np.median(candidate_lengths)),
                "TopK_Median": float(np.median(top_lengths)),
                "CaR_Median": float(np.median(car_lengths)),
                "Real_Candidate_W1": real_candidate_w1,
                "Real_TopK_W1": real_topk_w1,
                "Improvement_W1": real_candidate_w1 - real_topk_w1,
                "Length_Proximity_Spearman": rho,
            })

    table = pd.DataFrame(rows)
    out_csv = CAR / "real_candidate_topk_car_length_comparison.csv"
    out_md = CAR / "real_candidate_topk_car_length_comparison.md"
    out_xlsx = CAR / "real_candidate_topk_car_length_comparison.xlsx"
    table.to_csv(out_csv, index=False)
    table.to_markdown(out_md, index=False, floatfmt=".3f")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Length_Comparison", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
