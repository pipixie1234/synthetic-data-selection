"""Checks for error-set algebra, degenerate kappa, blind sampling and rankings."""
import collections
import json
import unittest

import run_redundancy as exp


class PilotTests(unittest.TestCase):
    def test_error_overlap_empty_is_undefined(self):
        self.assertIsNone(exp.jaccard(set(), set()))
        self.assertEqual(exp.jaccard({1,2}, {2,3}), 1/3)
        self.assertEqual(exp.jaccard({1}, set()), 0)

    def test_kappa_accounts_for_marginals(self):
        self.assertEqual(exp.kappa(["PASS", "FAIL"], ["PASS", "FAIL"]), 1)
        self.assertEqual(exp.kappa(["PASS", "FAIL"], ["FAIL", "PASS"]), -1)
        self.assertIsNone(exp.kappa(["PASS"]*5, ["PASS"]*5))
        self.assertEqual(exp.kappa(["PASS"]*4, ["PASS"]*3+["FAIL"]), 0)

    def test_parser_rejects_imputed_or_invalid_scores(self):
        raw = "predicted_label: SciTech\nquality: FAIL\nconfidence: 0.02\nreason: LABEL_MISMATCH"
        self.assertEqual(exp.parse(raw)["confidence"], 0.02)
        for bad in [raw+"\nExplanation", raw.replace("0.02", "NaN"), raw.replace("0.02", "1.2"), raw.replace("LABEL_MISMATCH", "FAIL_LABEL")]:
            with self.assertRaises(ValueError):
                exp.parse(bad)

    def test_matched_budget_preserves_ranking_and_size(self):
        rows = [dict(item_id=str(i), qwen_confidence=i/20) for i in range(20)]
        rejected, diagnostics = exp.bottom(rows, "qwen")
        self.assertEqual(rejected, {"0", "1", "2", "3"})
        self.assertFalse(diagnostics["ties_straddle_cutoff"])
        for x in rows:
            x["qwen_confidence"] = .5
        first, diagnostics = exp.bottom(rows, "qwen")
        second, _ = exp.bottom(list(reversed(rows)), "qwen")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(diagnostics["ties_straddle_cutoff"])

    def test_frozen_dataset_and_blind_prompt(self):
        rows = exp.read_jsonl(exp.OUT / "samples.jsonl")
        main = [x for x in rows if x["subset"] == "main"]
        counts = collections.Counter((x["source"],x["true_label"]) for x in main)
        for source in ["qwen", "llama", "real"]:
            for label in exp.LABELS:
                self.assertEqual(counts[source,label], 10 if source == "real" else 20)
        self.assertEqual(len(rows), 240)
        by_id = {x["item_id"]: x for x in main}
        for x in rows:
            if x["subset"] == "challenge":
                self.assertEqual(x["source"], "real")
                self.assertEqual(x["text"], by_id[x["parent_id"]]["text"])
                self.assertNotEqual(x["true_label"], x["presented_label"])
        prompts = exp.read_jsonl(exp.OUT / "blind_prompts.jsonl")
        for item, x in zip(prompts, rows):
            self.assertEqual(item["item_id"], x["item_id"])
            self.assertEqual(item["prompt"], exp.PROMPT.format(label=x["presented_label"],text=json.dumps(x["text"],ensure_ascii=False)))
            self.assertNotIn(x["original_id"], item["prompt"])
            self.assertNotIn(x["item_id"], item["prompt"])


if __name__ == "__main__":
    unittest.main()
