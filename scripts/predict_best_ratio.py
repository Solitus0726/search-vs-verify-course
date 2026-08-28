# predict_best_ratio.py -- Experiment 3 prediction interface + auto scoring + small probe
# Interface contract (will be written into Notebook 03):
#   predict_best_ratio(task_meta) -> "search" | "verify"
#     Input task_meta (dict):
#       dataset: dataset name (e.g., "GSM8K")
#       model:   model name (one of the six models, e.g., "qwen3-0.6b")
#       p:       single-candidate correctness (output of the probe estimate_p; students can run 5-10 problems to estimate it)
#       budget:  per-problem budget B (e.g., 50)
#       judge_quality: verifier quality estimate ("weak" | "strong", default "weak")
#     Output: predicted best allocation direction ("search" = search/voting wins; "verify" = verification/mean-score selection wins)
#   Auto-scoring rule: the prediction earns a point if it matches the measured direction (score_prediction, 1/0)
# Student-implementation placeholder: this file provides the contract and skeleton (predict returns None when unimplemented); students fill in the logic in Notebook 03.
# Usage: python scripts/predict_best_ratio.py --model qwen3-0.6b --n 10 (probe example)

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import MODEL_FULL_NAMES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS1_DIR = os.path.join(PROJECT_ROOT, "data", "cache_subset")


def estimate_p(model: str, n: int = 10, dataset: str = "MATH-500") -> Optional[float]:
    # Small probe: estimate the single-candidate correctness p from the precomputed pass@1 cache (pass1_*.jsonl, MATH-500)
    # Input: model name + number of probe problems (5-10) + dataset; output: p or None (missing data / unknown model)
    # Note: GSM8K has no precomputed cache -- return None and tell students to estimate from their own experiment records
    if dataset != "MATH-500":
        return None  # only MATH-500 has a pass1 cache; estimate from real experiments for other tasks
    full = MODEL_FULL_NAMES.get(model)
    if full is None:
        return None
    # pass1 file names use the short name (no -Instruct suffix): pass1_Qwen3-4B.jsonl, etc.
    short = full.replace("-Instruct", "")
    path = os.path.join(PASS1_DIR, f"pass1_{short}.jsonl")
    if not os.path.exists(path):
        return None
    correct = 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if total >= n:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("correct") is not None:
                correct += 1 if rec["correct"] else 0
                total += 1
    return correct / total if total > 0 else None


def score_prediction(prediction: str, actual_direction: str) -> int:
    # Auto-scoring rule: 1 point if the predicted direction matches the measured one, 0 otherwise
    # actual_direction is computed from experiment data (search-config accuracy vs verify-config accuracy)
    if prediction not in ("search", "verify") or actual_direction not in ("search", "verify"):
        return 0
    return 1 if prediction == actual_direction else 0


def actual_direction_from_configs(configs: list) -> Optional[str]:
    # Compute the measured direction from aggregated experiment results (configs of experiment1_*.json):
    # compare the accuracy of search configs (M=0) vs verify configs (M>0) at the same cost level
    searches = [c for c in configs if c["config"]["strategy"] == "search"]
    verifies = [c for c in configs if c["config"]["strategy"] == "verify"]
    if not searches or not verifies:
        return None
    # Average over all search configs and all verify configs (comparing within the same budget level)
    s_acc = sum(c["accuracy"] for c in searches) / len(searches)
    v_acc = sum(c["accuracy"] for c in verifies) / len(verifies)
    return "search" if s_acc >= v_acc else "verify"


def predict_best_ratio(task_meta: Dict) -> Optional[str]:
    # Placeholder function (student implementation): takes task metadata, returns the predicted direction
    # This file only provides the contract skeleton; students fill in the logic in Notebook 03 based on the core principles
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment-3 probe: estimate the single-candidate correctness p")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_FULL_NAMES))
    ap.add_argument("--n", type=int, default=10, help="probe problems (5-10)")
    ap.add_argument("--dataset", default="MATH-500", choices=["MATH-500", "GSM8K"],
                    help="dataset to estimate p on (only MATH-500 has a precomputed cache)")
    args = ap.parse_args()
    p = estimate_p(args.model, args.n, dataset=args.dataset)
    if p is not None:
        print(f"Probe estimate ({args.model}, {args.n} problems, {args.dataset}): p = {p:.3f}")
    else:
        print(f"No precomputed cache for {args.dataset} - estimate p from your own experiment records instead.")


if __name__ == "__main__":
    main()
