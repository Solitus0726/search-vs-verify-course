# analyze_experiment.py - Experiment-1 aggregation statistics
# Records are the truth: rebuild per-(N,M) config statistics from the merged records.jsonl (or shards)
# Output: experiment1_<model>.json - one entry per config, fields match the record format
#   config/{strategy,N,M} accuracy std runs seeds budget dataset model engine quantization
#   gguf_file gguf_sha256 sampling canonical
# Run: python scripts/analyze_experiment.py --records data/results/records.jsonl --model qwen3-4b --budget 50
#   (--records may also point at a shard directory data/results/, which scans part-*.jsonl)

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (CANONICAL, CANDIDATE_MAX_TOKENS, JUDGE_MAX_TOKENS,
                            MODEL_DIR, MODEL_FILES, MODEL_FULL_NAMES, QWEN3_MODELS,
                            TEMPERATURE, TOP_P, N_CTX, config_cost)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HASH_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "results", "model_hashes.json")

ENGINE = "llama-cpp"
QUANTIZATION = "GGUF-Q8_0"
CHUNK = 1024 * 1024  # hash chunk size

# Six-model p family (measured pass@1, 100-problem subset) - the model field carries p
MODEL_P = {"gemma-3-1b": 0.29, "gemma-3-4b": 0.48, "phi-4-mini": 0.39,
           "qwen3-0.6b": 0.41, "qwen3-1.7b": 0.62, "qwen3-4b": 0.70}


def sha256_file(path: str) -> str:
    # Model file hash (cached to model_hashes.json to avoid re-hashing large files)
    cache = {}
    if os.path.exists(HASH_CACHE_PATH):
        with open(HASH_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    if os.path.basename(path) in cache:
        return cache[os.path.basename(path)]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    cache[os.path.basename(path)] = digest
    with open(HASH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return digest


def load_records(records_arg: str) -> dict:
    # Read records (file or shard dir), dedup by idem_key (first writer wins, same semantics as merge_records)
    paths = []
    if os.path.isdir(records_arg):
        paths = sorted(glob.glob(os.path.join(records_arg, "part-*.jsonl")))
        if not paths:
            cand = os.path.join(records_arg, "records.jsonl")
            if os.path.exists(cand):
                paths = [cand]
    elif os.path.exists(records_arg):
        paths = [records_arg]
    recs = {}
    if not paths:
        # Empty dir/file: return empty records (notebooks show "no data", no crash)
        return recs
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = d.get("idem_key")
                if k and k not in recs:
                    recs[k] = d
    return recs


def main(argv: Optional[List[str]] = None) -> None:
    # argv may pass an argument list (notebooks call analyze_main([...]) in-process);
    # default: read sys.argv (CLI)
    ap = argparse.ArgumentParser(description="Experiment-1 aggregation (records are the truth: statistics rebuilt from generate/judge records)")
    ap.add_argument("--records", required=True, help="records.jsonl file or shard directory")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_FILES))
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--dataset", default="MATH-500")
    ap.add_argument("--seeds", default=None, help="comma-separated seed list (default: all); use --seeds 0 to match the shipped cache")
    ap.add_argument("--out", default=None, help="output JSON path (default data/results/experiment1_<model>.json)")
    args = ap.parse_args(argv)

    recs = load_records(args.records)
    from evaluate import is_correct, majority_vote
    from run_experiment import load_dataset_rows, usage_total

    dataset_type = "math" if args.dataset == "MATH-500" else "gsm8k"
    gt_by_pid = {pid: gt for (pid, _p, gt) in load_dataset_rows(args.dataset)}
    # Rebuild grouped by (strategy, N, M, seed, problem_id)
    llm = [r for r in recs.values() if r["call_type"] in ("generate", "judge")]
    groups = {}
    for r in llm:
        key = (r["config"]["strategy"], r["config"]["N"], r["config"]["M"], r["config"]["seed"], r["problem_id"])
        groups.setdefault(key, {"cands": {}, "judges": {}})
        if r["call_type"] == "generate":
            groups[key]["cands"][r["candidate_idx"]] = r
        else:
            # Judge final state: among multiple (candidate, verify) entries, the largest retry_idx wins
            jk = (r["candidate_idx"], r["verify_idx"])
            old = groups[key]["judges"].get(jk)
            if old is None or r["retry_idx"] > old["retry_idx"]:
                groups[key]["judges"][jk] = r
    seed_set = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    by_config = {}
    for (strategy, n, m, seed, pid), g in groups.items():
        if seed_set is not None and seed not in seed_set:
            continue
        cands = [g["cands"][i] for i in sorted(g["cands"])]
        if not cands:
            continue
        if m == 0:
            winner, _votes, _tie = majority_vote([c["output"] for c in cands], dataset_type)
            correct = is_correct(winner, gt_by_pid.get(pid), dataset_type) if winner is not None else False
        else:
            scores = {}
            for (ci, vi), j in sorted(g["judges"].items()):
                scores.setdefault(ci, []).append(j["score"])
            best_i, best_avg = None, None
            for ci in sorted(scores):
                valid = [s for s in scores[ci] if s is not None]
                avg = (sum(valid) / len(valid)) if valid else None
                if avg is not None and (best_avg is None or avg > best_avg):
                    best_i, best_avg = ci, avg
            if best_i is None:
                winner, correct = None, False
            else:
                winner = cands[best_i]["output"]
                correct = is_correct(winner, gt_by_pid.get(pid), dataset_type)
        key = (strategy, n, m)
        by_config.setdefault(key, []).append(
            {"seed": seed, "pid": pid, "correct": correct, "winner": winner,
             "cands": cands, "scores": g["judges"]})
    # tokens.actual: sum over all LLM call records (incl. retries) per config - not derived from final records
    # Filter by seed_set like by_config (else stale seeds in out-dir inflate the token field)
    tokens_by_config = {}
    for r in llm:
        if seed_set is not None and r["config"]["seed"] not in seed_set:
            continue
        key = (r["config"]["strategy"], r["config"]["N"], r["config"]["M"])
        tokens_by_config[key] = tokens_by_config.get(key, 0) + usage_total(r.get("tokens"))

    gguf = MODEL_FILES[args.model]
    gguf_path = os.path.join(MODEL_DIR, gguf)
    full_name = MODEL_FULL_NAMES[args.model]
    enable_thinking = args.model not in QWEN3_MODELS

    results = []
    for (strategy, n, m), items in sorted(by_config.items()):
        seeds = sorted({a["seed"] for a in items})
        corrects = [a["correct"] for a in items]
        accuracy = sum(corrects) / len(corrects) if corrects else 0.0
        std = statistics.stdev(corrects) if len(corrects) >= 2 else None
        tokens_upper = n * CANDIDATE_MAX_TOKENS + (n * m * (CANDIDATE_MAX_TOKENS + JUDGE_MAX_TOKENS) if m > 0 else 0)
        tokens_actual = tokens_by_config.get((strategy, n, m), 0)
        results.append({
            "config": {"strategy": strategy, "N": n, "M": m},
            "accuracy": round(accuracy, 4),
            "std": round(std, 4) if std is not None else None,
            "runs": len(seeds),
            "seeds": seeds,
            "budget": args.budget,
            "dataset": args.dataset,
            "model": "%s (p=%.2f)" % (full_name, MODEL_P.get(args.model, 0.0)),
            "engine": ENGINE,
            "quantization": QUANTIZATION,
            "gguf_file": gguf,
            "gguf_sha256": sha256_file(gguf_path),
            "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P,
                         "n_ctx": N_CTX, "enable_thinking": enable_thinking},
            "canonical": (n, m) in CANONICAL,
            "cost": config_cost(n, m),
            "n_problems": len(items),
            "tokens": {"upper_bound": tokens_upper, "actual": tokens_actual},
        })

    out = args.out or os.path.join(PROJECT_ROOT, "data", "results", f"experiment1_{args.model}.json")
    if not results:
        # No config data (empty dir / no records): skip writing an empty aggregate (notebooks show "no data",
        # so an empty exp1.json is not mistaken for valid data, e.g. by the glob in the export cell)
        print("Aggregation done: 0 configs (no records - run the experiment first)")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": "%s (p=%.2f)" % (full_name, MODEL_P.get(args.model, 0.0)),
                   "budget": args.budget, "dataset": args.dataset,
                   "configs": results}, f, ensure_ascii=False, indent=2)
    print(f"Aggregation done: {len(results)} configs -> {out}")
    for r in results:
        print(f"  {r['config']['strategy']} N={r['config']['N']:>2} M={r['config']['M']:>2} "
              f"acc={r['accuracy']:.3f} runs={r['runs']} problems={r['n_problems']}")


if __name__ == "__main__":
    main()
