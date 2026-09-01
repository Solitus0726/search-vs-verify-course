# run_experiment.py - experiment entry point + judge implementation
# Unified sampling params / judge six-point parsing / incremental recording (idempotent + resumable) / parallelism (parallel by problem + serial within problem)
# per-call seed: sha256(experiment_seed|dataset|problem_id|config|call_type|candidate_idx|verify_idx|retry_idx) % 2^32
#   (built-in hash() disabled - it is randomized by PYTHONHASHSEED and inconsistent across processes)
# Idempotency key: dataset + problem_id + config + seed + call_type + candidate_idx + verify_idx + retry_idx (8 dims, same dimensionality as the seed formula)
# Run examples:
#   python scripts/run_experiment.py --model qwen3-0.6b --seeds 0 --configs "N=10,M=0" "N=5,M=1" --max-problems 5
#   python scripts/run_experiment.py --model qwen3-4b --seeds 0,1,2 --budget 50 --judge unified --workers 2

import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import is_correct, majority_vote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SUBSET_DIR = os.path.join(PROJECT_ROOT, "data", "subsets")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# Six-model p family
MODEL_FILES = {
    "gemma-3-1b": "gemma-3-1b-it-Q8_0.gguf",
    "phi-4-mini": "Phi-4-Mini-Instruct-Q8_0.gguf",
    "qwen3-0.6b": "Qwen3-0.6B-Instruct-Q8_0.gguf",
    "gemma-3-4b": "gemma-3-4b-it-Q8_0.gguf",
    "qwen3-1.7b": "Qwen3-1.7B-Instruct-Q8_0.gguf",
    "qwen3-4b": "Qwen3-4B-Instruct-Q8_0.gguf",
}
MODEL_FULL_NAMES = {
    "gemma-3-1b": "Gemma-3-1B-Instruct",
    "phi-4-mini": "Phi-4-Mini-Instruct",
    "qwen3-0.6b": "Qwen3-0.6B-Instruct",
    "gemma-3-4b": "Gemma-3-4B-Instruct",
    "qwen3-1.7b": "Qwen3-1.7B-Instruct",
    "qwen3-4b": "Qwen3-4B-Instruct",
}
QWEN3_MODELS = {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"}

# Unified sampling parameters
TEMPERATURE = 0.8
TOP_P = 0.95
N_CTX = 4096
CANDIDATE_MAX_TOKENS = 2048
JUDGE_MAX_TOKENS = 256

# 0-shot prompt template (prompt_hash recorded in data logs)
PROMPT_MATH = "Please solve the following math problem step by step, and put your final answer after '####'."
PROMPT_GSM8K = PROMPT_MATH
PROMPT_HASH = hashlib.sha256(PROMPT_MATH.encode("utf-8")).hexdigest()[:16]

# Judge prompt (six-point parsing rule 1: "output only an integer score from 1-10, score first, may add one sentence of reasoning")
JUDGE_PROMPT_TEMPLATE = (
    "你是验证器。给以下数学解答打分（1-10 分）。\n"
    "只输出一个 1–10 的整数分数，先打分，可附一句理由。\n"
    "题目：{problem}\n"
    "候选解答：{answer}"
)

# 23 unique (N,M) configs (budget B=50):
# pure search 4 (N=5/10/20/50) + pure verify within budget 13 (N=5/M=1-9 total 9, N=10/M=1-4 total 4)
# + over-budget reference points 5 (N=10/M=5-9) + valid combo N=25/M=1; zone: in=within budget, over=over-budget reference point
BASE_CONFIGS = [(n, 0, "in") for n in (5, 10, 20, 50)]
BASE_CONFIGS += [(5, m, "in") for m in range(1, 10)]
BASE_CONFIGS += [(10, m, "in") for m in range(1, 5)]
BASE_CONFIGS += [(10, m, "over") for m in range(5, 10)]
BASE_CONFIGS.append((25, 1, "in"))
# B=60 extension (optional, appended with --budget 60)
B60_EXTRA = [(6, 9, "in"), (12, 4, "in"), (15, 3, "in"), (30, 1, "in"), (60, 0, "in")]
# headline three (canonical configs)
CANONICAL = {(50, 0), (25, 1), (5, 9)}

SCORE_LABEL_RE = re.compile(r"(?:分数|得分|score)\s*[:：]?\s*(-?\d+)", re.I)
CN_POINT_RE = re.compile(r"(-?\d+)\s*分")
TAIL_CHARS = 200  # conservative character window for the last ~50 tokens of output


def config_str(n: int, m: int) -> str:
    # config serialization (shared by idempotency key and seed formula)
    return f"verify:N={n},M={m}" if m > 0 else f"search:N={n},M=0"


def config_cost(n: int, m: int) -> int:
    # Budget identity: pure search costs N; pure verify costs N×(1+M)
    return n * (1 + m) if m > 0 else n


def assert_budget(configs: List[Tuple[int, int, str]], budget: int) -> None:
    # Budget accounting assertion: each config's actual cost == N or N×(1+M); zone/budget relation is correct
    for (n, m, zone) in configs:
        cost = config_cost(n, m)
        assert cost == (n if m == 0 else n * (1 + m)), f"identity failed ({n},{m})"
        if zone == "in":
            assert cost <= budget, f"in-budget config exceeds budget ({n},{m}) cost={cost}"
        elif zone == "over":
            assert cost > budget, f"over-budget label wrong ({n},{m}) cost={cost}"
    # Dedup check (23 unique configs)
    keys = [(n, m) for (n, m, _) in configs]
    assert len(keys) == len(set(keys)), "config table contains duplicates"


def build_configs(budget: int) -> List[Tuple[int, int, str]]:
    if budget == 50:
        configs = list(BASE_CONFIGS)
    elif budget == 60:
        configs = list(BASE_CONFIGS) + list(B60_EXTRA)
    else:
        raise SystemExit(f"--budget only supports 50 (primary) or 60 (extension), got {budget}")
    # zone normalized dynamically by budget: at B=60, (10,5) (cost 60) becomes in-budget, only (10,6)-(10,9) remain over-budget
    configs = [(n, m, "in" if config_cost(n, m) <= budget else "over") for (n, m, _) in configs]
    assert_budget(configs, budget)
    return configs


def derive_seed(experiment_seed: int, dataset: str, problem_id: int, config: str,
                call_type: str, candidate_idx: int, verify_idx: int = 0, retry_idx: int = 0) -> int:
    # Deterministic per-call seed (sha256, built-in hash() disabled)
    key = "|".join([str(experiment_seed), dataset, str(problem_id), config, call_type,
                    str(candidate_idx), str(verify_idx), str(retry_idx)])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def idem_key(experiment_seed: int, dataset: str, problem_id: int, config: str,
             call_type: str, candidate_idx: int, verify_idx: int = 0, retry_idx: int = 0) -> str:
    # Idempotency key: same 8 dims as the seed formula (judge retry key differs from the first call, so a retry is never skipped as "done")
    return "|".join([str(experiment_seed), dataset, str(problem_id), config, call_type,
                     str(candidate_idx), str(verify_idx), str(retry_idx)])


def clamp_score(v: int) -> int:
    # Clamp out-of-range values to 1-10 (six-point parsing rule 3)
    return max(1, min(10, v))


def usage_total(usage: Optional[dict]) -> int:
    # llama.cpp usage field-name compatibility: real model total_tokens; stub/test total
    if not usage:
        return 0
    return int(usage.get("total_tokens") or usage.get("total") or 0)


def parse_judge_score(output: Optional[str]) -> Optional[int]:
    # Strict judge score parsing (six-point parsing):
    # 1) Prefer label match (integer after the score label, full-string search, supports negatives)
    # 2) Trailing "X points" pattern (phrases like "give 8 points overall", avoids false grabs of prose digits)
    # 3) Otherwise match an integer only within the last ~50 tokens of output (avoids false grabs of candidate text)
    # 4) Clamp out-of-range; return None on parse failure (triggers retry, six-point parsing rules 4/5)
    if not output:
        return None
    m = SCORE_LABEL_RE.search(output)
    if m:
        return clamp_score(int(m.group(1)))
    tail = output[-TAIL_CHARS:]
    m = CN_POINT_RE.search(tail)
    if m:
        return clamp_score(int(m.group(1)))
    m = re.search(r"\d+", tail)
    if m:
        return clamp_score(int(m.group(0)))
    return None


def load_model(model: str) -> object:
    # Load GGUF (full GPU offload); inject enable_thinking=False for the Qwen3 family (0.3.34 has no native support)
    from llama_cpp import Llama

    path = os.path.join(MODEL_DIR, MODEL_FILES[model])
    if not os.path.exists(path):
        raise SystemExit(f"model file not found: {path}")
    llm = Llama(model_path=path, n_ctx=N_CTX, n_gpu_layers=-1, verbose=False, seed=0)
    if model in QWEN3_MODELS:
        formatter = llm._chat_handlers.get("chat_template.default")
        if formatter is not None:
            def wrapped(*f_args, **f_kwargs):
                f_kwargs["enable_thinking"] = False
                return formatter(*f_args, **f_kwargs)
            llm._chat_handlers["chat_template.default"] = wrapped
    return llm


def generate_candidate(llm: object, problem: str, call_seed: int, dataset: str) -> Tuple[str, dict]:
    # Single candidate generation (chat mode + unified sampling params + per-call seed)
    prompt = PROMPT_MATH if dataset == "MATH-500" else PROMPT_GSM8K
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt + "\n\n" + problem}],
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=CANDIDATE_MAX_TOKENS, seed=call_seed,
    )
    text = (out["choices"][0]["message"]["content"] or "").strip()
    usage = out.get("usage") or {}
    return text, usage


def judge_call(llm: object, problem: str, answer: str, call_seed: int) -> Tuple[Optional[int], str, dict]:
    # Single verification scoring (max_tokens 256 to avoid verbosity; independent seed sampling keeps the M verifications independent)
    prompt = JUDGE_PROMPT_TEMPLATE.format(problem=problem, answer=answer)
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=JUDGE_MAX_TOKENS, seed=call_seed,
    )
    text = (out["choices"][0]["message"]["content"] or "").strip()
    usage = out.get("usage") or {}
    return parse_judge_score(text), text, usage


def _base_record(dataset: str, problem_id: int, n: int, m: int, experiment_seed: int,
                 call_type: str, candidate_idx: int, verify_idx: int, retry_idx: int, call_seed: int) -> dict:
    # Common record fields (incremental record format)
    return {
        "dataset": dataset,
        "problem_id": problem_id,
        "config": {"strategy": "verify" if m > 0 else "search", "N": n, "M": m, "seed": experiment_seed},
        "call_type": call_type,
        "candidate_idx": candidate_idx,
        "verify_idx": verify_idx,
        "retry_idx": retry_idx,
        "call_seed": call_seed,
        "ts": time.time(),
    }


class JsonlRecorder:
    # Incremental recording: append + flush + fsync; scans existing shards at startup to build an idempotency index
    # Atomicity note: each line is flush+fsync'd right after write; a trailing line that fails to parse during the scan (half-written) is treated as incomplete
    #   -> after restart that call is re-run (idempotent recovery fallback, same goal as "temp file + rename": never lose completed work, re-run bad lines)
    def __init__(self, out_dir: str, part_index: int):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"part-{part_index}.jsonl")
        self.done = {}  # idem_key -> (file path, line offset)
        self.bad_lines = 0
        for p in sorted(glob.glob(os.path.join(out_dir, "part-*.jsonl"))):
            self._scan(p)

    def _scan(self, path: str) -> None:
        # readline() loop (iterator prefetch disables tell()); offsets recorded in bytes (multi-byte UTF-8)
        # errors="replace": disk-full/interrupted truncation can leave half a UTF-8 char (UnicodeDecodeError);
        # after replacement the line fails JSON parsing and is skipped as a bad line (interrupted writes may truncate the last line)
        with open(path, encoding="utf-8", errors="replace") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_lines += 1  # half-written trailing line: treat as incomplete, re-run after restart
                    offset = f.tell()
                    continue
                key = d.get("idem_key")
                if key:
                    self.done[key] = (path, offset)
                offset = f.tell()

    def try_get(self, key: str) -> Optional[dict]:
        # Idempotent read: completed calls return their record (reuse output/score), incomplete ones return None
        # Fault tolerance: if a record was externally edited causing offset misalignment/parse failure, treat it as incomplete (re-run overwrites), never crash
        loc = self.done.get(key)
        if loc is None:
            return None
        path, offset = loc
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                line = f.readline()
            return json.loads(line)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def write(self, rec: dict) -> None:
        # generate/judge records use the 8-dim idempotency key; aggregate records (no config) use a caller-preset key
        if "idem_key" not in rec:
            rec["idem_key"] = idem_key(
                rec["config"]["seed"], rec["dataset"], rec["problem_id"],
                config_str(rec["config"]["N"], rec["config"]["M"]), rec["call_type"],
                rec["candidate_idx"], rec["verify_idx"], rec["retry_idx"],
            )
        line = json.dumps(rec, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            start = f.tell()  # in append mode tell() = byte offset at the current end (multi-byte UTF-8)
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if rec["idem_key"] not in self.done:
            self.done[rec["idem_key"]] = (self.path, start)  # first writer wins (idempotent)


def aggregate(dataset: str, n: int, m: int, experiment_seed: int,
              cands: List[str], scores: List[List[Optional[int]]], gt: str,
              tokens_actual: int) -> dict:
    # Aggregation (records are ground truth: vote/average stats rebuilt from record fields)
    # Two token accounting views: upper bound = candidate 2048 + judge input 2048 + judge output 256;
    #   actual = sum of usage.total over all calls (judge input included)
    dataset_type = "math" if dataset == "MATH-500" else "gsm8k"
    tokens_upper = n * CANDIDATE_MAX_TOKENS + (n * m * (CANDIDATE_MAX_TOKENS + JUDGE_MAX_TOKENS) if m > 0 else 0)
    base = {
        "call_type": "aggregate", "strategy": "verify" if m > 0 else "search",
        "N": n, "M": m, "seed": experiment_seed,
        "tokens": {"upper_bound": tokens_upper, "actual": tokens_actual},
        "canonical": (n, m) in CANONICAL,
    }
    if m == 0:
        winner, votes, tie = majority_vote(cands, dataset_type)
        correct = is_correct(winner, gt, dataset_type) if winner is not None else False
        base.update({"winner": winner, "votes": votes, "tie": tie, "correct": correct})
        return base
    # Verify aggregation: per-candidate average (NaN dropped); ties pick the first; no valid score -> wrong
    best_i, best_avg = None, None
    avgs = []
    for i, sc in enumerate(scores):
        valid = [x for x in sc if x is not None]
        avg = (sum(valid) / len(valid)) if valid else None
        avgs.append(avg)
        if avg is not None and (best_avg is None or avg > best_avg):
            best_i, best_avg = i, avg
    if best_i is None:
        winner, correct = None, False
    else:
        winner = cands[best_i]
        correct = is_correct(winner, gt, dataset_type)
    base.update({"winner": winner, "avg_scores": avgs, "correct": correct})
    return base


def process_problem(llm: object, judge_llm: object, recorder: JsonlRecorder,
                    experiment_seed: int, dataset: str, problem_id: int,
                    problem: str, gt: str, configs: List[Tuple[int, int, str]]) -> dict:
    # Strictly serial within a problem: generate N candidates -> verify each M times -> aggregate (ordering constraint)
    # Returns stats {calls, judge_total, judge_fail} (used for the parse-failure-rate warning)
    stats = {"calls": 0, "judge_total": 0, "judge_fail": 0}
    tokens_actual = 0
    for (n, m, _zone) in configs:
        cfg = config_str(n, m)
        cands: List[str] = []
        for i in range(n):
            key = idem_key(experiment_seed, dataset, problem_id, cfg, "generate", i)
            rec = recorder.try_get(key)
            if rec is None:
                s = derive_seed(experiment_seed, dataset, problem_id, cfg, "generate", i)
                text, usage = generate_candidate(llm, problem, s, dataset)
                rec = _base_record(dataset, problem_id, n, m, experiment_seed, "generate", i, 0, 0, s)
                rec.update({"output": text, "tokens": usage})
                recorder.write(rec)
                stats["calls"] += 1
            tokens_actual += usage_total(rec.get("tokens"))
            cands.append(rec["output"])
        scores: List[List[Optional[int]]] = [[] for _ in range(n)]
        for i in range(n):
            for v in range(m):
                key = idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 0)
                rec = recorder.try_get(key)
                if rec is None:
                    # First call: persist regardless of outcome (retry_idx=0 key; on failure score=None triggers a retry)
                    s = derive_seed(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 0)
                    score, text, usage = judge_call(judge_llm, problem, cands[i], s)
                    rec = _base_record(dataset, problem_id, n, m, experiment_seed, "judge", i, v, 0, s)
                    rec.update({"score": score, "output": text, "tokens": usage})
                    recorder.write(rec)
                    stats["calls"] += 1
                    if score is None:
                        # On parse failure, resample once as a retry (six-point parsing rule 4: retry_idx=1 separate key + new seed)
                        key1 = idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                        rec1 = recorder.try_get(key1)
                        if rec1 is None:
                            s1 = derive_seed(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                            score1, text1, usage1 = judge_call(judge_llm, problem, cands[i], s1)
                            rec1 = _base_record(dataset, problem_id, n, m, experiment_seed, "judge", i, v, 1, s1)
                            rec1.update({"score": score1, "output": text1, "tokens": usage1})
                            recorder.write(rec1)
                            stats["calls"] += 1
                        rec = rec1  # use the retry result for aggregation (final state)
                else:
                    # First call already persisted: if it failed, look up the retry record (retry_idx=1) as the final state
                    # If a crash happened after retry0 was written but before retry1,
                    # retry1 would be missing on recovery -- replay retry1 here (retries are idempotent, safe to re-run),
                    # so this judge never ends up permanently scored None (dropped as NaN)
                    if rec["score"] is None:
                        key1 = idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                        rec1 = recorder.try_get(key1)
                        if rec1 is None:
                            s1 = derive_seed(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                            score1, text1, usage1 = judge_call(judge_llm, problem, cands[i], s1)
                            rec1 = _base_record(dataset, problem_id, n, m, experiment_seed, "judge", i, v, 1, s1)
                            rec1.update({"score": score1, "output": text1, "tokens": usage1})
                            recorder.write(rec1)
                            stats["calls"] += 1
                        rec = rec1
                stats["judge_total"] += 1
                if rec["score"] is None:
                    stats["judge_fail"] += 1  # final state still failed: dropped as NaN (six-point parsing rule 5)
                tokens_actual += usage_total(rec.get("tokens"))
                if rec.get("retry_idx", 0) == 1:
                    # Retry path: the first call also consumed inference, count it in the actual token view
                    rec0 = recorder.try_get(idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 0))
                    if rec0 is not None:
                        tokens_actual += usage_total(rec0.get("tokens"))
                scores[i].append(rec["score"])
        agg = aggregate(dataset, n, m, experiment_seed, cands, scores, gt, tokens_actual)
        agg["problem_id"] = problem_id
        agg["dataset"] = dataset
        agg["idem_key"] = f"aggregate:{experiment_seed}:{dataset}:{problem_id}:{cfg}"
        # Aggregate records are idempotent too: on resume, the aggregate of a finished problem is not rewritten (avoids duplicate lines)
        if recorder.try_get(agg["idem_key"]) is None:
            recorder.write(agg)
    return stats


def configure_hf_source() -> None:
    # Dataset source fast path (shared by the notebooks and the CLI):
    # 1) Local cache exists -> fully offline (zero network, zero retries, instant)
    #    Note: the huggingface_hub/datasets offline switches are module constants cached **at import time**
    #    (from evaluate import ... at the top of this module already pulled in hub) -- env vars have no effect,
    #    so the module constants must be patched directly (in the Notebook scenario import ordering is correct, env vars suffice)
    # 2) No cache -> socket-probe huggingface.co (3s): if reachable use the official source; if unreachable (e.g. CN network) switch to hf-mirror -- avoids blind retries of ~23s
    # 3) HF_DATASETS_OFFLINE/HF_HUB_OFFLINE already set (remote batch) -> do not interfere
    import socket
    if os.environ.get("HF_DATASETS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE"):
        return
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    ds_cache = os.path.join(hf_home, "datasets")
    if os.path.isdir(ds_cache):
        if any("math-500" in n.lower() or "gsm8k" in n.lower() for n in os.listdir(ds_cache)):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                import huggingface_hub.constants as _hub_const
                _hub_const.HF_HUB_OFFLINE = True
            except ImportError:
                pass
            try:
                import datasets.config as _ds_config
                _ds_config.HF_DATASETS_OFFLINE = True
            except ImportError:
                pass
            return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_dataset_rows(dataset: str) -> List[Tuple[int, str, str]]:
    # Load dataset + subset list -> [(problem_id, problem, gt)] (gt extracted with the same convention as evaluate)
    configure_hf_source()
    import datasets as hf_datasets
    from eval_reference.math_utils import last_boxed_only_string, remove_boxed

    if dataset == "MATH-500":
        with open(os.path.join(SUBSET_DIR, "math500_subset.json"), encoding="utf-8") as f:
            sub = json.load(f)
        ds = hf_datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = []
        for pid in sub["problem_ids"]:
            doc = ds[pid]
            ans = remove_boxed(last_boxed_only_string(doc["solution"]))
            rows.append((pid, doc["problem"], ans))
    elif dataset == "GSM8K":
        with open(os.path.join(SUBSET_DIR, "gsm8k_subset.json"), encoding="utf-8") as f:
            sub = json.load(f)
        ds = hf_datasets.load_dataset("openai/gsm8k", "main", split="train")
        rows = []
        for idx in sub["indices"]:
            doc = ds[idx]
            rows.append((idx, doc["question"], doc["answer"]))
    else:
        raise SystemExit(f"unknown dataset: {dataset}")
    return rows


def worker_main(worker_idx: int, task_queue, args: dict) -> None:
    # Each worker loads its own model and writes shard part-{idx}.jsonl; dynamic problem queue
    llm = StubLLM() if args.get("stub") else load_model(args["model"])
    judge_llm = llm if args["judge"] == "self" or args.get("stub") else load_model("qwen3-4b")
    recorder = JsonlRecorder(args["out_dir"], worker_idx)
    total = {"calls": 0, "judge_total": 0, "judge_fail": 0}
    while True:
        try:
            task = task_queue.get_nowait()
        except Exception:
            break
        seed, problem_id, problem, gt = task
        st = process_problem(llm, judge_llm, recorder, seed, args["dataset"],
                             problem_id, problem, gt, args["configs"])
        for k in total:
            total[k] += st[k]
        print(f"[worker{worker_idx}] problem {problem_id} (seed {seed}) done, {st['calls']} calls this round", flush=True)
    report_judge_failure(total, f"worker{worker_idx}")
    print(f"[worker{worker_idx}] finished, {total['calls']} calls total", flush=True)


def report_judge_failure(stats: dict, tag: str) -> None:
    # Parse-failure-rate stats (six-point parsing rule 6): warn about verifier implementation quality when >5%
    if stats["judge_total"] == 0:
        return
    rate = stats["judge_fail"] / stats["judge_total"]
    msg = f"[{tag}] judge parse failure rate {stats['judge_fail']}/{stats['judge_total']} = {rate:.1%}"
    if rate > 0.05:
        msg += " ⚠️ above 5% threshold, warning: check prompt format or lower temperature and resample"
    print(msg, flush=True)


def parse_configs(items: List[str]) -> List[Tuple[int, int, str]]:
    # --configs "N=5,M=9" list -> (N, M) filter; look up the zone in the config table
    out = []
    for item in items:
        m_ = re.match(r"N=(\d+),M=(\d+)", item.strip())
        if not m_:
            raise SystemExit(f"bad config format (expected N=<int>,M=<int>): {item}")
        n, m = int(m_.group(1)), int(m_.group(2))
        found = [c for c in BASE_CONFIGS + B60_EXTRA if c[0] == n and c[1] == m]
        if not found:
            raise SystemExit(f"config not in table: N={n},M={m}")
        out.append(found[0])
    return out


class StubLLM:
    # Testability stub model (--stub, test/CI only): fixed output -> deterministic; usage uses the same field names as llama.cpp
    # Output contains no digits (otherwise judge tail parsing would grab them as scores); makes idempotency/crash recovery/determinism/parallel consistency GPU-independent
    def create_chat_completion(self, messages, temperature=0.8, top_p=0.95, max_tokens=256, seed=0):
        text = "stub-answer"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return {"choices": [{"message": {"content": text}}], "usage": usage}


def finalize_records(out_dir: str, configs: List[Tuple[int, int, str]],
                     seeds: List[int], n_problems: int, budget: int) -> Tuple[bool, List[str]]:
    # One-shot finalization (--finalize): merge dedup -> validation -> report
    # Validation covers: corrupted lines / duplicate keys / LLM call count vs budget-accounting expectation / aggregate completeness
    # Issues fall into two classes: warnings (already handled, e.g. duplicate lines removed by merge, integrity unaffected)
    #                             errors (need fixing, e.g. corrupted lines/insufficient calls -- main re-runs automatically, students run nothing more)
    # Returns (is_complete, list of errors to fix)
    from merge_records import merge_part_files

    out_path, total, dropped, bad = merge_part_files(out_dir)
    errors, warnings = [], []
    if bad > 0:
        # Bad lines are half-written truncation fragments: merge skips them; the missing calls are rebuilt via the count-check re-run;
        # after the re-run, complete records counts pass the check (bad-line fragments do not block the completeness verdict)
        warnings.append(f"{bad} corrupted lines (skipped by merge; missing calls rebuilt via count-check re-run)")
    if dropped > 0:
        warnings.append(f"{dropped} duplicate lines (removed by merge -- residue of old-version aggregate rewrites, fixed, integrity unaffected)")
    # Check 2: LLM call count vs budget-accounting expectation (retries only add calls, so use >=)
    expected_llm = sum(config_cost(n, m) for (n, m, _) in configs) * len(seeds) * n_problems
    llm_count = 0
    with open(out_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("call_type") in ("generate", "judge"):
                llm_count += 1
    if llm_count < expected_llm:
        errors.append(f"LLM call count {llm_count} < expected {expected_llm} (not finished, will re-run automatically)")
    # Check 3: aggregate completeness (one per problem per config)
    expected_agg = len(configs) * len(seeds) * n_problems
    agg_count = 0
    with open(out_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("call_type") == "aggregate":
                agg_count += 1
    if agg_count < expected_agg:
        errors.append(f"aggregate {agg_count} < expected {expected_agg} (not finished, will re-run automatically)")

    print(f"[finalize] merge done -> {out_path} ({total} records; {dropped} duplicates removed; {bad} corrupted lines)")
    if warnings:
        print("[finalize] ℹ️ notes:")
        for msg in warnings:
            print(f"  - {msg}")
    if errors:
        print("[finalize] ⚠️ to fix:")
        for msg in errors:
            print(f"  - {msg}")
    else:
        print("[finalize] ✅ validation passed: call counts match / aggregate complete")
    return (len(errors) == 0), errors


def records_complete(out_dir: str, configs: List[Tuple[int, int, str]],
                     seeds: List[int], n_problems: int) -> bool:
    # Idempotent early-exit check: records.jsonl already covers every call of this run (same convention as finalize_records)
    path = os.path.join(out_dir, "records.jsonl")
    if not os.path.exists(path):
        return False
    expected_llm = sum(config_cost(n, m) for (n, m, _) in configs) * len(seeds) * n_problems
    expected_agg = len(configs) * len(seeds) * n_problems
    llm_count = agg_count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ct = d.get("call_type")
            if ct in ("generate", "judge"):
                llm_count += 1
            elif ct == "aggregate":
                agg_count += 1
    return llm_count >= expected_llm and agg_count >= expected_agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment entry point (search vs verify, six-model p family)")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_FILES), help="generation model")
    ap.add_argument("--seeds", default="0", help="comma-separated experiment seeds (default 0; headline uses 0,1,2)")
    ap.add_argument("--budget", type=int, default=50, help="budget B (default 50; 60 is the extension grid)")
    ap.add_argument("--dataset", default="MATH-500", choices=["MATH-500", "GSM8K"])
    ap.add_argument("--judge", default="self", choices=["self", "unified"], help="self=self-evaluation (weak verifier); unified=Qwen3-4B unified judge (relatively strong verifier)")
    ap.add_argument("--workers", type=int, default=1, help="number of parallel workers (default 1, serial; small models can run 2 processes in parallel)")
    ap.add_argument("--configs", nargs="*", default=None, help="config filter, e.g. N=5,M=9 N=10,M=0 (default: all)")
    ap.add_argument("--max-problems", type=int, default=0, help="only run the first N problems per seed (for quick tests; 0=all)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory (shards part-N.jsonl)")
    ap.add_argument("--stub", action="store_true", help="test mode: deterministic stub model (no GPU load, test only)")
    ap.add_argument("--finalize", action="store_true",
                    help="one-shot finalization after the run: merge dedup -> records.jsonl + validation report")
    args = ap.parse_args()

    configs = parse_configs(args.configs) if args.configs else build_configs(args.budget)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = load_dataset_rows(args.dataset)
    if args.max_problems > 0:
        rows = rows[: args.max_problems]
    print(f"model={args.model} dataset={args.dataset} seeds={seeds} configs={len(configs)} "
          f"problems={len(rows)} judge={args.judge} workers={args.workers} stub={args.stub}", flush=True)

    # Idempotent early exit: if every call of this run already exists in records.jsonl -> skip model loading and finalize directly.
    # (screen-recording/re-run scenario: avoids loading a big model and wasting VRAM for 0 new calls; especially needed with the unified judge's two models)
    # Note: if finalize is incomplete (e.g. corrupted lines skipped by merge) do **not** exit early -- fall through to the normal flow and re-run automatically
    if args.finalize and records_complete(args.out_dir, configs, seeds, len(rows)):
        print("[fast-path] records complete (0 new calls) -- skipping model loading, finalizing directly ...", flush=True)
        try:
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
        except SystemExit:
            # records.jsonl is complete but no part shards exist (e.g. data imported externally): records are the only complete source,
            # skip merge and pass directly (records_complete already validated counts with the same convention)
            print("[fast-path] no shard files -- records.jsonl is complete, skipping merge and passing", flush=True)
            complete = True
        if complete:
            sys.exit(0)
        # Incomplete (corrupted lines/missing aggregate): continue the normal flow, load the model and re-run

    if args.workers <= 1:
        llm = StubLLM() if args.stub else load_model(args.model)
        judge_llm = llm if args.judge == "self" or args.stub else load_model("qwen3-4b")
        recorder = JsonlRecorder(args.out_dir, 0)
        for attempt in range(3):
            total = {"calls": 0, "judge_total": 0, "judge_fail": 0}
            for seed in seeds:
                for (problem_id, problem, gt) in rows:
                    st = process_problem(llm, judge_llm, recorder, seed, args.dataset,
                                         problem_id, problem, gt, configs)
                    for k in total:
                        total[k] += st[k]
                    print(f"problem {problem_id} (seed {seed}) done, {total['calls']} calls total", flush=True)
            report_judge_failure(total, "main")
            if not args.finalize:
                break
            # One-shot loop: validation incomplete -> auto re-run (idempotent, only fills gaps, at most 2 extra rounds)
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
            if complete or attempt >= 2:
                break
            print(f"[finalize] auto re-run round {attempt + 2} (idempotent, only missing calls)...", flush=True)
        else:
            # Still incomplete after 3 re-run rounds: expose via exit code 1 (detectable by automation/scripts)
            print("[finalize] still incomplete after 3 re-run rounds (validation failed), exit code 1", flush=True)
            sys.exit(1)
    else:
        # Parallel by problem + serial within a problem: task = (seed, problem_id, problem, gt)
        worker_args = {"model": args.model, "judge": args.judge, "dataset": args.dataset,
                       "out_dir": args.out_dir, "configs": configs, "stub": args.stub}
        for attempt in range(3):
            if args.stub:
                # Stub-mode parallelism: no model loading; the main process drives worker_main logic directly (spawn needs no GPU)
                task_queue = multiprocessing.Queue()
                for seed in seeds:
                    for (problem_id, problem, gt) in rows:
                        task_queue.put((seed, problem_id, problem, gt))
                procs = [multiprocessing.Process(target=worker_main, args=(i, task_queue, worker_args))
                         for i in range(args.workers)]
                for p in procs:
                    p.start()
                for p in procs:
                    p.join()
            else:
                ctx = multiprocessing.get_context("spawn")
                task_queue = ctx.Queue()
                for seed in seeds:
                    for (problem_id, problem, gt) in rows:
                        task_queue.put((seed, problem_id, problem, gt))
                procs = [ctx.Process(target=worker_main, args=(i, task_queue, worker_args))
                         for i in range(args.workers)]
                for p in procs:
                    p.start()
                for p in procs:
                    p.join()
                print("parallel execution finished, shards written to", args.out_dir, flush=True)
            if not args.finalize:
                break
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
            if complete or attempt >= 2:
                break
            print(f"[finalize] auto re-run round {attempt + 2} (idempotent, only missing calls)...", flush=True)
        else:
            # Still incomplete after 3 re-run rounds: expose via exit code 1 (detectable by automation/scripts)
            print("[finalize] still incomplete after 3 re-run rounds (validation failed), exit code 1", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
