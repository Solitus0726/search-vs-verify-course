# run_experiment.py —— 实验入口 + judge 实现
# 统一采样参数 / judge 六点解析 / 增量记录（幂等+断点续跑）/ 并行（按题并行+题内串行）
# per-call seed：sha256(实验seed|dataset|problem_id|config|call_type|candidate_idx|verify_idx|retry_idx) % 2^32
#   （禁用 Python 内置 hash()——受 PYTHONHASHSEED 随机化影响跨进程不一致）
# 幂等键：dataset + problem_id + config + seed + call_type + candidate_idx + verify_idx + retry_idx（8 维，与 seed 公式同维度）
# 运行示例：
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

# 六模型 p 谱系
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

# 统一采样参数表
TEMPERATURE = 0.8
TOP_P = 0.95
N_CTX = 4096
CANDIDATE_MAX_TOKENS = 2048
JUDGE_MAX_TOKENS = 256

# 0-shot 提示模板（数据日志记录 prompt_hash）
PROMPT_MATH = "Please solve the following math problem step by step, and put your final answer after '####'."
PROMPT_GSM8K = PROMPT_MATH
PROMPT_HASH = hashlib.sha256(PROMPT_MATH.encode("utf-8")).hexdigest()[:16]

# judge 提示词（六点解析第 1 条："只输出一个 1–10 的整数分数，先打分，可附一句理由"）
JUDGE_PROMPT_TEMPLATE = (
    "你是验证器。给以下数学解答打分（1-10 分）。\n"
    "只输出一个 1–10 的整数分数，先打分，可附一句理由。\n"
    "题目：{problem}\n"
    "候选解答：{answer}"
)

# 23 个唯一 (N,M) 配置（预算 B=50）：
# 纯搜索 4（N=5/10/20/50）+ 纯验证预算内 13（N=5/M=1-9 共 9、N=10/M=1-4 共 4）
# + 超预算参考点 5（N=10/M=5-9）+ 合法组合 N=25/M=1；zone: in=预算内, over=超预算参考点
BASE_CONFIGS = [(n, 0, "in") for n in (5, 10, 20, 50)]
BASE_CONFIGS += [(5, m, "in") for m in range(1, 10)]
BASE_CONFIGS += [(10, m, "in") for m in range(1, 5)]
BASE_CONFIGS += [(10, m, "over") for m in range(5, 10)]
BASE_CONFIGS.append((25, 1, "in"))
# B=60 扩展（可选，--budget 60 追加）
B60_EXTRA = [(6, 9, "in"), (12, 4, "in"), (15, 3, "in"), (30, 1, "in"), (60, 0, "in")]
# syllabus §5.4 headline 三行（canonical 配置）
CANONICAL = {(50, 0), (25, 1), (5, 9)}

SCORE_LABEL_RE = re.compile(r"(?:分数|得分|score)\s*[:：]?\s*(-?\d+)", re.I)
CN_POINT_RE = re.compile(r"(-?\d+)\s*分")
TAIL_CHARS = 200  # 输出最后约 50 token 的保守字符口径


def config_str(n: int, m: int) -> str:
    # config 序列化（幂等键与 seed 公式共用）
    return f"verify:N={n},M={m}" if m > 0 else f"search:N={n},M=0"


def config_cost(n: int, m: int) -> int:
    # 预算恒等式：纯搜索消耗 N；纯验证消耗 N×(1+M)
    return n * (1 + m) if m > 0 else n


def assert_budget(configs: List[Tuple[int, int, str]], budget: int) -> None:
    # 预算会计断言：每个配置实际消耗 == N 或 N×(1+M)；zone 与预算关系正确
    for (n, m, zone) in configs:
        cost = config_cost(n, m)
        assert cost == (n if m == 0 else n * (1 + m)), f"恒等式失败 ({n},{m})"
        if zone == "in":
            assert cost <= budget, f"预算内配置超预算 ({n},{m}) cost={cost}"
        elif zone == "over":
            assert cost > budget, f"超预算标注错误 ({n},{m}) cost={cost}"
    # 去重检查（23 个唯一配置）
    keys = [(n, m) for (n, m, _) in configs]
    assert len(keys) == len(set(keys)), "配置表含重复项"


def build_configs(budget: int) -> List[Tuple[int, int, str]]:
    if budget == 50:
        configs = list(BASE_CONFIGS)
    elif budget == 60:
        configs = list(BASE_CONFIGS) + list(B60_EXTRA)
    else:
        raise SystemExit(f"--budget 仅支持 50（主口径）或 60（扩展），收到 {budget}")
    # zone 按预算动态归一：B=60 时 (10,5)（消耗 60）转为预算内，超预算仅剩 (10,6)-(10,9)
    configs = [(n, m, "in" if config_cost(n, m) <= budget else "over") for (n, m, _) in configs]
    assert_budget(configs, budget)
    return configs


def derive_seed(experiment_seed: int, dataset: str, problem_id: int, config: str,
                call_type: str, candidate_idx: int, verify_idx: int = 0, retry_idx: int = 0) -> int:
    # 确定性 per-call seed（sha256，禁用内置 hash()）
    key = "|".join([str(experiment_seed), dataset, str(problem_id), config, call_type,
                    str(candidate_idx), str(verify_idx), str(retry_idx)])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def idem_key(experiment_seed: int, dataset: str, problem_id: int, config: str,
             call_type: str, candidate_idx: int, verify_idx: int = 0, retry_idx: int = 0) -> str:
    # 幂等键：与 seed 公式同 8 维（judge 重试键与首次不同，不会把重试当"已完成"跳过）
    return "|".join([str(experiment_seed), dataset, str(problem_id), config, call_type,
                     str(candidate_idx), str(verify_idx), str(retry_idx)])


def clamp_score(v: int) -> int:
    # 越界值钳制到 1–10（六点解析第 3 条）
    return max(1, min(10, v))


def usage_total(usage: Optional[dict]) -> int:
    # llama.cpp usage 字段名兼容：真模型 total_tokens；stub/测试 total
    if not usage:
        return 0
    return int(usage.get("total_tokens") or usage.get("total") or 0)


def parse_judge_score(output: Optional[str]) -> Optional[int]:
    # judge 打分鲁棒解析（六点解析）：
    # 1) 优先标签匹配（分数/得分/score 后整数，全串搜索，支持负数）
    # 2) 尾部中文"X分"模式（"综合给8分"类表述，避免散文数字误抓）
    # 3) 否则仅输出最后约 50 token 内匹配整数（避免候选文本误抓）
    # 4) 越界钳制；解析失败返回 None（触发重试，六点解析第 4/5 条）
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
    # 加载 GGUF（全 GPU offload）；Qwen3 系注入 enable_thinking=False（0.3.34 无原生支持）
    from llama_cpp import Llama

    path = os.path.join(MODEL_DIR, MODEL_FILES[model])
    if not os.path.exists(path):
        raise SystemExit(f"模型文件不存在: {path}")
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
    # 单候选生成（chat 模式 + 统一采样参数 + per-call seed）
    prompt = PROMPT_MATH if dataset == "MATH-500" else PROMPT_GSM8K
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt + "\n\n" + problem}],
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=CANDIDATE_MAX_TOKENS, seed=call_seed,
    )
    text = (out["choices"][0]["message"]["content"] or "").strip()
    usage = out.get("usage") or {}
    return text, usage


def judge_call(llm: object, problem: str, answer: str, call_seed: int) -> Tuple[Optional[int], str, dict]:
    # 单次验证打分（max_tokens 256 防冗长；独立 seed 采样保证 M 次验证独立）
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
    # 记录公共字段（增量记录格式）
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
    # 增量记录系统：append + flush + fsync；启动扫描已有分片构建幂等索引
    # 原子性说明：每行记录写完立即 flush+fsync，扫描时解析失败的尾行（写一半）自动视为未完成
    #   → 重启后该调用重跑（幂等恢复兜底，与"临时文件+rename"目标一致：不丢已完成、坏行重跑）
    def __init__(self, out_dir: str, part_index: int):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"part-{part_index}.jsonl")
        self.done = {}  # idem_key -> (文件路径, 行偏移)
        self.bad_lines = 0
        for p in sorted(glob.glob(os.path.join(out_dir, "part-*.jsonl"))):
            self._scan(p)

    def _scan(self, path: str) -> None:
        # readline() 循环（迭代器预读会禁用 tell()）；偏移按字节记录（UTF-8 中文多字节）
        # errors="replace"：磁盘满/中断截断可能留下半个 UTF-8 字符（UnicodeDecodeError），
        # 替换后按 JSON 解析失败视为坏行跳过（2026-08-16 修复，重跑磁盘满中断暴露）
        with open(path, encoding="utf-8", errors="replace") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_lines += 1  # 写一半的尾行：视为未完成，重启重跑
                    offset = f.tell()
                    continue
                key = d.get("idem_key")
                if key:
                    self.done[key] = (path, offset)
                offset = f.tell()

    def try_get(self, key: str) -> Optional[dict]:
        # 幂等读取：已完成调用返回其记录（复用 output/score），未完成返回 None
        # 容错：记录被外部编辑导致偏移错位/解析失败时视为未完成（重跑覆盖），不崩溃
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
        # generate/judge 记录由 8 维幂等键生成；aggregate 记录（无 config）使用调用方预置键
        if "idem_key" not in rec:
            rec["idem_key"] = idem_key(
                rec["config"]["seed"], rec["dataset"], rec["problem_id"],
                config_str(rec["config"]["N"], rec["config"]["M"]), rec["call_type"],
                rec["candidate_idx"], rec["verify_idx"], rec["retry_idx"],
            )
        line = json.dumps(rec, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            start = f.tell()  # append 模式 tell() = 当前末尾字节偏移（UTF-8 中文多字节）
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if rec["idem_key"] not in self.done:
            self.done[rec["idem_key"]] = (self.path, start)  # 首写者胜（幂等）


def aggregate(dataset: str, n: int, m: int, experiment_seed: int,
              cands: List[str], scores: List[List[Optional[int]]], gt: str,
              tokens_actual: int) -> dict:
    # 聚合（记录即真相：投票/均分统计重建自记录字段）
    # token 双口径：上限口径 = 候选 2048 + judge 输入 2048 + judge 输出 256；
    #   实际口径 = 全部调用 usage.total 之和（judge 输入计入）
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
    # 验证聚合：每候选平均分（NaN 剔除）；并列取先出现；无有效分数判错
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
    # 题内严格串行：生成 N 候选 → 每候选验证 M 次 → 聚合（顺序约束）
    # 返回统计 {calls, judge_total, judge_fail}（解析失败率预警用）
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
                    # 首次调用：无论成败都落盘（retry_idx=0 键；失败时 score=None 触发重试）
                    s = derive_seed(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 0)
                    score, text, usage = judge_call(judge_llm, problem, cands[i], s)
                    rec = _base_record(dataset, problem_id, n, m, experiment_seed, "judge", i, v, 0, s)
                    rec.update({"score": score, "output": text, "tokens": usage})
                    recorder.write(rec)
                    stats["calls"] += 1
                    if score is None:
                        # 解析失败重采样 1 次重试（六点解析第 4 条：retry_idx=1 独立键 + 新 seed）
                        key1 = idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                        rec1 = recorder.try_get(key1)
                        if rec1 is None:
                            s1 = derive_seed(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 1)
                            score1, text1, usage1 = judge_call(judge_llm, problem, cands[i], s1)
                            rec1 = _base_record(dataset, problem_id, n, m, experiment_seed, "judge", i, v, 1, s1)
                            rec1.update({"score": score1, "output": text1, "tokens": usage1})
                            recorder.write(rec1)
                            stats["calls"] += 1
                        rec = rec1  # 聚合用重试结果（最终状态）
                else:
                    # 首次已落盘：若为失败记录，补找重试记录（retry_idx=1）作为最终状态
                    # 2026-08-16 修复（第二次自查 M13）：retry0 落盘后若在 retry1 落盘前崩溃，
                    # 恢复时 retry1 缺失——此处重放 retry1（重试幂等，可安全补执行），
                    # 避免该 judge 永久以 score=None（NaN 剔除）收尾
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
                    stats["judge_fail"] += 1  # 最终状态仍失败：NaN 剔除（六点解析第 5 条）
                tokens_actual += usage_total(rec.get("tokens"))
                if rec.get("retry_idx", 0) == 1:
                    # 重试路径：首次调用同样消耗了推理，计入实际 token 口径
                    rec0 = recorder.try_get(idem_key(experiment_seed, dataset, problem_id, cfg, "judge", i, v, 0))
                    if rec0 is not None:
                        tokens_actual += usage_total(rec0.get("tokens"))
                scores[i].append(rec["score"])
        agg = aggregate(dataset, n, m, experiment_seed, cands, scores, gt, tokens_actual)
        agg["problem_id"] = problem_id
        agg["dataset"] = dataset
        agg["idem_key"] = f"aggregate:{experiment_seed}:{dataset}:{problem_id}:{cfg}"
        # 聚合记录同样幂等：续跑时已完成题的 aggregate 不重写（避免记录文件重复行）
        if recorder.try_get(agg["idem_key"]) is None:
            recorder.write(agg)
    return stats


def configure_hf_source() -> None:
    # 数据源快路径（2026-08-25 新增，Notebook 与脚本层统一）：
    # 1) 本地缓存存在 → 完全离线（零网络、零重试、秒出）
    #    注意：huggingface_hub/datasets 的离线开关是**导入时缓存**的模块常量
    #    （本模块顶部 from evaluate import ... 已提前拉入 hub）——环境变量无效，
    #    必须直接改模块常量（Notebook 场景 import 时序正确，环境变量方案即可）
    # 2) 无缓存 → socket 探测 huggingface.co（3s）：可达用官方；不可达（如国内网络）切 hf-mirror——避免盲目重试 ~23s
    # 3) 已有 HF_DATASETS_OFFLINE/HF_HUB_OFFLINE（远程批处理）→ 不干预
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
    # 加载数据集 + 子集清单 → [(problem_id, problem, gt)]（gt 与 evaluate 同口径提取）
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
        raise SystemExit(f"未知数据集: {dataset}")
    return rows


def worker_main(worker_idx: int, task_queue, args: dict) -> None:
    # 每 worker 独立加载模型 + 写分片 part-{idx}.jsonl；动态题队列
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
        print(f"[worker{worker_idx}] problem {problem_id} (seed {seed}) 完成，本次 {st['calls']} 次调用", flush=True)
    report_judge_failure(total, f"worker{worker_idx}")
    print(f"[worker{worker_idx}] 结束，总 {total['calls']} 次调用", flush=True)


def report_judge_failure(stats: dict, tag: str) -> None:
    # 解析失败率统计（六点解析第 6 条）：>5% 时预警验证器实现质量
    if stats["judge_total"] == 0:
        return
    rate = stats["judge_fail"] / stats["judge_total"]
    msg = f"[{tag}] judge 解析失败率 {stats['judge_fail']}/{stats['judge_total']} = {rate:.1%}"
    if rate > 0.05:
        msg += " ⚠️ 超过 5% 阈值，预警：检查提示词格式或降温度重采样"
    print(msg, flush=True)


def parse_configs(items: List[str]) -> List[Tuple[int, int, str]]:
    # --configs "N=5,M=9" 列表 → (N, M) 过滤；在预算内配置表中查找 zone
    out = []
    for item in items:
        m_ = re.match(r"N=(\d+),M=(\d+)", item.strip())
        if not m_:
            raise SystemExit(f"配置格式错误（应为 N=<int>,M=<int>）: {item}")
        n, m = int(m_.group(1)), int(m_.group(2))
        found = [c for c in BASE_CONFIGS + B60_EXTRA if c[0] == n and c[1] == m]
        if not found:
            raise SystemExit(f"配置不在表中: N={n},M={m}")
        out.append(found[0])
    return out


class StubLLM:
    # 可测性伪模型（--stub，仅测试/CI 用）：固定输出 → 确定性；usage 与 llama.cpp 同字段名
    # 输出不含数字（否则 judge 尾部解析会误抓为分数）；使幂等/中断恢复/确定性/并行一致性等行为不依赖 GPU
    def create_chat_completion(self, messages, temperature=0.8, top_p=0.95, max_tokens=256, seed=0):
        text = "stub-answer"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return {"choices": [{"message": {"content": text}}], "usage": usage}


def finalize_records(out_dir: str, configs: List[Tuple[int, int, str]],
                     seeds: List[int], n_problems: int, budget: int) -> Tuple[bool, List[str]]:
    # 一键收尾（--finalize）：merge 去重 → 校验 → 报告
    # 校验内容：损坏行 / 重复键 / LLM 调用数 vs 预算会计期望 / aggregate 完整性
    # 问题分两类：warnings（已处理，如重复行已被 merge 剔除，不影响完整性）
    #           errors（待修复，如损坏行/调用不足——由 main 自动补跑，学生无需再运行任何命令）
    # 返回 (是否完整, 待修复错误列表)
    from merge_records import merge_part_files

    out_path, total, dropped, bad = merge_part_files(out_dir)
    errors, warnings = [], []
    if bad > 0:
        # 坏行是写一半的截断残片：merge 跳过，对应调用由计数校验触发补跑重建；
        # 补跑后 records 计数完整即视为通过（坏行残片不阻塞完整性判定）
        warnings.append(f"损坏行 {bad} 条（merge 已跳过；缺失调用由计数校验触发补跑重建）")
    if dropped > 0:
        warnings.append(f"重复行 {dropped} 条（merge 已剔除——旧版本 aggregate 重写残留，已修复，不影响完整性）")
    # 校验 2：LLM 调用数 vs 预算会计期望（重试只会多于期望，故用 ≥）
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
        errors.append(f"LLM 调用数 {llm_count} < 期望 {expected_llm}（未跑完，将自动补跑）")
    # 校验 3：aggregate 完整性（每题每配置一条）
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
        errors.append(f"aggregate {agg_count} < 期望 {expected_agg}（未跑完，将自动补跑）")

    print(f"[finalize] 合并完成 → {out_path}（{total} 条；剔除重复 {dropped}；损坏行 {bad}）")
    if warnings:
        print("[finalize] ℹ️ 提示：")
        for msg in warnings:
            print(f"  - {msg}")
    if errors:
        print("[finalize] ⚠️ 待修复：")
        for msg in errors:
            print(f"  - {msg}")
    else:
        print("[finalize] ✅ 校验通过：调用数一致 / aggregate 完整")
    return (len(errors) == 0), errors


def records_complete(out_dir: str, configs: List[Tuple[int, int, str]],
                     seeds: List[int], n_problems: int) -> bool:
    # 幂等早退判断：records.jsonl 已覆盖本次运行的全部调用（与 finalize_records 同口径）
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
    ap = argparse.ArgumentParser(description="实验入口（搜索 vs 验证，六模型 p 谱系）")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_FILES), help="生成模型")
    ap.add_argument("--seeds", default="0", help="实验种子列表，逗号分隔（默认 0；headline 用 0,1,2）")
    ap.add_argument("--budget", type=int, default=50, help="预算 B（默认 50；60 为扩展网格）")
    ap.add_argument("--dataset", default="MATH-500", choices=["MATH-500", "GSM8K"])
    ap.add_argument("--judge", default="self", choices=["self", "unified"], help="self=自评（弱验证器）；unified=Qwen3-4B 统一 judge（相对强验证器）")
    ap.add_argument("--workers", type=int, default=1, help="并行 worker 数（默认 1 串行；小模型可 2 进程并行）")
    ap.add_argument("--configs", nargs="*", default=None, help="配置过滤，如 N=5,M=9 N=10,M=0（默认全部）")
    ap.add_argument("--max-problems", type=int, default=0, help="每题种子只跑前 N 题（快速测试用；0=全部）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录（分片 part-N.jsonl）")
    ap.add_argument("--stub", action="store_true", help="测试模式：确定性伪模型（不加载 GPU，仅测试用）")
    ap.add_argument("--finalize", action="store_true",
                    help="运行完成后一键收尾：merge 去重 → records.jsonl + 校验报告")
    args = ap.parse_args()

    configs = parse_configs(args.configs) if args.configs else build_configs(args.budget)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = load_dataset_rows(args.dataset)
    if args.max_problems > 0:
        rows = rows[: args.max_problems]
    print(f"模型={args.model} 数据集={args.dataset} seeds={seeds} 配置数={len(configs)} "
          f"题数={len(rows)} judge={args.judge} workers={args.workers} stub={args.stub}", flush=True)

    # 幂等早退：本次运行的全部调用已存在于 records.jsonl → 不加载模型，直接收尾。
    # （录屏/复跑场景：避免为 0 次新调用加载大模型占用显存；unified judge 双模型场景尤其需要）
    # 注意：finalize 不完整（如损坏行被 merge 跳过）时**不早退**——落到正常流程加载模型自动补跑
    if args.finalize and records_complete(args.out_dir, configs, seeds, len(rows)):
        print("[fast-path] 记录已完整（0 次新调用）——跳过模型加载，直接 finalize ...", flush=True)
        try:
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
        except SystemExit:
            # records.jsonl 完整但无 part 分片（如数据由外部导入）：records 已是唯一完整来源，
            # 跳过合并直接通过（records_complete 已按同口径验证计数）
            print("[fast-path] 无分片文件——records.jsonl 已完整，跳过合并直接通过", flush=True)
            complete = True
        if complete:
            sys.exit(0)
        # 不完整（损坏行/缺 aggregate）：继续正常流程，加载模型补跑

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
                    print(f"problem {problem_id} (seed {seed}) 完成，累计 {total['calls']} 次调用", flush=True)
            report_judge_failure(total, "main")
            if not args.finalize:
                break
            # 一键闭环：校验不完整 → 自动补跑（幂等仅补缺失，最多 2 轮补跑）
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
            if complete or attempt >= 2:
                break
            print(f"[finalize] 自动补跑第 {attempt + 2} 轮（幂等，仅补缺失调用）...", flush=True)
        else:
            # 3 轮补跑后仍不完整：以退出码 1 暴露（自动化/脚本可检测）
            print("[finalize] 3 轮补跑后仍未完整（校验不通过），退出码 1", flush=True)
            sys.exit(1)
    else:
        # 按题并行 + 题内串行：任务 = (seed, problem_id, problem, gt)
        worker_args = {"model": args.model, "judge": args.judge, "dataset": args.dataset,
                       "out_dir": args.out_dir, "configs": configs, "stub": args.stub}
        for attempt in range(3):
            if args.stub:
                # stub 模式并行：无模型加载，主进程直接驱动 worker_main 逻辑（spawn 无需 GPU）
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
                print("并行执行完成，分片写入", args.out_dir, flush=True)
            if not args.finalize:
                break
            complete, _issues = finalize_records(args.out_dir, configs, seeds, len(rows), args.budget)
            if complete or attempt >= 2:
                break
            print(f"[finalize] 自动补跑第 {attempt + 2} 轮（幂等，仅补缺失调用）...", flush=True)
        else:
            # 3 轮补跑后仍不完整：以退出码 1 暴露（自动化/脚本可检测）
            print("[finalize] 3 轮补跑后仍未完整（校验不通过），退出码 1", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
