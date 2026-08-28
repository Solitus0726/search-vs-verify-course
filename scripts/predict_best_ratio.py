# predict_best_ratio.py —— 实验 3 预测接口 + 自动评分 + 小探针
# 接口约定（将写入 Notebook 03）：
#   predict_best_ratio(task_meta) -> "search" | "verify"
#     输入 task_meta（dict）：
#       dataset: 数据集名（如 "GSM8K"）
#       model:   模型名（六模型之一，如 "qwen3-0.6b"）
#       p:       单候选正确率（小探针 estimate_p 的输出，学生可先跑 5-10 题估计）
#       budget:  单题预算 B（如 50）
#       judge_quality: 验证器质量估计（"weak" | "strong"，默认 "weak"）
#     输出：预测最优配比方向（"search" = 搜索/投票胜出；"verify" = 验证/均分选优胜出）
#   自动评分规则：预测方向与实测一致即得分（score_prediction，1 分/0 分）
# 学生实现占位：本文件提供契约与骨架（predict 返回 None 表示未实现），Notebook 03 中学生填充逻辑。
# 运行：python scripts/predict_best_ratio.py --model qwen3-0.6b --n 10（探针示例）

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import MODEL_FULL_NAMES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS1_DIR = os.path.join(PROJECT_ROOT, "data", "results")


def estimate_p(model: str, n: int = 10, dataset: str = "MATH-500") -> Optional[float]:
    # 小探针：从预计算的 pass@1 缓存（pass1_*.jsonl，MATH-500）估计单候选正确率 p
    # 输入：模型名 + 探针题数（5-10 题）+ 数据集；输出：p 或 None（数据缺失/模型未知时）
    # 注：GSM8K 无预计算缓存——返回 None 并提示学生实验后用自己的数据估计
    if dataset != "MATH-500":
        return None  # 仅 MATH-500 有 pass1 缓存；其他任务请实测估计
    full = MODEL_FULL_NAMES.get(model)
    if full is None:
        return None
    # pass1 文件名用短名（无 -Instruct 后缀）：pass1_Qwen3-4B.jsonl 等
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
    # 自动评分规则：预测方向与实测一致即得分（1 分），否则 0 分
    # actual_direction 由实验数据计算（搜索配置准确率 vs 验证配置准确率对比）
    if prediction not in ("search", "verify") or actual_direction not in ("search", "verify"):
        return 0
    return 1 if prediction == actual_direction else 0


def actual_direction_from_configs(configs: list) -> Optional[str]:
    # 从实验聚合结果（experiment1_*.json 的 configs）计算实测方向：
    # 比较搜索配置（M=0）与验证配置（M>0）在相同消耗量级的准确率
    searches = [c for c in configs if c["config"]["strategy"] == "search"]
    verifies = [c for c in configs if c["config"]["strategy"] == "verify"]
    if not searches or not verifies:
        return None
    # 搜索取全部平均，验证取全部平均（同预算量级下比较）
    s_acc = sum(c["accuracy"] for c in searches) / len(searches)
    v_acc = sum(c["accuracy"] for c in verifies) / len(verifies)
    return "search" if s_acc >= v_acc else "verify"


def predict_best_ratio(task_meta: Dict) -> Optional[str]:
    # 占位函数（学生实现）：输入任务元信息，输出预测方向
    # 本文件仅提供契约骨架，Notebook 03 中学生基于核心原理填充逻辑
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
