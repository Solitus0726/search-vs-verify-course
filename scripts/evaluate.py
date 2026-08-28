# evaluate.py —— 官方评估等价关系适配
# 抽取 / 投票 / 判分共用同一 canonical form（与 scripts/eval_reference/math_utils.py 官方逻辑一致）
# 三环节同一等价关系：
#   - 抽取  extract_answer：\boxed{} -> #### -> $ 边界（官方 MATH 流程）-> 全文回退
#   - 投票  majority_vote：按 canonicalize 分组，平票取候选首次出现顺序最先者
#   - 判分  is_correct：pred 与 gt 经同一 extract_answer 后由官方 is_equiv 判等
# 设计约束：避免"投票认 X 而评估认 Y"的矛盾——投票归组与判分判等
# 使用同一等价关系（is_equiv / strip_string，官方 lm-eval hendrycks_math 移植版）。

import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_reference.math_utils import is_equiv, strip_string  # 官方评估等价关系

DATASET_MATH = "math"
DATASET_GSM8K = "gsm8k"

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)")


def _extract_boxed(text: str) -> Optional[str]:
    # 返回最后一个 \boxed{...} / \fbox{...} 的花括号内内容（支持嵌套，如 \boxed{\frac{1}{2}}）
    idx = -1
    for m in _BOXED_RE.finditer(text):
        idx = m.start()
    if idx < 0:
        return None
    j = text.find("{", idx)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def extract_answer(text: Optional[str], dataset_type: str = DATASET_MATH) -> str:
    # 提取最终答案：\boxed{} -> #### -> $ 边界（官方 MATH 流程）-> 全文回退
    if not text:
        return ""
    t = text.strip()
    # 1) \boxed{...} / \fbox{...}（循环提取，兼容 $\boxed{...}$ 等嵌套书写）
    for _ in range(4):
        inner = _extract_boxed(t)
        if inner is None:
            break
        t = inner.strip()
    # 2) #### 后内容（GSM8K 约定，MATH 兼容模型输出）
    if "####" in t:
        t = t.rsplit("####", 1)[1].strip()
    # 3) $ 边界（官方 MATH 判分流程复刻：取第一个 $ 后到最后一个 $ 前）
    if dataset_type == DATASET_MATH:
        indices = [i for i, ch in enumerate(t) if ch == "$"]
        if len(indices) > 1:
            t = t[indices[0] + 1 : indices[-1]].strip()
    return t


def canonicalize(text: Optional[str], dataset_type: str = DATASET_MATH) -> str:
    # 唯一 canonical form：抽取 + 官方字符串归一化（投票分组与判分共用）
    # 兜底：模型输出不可控（如 "1/2miles" 触发归一化异常），异常时回退原文（不崩溃，判分自然判错）
    try:
        return strip_string(extract_answer(text, dataset_type))
    except Exception:
        return (text or "").strip()


def is_correct(pred: Optional[str], gt: Optional[str], dataset_type: str = DATASET_MATH) -> bool:
    # 判分：pred 与 gt 经同一 extract_answer 后由官方等价关系判等（gt 兼容 \boxed 原始格式）
    return is_equiv(extract_answer(pred, dataset_type), extract_answer(gt, dataset_type))


def majority_vote(
    candidates: List[Optional[str]], dataset_type: str = DATASET_MATH
) -> Tuple[Optional[str], int, bool]:
    # 多数投票：按 canonical form 分组（与判分同一等价关系）
    # 平票规则：多个并列最高票组时，取候选首次出现顺序最先者
    # 返回 (胜者原始文本, 得票数, 是否平票)
    if not candidates:
        return None, 0, False
    groups = {}  # canonical form -> [首次候选下标, 票数, 首次候选原始文本]
    for i, cand in enumerate(candidates):
        key = canonicalize(cand, dataset_type)
        if key not in groups:
            groups[key] = [i, 0, cand]
        groups[key][1] += 1
    max_votes = max(g[1] for g in groups.values())
    leaders = [g for g in groups.values() if g[1] == max_votes]
    leaders.sort(key=lambda g: g[0])
    winner = leaders[0]
    return winner[2], winner[1], len(leaders) > 1
