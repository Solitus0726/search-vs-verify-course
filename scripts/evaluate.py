# evaluate.py - adapter for the official evaluation equivalence relation
# extraction / voting / scoring share the same canonical form (consistent with the official logic in scripts/eval_reference/math_utils.py)
# Same equivalence relation across all three stages:
#   - extraction  extract_answer: \boxed{} -> #### -> $ boundaries (official MATH pipeline) -> fall back to the full text
#   - voting  majority_vote: group by canonicalize; on a tie, take the candidate that appears first
#   - scoring  is_correct: pred and gt go through the same extract_answer, then are compared by the official is_equiv
# Design constraint: avoid the contradiction of "voting accepts X but evaluation accepts Y"; grouping for voting and scoring
# use the same equivalence relation (is_equiv / strip_string, ported from official lm-eval hendrycks_math).

import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_reference.math_utils import is_equiv, strip_string  # official evaluation equivalence relation

DATASET_MATH = "math"
DATASET_GSM8K = "gsm8k"

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)")


def _extract_boxed(text: str) -> Optional[str]:
    # Return the brace content of the last \boxed{...} / \fbox{...} (supports nesting, e.g. \boxed{\frac{1}{2}})
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
    # Extract the final answer: \boxed{} -> #### -> $ boundaries (official MATH pipeline) -> fall back to the full text
    if not text:
        return ""
    t = text.strip()
    # 1) \boxed{...} / \fbox{...} (extract repeatedly; tolerates nested forms like $\boxed{...}$)
    for _ in range(4):
        inner = _extract_boxed(t)
        if inner is None:
            break
        t = inner.strip()
    # 2) content after #### (GSM8K convention; MATH-compatible model outputs)
    if "####" in t:
        t = t.rsplit("####", 1)[1].strip()
    # 3) $ boundaries (mirroring the official MATH scoring pipeline: take from after the first $ to before the last $)
    if dataset_type == DATASET_MATH:
        indices = [i for i, ch in enumerate(t) if ch == "$"]
        if len(indices) > 1:
            t = t[indices[0] + 1 : indices[-1]].strip()
    return t


def canonicalize(text: Optional[str], dataset_type: str = DATASET_MATH) -> str:
    # Single canonical form: extraction + official string normalization (shared by voting groups and scoring)
    # Fallback: model output is uncontrollable (e.g. "1/2miles" triggers a normalization exception); on exception, fall back to the raw text (no crash; scoring will naturally mark it wrong)
    try:
        return strip_string(extract_answer(text, dataset_type))
    except Exception:
        return (text or "").strip()


def is_correct(pred: Optional[str], gt: Optional[str], dataset_type: str = DATASET_MATH) -> bool:
    # Scoring: pred and gt go through the same extract_answer, then are compared by the official equivalence relation (gt may keep the raw \boxed format)
    return is_equiv(extract_answer(pred, dataset_type), extract_answer(gt, dataset_type))


def majority_vote(
    candidates: List[Optional[str]], dataset_type: str = DATASET_MATH
) -> Tuple[Optional[str], int, bool]:
    # Majority vote: group by canonical form (same equivalence relation as scoring)
    # Tie rule: when multiple groups are tied at the top, take the one whose candidate appears first
    # Return (winner raw text, vote count, whether it was a tie)
    if not candidates:
        return None, 0, False
    groups = {}  # canonical form -> [first candidate index, votes, first candidate raw text]
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
