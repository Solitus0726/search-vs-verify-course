# Test-Time Compute Allocation: Search vs. Verify

A self-contained teaching unit for the **NeurIPS 2026 Education Track**: given a
fixed inference budget, how should a model spend it — generate many candidate
answers and take the most frequent one (*search*), or generate fewer answers and
check each one with a judge (*verify*)?

**Central claim (conditional, measured):** under a *weak verifier* (the model
grading its own answers) and a *moderate budget* (50 inference calls per problem),
search beats verify — measured +9.3 to +12.3 pp on three models (100 problems,
3 seeds, 95% CIs exclude 0). The materials teach *when* this holds and how to
find the best split on your own model.

## Repository layout

| Path | What it is |
|------|------------|
| `notebooks/` | Five Jupyter notebooks (00 self-check → 04 baseline review) — the hands-on core |
| `video/full_1080p.mp4` | Five-minute concept video (English narration, hardcoded subtitles, SRT included) |
| `slides/策略讲解.pptx`, `slides/策略讲解.pdf` | Lecture slides: the math behind the trade-off |
| `syllabus.md` | Course syllabus: concept, leveling, learning objectives, core principle, worked example, decision tree, progressive experiments |
| `paper/` | The two-page submission paper (PDF + LaTeX source) |
| `figures/` | Learning-path diagram and strategy comparison diagram |
| `data/cache_subset/` | Precomputed experiment results — the full course runs without a GPU |
| `data/subsets/` | Problem ID subsets for MATH-500 and GSM8K |
| `scripts/` | Experiment runner, answer evaluator, split predictor, plotting utilities |
| `references/` | The five papers the course builds on |
| `requirements-*.txt` | Dependencies for the GPU tier and the CPU tier |

## Learning path

```
00 Self-check (5 min)
   └─→ 01 Budget-accuracy curves → 02 Optimal mix ratio → 03 New-task prediction
       → 04 Baseline review (advanced)
```

Notebook 00 is a five-minute self-check: learners who are new to transformer
inference get a short on-demand primer; everyone else proceeds to the full
sequence. Each notebook states its own prerequisites. The whole course can be
completed without a GPU using the precomputed results in `data/cache_subset/`
(about one to two hours); running the experiments yourself on a small local
model is optional.

## Quick start

**No-GPU (recommended first pass):** open `notebooks/00_self_check.ipynb`, then
follow the sequence. All plots and tables load from `data/cache_subset/`.

**GPU (run the experiments yourself):** Python 3.10+, any CUDA 12.4 machine with
an NVIDIA GPU (8 GB+ VRAM):

```bash
pip install -r requirements-local.txt
python scripts/download_models.py     # six small GGUF models, Q8_0, ~15 GB
python scripts/run_experiment.py --help
```

No-GPU / CPU-only fallback: `pip install -r requirements-cpu.txt`. The six models
span measured single-answer accuracies from 0.29 to 0.70 on MATH-500, which is
what makes the search-vs-verify trade-off visible from a single machine.

## Reproducibility

- Reported numbers are means over 3 seeds with 95% confidence intervals.
- Per-call seeds are derived from a deterministic key via sha256 (no Python
  `hash()`); same seed, same code version, same machine, same output directory
  ⇒ identical output. Cross-machine reproducibility is not guaranteed for
  llama.cpp.
- Every LLM call is persisted immediately as a JSONL record (idempotent keys);
  interrupts resume with zero data loss.
- Statistics are rebuilt from the records; the included cache is verified against
  the source records.

## License

The course materials are licensed under [CC BY 4.0](LICENSE). Model licenses:
Gemma Terms of Use, Apache-2.0 (Qwen3), MIT (Phi-4-Mini). The papers in
`references/` are redistributed under their CC BY 4.0 licenses.
