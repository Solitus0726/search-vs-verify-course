# Test-Time Compute Allocation: Search vs. Verify

A self-contained teaching unit for the **NeurIPS 2026 Education Track**.

**Test-time compute allocation** is the problem of dividing a fixed number of
model calls between search and verification: *search* aggregates many candidate
answers by majority voting, while *verification* generates fewer and checks each
one with a judge. The motivating question: given a fixed inference budget, how
should a model spend it?

**Central claim (conditional, measured):** under a *weak judge* (the model
grading its own answers) and a *moderate budget* (50 inference calls per problem),
search beats verification by +7.3 to +10.0 pp, measured on three models
(100 problems, 3 seeds; 95% CIs exclude 0). The materials teach *when* this
holds and how to find the best split on your own model.

All materials are original and were created for the NeurIPS 2026 Education
Track. The public repository is https://github.com/Solitus0726/search-vs-verify-course.

## Authors

Tianxiang Xie, Jingxuan Wu, Jiaying Liu\*, and Shuo Yu

Dalian University of Technology

\*Corresponding author: jiayingliu@dlut.edu.cn

## Repository layout

| Path | What it is |
|------|------------|
| `notebooks/` | Five Jupyter notebooks (00 self-check through 04 baseline review), the hands-on core |
| `video/concept_video.mp4` | Five-minute concept video (English narration, hardcoded subtitles, SRT included) |
| `slides/lecture_slides.pptx`, `slides/lecture_slides.pdf` | Lecture slides: the math behind the trade-off |
| `paper/` | The two-page submission paper (PDF + LaTeX source) |
| `figures/` | Learning-path diagram and strategy comparison diagram |
| `data/cache_subset/` | Precomputed experiment results, so the full course runs without a GPU |
| `data/subsets/` | Problem ID subsets for MATH 500 and GSM8K |
| `scripts/` | Experiment runner, answer evaluator, split predictor, plotting utilities |
| `references/` | The five papers the course builds on |
| `requirements-*.txt` | Dependencies for the GPU tier and the CPU tier |

## References

The course builds on five recent papers (PDFs in `references/`):

1. Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning (ICLR 2025)
2. When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning (COLM 2025)
3. Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification (ICML 2025)
4. Let Me Think! A Long Chain of Thought Can Be Worth Exponentially Many Short Ones (NeurIPS 2025)
5. Provable Scaling Laws for the Test-Time Compute of Large Language Models (NeurIPS 2025)

## Learning path

The course targets graduate students and early-career researchers in LLM
reasoning (adaptable for advanced undergraduates); the video-only route needs
no prerequisites. The course is organized as a learning path (see
`figures/learning_path.png`):

1. **Notebook 00 · 5-minute self-check** is the entry point: three prerequisite
   questions that route you to the right materials.
2. **If the self-check fails**, work through the on-demand resources (the video,
   slides, strategy diagram, and the five key papers) and retry the self-check.
3. **If it passes**, follow the main sequence in the diagram, in one of three
   ways, each a concrete material sequence:
   - *Systematic Study (≈3-4 h):* video → strategy diagram → slides →
     Notebooks 01-03 (full run) → Notebook 04 (write the report)
   - *Intuition First (≈30 min):* video → strategy diagram → Notebook 01 (simplified run)
   - *Quick Reference (≈10 min):* strategy diagram → jump to the relevant
     notebook section
4. Every path ends at the **acceptance check**: calculation problems, curve
   interpretation, and an analysis report.

Each notebook states its own prerequisites. Every path runs without a GPU using
the precomputed results in `data/cache_subset/`; running the experiments
yourself on a small local model is optional (one to two hours from scratch, or
instantly with the pre-computed cache).

## Quick start

**CPU tier (recommended first pass):** open `notebooks/00_self_check.ipynb`, then
follow the learning path. All plots and tables load from `data/cache_subset/`.
Install with `pip install -r requirements-cpu.txt`. The six models span measured
single-answer accuracies from 0.37 to 0.73 on MATH 500, which is what makes the
search-vs-verify trade-off visible from a single machine. Notebooks preview
directly on GitHub; to run them interactively, also install Jupyter
(`pip install notebook`), which the requirements files do not cover.

**GPU tier (run the experiments yourself):** Python 3.10+, any CUDA 12.4 machine
with an NVIDIA GPU (8 GB+ VRAM):

```bash
pip install -r requirements-local.txt
python scripts/download_models.py     # six small GGUF models, Q8_0, ~15 GB
python scripts/run_experiment.py --help
```

### Installing the GPU tier (Windows)

PyPI does not publish the CUDA 12.4 (cu124) wheel of `llama-cpp-python==0.3.34`
for Windows. Download it from the project's GitHub Releases (file
`llama_cpp_python-0.3.34-cpXXX-cpXXX-win_amd64.whl`, replacing `cpXXX` with your
Python version, e.g. `cp311` for Python 3.11). If GitHub is slow or unreachable,
prefix the URL with the `ghfast.top` mirror: `https://ghfast.top/<original-url>`.
Then install everything:

```bash
pip install <path-or-url-to-the-cu124-wheel>
pip install -r requirements-local.txt
```

`torch` is a CUDA-DLL dependency only: add `torch/lib` to your `PATH` before
running (it provides the `cudart64_12` / `cublas64_12` runtime DLLs).

## Reproducibility

- Reported numbers are means over 3 seeds with 95% confidence intervals.
- Per-call seeds are derived from a deterministic key via sha256 (no Python
  `hash()`); with the same seed, code version, machine, and output directory,
  rerunning gives identical output. Cross-machine reproducibility is not
  guaranteed for llama.cpp.
- Every LLM call is persisted immediately as a JSONL record (idempotent keys);
  interrupts resume with zero data loss.
- Statistics are rebuilt from the records; the included cache is verified against
  the source records.

## License

The course materials are licensed under [CC BY 4.0](LICENSE). Model licenses:
Gemma Terms of Use, Apache-2.0 (Qwen3), MIT (Phi-4-Mini). The papers in
`references/` are included for educational use; see each paper's source venue
(arXiv/ICML/ICLR/NeurIPS/COLM) for its license terms.
