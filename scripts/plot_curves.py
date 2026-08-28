# plot_curves.py —— 实验 1 预算-准确率曲线图
# 输入：experiment1_<model>.json（一个或多个，画曲线族）
# 输出：data/figures/curve_<model>.png
# 要素：x 轴双刻度（调用次数 + token 实际口径）、B=50 参考竖线、
#   等候选数对照点（N=10 搜索 vs N=10/M=4 验证，虚线连接标注）
# 运行：python scripts/plot_curves.py --inputs data/results/experiment1_qwen3-4b.json \
#        --tokens-per-call 608 --out data/figures/curve_qwen3-4b.png

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
# 中文字体（作者侧 Windows 渲染；学生看 PNG 成品无字体依赖）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 1 曲线图（双刻度 x 轴 + B=50 + 等候选数对照）")
    ap.add_argument("--inputs", nargs="+", required=True, help="experiment1_*.json 路径（可多个）")
    ap.add_argument("--tokens-per-call", type=float, default=608.0,
                    help="token 副刻度换算：平均 token/调用（校准值 608）")
    ap.add_argument("--out", default=None, help="输出 PNG 路径（默认 data/figures/curve_<首模型>.png）")
    args = ap.parse_args()

    datas = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            datas.append(json.load(f))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, data in enumerate(datas):
        color = colors[i % len(colors)]
        model = data["model"]
        search = [c for c in data["configs"] if c["config"]["strategy"] == "search"]
        verify = [c for c in data["configs"] if c["config"]["strategy"] == "verify"]
        # 搜索线（按消耗排序）
        sx = [c["cost"] for c in search]
        sy = [c["accuracy"] for c in search]
        if sx:
            ax.plot(sx, sy, "o-", color=color, label=f"{model} 搜索（投票）")
        # 验证线
        vx = [c["cost"] for c in verify]
        vy = [c["accuracy"] for c in verify]
        if vx:
            ax.plot(vx, vy, "s--", color=color, label=f"{model} 验证（均分选优）")

        # 等候选数对照：N=10 搜索 vs N=10/M=4 验证（虚线连接，剥离候选数效应）
        s10 = next((c for c in search if c["config"]["N"] == 10), None)
        v10m4 = next((c for c in verify if c["config"]["N"] == 10 and c["config"]["M"] == 4), None)
        if s10 is not None and v10m4 is not None:
            ax.plot([s10["cost"], v10m4["cost"]], [s10["accuracy"], v10m4["accuracy"]],
                    "k:", linewidth=1.2)
            ax.annotate("等候选数对照\n(N=10)", xy=((s10["cost"] + v10m4["cost"]) / 2,
                        (s10["accuracy"] + v10m4["accuracy"]) / 2),
                        fontsize=8, ha="center", va="bottom")

    # B=50 参考竖线
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=1.0)
    ax.text(50, ax.get_ylim()[0] if ax.get_ylim()[0] >= 0 else 0, "  B=50",
            fontsize=9, color="gray", va="bottom")

    ax.set_xlabel("预算消耗（调用次数）")
    ax.set_ylabel("准确率（MATH-500 子集）")
    ax.set_title(f"预算-准确率曲线（x 轴双刻度：调用次数 / token×{args.tokens_per_call:.0f}）")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # 副 x 轴：token 口径（实际均值 = 调用次数 × 平均 token/调用）
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * args.tokens_per_call, ax.get_xlim()[1] * args.tokens_per_call)
    ax2.set_xlabel(f"token（实际口径 ≈ 调用次数 × {args.tokens_per_call:.0f}）")

    out = args.out or os.path.join(PROJECT_ROOT, "data", "figures",
                                   f"curve_{os.path.basename(args.inputs[0]).replace('experiment1_', '').replace('.json', '.png')}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"曲线图已生成：{out}")


if __name__ == "__main__":
    main()
