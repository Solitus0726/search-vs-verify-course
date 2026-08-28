# plot_heatmap.py —— 实验 2 热力图
# 输入：experiment1_<model>.json（analyze_experiment 产物，含全部配置的 accuracy）
# 输出：data/figures/heatmap_<model>.png
# 要素：
#   - x 轴 = N（候选数），y 轴 = M（验证次数），格子颜色 = 准确率
#   - 每格标注 accuracy + 实际消耗 N×(1+M)
#   - 4 个合法组合 (5,9)(10,4)(25,1)(50,0) 边框高亮（B=50 等消耗）
#   - 超预算参考点（消耗 >50）单独标注
# 运行：python scripts/plot_heatmap.py --inputs data/results/experiment1_qwen3-4b.json

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 个合法组合（B=50 恒等式 N×(1+M)=50）
LEGAL_COMBOS = [(5, 9), (10, 4), (25, 1), (50, 0)]
BUDGET = 50


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 2 热力图（(N,M) 准确率 + 消耗标注）")
    ap.add_argument("--inputs", nargs="+", required=True, help="experiment1_*.json 路径（可多个）")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data", "figures"))
    args = ap.parse_args()

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data["model"]
        configs = data["configs"]

        # 格子数据：N × M 稀疏网格
        ns = sorted({c["config"]["N"] for c in configs})
        ms = sorted({c["config"]["M"] for c in configs})
        acc = {}
        cost = {}
        for c in configs:
            n, m = c["config"]["N"], c["config"]["M"]
            acc[(n, m)] = c["accuracy"]
            cost[(n, m)] = c["cost"]

        # 热力图（pcolormesh 稀疏格 + 标注）
        fig, ax = plt.subplots(figsize=(10, 7))
        for (n, m), a in acc.items():
            color = plt.cm.viridis(a)
            rect = plt.Rectangle((n - 1, m - 1), 1.8, 1.8, color=color, alpha=0.9)
            ax.add_patch(rect)
            legal = (n, m) in LEGAL_COMBOS
            # 标注：accuracy + 消耗（合法组合高亮边框）
            ax.text(n, m + 0.35, f"{a:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if legal else "normal",
                    color="white" if a < 0.5 else "black")
            ax.text(n, m - 0.35, f"耗{int(cost[(n, m)])}", ha="center", va="center", fontsize=8, color="gray")
            if legal:
                ax.add_patch(plt.Rectangle((n - 1, m - 1), 1.8, 1.8, fill=False,
                                           edgecolor="red", linewidth=2.5))

        ax.set_xlim(0, max(ns) + 1)
        ax.set_ylim(0, max(ms) + 1)
        ax.set_xticks(ns)
        ax.set_yticks(ms)
        ax.set_xlabel("候选数 N")
        ax.set_ylabel("每条验证次数 M")
        ax.set_title(f"实验 2：最优混合配比热力图（{model}，B={BUDGET}）\n红框 = 合法组合 N×(1+M)={BUDGET}；每格标注准确率与实际消耗")
        ax.grid(True, alpha=0.3)

        # 颜色条
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, label="准确率")

        out = os.path.join(args.out_dir, f"heatmap_{os.path.basename(path).replace('experiment1_', '').replace('.json', '.png')}")
        os.makedirs(args.out_dir, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"热力图已生成：{out}（{len(configs)} 个配置点）")


if __name__ == "__main__":
    main()
