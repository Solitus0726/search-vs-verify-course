# export_data.py —— syllabus §5.4 表格数据导出 + 字段名校验
# 输入：experiment1_<model>.json（analyze_experiment 产物，含全部配置统计）
# 输出：syllabus §5.4 表格数据 JSON + CSV（headline 三行必有输出槽位，缺失为 null）
# 字段校验：对照 JSON 格式逐字段；canonical: true 时 gguf_sha256 必填
# 运行：python scripts/export_data.py --inputs data/results/experiment1_qwen3-4b.json --out-dir data/results

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# syllabus §5.4 headline 三行
HEADLINE = [(50, 0), (25, 1), (5, 9)]

# 必需字段（逐字段校验）
REQUIRED_FIELDS = [
    "config", "accuracy", "std", "runs", "seeds", "budget", "dataset", "model",
    "engine", "quantization", "gguf_file", "gguf_sha256", "sampling", "canonical",
]
REQUIRED_CONFIG = ["strategy", "N", "M"]
REQUIRED_SAMPLING = ["temperature", "top_p", "n_ctx", "enable_thinking"]


def validate_config_record(rec: dict) -> List[str]:
    # 字段名校验：逐字段；canonical: true 时 gguf_sha256 必填
    errors = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errors.append(f"缺字段: {f}")
    for f in REQUIRED_CONFIG:
        if f not in rec.get("config", {}):
            errors.append(f"config 缺字段: {f}")
    for f in REQUIRED_SAMPLING:
        if f not in rec.get("sampling", {}):
            errors.append(f"sampling 缺字段: {f}")
    if rec.get("canonical") is True and not rec.get("gguf_sha256"):
        errors.append("canonical: true 但 gguf_sha256 为空（必填）")
    if rec.get("runs") != len(rec.get("seeds", [])):
        errors.append(f"runs({rec.get('runs')}) 与 seeds({rec.get('seeds')}) 不一致")
    return errors


def export_section54(data: dict) -> dict:
    # 组织 syllabus §5.4 表格数据：每模型 headline 三行必有槽位（缺失 null）+ 全配置
    model = data["model"]
    configs = data["configs"]
    by_key = {(c["config"]["N"], c["config"]["M"]): c for c in configs}
    headline = {}
    for (n, m) in HEADLINE:
        rec = by_key.get((n, m))
        headline[f"N={n},M={m}"] = rec if rec else None
    return {
        "model": model,
        "budget": data["budget"],
        "dataset": data["dataset"],
        "headline": headline,
        "configs": configs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="syllabus §5.4 表格数据导出 + 字段校验")
    ap.add_argument("--inputs", nargs="+", required=True, help="experiment1_*.json 路径（可多个）")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data", "results"))
    args = ap.parse_args()

    sections = []
    all_errors = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 字段校验（逐配置）
        errors = []
        for c in data.get("configs", []):
            errors += validate_config_record(c)
        if errors:
            all_errors += [f"{os.path.basename(path)}: {e}" for e in errors]
        sections.append(export_section54(data))
        print(f"{os.path.basename(path)}: {len(data.get('configs', []))} 配置已处理")

    # 输出 JSON
    out_json = os.path.join(args.out_dir, "section54_data.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"models": sections}, f, ensure_ascii=False, indent=2)
    print(f"syllabus §5.4 数据 → {out_json}")

    # 输出 CSV（每模型 headline 三行；dataset 列区分同名模型跨数据集——MATH 与 GSM8K 同模型名同 p）
    out_csv = os.path.join(args.out_dir, "section54_table.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "strategy", "N", "M", "accuracy", "cost", "canonical"])
        for s in sections:
            for (n, m) in HEADLINE:
                rec = s["headline"][f"N={n},M={m}"]
                if rec:
                    w.writerow([s["model"], s["dataset"], rec["config"]["strategy"], n, m,
                                rec["accuracy"], rec.get("cost", ""), rec["canonical"]])
                else:
                    w.writerow([s["model"], s["dataset"], "", n, m, "", "", ""])  # 槽位保留
    print(f"syllabus §5.4 CSV → {out_csv}")

    # 校验报告
    if all_errors:
        print("⚠️ 字段校验发现问题：")
        for e in all_errors[:20]:
            print(f"  - {e}")
        sys.exit(1)
    print("✅ 字段校验全部通过（含 canonical:true 的 gguf_sha256 必填）")


if __name__ == "__main__":
    main()
