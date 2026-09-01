# export_data.py - export the headline table data + field-name validation
# Input: experiment1_<model>.json (analyze_experiment output, containing full configuration statistics)
# Output: headline table data as JSON + CSV (headline rows always have output slots; null when missing)
# Field validation: check each field against the JSON format; when canonical: true, gguf_sha256 is required
# Run: python scripts/export_data.py --inputs data/results/experiment1_qwen3-4b.json --out-dir data/results

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# headline rows
HEADLINE = [(50, 0), (25, 1), (5, 9)]

# Required fields (validated one by one)
REQUIRED_FIELDS = [
    "config", "accuracy", "std", "runs", "seeds", "budget", "dataset", "model",
    "engine", "quantization", "gguf_file", "gguf_sha256", "sampling", "canonical",
]
REQUIRED_CONFIG = ["strategy", "N", "M"]
REQUIRED_SAMPLING = ["temperature", "top_p", "n_ctx", "enable_thinking"]


def validate_config_record(rec: dict) -> List[str]:
    # Field-name validation: per field; when canonical: true, gguf_sha256 is required
    errors = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errors.append(f"missing field: {f}")
    for f in REQUIRED_CONFIG:
        if f not in rec.get("config", {}):
            errors.append(f"config missing field: {f}")
    for f in REQUIRED_SAMPLING:
        if f not in rec.get("sampling", {}):
            errors.append(f"sampling missing field: {f}")
    if rec.get("canonical") is True and not rec.get("gguf_sha256"):
        errors.append("canonical: true but gguf_sha256 is empty (required)")
    if rec.get("runs") != len(rec.get("seeds", [])):
        errors.append(f"runs({rec.get('runs')}) and seeds({rec.get('seeds')}) mismatch")
    return errors


def export_headline(data: dict) -> dict:
    # Organize the headline table data: every model has slots for the headline rows (null when missing) + all configs
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
    ap = argparse.ArgumentParser(description="export headline table data + field validation")
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to experiment1_*.json (multiple allowed)")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data", "results"))
    args = ap.parse_args()

    sections = []
    all_errors = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Field validation (per configuration)
        errors = []
        for c in data.get("configs", []):
            errors += validate_config_record(c)
        if errors:
            all_errors += [f"{os.path.basename(path)}: {e}" for e in errors]
        sections.append(export_headline(data))
        print(f"{os.path.basename(path)}: {len(data.get('configs', []))} configs processed")

    # Write JSON
    out_json = os.path.join(args.out_dir, "headline_data.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"models": sections}, f, ensure_ascii=False, indent=2)
    print(f"headline data → {out_json}")

    # Write CSV (headline rows per model; the dataset column distinguishes same-name models across datasets (MATH and GSM8K) share the same model name and p)
    out_csv = os.path.join(args.out_dir, "headline_table.csv")
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
                    w.writerow([s["model"], s["dataset"], "", n, m, "", "", ""])  # keep the slot
    print(f"headline CSV → {out_csv}")

    # Validation report
    if all_errors:
        print("⚠️ field validation found issues:")
        for e in all_errors[:20]:
            print(f"  - {e}")
        sys.exit(1)
    print("✅ all field validation passed (including required gguf_sha256 when canonical:true)")


if __name__ == "__main__":
    main()
