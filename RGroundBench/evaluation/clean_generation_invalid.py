"""Remove API-failure rows while preserving chemically invalid model answers."""

\
\
\
\
\
   

import json
import os
from collections import defaultdict

RESULTS_DIR = "results/generation"


def is_api_invalid_record(record):
    raw = (record.get("raw_response") or "").strip()
    if record.get("api_invalid") is True:
        return True
    if not raw:
        return True
    if raw.startswith("TIMEOUT") or raw.startswith("ERROR:"):
        return True
    return False


def read_jsonl_records(path):
    lines = []
    skipped_corrupt = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw_line = line.strip()
            if not raw_line:
                continue
            if "\ufffd" in raw_line:
                skipped_corrupt += 1
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            lines.append((record, raw_line + "\n"))

    return lines, skipped_corrupt


def clean_and_count(results_dir):
    summary = defaultdict(lambda: defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0}))

    for model_tag in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue

        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".jsonl"):
                continue

            fpath = os.path.join(model_dir, fname)
            key = fname[:-6]

            lines, skipped_corrupt = read_jsonl_records(fpath)
            valid_lines = [(rec, line) for rec, line in lines if not is_api_invalid_record(rec)]
            invalid_lines = [(rec, line) for rec, line in lines if is_api_invalid_record(rec)]

            n_invalid = len(invalid_lines)
            if n_invalid > 0 or skipped_corrupt > 0:
                with open(fpath, "w", encoding="utf-8") as f:
                    for _, line in valid_lines:
                        f.write(line)
                if skipped_corrupt > 0:
                    print(f"  [{model_tag}] {fname}: skipped {skipped_corrupt} corrupt lines")
                if n_invalid > 0:
                    print(f"  [{model_tag}] {fname}: removed {n_invalid} API-invalid, kept {len(valid_lines)}")

            summary[model_tag][key] = {
                "total": len(lines),
                "valid": len(valid_lines),
                "invalid": n_invalid,
            }

    return summary


def print_summary(summary):
    modes = ["smi", "img"]
    splits = ["easy", "hard"]

    print("\n" + "=" * 70)
    print("  Generation clean complete - valid answer counts")
    print("=" * 70)

    for model_tag in sorted(summary.keys()):
        print(f"\n  Model: {model_tag}")
        print(f"  {'mode_split':<20} {'total':>7} {'valid':>7} {'api_invalid':>12}")
        print(f"  {'-' * 52}")
        model_valid = 0
        model_invalid = 0

        for mode in modes:
            for split in splits:
                key = f"{mode}_{split}"
                if key not in summary[model_tag]:
                    continue
                stats = summary[model_tag][key]
                model_valid += stats["valid"]
                model_invalid += stats["invalid"]
                flag = " *" if stats["invalid"] > 0 else ""
                print(
                    f"  {key:<20} {stats['total']:>7} "
                    f"{stats['valid']:>7} {stats['invalid']:>12}{flag}"
                )

        print(f"  {'Total':<20} {model_valid + model_invalid:>7} {model_valid:>7} {model_invalid:>12}")

    print("\n" + "=" * 70)
    print("  Cross-model valid summary")
    print("=" * 70)
    print(f"  {'Model':<35} {'valid':>8} {'api_invalid':>12}")
    print(f"  {'-' * 58}")
    for model_tag in sorted(summary.keys()):
        valid = sum(stats["valid"] for stats in summary[model_tag].values())
        invalid = sum(stats["invalid"] for stats in summary[model_tag].values())
        print(f"  {model_tag:<35} {valid:>8} {invalid:>12}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory does not exist: {RESULTS_DIR}")
        return

    print(f"Scanning directory: {RESULTS_DIR}")
    summary = clean_and_count(RESULTS_DIR)
    print_summary(summary)
    print("\nDone. Rerun 30_eval_generation.py or 30b_eval_generation_r1.py to refill API-invalid questions.")


if __name__ == "__main__":
    main()
