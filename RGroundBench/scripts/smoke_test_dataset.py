"""Smoke-test released dataset directories and image references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


VQA_SPLITS = ["easy_basic", "medium_basic", "hard_basic"]
GEN_SPLITS = ["easy", "hard"]


def exists(path: str) -> bool:
    return bool(path) and Path(path).is_file()


def check_vqa(vqa_dir: Path, max_samples: int) -> list[dict]:
    reports = []
    for split in VQA_SPLITS:
        ds = load_from_disk(str(vqa_dir / split))
        rows = list(ds.select(range(min(max_samples, len(ds)))))
        missing = []
        for i, row in enumerate(rows):
            keys = ["original_image_path", "option_A_image", "option_B_image", "option_C_image"]
            if str(row.get("option_D_smiles", "")) != "None of the above":
                keys.append("option_D_image")
            for key in keys:
                if not exists(str(row.get(key, ""))):
                    missing.append({"index": i, "key": key, "path": row.get(key, "")})
        reports.append(
            {
                "dataset": "vqa",
                "split": split,
                "records": len(ds),
                "checked": len(rows),
                "missing_images": missing[:20],
                "missing_count": len(missing),
                "sample_keys": list(rows[0].keys()) if rows else [],
            }
        )
    return reports


def check_generation(gen_dir: Path, max_samples: int) -> list[dict]:
    reports = []
    for split in GEN_SPLITS:
        ds = load_from_disk(str(gen_dir / split))
        rows = list(ds.select(range(min(max_samples, len(ds)))))
        missing = []
        for i, row in enumerate(rows):
            for key in ["original_image_path", "output_image_path"]:
                if not exists(str(row.get(key, ""))):
                    missing.append({"index": i, "key": key, "path": row.get(key, "")})
        reports.append(
            {
                "dataset": "generation",
                "split": split,
                "records": len(ds),
                "checked": len(rows),
                "missing_images": missing[:20],
                "missing_count": len(missing),
                "sample_keys": list(rows[0].keys()) if rows else [],
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_dir", default="data/vqa_dataset")
    parser.add_argument("--gen_dir", default="data/generation_qa")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--output", default="results/dataset_smoke.json")
    args = parser.parse_args()

    reports = []
    reports.extend(check_vqa(Path(args.vqa_dir), args.max_samples))
    reports.extend(check_generation(Path(args.gen_dir), args.max_samples))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(reports, indent=2, ensure_ascii=False))
    failed = [r for r in reports if r["missing_count"]]
    if failed:
        raise SystemExit("Dataset smoke test failed: missing image paths remain.")
    print("Dataset smoke test passed.")


if __name__ == "__main__":
    main()
