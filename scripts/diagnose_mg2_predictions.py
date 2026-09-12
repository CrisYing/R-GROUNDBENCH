"""Diagnose MarkushGrapher-2 scaffold prediction quality."""

\
\
\
\
\
\
\
\
\
\
\
\
   

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from datasets import load_from_disk
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")


ANNOT_PAT = re.compile(r"<a>\d+:(?:R\[\d+\]|R\d+|R|X|Y|Z)</a>")
WILDCARD_PAT = re.compile(r"\[\d+\*\]|\[\*\]|\*|\[#0\]")


def bare_esmiles(esmiles: str) -> str:
    return esmiles.split("<sep>")[0].strip()


def bare_cxsmiles(cxsmiles: str) -> str:
    return cxsmiles.split("|")[0].strip()


def prediction_aliases(img_id: str) -> set[str]:
    base = Path(img_id).name
    stem = Path(base).stem
    return {img_id, base, stem, f"{stem}.png"}


def load_predictions(pred_file: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with pred_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            img_id = str(obj.get("id", ""))
            cxsmiles = obj.get("cxsmiles", "") or ""
            if not img_id or not cxsmiles:
                continue
            for key in prediction_aliases(img_id):
                preds[key] = cxsmiles
    return preds


def lookup_prediction(preds: dict[str, str], image_path: str) -> str:
    name = Path(str(image_path)).name
    stem = Path(name).stem
    for key in (name, stem, f"{stem}.png"):
        if key in preds:
            return preds[key]
    return ""


def wildcard_count(smiles: str) -> int:
    return len(WILDCARD_PAT.findall(smiles))


def normalize_for_scaffold(smiles: str) -> str:
    return WILDCARD_PAT.sub("[H]", smiles)


def canonical_or_none(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def scaffold_mcs_coverage(gt_bare: str, pred_bare: str) -> float | None:
    try:
        gt_mol = Chem.MolFromSmiles(normalize_for_scaffold(gt_bare))
        pred_mol = Chem.MolFromSmiles(normalize_for_scaffold(pred_bare))
        if gt_mol is None or pred_mol is None or gt_mol.GetNumAtoms() == 0:
            return None
        mcs = rdFMCS.FindMCS(
            [gt_mol, pred_mol],
            timeout=5,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        if mcs.canceled:
            return None
        return float(mcs.numAtoms / gt_mol.GetNumAtoms())
    except Exception:
        return None


def summarize_split(dataset_dir: Path, pred_file: Path, sample_limit: int) -> dict:
    dataset = load_from_disk(str(dataset_dir))
    preds = load_predictions(pred_file)
    rows = []
    examples = []

    for idx, record in enumerate(dataset):
        gt_bare = bare_esmiles(record.get("original_esmiles", ""))
        cx = lookup_prediction(preds, record.get("original_image_path", ""))
        pred_bare = bare_cxsmiles(cx) if cx else ""

        gt_norm = normalize_for_scaffold(gt_bare)
        pred_norm = normalize_for_scaffold(pred_bare) if pred_bare else ""
        gt_can = canonical_or_none(gt_norm)
        pred_can = canonical_or_none(pred_norm) if pred_norm else None
        mcs_cov = scaffold_mcs_coverage(gt_bare, pred_bare) if pred_bare else None

        row = {
            "index": idx,
            "missing": not bool(cx),
            "pred_valid": pred_can is not None,
            "exact_scaffold": gt_can is not None and pred_can is not None and gt_can == pred_can,
            "gt_wildcards": wildcard_count(gt_bare),
            "pred_wildcards": wildcard_count(pred_bare),
            "wildcard_count_match": wildcard_count(gt_bare) == wildcard_count(pred_bare),
            "mcs_coverage": mcs_cov,
        }
        rows.append(row)

        if len(examples) < sample_limit and (
            row["missing"]
            or not row["pred_valid"]
            or not row["wildcard_count_match"]
            or (mcs_cov is not None and mcs_cov < 0.8)
        ):
            examples.append({
                **row,
                "image": Path(str(record.get("original_image_path", ""))).name,
                "gt_bare": gt_bare[:300],
                "pred_bare": pred_bare[:300],
            })

    total = len(rows)
    present = [r for r in rows if not r["missing"]]
    valid = [r for r in present if r["pred_valid"]]
    mcs_values = [r["mcs_coverage"] for r in rows if r["mcs_coverage"] is not None]

    return {
        "total": total,
        "predictions_loaded_aliases": len(preds),
        "missing": sum(r["missing"] for r in rows),
        "prediction_coverage": round(len(present) / max(total, 1), 4),
        "pred_valid_rate": round(len(valid) / max(len(present), 1), 4),
        "exact_scaffold_rate": round(sum(r["exact_scaffold"] for r in rows) / max(total, 1), 4),
        "wildcard_count_match_rate": round(
            sum(r["wildcard_count_match"] for r in present) / max(len(present), 1), 4
        ),
        "mcs_coverage_mean": round(mean(mcs_values), 4) if mcs_values else 0.0,
        "mcs_coverage_ge_0.8": round(
            sum(v >= 0.8 for v in mcs_values) / max(len(mcs_values), 1), 4
        ),
        "failure_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["easy_basic", "medium_basic"])
    parser.add_argument("--sample_limit", type=int, default=10)
    args = parser.parse_args()

    vqa_dir = Path(args.vqa_dir)
    pred_dir = Path(args.pred_dir)
    summaries = {}

    for split in args.splits:
        pred_file = pred_dir / f"predictions_vqa_{split}.jsonl"
        if not pred_file.is_file():
            raise FileNotFoundError(pred_file)
        summaries[split] = summarize_split(
            vqa_dir / split,
            pred_file,
            sample_limit=args.sample_limit,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nSaved diagnostic summary to: {out}")


if __name__ == "__main__":
    main()
