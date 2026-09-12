"""Recompute scaffold-match metrics for generation result files."""

\
\
\
\
\
\
   

import json
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
except ImportError:
    print("ERROR: rdkit not installed. Run: pip install rdkit")
    exit(1)

                                                                               
GEN_DIR = r"results/generation"
MCS_THRESHOLD = 0.80
                                                                               

MODEL_DISPLAY = {
    "claude-sonnet-4-6":                 "Claude Sonnet 4",
    "deepseek-r1-0528":                  "DeepSeek-R1",
    "deepseek-v3-0324":                  "DeepSeek-V3",
    "gemini-2.5-flash-nothinking":       "Gemini 2.5 Flash",
    "gpt-4.1-2025-04-14":               "GPT-4.1",
    "gpt-4o":                            "GPT-4o",
    "gpt-5.5":                           "GPT-5.5",
    "llama-3.3-70b-instruct":            "LLaMA-3.3-70B",
    "llama-4-maverick":                  "Llama 4 Maverick",
    "meta-llama_llama-3.3-70b-instruct": "LLaMA-3.3-70B",
    "o3":                                "o3",
    "qwen3-max-2025-09-23":              "Qwen3-Max",
    "qwen3-vl-8b-instruct":             "Qwen3-VL-8B",
    "qwen3-vl-32b-instruct":            "Qwen3-VL-32B",
}

SKIP = {"gemini-2.5-pro-nothinking"}

VLM_ORDER = ["Qwen3-VL-8B", "Qwen3-VL-32B", "Claude Sonnet 4",
             "GPT-4o", "GPT-4.1", "GPT-5.5", "Gemini 2.5 Flash", "Llama 4 Maverick"]
LLM_ORDER = ["DeepSeek-V3", "DeepSeek-R1", "LLaMA-3.3-70B", "o3", "Qwen3-Max"]

VLM_MODELS = {
    "claude-sonnet-4-6", "gemini-2.5-flash-nothinking",
    "gpt-4.1-2025-04-14", "gpt-4o", "gpt-5.5",
    "llama-4-maverick", "qwen3-vl-8b-instruct", "qwen3-vl-32b-instruct",
}
LLM_MODELS = {
    "deepseek-r1-0528", "deepseek-v3-0324",
    "llama-3.3-70b-instruct", "meta-llama_llama-3.3-70b-instruct",
    "o3", "qwen3-max-2025-09-23",
}


def load_jsonl(path):
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
    except FileNotFoundError:
        pass
    return records


def scaffold_match_score(pred_smi: str, backbone_smi: str) -> bool:
    """Return whether the predicted molecule preserves the scaffold backbone by MCS coverage."""
                                                                              
    try:
        mol_pred = Chem.MolFromSmiles(pred_smi)
        mol_back = Chem.MolFromSmiles(backbone_smi)
        if mol_pred is None or mol_back is None:
            return False
        n_backbone = mol_back.GetNumAtoms()
        if n_backbone == 0:
            return False
        res = rdFMCS.FindMCS(
            [mol_pred, mol_back],
            timeout=2,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        return (res.numAtoms / n_backbone) >= MCS_THRESHOLD
    except:
        return False


def exact_match(pred_smi: str, gt_smi: str) -> bool:
    try:
        m1 = Chem.MolFromSmiles(pred_smi)
        m2 = Chem.MolFromSmiles(gt_smi)
        if m1 is None or m2 is None:
            return False
        return Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)
    except:
        return False


def fmt(v):
    return f"{v:.1f}" if v is not None else "N/A"


def dedup(rows):
    best = {}
    for r in rows:
        n = r["display"]
        if n not in best or r["total"] > best[n]["total"]:
            best[n] = r
    return list(best.values())


def sort_rows(rows, order):
    idx = {n: i for i, n in enumerate(order)}
    return sorted(rows, key=lambda r: idx.get(r["display"], 999))


def compute(gen_dir: Path):
    print("="*75)
    print("  Scaffold Match verification  (MCS >= 80% backbone atoms)")
    print("="*75)

                                                               
    print("\n[Inspecting field names in generation files...]")
    sample_shown = False
    for folder in sorted(gen_dir.iterdir()):
        if not folder.is_dir() or folder.name in SKIP:
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix in ('.jsonl', '.json') and f.stat().st_size > 1000:
                recs = load_jsonl(f)
                if recs:
                    print(f"  File : {folder.name}/{f.name}")
                    print(f"  Keys : {list(recs[0].keys())}")
                    print(f"  Row0 : { {k: str(v)[:60] for k,v in recs[0].items()} }")
                    sample_shown = True
                    break
        if sample_shown:
            break

    if not sample_shown:
        print("  ERROR: No generation files found. Check GEN_DIR path.")
        return

                                
    sample = load_jsonl(list(list(gen_dir.iterdir())[0].iterdir())[0])
    if not sample:
        print("  ERROR: Could not load sample file.")
        return

    keys = set(sample[0].keys())
                                
    pred_key  = next((k for k in ["predicted_smiles", "predicted", "output",
                                   "pred_smiles", "pred"] if k in keys), None)
    gt_key    = next((k for k in ["ground_truth", "gt_smiles", "target",
                                   "correct_smiles", "answer"] if k in keys), None)
    back_key  = next((k for k in ["backbone_smiles", "backbone", "scaffold_smiles",
                                   "scaffold", "markush_smiles"] if k in keys), None)
    split_key = next((k for k in ["split", "difficulty", "level"] if k in keys), None)
    mode_key  = next((k for k in ["mode", "modality", "input_mode"] if k in keys), None)

    print(f"\n  Detected fields:")
    print(f"    pred_key  = {pred_key}")
    print(f"    gt_key    = {gt_key}")
    print(f"    back_key  = {back_key}")
    print(f"    split_key = {split_key}")
    print(f"    mode_key  = {mode_key}")

    if pred_key is None or gt_key is None:
        print("\n  ERROR: Cannot find predicted/ground-truth SMILES fields.")
        print("  Please check the key names above and update the script.")
        return

                                       
    rows = []

    for folder in sorted(gen_dir.iterdir()):
        if not folder.is_dir() or folder.name in SKIP:
            continue
        fn = folder.name

        if fn in VLM_MODELS:
            model_type = "VLM"
        elif fn in LLM_MODELS:
            model_type = "LLM"
        else:
            model_type = "UNK"

                                                                              
        buckets = defaultdict(list)                      

        for f in sorted(folder.iterdir()):
            if f.suffix not in ('.jsonl', '.json'):
                continue
            recs = load_jsonl(f)
            if not recs:
                continue
            for r in recs:
                mode  = r.get(mode_key, "unknown") if mode_key else "unknown"
                split = r.get(split_key, "unknown") if split_key else "unknown"
                                       
                split_norm = "easy" if "easy" in str(split).lower() else\
                             "hard" if "hard" in str(split).lower() else split
                mode_norm  = "smi_smi" if "smi_smi" in str(mode).lower() or\
                                          str(mode).lower() in ("smi-smi", "smismi")\
                             else ("img_smi" if "img_smi" in str(mode).lower() or
                                   str(mode).lower() in ("img-smi", "imgsmi") else mode)
                buckets[(mode_norm, split_norm)].append(r)

        if not buckets:
            print(f"  [NO DATA] {fn}")
            continue

        print(f"\n  {fn}: buckets = {list(buckets.keys())}")

        display = MODEL_DISPLAY.get(fn, fn)
        result = {"display": display, "type": model_type, "total": 0}

        for (mode, split), recs in sorted(buckets.items()):
            em_list, sm_list = [], []
            for r in recs:
                pred = r.get(pred_key, "") or ""
                gt   = r.get(gt_key, "")   or ""
                back = r.get(back_key, gt) or gt                                  

                em_list.append(exact_match(pred, gt))

                if back_key:
                    sm_list.append(scaffold_match_score(pred, back))
                else:
                                                        
                    sm_list.append(scaffold_match_score(pred, gt))

            n = len(recs)
            em  = round(100 * sum(em_list) / n, 1) if n else None
            sm  = round(100 * sum(sm_list) / n, 1) if n else None
            key = f"EM_{mode}_{split}"
            result[f"EM_{mode}_{split}"]   = em
            result[f"Scaff_{mode}_{split}"] = sm
            result[f"N_{mode}_{split}"]     = n
            result["total"] += n

        rows.append(result)

                                 
    print("\n\n" + "="*75)
    print("  RESULTS SUMMARY")
    print("="*75)

    for mode in ["smi_smi", "img_smi"]:
        for split in ["easy", "hard"]:
            em_key = f"EM_{mode}_{split}"
            sm_key = f"Scaff_{mode}_{split}"
            n_key  = f"N_{mode}_{split}"

            group_rows = [r for r in rows if em_key in r]
            if not group_rows:
                continue

            print(f"\n  --- {mode} | {split} ---")
            print(f"  {'Model':<25} {'EM':>6} {'Scaff.Match':>12} {'N':>5}")
            print(f"  {'-'*52}")

            vlm = sort_rows(dedup([r for r in group_rows if r["type"]=="VLM"]), VLM_ORDER)
            llm = sort_rows(dedup([r for r in group_rows if r["type"]=="LLM"]), LLM_ORDER)

            for group, label in [(vlm, "VLMs"), (llm, "LLMs")]:
                if not group: continue
                print(f"  [{label}]")
                em_vals, sm_vals = [], []
                for r in group:
                    em = r.get(em_key)
                    sm = r.get(sm_key)
                    n  = r.get(n_key, 0)
                    print(f"  {r['display']:<25} {fmt(em):>6} {fmt(sm):>12} {n:>5}")
                    if em is not None: em_vals.append(em)
                    if sm is not None: sm_vals.append(sm)
                if em_vals:
                    print(f"  {'Average':<25} "
                          f"{sum(em_vals)/len(em_vals):>6.1f} "
                          f"{sum(sm_vals)/len(sm_vals) if sm_vals else 0:>12.1f}")

    print("\n\nDone. Compare Scaff. column with Table 6 R-Corr numbers.")
