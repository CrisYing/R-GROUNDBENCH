"""Evaluate the MarkushGrapher-2 plus symbolic executor baseline."""

import os, sys, json, re, argparse
from pathlib import Path
from collections import defaultdict

DEBUG_ENABLED = False
_diag_count = [0]


def diag(message: str, limit: int = 5):
    """Print bounded diagnostics only when debug mode is enabled."""
    if not DEBUG_ENABLED:
        return
    if _diag_count[0] >= limit:
        return
    _diag_count[0] += 1
    print(message, flush=True)

from datasets import load_from_disk
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdFMCS

RDLogger.DisableLog("rdApp.*")

                                                               
                                                                                
VQA_DIR     = os.environ.get("RGBENCH_VQA_DIR",
    "data/vqa_dataset")
GEN_QA_DIR  = os.environ.get("RGBENCH_GEN_DIR",
    "data/generation_qa")
RESULTS_DIR = os.environ.get("RGBENCH_RESULTS_DIR",
    "results/mg2_pipeline")

VQA_SPLITS = ["easy_basic", "medium_basic", "hard_basic"]
GEN_SPLITS = ["easy", "hard"]

                                                                                
                            
                                                                                
R_GROUP_SMILES: dict[str, str] = {
                              
    "phenyl":                   "[*]c1ccccc1",
    "4-fluorophenyl":           "[*]c1ccc(F)cc1",
    "4-chlorophenyl":           "[*]c1ccc(Cl)cc1",
    "4-methylphenyl":           "[*]c1ccc(C)cc1",
    "4-methoxyphenyl":          "[*]c1ccc(OC)cc1",
             
    "3-chlorophenyl":           "[*]c1cccc(Cl)c1",
    "3,4-dichlorophenyl":       "[*]c1ccc(Cl)c(Cl)c1",
    "4-trifluoromethylphenyl":  "[*]c1ccc(C(F)(F)F)cc1",
                      
    "2-pyridyl":                "[*]c1ccccn1",
    "3-pyridyl":                "[*]c1cccnc1",
    "4-pyridyl":                "[*]c1ccncc1",
                         
    "cyclohexyl":               "[*]C1CCCCC1",
    "cyclopentyl":              "[*]C1CCCC1",
    "cyclopropyl":              "[*]C1CC1",
                     
    "benzyl":                   "[*]Cc1ccccc1",
    "4-fluorobenzyl":           "[*]Cc1ccc(F)cc1",
    "4-chlorobenzyl":           "[*]Cc1ccc(Cl)cc1",
                                                          
    "benzoyl":                  "[*]C(=O)c1ccccc1",
    "phenylacetyl":             "[*]C(=O)Cc1ccccc1",
    "4-fluorobenzoyl":          "[*]C(=O)c1ccc(F)cc1",
                                                    
    "phenylsulfonyl":           "[*]S(=O)(=O)c1ccccc1",
    "4-methylsulfonylphenyl":   "[*]c1ccc(S(=O)(=O)C)cc1",
}

                              
_NORM_MAP = {k.lower().replace("-", "").replace(" ", ""): v
             for k, v in R_GROUP_SMILES.items()}

def lookup_rgroup(name: str) -> str | None:
                                    
    name = name.strip()
    if name in R_GROUP_SMILES:
        return R_GROUP_SMILES[name]
    key = name.lower().replace("-", "").replace(" ", "")
    return _NORM_MAP.get(key)


                                                                                
                                                      
                                                                                
 
                                                               
                                    
                                                                  
                                                              
 
_R_LABEL = r'R\[\d+\]|R\d+|R(?!\[|\d)|X|Y|Z'
_PAIR_PAT = re.compile(r'(' + _R_LABEL + r')\s+with\s+([\w,\-\(\) ]+?)(?=,\s+and\s+(?:' + _R_LABEL + r')|\.?\s*$)',
                       re.IGNORECASE)

def _norm_rlabel(label: str) -> str:
                                                                   
    return re.sub(r'R\[(\d+)\]', r'R\1', label)

def parse_instruction(instruction: str) -> dict[str, str] | None:
    """Parse direct R-group substitution instructions into substituent assignments."""
\
\
\
\
\
       
                                             
    if not re.match(r'^\s*(?:replace|substitute)\b', instruction, re.IGNORECASE):
        return None

    pairs = _PAIR_PAT.findall(instruction)
    if not pairs:
        return None

    result = {}
    for rlabel, subst_name in pairs:
        subst_name = subst_name.strip().rstrip(". ")
        if not lookup_rgroup(subst_name):
            return None                           
                                                                    
        norm_label = _norm_rlabel(rlabel).upper()
        result[norm_label] = subst_name
    return result if result else None


                                                                                
             
                                                                                

_ANNOT_PAT = re.compile(r'<a>(\d+):(R\[\d+\]|R\d+|R|X|Y|Z)</a>')

def parse_esmiles(esmiles: str) -> tuple[str, dict[str, int]]:
\
\
\
\
\
       
    parts = esmiles.split("<sep>")
    bare = parts[0].strip()
    label_to_idx: dict[str, int] = {}
    if len(parts) > 1:
        for m in _ANNOT_PAT.finditer(parts[1]):
            raw_label = m.group(2)                              
                             
            norm = re.sub(r'R\[(\d+)\]', r'R\1', raw_label)
            label_to_idx[norm] = int(m.group(1))
    return bare, label_to_idx


                                                                                
            
                                                                                

def apply_substitutions(
    bare_smiles: str,
    label_to_idx: dict[str, int],
    substitutions: dict[str, str],                               
    fill_remaining_with_h: bool = True,
) -> str | None:
    """Apply parsed R-group substitutions to a Markush scaffold using RDKit graph edits."""
\
\
\
\
\
\
       
    try:
        mol = Chem.MolFromSmiles(bare_smiles, sanitize=False)
        if mol is None:
            diag(f"[DIAG-A] MolFromSmiles(sanitize=False) failed: {bare_smiles!r}")
            return None

        edit = Chem.RWMol(mol)

                                                                                       
        sorted_rlabels = sorted(
            substitutions.keys(),
            key=lambda r: label_to_idx.get(r, -1),
            reverse=True,
        )

        for rlabel in sorted_rlabels:
            subst_name = substitutions[rlabel]
            widx = label_to_idx.get(rlabel)
            if widx is None:
                diag(f"[DIAG-B] idx None: rlabel={rlabel!r}, label_to_idx keys={sorted(label_to_idx.keys())}, bare={bare_smiles!r}")
                return None

            frag_smiles = lookup_rgroup(subst_name)
            if frag_smiles is None:
                diag(f"[DIAG-C] lookup failed: {subst_name!r}")
                return None
                                                 
                                                                         
                                                                           
                                                                                      
            if frag_smiles.startswith("[*]"):
                frag_smiles = frag_smiles[3:]
            sub_mol = Chem.MolFromSmiles(frag_smiles)
            if sub_mol is None:
                diag(f"[DIAG-C2] sub_mol None: frag={frag_smiles!r}")
                return None

            try:
                wc_atom = edit.GetAtomWithIdx(widx)
            except Exception:
                diag(f"[DIAG-D0] GetAtomWithIdx({widx}) failed, natoms={edit.GetNumAtoms()}, bare={bare_smiles!r}")
                return None

            if wc_atom.GetAtomicNum() != 0:
                diag(f"[DIAG-D] not wildcard: rlabel={rlabel!r}, idx={widx}, sym={wc_atom.GetSymbol()!r}(Z={wc_atom.GetAtomicNum()}), bare={bare_smiles!r}")
                return None

            neighbors = [n.GetIdx() for n in wc_atom.GetNeighbors()]
            if not neighbors:
                diag(f"[DIAG-D2] wildcard has no neighbors: rlabel={rlabel!r}, idx={widx}, bare={bare_smiles!r}")
                return None
            neighbor_idx = neighbors[0]

                                                                                             
            combined = Chem.CombineMols(edit.GetMol(), sub_mol)
            edit2    = Chem.RWMol(combined)
            offset   = edit.GetNumAtoms()
            edit2.AddBond(neighbor_idx, offset, Chem.BondType.SINGLE)
            edit2.RemoveAtom(widx)
            edit = edit2

                                                                  
        if fill_remaining_with_h:
            for atom in edit.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    atom.SetIsotope(0)

                                                                                
                                                                   
                                                                                  
                                                                                   
                                                                                    
                                                                                        
                                                                                     
        try:
            final_mol = Chem.RemoveHs(edit.GetMol())
        except Exception as e:
            diag(f"[DIAG-F] RemoveHs/sanitize failed: {e!r}\n"
                 f"         bare={bare_smiles!r}\n"
                 f"         subs={substitutions}", limit=10)
            return None

        try:
            smi = Chem.MolToSmiles(final_mol)
            return smi if smi else None
        except Exception as e:
            diag(f"[DIAG-G] MolToSmiles failed: {e}")
            return None

    except Exception as e:
        diag(f"[DIAG-X] Unexpected exception in apply_substitutions: {e}")
        return None


                                                                                
                                    
                                                                                

def canonical(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None

def is_valid(smi: str) -> bool:
    return canonical(smi) is not None

def tanimoto(smi1: str, smi2: str) -> float | None:
    try:
        m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return None
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None

def scaffold_match(pred_smi: str, scaf_clean: str) -> bool | None:
                                                           
    try:
        scaf_no_star = re.sub(r"\*|\[\*\]|\[#0\]", "[H]", scaf_clean.split("<sep>")[0])
        m_scaf = Chem.MolFromSmiles(scaf_no_star)
        m_pred = Chem.MolFromSmiles(pred_smi)
        if m_scaf is None or m_pred is None:
            return None
        if m_scaf.GetNumAtoms() > m_pred.GetNumAtoms():
            return False
        mcs = rdFMCS.FindMCS(
            [m_scaf, m_pred], timeout=5,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        return (not mcs.canceled) and (mcs.numAtoms >= m_scaf.GetNumAtoms() * 0.8)
    except Exception:
        return None


                                                                                
                
                                                                                

def load_mg2_predictions(pred_dir: str | None) -> dict[str, str]:
\
\
\
       
    if not pred_dir or not os.path.isdir(pred_dir):
        return {}
    preds = {}
    for f in Path(pred_dir).glob("predictions_*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    img_id = str(obj.get("id", ""))
                    cxsmiles = obj.get("cxsmiles", "") or ""
                    if img_id and cxsmiles:
                                                                                
                                                                               
                                                                                
                                                                          
                        base = Path(img_id).name
                        stem = Path(base).stem
                        for key in {img_id, base, stem, f"{stem}.png"}:
                            if key:
                                preds[key] = cxsmiles
                except Exception:
                    pass
    return preds

def lookup_mg2_prediction(mg2_preds: dict[str, str], image_path: str) -> str:
                                                                            
    name = Path(str(image_path)).name
    stem = Path(name).stem
    for key in (name, stem, f"{stem}.png"):
        cx = mg2_preds.get(key, "")
        if cx:
            return cx
    return ""

def cxsmiles_to_bare(cxsmiles: str) -> str:
\
\
\
       
    return cxsmiles.split("|")[0].strip()


                                                                                
        
                                                                                

def eval_vqa_record(
    record: dict,
    scaffold_mode: str,
    mg2_preds: dict[str, str],
) -> dict:
                             
    result = {
        "index":            record.get("index", -1),
        "split":            record.get("split", ""),
        "correct_answer":   record.get("correct_answer", ""),
        "is_nova":          record.get("is_none_of_above", False),
        "predicted":        "SKIP",
        "is_correct":       False,
        "executor_status":  "ok",                                                                                    
    }

    instruction = record.get("instruction", "")
    esmiles     = record.get("original_esmiles", "")
    opt_smiles  = {lbl: record.get(f"option_{lbl}_smiles", "") for lbl in "ABCD"}

                                                                 
    if scaffold_mode == "gt":
        bare, label_to_idx = parse_esmiles(esmiles)
    else:       
        img_path = record.get("original_image_path", "")
        cx = lookup_mg2_prediction(mg2_preds, img_path)
        if not cx:
            result["executor_status"] = "mg2_prediction_missing"
            return result
        bare = cxsmiles_to_bare(cx)
                                                        
        _, label_to_idx = parse_esmiles(esmiles)

                                                                   
    subs = parse_instruction(instruction)
    if subs is None:
        result["executor_status"] = "instruction_unparseable"
        return result

                                                                 
    predicted_smi = apply_substitutions(bare, label_to_idx, subs, fill_remaining_with_h=True)
    if predicted_smi is None:
        result["executor_status"] = "substitution_failed"
        return result

    pred_canon = canonical(predicted_smi)
    if pred_canon is None:
        result["executor_status"] = "substitution_failed"
        return result

                                                               
    matched_label = None
    best_tani = -1.0
    best_label = None
    for lbl, opt_smi in opt_smiles.items():
        if not opt_smi:
            continue
        opt_canon = canonical(opt_smi)
        if opt_canon and opt_canon == pred_canon:
            matched_label = lbl
            break
        t = tanimoto(predicted_smi, opt_smi)
        if t is not None and t > best_tani:
            best_tani, best_label = t, lbl

    if matched_label:
        result["predicted"] = matched_label
    elif best_label:
                                     
        result["predicted"] = best_label
        result["executor_status"] = "tanimoto_fallback"
    else:
        result["executor_status"] = "no_match"
        return result

    result["is_correct"] = (result["predicted"] == result["correct_answer"])
    result["predicted_smiles"] = pred_canon
    result["best_tanimoto"] = round(best_tani, 4) if matched_label is None else None
    return result


def run_vqa_eval(args):
    splits = args.split or VQA_SPLITS
    mg2_preds = load_mg2_predictions(args.mg2_pred_dir) if args.scaffold_mode == "mg2" else {}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summary = {}
    for split in splits:
        split_path = os.path.join(VQA_DIR, split)
        if not os.path.isdir(split_path):
            print(f" : {split_path}")
            continue

        records = list(load_from_disk(split_path))
        if args.max_samples:
            records = records[:args.max_samples]

        save_path = Path(RESULTS_DIR) / f"vqa_{args.scaffold_mode}_{split}.jsonl"
        counters = defaultdict(int)

        with open(save_path, "w", encoding="utf-8") as fout:
            for i, rec in enumerate(records):
                rec = dict(rec)
                rec["index"] = i
                r = eval_vqa_record(rec, args.scaffold_mode, mg2_preds)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                counters[r["executor_status"]] += 1
                if r["is_correct"]:
                    counters["correct"] += 1
                counters["total"] += 1

        parseable = (counters["total"]
                     - counters["instruction_unparseable"]
                     - counters["mg2_prediction_missing"])
        acc_all  = counters["correct"] / max(counters["total"], 1)
        acc_exec = counters["correct"] / max(parseable, 1)
        coverage = parseable / max(counters["total"], 1)

        summary = {
            "split": split, "total": counters["total"],
            "parseable": parseable, "coverage": round(coverage, 4),
            "accuracy_all":      round(acc_all, 4),
            "accuracy_parseable": round(acc_exec, 4),
            "status_counts": dict(counters),
        }
        all_summary[split] = summary
        print(f"  {split}: total={counters['total']} coverage={coverage:.1%} "
              f"acc(all)={acc_all:.3f} acc(parsed)={acc_exec:.3f}")
                                                    
        for k, v in sorted(counters.items()):
            if k not in ("total", "correct", "instruction_unparseable"):
                print(f"    {k}: {v}")

    out = Path(RESULTS_DIR) / f"vqa_{args.scaffold_mode}_summary.json"
    with open(out, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSaved summary: {out}")


                                                                                
               
                                                                                

def eval_gen_record(record: dict, scaffold_mode: str, mg2_preds: dict[str, str]) -> dict:
    result = {
        "index":           record.get("index", -1),
        "split":           record.get("split", ""),
        "ground_truth":    record.get("correct_smiles", ""),
        "executor_status": "ok",
        "predicted_smiles": None,
        "is_valid":        False,
        "is_exact_match":  False,
        "tanimoto":        None,
        "scaffold_match":  None,
    }

    instruction    = record.get("instruction", "")
    esmiles        = record.get("original_esmiles", "")
    scaffold_clean = record.get("scaffold_smiles_clean", "")

                  
    if scaffold_mode == "gt":
        bare, label_to_idx = parse_esmiles(esmiles)
    else:
        img_path = record.get("original_image_path", "")
        cx = lookup_mg2_prediction(mg2_preds, img_path)
        if not cx:
            result["executor_status"] = "mg2_prediction_missing"
            return result
        bare = cxsmiles_to_bare(cx)
        _, label_to_idx = parse_esmiles(esmiles)

                           
    subs = parse_instruction(instruction)
    if subs is None:
        result["executor_status"] = "instruction_unparseable"
        return result

                  
    pred_smi = apply_substitutions(bare, label_to_idx, subs, fill_remaining_with_h=True)
    if pred_smi is None:
        result["executor_status"] = "substitution_failed"
        return result

                
    gt = result["ground_truth"]
    result["predicted_smiles"] = pred_smi
    result["is_valid"]         = is_valid(pred_smi)
    pred_canon = canonical(pred_smi)
    gt_canon   = canonical(gt)
    result["is_exact_match"] = (pred_canon is not None and gt_canon is not None
                                 and pred_canon == gt_canon)
    result["tanimoto"]        = tanimoto(pred_smi, gt) if result["is_valid"] else None
    result["scaffold_match"]  = scaffold_match(pred_smi, scaffold_clean) if result["is_valid"] else None
    return result


def run_gen_eval(args):
    splits = args.split or GEN_SPLITS
    mg2_preds = load_mg2_predictions(args.mg2_pred_dir) if args.scaffold_mode == "mg2" else {}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summary = {}
    for split in splits:
        split_path = os.path.join(GEN_QA_DIR, split)
        if not os.path.isdir(split_path):
            print(f"Missing split directory: {split_path}")
            continue

        records = list(load_from_disk(split_path))
        if args.max_samples:
            records = records[:args.max_samples]

        save_path = Path(RESULTS_DIR) / f"gen_{args.scaffold_mode}_{split}.jsonl"
        rows = []
        counters = defaultdict(int)

        with open(save_path, "w", encoding="utf-8") as fout:
            for i, rec in enumerate(records):
                rec = dict(rec)
                rec["index"] = i
                r = eval_gen_record(rec, args.scaffold_mode, mg2_preds)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                rows.append(r)
                counters["total"] += 1
                if r["executor_status"] == "instruction_unparseable":
                    counters["unparseable"] += 1
                elif r["executor_status"] == "mg2_prediction_missing":
                    counters["missing_prediction"] += 1

        valid_rows = [r for r in rows if r["is_valid"]]
        exact_rows = [r for r in rows if r["is_exact_match"]]
        tani_vals  = [r["tanimoto"] for r in valid_rows if r["tanimoto"] is not None]
        scaf_true  = sum(1 for r in rows if r.get("scaffold_match") is True)
        parseable  = counters["total"] - counters["unparseable"] - counters["missing_prediction"]

        summary = {
            "split": split, "total": counters["total"],
            "parseable": parseable,
            "coverage":      round(parseable / max(counters["total"], 1), 4),
            "validity_rate": round(len(valid_rows) / max(parseable, 1), 4),
            "exact_match":   round(len(exact_rows) / max(parseable, 1), 4),
            "avg_tanimoto":  round(sum(tani_vals) / max(len(tani_vals), 1), 4),
            "scaffold_match": round(scaf_true / max(parseable, 1), 4),
        }
        all_summary[split] = summary
        n = parseable
        print(f"  {split} (n={n}): validity={summary['validity_rate']:.3f} "
              f"EM={summary['exact_match']:.3f} tani={summary['avg_tanimoto']:.3f} "
              f"scaf={summary['scaffold_match']:.3f}  coverage={summary['coverage']:.1%}")
                          
        status_counts = {k: v for k, v in counters.items() if k not in ("total", "unparseable")}
        for k, v in sorted(status_counts.items()):
            print(f"    {k}: {v}")

    out = Path(RESULTS_DIR) / f"gen_{args.scaffold_mode}_summary.json"
    with open(out, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSaved summary: {out}")


                                                                                
    
                                                                                

def main():
    parser = argparse.ArgumentParser(description="Evaluate the MarkushGrapher-2 symbolic-executor pipeline.")
    parser.add_argument("--task", choices=["vqa", "gen"], required=True)
    parser.add_argument("--scaffold_mode", choices=["gt", "mg2"], default="gt",
                        help="Use gt for ground-truth scaffolds or mg2 for MarkushGrapher-2 predictions.")
    parser.add_argument("--split", nargs="+", default=None,
                        help="VQA: easy_basic/medium_basic/hard_basic; Gen: easy/hard")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional maximum number of samples per split.")
    parser.add_argument("--mg2_pred_dir", type=str, default=None,
                        help="Directory containing MarkushGrapher-2 prediction JSONL files when scaffold_mode=mg2.")
    parser.add_argument("--debug", action="store_true",
                        help="Print bounded symbolic-executor diagnostics.")
    args = parser.parse_args()

    global DEBUG_ENABLED
    DEBUG_ENABLED = args.debug

    mode_label = f"scaffold_mode={args.scaffold_mode}"
    print(f"{'='*60}\n  R-GroundBench MarkushGrapher-2 Pipeline Evaluator\n"
          f"  task={args.task}  {mode_label}\n  splits={args.split}\n{'='*60}")

    if args.task == "vqa":
        run_vqa_eval(args)
    else:
        run_gen_eval(args)


if __name__ == "__main__":
    main()
