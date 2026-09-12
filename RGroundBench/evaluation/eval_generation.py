"""Evaluate models on R-GroundBench molecule generation tasks."""

import os, json, time, base64, re, argparse, asyncio
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from openai import AsyncOpenAI
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdFMCS

RDLogger.DisableLog('rdApp.*')

                                                         

                                          

                                                     

                                                        

                        

                                              



                                                          
                                                                    
GEN_QA_DIR  = os.environ.get("RGBENCH_GEN_DIR",
    "data/generation_qa")
RESULTS_DIR = os.environ.get("RGBENCH_RESULTS_DIR",
    "results/generation")
         
                                                    
                                               
            
                                    
                                    

ALL_SPLITS  = ["easy", "hard"]
ALL_MODES   = ["smi", "img"]

THINKING_MODELS = {
    "deepseek-r1",
    "o3",
}

DEFAULT_CONCURRENCY = 8
MAX_RETRIES         = 3
RETRY_DELAY         = 2.0
REQUEST_TIMEOUT     = 120

def get_max_tokens(model):
    return 20000 if model in THINKING_MODELS else 512


                                                            
          
                                                            

SYSTEM_PROMPT = (
    "You are an expert chemist. "
    "Given a Markush structure and an R-group substitution instruction, "
    "generate the SMILES string of the resulting molecule. "
    "Return ONLY the SMILES string enclosed in <smiles> and </smiles> tags. "
    "Do not include any explanation, reasoning, or additional text. "
    "Example format: <smiles>CCO</smiles>"
)

def encode_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return None

def make_image_content(b64: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}

def build_messages(record: dict, mode: str) -> list | None:
    """Build OpenAI-compatible chat messages for one generation record."""
\
\
\
\
\
       
    instruction = record.get("instruction", "").strip()
    esmiles     = record.get("original_esmiles", "")
    img_path    = record.get("original_image_path", "")

    content = []

    if mode == "img":
        b64 = encode_image(img_path)
        if b64 is None:
            return None
        content.append(make_image_content(b64))
        prompt = (
            "The image above shows a Markush structure with R-group position(s).\n\n"
            f"Instruction: {instruction}\n\n"
            "Generate the SMILES of the resulting molecule after applying this instruction. "
            "Return ONLY the SMILES enclosed in <smiles></smiles> tags."
        )
    else:
                                
        prompt = (
            f"Markush structure (SMILES):\n{esmiles}\n\n"
            f"Instruction: {instruction}\n\n"
            "Generate the SMILES of the resulting molecule after applying this instruction. "
            "Return ONLY the SMILES enclosed in <smiles></smiles> tags."
        )

    content.append({"type": "text", "text": prompt})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


                                                            
                 
                                                            

def extract_smiles_from_response(text: str) -> str:
    """Extract a SMILES candidate from a tagged or plain-text model response."""
\
\
\
       
    if not text:
        return ""

    def clean_candidate(candidate: str) -> str:
        candidate = candidate.strip().strip("`")
        candidate = re.sub(r"^SMILES\s*(?:is|:)\s*", "", candidate, flags=re.IGNORECASE).strip()
                                                                          
                                                                              
                                                                  
        candidate = candidate.rstrip("")
        while candidate.endswith("."):
            candidate = candidate[:-1].rstrip()
        return candidate

          
    m = re.search(r"<smiles>(.*?)</smiles>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return clean_candidate(m.group(1))

                                                       
    m = re.search(r"\bSMILES\s+is\s+([^\s]+)", text, re.IGNORECASE)
    if m:
        return clean_candidate(m.group(1))

                       
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        candidate = clean_candidate(lines[-1])
                                       
        if len(candidate) < 300 and " " not in candidate:
            return candidate
    return ""

def canonical_smiles(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def is_valid_smiles(smi: str) -> bool:
    if not smi: return False
    return canonical_smiles(smi) is not None

def tanimoto_similarity(smi1: str, smi2: str) -> float | None:
                                                   
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None: return None
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except:
        return None

def scaffold_match(predicted_smi: str, scaffold_clean_smi: str) -> bool | None:
\
\
\
\
\
\
\
       
    try:
                                                      
                                 
        scaffold_no_star = re.sub(r"\*|\[\*\]|\[#0\]", "[H]", scaffold_clean_smi)
        mol_scaffold = Chem.MolFromSmiles(scaffold_no_star)
        mol_pred     = Chem.MolFromSmiles(predicted_smi)
        if mol_scaffold is None or mol_pred is None:
            return None

                          
        if mol_scaffold.GetNumAtoms() > mol_pred.GetNumAtoms():
            return False

        mcs_result = rdFMCS.FindMCS(
            [mol_scaffold, mol_pred],
            timeout=5,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=False,
            ringMatchesRingOnly=False,
        )
        if mcs_result.canceled:
            return None

        mcs_size = mcs_result.numAtoms
        scaffold_size = mol_scaffold.GetNumAtoms()
                                
        return mcs_size >= scaffold_size * 0.8
    except:
        return None

def evaluate_prediction(predicted_raw: str, ground_truth: str,
                         scaffold_clean: str) -> dict:
\
\
\
\
\
\
\
\
\
       
                                    
    scaffold_clean = scaffold_clean.split("<sep>")[0].strip()

    extracted = extract_smiles_from_response(predicted_raw)
    valid  = is_valid_smiles(extracted)
    canon  = canonical_smiles(extracted) if valid else None
    gt_canon = canonical_smiles(ground_truth)

    exact = (canon is not None and gt_canon is not None and canon == gt_canon)
    tani  = tanimoto_similarity(extracted, ground_truth) if valid else None
    s_match = scaffold_match(extracted, scaffold_clean) if valid else None

    return {
        "predicted_smiles": extracted,
        "is_valid":          valid,
        "is_exact_match":    exact,
        "tanimoto":          tani,
        "scaffold_match":    s_match,
    }


                                                            
       
                                                            

async def call_api_with_retry(client, model, messages, semaphore):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model, messages=messages,
                        max_tokens=get_max_tokens(model), temperature=0,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = time.time() - t0
                raw = (resp.choices[0].message.content or "").strip()
                                                           
                if not raw:
                    reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
                    raw = reasoning
                return raw, elapsed
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return "TIMEOUT", 0.0
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return f"ERROR:{str(e)[:80]}", 0.0
    return "", 0.0


                                                            
                
                                                            

async def eval_split_async(client, model, mode, split, records,
                            save_path: Path, concurrency: int, max_samples: int):
    done_indices = set()
    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["index"])
                except:
                    pass

    if max_samples:
        records = records[:max_samples]
    total = len(records)
    todo  = [(i, r) for i, r in enumerate(records) if i not in done_indices]

    print(f"\n  {model} | {mode} | {split}: {total}, {len(done_indices)}, {len(todo)}")
    if not todo:
        print(f"   ")
        return load_results_from_file(save_path, total)

    semaphore   = asyncio.Semaphore(concurrency)
    result_lock = asyncio.Lock()
    counters    = {"exact": 0, "valid": 0, "invalid": 0, "done": len(done_indices)}
    start_time  = time.time()

    async def process_one(idx, record):
        messages = build_messages(record, mode)
        if messages is None:
            async with result_lock:
                counters["done"] += 1
            return

        raw, elapsed = await call_api_with_retry(client, model, messages, semaphore)

        ground_truth    = record.get("correct_smiles", "")
        scaffold_clean  = record.get("scaffold_smiles_clean", "")
        difficulty      = record.get("difficulty", split)

        metrics = evaluate_prediction(raw, ground_truth, scaffold_clean)

        result = {
            "index":            idx,
            "split":            split,
            "difficulty":       difficulty,
            "mode":             mode,
            "model":            model,
            "ground_truth":     ground_truth,
            "raw_response":     raw[:500],              
            "elapsed":          round(elapsed, 3),
            **metrics,
        }

        async with result_lock:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if metrics["is_exact_match"]: counters["exact"] += 1
            if metrics["is_valid"]:       counters["valid"] += 1
            else:                         counters["invalid"] += 1
            counters["done"] += 1
            done = counters["done"]
            if done % 50 == 0 or done == total:
                processed = done - len(done_indices)
                elapsed_total = time.time() - start_time
                exact_acc  = counters["exact"]  / max(processed, 1)
                valid_rate = counters["valid"]   / max(processed, 1)
                invalid_n  = counters["invalid"]
                speed = processed / max(elapsed_total, 1)
                eta   = (total - done) / max(speed, 0.001)
                print(f"    [{done}/{total}] exact={exact_acc:.3f} valid={valid_rate:.3f} "
                      f"invalid={invalid_n} speed={speed:.1f}/s ETA={eta/60:.1f}min")

    await asyncio.gather(*[process_one(idx, r) for idx, r in todo])
    return load_results_from_file(save_path, total)


def load_results_from_file(save_path: Path, total: int) -> list:
    results = []
    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass
    return results


                                                            
      
                                                            

def compute_summary(all_results: dict) -> dict:
    summary = {}
    for split, results in all_results.items():
        if not results:
            summary[split] = {}
            continue

        valid_results  = [r for r in results if r.get("is_valid")]
        exact_results  = [r for r in results if r.get("is_exact_match")]
        tani_values    = [r["tanimoto"] for r in valid_results if r.get("tanimoto") is not None]

                                              
                                  
        n = len(results)
        scaffold_true = sum(1 for r in results if r.get("scaffold_match") is True)

        summary[split] = {
            "total":          n,
            "validity_rate":  round(len(valid_results) / max(n, 1), 4),
            "exact_match":    round(len(exact_results) / max(n, 1), 4),
            "avg_tanimoto":   round(sum(tani_values) / max(len(tani_values), 1), 4),
            "scaffold_match": round(scaffold_true / max(n, 1), 4),
        }

    return summary


def print_summary(summary: dict, model: str, mode: str):
    print(f"\n{'='*65}")
    print(f"  {model} | {mode} | Generation QA Summary")
    print(f"{'='*65}")
    print(f"  {'Split':<12} {'N':>5} {'Validity':>9} {'ExactMatch':>11} {'AvgTani':>9} {'ScaffMatch':>11}")
    print(f"  {'-'*60}")
    for split in ALL_SPLITS:
        s = summary.get(split, {})
        if not s:
            print(f"  {split:<12} {'N/A':>5}")
            continue
        print(f"  {split:<12} {s['total']:>5} "
              f"{s['validity_rate']:>9.3f} "
              f"{s['exact_match']:>11.3f} "
              f"{s['avg_tanimoto']:>9.3f} "
              f"{s['scaffold_match']:>11.3f}")


                                                            
     
                                                            

async def run_eval(args):
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    modes   = [args.mode] if args.mode else ALL_MODES
    splits  = args.splits if args.splits else ALL_SPLITS
    models  = args.model

    for model in models:
        for mode in modes:
            model_dir = Path(RESULTS_DIR) / model
            model_dir.mkdir(parents=True, exist_ok=True)

            all_results = {}

            for split in splits:
                split_path = Path(GEN_QA_DIR) / split
                if not split_path.exists():
                    print(f"    Missing split directory for {split}: {split_path}")
                    continue

                try:
                    ds = load_from_disk(str(split_path))
                    records = list(ds)
                except Exception as e:
                    print(f"    Failed to load split {split}: {e}")
                    continue

                save_path = model_dir / f"{mode}_{split}.jsonl"
                results   = await eval_split_async(
                    client, model, mode, split, records,
                    save_path, args.concurrency, args.max_samples
                )
                all_results[split] = results

                  
            summary = compute_summary(all_results)
            print_summary(summary, model, mode)

            summary_path = model_dir / f"{mode}_generation_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"model": model, "mode": mode, "summary": summary},
                          f, indent=2, ensure_ascii=False)
            print(f"  Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate R-GroundBench molecule generation tasks.")
    parser.add_argument("--model", nargs="+", required=True,
                        help="One or more OpenAI-compatible model names, e.g. --model gpt-4o.")
    parser.add_argument("--mode", type=str, default=None, choices=ALL_MODES,
                        help="Generation mode: smi for symbolic input or img for image input. Defaults to both.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Dataset splits to evaluate. Defaults to easy and hard.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional maximum number of samples per split; 0 means all samples.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Maximum number of concurrent API requests.")
    args = parser.parse_args()

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
