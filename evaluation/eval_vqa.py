"""Evaluate models on R-GroundBench multiple-choice VQA tasks."""

import os, json, time, base64, re, argparse, asyncio
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from openai import AsyncOpenAI

                                          

                                                     

                                                        

                        

                                              



VLM_VQA_DIR = os.environ.get("RGBENCH_VQA_DIR",
    "data/vqa_dataset")
LLM_VQA_DIR = os.environ.get("RGBENCH_VQA_DIR",
    "data/llm_vqa_dataset")
RESULTS_DIR = os.environ.get("RGBENCH_RESULTS_DIR",
    "results/vqa")
         
                                                  
                                        
            

ALL_SPLITS = ["easy_basic", "easy_advanced",
              "medium_basic", "medium_advanced",
              "hard_basic", "hard_advanced"]
ALL_MODES  = ["img_img", "img_smi", "smi_img", "smi_smi"]
                                                   
MODE_TO_DATASET = {
    "img_img": VLM_VQA_DIR, "img_smi": VLM_VQA_DIR,
    "smi_img": VLM_VQA_DIR, "smi_smi": VLM_VQA_DIR,
}

                                                   
THINKING_MODELS = {
    "deepseek-r1",
    "o3",
}

def get_max_tokens(model):
    return 16000 if model in THINKING_MODELS else 1000

DEFAULT_CONCURRENCY = 8
MAX_RETRIES         = 3
RETRY_DELAY         = 2.0
REQUEST_TIMEOUT     = 60                     

DEBUG_ENABLED = False

SYSTEM_PROMPT = (
    "You are an expert chemist answering multiple choice questions. "
    "YOUR ONLY OUTPUT MUST BE A SINGLE LETTER: A, B, C, or D. "
    "DO NOT write any explanation, reasoning, analysis, or punctuation. "
    "DO NOT start your response with words. "
    "ONLY output one of these four characters: A B C D"
)
NOVA_NOTE = (
    "Note: Option D is 'None of the above'. "
    "Select D if no option is chemically valid or correct."
)


def encode_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def make_image_content(b64):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}

def build_messages(record, mode):
    """Build OpenAI-compatible chat messages for one VQA record and input/output mode."""
    is_basic   = "basic" in record.get("split", "")
    is_nova    = record.get("is_none_of_above", False)
    instruction = record.get("instruction", "").strip()
    question    = record.get("question", "").strip()
    nova_note   = NOVA_NOTE if is_nova else ""
    esmiles     = record.get("original_esmiles", "")
    opt_labels  = ["A", "B", "C", "D"]
    opt_smiles  = {lbl: record.get(f"option_{lbl}_smiles", "") for lbl in opt_labels}

    content = []

          
    if mode in ("img_img", "img_smi"):
        b64 = encode_image(record.get("original_image_path", ""))
        if b64 is None:
            return None
        content.append(make_image_content(b64))

             
    opt_b64 = {}
    if mode in ("img_img", "smi_img"):
        for lbl in opt_labels:
            img_path = record.get(f"option_{lbl}_image", "")
            if not img_path:
                opt_b64[lbl] = None
            else:
                b64 = encode_image(img_path)
                if b64 is None:
                    return None
                opt_b64[lbl] = b64

                
    if is_basic:
        if mode in ("img_img", "smi_img"):
            input_line = ("The image above shows a Markush structure."
                          if mode == "img_img"
                          else f"Markush structure (SMILES):\n{esmiles}")
            prompt = (f"{input_line}\n\nInstruction: {instruction}\n\n{question}\n\n"
                      f"The four candidate molecules are shown below as images (A, B, C, D).\n{nova_note}")
        else:
            input_line = ("The image above shows a Markush structure."
                          if mode == "img_smi"
                          else f"Markush structure (SMILES):\n{esmiles}")
            prompt = (f"{input_line}\n\nInstruction: {instruction}\n\n{question}\n\n"
                      f"Candidate molecules (SMILES):\nA: {opt_smiles['A']}\nB: {opt_smiles['B']}\n"
                      f"C: {opt_smiles['C']}\nD: {opt_smiles['D']}\n{nova_note}")
    else:
        prop = record.get("property", "")
        is_indication = (prop == "indication_mechanism")
        if mode in ("img_img", "smi_img"):
            input_line = ("The image above shows a Markush structure with an R-group position."
                          if mode == "img_img"
                          else f"Markush structure (SMILES):\n{esmiles}")
            option_desc = (
                "The four R-group options are shown below as images (A, B, C, D)."
                if is_indication else
                "The four candidate molecules are shown below as images (A, B, C, D)."
            )
            prompt = f"{input_line}\n\n{question}\n\n{option_desc}"
        else:
            input_line = ("The image above shows a Markush structure with an R-group position."
                          if mode == "img_smi"
                          else f"Markush structure (SMILES):\n{esmiles}")
            option_label = "Candidate R-groups" if is_indication else "Candidate molecules"
            prompt = (f"{input_line}\n\n{question}\n\n"
                      f"{option_label} (SMILES):\nA: {opt_smiles['A']}\nB: {opt_smiles['B']}\n"
                      f"C: {opt_smiles['C']}\nD: {opt_smiles['D']}")

            
    if mode in ("img_img", "smi_img"):
        content.append({"type": "text", "text": prompt})
        for lbl in opt_labels:
            b64 = opt_b64[lbl]
            if b64 is not None:
                content.append(make_image_content(b64))
            else:
                content.append({"type": "text", "text": f"Option {lbl}: None of the above"})
        content.append({"type": "text", "text": "Your answer (single letter only, no explanation): "})
    else:
        content.append({"type": "text", "text": prompt})
        content.append({"type": "text", "text": "Your answer (single letter only, no explanation): "})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


def extract_answer_from_text(text):
    """Extract a single multiple-choice answer letter from model text."""
    if not text:
        return "INVALID"
    t = text.strip().upper()
    if t in ("A", "B", "C", "D"):
        return t
                                                  
    for pat in [
        r"^([ABCD])[.\):\s]",                                      
        r"^\(([ABCD])\)",                                  
        r"ANSWER[:\s]+([ABCD])",                     
        r"CORRECT[^.]*?([ABCD])",                              
        r"[:]+([ABCD])",                       
        r"([ABCD])[^A-Z]",                          
        r"([ABCD])",                               
        r"([ABCD])[,\s]",                                   
    ]:
        m = re.search(pat, t)
        if m:
            return m.group(1)
                                                                          
                                                                          
    all_matches = re.findall(r"\b([ABCD])\b", t)
    if all_matches:
        return all_matches[-1]
    return "INVALID"

def extract_answer(resp, model):
                                         
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return extract_answer_from_text(content)
                                        
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if reasoning:
        ans = extract_answer_from_text(reasoning[-300:])
        if ans != "INVALID":
            return ans
    if DEBUG_ENABLED:
        print(f"[DEBUG] content={repr(content[:100])} reasoning_tail={repr(reasoning[-200:])}")
    return "INVALID"


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
                predicted = extract_answer(resp, model)
                return raw, predicted, elapsed
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return "TIMEOUT", "INVALID", 0.0
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return f"ERROR:{str(e)[:80]}", "INVALID", 0.0
    return "", "INVALID", 0.0


async def eval_split_async(client, model, mode, split, records, save_path, concurrency, max_samples, force=False, debug=False):
    done_indices = set()
    if save_path.exists() and not force:
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["index"])
                except:
                    pass
    elif save_path.exists() and force:
        save_path.unlink()

    if max_samples:
        records = records[:max_samples]
    total = len(records)
    todo  = [(i, r) for i, r in enumerate(records) if i not in done_indices]

    print(f"\n  {model} | {mode} | {split}: total={total}, completed={len(done_indices)}, pending={len(todo)}")
    if not todo:
        print("   No pending records; loading existing results.")
        return load_results_from_file(save_path, total)

    semaphore   = asyncio.Semaphore(concurrency)
    result_lock = asyncio.Lock()
    counters    = {"correct": 0, "invalid": 0, "done": len(done_indices)}
    start_time  = time.time()
    debug_count = [0]                                             

    async def process_one(idx, record):
        messages = build_messages(record, mode)
        if messages is None:
            async with result_lock:
                counters["done"] += 1
            return
        raw, predicted, elapsed = await call_api_with_retry(client, model, messages, semaphore)
        if debug:
            async with result_lock:
                if debug_count[0] >= 3:
                    pass
                else:
                    debug_count[0] += 1
                    print(f"[DEBUG-RAW {model}|{mode}|{split}|{debug_count[0]}] "
                          f"predicted={predicted!r} raw={raw[:200]!r}", flush=True)
        correct_answer = record.get("correct_answer", "")
        is_correct = (predicted == correct_answer)
        result = {
            "index": idx, "split": record.get("split", split), "mode": mode, "model": model,
            "correct_answer": correct_answer, "predicted": predicted, "raw_response": raw,
            "is_correct": is_correct, "elapsed": round(elapsed, 3),
            "is_none_of_above": record.get("is_none_of_above", False),
            "nova_answer_is_d": record.get("nova_answer_is_d", False),
            "nova_type":        record.get("nova_type", ""),
            "property": record.get("property", ""),
        }
        async with result_lock:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if is_correct:
                counters["correct"] += 1
            if predicted == "INVALID":
                counters["invalid"] += 1
            counters["done"] += 1
            done = counters["done"]
            if done % 50 == 0 or done == total:
                processed = done - len(done_indices)
                elapsed_total = time.time() - start_time
                acc   = counters["correct"] / max(processed, 1)
                speed = processed / max(elapsed_total, 1)
                eta   = (total - done) / max(speed, 0.001)
                print(f"    [{done}/{total}] acc={acc:.3f} invalid={counters['invalid']} "
                      f"speed={speed:.1f}/s ETA={eta/60:.1f}min")

    await asyncio.gather(*[process_one(idx, r) for idx, r in todo])
    return load_results_from_file(save_path, total)


def load_results_from_file(save_path, total):
    results = []
    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass
    if not results:
        return {"accuracy": None, "total": total, "evaluated": 0}

    correct   = sum(1 for r in results if r.get("is_correct"))
    invalid   = sum(1 for r in results if r.get("predicted") == "INVALID")
    evaluated = len(results)
    normal    = [r for r in results if not r.get("is_none_of_above")]
    type_b    = [r for r in results if r.get("nova_type") == "type_b"]
    type_c    = [r for r in results if r.get("nova_type") == "type_c"]
    valid_d   = [r for r in results if r.get("nova_type") == "valid_d"]

    def acc(lst): return round(sum(r["is_correct"] for r in lst) / len(lst), 4) if lst else None

    prop_acc = {}
    prop_groups = defaultdict(list)
    for r in results:
        if r.get("property"):
            prop_groups[r["property"]].append(r)
    for prop, recs in prop_groups.items():
        prop_acc[prop] = round(sum(r["is_correct"] for r in recs) / len(recs), 4)

    return {
        "accuracy":         round(correct / evaluated, 4) if evaluated else None,
        "correct":          correct, "evaluated": evaluated, "total": total,
        "invalid_count":    invalid,
        "normal_acc":       acc(normal),
        "normal_count":     len(normal),
        "type_b_acc":       acc(type_b),                   
        "type_b_count":     len(type_b),
        "type_c_acc":       acc(type_c),                               
        "type_c_count":     len(type_c),
        "valid_d_acc":      acc(valid_d),                  
        "valid_d_count":    len(valid_d),
        "property_acc":     prop_acc,
    }


async def run_eval(args):
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    models = args.model
    modes  = args.mode  if args.mode  else ALL_MODES
    splits = args.split if args.split else ALL_SPLITS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_summary = {}

    for model in models:
        print(f"\n{'='*65}\n  Model: {model}")
        if model in THINKING_MODELS:
            print(f"   Thinking: max_tokens={get_max_tokens(model)}, timeout={REQUEST_TIMEOUT}s")
        print(f"  Modes: {modes}  Splits: {splits}\n{'='*65}")

        model_summary = defaultdict(dict)
        model_tag = model.replace("/", "_").replace(":", "_")
        save_dir  = Path(RESULTS_DIR) / model_tag
        save_dir.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            for split in splits:
                split_path = os.path.join(MODE_TO_DATASET[mode], split)
                if not os.path.isdir(split_path):
                    print(f"   Missing split directory: {split_path}")
                    continue
                records   = list(load_from_disk(split_path))
                save_path = save_dir / f"{mode}_{split}.jsonl"
                stats = await eval_split_async(
                    client, model, mode, split, records,
                    save_path, args.concurrency, args.max_samples,
                    force=args.force,
                    debug=args.debug
                )
                model_summary[mode][split] = stats
                print(f"   {mode} | {split}: acc={stats['accuracy']} ({stats['correct']}/{stats['evaluated']})")

                                               
        for mode in modes:
            if mode not in model_summary:
                continue
            mode_result = {}
            for split in splits:
                stats = model_summary[mode].get(split)
                if stats is None:
                    continue
                mode_result[split] = {
                    "accuracy":      stats.get("accuracy"),
                    "correct":       stats.get("correct"),
                    "evaluated":     stats.get("evaluated"),
                    "invalid_count": stats.get("invalid_count"),
                                     
                    "normal_acc":    stats.get("normal_acc"),
                    "normal_count":  stats.get("normal_count"),
                    "type_b_acc":    stats.get("type_b_acc"),
                    "type_b_count":  stats.get("type_b_count"),
                    "type_c_acc":    stats.get("type_c_acc"),
                    "type_c_count":  stats.get("type_c_count"),
                    "valid_d_acc":   stats.get("valid_d_acc"),
                    "valid_d_count": stats.get("valid_d_count"),
                                     
                    "property_acc":  stats.get("property_acc", {}),
                }
            mode_json_path = save_dir / f"{mode}.json"
            with open(mode_json_path, "w", encoding="utf-8") as f:
                json.dump(mode_result, f, indent=2, ensure_ascii=False)
            print(f"  Saved summary: {mode_json_path}")

        all_summary[model] = dict(model_summary)

    print_summary_table(all_summary, modes, splits)


def print_summary_table(all_summary, modes, splits):
    print(f"\n{'='*75}\n  accuracy\n{'='*75}")
    header = f"  {'Model|Mode':<35}" + "".join(f" {s[:12]:>13}" for s in splits)
    print(header)
    print(f"  {'-'*73}")
    for model, mode_data in all_summary.items():
        for mode in modes:
            split_data = mode_data.get(mode, {})
            row = f"  {(model[:15]+'|'+mode):<35}"
            for split in splits:
                acc = split_data.get(split, {}).get("accuracy")
                row += f" {(f'{acc:.3f}' if acc is not None else 'N/A'):>13}"
            print(row)

    print(f"\n  None-of-Above Basic splits:")
    print(f"  {'Key':<38} {'normal':>8} {'type_b':>8} {'type_c':>8} {'valid_d':>8}")
    print(f"  {'-'*76}")
    for model, mode_data in all_summary.items():
        for mode in modes:
            for split in [s for s in splits if "basic" in s]:
                stats = mode_data.get(mode, {}).get(split, {})
                if not stats:
                    continue
                key = f"{model[:12]}|{mode}|{split[:10]}"
                def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
                print(f"  {key:<38} "
                      f"{fmt(stats.get('normal_acc')):>8} "
                      f"{fmt(stats.get('type_b_acc')):>8} "
                      f"{fmt(stats.get('type_c_acc')):>8} "
                      f"{fmt(stats.get('valid_d_acc')):>8}")
    print(f"{'='*75}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R-GroundBench multiple-choice VQA tasks.")
    parser.add_argument("--model", nargs="+", default=["claude-sonnet-4-6"],
                        help="One or more OpenAI-compatible model names.")
    parser.add_argument("--mode",  nargs="+", choices=ALL_MODES,  default=None,
                        help="Evaluation modes. Defaults to all modes.")
    parser.add_argument("--split", nargs="+", choices=ALL_SPLITS, default=None,
                        help="Dataset splits. Defaults to all splits.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional maximum number of samples per split.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Maximum number of concurrent API requests.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing JSONL results (re-evaluate all questions)")
    parser.add_argument("--debug", action="store_true",
                        help="Print a small number of raw model responses for parser diagnostics.")
    return parser.parse_args()

def main():
    args = parse_args()
    global DEBUG_ENABLED
    DEBUG_ENABLED = args.debug
    print(f"{'='*65}\n  R-GroundBench VQA Evaluator")
    print(f"  Models: {args.model}\n  Modes: {args.mode or ALL_MODES}")
    print(f"  Splits: {args.split or ALL_SPLITS}\n  Concurrency: {args.concurrency}")
    print(f"  Max samples: {args.max_samples or 'all'}\n  Results directory: {RESULTS_DIR}\n{'='*65}")
    asyncio.run(run_eval(args))

if __name__ == "__main__":
    main()
