# 文件位置: Moleditor/code/03_build_dataset.py

from datasets import load_from_disk, Dataset
from rdkit import Chem
from rdkit.Chem import Draw, SanitizeMol
import re
import random
import json
import os
from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────
INPUT_DIR  = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/molparser_markush"
OUTPUT_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/rgroup_edit_dataset"
NUM_VARIANTS = 3
RANDOM_SEED  = 42
BATCH_SIZE   = 5000
MAX_RGROUPS  = 10      # ← 新增：R-group数量上限，超过则跳过
# ─────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

GENERAL_SUBSTITUENTS = [
    {"key": "methyl",          "smiles": "C",        "en": "methyl",          "zh": "甲基"},
    {"key": "ethyl",           "smiles": "CC",       "en": "ethyl",           "zh": "乙基"},
    {"key": "propyl",          "smiles": "CCC",      "en": "propyl",          "zh": "丙基"},
    {"key": "isopropyl",       "smiles": "C(C)C",    "en": "isopropyl",       "zh": "异丙基"},
    {"key": "tert-butyl",      "smiles": "C(C)(C)C", "en": "tert-butyl",      "zh": "叔丁基"},
    {"key": "phenyl",          "smiles": "c1ccccc1", "en": "phenyl",          "zh": "苯基"},
    {"key": "hydroxyl",        "smiles": "O",        "en": "hydroxyl",        "zh": "羟基"},
    {"key": "methoxy",         "smiles": "OC",       "en": "methoxy",         "zh": "甲氧基"},
    {"key": "fluoro",          "smiles": "F",        "en": "fluoro",          "zh": "氟"},
    {"key": "chloro",          "smiles": "Cl",       "en": "chloro",          "zh": "氯"},
    {"key": "bromo",           "smiles": "Br",       "en": "bromo",           "zh": "溴"},
    {"key": "amino",           "smiles": "N",        "en": "amino",           "zh": "氨基"},
    {"key": "trifluoromethyl", "smiles": "C(F)(F)F", "en": "trifluoromethyl", "zh": "三氟甲基"},
    {"key": "cyano",           "smiles": "C#N",      "en": "cyano",           "zh": "氰基"},
    {"key": "acetyl",          "smiles": "C(=O)C",   "en": "acetyl",          "zh": "乙酰基"},
]

HALOGEN_SUBSTITUENTS = [
    {"key": "fluoro",  "smiles": "F",  "en": "fluoro",  "zh": "氟"},
    {"key": "chloro",  "smiles": "Cl", "en": "chloro",  "zh": "氯"},
    {"key": "bromo",   "smiles": "Br", "en": "bromo",   "zh": "溴"},
]

N_SUBSTITUENTS = [
    {"key": "methyl",          "smiles": "C",        "en": "methyl",          "zh": "甲基"},
    {"key": "ethyl",           "smiles": "CC",       "en": "ethyl",           "zh": "乙基"},
    {"key": "propyl",          "smiles": "CCC",      "en": "propyl",          "zh": "丙基"},
    {"key": "isopropyl",       "smiles": "C(C)C",    "en": "isopropyl",       "zh": "异丙基"},
    {"key": "tert-butyl",      "smiles": "C(C)(C)C", "en": "tert-butyl",      "zh": "叔丁基"},
    {"key": "phenyl",          "smiles": "c1ccccc1", "en": "phenyl",          "zh": "苯基"},
    {"key": "acetyl",          "smiles": "C(=O)C",   "en": "acetyl",          "zh": "乙酰基"},
    {"key": "trifluoromethyl", "smiles": "C(F)(F)F", "en": "trifluoromethyl", "zh": "三氟甲基"},
]

O_SUBSTITUENTS = [
    {"key": "methyl",     "smiles": "C",        "en": "methyl",     "zh": "甲基"},
    {"key": "ethyl",      "smiles": "CC",       "en": "ethyl",      "zh": "乙基"},
    {"key": "propyl",     "smiles": "CCC",      "en": "propyl",     "zh": "丙基"},
    {"key": "isopropyl",  "smiles": "C(C)C",    "en": "isopropyl",  "zh": "异丙基"},
    {"key": "tert-butyl", "smiles": "C(C)(C)C", "en": "tert-butyl", "zh": "叔丁基"},
    {"key": "phenyl",     "smiles": "c1ccccc1", "en": "phenyl",     "zh": "苯基"},
]

# ── 5种Instruction风格，每种3个模板 ──────────────────────

INSTRUCTION_STYLES = ["direct", "chemical", "conversational", "result", "conditional"]

INSTRUCTION_TEMPLATES = {
    "direct": [
        lambda s: f"Replace {s}.",
        lambda s: f"Replace {s} in this molecule.",
        lambda s: f"Perform the following substitution: replace {s}.",
    ],
    "chemical": [
        lambda s: f"Substitute {s}.",
        lambda s: f"Carry out the substitution of {s}.",
        lambda s: f"Please substitute {s} on the core scaffold.",
    ],
    "conversational": [
        lambda s: f"Can you change {s}?",
        lambda s: f"Could you change {s} for me?",
        lambda s: f"Hey, change {s} please.",
    ],
    "result": [
        lambda s: f"I need {s}.",
        lambda s: f"I want {s} in the final molecule.",
        lambda s: f"The target molecule should have {s}.",
    ],
    "conditional": [
        lambda s: f"Set {s}.",
        lambda s: f"Assign {s} to this structure.",
        lambda s: f"Define {s} for this Markush scaffold.",
    ],
}

def format_rgroup_list(assignment: dict, style: str) -> str:
    items = list(assignment.items())
    if style == "direct":
        parts = [f"{name} with {info['en']}" for name, info in items]
    elif style == "chemical":
        parts = [f"the {name} substituent with a {info['en']} group" for name, info in items]
    elif style == "conversational":
        parts = [f"{name} to {info['en']}" for name, info in items]
    elif style == "result":
        parts = [f"the {name} position to be {info['en']}" for name, info in items]
    elif style == "conditional":
        parts = [f"{name} as {info['en']}" for name, info in items]
    else:
        parts = [f"{name} with {info['en']}" for name, info in items]
    return ", and ".join(parts) if len(parts) > 1 else parts[0]


def generate_all_instructions(assignment: dict) -> list:
    results = []
    for style in INSTRUCTION_STYLES:
        rgroup_str = format_rgroup_list(assignment, style)
        template = random.choice(INSTRUCTION_TEMPLATES[style])
        results.append({
            "instruction":       template(rgroup_str),
            "instruction_style": style,
        })
    return results


def get_candidate_pool(group_name: str, esmiles: str = None, atom_idx: int = None) -> list:
    if group_name in {"X", "Y", "Z"}:
        return HALOGEN_SUBSTITUENTS
    if esmiles is not None and atom_idx is not None:
        neighbor_symbol = get_attachment_atom(esmiles, atom_idx)
        if neighbor_symbol == "N":
            return N_SUBSTITUENTS
        if neighbor_symbol == "O":
            return O_SUBSTITUENTS
    return GENERAL_SUBSTITUENTS


def get_attachment_atom(esmiles: str, wildcard_idx: int) -> str:
    try:
        core = esmiles.split("<sep>")[0]
        mol = Chem.MolFromSmiles(core, sanitize=False)
        if mol is None:
            return None
        target_atom = mol.GetAtomWithIdx(wildcard_idx)
        if target_atom.GetSymbol() != "*":
            return None
        neighbors = target_atom.GetNeighbors()
        if not neighbors:
            return None
        return neighbors[0].GetSymbol()
    except Exception:
        return None


def is_real_rgroup(name: str) -> bool:
    if re.match(r"^R\[?\d+\]?$", name):
        return True
    if name == "R":
        return True
    if re.match(r"^[XYZWL]$", name):
        return True
    return False

def parse_rgroups(esmiles):
    matches = re.findall(r"<a>(\d+):([^<]+)</a>", esmiles)
    return {gname: int(idx) for idx, gname in matches if is_real_rgroup(gname)}

def is_clean_sample(esmiles):
    if any(t in esmiles for t in ["<r>", "<c>", "<dum>"]):
        return False
    rgroups = parse_rgroups(esmiles)
    if len(rgroups) == 0:
        return False
    if len(rgroups) > MAX_RGROUPS:        # ← 新增过滤条件
        return False
    return True

def replace_rgroups(esmiles, assignment):
    result = esmiles.split("<sep>")[0]
    for sub_info in assignment.values():
        if "*" not in result:
            return None
        result = result.replace("*", sub_info["smiles"], 1)
    return result

def strict_validate(smiles: str):
    if "*" in smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        if Chem.SanitizeMol(mol, catchErrors=True) != 0:
            return None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetSymbol() != "*":
                return None
        return mol
    except Exception:
        return None

def smiles_to_image(smiles, size=(300, 300)):
    try:
        mol = strict_validate(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None

def generate_records_for_variant(esmiles, original_image, assignment,
                                  new_smiles, new_image):
    rgroup_def = {
        n: {"substituent_key": v["key"], "substituent_smiles": v["smiles"],
            "substituent_en": v["en"], "substituent_zh": v["zh"]}
        for n, v in assignment.items()
    }
    rgroup_json = json.dumps(rgroup_def, ensure_ascii=False)
    all_instructions = generate_all_instructions(assignment)
    records = []
    for instr_info in all_instructions:
        records.append({
            "original_esmiles":  esmiles,
            "original_image":    original_image,
            "instruction":       instr_info["instruction"],
            "instruction_style": instr_info["instruction_style"],
            "rgroup_assignment": rgroup_json,
            "output_smiles":     new_smiles,
            "output_image":      new_image,
        })
    return records


def save_records(records, output_dir):
    images_dir    = os.path.join(output_dir, "output_images")
    orig_imgs_dir = os.path.join(output_dir, "original_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(orig_imgs_dir, exist_ok=True)

    print("保存图像...")
    text_records = []
    for idx, r in enumerate(records):
        if idx % 5000 == 0:
            print(f"  图像进度: {idx}/{len(records)}")

        img_filename = f"{idx:06d}.png"

        out_img_path = os.path.join(images_dir, img_filename)
        if r["output_image"] is not None:
            try:
                r["output_image"].save(out_img_path)
            except Exception:
                out_img_path = ""

        orig_img_path = os.path.join(orig_imgs_dir, img_filename)
        if r["original_image"] is not None:
            try:
                if isinstance(r["original_image"], Image.Image):
                    r["original_image"].save(orig_img_path)
                elif isinstance(r["original_image"], dict) and "bytes" in r["original_image"]:
                    with open(orig_img_path, "wb") as f:
                        f.write(r["original_image"]["bytes"])
                else:
                    orig_img_path = ""
            except Exception:
                orig_img_path = ""

        text_records.append({
            "original_esmiles":    r["original_esmiles"],
            "instruction":         r["instruction"],
            "instruction_style":   r["instruction_style"],
            "rgroup_assignment":   r["rgroup_assignment"],
            "output_smiles":       r["output_smiles"],
            "output_image_path":   out_img_path,
            "original_image_path": orig_img_path,
        })

    print("保存文本数据集（分批）...")
    for start in range(0, len(text_records), BATCH_SIZE):
        batch = text_records[start: start + BATCH_SIZE]
        Dataset.from_list(batch).save_to_disk(
            os.path.join(output_dir, f"batch_{start:06d}")
        )
        print(f"  batch {start}~{start+len(batch)-1} 已保存")

    preview_path = os.path.join(output_dir, "preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if "image" not in k}
                   for r in text_records[:20]],
                  f, indent=2, ensure_ascii=False)
    print(f"预览: {preview_path}")


def main():
    print("加载数据集...")
    ds = load_from_disk(INPUT_DIR)

    print("过滤干净样本...")
    clean_samples = [s for s in ds if is_clean_sample(s["SMILES"])]
    print(f"干净样本（R-group ≤ {MAX_RGROUPS}）: {len(clean_samples)} 条\n")

    records = []
    skipped = {"replace": 0, "validate": 0, "render": 0}

    for i, sample in enumerate(clean_samples):
        if i % 200 == 0:
            print(f"  进度 {i}/{len(clean_samples)} | 已生成记录 {len(records)} | "
                  f"跳过 替换{skipped['replace']} 验证{skipped['validate']} 渲染{skipped['render']}")

        esmiles = sample["SMILES"]
        rgroups = parse_rgroups(esmiles)
        used_keys = []
        variant_count = 0

        for attempt in range(NUM_VARIANTS * 6):
            if variant_count >= NUM_VARIANTS:
                break

            assignment = {
                g: random.choice(get_candidate_pool(g, esmiles, rgroups[g]))
                for g in rgroups.keys()
            }
            akey = str(sorted((k, v["key"]) for k, v in assignment.items()))
            if akey in used_keys:
                continue
            used_keys.append(akey)

            new_smiles = replace_rgroups(esmiles, assignment)
            if new_smiles is None:
                skipped["replace"] += 1
                continue

            if strict_validate(new_smiles) is None:
                skipped["validate"] += 1
                continue

            new_image = smiles_to_image(new_smiles)
            if new_image is None:
                skipped["render"] += 1
                continue

            variant_records = generate_records_for_variant(
                esmiles, sample["image"], assignment, new_smiles, new_image
            )
            records.extend(variant_records)
            variant_count += 1

    print(f"\n生成完成: 有效记录 {len(records)} 条 "
          f"（化学变体 {len(records)//5} 个 × 5种style）")
    print(f"跳过: 替换{skipped['replace']} 验证{skipped['validate']} 渲染{skipped['render']}")

    from collections import Counter
    style_dist = Counter(r["instruction_style"] for r in records)
    print("\nInstruction style 分布:")
    for s, c in style_dist.most_common():
        print(f"  {s}: {c} 条 ({c/len(records)*100:.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_records(records, OUTPUT_DIR)
    print(f"\n全部完成！数据集位于: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()