# 文件位置: Moleditor/code/25_build_new_edit_dataset.py
#
# 作用: 构建新版统一编辑数据集，用于 Easy + Hard 两个 Level 的 VQA
#
# 设计：
#   - 取代基：全部使用复杂取代基（27种，含环或多原子），与旧Hard level一致
#   - instruction字段：同时保存两种
#       instruction            → 直接版 "Replace R1 with 4-chlorophenyl."  (Easy VQA用)
#       descriptive_instruction→ 描述性版 "Replace R1 with a halogenated phenyl ring." (Hard VQA用)
#   - similar_group_idx 字段：记录该取代基属于哪个similar group（-1表示不属于任何group）
#     方便VQA构建时找同组取代基做同骨架干扰项
#   - 每个骨架生成 NUM_VARIANTS 个变体（默认4个，保证VQA有足够候选）
#
# 输出:
#   new/edit_dataset/
#     batch_000000/   ... (HuggingFace Dataset)
#     output_images/      (RDKit渲染，300x300)
#     original_images/    (原始Markush图)
#     preview.json
#
# 用法:
#   python 25_build_new_edit_dataset.py
#
# 注意: 本地路径已配置，集群路径注释在下方

import json
import os
import random
import re
from collections import Counter

from datasets import Dataset, load_from_disk
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# ── 路径配置 ──────────────────────────────────────────────
# 本地路径
INPUT_DIR  = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/molparser_markush"
OUTPUT_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/edit_dataset"

# 集群路径（上传集群时切换）
# INPUT_DIR  = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/molparser_markush"
# OUTPUT_DIR = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/edit_dataset"

NUM_VARIANTS = 4    # 每个骨架生成几个变体（至少4个才能凑VQA选项）
RANDOM_SEED  = 42
BATCH_SIZE   = 5000
# ──────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════
# Similar Group 定义（与15号脚本完全一致）
# 每个group内的取代基视觉上相似，用于Hard VQA同骨架干扰项
# ══════════════════════════════════════════════════════════

SIMILAR_GROUPS = [
    # Group 0: 单取代苯（卤素/甲基/甲氧基）
    [
        {"key": "phenyl",          "smiles": "c1ccccc1",       "en": "phenyl",          "zh": "苯基"},
        {"key": "4-fluorophenyl",  "smiles": "c1ccc(F)cc1",    "en": "4-fluorophenyl",  "zh": "4-氟苯基"},
        {"key": "4-chlorophenyl",  "smiles": "c1ccc(Cl)cc1",   "en": "4-chlorophenyl",  "zh": "4-氯苯基"},
        {"key": "4-methylphenyl",  "smiles": "c1ccc(C)cc1",    "en": "4-methylphenyl",  "zh": "4-甲基苯基"},
        {"key": "4-methoxyphenyl", "smiles": "c1ccc(OC)cc1",   "en": "4-methoxyphenyl", "zh": "4-甲氧基苯基"},
    ],
    # Group 1: 间/多取代苯
    [
        {"key": "3-chlorophenyl",           "smiles": "c1cccc(Cl)c1",        "en": "3-chlorophenyl",           "zh": "3-氯苯基"},
        {"key": "3,4-dichlorophenyl",        "smiles": "c1ccc(Cl)c(Cl)c1",   "en": "3,4-dichlorophenyl",       "zh": "3,4-二氯苯基"},
        {"key": "4-trifluoromethylphenyl",   "smiles": "c1ccc(C(F)(F)F)cc1", "en": "4-trifluoromethylphenyl",  "zh": "4-三氟甲基苯基"},
    ],
    # Group 2: 吡啶系列（位置异构体）
    [
        {"key": "2-pyridyl", "smiles": "c1ccncc1", "en": "2-pyridyl", "zh": "2-吡啶基"},
        {"key": "3-pyridyl", "smiles": "c1cnccc1", "en": "3-pyridyl", "zh": "3-吡啶基"},
        {"key": "4-pyridyl", "smiles": "c1ccncc1", "en": "4-pyridyl", "zh": "4-吡啶基"},
    ],
    # Group 3: 脂环系列
    [
        {"key": "cyclohexyl",  "smiles": "C1CCCCC1", "en": "cyclohexyl",  "zh": "环己基"},
        {"key": "cyclopentyl", "smiles": "C1CCCC1",  "en": "cyclopentyl", "zh": "环戊基"},
        {"key": "cyclopropyl", "smiles": "C1CC1",    "en": "cyclopropyl", "zh": "环丙基"},
    ],
    # Group 4: 苄基系列
    [
        {"key": "benzyl",         "smiles": "Cc1ccccc1",     "en": "benzyl",         "zh": "苄基"},
        {"key": "4-fluorobenzyl", "smiles": "Cc1ccc(F)cc1",  "en": "4-fluorobenzyl", "zh": "4-氟苄基"},
        {"key": "4-chlorobenzyl", "smiles": "Cc1ccc(Cl)cc1", "en": "4-chlorobenzyl", "zh": "4-氯苄基"},
    ],
    # Group 5: 苯基羰基/磺酰基系列
    [
        {"key": "benzoyl",        "smiles": "C(=O)c1ccccc1",     "en": "benzoyl",        "zh": "苯甲酰基"},
        {"key": "phenylacetyl",   "smiles": "C(=O)Cc1ccccc1",    "en": "phenylacetyl",   "zh": "苯乙酰基"},
        {"key": "phenylsulfonyl", "smiles": "S(=O)(=O)c1ccccc1", "en": "phenylsulfonyl", "zh": "苯磺酰基"},
    ],
    # Group 6: 磺酰基系列
    [
        {"key": "methylsulfonyl",        "smiles": "S(=O)(=O)C",            "en": "methylsulfonyl",        "zh": "甲磺酰基"},
        {"key": "phenylsulfonyl",        "smiles": "S(=O)(=O)c1ccccc1",    "en": "phenylsulfonyl",        "zh": "苯磺酰基"},
        {"key": "4-methylsulfonylphenyl","smiles": "c1ccc(S(=O)(=O)C)cc1", "en": "4-methylsulfonylphenyl","zh": "4-甲磺酰基苯基"},
    ],
]

# 描述性 instruction 模板（每个similar group一套）
DESCRIPTIVE_TEMPLATES = {
    0: [
        "Replace {rgroup} with a monosubstituted phenyl ring.",
        "Substitute {rgroup} with a phenyl group bearing a single substituent.",
        "Set {rgroup} to a para-substituted or unsubstituted benzene ring.",
    ],
    1: [
        "Replace {rgroup} with a polyhalogenated or strongly electron-withdrawing phenyl group.",
        "Substitute {rgroup} with a phenyl ring bearing multiple or bulky substituents.",
        "Set {rgroup} to a phenyl group with strong electron-withdrawing character.",
    ],
    2: [
        "Replace {rgroup} with a nitrogen-containing six-membered aromatic ring.",
        "Substitute {rgroup} with a pyridyl group.",
        "Set {rgroup} to a heteroaromatic ring with one ring nitrogen.",
    ],
    3: [
        "Replace {rgroup} with a saturated aliphatic ring.",
        "Substitute {rgroup} with a cycloalkyl group.",
        "Set {rgroup} to a non-aromatic carbocyclic substituent.",
    ],
    4: [
        "Replace {rgroup} with a benzyl-type group (phenyl connected via a methylene linker).",
        "Substitute {rgroup} with a CH2-phenyl or substituted CH2-phenyl group.",
        "Set {rgroup} to a benzylic substituent.",
    ],
    5: [
        "Replace {rgroup} with a phenyl-containing carbonyl or sulfonyl group.",
        "Substitute {rgroup} with a benzoyl, phenylacetyl, or phenylsulfonyl group.",
        "Set {rgroup} to a phenyl-bearing acyl or sulfonyl substituent.",
    ],
    6: [
        "Replace {rgroup} with a sulfonyl-containing group.",
        "Substitute {rgroup} with an alkyl or aryl sulfonyl substituent.",
        "Set {rgroup} to a methylsulfonyl or phenylsulfonyl group.",
    ],
    -1: [
        "Replace {rgroup} with an appropriate substituent.",
        "Substitute {rgroup} with a suitable chemical group.",
        "Set {rgroup} to a compatible substituent for this position.",
    ],
}

# 建立 key → (group_idx, substituent_dict) 的查找表，加速运行
KEY_TO_GROUP: dict = {}
for gidx, group in enumerate(SIMILAR_GROUPS):
    for sub in group:
        KEY_TO_GROUP[sub["key"]] = gidx

# 所有取代基的扁平列表（去重，key唯一）
ALL_SUBSTITUENTS: list = []
seen_keys = set()
for group in SIMILAR_GROUPS:
    for sub in group:
        if sub["key"] not in seen_keys:
            ALL_SUBSTITUENTS.append(sub)
            seen_keys.add(sub["key"])

# N / O 连接位点的白名单（从 ALL_SUBSTITUENTS 中筛选）
N_ALLOWED_KEYS = {
    "phenyl", "4-fluorophenyl", "4-chlorophenyl", "4-methylphenyl",
    "benzyl", "4-fluorobenzyl", "4-chlorobenzyl",
    "cyclohexyl", "cyclopentyl",
    "2-pyridyl", "3-pyridyl", "4-pyridyl",
    "benzoyl", "phenylacetyl", "methylsulfonyl", "phenylsulfonyl",
}
O_ALLOWED_KEYS = {
    "phenyl", "4-fluorophenyl", "4-chlorophenyl", "4-methylphenyl", "4-methoxyphenyl",
    "benzyl", "4-fluorobenzyl", "4-chlorobenzyl",
    "cyclohexyl",
    "benzoyl",
}

N_SUBSTITUENTS = [s for s in ALL_SUBSTITUENTS if s["key"] in N_ALLOWED_KEYS]
O_SUBSTITUENTS = [s for s in ALL_SUBSTITUENTS if s["key"] in O_ALLOWED_KEYS]


# ══════════════════════════════════════════════════════════
# 化学工具函数（与12/15号脚本一致）
# ══════════════════════════════════════════════════════════

def is_real_rgroup(name: str) -> bool:
    if re.match(r"^R\[?\d+\]?$", name):
        return True
    if name == "R":
        return True
    if re.match(r"^[XYZWL]$", name):
        return True
    return False


def parse_rgroups(esmiles: str) -> dict:
    """返回 {group_name: atom_idx}"""
    matches = re.findall(r"<a>(\d+):([^<]+)</a>", esmiles)
    return {gname: int(idx) for idx, gname in matches if is_real_rgroup(gname)}


def is_clean_sample(esmiles: str) -> bool:
    if any(t in esmiles for t in ["<r>", "<c>", "<dum>"]):
        return False
    return len(parse_rgroups(esmiles)) > 0


def get_attachment_atom(esmiles: str, wildcard_idx: int) -> str:
    try:
        core = esmiles.split("<sep>")[0]
        mol  = Chem.MolFromSmiles(core, sanitize=False)
        if mol is None:
            return None
        atom = mol.GetAtomWithIdx(wildcard_idx)
        if atom.GetSymbol() != "*":
            return None
        neighbors = atom.GetNeighbors()
        return neighbors[0].GetSymbol() if neighbors else None
    except Exception:
        return None


def get_candidate_pool(group_name: str, esmiles: str, atom_idx: int) -> list:
    """根据连接位点返回候选取代基池"""
    if group_name in {"X", "Y", "Z"}:
        # 卤素位点：只用含卤素的芳香基，来自Group 0/1
        halogen_keys = {
            "4-fluorophenyl", "4-chlorophenyl", "3-chlorophenyl",
            "3,4-dichlorophenyl", "4-trifluoromethylphenyl",
            "4-fluorobenzyl", "4-chlorobenzyl",
        }
        return [s for s in ALL_SUBSTITUENTS if s["key"] in halogen_keys]
    neighbor = get_attachment_atom(esmiles, atom_idx)
    if neighbor == "N":
        return N_SUBSTITUENTS
    if neighbor == "O":
        return O_SUBSTITUENTS
    return ALL_SUBSTITUENTS


def replace_rgroups(esmiles: str, assignment: dict) -> str:
    """把esmiles里的*替换成取代基，返回SMILES或None"""
    try:
        core = esmiles.split("<sep>")[0]
        mol  = Chem.MolFromSmiles(core, sanitize=False)
        if mol is None:
            return None
        edit = Chem.RWMol(mol)
        rgroups = parse_rgroups(esmiles)

        # 从大到小替换，避免atom index错位
        for gname in sorted(rgroups, key=lambda g: rgroups[g], reverse=True):
            if gname not in assignment:
                continue
            sub_smi = assignment[gname]["smiles"]
            sub_mol = Chem.MolFromSmiles(sub_smi)
            if sub_mol is None:
                return None
            widx = rgroups[gname]
            # 找通配符邻居
            try:
                wc_atom = edit.GetAtomWithIdx(widx)
            except Exception:
                return None
            if wc_atom.GetSymbol() != "*":
                return None
            neighbors = [n.GetIdx() for n in wc_atom.GetNeighbors()]
            if not neighbors:
                return None
            neighbor_idx = neighbors[0]

            # 合并mol，连接取代基
            combined   = Chem.CombineMols(edit.GetMol(), sub_mol)
            edit2      = Chem.RWMol(combined)
            offset     = edit.GetNumAtoms()
            # 取代基的连接点（第一个重原子）
            attach_idx = offset  # sub_mol第一个原子
            edit2.AddBond(neighbor_idx, attach_idx, Chem.BondType.SINGLE)
            edit2.RemoveAtom(widx if widx < offset else widx)
            edit = edit2

        smi = Chem.MolToSmiles(Chem.RemoveHs(edit.GetMol()))
        return smi
    except Exception:
        return None


def strict_validate(smiles: str):
    """严格验证：RDKit能正常sanitize"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def smiles_to_image(smiles: str, size=(300, 300)):
    try:
        mol = strict_validate(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# Instruction 生成
# ══════════════════════════════════════════════════════════

def make_direct_instruction(assignment: dict) -> str:
    """直接版：Replace R1 with 4-chlorophenyl, and R2 with cyclohexyl."""
    parts = [f"{gname} with {info['en']}" for gname, info in assignment.items()]
    return "Replace " + ", and ".join(parts) + "."


def make_descriptive_instruction(assignment: dict) -> str:
    """
    描述性版：不给出取代基名称，只描述结构特征。
    多个R基时每个R基单独描述，合并成一句。
    """
    parts = []
    for gname, info in assignment.items():
        gidx = KEY_TO_GROUP.get(info["key"], -1)
        templates = DESCRIPTIVE_TEMPLATES.get(gidx, DESCRIPTIVE_TEMPLATES[-1])
        # 从模板中随机选一个，并填入R基名称
        template = random.choice(templates)
        parts.append(template.format(rgroup=gname))

    if len(parts) == 1:
        return parts[0]
    # 多R基：把第一个取末尾句号，后面的全部连起来
    combined = parts[0].rstrip(".")
    for p in parts[1:]:
        combined += "; " + p[0].lower() + p[1:]
    return combined


# ══════════════════════════════════════════════════════════
# 记录生成
# ══════════════════════════════════════════════════════════

def generate_record(esmiles, original_image, assignment, new_smiles, new_image) -> dict:
    """
    生成单条记录，包含：
      - instruction            (直接版)
      - descriptive_instruction(描述性版)
      - similar_group_idx      (首个R基对应的similar group，-1表示不在任何group)
      - rgroup_assignment      (JSON字符串，含key/smiles/en/zh/group_idx)
    """
    rgroup_def = {}
    group_indices = []
    for gname, info in assignment.items():
        gidx = KEY_TO_GROUP.get(info["key"], -1)
        group_indices.append(gidx)
        rgroup_def[gname] = {
            "substituent_key":    info["key"],
            "substituent_smiles": info["smiles"],
            "substituent_en":     info["en"],
            "substituent_zh":     info["zh"],
            "similar_group_idx":  gidx,
        }

    # similar_group_idx：如果所有R基同组就记录该组，否则记-1
    unique_groups = set(g for g in group_indices if g >= 0)
    overall_group = list(unique_groups)[0] if len(unique_groups) == 1 else -1

    return {
        "original_esmiles":          esmiles,
        "original_image":            original_image,
        "instruction":               make_direct_instruction(assignment),
        "descriptive_instruction":   make_descriptive_instruction(assignment),
        "similar_group_idx":         overall_group,
        "rgroup_assignment":         json.dumps(rgroup_def, ensure_ascii=False),
        "output_smiles":             new_smiles,
        "output_image":              new_image,
    }


# ══════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════

def save_records(records: list, output_dir: str):
    images_dir    = os.path.join(output_dir, "output_images")
    orig_imgs_dir = os.path.join(output_dir, "original_images")
    os.makedirs(images_dir,    exist_ok=True)
    os.makedirs(orig_imgs_dir, exist_ok=True)

    print("保存图像...")
    text_records = []
    for idx, r in enumerate(records):
        if idx % 2000 == 0:
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
            "original_esmiles":          r["original_esmiles"],
            "instruction":               r["instruction"],
            "descriptive_instruction":   r["descriptive_instruction"],
            "similar_group_idx":         r["similar_group_idx"],
            "rgroup_assignment":         r["rgroup_assignment"],
            "output_smiles":             r["output_smiles"],
            "output_image_path":         out_img_path,
            "original_image_path":       orig_img_path,
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
        json.dump(
            [{k: v for k, v in r.items() if "image" not in k}
             for r in text_records[:20]],
            f, indent=2, ensure_ascii=False
        )
    print(f"预览已保存: {preview_path}")


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("新版编辑数据集构建（Easy + Hard 共用）")
    print(f"取代基总数: {len(ALL_SUBSTITUENTS)} 种（全部含环或多原子）")
    print(f"Similar groups: {len(SIMILAR_GROUPS)} 个")
    print(f"每骨架变体数: {NUM_VARIANTS}")
    print("输出字段: instruction + descriptive_instruction + similar_group_idx")
    print("=" * 60)

    print(f"\n加载原始Markush数据集: {INPUT_DIR}")
    ds = load_from_disk(INPUT_DIR)
    print(f"原始样本数: {len(ds)}")

    print("\n过滤干净样本（无<r>/<c>/<dum>，且有R基）...")
    clean_samples = [s for s in ds if is_clean_sample(s["SMILES"])]
    print(f"干净样本: {len(clean_samples)} 条\n")

    records = []
    skipped = {"replace": 0, "validate": 0, "render": 0, "pool_empty": 0}

    for i, sample in enumerate(clean_samples):
        if i % 500 == 0:
            print(f"  进度 {i}/{len(clean_samples)} | 已生成 {len(records)} | "
                  f"跳过 替换{skipped['replace']} 验证{skipped['validate']} "
                  f"渲染{skipped['render']} 池空{skipped['pool_empty']}")

        esmiles  = sample["SMILES"]
        rgroups  = parse_rgroups(esmiles)
        used_keys = []

        for _ in range(NUM_VARIANTS * 8):
            if len(used_keys) >= NUM_VARIANTS:
                break

            # 为每个R基独立采样
            assignment = {}
            valid_pool  = True
            for gname, aidx in rgroups.items():
                pool = get_candidate_pool(gname, esmiles, aidx)
                if not pool:
                    skipped["pool_empty"] += 1
                    valid_pool = False
                    break
                assignment[gname] = random.choice(pool)

            if not valid_pool:
                continue

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

            records.append(generate_record(
                esmiles, sample["image"], assignment, new_smiles, new_image
            ))

    print(f"\n生成完成: 有效 {len(records)} 条")
    print(f"跳过: 替换失败{skipped['replace']} | 验证失败{skipped['validate']} | "
          f"渲染失败{skipped['render']} | 池空{skipped['pool_empty']}")

    # 统计取代基分布
    sub_counter   = Counter()
    group_counter = Counter()
    for r in records:
        assignment = json.loads(r["rgroup_assignment"])
        for info in assignment.values():
            sub_counter[info["substituent_en"]] += 1
            group_counter[info["similar_group_idx"]] += 1

    print(f"\n【取代基分布 Top 15】")
    for sub, cnt in sub_counter.most_common(15):
        print(f"  {sub:<40} {cnt:>6} 次")

    print(f"\n【Similar Group 分布】")
    for gidx in sorted(group_counter):
        cnt = group_counter[gidx]
        label = f"Group {gidx}" if gidx >= 0 else "No group"
        print(f"  {label}: {cnt} 次 ({cnt/sum(group_counter.values())*100:.1f}%)")

    # 描述性instruction示例
    print(f"\n【Instruction 示例（前5条）】")
    for r in records[:5]:
        assign = json.loads(r["rgroup_assignment"])
        print(f"  直接版:   {r['instruction']}")
        print(f"  描述性:   {r['descriptive_instruction']}")
        print(f"  取代基:   {[(k, v['substituent_en']) for k, v in assign.items()]}")
        print(f"  group_idx:{r['similar_group_idx']}")
        print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_records(records, OUTPUT_DIR)

    print(f"\n✓ 数据集已保存: {OUTPUT_DIR}")
    print(f"✓ 总条数: {len(records)}")
    print(f"\n下一步: 运行 26_build_new_vqa.py 构建 Easy/Hard VQA")


if __name__ == "__main__":
    main()
