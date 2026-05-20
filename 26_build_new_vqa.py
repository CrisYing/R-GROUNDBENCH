# 文件位置: Moleditor/code/26_build_new_vqa.py
#
# 作用: 基于 25_build_new_edit_dataset.py 的输出 + chembl_reverse_markush.jsonl
#       构建新版6个sub-benchmark（VLM track用，图片输入+图片选项）
#
# ── Basic（R基替换题）──────────────────────────────────────
#   Easy Basic   : 跨骨架干扰项 + 直接instruction
#                  ~80%正常4选1；~20%含D=None of the above：
#                    ~5% TYPE B（R基位点不存在）→ 正确答案D
#                    ~10% TYPE C（Subtle Wrong Answer）→ 正确答案D
#                    ~5% valid instruction + D占位 → 正确答案A/B/C
#
#   Medium Basic : 同骨架干扰项 + 直接instruction
#                  ~80%正常4选1；~20%含D=None of the above（同Easy Basic）
#
#   Hard Basic   : 同骨架干扰项（R基相似度最高） + 描述性instruction
#                  无None of Above题（100%正常题）
#                  干扰项选法：按R基ECFP4 Tanimoto相似度最高的3个
#
# ── Advanced（性质推理题）─────────────────────────────────
#   Easy Advanced  : 跨骨架/同靶点干扰，直接性提问，来自ChEMBL
#   Medium Advanced: 同骨架干扰，直接性提问，来自ChEMBL SAR
#   Hard Advanced  : 同骨架干扰，描述性提问，来自ChEMBL SAR
#
# ── 改动说明（v7）─────────────────────────────────────────
#   1. 新增 Medium split（同骨架+直接instruction+NoA）
#   2. Hard Basic去掉None of Above，100%正常题
#   3. Hard Basic干扰项改为R基ECFP4 Tanimoto相似度最高的3个
#   4. Medium/Hard Advanced的same_scaffold逻辑分开处理
#
# 用法: python 26_build_new_vqa.py
# ─────────────────────────────────────────────────────────────────────────────

import json, os, random, re as _re
from collections import defaultdict, Counter
from datasets import Dataset, load_from_disk
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

# ── 路径 ──────────────────────────────────────────────────
EDIT_DATASET_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/edit_dataset"
CHEMBL_JSONL     = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/chembl_reverse_markush/chembl_reverse_markush.jsonl"
SAR_JSONL        = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/chembl_sar_scaffold/chembl_sar_scaffold.jsonl"
VQA_DIR          = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/vqa_dataset"
# 集群:
# EDIT_DATASET_DIR = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/edit_dataset"
# CHEMBL_JSONL     = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/chembl_reverse_markush/chembl_reverse_markush.jsonl"
# SAR_JSONL        = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/chembl_sar_scaffold/chembl_sar_scaffold.jsonl"
# VQA_DIR          = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/vqa_dataset"

RANDOM_SEED       = 42
TARGET_BASIC      = 940
TARGET_ADVANCED   = 634
MIN_SAME_SCAFFOLD = 4

random.seed(RANDOM_SEED)

# ── None of Above 比例配置 ─────────────────────────────────
NOVA_TYPE_B_RATE = 0.05   # 5%:  TYPE B R基位点不存在 → 正确答案D
NOVA_TYPE_C_RATE = 0.11   # ~11%: TYPE C Subtle Wrong Answer → 正确答案D
NOVA_VALID_RATE  = 0.05   # 5%:  valid instruction + D占位 → 正确答案A/B/C
# 合计约20%为None of Above题（仅Easy和Medium Basic）

# ══════════════════════════════════════════════════════════
# RDKit 性质配置
# ══════════════════════════════════════════════════════════
PROPERTY_CONFIGS = {
    "mol_weight": {
        "func": lambda m: round(Descriptors.MolWt(m), 2),
        "questions_easy": [
            "Which substitution results in the highest molecular weight?",
            "Which R-group choice gives the heaviest molecule?",
        ],
        "questions_hard": [
            "Which substitution gives the most massive compound in terms of atomic composition?",
            "Which R-group choice produces a molecule that would sediment fastest in a centrifuge due to its size?",
        ],
    },
    "logP": {
        "func": lambda m: round(Descriptors.MolLogP(m), 3),
        "questions_easy": [
            "Which substitution gives the most lipophilic molecule?",
            "Which R-group choice results in the highest logP value?",
        ],
        "questions_hard": [
            "Which substitution produces the compound most likely to accumulate in lipid membranes?",
            "Which R-group choice gives the compound with the best passive membrane permeability?",
        ],
    },
    "tpsa": {
        "func": lambda m: round(Descriptors.TPSA(m), 2),
        "questions_easy": [
            "Which substitution results in the highest polar surface area?",
            "Which R-group choice gives the largest TPSA?",
        ],
        "questions_hard": [
            "Which substitution gives the compound most likely to have poor oral absorption due to polarity?",
            "Which R-group choice produces a molecule with the greatest hydrogen-bonding surface exposed to solvent?",
        ],
    },
    "hbd": {
        "func": lambda m: Descriptors.NumHDonors(m),
        "questions_easy": [
            "Which substitution gives the most hydrogen bond donors?",
            "Which R-group choice maximizes hydrogen bond donor count?",
        ],
        "questions_hard": [
            "Which substitution produces the compound most capable of donating protons in aqueous solution?",
            "Which R-group choice gives the molecule that would interact most strongly with acceptor residues in a binding pocket?",
        ],
    },
    "hba": {
        "func": lambda m: Descriptors.NumHAcceptors(m),
        "questions_easy": [
            "Which substitution gives the most hydrogen bond acceptors?",
            "Which R-group choice maximizes hydrogen bond acceptor count?",
        ],
        "questions_hard": [
            "Which substitution gives the compound with the most lone pairs available for hydrogen bonding?",
            "Which R-group choice produces the molecule most likely to engage multiple donor residues in a receptor?",
        ],
    },
    "rot_bonds": {
        "func": lambda m: Descriptors.NumRotatableBonds(m),
        "questions_easy": [
            "Which substitution results in the most rotatable bonds?",
            "Which R-group choice gives the most flexible molecule?",
        ],
        "questions_hard": [
            "Which substitution produces the compound with the highest conformational entropy in solution?",
            "Which R-group choice gives the molecule most likely to suffer entropic penalty upon binding?",
        ],
    },
}
PROPERTY_NAMES = list(PROPERTY_CONFIGS.keys())

def compute_rdkit_prop(smiles, prop):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return PROPERTY_CONFIGS[prop]["func"](mol)
    except: return None

# ══════════════════════════════════════════════════════════
# 生物活性/Indication 问题模板
# ══════════════════════════════════════════════════════════

# Easy/Medium Advanced: 直接性提问
BIO_QUESTIONS_EASY_GENERIC = [
    "Which substitution gives the most potent molecule against this target?",
    "Which R-group choice results in the highest binding affinity (pIC50)?",
    "Which substitution produces the strongest inhibitory activity?",
    "Which R-group selection leads to the most active compound?",
    "Which substitution results in the compound with the best potency?",
]
BIO_QUESTIONS_EASY_WITH_TARGET = [
    "Which substitution gives the most potent {target} inhibitor?",
    "Which R-group choice produces the strongest {target} binding?",
    "Which substitution results in the highest activity against {target}?",
]

# Hard Advanced: 描述性提问（不直接说pIC50）
BIO_QUESTIONS_HARD_GENERIC = [
    "Which substitution produces the compound most likely to exhibit superior target engagement in biochemical assays?",
    "Which R-group choice gives the molecule with the best prospects for advancing in lead optimization?",
    "Which substitution produces the compound most likely to show efficacy at the lowest concentration?",
    "Which R-group selection would a medicinal chemist prioritize for in vivo potency studies?",
    "Which substitution yields the compound with the tightest binding to its biological target?",
]
BIO_QUESTIONS_HARD_WITH_TARGET = [
    "Which substitution produces the compound most likely to show clinical relevance in {target} inhibition studies?",
    "Which R-group choice gives the molecule a medicinal chemist would advance for {target}-related therapeutic development?",
    "Which substitution is most likely to result in a compound worthy of further profiling against {target}?",
]

# Indication: Easy/Medium（直接）—— R基版本
INDICATION_QUESTIONS_EASY_RGROUP = [
    "Which R-group, when substituted into the scaffold, gives a compound indicated for the treatment of {indication}?",
    "Which R-group choice produces a molecule approved for {indication}?",
    "Which R-group substitution yields the compound used to treat {indication}?",
    "Which R-group results in a known therapeutic agent for {indication}?",
]
MECHANISM_QUESTIONS_EASY_RGROUP = [
    "Which R-group, when substituted into the scaffold, gives a compound that acts as a {mechanism}?",
    "Which R-group choice results in a molecule with the mechanism: {mechanism}?",
    "Which R-group substitution produces a {mechanism}?",
]

# Indication: Hard（描述性，不直接说indication名称）—— R基版本
INDICATION_QUESTIONS_HARD_RGROUP = [
    "Which R-group substitution produces a compound that has received regulatory approval for managing a condition characterized by {indication_desc}?",
    "Which R-group choice gives a molecule that clinicians would consider for patients suffering from {indication_desc}?",
    "Which R-group substitution yields a compound with established clinical use in conditions involving {indication_desc}?",
]
MECHANISM_QUESTIONS_HARD_RGROUP = [
    "Which R-group substitution gives a compound whose primary pharmacological action involves {mechanism_desc}?",
    "Which R-group choice results in a molecule whose therapeutic effect is mediated through {mechanism_desc}?",
    "Which R-group substitution produces a compound that exerts its effect by {mechanism_desc}?",
]

# 将indication名称转换为描述性表达
INDICATION_TO_DESC = {
    "pain":                                  "persistent or acute nociceptive signaling",
    "cardiovascular disease":                "impaired cardiac or vascular function",
    "neoplasm":                              "uncontrolled cellular proliferation",
    "hiv infection":                         "retroviral immunodeficiency",
    "hypertension":                          "chronically elevated arterial pressure",
    "infection":                             "pathogenic microbial invasion",
    "epilepsy":                              "recurrent seizure episodes",
    "rheumatic disease":                     "autoimmune joint inflammation",
    "heart failure":                         "inadequate cardiac output",
    "chronic obstructive pulmonary disease": "progressive airflow limitation in the lungs",
    "non-small cell lung carcinoma":         "aggressive pulmonary malignancy of epithelial origin",
    "rheumatoid arthritis":                  "chronic inflammatory polyarthritis",
    "diabetes mellitus":                     "dysregulated glucose homeostasis",
    "multiple myeloma":                      "malignant plasma cell proliferation in bone marrow",
    "anxiety":                               "excessive and persistent psychological distress",
    "glioblastoma multiforme":               "highly aggressive primary brain malignancy",
}

def indication_to_descriptive(ind_term):
    key = ind_term.strip().lower()
    if key in INDICATION_TO_DESC:
        return INDICATION_TO_DESC[key]
    return f"conditions related to {ind_term}"

def mechanism_to_descriptive(mech_term):
    return f"modulation of {mech_term.lower()} pathways"


# ══════════════════════════════════════════════════════════
# R基 Tanimoto 相似度计算（Hard Basic 负样本选择用）
# ══════════════════════════════════════════════════════════

def _get_rgroup_smiles(record: dict) -> str:
    """从edit record中获取输出分子SMILES（用于相似度计算）"""
    return record.get("output_smiles", "")

def _smiles_to_fp(smiles: str):
    """将SMILES转为ECFP4 fingerprint，失败返回None"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except:
        return None

def _tanimoto(fp1, fp2) -> float:
    """计算两个fingerprint的Tanimoto相似度"""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def select_hard_distractors(correct: dict, candidates: list, n: int = 3) -> list:
    """
    从candidates中按输出分子Tanimoto相似度选最高的n个作为干扰项。
    如果candidates不足n个或相似度计算失败，退化为随机选取。
    """
    if len(candidates) <= n:
        return candidates[:n] if len(candidates) == n else None

    correct_smi = _get_rgroup_smiles(correct)
    correct_fp  = _smiles_to_fp(correct_smi)

    if correct_fp is None:
        # 计算失败，随机选
        return random.sample(candidates, n)

    scored = []
    for c in candidates:
        fp = _smiles_to_fp(_get_rgroup_smiles(c))
        sim = _tanimoto(correct_fp, fp)
        scored.append((sim, c))

    # 按相似度降序，取前n个（相似度最高=最难分辨）
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:n]]


# ══════════════════════════════════════════════════════════
# TYPE C: Subtle Wrong Answer
# ══════════════════════════════════════════════════════════

def make_subtle_wrong_answer_instruction(record: dict, instr_style: str = "direct") -> tuple | None:
    esmiles = record.get("original_esmiles", "")
    matches = _re.findall(r"<a>(\d+):([^<]+)</a>", esmiles)
    rgroup_map = {gname: int(idx) for idx, gname in matches
                  if _re.match(r"^R\[?\d+\]?$|^R$|^[XYZWL]$", gname)}
    if not rgroup_map:
        return None

    if instr_style == "direct":
        instr = record.get("instruction", "")
    else:
        instr = record.get("descriptive_instruction", record.get("instruction", ""))

    if not instr:
        return None

    return instr, "subtle_wrong_answer"


# ══════════════════════════════════════════════════════════
# TYPE B: R基位点不存在
# ══════════════════════════════════════════════════════════

_RGROUP_REPLACE_TARGETS = [
    "a fluorine atom",
    "a chlorine atom",
    "a methyl group",
    "a hydroxyl group",
    "a cyano group",
    "a trifluoromethyl group",
    "an amino group",
    "a methoxy group",
    "an ethyl group",
    "a bromine atom",
]

def _parse_rgroup_names(esmiles: str) -> list[str]:
    matches = _re.findall(r"<a>\d+:([^<]+)</a>", esmiles)
    return [g for g in matches if _re.match(r"^R\[?\d+\]?$|^R$|^[XYZWL]$", g)]

def make_nonexistent_rgroup_instruction(record: dict, instr_style: str = "direct") -> tuple | None:
    esmiles = record.get("original_esmiles", "")
    existing_rgroups = _parse_rgroup_names(esmiles)
    if not existing_rgroups:
        return None

    candidate_names = [f"R{i}" for i in range(1, 7)]
    nonexistent = [r for r in candidate_names if r not in existing_rgroups]
    if not nonexistent:
        nonexistent = ["R7"]

    fake_rgroup = random.choice(nonexistent)
    replace_with = random.choice(_RGROUP_REPLACE_TARGETS)

    if instr_style == "direct":
        instr = f"Replace {fake_rgroup} with {replace_with}."
    else:
        instr = (f"Substitute the group at position {fake_rgroup} "
                 f"with {replace_with}.")

    return instr, f"nonexistent_{fake_rgroup}"


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════

def load_all_batches(input_dir):
    records = []
    batch_dirs = sorted([d for d in os.listdir(input_dir)
                         if d.startswith("batch_") and os.path.isdir(os.path.join(input_dir, d))])
    print(f"  找到 {len(batch_dirs)} 个batch")
    for bd in batch_dirs:
        ds = load_from_disk(os.path.join(input_dir, bd))
        records.extend(list(ds))
    print(f"  总记录数: {len(records)}")
    return records

def filter_valid_edit(records):
    valid = [r for r in records
             if os.path.isfile(r.get("output_image_path",""))
             and os.path.isfile(r.get("original_image_path",""))
             and r.get("output_smiles")]
    print(f"  有效edit记录: {len(valid)} / {len(records)}")
    return valid

def load_chembl_records(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"  ⚠️  ChEMBL数据不存在: {jsonl_path}")
        return []
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: continue
    print(f"  ChEMBL记录数: {len(records)}")
    return records

def load_sar_records(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"  ⚠️  SAR数据不存在: {jsonl_path}")
        return [], []
    bio_clusters = []
    ind_clusters = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                if rec.get("data_type") == "bio_activity":
                    bio_clusters.append(rec)
                elif rec.get("data_type") == "indication_mechanism":
                    ind_clusters.append(rec)
            except: continue
    print(f"  SAR bio簇: {len(bio_clusters)}，indication簇: {len(ind_clusters)}")
    return bio_clusters, ind_clusters

def filter_valid_chembl(records):
    valid = [r for r in records
             if os.path.isfile(r.get("markush_image_path",""))
             and os.path.isfile(r.get("output_image_path",""))
             and r.get("markush_smiles") and "*" in r.get("markush_smiles","")]
    print(f"  有效ChEMBL记录: {len(valid)} / {len(records)}")
    return valid

def group_by_scaffold(records, smiles_key="original_esmiles"):
    groups = defaultdict(list)
    for r in records:
        groups[r[smiles_key]].append(r)
    return dict(groups)

def get_activity_value(r):
    best = r.get("best_activity")
    if not best: return None
    try: return float(best.get("pchembl_value", 0) or 0)
    except: return None

def get_target_short(r):
    best = r.get("best_activity", {}) or {}
    t = best.get("target_name", "this target")
    return t[:40] if t else "this target"

def get_target_chembl_id(r):
    best = r.get("best_activity", {}) or {}
    return best.get("target_chembl_id", None)


# ══════════════════════════════════════════════════════════
# 通用record构建工具
# ══════════════════════════════════════════════════════════

def make_edit_vqa(split, esmiles, orig_img, question, instruction,
                  candidates, correct_idx, prop="", prop_values=None, correct_val=None):
    lbl = ["A","B","C","D"][correct_idx]
    return {
        "split": split, "original_esmiles": esmiles, "original_image_path": orig_img,
        "instruction": instruction, "question": question, "property": prop,
        "correct_answer": lbl, "correct_smiles": candidates[correct_idx]["output_smiles"],
        "correct_prop_value": correct_val,
        "prop_values": json.dumps(prop_values or {}),
        "is_none_of_above": False, "nova_answer_is_d": False, "nova_type": "",
        "option_A_image":  candidates[0]["output_image_path"],
        "option_B_image":  candidates[1]["output_image_path"],
        "option_C_image":  candidates[2]["output_image_path"],
        "option_D_image":  candidates[3]["output_image_path"],
        "option_A_smiles": candidates[0]["output_smiles"],
        "option_B_smiles": candidates[1]["output_smiles"],
        "option_C_smiles": candidates[2]["output_smiles"],
        "option_D_smiles": candidates[3]["output_smiles"],
    }

def make_chembl_vqa(split, prop_label, orig_esmiles, orig_img_path,
                    question, instruction, candidates, correct_idx,
                    correct_val, prop_values) -> dict:
    lbl = ["A","B","C","D"][correct_idx]
    return {
        "split": split, "property": prop_label,
        "original_esmiles": orig_esmiles,
        "original_image_path": orig_img_path,
        "instruction": instruction, "question": question,
        "correct_answer": lbl,
        "correct_smiles": candidates[correct_idx].get("full_molecule_smiles",""),
        "correct_prop_value": correct_val,
        "prop_values": json.dumps(prop_values),
        "is_none_of_above": False, "nova_answer_is_d": False, "nova_type": "",
        "option_A_image":  candidates[0].get("output_image_path",""),
        "option_B_image":  candidates[1].get("output_image_path",""),
        "option_C_image":  candidates[2].get("output_image_path",""),
        "option_D_image":  candidates[3].get("output_image_path",""),
        "option_A_smiles": candidates[0].get("full_molecule_smiles",""),
        "option_B_smiles": candidates[1].get("full_molecule_smiles",""),
        "option_C_smiles": candidates[2].get("full_molecule_smiles",""),
        "option_D_smiles": candidates[3].get("full_molecule_smiles",""),
    }

def make_none_of_above_record(
    split, esmiles, orig_img_path, instruction, three_candidates,
    answer_is_d: bool, nova_type: str = "",
    correct_label: str = "D",
    correct_smiles: str = "None of the above"
) -> dict:
    assert len(three_candidates) == 3
    correct_answer = "D" if answer_is_d else correct_label
    real_correct_smiles = "None of the above" if answer_is_d else correct_smiles
    return {
        "split":               split,
        "original_esmiles":    esmiles,
        "original_image_path": orig_img_path,
        "instruction":         instruction,
        "question":            "Which molecule is the correct result of the following edit instruction applied to the Markush structure?",
        "property":            "",
        "correct_answer":      correct_answer,
        "correct_smiles":      real_correct_smiles,
        "correct_prop_value":  None,
        "prop_values":         json.dumps({}),
        "is_none_of_above":    True,
        "nova_answer_is_d":    answer_is_d,
        "nova_type":           nova_type,
        "option_A_image":      three_candidates[0]["output_image_path"],
        "option_B_image":      three_candidates[1]["output_image_path"],
        "option_C_image":      three_candidates[2]["output_image_path"],
        "option_D_image":      "",
        "option_A_smiles":     three_candidates[0]["output_smiles"],
        "option_B_smiles":     three_candidates[1]["output_smiles"],
        "option_C_smiles":     three_candidates[2]["output_smiles"],
        "option_D_smiles":     "None of the above",
    }


# ══════════════════════════════════════════════════════════
# Easy Basic: 跨骨架 + 直接instruction + None of Above
# ══════════════════════════════════════════════════════════
def build_easy_basic(records, groups, target):
    vqa = []
    skipped = {"normal": 0, "type_b": 0, "type_c": 0, "valid_d": 0}
    count   = {"normal": 0, "type_b": 0, "type_c": 0, "valid_d": 0}

    scaffold_list = list(groups.keys())
    random.shuffle(scaffold_list)

    for esmiles in scaffold_list:
        if len(vqa) >= target: break
        variants = groups[esmiles]
        if not variants: continue

        rnd = random.random()

        # ── TYPE B: R基位点不存在（~5%）──
        if rnd < NOVA_TYPE_B_RATE:
            ref = random.choice(variants)
            result = make_nonexistent_rgroup_instruction(ref, instr_style="direct")
            if result is None:
                skipped["type_b"] += 1
                rnd = 1.0
            else:
                instr, _ = result
                other = [r for r in records if r["original_esmiles"] != esmiles]
                if len(other) < 3: skipped["type_b"] += 1; continue
                three = random.sample(other, 3)
                vqa.append(make_none_of_above_record(
                    "easy_basic", esmiles, ref["original_image_path"],
                    instr, three, answer_is_d=True, nova_type="type_b"
                ))
                count["type_b"] += 1
                continue

        # ── TYPE C: Subtle Wrong Answer（~10%）──
        elif rnd < NOVA_TYPE_B_RATE + NOVA_TYPE_C_RATE:
            ref = random.choice(variants)
            result = make_subtle_wrong_answer_instruction(ref, instr_style="direct")
            if result is None:
                skipped["type_c"] += 1
                rnd = 1.0
            else:
                instr, _ = result
                other = [r for r in records if r["original_esmiles"] != esmiles]
                if len(other) < 3: skipped["type_c"] += 1; continue
                three = random.sample(other, 3)
                vqa.append(make_none_of_above_record(
                    "easy_basic", esmiles, ref["original_image_path"],
                    instr, three, answer_is_d=True, nova_type="type_c"
                ))
                count["type_c"] += 1
                continue

        # ── valid instruction + D占位（~5%）──
        elif rnd < NOVA_TYPE_B_RATE + NOVA_TYPE_C_RATE + NOVA_VALID_RATE:
            correct = random.choice(variants)
            other   = [r for r in records if r["original_esmiles"] != esmiles]
            if len(other) < 2: skipped["valid_d"] += 1; continue
            distractors = random.sample(other, 2)
            three_candidates = distractors + [correct]
            random.shuffle(three_candidates)
            correct_idx = three_candidates.index(correct)
            answer_lbl  = ["A", "B", "C"][correct_idx]
            vqa.append(make_none_of_above_record(
                "easy_basic", esmiles, correct["original_image_path"],
                correct["instruction"], three_candidates,
                answer_is_d=False, nova_type="valid_d",
                correct_label=answer_lbl, correct_smiles=correct["output_smiles"]
            ))
            count["valid_d"] += 1
            continue

        # ── 正常题（~80%）──
        correct     = random.choice(variants)
        other       = [r for r in records if r["original_esmiles"] != esmiles]
        if len(other) < 3: skipped["normal"] += 1; continue
        distractors = random.sample(other, 3)
        candidates  = distractors + [correct]
        random.shuffle(candidates)
        vqa.append(make_edit_vqa(
            "easy_basic", esmiles, correct["original_image_path"],
            "Which molecule is the correct result of the following edit instruction applied to the Markush structure?",
            correct["instruction"], candidates, candidates.index(correct)
        ))
        count["normal"] += 1

    total = len(vqa)
    print(f"  Easy Basic: {total} 条")
    print(f"    正常题:    {count['normal']:>4} ({count['normal']/max(total,1)*100:.1f}%)")
    print(f"    TYPE B(R基不存在→D): {count['type_b']:>4} ({count['type_b']/max(total,1)*100:.1f}%)")
    print(f"    TYPE C(subtle wrong→D): {count['type_c']:>4} ({count['type_c']/max(total,1)*100:.1f}%)")
    print(f"    占位D(答案A/B/C): {count['valid_d']:>4} ({count['valid_d']/max(total,1)*100:.1f}%)")
    return vqa


# ══════════════════════════════════════════════════════════
# Medium Basic: 同骨架 + 直接instruction + None of Above（新增）
# 与Easy Basic的区别：干扰项从同骨架其他变体中选（而非跨骨架）
# ══════════════════════════════════════════════════════════
def build_medium_basic(groups, target):
    vqa = []
    skipped = {"normal": 0, "type_b": 0, "type_c": 0, "valid_d": 0}
    count   = {"normal": 0, "type_b": 0, "type_c": 0, "valid_d": 0}

    all_records   = [r for variants in groups.values() for r in variants]
    scaffold_list = list(groups.keys())
    random.shuffle(scaffold_list)

    for esmiles in scaffold_list:
        if len(vqa) >= target: break
        variants = groups[esmiles]
        if len(variants) < MIN_SAME_SCAFFOLD:
            skipped["normal"] += 1; continue

        rnd = random.random()

        # ── TYPE B: R基位点不存在（~5%）──
        if rnd < NOVA_TYPE_B_RATE:
            ref = random.choice(variants)
            result = make_nonexistent_rgroup_instruction(ref, instr_style="direct")
            if result is None:
                skipped["type_b"] += 1
                rnd = 1.0
            else:
                instr, _ = result
                # 干扰项也从同骨架中选（保持Medium的同骨架特性）
                other_same = [r for r in variants if r is not ref]
                if len(other_same) < 3:
                    # 同骨架不够则fallback到跨骨架
                    other_same = [r for r in all_records if r["original_esmiles"] != esmiles]
                if len(other_same) < 3: skipped["type_b"] += 1; continue
                three = random.sample(other_same, 3)
                vqa.append(make_none_of_above_record(
                    "medium_basic", esmiles, ref["original_image_path"],
                    instr, three, answer_is_d=True, nova_type="type_b"
                ))
                count["type_b"] += 1
                continue

        # ── TYPE C: Subtle Wrong Answer（~10%）──
        elif rnd < NOVA_TYPE_B_RATE + NOVA_TYPE_C_RATE:
            ref = random.choice(variants)
            result = make_subtle_wrong_answer_instruction(ref, instr_style="direct")
            if result is None:
                skipped["type_c"] += 1
                rnd = 1.0
            else:
                instr, _ = result
                # 干扰项从同骨架其他变体中选
                other_same = [r for r in variants if r is not ref]
                if len(other_same) < 3:
                    other_same = [r for r in all_records if r["original_esmiles"] != esmiles]
                if len(other_same) < 3: skipped["type_c"] += 1; continue
                three = random.sample(other_same, 3)
                vqa.append(make_none_of_above_record(
                    "medium_basic", esmiles, ref["original_image_path"],
                    instr, three, answer_is_d=True, nova_type="type_c"
                ))
                count["type_c"] += 1
                continue

        # ── valid instruction + D占位（~5%）──
        elif rnd < NOVA_TYPE_B_RATE + NOVA_TYPE_C_RATE + NOVA_VALID_RATE:
            chosen  = random.sample(variants, min(3, len(variants)))
            correct = random.choice(chosen)
            three_candidates = list(chosen)
            random.shuffle(three_candidates)
            correct_idx = three_candidates.index(correct)
            answer_lbl  = ["A", "B", "C"][correct_idx]
            vqa.append(make_none_of_above_record(
                "medium_basic", esmiles, correct["original_image_path"],
                correct["instruction"], three_candidates,
                answer_is_d=False, nova_type="valid_d",
                correct_label=answer_lbl, correct_smiles=correct["output_smiles"]
            ))
            count["valid_d"] += 1
            continue

        # ── 正常题（同骨架4选1，直接instruction）──
        chosen     = random.sample(variants, MIN_SAME_SCAFFOLD)
        correct    = random.choice(chosen)
        candidates = list(chosen)
        random.shuffle(candidates)
        vqa.append(make_edit_vqa(
            "medium_basic", esmiles, correct["original_image_path"],
            "Which molecule is the correct result of the following edit instruction applied to the Markush structure?",
            correct["instruction"],   # ← 直接instruction（区别于Hard的descriptive_instruction）
            candidates, candidates.index(correct)
        ))
        count["normal"] += 1

    total = len(vqa)
    print(f"  Medium Basic: {total} 条")
    print(f"    正常题:    {count['normal']:>4} ({count['normal']/max(total,1)*100:.1f}%)")
    print(f"    TYPE B(R基不存在→D): {count['type_b']:>4} ({count['type_b']/max(total,1)*100:.1f}%)")
    print(f"    TYPE C(subtle wrong→D): {count['type_c']:>4} ({count['type_c']/max(total,1)*100:.1f}%)")
    print(f"    占位D(答案A/B/C): {count['valid_d']:>4} ({count['valid_d']/max(total,1)*100:.1f}%)")
    return vqa


# ══════════════════════════════════════════════════════════
# Hard Basic: 同骨架 + 描述性instruction + 无None of Above（v7改动）
# 干扰项：按R基Tanimoto相似度最高的3个（最难分辨）
# ══════════════════════════════════════════════════════════
def build_hard_basic(groups, target):
    """
    v7改动：
    1. 去掉所有None of Above题（100%正常题）
    2. 干扰项选法：按输出分子ECFP4 Tanimoto相似度最高的3个同骨架变体
       （相似度越高越难区分，比随机选难度更大）
    """
    vqa = []
    skipped = {"not_enough": 0, "no_descriptive": 0}

    scaffold_list = list(groups.keys())
    random.shuffle(scaffold_list)

    for esmiles in scaffold_list:
        if len(vqa) >= target: break
        variants = groups[esmiles]
        if len(variants) < MIN_SAME_SCAFFOLD:
            skipped["not_enough"] += 1
            continue

        # 选correct变体（需要有descriptive_instruction）
        valid_variants = [v for v in variants if v.get("descriptive_instruction")]
        if len(valid_variants) < MIN_SAME_SCAFFOLD:
            skipped["no_descriptive"] += 1
            continue

        correct = random.choice(valid_variants)

        # 干扰项：从同骨架其他变体中按Tanimoto相似度最高选3个
        other_variants = [v for v in valid_variants if v is not correct]
        if len(other_variants) < 3:
            skipped["not_enough"] += 1
            continue

        distractors = select_hard_distractors(correct, other_variants, n=3)
        if distractors is None or len(distractors) < 3:
            skipped["not_enough"] += 1
            continue

        candidates = distractors + [correct]
        random.shuffle(candidates)

        vqa.append(make_edit_vqa(
            "hard_basic", esmiles, correct["original_image_path"],
            "Which molecule is the correct result of the following edit instruction applied to the Markush structure?",
            correct["descriptive_instruction"],  # 描述性instruction
            candidates, candidates.index(correct)
        ))

    total = len(vqa)
    print(f"  Hard Basic: {total} 条（无NoA，R基相似度最高干扰项）")
    print(f"    跳过: 变体不足={skipped['not_enough']} 无descriptive_instr={skipped['no_descriptive']}")
    return vqa


# ══════════════════════════════════════════════════════════
# Advanced Bio: 同靶点干扰项
# Easy/Medium: 直接性问题；Hard: 描述性问题 + 最难干扰
# ══════════════════════════════════════════════════════════
def _build_advanced_bio(bio_clusters, target, split, same_scaffold):
    vqa = []
    use_hard_questions = ("hard" in split)

    if not bio_clusters:
        print(f"  ⚠️  {split} 无SAR bio簇数据")
        return []

    random.shuffle(bio_clusters)
    skipped = {"variants": 0, "tie": 0, "spread": 0, "image": 0}

    for cluster in bio_clusters:
        if len(vqa) >= target: break

        variants = cluster.get("variants", [])
        if len(variants) < 4:
            skipped["variants"] += 1
            continue

        valid_variants = [
            v for v in variants
            if os.path.isfile(v.get("mol_image_path", ""))
            and v.get("full_mol_smiles")
            and v.get("pchembl_value") is not None
        ]
        if len(valid_variants) < 4:
            skipped["image"] += 1
            continue

        best = max(valid_variants, key=lambda x: x["pchembl_value"])
        others = [v for v in valid_variants if v is not best]
        if len(others) < 3:
            skipped["variants"] += 1
            continue

        # Hard: 选pIC50最接近best的3个（让题目更难）
        # Easy/Medium: 随机选3个
        if use_hard_questions and len(others) >= 3:
            others_sorted = sorted(others, key=lambda x: x["pchembl_value"], reverse=True)
            chosen_others = others_sorted[:3]
        else:
            chosen_others = random.sample(others, 3)

        candidates = chosen_others + [best]
        random.shuffle(candidates)

        pvals = [v["pchembl_value"] for v in candidates]
        max_val = max(pvals)
        tied = [v for v in candidates if v["pchembl_value"] == max_val]
        best = random.choice(tied)
        correct_idx = candidates.index(best)
        max_val = best["pchembl_value"]

        target_name  = cluster.get("target_name", "this target")
        focus_pos    = cluster.get("focus_rgroup_pos", 0)
        n_rg         = cluster.get("n_rgroup_positions", 1)

        if use_hard_questions:
            if random.random() < 0.5 and target_name != "this target":
                q = random.choice(BIO_QUESTIONS_HARD_WITH_TARGET).format(target=target_name[:40])
            else:
                q = random.choice(BIO_QUESTIONS_HARD_GENERIC)
        else:
            if random.random() < 0.5 and target_name != "this target":
                q = random.choice(BIO_QUESTIONS_EASY_WITH_TARGET).format(target=target_name[:40])
            else:
                q = random.choice(BIO_QUESTIONS_EASY_GENERIC)

        if n_rg > 1:
            pos_label = f"R{focus_pos + 1}"
            q = q.replace("substitution", f"substitution at {pos_label}")
            q = q.replace("R-group choice", f"R-group choice at {pos_label}")

        prop_values = {lbl: v["pchembl_value"]
                       for lbl, v in zip(["A","B","C","D"], candidates)}

        vqa.append({
            "split":               split,
            "property":            "bio_activity",
            "original_esmiles":    cluster.get("scaffold_star_smiles", ""),
            "original_image_path": cluster.get("scaffold_image_path", ""),
            "instruction":         "",
            "question":            q,
            "correct_answer":      ["A","B","C","D"][correct_idx],
            "correct_smiles":      candidates[correct_idx]["full_mol_smiles"],
            "correct_prop_value":  max_val,
            "prop_values":         json.dumps(prop_values),
            "is_none_of_above":    False,
            "nova_answer_is_d":    False,
            "nova_type":           "",
            "target_chembl_id":    cluster.get("target_chembl_id", ""),
            "target_name":         target_name,
            "option_A_image":      candidates[0]["mol_image_path"],
            "option_B_image":      candidates[1]["mol_image_path"],
            "option_C_image":      candidates[2]["mol_image_path"],
            "option_D_image":      candidates[3]["mol_image_path"],
            "option_A_smiles":     candidates[0]["full_mol_smiles"],
            "option_B_smiles":     candidates[1]["full_mol_smiles"],
            "option_C_smiles":     candidates[2]["full_mol_smiles"],
            "option_D_smiles":     candidates[3]["full_mol_smiles"],
        })

    q_style = "描述性+最难3干扰" if use_hard_questions else "直接+随机干扰"
    print(f"  {split} Bio(SAR同骨架,{q_style}): {len(vqa)} 条"
          f"（变体不足{skipped['variants']} 图像失败{skipped['image']}）")
    return vqa


# ══════════════════════════════════════════════════════════
# Advanced Indication: 骨架 + 4个R基选项（v6设计保留）
# ══════════════════════════════════════════════════════════
def _build_advanced_indication(chembl_records, ind_clusters, target, split):
    use_hard_questions = ("hard" in split)
    vqa = []

    # ── 优先：SAR indication簇（同骨架，R基选项）─────────────────
    sar_vqa = []
    if ind_clusters:
        random.shuffle(ind_clusters)
        for cluster in ind_clusters:
            if len(sar_vqa) >= target: break
            variants = cluster.get("variants", [])

            valid = [v for v in variants
                     if (v.get("indications") or v.get("mechanisms"))
                     and os.path.isfile(v.get("mol_image_path", ""))]
            if len(valid) < 4:
                continue

            correct = None
            term = None
            use_ind = None
            best_uniqueness = 0

            for v in valid:
                for ind in v.get("indications", []):
                    others_with_term = sum(
                        1 for u in valid
                        if u is not v and ind in u.get("indications", [])
                    )
                    uniqueness = len(valid) - 1 - others_with_term
                    if uniqueness > best_uniqueness:
                        best_uniqueness = uniqueness
                        correct = v
                        term = ind
                        use_ind = True
                for mech in v.get("mechanisms", []):
                    others_with_term = sum(
                        1 for u in valid
                        if u is not v and mech in u.get("mechanisms", [])
                    )
                    uniqueness = len(valid) - 1 - others_with_term
                    if uniqueness > best_uniqueness:
                        best_uniqueness = uniqueness
                        correct = v
                        term = mech
                        use_ind = False

            if correct is None or term is None:
                continue

            if use_ind:
                distractors = [v for v in valid
                               if v is not correct
                               and term not in v.get("indications", [])]
            else:
                distractors = [v for v in valid
                               if v is not correct
                               and term not in v.get("mechanisms", [])]

            if len(distractors) < 3:
                continue
            chosen_dist = random.sample(distractors, 3)
            candidates  = chosen_dist + [correct]
            random.shuffle(candidates)
            correct_idx = candidates.index(correct)

            if use_hard_questions:
                if use_ind:
                    desc = indication_to_descriptive(term)
                    q = random.choice(INDICATION_QUESTIONS_HARD_RGROUP).format(indication_desc=desc)
                else:
                    desc = mechanism_to_descriptive(term)
                    q = random.choice(MECHANISM_QUESTIONS_HARD_RGROUP).format(mechanism_desc=desc)
            else:
                if use_ind:
                    q = random.choice(INDICATION_QUESTIONS_EASY_RGROUP).format(indication=term[:50])
                else:
                    q = random.choice(MECHANISM_QUESTIONS_EASY_RGROUP).format(mechanism=term[:60])

            def get_option_img(v):
                rg_img = v.get("rgroup_image_path", "")
                if rg_img and os.path.isfile(rg_img):
                    return rg_img
                return v.get("mol_image_path", "")

            def get_option_smi(v):
                rg_smi = v.get("focus_rgroup_smiles", "")
                if rg_smi:
                    return rg_smi
                return v.get("full_mol_smiles", "")

            sar_vqa.append({
                "split":               split,
                "property":            "indication_mechanism",
                "original_esmiles":    cluster.get("scaffold_star_smiles", ""),
                "original_image_path": cluster.get("scaffold_image_path", ""),
                "instruction": "", "question": q,
                "correct_answer":      ["A","B","C","D"][correct_idx],
                "correct_smiles":      get_option_smi(correct),
                "correct_prop_value":  1.0,
                "prop_values":         json.dumps({}),
                "is_none_of_above": False, "nova_answer_is_d": False, "nova_type": "",
                "option_A_image":  get_option_img(candidates[0]),
                "option_B_image":  get_option_img(candidates[1]),
                "option_C_image":  get_option_img(candidates[2]),
                "option_D_image":  get_option_img(candidates[3]),
                "option_A_smiles": get_option_smi(candidates[0]),
                "option_B_smiles": get_option_smi(candidates[1]),
                "option_C_smiles": get_option_smi(candidates[2]),
                "option_D_smiles": get_option_smi(candidates[3]),
            })

    vqa.extend(sar_vqa)
    style = "描述性" if use_hard_questions else "直接"
    print(f"  {split} Indication(SAR同骨架R基选项,{style}): {len(sar_vqa)} 条")

    remaining = target - len(vqa)
    if remaining > 0:
        fallback = _build_indication_fallback(chembl_records, remaining, split)
        vqa.extend(fallback)

    return vqa


def _build_indication_fallback(chembl_records, target, split):
    use_hard_questions = ("hard" in split)
    vqa = []
    approved = [r for r in chembl_records
                if (r.get("indications") or r.get("mechanisms"))
                and os.path.isfile(r.get("output_image_path",""))]
    if not approved: return []
    random.shuffle(approved)
    used = set()
    skipped = 0

    for correct in approved:
        if len(vqa) >= target: break
        cid = correct.get("chembl_id", id(correct))
        if cid in used: continue
        inds  = correct.get("indications", [])
        mechs = correct.get("mechanisms", [])
        use_ind = bool(inds) and (not mechs or random.random() < 0.6)
        if use_ind:
            term = random.choice(inds)
            if use_hard_questions:
                desc = indication_to_descriptive(term[:50])
                q = random.choice(INDICATION_QUESTIONS_HARD_RGROUP).format(indication_desc=desc)
            else:
                q = random.choice(INDICATION_QUESTIONS_EASY_RGROUP).format(indication=term[:50])
            distractor_pool = [r for r in approved
                               if cid != r.get("chembl_id", id(r))
                               and term not in (r.get("indications") or [])]
        else:
            term = random.choice(mechs)
            if use_hard_questions:
                desc = mechanism_to_descriptive(term[:60])
                q = random.choice(MECHANISM_QUESTIONS_HARD_RGROUP).format(mechanism_desc=desc)
            else:
                q = random.choice(MECHANISM_QUESTIONS_EASY_RGROUP).format(mechanism=term[:60])
            distractor_pool = [r for r in approved
                               if cid != r.get("chembl_id", id(r))
                               and term not in (r.get("mechanisms") or [])]
        if len(distractor_pool) < 3:
            skipped += 1; continue
        distractors = random.sample(distractor_pool, 3)
        candidates  = distractors + [correct]
        random.shuffle(candidates)
        correct_idx = candidates.index(correct)
        vqa.append(make_chembl_vqa(
            split, "indication_mechanism",
            correct.get("markush_smiles",""), correct.get("markush_image_path",""),
            q, "", candidates, correct_idx, 1.0, {}
        ))
        used.add(cid)

    style = "描述性" if use_hard_questions else "直接"
    print(f"  {split} Indication(fallback 4药物,{style}): {len(vqa)} 条（跳过{skipped}）")
    return vqa


# ══════════════════════════════════════════════════════════
# Advanced RDKit: Easy/Medium直接问题，Hard描述性问题
# ══════════════════════════════════════════════════════════
def _build_advanced_rdkit(chembl_records, target, split, same_scaffold):
    use_hard_questions = ("hard" in split)
    vqa = []
    all_valid = [r for r in chembl_records
                 if os.path.isfile(r.get("output_image_path",""))
                 and r.get("full_molecule_smiles")]
    groups = defaultdict(list)
    for r in all_valid:
        groups[r["markush_smiles"]].append(r)

    scaffold_list = list(groups.keys()); random.shuffle(scaffold_list)
    prop_cycle = PROPERTY_NAMES * (target // len(PROPERTY_NAMES) + 2)
    random.shuffle(prop_cycle)
    prop_idx = 0
    skipped  = {"prop": 0, "tie": 0, "distractor": 0}

    for scaffold_smi in scaffold_list:
        if len(vqa) >= target: break
        variants = groups[scaffold_smi]
        prop = prop_cycle[prop_idx % len(prop_cycle)]; prop_idx += 1
        cfg  = PROPERTY_CONFIGS[prop]

        if same_scaffold and len(variants) >= MIN_SAME_SCAFFOLD:
            chosen   = random.sample(variants, MIN_SAME_SCAFFOLD)
            val_list = [(compute_rdkit_prop(v.get("full_molecule_smiles",""), prop), v) for v in chosen]
            valid    = [(val, v) for val, v in val_list if val is not None]
            if len(valid) < MIN_SAME_SCAFFOLD: skipped["prop"] += 1; continue
            valid.sort(key=lambda x: x[0], reverse=True)
            if valid[0][0] == valid[1][0]: skipped["tie"] += 1; continue
            correct_val, correct = valid[0]
            candidates = [v for _, v in valid]
            random.shuffle(candidates)
        else:
            val_map = {}
            for v in variants:
                val = compute_rdkit_prop(v.get("full_molecule_smiles",""), prop)
                if val is not None: val_map[id(v)] = (val, v)
            if not val_map: skipped["prop"] += 1; continue
            best_id = max(val_map, key=lambda k: val_map[k][0])
            correct_val, correct = val_map[best_id]
            other = [r for r in all_valid if r["markush_smiles"] != scaffold_smi]
            if len(other) < 3: skipped["distractor"] += 1; continue
            distractors = random.sample(other, 3)
            candidates  = distractors + [correct]
            random.shuffle(candidates)
            pv_check = {lbl: compute_rdkit_prop(c.get("full_molecule_smiles",""), prop)
                        for lbl, c in zip(["A","B","C","D"], candidates)}
            max_val = max(v for v in pv_check.values() if v is not None)
            if pv_check[["A","B","C","D"][candidates.index(correct)]] != max_val:
                skipped["tie"] += 1; continue

        correct_idx = candidates.index(correct)
        prop_values = {lbl: compute_rdkit_prop(c.get("full_molecule_smiles",""), prop)
                       for lbl, c in zip(["A","B","C","D"], candidates)}

        if use_hard_questions:
            q = random.choice(cfg["questions_hard"])
        else:
            q = random.choice(cfg["questions_easy"])

        vqa.append(make_chembl_vqa(
            split, prop, scaffold_smi,
            correct.get("markush_image_path",""),
            q, "", candidates, correct_idx, correct_val, prop_values
        ))

    label = "同骨架优先" if same_scaffold else "跨骨架"
    print(f"  {split} RDKit({label},{('描述性' if use_hard_questions else '直接')}): {len(vqa)} 条"
          f"（性质失败{skipped['prop']} 并列{skipped['tie']} 干扰不足{skipped['distractor']}）")
    return vqa


# ══════════════════════════════════════════════════════════
# Advanced 组合入口
# ══════════════════════════════════════════════════════════
def build_advanced(chembl_records, bio_clusters, ind_clusters, target, split, same_scaffold):
    bio_target = int(target * 0.45)
    ind_target = int(target * 0.30)
    rdk_target = target - bio_target - ind_target

    bio_vqa = _build_advanced_bio(bio_clusters, bio_target, split, same_scaffold)
    ind_vqa = _build_advanced_indication(chembl_records, ind_clusters, ind_target, split)
    rdk_vqa = _build_advanced_rdkit(chembl_records, rdk_target, split, same_scaffold)

    all_vqa = bio_vqa + ind_vqa + rdk_vqa
    random.shuffle(all_vqa)
    print(f"  {split} 合计: {len(all_vqa)} 条 "
          f"(bio={len(bio_vqa)} ind={len(ind_vqa)} rdkit={len(rdk_vqa)})")
    return all_vqa


# ══════════════════════════════════════════════════════════
# 保存（含分类统计）
# ══════════════════════════════════════════════════════════
def save_split(vqa, name, vqa_dir):
    if not vqa:
        print(f"  ⚠️  {name} 为空，跳过保存")
        return
    os.makedirs(vqa_dir, exist_ok=True)
    Dataset.from_list(vqa).save_to_disk(os.path.join(vqa_dir, name))
    preview_path = os.path.join(vqa_dir, f"{name}_preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in r.items() if "image" not in k or k == "original_image_path"}
             for r in vqa[:10]],
            f, indent=2, ensure_ascii=False
        )
    total = len(vqa)
    ans_dist = Counter(r["correct_answer"] for r in vqa)
    print(f"    保存: {os.path.join(vqa_dir, name)} ({total}条)")
    print(f"    答案分布: " + " ".join(f"{a}:{ans_dist.get(a,0)}" for a in "ABCD"))

    nova_recs = [r for r in vqa if r.get("is_none_of_above")]
    if nova_recs:
        type_b  = [r for r in nova_recs if r.get("nova_type") == "type_b"]
        type_c  = [r for r in nova_recs if r.get("nova_type") == "type_c"]
        valid_d = [r for r in nova_recs if r.get("nova_type") == "valid_d"]
        normal  = [r for r in vqa if not r.get("is_none_of_above")]
        print(f"    None of Above分布:")
        print(f"      正常题:          {len(normal):>4} ({len(normal)/max(total,1)*100:.1f}%)")
        print(f"      TYPE B(R基不存在→D): {len(type_b):>4} ({len(type_b)/max(total,1)*100:.1f}%)")
        print(f"      TYPE C(→D):      {len(type_c):>4} ({len(type_c)/max(total,1)*100:.1f}%)")
        print(f"      占位D(→A/B/C):   {len(valid_d):>4} ({len(valid_d)/max(total,1)*100:.1f}%)")

    if vqa[0].get("property"):
        print(f"    性质分布: {dict(Counter(r['property'] for r in vqa).most_common())}")


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("新版 VQA Benchmark 构建（v7 - 6个splits）")
    print("  新增: Medium split（同骨架+直接instruction+NoA）")
    print("  改动: Hard Basic去掉NoA，改用R基相似度最高干扰项")
    print("=" * 65)

    print(f"\n[1] 加载edit数据集（Basic题用）...")
    edit_records = load_all_batches(EDIT_DATASET_DIR)
    edit_records = filter_valid_edit(edit_records)
    edit_groups  = group_by_scaffold(edit_records)
    enough = sum(1 for v in edit_groups.values() if len(v) >= MIN_SAME_SCAFFOLD)
    print(f"  骨架数: {len(edit_groups)}，变体>={MIN_SAME_SCAFFOLD}的骨架: {enough}")

    print(f"\n[2] 加载ChEMBL数据（Advanced题indication/rdkit用）...")
    chembl_records = load_chembl_records(CHEMBL_JSONL)
    chembl_records = filter_valid_chembl(chembl_records) if chembl_records else []

    print(f"\n[2b] 加载SAR scaffold数据（Advanced题bio/indication用）...")
    bio_clusters, ind_clusters = load_sar_records(SAR_JSONL)

    print(f"\n[3] 构建 Basic sub-benchmarks...")
    easy_basic   = build_easy_basic(edit_records, edit_groups, TARGET_BASIC)
    medium_basic = build_medium_basic(edit_groups, TARGET_BASIC)
    hard_basic   = build_hard_basic(edit_groups, TARGET_BASIC)

    print(f"\n[4] 构建 Advanced sub-benchmarks...")
    print(f"  构建 Easy Advanced（跨骨架+直接性问题）...")
    easy_advanced = build_advanced(chembl_records, bio_clusters, ind_clusters,
                                   TARGET_ADVANCED, "easy_advanced", same_scaffold=False)
    print(f"  构建 Medium Advanced（同骨架+直接性问题）...")
    medium_advanced = build_advanced(chembl_records, bio_clusters, ind_clusters,
                                     TARGET_ADVANCED, "medium_advanced", same_scaffold=True)
    print(f"  构建 Hard Advanced（同骨架+描述性问题+最难干扰）...")
    hard_advanced = build_advanced(chembl_records, bio_clusters, ind_clusters,
                                   TARGET_ADVANCED, "hard_advanced", same_scaffold=True)

    print(f"\n[5] 保存到: {VQA_DIR}")
    save_split(easy_basic,      "easy_basic",      VQA_DIR)
    save_split(medium_basic,    "medium_basic",     VQA_DIR)
    save_split(hard_basic,      "hard_basic",       VQA_DIR)
    save_split(easy_advanced,   "easy_advanced",    VQA_DIR)
    save_split(medium_advanced, "medium_advanced",  VQA_DIR)
    save_split(hard_advanced,   "hard_advanced",    VQA_DIR)

    total = (len(easy_basic) + len(medium_basic) + len(hard_basic) +
             len(easy_advanced) + len(medium_advanced) + len(hard_advanced))
    print(f"\n{'='*65}")
    print(f"✓ 完成（v7）")
    print(f"  Easy Basic:       {len(easy_basic):>4} 条")
    print(f"  Medium Basic:     {len(medium_basic):>4} 条")
    print(f"  Hard Basic:       {len(hard_basic):>4} 条")
    print(f"  Easy Advanced:    {len(easy_advanced):>4} 条")
    print(f"  Medium Advanced:  {len(medium_advanced):>4} 条")
    print(f"  Hard Advanced:    {len(hard_advanced):>4} 条")
    print(f"  合计:             {total:>4} 条")

if __name__ == "__main__":
    main()