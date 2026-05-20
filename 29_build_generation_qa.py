# 文件位置: Moleditor/code/29_build_generation_qa.py
#
# 作用: 构建填空题（Generation QA）数据集
#       输入：Markush骨架（图片 or E-SMILES）+ R基替换instruction
#       输出：模型直接生成替换后的完整分子标准SMILES（而非选择题）
#
# ── Split设计 ─────────────────────────────────────────────
#   Easy : 直接instruction（"Replace R1 with methyl"）
#   Hard : 描述性instruction（"Replace R1 with a saturated alicyclic ring"）
#   （填空题无干扰项，Medium跨/同骨架的区别在此无意义，故只保留两档）
#
# ── 数据结构 ─────────────────────────────────────────────
#   每条记录：
#     - original_esmiles        : 骨架E-SMILES（LLM track输入）
#     - original_image_path     : 骨架图片路径（VLM track输入）
#     - instruction             : 替换指令（easy: direct, hard: descriptive）
#     - correct_smiles          : ground truth（RDKit canonical SMILES）
#     - difficulty              : "easy" / "hard"
#     - scaffold_smiles_clean   : strip掉<a>标签后的骨架SMILES（供参考）
#
# ── 评测指标（由30_eval_generation.py计算）────────────────
#   1. Validity Rate   : 模型输出是否为合法SMILES
#   2. Exact Match     : canonical SMILES是否与ground truth完全一致
#   3. Tanimoto Sim    : 与ground truth的ECFP4 Tanimoto相似度（0~1，连续分）
#   4. Scaffold Match  : MCS是否包含原骨架核心结构（验证骨架未被改动）
#
# 用法: python 29_build_generation_qa.py
# ─────────────────────────────────────────────────────────

import json, os, random, re as _re
from datasets import Dataset, load_from_disk
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# ── 路径 ──────────────────────────────────────────────────
EDIT_DATASET_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/edit_dataset"
GEN_QA_DIR       = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/generation_qa"
# 集群:
# EDIT_DATASET_DIR = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/edit_dataset"
# GEN_QA_DIR       = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/generation_qa"

RANDOM_SEED       = 42
TARGET_PER_SPLIT  = 500   # 每个难度档的题目数量

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════
# 数据加载（复用26的逻辑）
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
    """过滤有效记录：图片存在、output_smiles存在、且RDKit能解析"""
    valid = []
    for r in records:
        if not os.path.isfile(r.get("output_image_path", "")): continue
        if not os.path.isfile(r.get("original_image_path", "")): continue
        smi = r.get("output_smiles", "")
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        valid.append(r)
    print(f"  有效edit记录: {len(valid)} / {len(records)}")
    return valid

def canonical_smiles(smi: str) -> str | None:
    """返回RDKit canonical SMILES，无效则返回None"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def strip_esmiles_annotations(esmiles: str) -> str:
    """去除E-SMILES的<a>...</a>注解标签，返回可用SMILES"""
    return _re.sub(r"<a>[^<]*</a>", "", esmiles)


# ══════════════════════════════════════════════════════════
# 构建函数
# ══════════════════════════════════════════════════════════

def build_easy_gen(records, target):
    """
    Easy：直接instruction，从所有有效记录里随机采样。
    """
    vqa = []
    skipped = 0

    # 随机打乱所有记录，直接逐条出题
    shuffled = list(records)
    random.shuffle(shuffled)

    for rec in shuffled:
        if len(vqa) >= target: break

        instr = rec.get("instruction", "")
        if not instr: skipped += 1; continue

        ground_truth = canonical_smiles(rec.get("output_smiles", ""))
        if ground_truth is None: skipped += 1; continue

        vqa.append({
            "difficulty":           "easy",
            "original_esmiles":     rec["original_esmiles"],
            "original_image_path":  rec["original_image_path"],
            "instruction":          instr,
            "correct_smiles":       ground_truth,
            "scaffold_smiles_clean": strip_esmiles_annotations(rec["original_esmiles"]),
            # 额外字段，方便分析
            "output_image_path":    rec.get("output_image_path", ""),
        })

    print(f"  Easy Gen: {len(vqa)} 条（跳过{skipped}）")
    return vqa


def build_hard_gen(records, target):
    """
    Hard：使用描述性instruction（descriptive_instruction字段）。
    填空题无干扰项，不需要同骨架限制，直接从所有有descriptive_instruction的记录里选。
    """
    vqa = []
    skipped = {"no_instr": 0, "invalid_smi": 0}

    shuffled = [r for r in records if r.get("descriptive_instruction")]
    random.shuffle(shuffled)

    for rec in shuffled:
        if len(vqa) >= target: break

        instr = rec.get("descriptive_instruction", "")
        if not instr: skipped["no_instr"] += 1; continue

        ground_truth = canonical_smiles(rec.get("output_smiles", ""))
        if ground_truth is None: skipped["invalid_smi"] += 1; continue

        vqa.append({
            "difficulty":            "hard",
            "original_esmiles":      rec["original_esmiles"],
            "original_image_path":   rec["original_image_path"],
            "instruction":           instr,
            "correct_smiles":        ground_truth,
            "scaffold_smiles_clean": strip_esmiles_annotations(rec["original_esmiles"]),
            "output_image_path":     rec.get("output_image_path", ""),
        })

    print(f"  Hard Gen: {len(vqa)} 条"
          f"（跳过: 无descriptive_instr={skipped['no_instr']} "
          f"SMILES无效={skipped['invalid_smi']}）")
    return vqa


# ══════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════

def save_split(vqa, name, out_dir):
    if not vqa:
        print(f"  ⚠️  {name} 为空，跳过保存")
        return
    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_list(vqa).save_to_disk(os.path.join(out_dir, name))

    # 保存preview（不含图片二进制，方便查看）
    preview_path = os.path.join(out_dir, f"{name}_preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(vqa[:10], f, indent=2, ensure_ascii=False)

    print(f"    保存: {os.path.join(out_dir, name)} ({len(vqa)}条)")
    # 展示几条样例
    for rec in vqa[:2]:
        print(f"      instruction: {rec['instruction'][:80]}...")
        print(f"      answer:      {rec['correct_smiles']}")


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Generation QA Benchmark 构建（v2）")
    print("  Easy : 直接instruction（跨所有记录随机采样）")
    print("  Hard : 描述性instruction（跨所有记录随机采样）")
    print("=" * 65)

    print(f"\n[1] 加载edit数据集...")
    edit_records = load_all_batches(EDIT_DATASET_DIR)
    edit_records = filter_valid_edit(edit_records)
    has_descriptive = sum(1 for r in edit_records if r.get("descriptive_instruction"))
    print(f"  有descriptive_instruction的记录: {has_descriptive} / {len(edit_records)}")

    print(f"\n[2] 构建各难度的Generation QA...")
    easy_gen = build_easy_gen(edit_records, TARGET_PER_SPLIT)
    hard_gen = build_hard_gen(edit_records, TARGET_PER_SPLIT)

    print(f"\n[3] 保存到: {GEN_QA_DIR}")
    save_split(easy_gen, "easy", GEN_QA_DIR)
    save_split(hard_gen, "hard", GEN_QA_DIR)

    total = len(easy_gen) + len(hard_gen)
    print(f"\n{'='*65}")
    print(f"✓ 完成（v2）")
    print(f"  Easy Gen: {len(easy_gen):>4} 条（直接instruction）")
    print(f"  Hard Gen: {len(hard_gen):>4} 条（描述性instruction）")
    print(f"  合计:     {total:>4} 条")
    print(f"\n次步: 运行 30_eval_generation.py 进行评测")

if __name__ == "__main__":
    main()