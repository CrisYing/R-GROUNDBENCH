# 文件位置: Moleditor/code/04_dataset_stats.py
# 作用: 统计数据集质量指标，为论文数据集章节提供数据支撑

from datasets import load_from_disk
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import json
import os
from collections import Counter

OUTPUT_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/rgroup_edit_dataset"

def load_all_batches(output_dir):
    records = []
    batch_dirs = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith("batch_") and os.path.isdir(os.path.join(output_dir, d))
    ])
    print(f"找到 {len(batch_dirs)} 个batch")
    for bd in batch_dirs:
        ds = load_from_disk(os.path.join(output_dir, bd))
        records.extend(list(ds))
    print(f"共加载 {len(records)} 条记录")
    return records

def mol_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "num_atoms":  mol.GetNumAtoms(),
            "num_bonds":  mol.GetNumBonds(),
            "mol_weight": round(Descriptors.MolWt(mol), 2),
            "num_rings":  rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return None

def main():
    records = load_all_batches(OUTPUT_DIR)
    total = len(records)

    has_style = "instruction_style" in records[0] if records else False

    # ── 统计instruction多样性 ──────────────────────────────
    instruction_counter = Counter(r["instruction"] for r in records)
    unique_instructions = len(instruction_counter)

    # ── 统计唯一化学操作数（按rgroup_assignment去重）─────────
    # 同一个化学替换有5种措辞，去重后才是真实的化学变体数
    unique_chem_ops = len(set(
        r["original_esmiles"] + "||" + r["rgroup_assignment"]
        for r in records
    ))

    # ── 统计instruction_style分布 ─────────────────────────
    if has_style:
        style_counter = Counter(r["instruction_style"] for r in records)
    else:
        style_counter = Counter()
        print("⚠️  当前数据集无instruction_style字段（旧版数据集）")

    # ── 统计R-group替换类型分布 ────────────────────────────
    substituent_counter = Counter()
    rgroup_name_counter = Counter()
    num_rgroups_counter = Counter()

    # 按化学操作去重后再统计取代基分布，避免5倍重复计数
    seen_ops = set()
    for r in records:
        op_key = r["original_esmiles"] + "||" + r["rgroup_assignment"]
        if op_key in seen_ops:
            continue
        seen_ops.add(op_key)

        assignment = json.loads(r["rgroup_assignment"])
        num_rgroups_counter[len(assignment)] += 1
        for rname, rinfo in assignment.items():
            substituent_counter[rinfo["substituent_en"]] += 1
            rgroup_name_counter[rname] += 1

    # ── 统计输出分子属性（同样去重）──────────────────────────
    mol_weights = []
    num_atoms_list = []
    invalid_smiles = 0
    seen_smiles = set()

    for r in records:
        if r["output_smiles"] in seen_smiles:
            continue
        seen_smiles.add(r["output_smiles"])

        props = mol_properties(r["output_smiles"])
        if props is None:
            invalid_smiles += 1
        else:
            mol_weights.append(props["mol_weight"])
            num_atoms_list.append(props["num_atoms"])

    # ── 统计原始分子多样性 ─────────────────────────────────
    unique_source_mols = len(set(r["original_esmiles"] for r in records))

    # ── 打印结果 ──────────────────────────────────────────
    print("\n" + "="*55)
    print("           数据集统计报告")
    print("="*55)

    print(f"\n【基本信息】")
    print(f"  总记录数（含5种style）:   {total}")
    print(f"  唯一化学操作数:           {unique_chem_ops}")
    print(f"  来源分子数:               {unique_source_mols}")
    print(f"  平均每分子化学变体数:     {unique_chem_ops/unique_source_mols:.2f}")
    print(f"  唯一instruction文本数:    {unique_instructions}")
    print(f"  SMILES验证失败（去重后）: {invalid_smiles}")

    if has_style:
        print(f"\n【Instruction Style 分布】")
        for style, count in style_counter.most_common():
            print(f"  {style:>15s}: {count:>6} 条 ({count/total*100:.1f}%)")
        styles_per_op = total / unique_chem_ops if unique_chem_ops else 0
        print(f"  → 每个化学操作平均覆盖 {styles_per_op:.1f} 种instruction")

    print(f"\n【Instruction 模板多样性】")
    print(f"  唯一instruction文本数: {unique_instructions}")
    print(f"  重复最多的TOP 5:")
    for instr, cnt in instruction_counter.most_common(5):
        preview = instr[:60] + "..." if len(instr) > 60 else instr
        print(f"    [{cnt}次] {preview}")

    print(f"\n【每条样本R-group数量分布（化学操作去重后）】")
    for n, count in sorted(num_rgroups_counter.items()):
        print(f"  {n}个R-group: {count} 条 ({count/unique_chem_ops*100:.1f}%)")

    print(f"\n【取代基使用频率 TOP 15（化学操作去重后）】")
    total_sub = sum(substituent_counter.values())
    for sub, count in substituent_counter.most_common(15):
        print(f"  {sub}: {count} ({count/total_sub*100:.1f}%)")

    print(f"\n【R-group名称分布 TOP 15（化学操作去重后）】")
    for name, count in rgroup_name_counter.most_common(15):
        print(f"  {name}: {count}")

    print(f"\n【输出分子属性（SMILES去重后）】")
    if mol_weights:
        print(f"  分子量 均值: {sum(mol_weights)/len(mol_weights):.1f}  "
              f"最小: {min(mol_weights):.1f}  最大: {max(mol_weights):.1f}")
    if num_atoms_list:
        print(f"  原子数 均值: {sum(num_atoms_list)/len(num_atoms_list):.1f}  "
              f"最小: {min(num_atoms_list)}  最大: {max(num_atoms_list)}")

    # ── 保存报告 ──────────────────────────────────────────
    report = {
        "total_records":              total,
        "unique_chem_ops":            unique_chem_ops,
        "unique_source_mols":         unique_source_mols,
        "avg_variants_per_mol":       round(unique_chem_ops/unique_source_mols, 2),
        "styles_per_op":              round(total/unique_chem_ops, 2) if unique_chem_ops else 0,
        "unique_instructions":        unique_instructions,
        "invalid_smiles_dedup":       invalid_smiles,
        "instruction_style_dist":     dict(style_counter.most_common()) if has_style else {},
        "num_rgroups_dist":           dict(sorted(num_rgroups_counter.items())),
        "substituent_freq":           dict(substituent_counter.most_common()),
        "rgroup_name_freq":           dict(rgroup_name_counter.most_common(30)),
        "mol_weight_mean":            round(sum(mol_weights)/len(mol_weights), 1) if mol_weights else 0,
        "num_atoms_mean":             round(sum(num_atoms_list)/len(num_atoms_list), 1) if num_atoms_list else 0,
    }

    report_path = os.path.join(OUTPUT_DIR, "dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存至: {report_path}")
    print("="*55)

if __name__ == "__main__":
    main()