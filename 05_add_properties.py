# 文件位置: Moleditor/code/05_add_properties.py
# 作用: 为数据集中所有output_smiles计算RDKit理化性质，更新batch文件

from datasets import load_from_disk, Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import os
import shutil

OUTPUT_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/rgroup_edit_dataset"

def compute_properties(smiles: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "mol_weight": round(Descriptors.MolWt(mol), 2),
            "logP":       round(Crippen.MolLogP(mol), 3),
            "tpsa":       round(rdMolDescriptors.CalcTPSA(mol), 2),
            "hbd":        Lipinski.NumHDonors(mol),
            "hba":        Lipinski.NumHAcceptors(mol),
            "rot_bonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
    except Exception:
        return None


def process_batch(batch_dir: str) -> int:
    ds = load_from_disk(batch_dir)
    failed = 0

    new_columns = {
        "mol_weight": [],
        "logP":       [],
        "tpsa":       [],
        "hbd":        [],
        "hba":        [],
        "rot_bonds":  [],
    }

    for record in ds:
        props = compute_properties(record["output_smiles"])
        if props is None:
            failed += 1
            new_columns["mol_weight"].append(-1.0)
            new_columns["logP"].append(-1.0)
            new_columns["tpsa"].append(-1.0)
            new_columns["hbd"].append(-1)
            new_columns["hba"].append(-1)
            new_columns["rot_bonds"].append(-1)
        else:
            for key in new_columns:
                new_columns[key].append(props[key])

    for col_name, col_values in new_columns.items():
        ds = ds.add_column(col_name, col_values)

    # ── 核心修改：先存到临时目录，再替换原目录 ────────────
    tmp_dir = batch_dir + "_tmp"
    ds.save_to_disk(tmp_dir)
    shutil.rmtree(batch_dir)
    shutil.move(tmp_dir, batch_dir)

    return failed


def main():
    batch_dirs = sorted([
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("batch_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])
    print(f"找到 {len(batch_dirs)} 个batch\n")

    total_failed = 0
    total_records = 0

    for i, bd in enumerate(batch_dirs):
        print(f"处理 [{i+1}/{len(batch_dirs)}]: {os.path.basename(bd)}")
        failed = process_batch(bd)
        ds = load_from_disk(bd)
        total_records += len(ds)
        total_failed += failed
        print(f"  完成: {len(ds)} 条, 失败: {failed} 条")

    print(f"\n全部完成: 共 {total_records} 条, 性质计算失败: {total_failed} 条")
    print("各batch已更新，新增字段: mol_weight, logP, tpsa, hbd, hba, rot_bonds")


if __name__ == "__main__":
    main()