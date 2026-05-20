# 文件位置: Moleditor/code/02_analyze_data.py
# 作用: 分析Markush数据集的结构，统计各类标签和R-group名称的分布

from datasets import load_from_disk
import re
from collections import Counter

SAVE_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/molparser_markush"

def main():
    print("加载本地数据集...")
    ds = load_from_disk(SAVE_DIR)
    print(f"共 {len(ds)} 条样本\n")

    # 统计各类标签出现次数
    tag_counter = Counter()
    # 统计所有group name（<a>里冒号后面的内容）
    a_group_counter = Counter()
    r_group_counter = Counter()
    
    # 统计每条样本里有几个<a>标签（R-group数量分布）
    num_a_tags = Counter()
    
    # 统计只含<a>标签（不含<r><c>）的"干净"样本数
    clean_count = 0

    for sample in ds:
        smiles = sample["SMILES"]
        
        # 统计标签类型
        if "<a>" in smiles:
            tag_counter["<a>"] += 1
        if "<r>" in smiles:
            tag_counter["<r>"] += 1
        if "<c>" in smiles:
            tag_counter["<c>"] += 1
        if "<dum>" in smiles:
            tag_counter["<dum>"] += 1
        
        # 提取<a>里的group name
        a_groups = re.findall(r"<a>\d+:([^<]+)</a>", smiles)
        for g in a_groups:
            a_group_counter[g] += 1
        num_a_tags[len(a_groups)] += 1
        
        # 提取<r>里的group name
        r_groups = re.findall(r"<r>\d+:([^<]+)</r>", smiles)
        for g in r_groups:
            r_group_counter[g] += 1
        
        # 判断是否"干净"样本：只有<a>标签，没有<r><c>，且<a>里全是R[x]或X/Y/Z
        if "<a>" in smiles and "<r>" not in smiles and "<c>" not in smiles:
            a_names = re.findall(r"<a>\d+:([^<]+)</a>", smiles)
            # 只含真正R-group的：R[x], X, Y, Z, R（不含<dum>，不含具体基团）
            is_real_rgroup = lambda name: (
                re.match(r"R\[?\d*\]?$", name) or name in {"X", "Y", "Z", "R"}
            )
            if a_names and all(is_real_rgroup(n) for n in a_names):
                clean_count += 1

    # ── 打印结果 ──────────────────────────────────────────
    print("=" * 50)
    print("【标签类型分布】")
    for tag, count in tag_counter.most_common():
        print(f"  {tag}: {count} 条 ({count/len(ds)*100:.1f}%)")

    print("\n【<a>标签中 group name TOP 30】")
    for name, count in a_group_counter.most_common(30):
        print(f"  {name}: {count}")

    print("\n【<r>标签中 group name TOP 15】")
    for name, count in r_group_counter.most_common(15):
        print(f"  {name}: {count}")

    print("\n【每条样本含<a>标签数量分布】")
    for n, count in sorted(num_a_tags.items()):
        print(f"  {n}个<a>标签: {count} 条")

    print("\n【'干净'Markush样本数（只含真正R-group，无<r><c><dum>）】")
    print(f"  {clean_count} 条 ({clean_count/len(ds)*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()