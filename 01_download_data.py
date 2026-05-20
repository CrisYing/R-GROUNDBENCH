# 文件位置: MolEditor/code/01_download_data.py
# 作用: 从HuggingFace下载sft_real子集，过滤出含Markush结构的样本，保存到本地

from datasets import load_dataset
import os

# ── 配置路径 ──────────────────────────────────────────────
SAVE_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/molparser_markush"   # 相对于code文件夹的路径
# D:\HuaweiMoveData\Users\David\Desktop\Moleditor\dataset\molparser_markush
# ─────────────────────────────────────────────────────────

def main():
    print("Step 1: 下载 sft_real 数据集...")
    
    # 下载sft_real子集（91.2k行，会自动缓存，第二次运行不会重新下载）
    ds = load_dataset(
        "UniParser/MolParser-7M",
        name="sft_real",         # 指定subset名称
        split="train"
    )
    
    print(f"下载完成，共 {len(ds)} 条样本")
    print(f"字段: {ds.column_names}")
    print(f"\n前3条样本预览:")
    for i in range(min(3, len(ds))):
        print(f"  [{i}] SMILES: {ds[i]['SMILES'][:80]}...")  # 只打印前80字符

    # ── 过滤含Markush结构的样本 ────────────────────────────
    # 含<a>标签 = 有R-group替换点（我们需要的）
    print("\nStep 2: 过滤含Markush结构的样本 (<a>标签)...")
    
    markush_ds = ds.filter(
        lambda x: "<a>" in x["SMILES"],
        desc="过滤中"
    )
    
    print(f"过滤完成: {len(ds)} → {len(markush_ds)} 条含Markush样本")
    print(f"占比: {len(markush_ds)/len(ds)*100:.1f}%")
    
    # ── 打印几条看看长什么样 ───────────────────────────────
    print("\nMarkush样本预览:")
    for i in range(min(5, len(markush_ds))):
        print(f"  [{i}] {markush_ds[i]['SMILES']}")
    
    # ── 保存到本地 ─────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    markush_ds.save_to_disk(SAVE_DIR)
    print(f"\n已保存到: {SAVE_DIR}")
    print("完成！")

if __name__ == "__main__":
    main()