# MolEdit Benchmark

Markush结构R基编辑的多模态benchmark，数据来自MolParser-7M专利分子 + ChEMBL SAR数据。目标投NeurIPS。

---

## 项目概览

测试模型对Markush结构（专利药物骨架）进行R基替换编辑和性质推理的理解能力，分为VLM track（图像输入）和LLM track（SMILES文字输入）两个track，共16种测试配置。

---

## Benchmark设计

### Split设计

| Split | 题数 | 干扰项 | Instruction风格 | 数据来源 |
|---|---|---|---|---|
| Easy Basic | 940 | 跨骨架 | 直接（"Replace R1 with fluorine"） | MolParser |
| Easy Advanced | ~663 | 同骨架SAR | 直接（"Which has highest pIC50?"） | ChEMBL SAR |
| Hard Basic | 940 | 同骨架 | 描述性（"Replace R with a halogenated group"） | MolParser |
| Hard Advanced | ~663 | 同骨架SAR+最难干扰 | 描述性（"Which compound most likely to show efficacy..."） | ChEMBL SAR |

### Advanced题性质分布（~44% bio + ~28% indication + ~24% RDKit）

固定数量：bio=290，indication=186，RDKit=158（合计≈634，RDKit占比≈25%）

**bio_activity（生物活性）：**
- Input：Markush骨架图；Options：同一骨架不同R基的4个完整分子
- 数据来自20b生成的290个SAR簇（12个靶点：EGFR/VEGFR2/CDK2/HDAC1/DopamineD2/AdenosineA2a/COX2/BTK/JAK2/ALK/PARP1/BRAF）
- 正确答案：pIC50最高的变体；Hard模式干扰项选pIC50最接近correct的3个（更难区分）
- 模型必须理解R基变化对活性的影响，不能靠骨架比较

**indication_mechanism（适应症/作用机制，v6新设计）：**
- Input：骨架图；Options：4个R基片段图（而非完整分子图）
- Question：问"哪个R基的替换结果是被批准用于XX的药物"
- 模型必须先理解R基→完整分子的映射，再判断indication，不能靠"哪个更像药物"走捷径
- 正确答案选法：找该簇中uniqueness最强的term（其他变体没有该indication/mechanism的变体）
- 优先用SAR indication簇（同骨架药物，18簇）；不足时fallback到"4个都是approved drugs"方案（186条中大部分走fallback）

**RDKit理化性质（logP/HBD/HBA/TPSA/rot_bonds/mol_weight）：**
- Easy：直接问（"Which substitution results in the highest logP value?"）
- Hard：描述性问（"Which substitution produces the compound most likely to accumulate in lipid membranes?"）

### None of Above设计（仅Basic题，占~20%）

```
Basic题 100%
├── ~80%  正常题（正确答案在A/B/C/D中某一项）
└── ~20%  D = "None of the above"
    ├── ~5%   占位D：valid instruction，D只是占位，正确答案在A/B/C
    └── ~15%  正确答案是D：
              ├── ~5%  TYPE B：R基位点不存在
              │         instruction指定的R基位点在骨架中不存在
              │         （如骨架只有R1，instruction说"Replace R2 with fluorine"）
              │         模型必须看懂骨架图才能识别位点不存在 → 答案D
              └── ~12% TYPE C：Subtle Wrong Answer
                          instruction合理，但所有选项都是错误的替换结果
                          → 答案D
```

字段：`is_none_of_above`（bool）、`nova_answer_is_d`（bool）、`nova_type`（"type_b"/"type_c"/"valid_d"）

**TYPE B设计说明：** 旧版TYPE B是"键型不兼容"（instruction化学上不合理），改为"R基位点不存在"后，题目直接考察模型识别骨架R基位点的能力——模型必须读懂骨架结构图（或SMILES）才能发现R基不存在，而不是靠化学键知识猜答案。对img输入特别有效，因为必须真正看懂图。

### Easy vs Hard的差异化维度

| 维度 | Easy | Hard |
|---|---|---|
| 干扰项（Basic） | 跨骨架 | 同骨架 |
| Instruction风格（Basic） | 直接 | 描述性 |
| 干扰项（Advanced Bio） | 随机同骨架3个 | pIC50最接近correct的3个 |
| 问题风格（Advanced） | 直接命名性质/indication | 描述性表达（不直说性质名/疾病名） |

### 16种测试配置

**VLM track（来自vqa_dataset/）：**
- `img_img`：输入图片 + 选项图片 × 4 splits = 4套
- `img_smi`：输入图片 + 选项SMILES × 4 splits = 4套
- `smi_img`：输入SMILES + 选项图片 × 4 splits = 4套

**LLM track（来自llm_vqa_dataset/）：**
- `smi_smi`：输入SMILES + 选项SMILES × 4 splits = 4套

---

## 数据路径

### 本地（Windows）

```
D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/
  new/
    edit_dataset/              ← 56,500条编辑数据（脚本25输出）
    vqa_dataset/               ← VLM VQA（脚本26输出）
      easy_basic/ easy_advanced/ hard_basic/ hard_advanced/
    llm_vqa_dataset/           ← LLM VQA（脚本27输出）
      easy_basic/ easy_advanced/ hard_basic/ hard_advanced/
    chembl_sar_scaffold/       ← SAR数据（脚本20b输出）
      chembl_sar_scaffold.jsonl  ← 308个SAR簇（290 bio + 18 indication）
      images/scaffold/           ← 骨架图（含*）
      images/molecule/           ← 完整分子图
      images/rgroup/             ← R基片段图（v6新增，indication题option用）
      cache/                     ← ChEMBL API缓存（断点续跑用）

  chembl_reverse_markush/
    chembl_reverse_markush.jsonl   ← 6,936条原ChEMBL数据（indication fallback + RDKit题用）

eval_results:
  D:/HuaweiMoveData/Users/David/Desktop/Moleditor/code/result/
    new/      ← 旧版评测结果（基于旧数据，仅供参考）
    update/   ← 新版评测结果（当前使用）
      {model_name}/
        {mode}_{split}.jsonl   ← 每题结果（断点续跑）
        {mode}.json            ← 每个mode的汇总（含4个split详细准确率+nova分类统计）
```

### 集群（P集群）

```
/mnt/petrelfs/wangxin1/image_edit/MolEdit/
  分区: DataFrontier_Share
  用户: wangxin1
  Conda环境: ai_scientist（transformers 4.57.3, torch 2.4.0+cu121）
  模型路径: /mnt/petrelfs/share/yejinhui/Models/Pretrained_models/
```

---

## 脚本说明

| 脚本 | 状态 | 说明 |
|---|---|---|
| `25_build_new_edit_dataset.py` | ✅ 已完成，无需重跑 | 生成56,500条编辑数据 |
| `20b_chembl_sar_scaffold.py` | ✅ 已完成，无需重跑 | 从ChEMBL API下载SAR数据，Butina聚类+MCS，生成同骨架R基变体簇；v6新增渲染R基片段图（images/rgroup/） |
| `26_build_new_vqa.py` | ✅ 最新版（v6） | VLM VQA构建；TYPE B改为"R基位点不存在"；indication改为骨架+R基片段选项；RDKit固定158题使占比≈25%；TYPE C比例提至12% |
| `27_build_llm_vqa.py` | ✅ 最新版 | LLM VQA构建（从26输出转换，去图片保留SMILES，透传nova_type/target字段） |
| `28_eval_vqa.py` | ✅ 最新版 | 主评测脚本；16种配置、并发、断点续跑；indication题prompt区分R基/完整分子描述；输出4个mode.json |

---

## 评测运行方式

### 重新生成数据

```bash
# 先删除旧数据：dataset/new/vqa_dataset/ 和 llm_vqa_dataset/ 下4个文件夹
# 注意：chembl_sar_scaffold/cache/ 可以保留（节省API调用时间）
python 26_build_new_vqa.py
python 27_build_llm_vqa.py
```

### API模型评测（本地）

```bash
pip install openai datasets rdkit requests

python 28_eval_vqa.py --model claude-sonnet-4-6
python 28_eval_vqa.py --model claude-sonnet-4-6 gpt-4.1-2025-04-14 gemini-2.5-flash
python 28_eval_vqa.py --model claude-sonnet-4-6 --mode smi_smi --max_samples 10
```

API配置（28_eval_vqa.py开头）：
```python
os.environ.setdefault("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
RESULTS_DIR = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/code/result/update"
```

### 本地模型（集群）

```bash
sbatch -p DataFrontier_Share --gres=gpu:1 --quotatype=spot --mem=32G --time=6:00:00 \
  --output=/mnt/petrelfs/wangxin1/image_edit/MolEdit/logs/模型名_%j.log \
  --wrap="source /mnt/petrelfs/wangxin1/miniconda3/etc/profile.d/conda.sh \
    && conda activate ai_scientist \
    && cd /mnt/petrelfs/wangxin1/image_edit/MolEdit/code \
    && python 28_eval_vqa.py --model Qwen2.5-VL-7B-Instruct --mode img_img"
```

---

## 重要发现与注意事项

### img输入在Advanced题上虚高问题

**现象：** `img_smi`/`img_img`在Advanced题上比`smi_smi`/`smi_img`高约30%。

**原因：** E-SMILES是MolParser自定义格式（非标准CXSMILES），大模型预训练数据中几乎没有，导致`smi_*`模式下模型无法理解Markush骨架，分数偏低；`img_*`模式图片直观，分数虚高。

**可信配置：** `smi_smi`和`smi_img`结果可信。

**待做控制实验：**
1. Wrong Scaffold Control：把输入骨架图换成无关骨架，验证img虚高
2. Property Counterfactual：把question改成反向（问活性最低），验证模型是否真正理解性质

### TYPE B子集解读

新版TYPE B（R基位点不存在）对不同输入模式的预期难度不同：
- `smi_*`模式：模型可以从SMILES文本中数出R基位点，相对简单
- `img_*`模式：模型必须真正看懂骨架结构图才能识别位点，更有挑战性

因此TYPE B准确率的`img vs smi`对比是一个有意义的分析维度，可以量化模型"真正读懂骨架图"的能力。

### Advanced题旧版设计缺陷（已修复）

旧版Advanced选项来自不同骨架的完整分子，模型只需比较4个独立分子的性质，不需要理解Markush/R基。新版改为同骨架SAR数据，真正考察R基理解能力。

旧版Indication题correct是"有indication的药物"、distractor是"无indication的分子"，模型只需识别"哪个是药物"即可，不考察化学知识。新版改为4个都是approved drugs，且选项改为R基片段图。

---

## 新版评测结果（待填）

> 重跑26/27后更新此处。

---

## 旧版评测结果（基于旧版数据，仅供参考）

> ⚠️ 以下结果基于旧版数据集（Advanced题选项为跨骨架完整分子，TYPE B为键型不兼容），新版数据出来后需重测。

**claude-sonnet-4-6：**

| split | smi_smi | smi_img | img_smi | img_img |
|---|---|---|---|---|
| easy_basic | 0.9787 | 0.9649 | 0.9745 | 0.9780 |
| easy_advanced | 0.5791 | 0.5818 | 0.8874 | 0.8150 |
| hard_basic | 0.6617 | 0.6426 | 0.7638 | 0.7000 |
| hard_advanced | 0.5416 | 0.5456 | 0.8794 | 0.8298 |

**gpt-4.1-2025-04-14：**

| split | smi_smi | smi_img | img_smi | img_img |
|---|---|---|---|---|
| easy_basic | 0.9840 | 0.9809 | 0.9851 | 0.9830 |
| easy_advanced | 0.5697 | 0.6488 | 0.9464 | 0.8646 |
| hard_basic | 0.7064 | 0.6872 | 0.7202 | 0.6872 |
| hard_advanced | 0.5322 | 0.5764 | 0.9531 | 0.8512 |

---

## 下一步任务

1. 重新评测：claude/gpt/gemini/qwen全量（result/update/）
2. 集群评测：Qwen2.5-VL-7B/72B，Qwen3-VL-4B
3. 控制实验：Wrong Scaffold + Property Counterfactual
4. 错误分析：按靶点、按R基类型分析hard_advanced失败模式
5. 写paper