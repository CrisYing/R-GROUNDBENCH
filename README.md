# R-GROUNDBENCH

Markush结构R基编辑与性质推理的多模态benchmark，数据来自MolParser-7M专利分子 + ChEMBL SAR数据。

---

## 项目概览

测试LLM/VLM对Markush结构（专利药物骨架）进行R基替换编辑和性质推理的能力。包含两种题型：**选择题（VQA）** 和 **填空题（Generation QA）**。

---

## Benchmark设计

### 一、选择题（VQA）

给定Markush骨架 + 自然语言instruction，从4个候选分子中选出正确答案。

#### Split设计（3难度 × 2题型 = 6个split）

| Split | 题数 | 干扰项 | Instruction风格 | NOTA |
|---|---|---|---|---|
| easy_basic | 940 | 跨骨架 | 直接（"Replace R1 with methyl"） | ✓ 20% |
| medium_basic | 940 | 同骨架 | 直接 | ✓ 20% |
| hard_basic | 940 | 同骨架（R基Tanimoto相似度最高） | 描述性（"Replace R1 with a halogenated ring"） | ✗ |
| easy_advanced | 634 | 跨骨架/混合 | 直接 | ✗ |
| medium_advanced | 634 | 同骨架 | 直接 | ✗ |
| hard_advanced | 634 | 同骨架 | 描述性 | ✗ |
| **合计** | **4,722** | | | |

**难度递进逻辑：**
- Easy → Medium：仅干扰项变化（跨骨架→同骨架），instruction不变，单独量化同骨架干扰的影响
- Medium → Hard：仅instruction变化（直接→描述性），干扰项不变，单独量化描述性instruction的影响
- Hard额外：干扰项按R基ECFP4 Tanimoto相似度最高的3个选取，去掉NOTA题

#### Advanced题性质分布（每个Advanced split）

| 性质类型 | 题数 | 占比 | 数据来源 |
|---|---|---|---|
| bio_activity（生物活性pIC50） | 285 | 45% | ChEMBL SAR同骨架簇 |
| indication_mechanism（适应症/机制） | 190 | 30% | ChEMBL approved drugs |
| RDKit理化性质（logP/MW/TPSA/HBD/HBA/RotBonds） | 159 | 25% | RDKit计算 |

#### None of Above（NOTA）设计（仅Easy/Medium Basic，占20%）

```
Basic题 100%
├── 80%  正常题（正确答案在A/B/C/D中某一项）
└── 20%  D = "None of the above"
    ├── 5%   TYPE valid_d：valid instruction，正确答案在A/B/C，D只是占位
    ├── 5%   TYPE B：instruction引用了骨架中不存在的R基位点 → 正确答案D
    └── 10%  TYPE C：Subtle Wrong Answer（所有选项结构都有细微错误）→ 正确答案D
```

字段：`is_none_of_above`（bool）、`nova_answer_is_d`（bool）、`nova_type`（"type_b"/"type_c"/"valid_d"）

**Hard Basic无NOTA**：实验发现Hard中NOTA准确率异常偏高，说明模型在利用NOTA作为捷径，故去掉。

#### 4种模态配置

| 模式 | 骨架输入 | 选项格式 | 适用模型 |
|---|---|---|---|
| img_img | Markush图片 | 分子图片 | VLM |
| img_smi | Markush图片 | SMILES字符串 | VLM |
| smi_img | E-SMILES文字 | 分子图片 | VLM |
| smi_smi | E-SMILES文字 | SMILES字符串 | VLM + LLM |

**注意：** E-SMILES是MolParser定义的专有格式（非标准SMILES），RDKit无法直接解析。`smi_*`模式下模型几乎没见过这种格式，分数偏低；`img_*`模式下Advanced分数虚高（模型可能靠记忆而非视觉推理）。可信配置：`smi_smi`和`smi_img`。

---

### 二、填空题（Generation QA）

给定Markush骨架 + instruction，直接生成编辑后分子的SMILES（无候选选项）。

#### Split设计（2个split）

| Split | 题数 | Instruction风格 |
|---|---|---|
| easy | 500 | 直接命名（"Replace R1 with a 4-chlorophenyl group"） |
| hard | 500 | 描述性（"Replace R1 with a halogenated six-membered aromatic ring"） |

Easy vs Hard仅instruction风格不同，无干扰项设计（填空题无候选）。

#### 评测指标（4层）

| 指标 | 说明 |
|---|---|
| Validity Rate | RDKit能否解析模型输出的SMILES |
| Exact Match | canonical SMILES是否与ground truth完全一致 |
| Avg Tanimoto | ECFP4 Tanimoto相似度（0~1，连续分） |
| Scaffold Match | MCS覆盖≥80%骨架原子（验证骨架未被改动） |

#### 模态配置

| 模式 | 输入 | 适用模型 |
|---|---|---|
| smi | E-SMILES文字 + instruction | LLM + VLM |
| img | 骨架图片 + instruction | VLM |

---

## 数据路径

### 本地（Windows）

```
D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/
  new/
    edit_dataset/              ← 编辑数据（脚本25输出）
    vqa_dataset/               ← 选择题数据集（脚本26输出）
      easy_basic/
      medium_basic/
      hard_basic/
      easy_advanced/
      medium_advanced/
      hard_advanced/
    generation_qa/             ← 填空题数据集（脚本29输出）
      easy/
      hard/
    chembl_sar_scaffold/       ← SAR数据（脚本20b输出）
      chembl_sar_scaffold.jsonl
      images/scaffold/
      images/molecule/
      cache/

  chembl_reverse_markush/
    chembl_reverse_markush.jsonl   ← ChEMBL数据（indication fallback + RDKit题用）

eval_results:
  D:/HuaweiMoveData/Users/David/Desktop/Moleditor/code/result/
    update/   ← 选择题评测结果
      {model_name}/
        {mode}_{split}.jsonl   ← 每题结果（断点续跑）
        {mode}.json            ← 每个mode的汇总
    generation/  ← 填空题评测结果
      {model_name}/
        {mode}_{split}.jsonl
        {mode}_generation_summary.json
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
| `25_build_new_edit_dataset.py` | ✅ 已完成，无需重跑 | 生成编辑数据（含direct + descriptive instruction） |
| `20b_chembl_sar_scaffold.py` | ✅ 已完成，无需重跑 | ChEMBL SAR数据下载、聚类、R基变体生成 |
| `26_build_new_vqa.py` | ✅ v7最新版 | 选择题构建（6个split，Easy/Medium/Hard × Basic/Advanced） |
| `28_eval_vqa.py` | ✅ v7最新版 | 选择题评测（支持6个split，4种模态，断点续跑） |
| `29_build_generation_qa.py` | ✅ v2最新版 | 填空题数据集构建（easy/hard两个split） |
| `30_eval_generation.py` | ✅ v1最新版 | 填空题评测（Validity/ExactMatch/Tanimoto/ScaffoldMatch） |

---

## 运行方式

### 生成选择题数据

```bash
# vqa_dataset/下只保留easy_basic和easy_advanced，删除其余文件夹
python 26_build_new_vqa.py
```

### 生成填空题数据

```bash
python 29_build_generation_qa.py
```

### 选择题评测

```bash
# 单模型
python 28_eval_vqa.py --model claude-sonnet-4-6

# 多模型
python 28_eval_vqa.py --model claude-sonnet-4-6 gpt-4.1-2025-04-14 gemini-2.5-flash-nothinking

# 指定split（只测新增的medium和重跑的hard）
python 28_eval_vqa.py --model claude-sonnet-4-6 --split medium_basic medium_advanced hard_basic hard_advanced

# 指定模式
python 28_eval_vqa.py --model claude-sonnet-4-6 --mode smi_smi

# 小样本测试
python 28_eval_vqa.py --model claude-sonnet-4-6 --max_samples 10
```

### 填空题评测

```bash
# VLM（两种模式）
python 30_eval_generation.py --model claude-sonnet-4-6 --mode smi
python 30_eval_generation.py --model claude-sonnet-4-6 --mode img

# LLM（只跑smi）
python 30_eval_generation.py --model deepseek-r1-0528 --mode smi
```

### 集群评测（Qwen2.5-VL）

```bash
sbatch -p DataFrontier_Share --gres=gpu:1 --quotatype=spot --mem=32G --time=6:00:00 \
  --output=/mnt/petrelfs/wangxin1/image_edit/MolEdit/logs/模型名_%j.log \
  --wrap="source /mnt/petrelfs/wangxin1/miniconda3/etc/profile.d/conda.sh \
    && conda activate ai_scientist \
    && cd /mnt/petrelfs/wangxin1/image_edit/MolEdit/code \
    && python 28_eval_vqa.py --model Qwen2.5-VL-72B-Instruct --mode img_img"
```

---

## API配置

```python
# 28_eval_vqa.py / 30_eval_generation.py 开头配置
os.environ.setdefault("OPENAI_API_KEY", "your_key")
os.environ.setdefault("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")

# 已测试可用模型
# VLM: claude-sonnet-4-6, gpt-4.1-2025-04-14, gemini-2.5-flash-nothinking
#      qwen-vl-max-latest, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct
# LLM: deepseek-v3-0324, deepseek-r1-0528, llama-3.3-70b-instruct, qwq-32b
```

---

## HuggingFace数据集

| Dataset | 内容 | 链接 |
|---|---|---|
| `Crisying/rgroup-vqa-benchmark` | 选择题，6个split，4722题 | VQA track |
| `Crisying/rgroup-generation-benchmark` | 填空题，2个split，1000题 | Generation track |

---

## 重要技术细节

### E-SMILES格式

E-SMILES是MolParser定义的专有格式，在标准SMILES后用`<sep>`分隔R基注解：

```
N1=CC(*)=C(*)N=C1*<sep><a>3:R[1]</a><a>5:R[2]</a><a>8:X</a>
```

RDKit无法直接解析，需要先strip `<a>...</a>` 标签。

### img模式虚高问题

`img_smi`/`img_img`在Advanced题上比`smi_smi`/`smi_img`高约30%，原因是E-SMILES稀有格式导致smi模式分数异常低，不代表真正视觉理解能力。可信配置：`smi_smi`和`smi_img`。

### Thinking模型

DeepSeek-R1-0528等thinking模型需在`28_eval_vqa.py`中加入`THINKING_MODELS`集合，并把`REQUEST_TIMEOUT`从120改为300。

---

## 当前状态与下一步

**数据：**
- ✅ easy_basic / easy_advanced：已生成并评测完成
- ✅ medium_basic / medium_advanced / hard_basic / hard_advanced：已生成（v7新版）
- ✅ generation_qa easy/hard：已生成

**下一步任务（按优先级）：**
1. 全量评测Medium + Hard split（claude/gpt/gemini/qwen/deepseek）
2. Qwen2.5-VL-7B/72B集群评测
3. 填空题评测（30_eval_generation.py）
4. 控制实验：Wrong Scaffold + Property Counterfactual
5. 写paper