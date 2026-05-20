# 文件位置: Moleditor/code/20b_chembl_sar_scaffold.py
#
# 作用: 从ChEMBL下载SAR数据，用Butina聚类+MCS找同骨架簇，
#       输出"同骨架不同R基+pIC50"的数据，供26重写Advanced题用
#
# 与20的核心区别:
#   20: 每个分子独立做反向构造，每条记录自成一体，骨架互不相关
#   20b: 对同一靶点的所有分子做聚类，找出"同骨架家族"，
#        同一簇内的分子共享骨架，只有R基不同 → 才能出真正的SAR题
#
# 数据流:
#   ChEMBL API → 每个靶点的所有pIC50分子
#   → Butina聚类（Tanimoto >= 0.4）→ 同类分子簇
#   → 簇内MCS → 共同骨架（含*位点）
#   → 每个分子提取R基SMILES（full_mol - core）
#   → 过滤：簇内>=4个变体，pIC50 spread>=1.0，R基有差异
#   → 渲染图像（骨架图 + 完整分子图）
#   → 输出 chembl_sar_scaffold.jsonl
#
# 同时处理approved drugs，用同样逻辑找同骨架indication变体
#
# 用法:
#   python 20b_chembl_sar_scaffold.py --mode explore   # 测试单靶点
#   python 20b_chembl_sar_scaffold.py --mode full      # 完整运行
#   python 20b_chembl_sar_scaffold.py --mode debug --target CHEMBL279
#
# 输出: D:/HuaweiMoveData/.../dataset/new/chembl_sar_scaffold/
#   chembl_sar_scaffold.jsonl   ← 主数据文件（簇级别，每条=一个骨架家族）
#   images/scaffold/            ← 骨架图（含*）
#   images/molecule/            ← 完整分子图
#   cache/                      ← API缓存，断点续跑
# ─────────────────────────────────────────────────────────────────────────────

import os, json, time, random, argparse
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import (Draw, Descriptors, Crippen, rdMolDescriptors,
                         Lipinski, AllChem, DataStructs)
from rdkit.Chem.rdFMCS import FindMCS, MCSParameters
import requests

RDLogger.DisableLog('rdApp.*')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── 路径 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "D:/HuaweiMoveData/Users/David/Desktop/Moleditor/dataset/new/chembl_sar_scaffold"
# 集群:
# OUTPUT_DIR = "/mnt/petrelfs/wangxin1/image_edit/MolEdit/dataset/new/chembl_sar_scaffold"

RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

IMG_SIZE     = (300, 300)

# ── ChEMBL API ────────────────────────────────────────────────────────────────
CHEMBL_API_BASE   = "https://www.ebi.ac.uk/chembl/api/data"
REQUEST_DELAY     = 0.5
MAX_RETRIES       = 3
MAX_ACTS_PER_TARGET = 2000   # 每个靶点最多拉2000条活性记录（ChEMBL有丰富数据）

# ── 聚类 & MCS 参数 ───────────────────────────────────────────────────────────
BUTINA_CUTOFF       = 0.35   # 略微调紧，让簇内分子更相似（原0.4）
MCS_TIMEOUT         = 10     # MCS超时（秒）
MCS_MIN_ATOMS       = 8      # MCS最小原子数（骨架太小无意义）
MIN_RGROUP_ATOMS    = 1      # R基最少原子数（排除H）
MAX_RGROUP_ATOMS    = 15     # R基最多原子数（太大的基团排除）
MIN_VARIANTS        = 4      # 每个簇最少变体数（够出4选1题）
MIN_PICSA50_SPREAD  = 0.8    # pIC50最小spread
MAX_PICSA50_SPREAD  = 3.0    # pIC50最大spread（从5.0降到3.0，防止差异太明显被猜）
MIN_MOL_WEIGHT      = 180    # 完整分子最小MW（排除太小的分子）
MAX_RGROUP_POSITIONS = 2     # 骨架最多允许几个*位点（超过则只保留单R基变体）

# ── 靶点列表（优先选ChEMBL数据丰富的激酶/GPCR类）────────────────────────────
SAR_TARGETS = [
    # 靶点CHEMBL ID    简称              中文名
    ("CHEMBL279",  "EGFR",        "表皮生长因子受体"),
    ("CHEMBL203",  "VEGFR2",      "血管内皮生长因子受体2"),
    ("CHEMBL325",  "CDK2",        "细胞周期蛋白依赖激酶2"),
    ("CHEMBL2842", "HDAC1",       "组蛋白去乙酰化酶1"),
    ("CHEMBL217",  "DopamineD2",  "多巴胺D2受体"),
    ("CHEMBL251",  "AdenosineA2a","腺苷A2a受体"),
    ("CHEMBL264",  "COX2",        "环氧合酶2"),
    ("CHEMBL2179", "BTK",         "布鲁顿酪氨酸激酶"),
    ("CHEMBL1824", "JAK2",        "Janus激酶2"),
    ("CHEMBL2971", "ALK",         "间变性淋巴瘤激酶"),
    ("CHEMBL1163125","PARP1",     "聚ADP-核糖聚合酶1"),
    ("CHEMBL301",  "BRAF",        "B-Raf激酶"),
]

# ── Indication相关靶点（approved drugs，用于indication题）────────────────────
MAX_APPROVED_DRUGS = 2000    # 拉取批准药物数量


# ══════════════════════════════════════════════════════════════════════════════
# 【模块1】ChEMBL REST API
# ══════════════════════════════════════════════════════════════════════════════

def _get(url, params=None, timeout=30):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (400, 404):
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 2)
    return None


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except: return None


def fetch_activities_for_target(target_id, max_records=MAX_ACTS_PER_TARGET):
    """拉取某靶点所有有pchembl_value的活性记录"""
    url    = f"{CHEMBL_API_BASE}/activity.json"
    params = {
        "target_chembl_id":      target_id,
        "pchembl_value__isnull": False,
        "limit": 100, "offset": 0,
    }
    records = []
    while len(records) < max_records:
        data = _get(url, params=params)
        if not data: break
        batch = data.get("activities", [])
        if not batch: break
        for a in batch:
            mol_id  = a.get("molecule_chembl_id","")
            pchembl = a.get("pchembl_value")
            stype   = a.get("standard_type","")
            # 只保留常见定量活性类型
            if mol_id and pchembl and stype in ("IC50","Ki","Kd","EC50","GI50","Potency"):
                try:
                    records.append({
                        "molecule_chembl_id": mol_id,
                        "pchembl_value": float(pchembl),
                        "activity_type": stype,
                        "assay_description": a.get("assay_description","")[:100],
                    })
                except: pass
        if not data.get("page_meta",{}).get("next") or len(records) >= max_records:
            break
        params["offset"] += 100
        time.sleep(REQUEST_DELAY)
    return records


def fetch_smiles_batch(mol_ids):
    """批量获取分子SMILES，无数量限制，返回{mol_id: canonical_smiles}"""
    result = {}
    for i in range(0, len(mol_ids), 50):
        batch = mol_ids[i:i+50]
        data  = _get(f"{CHEMBL_API_BASE}/molecule.json", params={
            "molecule_chembl_id__in": ",".join(batch),
            "limit": 50,
        })
        if data:
            for mol in data.get("molecules", []):
                mid  = mol.get("molecule_chembl_id","")
                smi  = (mol.get("molecule_structures") or {}).get("canonical_smiles","")
                if mid and smi:
                    canon = canonical_smiles(smi)
                    if canon: result[mid] = canon
        time.sleep(REQUEST_DELAY)
    return result


def fetch_approved_drugs(cache_path, max_drugs=MAX_APPROVED_DRUGS):
    """拉取max_phase=4的批准药物，附带indication和mechanism"""
    if os.path.exists(cache_path):
        with open(cache_path) as f: return json.load(f)

    print(f"  拉取批准药物（max_phase=4，目标{max_drugs}个）...")
    drug_smiles = {}
    offset = 0
    # 新版ChEMBL API用 max_phase__gte=4 或 max_phase=4 均可，加fallback
    for phase_param in [{"max_phase__gte": 4}, {"max_phase": 4}, {"max_phase__exact": 4}]:
        if drug_smiles: break
        offset = 0
        while len(drug_smiles) < max_drugs:
            params = {"molecule_type": "Small molecule", "limit": 100, "offset": offset}
            params.update(phase_param)
            data = _get(f"{CHEMBL_API_BASE}/molecule.json", params=params)
            if not data: break
            batch = data.get("molecules",[])
            if not batch: break
            for mol in batch:
                mid = mol.get("molecule_chembl_id","")
                smi = (mol.get("molecule_structures") or {}).get("canonical_smiles","")
                if mid and smi:
                    canon = canonical_smiles(smi)
                    if canon: drug_smiles[mid] = canon
            if not data.get("page_meta",{}).get("next") or len(drug_smiles) >= max_drugs:
                break
            offset += 100
            time.sleep(REQUEST_DELAY)
        if drug_smiles:
            print(f"  API参数 {phase_param} 有效，获得 {len(drug_smiles)} 个药物SMILES")

    print(f"  获得药物SMILES: {len(drug_smiles)} 个，拉取indication/mechanism...")
    mol_ids = list(drug_smiles.keys())
    ind_map  = defaultdict(list)
    mech_map = defaultdict(list)

    for i in range(0, len(mol_ids), 50):
        batch   = mol_ids[i:i+50]
        ids_str = ",".join(batch)
        d = _get(f"{CHEMBL_API_BASE}/drug_indication.json", params={
            "molecule_chembl_id__in": ids_str, "limit": 200,
        })
        if d:
            for rec in d.get("drug_indications",[]):
                mid  = rec.get("molecule_chembl_id","")
                term = rec.get("efo_term","")
                if mid and term and "unknown" not in term.lower():
                    ind_map[mid].append(term)
        time.sleep(REQUEST_DELAY)

        d = _get(f"{CHEMBL_API_BASE}/mechanism.json", params={
            "molecule_chembl_id__in": ids_str, "limit": 200,
        })
        if d:
            for rec in d.get("mechanisms",[]):
                mid = rec.get("molecule_chembl_id","")
                moa = rec.get("mechanism_of_action","")
                if mid and moa and "unknown" not in moa.lower():
                    mech_map[mid].append(moa)
        time.sleep(REQUEST_DELAY)

    result = []
    for mid, smi in drug_smiles.items():
        inds  = list(set(ind_map[mid]))
        mechs = list(set(mech_map[mid]))
        if not inds and not mechs: continue
        # 过滤MW太小
        mw = mol_weight(smi)
        if mw and mw < MIN_MOL_WEIGHT: continue
        result.append({
            "chembl_id": mid, "smiles": smi,
            "indications": inds, "mechanisms": mechs,
            "is_approved_drug": True,
        })

    print(f"  有indication/mechanism的药物: {len(result)} 个")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f: json.dump(result, f, ensure_ascii=False)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 【模块2】RDKit工具
# ══════════════════════════════════════════════════════════════════════════════

def mol_weight(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.CalcExactMolWt(mol) if mol else None
    except: return None


def compute_properties(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return {
            "mol_weight": round(Descriptors.MolWt(mol), 2),
            "logP":       round(Crippen.MolLogP(mol), 3),
            "tpsa":       round(rdMolDescriptors.CalcTPSA(mol), 2),
            "hbd":        Lipinski.NumHDonors(mol),
            "hba":        Lipinski.NumHAcceptors(mol),
            "rot_bonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
    except: return None


def morgan_fp(smi, radius=2, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    except: return None


def butina_cluster(smiles_list, cutoff=BUTINA_CUTOFF):
    """
    Butina聚类，返回clusters列表，每个cluster是smiles_list的索引列表。
    cutoff是Tanimoto距离上限（distance = 1 - similarity）。
    兼容新旧版RDKit（ClusterData在新版移到rdSimDivPickers）。
    """
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)

    if len(fps) < 2:
        return [[i] for i in valid_idx]

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    # 兼容新旧RDKit
    cluster_res = None
    try:
        from rdkit.ML.Cluster import Butina
        cluster_res = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    except Exception:
        pass
    if cluster_res is None:
        try:
            from rdkit.Chem.rdSimDivPickers import ClusterData
            cluster_res = ClusterData(dists, n, cutoff, isDistData=True)
        except Exception:
            pass
    if cluster_res is None:
        cluster_res = _manual_butina(dists, n, cutoff)

    clusters = []
    for c in cluster_res:
        clusters.append([valid_idx[i] for i in c])
    return clusters


def _manual_butina(dists, n, cutoff):
    """手动实现Butina聚类（fallback）"""
    dist_matrix = [[0.0] * n for _ in range(n)]
    idx = 0
    for i in range(1, n):
        for j in range(i):
            dist_matrix[i][j] = dists[idx]
            dist_matrix[j][i] = dists[idx]
            idx += 1

    neighbors = []
    for i in range(n):
        nb = [j for j in range(n) if i != j and dist_matrix[i][j] <= cutoff]
        neighbors.append(nb)

    assigned = [False] * n
    clusters  = []
    order     = sorted(range(n), key=lambda i: len(neighbors[i]), reverse=True)
    for i in order:
        if assigned[i]: continue
        cluster = [i]
        assigned[i] = True
        for j in neighbors[i]:
            if not assigned[j]:
                cluster.append(j)
                assigned[j] = True
        clusters.append(tuple(cluster))
    return clusters


def find_mcs_scaffold(smiles_list, timeout=MCS_TIMEOUT):
    """
    在一组分子里找MCS（最大公共子结构），返回骨架SMILES（含*位点）。
    返回 (scaffold_smiles_with_wildcards, mcs_smarts) 或 None。
    """
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol: mols.append(mol)
    if len(mols) < 2: return None

    try:
        params = MCSParameters()
        params.Timeout           = timeout
        params.AtomCompareParameters.CompleteRingsOnly = True
        params.BondCompareParameters.CompleteRingsOnly = True

        result = FindMCS(mols, params)
        if result.canceled or result.numAtoms < MCS_MIN_ATOMS:
            return None

        mcs_smarts = result.smartsString
        mcs_mol    = Chem.MolFromSmarts(mcs_smarts)
        if mcs_mol is None: return None

        return mcs_smarts
    except: return None


def extract_rgroup(full_smi, mcs_smarts):
    """
    从完整分子中提取R基SMILES（full_mol减去MCS骨架之外的部分）。
    返回 (rgroup_smiles, attachment_smiles) 或 None。

    策略：
    1. 找MCS在full_mol中的匹配位置
    2. 骨架原子之外的连通片段就是R基
    3. 如果有多个非骨架片段（多个R基位点），分别提取
    """
    try:
        full_mol = Chem.MolFromSmiles(full_smi)
        mcs_mol  = Chem.MolFromSmarts(mcs_smarts)
        if full_mol is None or mcs_mol is None: return None

        matches = full_mol.GetSubstructMatches(mcs_mol)
        if not matches: return None

        # 取第一个匹配
        core_atoms = set(matches[0])
        all_atoms  = set(range(full_mol.GetNumAtoms()))
        rgroup_atoms = all_atoms - core_atoms

        if not rgroup_atoms: return None

        # 找R基和骨架的连接键（attachment bonds）
        attachment_points = []
        for bond in full_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in core_atoms) != (a2 in core_atoms):
                # 这是骨架-R基连接键
                rg_atom    = a2 if a1 in core_atoms else a1
                core_atom  = a1 if a1 in core_atoms else a2
                attachment_points.append((core_atom, rg_atom))

        if not attachment_points: return None

        # 提取R基片段（所有非骨架原子）
        rgroup_mol = Chem.RWMol(full_mol)
        # 用*替换骨架原子（attachment points）
        # 简化：直接用MolFragmentToSmiles提取R基片段
        # BFS收集每个attachment point对应的R基连通分量
        rgroups = []
        visited_rg = set()

        for core_atom, rg_atom in attachment_points:
            if rg_atom in visited_rg: continue
            # BFS收集此R基连通分量
            from collections import deque
            frag = set()
            q = deque([rg_atom])
            while q:
                cur = q.popleft()
                if cur in frag or cur in core_atoms: continue
                frag.add(cur)
                for nb in full_mol.GetAtomWithIdx(cur).GetNeighbors():
                    nidx = nb.GetIdx()
                    if nidx not in frag and nidx not in core_atoms:
                        q.append(nidx)
            visited_rg.update(frag)

            if not frag: continue

            # 检查R基原子数
            if len(frag) < MIN_RGROUP_ATOMS or len(frag) > MAX_RGROUP_ATOMS:
                continue

            try:
                rg_smi = Chem.MolFragmentToSmiles(
                    full_mol, tuple(sorted(frag)),
                    atomSymbols=None, canonical=True
                )
                if rg_smi:
                    canon_rg = canonical_smiles(rg_smi)
                    if canon_rg:
                        rgroups.append(canon_rg)
            except: continue

        if not rgroups: return None
        # 返回所有R基（通常1个，偶尔多个）
        return rgroups

    except: return None


def build_scaffold_smiles_with_wildcard(full_smi, mcs_smarts):
    """
    把full_mol中非MCS的部分替换为*，构造含*的骨架SMILES。
    用于渲染Markush骨架图。
    返回 scaffold_smiles (含*) 或 None。
    """
    try:
        full_mol = Chem.MolFromSmiles(full_smi)
        mcs_mol  = Chem.MolFromSmarts(mcs_smarts)
        if full_mol is None or mcs_mol is None: return None

        matches = full_mol.GetSubstructMatches(mcs_mol)
        if not matches: return None
        core_atoms = set(matches[0])

        rw = Chem.RWMol(full_mol)

        # 找所有R基-骨架连接点
        attach_bonds = []
        for bond in full_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in core_atoms) != (a2 in core_atoms):
                rg_atom   = a2 if a1 in core_atoms else a1
                core_atom = a1 if a1 in core_atoms else a2
                attach_bonds.append((core_atom, rg_atom))

        if not attach_bonds: return None

        # 对每个连接点：添加*，删除R基原子
        all_rg_atoms = set(range(full_mol.GetNumAtoms())) - core_atoms
        wc_atoms = []
        for core_atom, rg_atom in attach_bonds:
            wc_idx = rw.AddAtom(Chem.Atom(0))  # wildcard *
            rw.AddBond(core_atom, wc_idx, Chem.BondType.SINGLE)
            wc_atoms.append(wc_idx)

        # 删除R基原子（从大到小，避免index错位）
        for idx in sorted(all_rg_atoms, reverse=True):
            rw.RemoveAtom(idx)

        try: Chem.SanitizeMol(rw)
        except: pass

        smi = Chem.MolToSmiles(rw, canonical=True)
        return smi if smi and "*" in smi else None
    except: return None


def _fix_other_rgroups(scaffold_star_smi, other_rgroups, focus_pos, n_star):
    """
    把骨架中非focus_pos位置的*替换回具体R基SMILES，
    只保留focus_pos位置的*，让题目只比较一个R基位点。

    scaffold_star_smi: 含n_star个*的骨架SMILES
    other_rgroups:     长度为n_star-1的tuple，按顺序对应非focus_pos位置的R基
    focus_pos:         要保留*的位点索引（0-based，在原始n_star个*里的位置）
    返回只含1个*的骨架SMILES，或None（失败时）
    """
    try:
        result = scaffold_star_smi
        other_iter = iter(other_rgroups)
        new_smi = ""
        star_count = 0
        i = 0
        # 逐字符扫描，遇到*时决定替换还是保留
        while i < len(result):
            if result[i] == "*":
                if star_count == focus_pos:
                    new_smi += "*"     # 保留这个位点
                else:
                    rg = next(other_iter, None)
                    if rg is None:
                        return None
                    # 把R基SMILES插入（加括号避免歧义）
                    new_smi += f"({rg})"
                star_count += 1
            else:
                new_smi += result[i]
            i += 1

        # 验证结果合法且只含1个*
        canon = canonical_smiles(new_smi)
        if canon and canon.count("*") == 1:
            return canon
        # canonical_smiles可能失败（含*的SMILES），直接返回new_smi
        if new_smi.count("*") == 1:
            return new_smi
        return None
    except:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 【模块3】图像渲染
# ══════════════════════════════════════════════════════════════════════════════

def render_molecule(smi, size=IMG_SIZE):
    """渲染完整分子图像"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Draw.MolToImage(mol, size=size)
    except: return None


def render_scaffold(scaffold_smi_with_star, size=IMG_SIZE):
    """渲染含*的Markush骨架图像"""
    try:
        mol = Chem.MolFromSmiles(scaffold_smi_with_star, sanitize=False)
        if mol is None: return None
        try: Chem.SanitizeMol(mol, catchErrors=True)
        except: pass
        return Draw.MolToImage(mol, size=size)
    except: return None


def render_rgroup(rgroup_smi, size=IMG_SIZE):
    """渲染R基片段图像（rgroup_smi可能不含*，直接渲染SMILES片段）"""
    try:
        # R基SMILES可能是片段（如CC、c1ccccc1等），直接解析渲染
        mol = Chem.MolFromSmiles(rgroup_smi, sanitize=False)
        if mol is None: return None
        try: Chem.SanitizeMol(mol, catchErrors=True)
        except: pass
        return Draw.MolToImage(mol, size=size)
    except: return None


def save_image(img, path):
    if img is None: return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try: img.save(path); return True
    except: return False


# ══════════════════════════════════════════════════════════════════════════════
# 【模块4】核心：构建SAR簇
# ══════════════════════════════════════════════════════════════════════════════

def build_sar_clusters_for_target(target_id, target_name, smiles_picsa50_list,
                                   images_dir, record_id_start):
    """
    对单个靶点的所有分子，做聚类+MCS，构建SAR簇。

    smiles_picsa50_list: [(mol_id, smiles, picsa50), ...]
    返回 (clusters_records, next_record_id)
    clusters_records: list of cluster_record（见下方格式）
    """
    smiles_list = [s for _, s, _ in smiles_picsa50_list]
    mol_id_list = [m for m, _, _ in smiles_picsa50_list]
    picsa50_list= [p for _, _, p in smiles_picsa50_list]

    print(f"    Butina聚类（{len(smiles_list)}个分子，cutoff={BUTINA_CUTOFF}）...")
    clusters = butina_cluster(smiles_list, cutoff=BUTINA_CUTOFF)
    print(f"    聚类结果: {len(clusters)} 个簇，"
          f"最大簇大小={max(len(c) for c in clusters) if clusters else 0}")

    cluster_records = []
    record_id = record_id_start
    skipped   = {"too_small": 0, "mcs_fail": 0, "spread": 0, "rgroup": 0, "image": 0}

    for cluster_idx in clusters:
        if len(cluster_idx) < MIN_VARIANTS:
            skipped["too_small"] += 1
            continue

        cluster_smiles = [smiles_list[i] for i in cluster_idx]
        cluster_molids = [mol_id_list[i] for i in cluster_idx]
        cluster_picsa50= [picsa50_list[i] for i in cluster_idx]

        # pIC50 spread 检查
        spread = max(cluster_picsa50) - min(cluster_picsa50)
        if spread < MIN_PICSA50_SPREAD or spread > MAX_PICSA50_SPREAD:
            skipped["spread"] += 1
            continue

        # MCS找共同骨架
        mcs_smarts = find_mcs_scaffold(cluster_smiles)
        if mcs_smarts is None:
            skipped["mcs_fail"] += 1
            continue

        # 构建含*的骨架SMILES（用第一个分子作为模板）
        scaffold_star_smi = build_scaffold_smiles_with_wildcard(
            cluster_smiles[0], mcs_smarts)
        if scaffold_star_smi is None:
            skipped["mcs_fail"] += 1
            continue

        # 对每个分子提取R基
        n_star = scaffold_star_smi.count("*")
        variants = []
        for mol_id, smi, pIC50 in zip(cluster_molids, cluster_smiles, cluster_picsa50):
            rgroups = extract_rgroup(smi, mcs_smarts)
            if rgroups is None: continue
            # 严格过滤：R基数量必须和骨架*数量一致
            if len(rgroups) != n_star: continue
            mw = mol_weight(smi)
            if mw and mw < MIN_MOL_WEIGHT: continue
            props = compute_properties(smi)
            variants.append({
                "chembl_id":        mol_id,
                "full_mol_smiles":  smi,
                "pchembl_value":    pIC50,
                "rgroup_smiles":    rgroups,
                "n_rgroups":        len(rgroups),
                "rdkit_properties": props,
                "mol_image_path":   "",
            })

        if len(variants) < MIN_VARIANTS:
            skipped["rgroup"] += 1
            continue

        # ── 多R基位点处理 ────────────────────────────────────────────
        # 如果骨架有多个*，需要找出"只有某一个位点不同、其他位点相同"的子集，
        # 这样题目才有明确的比较维度。
        # 策略：对每个R基位点pos，按其他位点的取值分组，
        #       找出组内该位点有>=4种不同取值的分组 → 作为独立的出题单元。
        # 如果只有1个*，直接用全部变体。
        if n_star <= 1:
            question_units = [{"focus_pos": 0, "variants": variants,
                               "scaffold_star_smiles": scaffold_star_smi,
                               "scaffold_smarts": mcs_smarts}]
        else:
            question_units = []
            # 对每个R基位点尝试出题
            for focus_pos in range(n_star):
                # 按"除focus_pos外的其他R基取值"分组
                other_key_groups = defaultdict(list)
                for v in variants:
                    rgs = v["rgroup_smiles"]
                    if len(rgs) < n_star:
                        continue  # R基数量不匹配，跳过
                    # 其他位点的R基组合作为key
                    other_key = tuple(
                        rgs[i] for i in range(n_star) if i != focus_pos
                    )
                    other_key_groups[other_key].append(v)

                for other_key, sub_variants in other_key_groups.items():
                    # 该sub_group内：其他R基相同，只有focus_pos位置的R基不同
                    focus_rgroups = [v["rgroup_smiles"][focus_pos]
                                     for v in sub_variants]
                    if len(set(focus_rgroups)) < MIN_VARIANTS:
                        continue  # focus位点的多样性不够

                    # pIC50 spread检查
                    pvals  = [v["pchembl_value"] for v in sub_variants]
                    spread = max(pvals) - min(pvals)
                    if spread < MIN_PICSA50_SPREAD or spread > MAX_PICSA50_SPREAD:
                        continue

                    # 构建"固定其他R基、只露出focus_pos位点"的骨架SMILES
                    # 用sub_variants[0]作为模板，把非focus位置的*替换回具体R基
                    fixed_scaffold = _fix_other_rgroups(
                        scaffold_star_smi, other_key, focus_pos, n_star)

                    question_units.append({
                        "focus_pos":           focus_pos,
                        "fixed_other_rgroups": list(other_key),
                        "variants":            sub_variants,
                        "scaffold_star_smiles": fixed_scaffold or scaffold_star_smi,
                        "scaffold_smarts":     mcs_smarts,
                        "pchembl_spread":      round(spread, 3),
                    })

            # 如果没找到合适的单位，尝试用全部变体+只看第一个R基位点
            if not question_units:
                skipped["rgroup"] += 1
                continue

        # ── 渲染并输出每个出题单元 ───────────────────────────────────
        for unit in question_units:
            unit_variants  = unit["variants"]
            unit_scaffold  = unit["scaffold_star_smiles"]
            unit_focus_pos = unit.get("focus_pos", 0)

            # 渲染骨架图像
            scaffold_img_path = os.path.join(
                images_dir, "scaffold", f"{record_id:06d}_scaffold.png")
            scaffold_img = render_scaffold(unit_scaffold)
            if not save_image(scaffold_img, scaffold_img_path):
                skipped["image"] += 1
                continue

            # 渲染每个变体的完整分子图像 + focus R基图像
            mol_img_ok = True
            for i, v in enumerate(unit_variants):
                mol_img_path = os.path.join(
                    images_dir, "molecule", f"{record_id:06d}_mol_{i:02d}.png")
                mol_img = render_molecule(v["full_mol_smiles"])
                if save_image(mol_img, mol_img_path):
                    v["mol_image_path"] = mol_img_path
                else:
                    mol_img_ok = False
                    break

                # 渲染focus_rgroup_pos对应的R基图像
                rgroups = v.get("rgroup_smiles", [])
                focus_rg_smi = rgroups[unit_focus_pos] if len(rgroups) > unit_focus_pos else None
                if focus_rg_smi:
                    rg_img_path = os.path.join(
                        images_dir, "rgroup", f"{record_id:06d}_rg_{i:02d}.png")
                    rg_img = render_rgroup(focus_rg_smi)
                    v["rgroup_image_path"] = rg_img_path if save_image(rg_img, rg_img_path) else ""
                    v["focus_rgroup_smiles"] = focus_rg_smi
                else:
                    v["rgroup_image_path"] = ""
                    v["focus_rgroup_smiles"] = ""

            if not mol_img_ok:
                skipped["image"] += 1
                continue
            
            

            unit_pvals = [v["pchembl_value"] for v in unit_variants]
            unit_spread = unit.get("pchembl_spread",
                                   round(max(unit_pvals) - min(unit_pvals), 3))

            cluster_records.append({
                "record_id":            record_id,
                "target_chembl_id":     target_id,
                "target_name":          target_name,
                "scaffold_smarts":      unit["scaffold_smarts"],
                "scaffold_star_smiles": unit_scaffold,
                "scaffold_image_path":  scaffold_img_path,
                "n_rgroup_positions":   unit_scaffold.count("*"),
                "focus_rgroup_pos":     unit_focus_pos,   # 题目比较的R基位点
                "fixed_other_rgroups":  unit.get("fixed_other_rgroups", []),
                "pchembl_spread":       unit_spread,
                "n_variants":           len(unit_variants),
                "variants":             unit_variants,
                "data_type":            "bio_activity",
            })
            record_id += 1

    print(f"    有效SAR簇: {len(cluster_records)}  "
          f"跳过: 太小={skipped['too_small']} spread={skipped['spread']} "
          f"mcs失败={skipped['mcs_fail']} R基提取失败={skipped['rgroup']} "
          f"图像失败={skipped['image']}")
    return cluster_records, record_id


def build_indication_clusters(approved_drugs, images_dir, record_id_start):
    """
    对approved drugs做聚类，找同骨架不同indication的簇。
    簇内分子：同骨架（相似度高），但indication不完全相同。
    每道题：4个选项都是同骨架的药物，只有1个有目标indication。
    """
    print(f"\n  构建Indication SAR簇（{len(approved_drugs)}个approved drugs）...")

    # 只取有indication的
    ind_drugs = [d for d in approved_drugs if d.get("indications")]
    print(f"  有indication的药物: {len(ind_drugs)} 个")

    smiles_list = [d["smiles"] for d in ind_drugs]
    print(f"  Butina聚类...")
    clusters = butina_cluster(smiles_list, cutoff=BUTINA_CUTOFF)
    print(f"  聚类结果: {len(clusters)} 个簇，"
          f"最大={max(len(c) for c in clusters) if clusters else 0}")

    cluster_records = []
    record_id = record_id_start
    skipped = {"too_small": 0, "mcs_fail": 0, "no_diversity": 0, "image": 0}

    for cluster_idx in clusters:
        if len(cluster_idx) < MIN_VARIANTS:
            skipped["too_small"] += 1
            continue

        cluster_drugs  = [ind_drugs[i] for i in cluster_idx]
        cluster_smiles = [d["smiles"] for d in cluster_drugs]

        # indication多样性检查：簇内至少有2种不同indication
        all_inds = set()
        for d in cluster_drugs:
            all_inds.update(d.get("indications",[]))
        if len(all_inds) < 2:
            skipped["no_diversity"] += 1
            continue

        # MCS
        mcs_smarts = find_mcs_scaffold(cluster_smiles)
        if mcs_smarts is None:
            skipped["mcs_fail"] += 1
            continue

        scaffold_star_smi = build_scaffold_smiles_with_wildcard(
            cluster_smiles[0], mcs_smarts)
        if scaffold_star_smi is None:
            skipped["mcs_fail"] += 1
            continue

        # 渲染骨架图
        scaffold_img_path = os.path.join(
            images_dir, "scaffold", f"ind_{record_id:06d}_scaffold.png")
        scaffold_img = render_scaffold(scaffold_star_smi)
        if not save_image(scaffold_img, scaffold_img_path):
            skipped["image"] += 1
            continue

        # 渲染每个药物图像 + 提取R基图像
        variants = []
        all_ok = True
        for i, drug in enumerate(cluster_drugs):
            mol_img_path = os.path.join(
                images_dir, "molecule", f"ind_{record_id:06d}_mol_{i:02d}.png")
            mol_img = render_molecule(drug["smiles"])
            if not save_image(mol_img, mol_img_path):
                all_ok = False; break

            # 提取R基（用簇的MCS骨架）
            rgroups = extract_rgroup(drug["smiles"], mcs_smarts)
            focus_rg_smi = rgroups[0] if rgroups else None   # indication簇只取第一个R基位点

            rg_img_path = ""
            if focus_rg_smi:
                rg_img_path = os.path.join(
                    images_dir, "rgroup", f"ind_{record_id:06d}_rg_{i:02d}.png")
                rg_img = render_rgroup(focus_rg_smi)
                if not save_image(rg_img, rg_img_path):
                    rg_img_path = ""

            variants.append({
                "chembl_id":           drug["chembl_id"],
                "full_mol_smiles":     drug["smiles"],
                "indications":         drug.get("indications", []),
                "mechanisms":          drug.get("mechanisms", []),
                "mol_image_path":      mol_img_path,
                "rgroup_smiles":       rgroups or [],
                "focus_rgroup_smiles": focus_rg_smi or "",
                "rgroup_image_path":   rg_img_path,
                "rdkit_properties":    compute_properties(drug["smiles"]),
            })

        if not all_ok or len(variants) < MIN_VARIANTS:
            skipped["image"] += 1
            continue

        cluster_records.append({
            "record_id":            record_id,
            "target_chembl_id":     "approved_drugs",
            "target_name":          "Approved Drugs",
            "scaffold_smarts":      mcs_smarts,
            "scaffold_star_smiles": scaffold_star_smi,
            "scaffold_image_path":  scaffold_img_path,
            "n_rgroup_positions":   scaffold_star_smi.count("*"),
            "n_variants":           len(variants),
            "variants":             variants,
            "all_indications":      list(all_inds),
            "data_type":            "indication_mechanism",
        })
        record_id += 1

    print(f"  Indication SAR簇: {len(cluster_records)}  "
          f"跳过: 太小={skipped['too_small']} mcs失败={skipped['mcs_fail']} "
          f"无indication多样性={skipped['no_diversity']} 图像失败={skipped['image']}")
    return cluster_records, record_id


# ══════════════════════════════════════════════════════════════════════════════
# 【模块5】主流程
# ══════════════════════════════════════════════════════════════════════════════

def run_full(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images_dir      = os.path.join(output_dir, "images")
    cache_dir       = os.path.join(output_dir, "cache")
    jsonl_path      = os.path.join(output_dir, "chembl_sar_scaffold.jsonl")
    drug_cache_path = os.path.join(cache_dir,  "approved_drugs_cache.json")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir,  exist_ok=True)

    print("=" * 65)
    print("  20b: ChEMBL SAR Scaffold 构建")
    print("=" * 65)

    all_clusters   = []
    record_id      = 0

    # ── 1. Bio_activity 靶点 ──────────────────────────────────────
    print(f"\n[1] 构建Bio_activity SAR簇（{len(SAR_TARGETS)} 个靶点）...")
    for target_id, target_name, target_cn in SAR_TARGETS:
        print(f"\n  靶点: {target_id} ({target_name} / {target_cn})")
        cache_path = os.path.join(cache_dir, f"acts_{target_id}.json")

        # 读缓存或下载
        if os.path.exists(cache_path):
            with open(cache_path) as f: acts = json.load(f)
            print(f"  读缓存: {len(acts)} 条活性记录")
        else:
            print(f"  拉取活性数据...")
            acts = fetch_activities_for_target(target_id)
            print(f"  获得 {len(acts)} 条活性记录")
            with open(cache_path, "w") as f: json.dump(acts, f)

        if not acts:
            print(f"  ⚠ 无活性记录，跳过")
            continue

        # 去重：同mol取最高pIC50
        mol_best = {}
        for a in acts:
            mid = a["molecule_chembl_id"]
            pv  = a["pchembl_value"]
            if mid not in mol_best or pv > mol_best[mid]:
                mol_best[mid] = pv

        print(f"  唯一分子: {len(mol_best)} 个，批量获取SMILES...")
        smi_cache = os.path.join(cache_dir, f"smiles_{target_id}.json")
        if os.path.exists(smi_cache):
            with open(smi_cache) as f: smi_map = json.load(f)
            # 如果缓存的SMILES数量不足唯一分子数的50%，认为缓存不完整，重新拉取
            if len(smi_map) < len(mol_best) * 0.5:
                print(f"  SMILES缓存不完整（{len(smi_map)}/{len(mol_best)}），重新拉取...")
                os.remove(smi_cache)
                smi_map = fetch_smiles_batch(list(mol_best.keys()))
                with open(smi_cache, "w") as f: json.dump(smi_map, f)
        else:
            smi_map = fetch_smiles_batch(list(mol_best.keys()))
            with open(smi_cache, "w") as f: json.dump(smi_map, f)
        print(f"  获得SMILES: {len(smi_map)} 个")

        # 过滤MW太小的分子
        smiles_picsa50_list = []
        for mid, smi in smi_map.items():
            mw = mol_weight(smi)
            if mw and mw >= MIN_MOL_WEIGHT:
                smiles_picsa50_list.append((mid, smi, mol_best[mid]))
        print(f"  MW>={MIN_MOL_WEIGHT}的分子: {len(smiles_picsa50_list)} 个")

        if len(smiles_picsa50_list) < MIN_VARIANTS:
            print(f"  ⚠ 分子数不足，跳过")
            continue

        # 构建SAR簇
        clusters, record_id = build_sar_clusters_for_target(
            target_id, target_name, smiles_picsa50_list,
            images_dir, record_id
        )
        all_clusters.extend(clusters)
        print(f"  {target_name}: +{len(clusters)} 个SAR簇，"
              f"累计 {len(all_clusters)} 个")

    # ── 2. Indication 簇 ─────────────────────────────────────────
    print(f"\n[2] 构建Indication SAR簇...")
    approved_drugs = fetch_approved_drugs(drug_cache_path)
    ind_clusters, record_id = build_indication_clusters(
        approved_drugs, images_dir, record_id)
    all_clusters.extend(ind_clusters)
    print(f"  Indication SAR簇: {len(ind_clusters)} 个")

    # ── 3. 写入jsonl ──────────────────────────────────────────────
    print(f"\n[3] 写入 {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_clusters:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 4. 统计 ───────────────────────────────────────────────────
    bio_clusters = [r for r in all_clusters if r["data_type"] == "bio_activity"]
    ind_clusters = [r for r in all_clusters if r["data_type"] == "indication_mechanism"]
    bio_variants = sum(r["n_variants"] for r in bio_clusters)
    ind_variants = sum(r["n_variants"] for r in ind_clusters)

    print(f"\n{'='*65}")
    print(f"✓ 完成")
    print(f"  Bio_activity SAR簇:      {len(bio_clusters):>5} 个  ({bio_variants} 个变体)")
    print(f"  Indication SAR簇:        {len(ind_clusters):>5} 个  ({ind_variants} 个变体)")
    print(f"  合计SAR簇:               {len(all_clusters):>5} 个")
    print(f"  输出: {jsonl_path}")

    spread_vals = [r["pchembl_spread"] for r in bio_clusters if "pchembl_spread" in r]
    if spread_vals:
        import statistics
        print(f"\n  Bio pIC50 spread统计:")
        print(f"    平均={statistics.mean(spread_vals):.2f}  "
              f"中位={statistics.median(spread_vals):.2f}  "
              f"范围=[{min(spread_vals):.2f},{max(spread_vals):.2f}]")

    print(f"\n  下一步: 更新26_build_new_vqa.py读取此数据")


def run_explore(output_dir, target_id="CHEMBL279"):
    """只跑单个靶点，测试流程"""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    cache_dir  = os.path.join(output_dir, "cache")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    target_name = next((n for tid, n, _ in SAR_TARGETS if tid == target_id), target_id)
    print(f"=== explore模式: {target_id} ({target_name}) ===")

    cache_path = os.path.join(cache_dir, f"acts_{target_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f: acts = json.load(f)
    else:
        acts = fetch_activities_for_target(target_id, max_records=500)
        with open(cache_path, "w") as f: json.dump(acts, f)
    print(f"活性记录: {len(acts)} 条")

    mol_best = {}
    for a in acts:
        mid, pv = a["molecule_chembl_id"], a["pchembl_value"]
        if mid not in mol_best or pv > mol_best[mid]:
            mol_best[mid] = pv

    smi_cache = os.path.join(cache_dir, f"smiles_{target_id}.json")
    if os.path.exists(smi_cache):
        with open(smi_cache) as f: smi_map = json.load(f)
    else:
        smi_map = fetch_smiles_batch(list(mol_best.keys())[:200])
        with open(smi_cache, "w") as f: json.dump(smi_map, f)
    print(f"SMILES: {len(smi_map)} 个")

    smiles_picsa50_list = [
        (mid, smi, mol_best[mid]) for mid, smi in smi_map.items()
        if mol_weight(smi) and mol_weight(smi) >= MIN_MOL_WEIGHT
    ]
    print(f"有效分子: {len(smiles_picsa50_list)} 个\n")

    clusters, _ = build_sar_clusters_for_target(
        target_id, target_name, smiles_picsa50_list, images_dir, 0)

    print(f"\n=== Explore结果: {len(clusters)} 个SAR簇 ===")
    for i, c in enumerate(clusters[:5]):
        print(f"\n  簇{i+1}: scaffold={c['scaffold_star_smiles'][:60]}")
        print(f"    变体数={c['n_variants']}  spread={c['pchembl_spread']}")
        for v in sorted(c["variants"], key=lambda x: x["pchembl_value"], reverse=True)[:4]:
            print(f"    pIC50={v['pchembl_value']:.2f}  "
                  f"R基={v['rgroup_smiles']}  "
                  f"smi={v['full_mol_smiles'][:50]}")

    if len(clusters) >= 5:
        print(f"\n✅ 数据充足，建议运行 --mode full")
    else:
        print(f"\n⚠️  簇数较少，可以调整参数后再试")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full","explore"], default="explore")
    parser.add_argument("--target", default="CHEMBL279",
                        help="explore模式下的靶点ID")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.mode == "explore":
        run_explore(args.output_dir, args.target)
    else:
        run_full(args.output_dir)


if __name__ == "__main__":
    main()