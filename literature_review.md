# 文献综述：深度学习 + 动态心电图（Holter ECG）

**生成日期**: 2026-04-24
**研究方向**: 深度学习 + 动态心电图，方法导向，目标期刊 IF 5-15
**数据条件**: 5k-80k 条 × 3导联 × 24h × 128Hz + R波标注(N/V/F) + 医生中文报告

---

## 一、数据集概况

每条数据 = 1次24h Holter监测，来源仁济医院 DMS 系统：
- **波形**: 3通道交织排列，128Hz，uint8，~31MB/条
- **报告摘要**: GBK编码CSV，含心率统计、室早/室上早计数、医生诊断结论（自由文本）
- **R波标注**: `时间戳:节拍类型`，类型包括 N(正常)、V(室早)、F(融合波)
- **示例**: 35岁女性，24h记录103927个心搏，N=102100, V=1827(PVC 1.76%), F=8

---

## 二、领域全景

ECG + DL 领域正从"任务特定模型"向"基础模型"范式转移。以下按主题分组。

### 1. ECG 基础模型（Foundation Models）— 当前最热方向

| 论文 | 时间 | arXiv/DOI | 核心贡献 | 与本项目关系 |
|------|------|-----------|---------|-------------|
| **ECGFounder** (Li et al.) | 2024 | Nature Medicine | 首个大规模ECG基础模型，150种诊断 | 标杆。可验证其在Holter场景的泛化性 |
| **AnyECG** (Li, Zhu et al.) | 2026.01 | arXiv 2601.10748 | ECGFounder扩展版，1330万ECG，全身健康画像 | 同上，更强版本 |
| **ECG-FM** (McKeen et al.) | 2024.08 | arXiv 2408.05178 | 开源ECG基础模型，自监督预训练 | 开源可用，可直接fine-tune |
| **ECG-MoE** (Xu et al.) | 2026.03 | arXiv 2603.04589 | 混合专家架构，捕获周期性+多样特征 | 架构创新方向 |
| **Foundation Model via Masked Latent Attention** (Vandenhirtz et al.) | 2026.03 | arXiv 2603.26475 | 跨导联建模，masked latent attention | 方法新颖，关注导联间关系 |
| **CLEF** (Shu et al.) | 2025.12 | arXiv 2512.02180 | 临床引导的对比学习，利用风险评分加权负样本 | 单导联+可穿戴场景，与Holter相关 |
| **CoRe-ECG** (Qin et al.) | 2026.04 | arXiv 2604.11359 | 对比+重建协同的自监督学习 | 最新SSL方法 |
| **RhythmBERT** (Wang et al.) | 2026.02 | arXiv 2602.23060 | 基于心搏节律语义的自监督模型 | 直接利用R波标注的节律结构 |
| **ECG-Byte** (Han et al.) | 2024.12 | arXiv 2412.14373 | ECG tokenizer，端到端生成式建模 | 将ECG当作"语言"处理 |
| **PanLUNA** (Zelic et al.) | 2026.04 | arXiv 2604.04297 | 5.4M参数泛模态FM，支持ECG/EEG/PPG | 轻量级多模态方向 |

**关键观察**: 几乎所有基础模型都基于 **标准12导联、10秒静态ECG** 训练。没有一个专门为3导联、24小时连续Holter设计。这是明确的gap。

### 2. Holter / 长程ECG 专项研究

| 论文 | 时间 | arXiv/DOI | 核心 | 备注 |
|------|------|-----------|------|------|
| **Zvuloni et al.** "Day-Long ECG to Predict HF" | 2025.12 | arXiv 2601.00014 | 24h单导联ECG预测心衰，可解释AI | 最接近本项目数据形态，但有结局标签 |
| **AI-HEART** (Kontou et al.) | 2026.02 | arXiv 2603.16891 | 端到端云平台，多日3导联Holter分析 | 系统工程导向，非方法创新 |
| **SHDB-AF** (Tsutsui et al.) | 2024.06 | arXiv 2406.16974 | 日本Holter房颤数据库 | 数据集贡献型论文 |
| **Robust Peak Detection for Holter** (Gabbouj et al.) | 2021 | arXiv 2110.02381 | 自组织神经网络做Holter R波检测 | 较老，直接针对Holter噪声问题 |

**关键观察**: Holter级别的DL研究非常少，且大多是应用型（检测某种心律失常），缺乏方法论创新。

### 3. PVC（室早）检测 — 数据中最明确的标签

| 论文 | 时间 | arXiv/DOI | 核心 | 备注 |
|------|------|-----------|------|------|
| **uPVC-Net** (Hamami et al.) | 2025.06 | arXiv 2506.11238 | 通用PVC检测算法，跨数据集泛化 | 直接竞品，但只用短片段 |
| **ECGFounder → VT/VF预测** (Huang et al.) | 2025.10 | arXiv 2510.17172 | ECGFounder + XGBoost预测AMI后恶性室性心律失常 | 基础模型+传统ML混合 |

### 4. 泛化性与域适应

| 论文 | 时间 | arXiv/DOI | 核心 | 备注 |
|------|------|-----------|------|------|
| **ECG-RAMBA** (Nguyen et al.) | 2025.12 | arXiv 2512.23347 | 形态-节律解耦 + 长程建模，零样本泛化 | 方法创新，直接相关 |
| **CREMA** (Song et al.) | 2024.06 | arXiv 2407.07110 | 对比正则化MAE，跨临床域鲁棒性 | 自监督+泛化 |
| **Benchmarking ECG FMs** (Al-Masud et al.) | 2025.09 | arXiv 2509.25095 | ECG基础模型的系统benchmark | 揭示现有FM的局限性 |
| **Demographic-Aware SSL** (Huang et al.) | 2026.03 | arXiv 2603.19695 | 人口统计学感知的自监督异常检测 | 公平性+罕见病检测 |

### 5. ECG + LLM / 多模态

| 论文 | 时间 | arXiv/DOI | 核心 | 备注 |
|------|------|-----------|------|------|
| **ECG-R1** (Jin et al.) | 2026.02 | arXiv 2602.04279 | 协议引导的可靠ECG解读MLLM | 最新ECG-LLM |
| **Encoder-Free ECG-LM** (Han et al.) | 2026.01 | arXiv 2601.18798 | 无编码器的ECG语言模型 | 简化架构 |
| **ECG Foundation Models Survey** (Khan et al.) | 2026.04 | arXiv 2604.02501 | ECG FM + Medical LLM综述 | 最新综述，部署视角 |
| **ESI** (Yu et al.) | 2024.05 | arXiv 2405.19366 | LLM增强的ECG语义预训练 | 文本-波形对齐 |

### 6. 其他相关

| 论文 | 时间 | arXiv/DOI | 核心 | 备注 |
|------|------|-----------|------|------|
| **SignalMC-MED** (Gustafsson et al.) | 2026.03 | arXiv 2603.09940 | 生物信号FM评估benchmark，单导联ECG+PPG | 长时多模态评估 |
| **Sampling Matters** (Mahmuod et al.) | 2026.04 | arXiv 2604.16437 | ECG采样率对AF检测的影响 | 与128Hz低采样率Holter直接相关 |
| **Pocket-K** (Tang et al.) | 2026.03 | arXiv 2603.14177 | 单导联ECG检测高钾血症，基于ECGFounder | FM下游应用范例 |
| **BioMamba** (Qian et al.) | 2025.03 | arXiv 2503.11741 | 双向Mamba用于生物信号分类 | SSM架构在ECG上的应用 |

---

## 三、Gap 分析

### 核心发现

1. **所有ECG基础模型都基于10秒标准ECG**，没有针对24h连续Holter的预训练方案
2. **Holter DL研究稀少且应用导向**，缺乏方法论创新
3. **3导联低采样率(128Hz)场景被忽视**，现有模型假设12导联500Hz
4. **R波标注序列（节律信息）未被充分利用**，RhythmBERT是唯一尝试但仅用于短ECG
5. **中文Holter报告文本**是独特资产，无人做过Holter波形↔中文报告对齐

---

## 四、可行方向（基于仅有Holter数据的约束）

### 方向 A：Holter 级别的自监督预训练基础模型（最推荐）

**Gap**: 现有ECG基础模型全部基于10秒标准ECG。没有人做过专门针对24小时连续Holter信号的自监督预训练。

**为什么是真问题**:
- 10秒ECG捕获瞬时状态；24h Holter捕获动态变化（昼夜节律、运动响应、间歇性心律失常）
- 12导联FM迁移到3导联Holter有严重域偏移（导联数、采样率、噪声特性、信号长度差3个数量级）
- R波标注(N/V/F)可作为自监督的天然信号——节律序列本身是一种"语言"

**做法**:
1. 在5k条24h数据上做自监督预训练（masked waveform modeling + 节律序列预测）
2. 在公开数据集（MIT-BIH, INCART, SHDB-AF等）上验证下游任务泛化性
3. 与ECGFounder/ECG-FM直接对比在Holter场景的表现

**目标期刊**: npj Digital Medicine (IF~15), Medical Image Analysis (IF~10), IEEE JBHI (IF~7)

**风险**: 5k条数据做预训练可能不够，扩到80k后故事更强。

### 方向 B：利用报告文本做ECG-Language对齐（差异化优势）

**Gap**: 每条Holter有医生中文诊断结论（自由文本）。现有ECG-Language工作都基于英文标准ECG报告。

**做法**: Holter波形 ↔ 中文报告的对比学习，结合RhythmBERT的节律语义思路做多粒度对齐。

**风险**: 中文医学NLP的评审接受度可能有限，需证明方法的语言无关性。

### 方向 C：长程ECG的高效推理架构

**Gap**: 24h × 128Hz × 3通道 = ~3300万采样点。没有现成transformer能直接处理。

**做法**: 层次化架构——心搏级编码 → 片段级聚合 → 全天级推理，利用R波标注做自然分割。可结合Mamba/SSM架构。

**风险**: 纯架构创新越来越难发，需配合强实验。

---

## 五、关于 IF≥20 的说明

Nature Medicine / Lancet Digital Health 级别的 ECG-AI 论文 100% 需要回答临床问题（预测心衰、筛查房颤、检测低EF等），验证必须有 ground truth 结局标签。仅有 Holter 波形 + 报告摘要，没有临床结局数据，IF≥20 几乎不可能。如未来能关联患者后续诊断、心超、住院记录、随访结局等数据，可重新评估。
