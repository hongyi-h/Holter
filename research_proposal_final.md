# 研究提案（终版）：首个24小时连续心电图基础模型

> 目标期刊：Nature / Nature Medicine
> 团队：临床（心内科）+ CS/AI
> 数据：10,000例 24h Holter（3通道, 128Hz, 逐拍标注）+ 外部验证集
> 计算资源：4×A100 (80GB)
> 时间预算：12个月+

---

## 0. 一句话总结

**在10,000例24小时连续心电图（330亿采样点）上训练首个长程ECG基础模型，展示24h连续数据能学到10秒静态ECG学不到的心脏时序特征，并在多个下游任务和外部验证集上证明其优越性。**

---

## 1. 为什么这能发Nature

### 1.1 文献空白

现有所有ECG基础模型都在**10秒静态ECG片段**上训练：

| 模型 | 期刊 | 训练数据 | 局限 |
|------|------|---------|------|
| Oh et al. | Nature Medicine, 2024 | ~1.1M × 10sec 12导联 | 无时序动态 |
| ECG-FM (Wornow et al.) | arXiv, 2024 | ~800k × 10sec 12导联 | 无时序动态 |
| Hughes et al. | Lancet Digital Health, 2024-25 | ~500k × 10sec | 无时序动态 |
| MERL (Li et al.) | arXiv, 2024 | ~800k × 10sec 12导联 | 无时序动态 |

**没有人在24h连续ECG上做过基础模型。** 这不是因为没人想到，而是因为Holter数据难获取、格式混乱、处理复杂。你已经解决了数据获取问题。

### 1.2 数据规模优势

| 指标 | 现有最大ECG FM | 我们 |
|------|--------------|------|
| 患者数 | ~1.1M | 10k（劣势） |
| 原始采样点 | ~66B | ~330B（**5倍优势**） |
| 时间跨度/例 | 10秒 | 24小时（**8640倍**） |
| 能学到的时序尺度 | 单拍形态 | 昼夜节律、心律失常动态、自主神经调节 |

### 1.3 Nature的叙事

"我们证明了：当深度学习从10秒心电图片段扩展到24小时连续监测时，模型学到了一组全新的心脏时序特征——昼夜节律编码、心律失常发作/终止动态、自主神经调节模式——这些是静态ECG基础模型根本无法捕获的。这一发现重新定义了心电AI的能力边界。"

---

## 2. 技术方案

### 2.1 模型架构：Beat-Synchronous Hierarchical Encoder

核心设计原则：**局部形态在拍/episode尺度处理，全天时序关系通过全局注意力建模。**

```
Level 0 (Beat Tokenizer): R-peak对齐的80样本窗口 → 256-d beat embedding + VQ code
  输入: 80 samples × 3ch (625ms, 以R波为中心)
  架构: Conv1d stem + 4 ResBlocks + SE + AttentivePool + VQ codebook (512 codes)
  输出: 连续embedding h_i ∈ R^256 + 离散形态码 c_i ∈ {1..512}
  参数: ~1.1M

Level 1 (Episode Encoder): 64拍 → episode token
  输入: 64个beat embeddings (约38-64秒)
  架构: 6层Transformer, d=384, 6头, RoPE, CLS token
  输出: 上下文化beat states (64, 384) + episode token (384,)
  参数: ~10.7M

Level 1b (Rhythm Branch): 64拍节律特征 → episode rhythm token
  输入: VQ code + RR bins + clock features, 每episode独立处理
  架构: 4层 local-conv + MLP blocks (kernel=7), attentive pooling
  输出: per-beat rhythm state (128,) + episode rhythm summary (128,)
  参数: ~0.4M
  设计理由: 节律上下文（耦合间期、HRV、ectopy模式）在±64拍内完备，
            不需要跨episode的全天序列建模。昼夜信息由clock features提供。

Level 2 (Day Encoder): ~1,563 episode tokens → 24h全局表征
  输入: fused episode tokens [waveform_ep; rhythm_ep] → 512-d
  架构: 12层标准Transformer, d=512, 8头, sinusoidal位置编码
  序列长度: ~1,563 (标准注意力完全可行: 1563² ≈ 244万元素)
  输出: day embedding (512,) + 上下文化episode states (n_ep, 512)
  参数: ~39.4M
  设计理由: 全局注意力让任意两个时间段直接交互，
            这正是论文核心叙事"24h上下文"的最佳体现。
```

**总参数量: ~52.3M**

**为什么不用Mamba/SSM：**
- RhythmBranch: 64拍episode内的局部conv已足够，不需要100k token的全天序列
- DayEncoder: 1563 tokens对Transformer无压力，且全局注意力比SSM更直接地建模时序关系
- 工程收益: 纯PyTorch实现，无外部依赖，任何GPU环境可运行

### 2.2 预训练策略（六个自监督任务）

全部不需要标签，仅用原始波形和R波标注：

**Beat级:**
- Masked Patch MAE: 50% patch mask, L1 + derivative-L1重建
- 迫使beat tokenizer学习完整的P-QRS-T形态

**Episode级:**
- Contrastive Predictive Coding (CPC): 从当前episode预测下2个episode
- Waveform-Rhythm Alignment: 对齐波形episode token和节律episode token (InfoNCE)
- Temporal Order Prediction: 判断相邻episode是否被交换

**Day级:**
- Masked Episode Modeling: mask 15% episode tokens, 从day encoder输出重建
- Day Statistics Prediction: 预测全天统计量 (HR, HRV, PVC burden等)
- Report Concept Prediction (弱监督): 预测医生结论中的19个概念标签

**Rhythm级:**
- Span-Masked Rhythm Prediction: mask 30%节律token spans, 预测VQ code和RR bin
- Next-RR Prediction: 从节律状态预测下一个RR间期

### 2.3 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| Beat Tokenizer | Conv1d + 4 ResBlocks + VQ | 轻量但足够捕获形态 |
| Episode Encoder | 6层Transformer, d=384, 6头 | 64拍序列，标准注意力 |
| Rhythm Branch | 4层local-conv+MLP, d=128 | Episode-local，无需SSM |
| Day Encoder | 12层Transformer, d=512, 8头 | 1563 tokens全局注意力 |
| 总参数量 | ~52.3M | 与现有ECG FM可比 |
| Batch size | 1/GPU × 8 GPU = 8 | 每个样本是一个完整24h记录 |
| 优化器 | AdamW, lr=2e-4, cosine decay, warmup 200步 | 标准配置 |
| 预训练epochs | 40 | 1170例 × 40 = ~4000步 |
| 预计训练时间 | 1-2天（8×GPU） | 可接受 |
| 混合精度 | BF16 | 标准 |
| Loss权重 | beat 0.35, episode 0.20, day 0.20, rhythm 0.20, report 0.05 | Day/report前5 epoch ramp-up |

### 2.4 数据处理Pipeline

```
原始数据 (data/DMS/)
  ├── .dat文件 → 读取uint8, reshape为(n_samples, 3)
  ├── RPointProperty.txt → 提取R点时间戳和类型标签(N/V/F)
  └── HolterSummary.csv → 提取人口学信息和医生结论

预处理:
  1. 质控: beat count差异>10%或报告缺失的排除 (8/1178条)
  2. 波形标准化: per-channel median/MAD, clip [-5, 5]
  3. Beat窗口: 以R波为中心, 24 pre + 56 post samples = 80 samples
  4. Episode切分: 每64拍为一个episode (~1563 episodes/day)
  5. 节律token: VQ code + 32-bin log-RR + hour_sin/cos
  6. 数据增强: 幅度缩放、高斯噪声、基线漂移、通道dropout、时间抖动
```

---

## 3. 下游任务评估

预训练完成后，用以下任务评估模型质量。**所有标签均来自现有数据，不需要额外标注。**

### 3.1 任务列表

| # | 任务 | 标签来源 | 评估指标 | 为什么重要 |
|---|------|---------|---------|-----------|
| 1 | 逐拍心律失常分类 (N/V/F) | RPointProperty | F1, AUROC | 基线任务，与Hannun 2019对比 |
| 2 | 室早负荷回归 | CSV: 室早百分比 | MAE, R² | 临床核心指标 |
| 3 | 室上早负荷回归 | CSV: 室上早百分比 | MAE, R² | 临床核心指标 |
| 4 | 心率统计预测 | CSV: 平均/最快/最慢心率 | MAE | 验证模型理解全局心率动态 |
| 5 | 医生结论多标签分类 | NLP从结论文本提取 | F1, AUROC | 最接近临床应用的任务 |
| 6 | 年龄/性别预测 | CSV | MAE / AUROC | 表征质量的proxy |
| 7 | 异常时间段定位 | 结论中提到的时间点 | mAP | 临床实用性 |
| 8 | 昼夜心率模式分类 | 从数据自动提取 | NMI, ARI | **只有24h数据能做的任务** |
| 9 | 下一窗口预测 | 自监督 | MSE | 时序建模能力 |

### 3.2 关键对比实验

| 对比 | 目的 |
|------|------|
| 我们 vs 现有ECG FM (在10sec片段上预训练) | 证明24h连续预训练的优势 |
| 我们 vs 从头训练 (无预训练) | 证明预训练的价值 |
| 我们 vs 传统HRV特征 + XGBoost | 证明深度表征优于手工特征 |
| 我们 (24h输入) vs 我们 (随机10sec输入) | 证明长程上下文的价值 |
| 层次化Transformer vs Flat Transformer | 验证架构选择 |

### 3.3 核心实验：24h vs 10sec的消融

这是论文最关键的实验——直接回答"24h连续数据到底比10sec片段多学到了什么？"

设计：
1. 用同样的模型架构，分别在24h连续数据和随机10sec片段上预训练
2. 在所有下游任务上对比
3. 预期结果：
   - 任务1-3（逐拍分类、室早/室上早负荷）：24h应该显著更好（因为有上下文）
   - 任务5（医生结论）：24h应该显著更好（结论是基于全天数据的）
   - 任务6（年龄/性别）：可能差异不大（10sec也能学到）
   - 任务8（昼夜模式）：24h应该远超10sec（10sec根本做不了这个任务）

---

## 4. 论文结构

### 核心Figures

**Fig 1: 概览**
- (a) 数据：10k例24h Holter的规模可视化
- (b) 模型：层次化Transformer架构图
- (c) 预训练：三个自监督任务示意

**Fig 2: 预训练表征分析**
- (a) UMAP：不同患者、不同时间段的表征分布
- (b) 时间编码：模型是否自动学到了昼夜节律？（可视化Level 2的注意力权重）
- (c) 与传统HRV指标的相关性：模型表征是否包含了SDNN/RMSSD等信息？

**Fig 3: 下游任务性能**
- (a) 9个下游任务的性能对比表
- (b) 关键任务的ROC曲线/混淆矩阵

**Fig 4: 24h vs 10sec消融**
- (a) 各任务上24h预训练 vs 10sec预训练的性能差异
- (b) 24h模型独有的能力展示（昼夜模式、异常定位）
- (c) 注意力可视化：24h模型在做决策时关注了哪些时间段

**Fig 5: 外部验证**
- 在外部验证集上复现主要结果

### 补充材料
- 完整的消融实验
- 超参数敏感性分析
- 更多可视化
- 数据处理细节

---

## 5. 风险评估与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 24h预训练 vs 10sec预训练差异不显著 | 20% | 致命 | 重新设计预训练任务，强化时序学习；如果确实无差异，说明24h连续数据的价值不在表征学习而在其他地方，需要转向 |
| 128Hz采样率限制波形细节学习 | 15% | 中等 | 128Hz对QRS波群足够（主频<40Hz），对高频成分（如late potentials）不够；在论文中明确讨论，定位为"Holter级别"而非"诊断级别"的基础模型 |
| 3通道 vs 12导联信息量不足 | 15% | 中等 | 3通道（含II和V5）覆盖了主要的心电信息；在对比实验中与12导联模型公平比较（只用对应通道） |
| 10k患者多样性不够，外部验证失败 | 25% | 高 | 这是最大风险；应对：(1) 分析训练集的人口学分布，(2) 如果外部验证集人群差异大，做domain adaptation，(3) 在论文中诚实讨论局限性 |
| 训练不稳定/不收敛 | 10% | 中等 | 先在1k例子集上做小规模实验验证方案可行性 |
| 审稿人质疑"10k太少" | 30% | 中等 | 强调信号量（330B采样点）而非患者数；强调24h连续性是独特贡献；展示外部验证泛化性 |

---

## 6. 时间线

| 阶段 | 时间 | 里程碑 | 关键产出 |
|------|------|--------|---------|
| **M1** | 数据Pipeline | 10k例全部读取、质控、切分完成 | 预处理代码 + 数据质量报告 |
| **M2** | 小规模验证 | 在1k例上跑通预训练 + 1个下游任务 | 确认方案可行性 |
| **M3-4** | 全量预训练 | 10k例预训练完成，多轮调参 | 预训练模型checkpoint |
| **M5-6** | 下游任务评估 | 9个下游任务 + 所有对比实验完成 | 性能数据表 |
| **M7** | 24h vs 10sec消融 | 核心消融实验完成 | 论文核心figure |
| **M8** | 外部验证 | 在外部数据上复现 | 泛化性证据 |
| **M9-10** | 论文撰写 | 初稿完成 | 论文草稿 |
| **M11** | 内部审阅 + 修改 | 团队审阅、临床同事确认 | 终稿 |
| **M12** | 投稿 | 提交Nature/Nature Medicine | — |

### 关键决策点

- **M2结束时**：小规模实验结果是否支持继续？如果预训练loss不收敛或下游任务无提升，需要调整方案
- **M6结束时**：24h vs 10sec是否有显著差异？如果没有，需要重新评估论文叙事
- **M8结束时**：外部验证是否通过？如果失败，需要分析原因并决定是否补充实验

---

## 7. 团队分工

| 角色 | 负责 | 时间投入 |
|------|------|---------|
| CS/AI（主力） | 数据pipeline、模型实现、训练、实验、可视化 | M1-M8全程 |
| CS/AI（协助） | 代码review、对比实验、消融实验 | M3-M7 |
| 临床 | 下游任务标签定义、结论文本NLP标签审核、结果临床解读 | M1(标签), M5-M6(解读), M9-M10(写作) |
| 临床 | 外部验证集获取和预处理、论文临床部分撰写 | M1-M2(数据), M8(验证), M9-M10(写作) |

---

## 8. 与方向A（表型发现）的关系

方向A不需要单独做。在方向B的框架下：

- **表型发现 = 下游任务8（昼夜心率模式分类）的扩展版**
- 预训练模型的Level 2表征天然就是24h心率动态的压缩表示
- 在这个表征空间上做聚类，就是方向A的核心工作
- 如果聚类发现有意义的亚型，可以作为论文的一个亮点section
- 如果没有，不影响论文主体（基础模型本身就是贡献）

**方向A变成了方向B的一个bonus，而不是独立的赌注。**

---

## 9. 如果后续拿到结局数据

基础模型训练好后，接入结局数据只需要：
1. 用预训练模型提取每个患者的24h表征向量
2. 以表征向量为输入，训练一个简单的Cox回归或分类器
3. 验证是否预测不良事件

这可以作为第二篇论文（Nature Medicine），与第一篇（基础模型本身）形成系列。

---

## 10. 下一步行动

1. **立即**：确认外部验证集的格式细节（采样率、通道数、标注格式）
2. **本周**：搭建数据读取pipeline，跑通10k例的批量读取和质控
3. **两周内**：在1k例子集上实现并验证预训练方案
4. **一个月内**：启动全量预训练

---

## 附录：关键参考文献

### 直接竞品（ECG基础模型）

1. **ECGFounder** "A foundation model for ECG analysis." *NEJM AI*, 2025.
   - >10M ECGs, 150 labels, reduced-lead support
   - 我们的差异：24h连续 vs 10sec片段；Holter-native任务

2. **ECG-LFM** "A large foundation model for ECG." *Nature Communications*, 2026.
   - >10M 12-lead ECGs, SSL, CVD prediction + genetics
   - 我们的差异：时序动态 vs 静态形态

3. **AnyECG** "Universal ECG foundation model." *arXiv*, 2026.
   - 13.3M ECGs, broad health profiling
   - 我们的差异：同上

4. **ECG-FM (Wornow et al.)** "An open electrocardiogram foundation model." *arXiv*, 2024.
   - 自监督预训练，开源
   - 我们的差异：时序动态 vs 静态形态

5. **Oh et al.** "A foundation model for clinician-level ECG interpretation." *Nature Medicine*, 2024.
   - 1.1M 12导联ECG预训练，多任务微调
   - 我们的差异：24h连续 vs 10sec片段

### 最接近的Holter DL竞品

6. **DeepHHF** "Deep learning on 24h Holter for heart failure prediction." 2025/2026.
   - 69,663条24h单导联Holter预测5年心衰
   - 关键区别：预测模型（非基础模型），单导联，无SSL预训练
   - 我们的优势：基础模型范式，多任务，可迁移

7. **RhythmBERT** "ECG-as-language rhythm SSL." *arXiv*, 2026.
   - 概念上接近我们的rhythm branch
   - 我们的差异：全天多尺度 vs 片段级

### 方法参考

8. **He et al.** "Masked Autoencoders Are Scalable Vision Learners." *CVPR*, 2022.
   - MAE预训练策略参考

9. **Hannun et al.** "Cardiologist-level arrhythmia detection using a deep neural network." *Nature Medicine*, 2019.
   - DL+ECG范式奠基

10. **Attia et al.** "AI-enabled ECG for detection of cardiac contractile dysfunction." *Nature Medicine*, 2019.
    - "ECG中发现隐匿信息"的叙事先驱
