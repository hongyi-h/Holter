# 架构决策：去除 Mamba，改用 Episode-Local Rhythm + Day Transformer

日期: 2026-04-29
触发: M2 预训练 OOM（mamba-ssm 依赖曦云 libmaca_mathlib_host.so 缺失，fallback SSM 在 100k tokens 上 OOM）

## 问题根因

RhythmBranch 设计为全天 100k beat-level BiMamba 序列。这在工程和临床逻辑上都有问题：
1. **工程**：100k tokens 的 SSM 即使用官方 mamba-ssm 也需要大量显存；fallback 纯 PyTorch 实现直接 OOM
2. **临床**：没有节律现象需要单拍看到 8 小时前的上下文

## Codex (GPT-5.5) 交叉验证结论

### RhythmBranch 不需要全天序列

节律特征的真实时间尺度：
- PVC 耦合间期、代偿间歇：±5 拍
- 二联律/三联律/成对/短阵室速：连续 3-20 拍
- HRV (RMSSD, SDNN)：5 分钟窗口 = 300-500 拍
- 昼夜节律：hour_sin/hour_cos 已编码时钟时间

**但 64 拍 episode 太短**（38-64 秒），不够 5 分钟 HRV。

### DayEncoder 应该用 Transformer

- 1563² = 244 万注意力元素，标准 Transformer 无压力
- 全局注意力直接建模"凌晨心律失常 vs 下午运动心率"的关系
- 这正是论文核心叙事"24h 上下文"的最佳体现
- 注意力图可用于可视化（Figure 4 候选）

### 对论文叙事的影响

**加强而非削弱**。正确的临床叙事是：
> "局部 ECG 形态和短时节律模式在拍/episode 尺度处理；临床有意义的 Holter 解读来自在分钟、小时、全天尺度上聚合这些模式。"

而不是"每个拍需要看全天"。

## 决策：新架构

### RhythmBranch → Episode-Local + 多尺度统计

- 每 64 拍 episode 独立处理（轻量 MLP 或 1-2 层小 Transformer）
- 输入不变：VQ code + RR bins + clock features
- 输出：per-beat rhythm state (128-d) + episode rhythm token (128-d)
- 额外：每 episode 计算确定性统计特征（mean RR, RMSSD, ectopy count 等）拼入 episode token

### DayEncoder → 12-layer Transformer

- d_model=512, n_heads=8, mlp_ratio=4
- 输入：1563 个 fused episode tokens（waveform + rhythm + stats）
- 位置编码：绝对时间编码（hour_sin/cos）+ 可学习位置编码
- 输出：day embedding (512-d) + contextualized episode states

### 收益

1. 完全不依赖 mamba-ssm，纯 PyTorch
2. 消除 100k token OOM 问题
3. 参数量可能略降（RhythmBranch 从 1.8M 降到 ~1M）
4. DayEncoder 参数量略升（Transformer 比 Mamba 稍大）
5. 注意力图可直接用于论文可视化

## 关键竞品（Codex 提供）

- **DeepHHF** (2025/2026): 69,663 条 24h 单导联 Holter 预测 5 年心衰。最接近的全天 Holter DL 竞争者，但是预测模型不是基础模型。
- **ECGFounder** (NEJM AI 2025): >10M ECGs, 150 labels。10s ECG FM 最强 baseline。
- **ECG-LFM** (Nature Communications 2026): >10M 12-lead ECGs, SSL, CVD prediction。
- **RhythmBERT** (2026 preprint): ECG-as-language rhythm SSL，概念上接近我们的 rhythm branch。

## 需要保护的消融实验

- local-only model (no DayEncoder)
- local + hour_sin/hour_cos only
- Day Transformer with shuffled episode order
- context cap: 1 min vs 5 min vs 1 h vs 24 h
