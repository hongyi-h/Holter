# HolterFM

首个 24 小时连续心电图基础模型。

## 项目结构

```
├── src/
│   ├── data/           # 数据加载、预处理、报告概念提取
│   ├── models/         # 模型架构 (BeatTokenizer, EpisodeEncoder, RhythmBranch, DayEncoder, HolterFM)
│   ├── training/       # 预训练循环、损失函数、消融实验
│   └── evaluation/     # 下游任务评估头和训练
├── scripts/            # 运行脚本 (sanity check, pretrain, downstream, ablations)
├── configs/            # YAML 配置文件
├── notes/              # 实验记录和决策文档
├── research_proposal_final.md  # 研究提案
├── valid_records.txt   # M0 质控后的可用记录 ID 列表
└── requirements.txt    # Python 依赖
```

## 架构 (~48M params)

```
Beat Tokenizer (1.1M)     Conv1d + VQ → 256-d per beat
        ↓
Episode Encoder (10.7M)   6-layer Transformer, 64 beats → 384-d episode token
        ↓
Rhythm Branch (0.8M)      Episode-local conv+MLP → 128-d rhythm per beat/episode
        ↓
Day Encoder (35M)         12-layer Transformer over ~1563 episodes → 512-d day embedding
```

- 纯 PyTorch，无外部依赖
- 全局注意力建模 24h 时序关系

## 快速开始

```bash
# 环境
pip install -r requirements.txt

# 验证模型 forward/backward
python scripts/verify_model.py

# 预训练 (8 GPU)
NUM_WORKERS=4 bash scripts/run_m2_pretrain.sh

# 下游评估
CHECKPOINT=checkpoints/pretrain/holter_fm_best.pt bash scripts/run_m3_downstream.sh
```

## 数据

- 1170 条 24h Holter 记录 (3 通道, 128Hz, 逐拍 N/V/F 标注)
- 来源: 仁济医院 DMS 系统
- 格式: `.dat` (uint8 波形) + `_RPointProperty.txt` (R 波标注) + `_HolterSummary.csv` (报告)
- 预处理: per-channel median/MAD 标准化, clip [-5, 5]

## 实验里程碑

| 阶段 | 内容 | 状态 |
|------|------|------|
| M0 | 数据质控 | ✅ 完成 |
| M2 | 预训练 40 epochs | 待运行 |
| M3 | 下游评估 (beat分类, PVC burden, 报告概念) | 待运行 |
| M4 | 消融实验 (8 variants) | 待运行 |
| M5 | 外部验证 (MIT-BIH, INCART) | 待运行 |
