# M2 预训练执行计划

日期: 2026-04-29 (更新)

## 架构变更 (2026-04-29)

**重大变更**: 去除 mamba-ssm 依赖，改用纯 PyTorch 架构。

- RhythmBranch: 全天 100k token BiMamba → episode-local (64拍独立处理, 4层 conv+MLP)
- DayEncoder: 12层 BiMamba → 12层标准 Transformer (d=512, 8头, 全局注意力)
- 原因: (1) mamba-ssm 在曦云 C500 上缺 libmaca_mathlib_host.so; (2) 100k token 全天节律建模临床上不必要; (3) 1563 token 的 Transformer 全局注意力更直接支持"24h上下文"叙事
- 详见: notes/architecture_decision_remove_mamba.md

## 前置条件 (已完成)

- M0 数据质量通过: 1170/1178 可用, valid_records.txt 已生成
- 架构重构完成: 纯 PyTorch, 无外部依赖
- 16 个有效报告概念 (阈值 2-90%)
- verify_model.py sanity check 通过 (512 beats, forward + loss)

## 云端执行步骤

```bash
cd /mnt/afs/zhengmingkai/hhy/Holter
git pull

# Step 1: 验证新架构 forward/backward
python scripts/verify_model.py

# Step 2: 启动预训练 (8×C500)
NUM_WORKERS=4 bash scripts/run_m2_pretrain.sh
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 数据 | 1170 条 24h Holter (70/10/20 split) |
| 模型 | HolterFM ~48M params |
| 架构 | BeatTokenizer(1.1M) + EpisodeEncoder(10.7M) + RhythmBranch(0.8M) + DayEncoder(35M) + BeatProj(0.7M) |
| Epochs | 40 |
| Batch size | 1 day/GPU × 8 GPU = 8 |
| 优化器 | AdamW, lr=2e-4, cosine → 2e-5 |
| Warmup | 2000 steps |
| 精度 | bf16 |
| 梯度裁剪 | 1.0 |
| 保存 | 每 2 epoch + best model |
| Loss 权重 | beat 0.35, episode 0.20, day 0.20, rhythm 0.20, report 0.05, vq 0.10 |
| Day/report ramp | 前 5 epoch 从 0 线性增到 1 |

## 预期

- 训练集: ~819 条, 验证集: ~117 条, 测试集: ~234 条
- Steps/epoch: ~103 (8 GPU, bs=1/GPU)
- 总 steps: ~4120
- 预计时间: 20-40h (取决于 C500 性能)

## 监控要点

1. 前 10 步: loss 是否下降, 无 NaN
2. Epoch 0 结束: 各 loss 分量是否合理
3. Epoch 5: day/report loss ramp-up 后是否稳定
4. 验证 loss 是否持续下降
5. beat_mae 应从 ~19 逐步下降
6. ep_cpc 应从 ~1.9 (random) 下降

## 如果出问题

- OOM: 减少 num_workers (`NUM_WORKERS=2`) 或加 gradient checkpointing
- Loss 不降: 检查 lr, 可能需要降到 1e-4
- NaN: 检查 grad_clip, 可能需要降到 0.5
