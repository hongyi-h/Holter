# M2 Smoke Test: 5 Epoch 预训练结果分析

日期: 2026-05-03
环境: 8×MetaX C500 (64GB), DDP, bf16
数据: 1170条 (819 train / 117 val / 234 test), max_beats=80000
训练时间: ~66 min (5 epochs × ~790s/epoch), 103 steps/epoch, 515 total steps

## Loss 趋势 (epoch 平均)

| Loss | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | 趋势 |
|------|---------|---------|---------|---------|---------|------|
| **total** | 7.80 | 13.13 | 13.82 | 18.80 | 26.13 | ⚠️ 上升 |
| **beat_mae** | 12.34 | 11.50 | 9.76 | 8.67 | 8.37 | ✅ 稳定下降 |
| **ep_cpc** | 6.66 | 5.80 | 5.40 | 4.87 | 4.71 | ✅ 稳定下降 |
| **ep_align** | 6.60 | 6.31 | 6.23 | 5.54 | 5.26 | ✅ 稳定下降 |
| **ep_order** | 0.70 | 0.71 | 0.71 | 0.70 | 0.70 | ⚠️ 不动 (~random) |
| **rhythm_mask** | 7.32 | 5.08 | 3.16 | 2.23 | 1.87 | ✅ 快速下降 |
| **rr_next** | 0.13 | 0.07 | 0.05 | 0.05 | 0.05 | ✅ 收敛 |
| **day_mask** | 0.069 | 0.045 | 0.048 | 0.039 | 0.034 | ✅ 缓慢下降 |
| **day_stats** | 75.8 | 313.5 | 201.1 | 229.2 | 266.4 | ❌ 爆炸 |
| **report** | 0.71 | 0.49 | 0.42 | 0.40 | 0.42 | ✅ 下降后平稳 |
| **vq** | 0.011 | 0.011 | 0.022 | 0.019 | 0.019 | ✅ 稳定 |
| **gnorm** | 0.6-7 | 2-11 | 5-25 | 3-44 | 2-38 | ⚠️ 偶尔 spike |

## Val Loss

| Loss | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|------|---------|---------|---------|---------|---------|
| total | 7.63 | 10.15 | 12.80 | 15.07 | 18.43 |
| beat_mae | 13.11 | 11.54 | 9.93 | 9.42 | 9.23 |
| ep_cpc | 6.28 | 5.87 | 5.51 | 5.38 | 5.37 |
| rhythm_mask | 6.53 | 4.25 | 2.48 | 1.79 | 1.57 |
| day_mask | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| day_stats | 236.9 | 315.7 | 271.3 | 267.7 | 292.2 |

## 关键发现

### ✅ 正常工作的组件
1. **Beat MAE**: 12.3 → 8.4，稳定下降。Beat tokenizer 在学习 P-QRS-T 形态重建。
2. **Episode CPC**: 6.66 → 4.71（random baseline ~7.1），episode encoder 学到了时序预测能力。
3. **Episode Align**: 6.60 → 5.26，波形和节律 episode token 在对齐。
4. **Rhythm Mask**: 7.32 → 1.87，快速下降。Rhythm branch 在学习 VQ code 和 RR 间期预测。
5. **RR Next**: 0.13 → 0.05，收敛。
6. **Report Concepts**: 0.71 → 0.42，下降后平稳。弱监督信号在工作。
7. **Day Mask**: 0.069 → 0.034，缓慢下降。Day encoder 的 masked episode 重建在学习。

### ❌ 严重问题: day_stats loss 爆炸
- `day_stats` 从 75 → 313 → 201 → 229 → 266，完全不收敛
- 这是 `DayStatsLoss` 的 running mean/std 归一化 + Huber loss
- 根因: running_mean/std 用 EMA (0.01) 更新太慢，前几个 batch 的统计量不稳定，导致归一化后的 target 值极大
- **day_stats 通过 ramp-up 权重影响 total loss**: epoch 0 ramp=0, epoch 5 ramp=1.0，所以 total loss 在上升
- 这个 loss 在拖累整个训练

### ⚠️ 需要关注
1. **ep_order**: 始终 ~0.70，接近 random (0.693)。这个任务可能太简单或太难，需要检查。
2. **Val day_mask = 0.00**: 因为 eval 模式下不生成 mask，所以 val 的 day_mask 始终为 0。这是预期行为但意味着无法用 val day_mask 做 model selection。
3. **Grad norm spikes**: 偶尔到 30-44，说明某些 batch 的梯度不稳定，可能与 day_stats 爆炸有关。

## 建议修复

### 必须修复: day_stats loss
选项:
1. **降低 day_stats 权重或移除**: day_stats 只是辅助任务，不是核心。可以设权重为 0 或极小值。
2. **修复归一化**: 用整个训练集的统计量预计算 mean/std，而非 EMA。
3. **Clip target**: 归一化后 clip 到 [-10, 10] 防止极端值。

推荐: 方案 1 最简单有效。day_stats 的信息已经被 report concepts 和 day_mask 覆盖。

### 可选优化
- ep_order 权重可以降低（当前已经是 episode loss 的一部分，权重不大）
- grad_clip 从 1.0 降到 0.5 可能帮助稳定性
