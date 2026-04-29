# M0 数据质量审查结果

日期: 2026-04-28
数据集: 1178 条记录 (data/DMS), 完整数据集 1585 条 (/Volumes/My Passport/Datasets/DMS/V1/)

## R001: Beat extraction & RR stats

- Pass: 901, Fail: 277
- 失败分类:
  - 仅 duration 不匹配: 181 条 (报告 duration 始终 > 标注 duration, 中位差 3420s)
  - Beat count 不匹配: 95 条 (74 条 <5%, 13 条 5-10%, 8 条 >10%)
  - HR 不匹配: 8 条 (7 条未被排除)
- **Duration 不匹配原因**: 报告中的 duration 是设备佩戴时间, 标注 duration 是最后一个 R 波时间戳。差异来自设备启动/结束时的无标注段。不影响训练。
- **HR 不匹配根因**: 标注文件中存在时间跳跃 (RR > 5s), 占比 0.1%-2.0%。过滤 [0.2s, 3.0s] 范围外的 RR 后, HR 与报告一致。已在 `_compute_day_stats` 中修复。
- **Beat count >10% 的 8 条**: 全部已被 m0_fixes.py 排除。

## R002: PVC consistency

- Pass: 1177, Fail: 1 (差 1 个 PVC, 可忽略)
- 结论: 标注中的 V 标签与报告 PVC 计数高度一致。

## R003: Report concept ontology

- 总报告: 1177
- 有效概念: 16 个 (阈值 2-90%)
- 排除: sinus_rhythm (97.0%), pause_long (99.7%), av_block (0.5%)
- svt_run: 355 条 (30.2%) — 正常工作
- 概念分布合理, 覆盖了主要的 Holter 诊断类别

## R004: Ventricular event prevalence

- bigeminy: 172,047 事件, 131 条记录 (11.1%)
- trigeminy: 186,029 事件
- couplet: 10,638 事件, 122 条记录 (10.4%)
- V-run: 2,971 事件, 39 条记录 (3.3%)
- isolated PVC: 1,045,640 事件

## 数据质量结论

- 1170/1178 条记录可用 (99.3%)
- 排除 8 条: 7 条 beat count 差异 >10%, 1 条报告缺失
- 7 条 HR 不匹配记录不需要排除, 已通过 RR 过滤修复
- 数据质量足够支撑预训练

## 代码修复记录

1. `holter_dataset.py._compute_day_stats`: HR/HRV 统计现在过滤 [0.2s, 3.0s] 范围外的 RR 间期
2. `holter_dataset.py._compute_day_stats`: label=-1 (未知拍型) 不再被错误映射为 "F"
3. `holter_dataset.py._compute_day_stats`: HR 统计使用 rr_real 而非包含重复末尾的 rr
4. `rhythm_branch.py`: 改用官方 mamba-ssm Mamba2, MPS 回退
5. `day_encoder.py`: 同上, 输出 _fused_input 供 DayMaskLoss 使用
6. `pretrain_losses.py`: DayMaskLoss 已接入训练, valid_len 转 int
7. `downstream_eval.py`: autocast import 更新
