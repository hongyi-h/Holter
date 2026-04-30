# M0 数据质量审查结果

日期: 2026-04-28
运行环境: 云端 8×C500, /mnt/afs/zhengmingkai/hhy/Holter
数据集: 1178 条记录 (data/DMS on cloud)
完整数据集: 1585 条 (/Volumes/My Passport/Datasets/DMS/V1/, 本地)
运行命令: `bash scripts/run_m0_sanity.sh` + `python scripts/m0_fixes.py --data_dir data/DMS`

## R001: Beat extraction & RR stats

- Pass: 901, Fail: 277
- 失败分类:
  - 仅 duration 不匹配: ~181 条 (报告 duration > 标注 duration, 中位差 ~3420s)
  - Beat count 不匹配: 95 条 (74 条 <5%, 13 条 5-10%, 8 条 >10%)
  - HR 不匹配: 8 条 (7 条未被排除)
- **Duration 不匹配原因**: 报告中的 duration 是设备佩戴时间, 标注 duration 是最后一个 R 波时间戳。差异来自设备启动/结束时的无标注段。不影响训练。
- **HR 不匹配根因**: 标注文件中存在时间跳跃 (RR > 5s), 占比 0.1%-2.0%。过滤 [0.2s, 3.0s] 范围外的 RR 后, HR 与报告一致。
- **Beat count >10% 的 8 条**: 全部已被 m0_fixes.py 排除。
- 示例 fail (duration only):
  - 202108120804_青山山: ann=86399s, report=87600s (差 1201s)
  - 202108120825_蒲欢: ann=86399s, report=87000s (差 601s)

## R002: PVC consistency

- Pass: 1177, Fail: 1
- 唯一 fail: 202108170809_朱荣锁 (v_ann=21770, v_report=21769, 差 1)
- 结论: 标注中的 V 标签与报告 PVC 计数高度一致, 可信赖。

## R003: Report concept ontology

- 总报告: 1177 (1 条无法加载)
- 有效概念 (阈值 2-90%): 13 个
  - pvc_present: 854 (72.6%)
  - st_t_change: 695 (59.0%)
  - st_t_normal: 657 (55.8%)
  - pvc_couplet: 453 (38.5%)
  - svt_run: 355 (30.2%)
  - sinus_arrhythmia: 344 (29.2%)
  - pvc_trigeminy: 277 (23.5%)
  - frequent_pvc: 261 (22.2%)
  - pvc_bigeminy: 257 (21.8%)
  - frequent_pac: 241 (20.5%)
  - bundle_branch_block: 61 (5.2%)
  - atrial_fibrillation: 55 (4.7%)
  - pvc_run: 43 (3.7%)
- 排除的概念:
  - sinus_rhythm: 97.0% (太高, 无区分度)
  - sinus_bradycardia: 87.9% (太高)
  - sinus_tachycardia: 80.3% (太高)
  - pac_present: 83.1% (太高)
  - pause_long: 99.7% (太高)
  - av_block: 0.5% (太低)
- 注: m0_fixes.log 中 concept prevalence 显示 15 个 valid (阈值 2-90%), sanity_check.py 用了更严格的 5-90% 阈值得到 13 个。训练中使用全部 19 个概念的 extract_vector, 下游评估时按阈值过滤。

## R004: Ventricular event prevalence

- bigeminy: 172,047 事件, 131/1178 条记录 (11.1%)
- trigeminy: 186,029 事件
- couplet: 10,638 事件, 122/1178 条记录 (10.4%)
- V-run (≥3 consecutive V): 2,971 事件, 39/1178 条记录 (3.3%)
- isolated PVC: 1,045,640 事件

## 排除记录 (8 条)

| 记录 ID | 排除原因 |
|---------|---------|
| 202108170812_王倩_B504DF3B3 | beat_count_diff 22.2% |
| 202111121010_范金南_L00614208 | beat_count_diff 16.0% |
| 202112010945_陈丽珍_P06222335 | beat_count_diff 15.7% |
| 202112021017_蔡婷_H50850963 | beat_count_diff 14.9% |
| 202112030845_陈铭奎_P09280856 | beat_count_diff 47.8% |
| 202112031054_徐晶_K17180488 | beat_count_diff 10.4% |
| 202112300945_丁美华_P29575731 | beat_count_diff 10.4% |
| 202201030744_朱玉卿_P05895984 | report_beats_zero |

## 数据质量结论

- 1170/1178 条记录可用 (99.3%), 已写入 valid_records.txt
- 数据质量足够支撑预训练
- 不需要重跑 M0: 架构变更不影响数据质控结果

## 代码修复记录 (在 M0 审查过程中发现并修复)

1. `holter_dataset.py._compute_day_stats`: label=-1 (未知拍型) 不再被错误映射为 "F", 改为 "X"
2. `holter_dataset.py._compute_day_stats`: HR 统计使用 rr_real 而非包含重复末尾的 rr
3. `pretrain_losses.py._generate_span_mask`: valid_len 转 int 避免 torch.randint 报错
4. `downstream_eval.py`: `from torch.cuda.amp import autocast` → `from torch.amp import autocast`
5. `pretrain_losses.py`: DayMaskLoss 已接入训练 (之前定义了但未调用)
