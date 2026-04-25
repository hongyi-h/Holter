"""M0 修复：数据过滤 + 概念调整 + SVT regex诊断。

在服务器上运行：python scripts/m0_fixes.py --data_dir data/DMS

输出：
1. data_quality_report.json — 每条记录的质量评估 + 排除列表
2. concept_diagnosis.json — SVT相关文本模式 + 调整后的概念集统计
3. valid_records.txt — 可用记录ID列表（供后续训练使用）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.holter_record import HolterRecord
from src.data.report_concepts import ReportConceptExtractor, CONCEPT_PATTERNS


def assess_record_quality(rec: HolterRecord) -> dict:
    """Evaluate a single record, return quality assessment."""
    result = {"record_id": rec.record_id, "usable": True, "issues": []}

    try:
        beats = rec.load_beats()
        report = rec.load_report()
    except Exception as e:
        result["usable"] = False
        result["issues"].append(f"load_error: {e}")
        rec.free()
        return result

    n_ann = len(beats.times)
    n_report = report.total_beats

    # Exclusion 1: report total_beats = 0 (missing report)
    if n_report == 0:
        result["usable"] = False
        result["issues"].append(f"report_beats_zero")
        rec.free()
        return result

    # Exclusion 2: beat count difference > 10%
    beat_diff_pct = abs(n_ann - n_report) / max(n_report, 1) * 100
    if beat_diff_pct > 10:
        result["usable"] = False
        result["issues"].append(f"beat_count_diff_{beat_diff_pct:.1f}pct (ann={n_ann}, report={n_report})")

    # Exclusion 3: too few beats (< 1000)
    if n_ann < 1000:
        result["usable"] = False
        result["issues"].append(f"too_few_beats_{n_ann}")

    # Exclusion 4: annotation duration < 1 hour
    if len(beats.times) > 1:
        ann_duration = beats.times[-1] - beats.times[0]
        if ann_duration < 3600:
            result["usable"] = False
            result["issues"].append(f"too_short_{ann_duration:.0f}s")

    # Info (not exclusion): beat count diff 5-10%
    if 5 <= beat_diff_pct <= 10:
        result["issues"].append(f"beat_count_diff_{beat_diff_pct:.1f}pct_warning")

    # Stats for reference
    rr = np.diff(beats.times)
    result["stats"] = {
        "n_beats_ann": n_ann,
        "n_beats_report": n_report,
        "beat_diff_pct": round(beat_diff_pct, 2),
        "ann_duration_s": round(float(beats.times[-1]), 1) if len(beats.times) > 0 else 0,
        "mean_hr": round(60.0 / np.mean(rr), 1) if len(rr) > 0 else 0,
        "pvc_count": int(np.sum(beats.labels == 1)),
        "pvc_burden_pct": round(float(np.sum(beats.labels == 1)) / max(n_ann, 1) * 100, 2),
    }

    rec.free()
    return result


def diagnose_svt_patterns(records: list[HolterRecord]) -> dict:
    """Search all reports for SVT-related text to fix the regex."""
    keywords = ["室上速", "室上性心动过速", "阵发性室上", "短阵室上",
                "SVT", "房速", "房性心动过速", "室上性心律失常"]
    hits = []
    total_reports = 0

    for rec in records:
        try:
            report = rec.load_report()
            c = report.conclusion
            if not c:
                rec.free()
                continue
            total_reports += 1
            for kw in keywords:
                if kw in c:
                    idx = c.index(kw)
                    snippet = c[max(0, idx - 30):idx + 50]
                    hits.append({
                        "record": rec.record_id,
                        "keyword": kw,
                        "snippet": snippet,
                    })
                    break
            rec.free()
        except Exception:
            rec.free()

    return {
        "total_reports": total_reports,
        "svt_hits": len(hits),
        "examples": hits[:30],
        "all_keywords_searched": keywords,
    }


def recount_concepts_with_adjusted_threshold(records: list[HolterRecord],
                                              min_pct: float = 2.0,
                                              max_pct: float = 90.0) -> dict:
    """Re-evaluate concept prevalence with adjusted threshold (90% instead of 80%)."""
    extractor = ReportConceptExtractor()
    concept_counts = Counter()
    total = 0

    for rec in records:
        try:
            report = rec.load_report()
            if not report.conclusion:
                rec.free()
                continue
            total += 1
            hits = extractor.extract_hits(report.conclusion)
            for h in hits:
                concept_counts[h.name] += 1
            rec.free()
        except Exception:
            rec.free()

    prevalence = {}
    for name in extractor.concept_names:
        count = concept_counts.get(name, 0)
        pct = count / max(total, 1) * 100
        in_range = min_pct <= pct <= max_pct
        prevalence[name] = {
            "count": count,
            "pct": round(pct, 1),
            "valid": in_range,
            "reason": "" if in_range else (f"too_high_{pct:.1f}%" if pct > max_pct else f"too_low_{pct:.1f}%"),
        }

    valid = [n for n, v in prevalence.items() if v["valid"]]
    return {
        "total_reports": total,
        "threshold": {"min_pct": min_pct, "max_pct": max_pct},
        "n_valid": len(valid),
        "valid_concepts": valid,
        "prevalence": prevalence,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M0 fixes: data filter + concept adjustment")
    parser.add_argument("--data_dir", type=str, default="data/DMS")
    args = parser.parse_args()

    records = HolterRecord.discover(args.data_dir)
    print(f"Found {len(records)} recordings")

    # --- 1. Data quality assessment ---
    print("\n=== Step 1: Record quality assessment ===")
    assessments = []
    for i, rec in enumerate(records):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(records)}...")
        assessments.append(assess_record_quality(rec))

    usable = [a for a in assessments if a["usable"]]
    excluded = [a for a in assessments if not a["usable"]]
    print(f"  Usable: {len(usable)}, Excluded: {len(excluded)}")
    for e in excluded:
        print(f"    EXCLUDE {e['record_id']}: {', '.join(e['issues'])}")

    # Save valid record IDs
    valid_ids = [a["record_id"] for a in usable]
    with open("valid_records.txt", "w") as f:
        for rid in valid_ids:
            f.write(rid + "\n")
    print(f"  Saved {len(valid_ids)} valid IDs to valid_records.txt")

    # --- 2. SVT regex diagnosis ---
    print("\n=== Step 2: SVT pattern diagnosis ===")
    svt_diag = diagnose_svt_patterns(records)
    print(f"  SVT-related reports found: {svt_diag['svt_hits']}/{svt_diag['total_reports']}")
    for ex in svt_diag["examples"][:10]:
        print(f"    [{ex['keyword']}] {ex['record']}: ...{ex['snippet']}...")

    # --- 3. Concept prevalence with 90% threshold ---
    print("\n=== Step 3: Concept prevalence (threshold 2-90%) ===")
    concept_result = recount_concepts_with_adjusted_threshold(records, min_pct=2.0, max_pct=90.0)
    print(f"  Valid concepts: {concept_result['n_valid']}")
    for name in concept_result["prevalence"]:
        p = concept_result["prevalence"][name]
        status = "OK" if p["valid"] else f"SKIP ({p['reason']})"
        print(f"    {name:<25} {p['count']:>5} ({p['pct']:>5.1f}%) — {status}")

    # --- Save all results ---
    all_results = {
        "quality": {
            "total": len(assessments),
            "usable": len(usable),
            "excluded": len(excluded),
            "excluded_records": [{"id": e["record_id"], "issues": e["issues"]} for e in excluded],
        },
        "svt_diagnosis": svt_diag,
        "concepts": concept_result,
    }
    with open("m0_fixes_report.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull report saved to m0_fixes_report.json")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Records: {len(usable)}/{len(records)} usable")
    print(f"  Concepts: {concept_result['n_valid']} valid (threshold 2-90%)")
    print(f"  SVT patterns: {svt_diag['svt_hits']} found")
    print(f"  Next: review m0_fixes_report.json, then proceed to M1")


if __name__ == "__main__":
    main()
