"""R001-R003: Data sanity checks for M0 milestone."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.holter_record import HolterRecord, SAMPLE_RATE
from src.data.report_concepts import ReportConceptExtractor


def r001_beat_extraction(records: list[HolterRecord]) -> dict:
    """Validate beat extraction and RR stats against report."""
    results = {"pass": 0, "fail": 0, "errors": []}
    for rec in records:
        try:
            beats = rec.load_beats()
            report = rec.load_report()

            n_beats_ann = len(beats.times)
            n_beats_report = report.total_beats

            # beat count: allow ±1% tolerance (annotation may include/exclude boundary beats)
            tol = max(n_beats_report * 0.01, 50)
            count_ok = abs(n_beats_ann - n_beats_report) <= tol

            # mean HR from RR intervals
            rr = np.diff(beats.times)
            if len(rr) > 0:
                mean_hr_computed = 60.0 / np.mean(rr)
            else:
                mean_hr_computed = 0
            hr_ok = abs(mean_hr_computed - report.mean_hr) <= 5

            # duration check — annotations may not start at t=0
            duration_ann = beats.times[-1] if len(beats.times) > 1 else 0
            h, m = map(int, report.duration_hhmm.split(":"))
            duration_report = h * 3600 + m * 60
            dur_ok = abs(duration_ann - duration_report) <= 600  # 10 min tolerance

            ok = count_ok and hr_ok and dur_ok
            if ok:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["errors"].append({
                    "record": rec.record_id,
                    "beat_count": {"ann": n_beats_ann, "report": n_beats_report, "ok": count_ok},
                    "mean_hr": {"computed": round(mean_hr_computed, 1), "report": report.mean_hr, "ok": hr_ok},
                    "duration": {"ann": round(duration_ann), "report": duration_report, "ok": dur_ok},
                })
            rec.free()
        except Exception as e:
            results["fail"] += 1
            results["errors"].append({"record": rec.record_id, "exception": str(e)})
    return results


def r002_pvc_consistency(records: list[HolterRecord]) -> dict:
    """Validate PVC counts from annotations match report."""
    results = {"pass": 0, "fail": 0, "errors": [], "stats": []}
    for rec in records:
        try:
            beats = rec.load_beats()
            report = rec.load_report()

            v_count_ann = int(np.sum(beats.labels == 1))
            v_count_report = report.pvc_count

            # strict: must match exactly (both come from same system)
            ok = v_count_ann == v_count_report

            burden_ann = v_count_ann / max(len(beats.labels), 1) * 100
            burden_report = report.pvc_pct

            results["stats"].append({
                "record": rec.record_id,
                "v_ann": v_count_ann, "v_report": v_count_report,
                "burden_ann": round(burden_ann, 2), "burden_report": burden_report,
                "match": ok,
            })

            if ok:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["errors"].append({
                    "record": rec.record_id,
                    "v_ann": v_count_ann, "v_report": v_count_report,
                    "diff": v_count_ann - v_count_report,
                })
            rec.free()
        except Exception as e:
            results["fail"] += 1
            results["errors"].append({"record": rec.record_id, "exception": str(e)})
    return results


def r003_report_concepts(records: list[HolterRecord]) -> dict:
    """Build and validate report concept ontology."""
    extractor = ReportConceptExtractor()
    concept_counts = Counter()
    total = 0
    examples = {}

    for rec in records:
        try:
            report = rec.load_report()
            conclusion = report.conclusion
            if not conclusion:
                continue
            total += 1
            hits = extractor.extract_hits(conclusion)
            for h in hits:
                concept_counts[h.name] += 1
                if h.name not in examples:
                    examples[h.name] = {"matched": h.matched, "record": rec.record_id}
            rec.free()
        except Exception:
            continue

    prevalence = {k: {"count": v, "pct": round(v / max(total, 1) * 100, 1)}
                  for k, v in concept_counts.most_common()}

    # filter: keep concepts with prevalence 2%-80%
    valid_concepts = [k for k, v in prevalence.items()
                      if 2.0 <= v["pct"] <= 80.0]

    return {
        "total_reports": total,
        "n_concepts_defined": extractor.n_concepts,
        "n_concepts_valid": len(valid_concepts),
        "valid_concepts": valid_concepts,
        "prevalence": prevalence,
        "examples": examples,
    }


def r004_ventricular_events(records: list[HolterRecord]) -> dict:
    """Count ventricular event patterns across dataset for Task 5 feasibility."""
    import re
    totals = {"bigeminy": 0, "trigeminy": 0, "couplet": 0, "v_run": 0, "isolated_pvc": 0}
    per_record = []

    for rec in records:
        try:
            beats = rec.load_beats()
            label_str = "".join(["N" if l == 0 else "V" if l == 1 else "F" for l in beats.labels])

            bi = len(re.findall(r"(?=NVNVNV)", label_str))
            tri = len(re.findall(r"(?=NNVNNV)", label_str))
            runs = re.findall(r"V{2,}", label_str)
            couplets = sum(1 for r in runs if len(r) == 2)
            v_runs = sum(1 for r in runs if len(r) >= 3)
            isolated = label_str.count("V") - sum(len(r) for r in runs)

            totals["bigeminy"] += bi
            totals["trigeminy"] += tri
            totals["couplet"] += couplets
            totals["v_run"] += v_runs
            totals["isolated_pvc"] += isolated

            per_record.append({
                "record": rec.record_id,
                "bigeminy": bi, "trigeminy": tri,
                "couplet": couplets, "v_run": v_runs,
                "total_pvc": int(np.sum(beats.labels == 1)),
            })
            rec.free()
        except Exception:
            continue

    return {
        "totals": totals,
        "n_records": len(per_record),
        "records_with_bigeminy": sum(1 for r in per_record if r["bigeminy"] > 0),
        "records_with_trigeminy": sum(1 for r in per_record if r["trigeminy"] > 0),
        "records_with_couplet": sum(1 for r in per_record if r["couplet"] > 0),
        "records_with_v_run": sum(1 for r in per_record if r["v_run"] > 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M0: Data sanity checks")
    parser.add_argument("--data_dir", type=str, default="data/DMS")
    parser.add_argument("--output", type=str, default="sanity_results.json")
    args = parser.parse_args()

    records = HolterRecord.discover(args.data_dir)
    print(f"Found {len(records)} recordings")

    print("\n=== R001: Beat extraction & RR stats ===")
    r001 = r001_beat_extraction(records)
    print(f"  Pass: {r001['pass']}, Fail: {r001['fail']}")
    if r001["errors"]:
        for e in r001["errors"][:5]:
            print(f"  Error: {e}")

    print("\n=== R002: PVC consistency ===")
    r002 = r002_pvc_consistency(records)
    print(f"  Pass: {r002['pass']}, Fail: {r002['fail']}")
    if r002["errors"]:
        for e in r002["errors"][:5]:
            print(f"  Error: {e}")

    print("\n=== R003: Report concept ontology ===")
    r003 = r003_report_concepts(records)
    print(f"  Total reports: {r003['total_reports']}")
    print(f"  Valid concepts ({r003['n_concepts_valid']}):")
    for c in r003["valid_concepts"]:
        p = r003["prevalence"][c]
        ex = r003["examples"].get(c, {}).get("matched", "")
        print(f"    {c}: {p['count']} ({p['pct']}%) — e.g. '{ex}'")

    print("\n=== R004: Ventricular event prevalence ===")
    r004 = r004_ventricular_events(records)
    print(f"  Totals: {r004['totals']}")
    print(f"  Records with bigeminy: {r004['records_with_bigeminy']}/{r004['n_records']}")
    print(f"  Records with couplet: {r004['records_with_couplet']}/{r004['n_records']}")
    print(f"  Records with V-run: {r004['records_with_v_run']}/{r004['n_records']}")

    all_results = {"r001": r001, "r002": r002, "r003": r003, "r004": r004}
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
