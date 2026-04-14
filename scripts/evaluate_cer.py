"""Evaluate CER metric across arms and compute orthogonality."""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.cer import compute_cer, compute_split_half_icc
from src.metrics.evaluation import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, nargs="+", help="Embedding npz files (one per arm)")
    parser.add_argument("--hrv", default=None, help="HRV features JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}

    for emb_path in args.embeddings:
        arm_name = os.path.splitext(os.path.basename(emb_path))[0]
        data = np.load(emb_path, allow_pickle=True)
        window_embeds = data.get("window_embeddings", None)
        quality_masks = data.get("quality_masks", None)

        if window_embeds is None:
            print(f"  {arm_name}: No window_embeddings found")
            continue

        # window_embeds and quality_masks are dicts (allow_pickle)
        if isinstance(window_embeds, np.ndarray) and window_embeds.ndim == 0:
            window_embeds = window_embeds.item()
        if isinstance(quality_masks, np.ndarray) and quality_masks.ndim == 0:
            quality_masks = quality_masks.item()

        arm_results = {}
        for pid in window_embeds:
            we = window_embeds[pid]
            qm = quality_masks[pid] if quality_masks is not None and pid in quality_masks else None
            cer = compute_cer(we, qm)
            icc = compute_split_half_icc(we, qm) if cer["n_valid"] >= 20 else 0.0
            arm_results[pid] = {"cer": cer["CER"], "h1": cer["H1"], "f": cer["F"],
                                "icc": icc, "n_valid": cer["n_valid"]}

        cer_values = [v["cer"] for v in arm_results.values()]
        results[arm_name] = {
            "mean_cer": float(np.mean(cer_values)) if cer_values else 0.0,
            "std_cer": float(np.std(cer_values)) if cer_values else 0.0,
            "n_patients": len(arm_results),
            "patients": arm_results,
        }
        print(f"  {arm_name}: CER={results[arm_name]['mean_cer']:.4f} +/- {results[arm_name]['std_cer']:.4f}")

    # Orthogonality with HRV
    if args.hrv and os.path.exists(args.hrv):
        with open(args.hrv) as f:
            hrv_data = json.load(f)

        for arm_name in results:
            patients = results[arm_name]["patients"]
            common_pids = set(patients.keys()) & set(hrv_data.keys())
            if len(common_pids) < 10:
                continue
            cer_arr = np.array([patients[p]["cer"] for p in common_pids])
            sdnn_arr = np.array([hrv_data[p]["SDNN"] for p in common_pids])
            corr = np.corrcoef(cer_arr, sdnn_arr)[0, 1]
            results[arm_name]["cer_sdnn_corr"] = float(corr)
            print(f"  {arm_name} CER-SDNN correlation: {corr:.3f}")

    save_results(results, args.output)
    print(f"\nSaved CER results to {args.output}")


if __name__ == "__main__":
    main()
