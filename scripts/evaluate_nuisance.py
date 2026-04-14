"""Evaluate nuisance/artifact confound check."""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.cer import compute_cer
from src.metrics.evaluation import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    window_embeds = data.get("window_embeddings", None)
    quality_masks = data.get("quality_masks", None)

    if window_embeds is None:
        print("No window_embeddings found")
        return

    if isinstance(window_embeds, np.ndarray) and window_embeds.ndim == 0:
        window_embeds = window_embeds.item()
    if isinstance(quality_masks, np.ndarray) and quality_masks.ndim == 0:
        quality_masks = quality_masks.item()

    results = {}
    for pid in window_embeds:
        we = window_embeds[pid]
        qm = quality_masks[pid] if quality_masks and pid in quality_masks else None

        # Original CER
        cer_orig = compute_cer(we, qm)

        # Permuted CER (shuffle time order)
        perm_cers = []
        for _ in range(20):
            idx = np.random.permutation(len(we))
            cer_perm = compute_cer(we[idx], qm[idx] if qm is not None else None)
            perm_cers.append(cer_perm["CER"])

        results[pid] = {
            "cer_original": cer_orig["CER"],
            "cer_permuted_mean": float(np.mean(perm_cers)),
            "cer_permuted_std": float(np.std(perm_cers)),
            "cer_diff": cer_orig["CER"] - float(np.mean(perm_cers)),
        }
        print(f"  {pid}: CER_orig={cer_orig['CER']:.4f} CER_perm={np.mean(perm_cers):.4f}")

    save_results(results, args.output)
    print(f"\nSaved nuisance results to {args.output}")


if __name__ == "__main__":
    main()
