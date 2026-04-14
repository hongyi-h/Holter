"""5-fold CV logistic regression linear probe evaluation."""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.evaluation import linear_probe_cv, save_results
from src.nlp.report_parser import LABEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npz")
    parser.add_argument("--labels", required=True, help="Path to labels .npz")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    emb_data = np.load(args.embeddings, allow_pickle=True)
    lab_data = np.load(args.labels, allow_pickle=True)

    X = emb_data["day_embeddings"]
    label_matrix = lab_data["labels"]
    label_names = list(lab_data.get("label_names", LABEL_NAMES))

    print(f"Embeddings: {X.shape}")
    print(f"Labels: {label_matrix.shape}")

    results = {}
    for li, name in enumerate(label_names):
        y = label_matrix[:, li]
        n_pos = y.sum()
        if n_pos < 5 or (len(y) - n_pos) < 5:
            print(f"  {name}: SKIP (pos={n_pos}, neg={len(y)-n_pos})")
            continue

        probe = linear_probe_cv(X, y, n_folds=args.n_folds)
        auroc_std = float(np.std(probe["fold_aucs"]))
        print(f"  {name}: AUROC={probe['auroc']:.3f} +/- {auroc_std:.3f}")
        results[name] = {
            "auroc": probe["auroc"],
            "auroc_std": auroc_std,
            "auroc_ci": probe["auroc_ci"],
            "auprc": probe["auprc"],
            "n_pos": int(n_pos),
            "n_total": len(y),
        }

    save_results(results, args.output)
    print(f"\nSaved probe results to {args.output}")


if __name__ == "__main__":
    main()
