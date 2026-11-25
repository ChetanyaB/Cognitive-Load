from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from tqdm import tqdm

from src.detection.predictor import CognitiveLoadPredictor


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


LABEL_TO_IDX = {"Low": 0, "Medium": 1, "High": 2}


def evaluate(
    predictor: CognitiveLoadPredictor,
    records: list[dict],
    max_samples: int | None = None,
) -> dict:
    if max_samples:
        records = records[:max_samples]

    pred_scores: list[float] = []
    pred_labels: list[str] = []
    true_scores: list[float] = []
    true_labels: list[str] = []

    for r in tqdm(records, desc="Evaluating"):
        result = predictor.predict(r["text"])
        pred_scores.append(result["load_score"])
        pred_labels.append(result["load_label"])
        true_scores.append(float(r["load_score"]))
        true_labels.append(r["load_label"])

    pred_idx = [LABEL_TO_IDX[l] for l in pred_labels]
    true_idx = [LABEL_TO_IDX[l] for l in true_labels]

    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = float(np.sqrt(mean_squared_error(true_scores, pred_scores)))
    acc = accuracy_score(true_idx, pred_idx)

    print("\n=== Score Regression ===")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    print("\n=== Classification ===")
    print(f"  Accuracy: {acc:.4f}")
    # labels= fixes the row order: without it sklearn sorts the strings
    # alphabetically (High, Low, Medium) and the target_names were attached to the wrong rows
    print(classification_report(true_labels, pred_labels, labels=["Low", "Medium", "High"], zero_division=0))
    report = classification_report(true_labels, pred_labels, labels=["Low", "Medium", "High"],
                                   zero_division=0, output_dict=True)
    from scipy.stats import spearmanr
    rho = float(spearmanr(true_scores, pred_scores).correlation)
    print(f"  Spearman rho (score vs. level): {rho:.3f}")

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(true_idx, pred_idx, labels=[0, 1, 2])
    print(cm)

    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "spearman_rho": rho,
        "per_class": {k: report[k] for k in ["Low", "Medium", "High"]},
        "confusion_matrix": cm.tolist(),
        "num_samples": len(records),
        "method": predictor.predict(records[0]["text"])["method"],
        "_scores": {"true": true_scores, "pred": pred_scores},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cognitive load predictor")
    parser.add_argument("--model_path", type=str, default="models/detector/")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--plot_dir", type=str, default=None, help="Save confusion matrix + score plot here")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    data_path = (Path("data/raw/onestop_parsed.jsonl") if args.split == "all"
                 else Path(f"data/processed/{args.split}.jsonl"))
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run scripts/prepare_dataset.py first.")
        return

    records = load_jsonl(data_path)
    print(f"Loaded {len(records):,} records from {data_path}")

    predictor = CognitiveLoadPredictor(model_path=args.model_path)
    results = evaluate(predictor, records, max_samples=args.max_samples)

    scores = results.pop("_scores")
    results["scores"] = scores
    if args.plot_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        ConfusionMatrixDisplay(np.array(results["confusion_matrix"]),
                               display_labels=["Low", "Medium", "High"]).plot(ax=ax[0], colorbar=False, cmap="Blues")
        ax[0].set_title(f"Confusion matrix · acc {results['accuracy']:.2f} · macro-F1 {results['macro_f1']:.2f}")
        levels = {20.0: "Elementary", 55.0: "Intermediate", 85.0: "Advanced"}
        groups = [[p for t, p in zip(scores["true"], scores["pred"]) if t == lv] for lv in levels]
        ax[1].boxplot(groups)
        ax[1].set_xticks([1, 2, 3], list(levels.values()))
        ax[1].axhline(40, ls="--", c="grey", lw=1); ax[1].axhline(70, ls="--", c="grey", lw=1)
        ax[1].set_ylabel("Predicted load score (0–100)")
        ax[1].set_title(f"Score by reading level · Spearman ρ = {results['spearman_rho']:.2f}")
        ax[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(Path(args.plot_dir) / f"evaluation_{args.split}.png", dpi=110)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
