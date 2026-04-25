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
    print(classification_report(true_labels, pred_labels, target_names=["Low", "Medium", "High"]))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(true_idx, pred_idx)
    print(cm)

    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": acc,
        "num_samples": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cognitive load predictor")
    parser.add_argument("--model_path", type=str, default="models/detector/")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(f"data/processed/{args.split}.jsonl")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run scripts/prepare_dataset.py first.")
        return

    records = load_jsonl(data_path)
    print(f"Loaded {len(records):,} records from {data_path}")

    predictor = CognitiveLoadPredictor(model_path=args.model_path)
    results = evaluate(predictor, records, max_samples=args.max_samples)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
