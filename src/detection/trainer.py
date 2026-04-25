from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.detection.classifier import (
    LABEL_TO_IDX,
    CognitiveLoadModel,
    compute_loss,
)
from src.features.extractor import FeatureExtractor


class CognitiveLoadDataset(Dataset):
    """Dataset that loads from JSONL, tokenizes text, and extracts features."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: AutoTokenizer,
        feature_extractor: FeatureExtractor,
        max_length: int = 512,
        cache_features: bool = True,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.records: list[dict] = []

        path = Path(jsonl_path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self._feature_cache: dict[int, list[float]] = {}
        if cache_features:
            print(f"Pre-extracting features for {len(self.records)} samples …")
            for i, rec in enumerate(tqdm(self.records, desc="Features")):
                fv = self.feature_extractor.extract(rec["text"])
                self._feature_cache[i] = fv.to_list()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec["text"]
        score = float(rec["load_score"])
        label_str = rec.get("load_label", "Medium")
        label = LABEL_TO_IDX.get(label_str, 1)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        if idx in self._feature_cache:
            features = self._feature_cache[idx]
        else:
            fv = self.feature_extractor.extract(text)
            features = fv.to_list()

        # Pad/truncate feature vector to 20 dims
        target_len = 20
        if len(features) < target_len:
            features = features + [0.0] * (target_len - len(features))
        else:
            features = features[:target_len]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "feature_vector": torch.tensor(features, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train(
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/val.jsonl",
    output_dir: str = "models/detector/",
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    max_length: int = 512,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extractor = FeatureExtractor(hf_model_name=model_name)

    print("Loading training data …")
    train_dataset = CognitiveLoadDataset(train_path, tokenizer, feature_extractor, max_length)
    val_dataset = CognitiveLoadDataset(val_path, tokenizer, feature_extractor, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CognitiveLoadModel(model_name=model_name, num_features=20).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        train_loss_total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feature_vector = batch["feature_vector"].to(device)
            scores = batch["score"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            score_pred, logits = model(input_ids, attention_mask, feature_vector)
            loss = compute_loss(score_pred, scores, logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        # ---------- VALIDATE ----------
        model.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                feature_vector = batch["feature_vector"].to(device)
                scores = batch["score"].to(device)
                labels = batch["label"].to(device)

                score_pred, logits = model(input_ids, attention_mask, feature_vector)
                loss = compute_loss(score_pred, scores, logits, labels)
                val_loss_total += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "model_name": model_name,
                    "num_features": 20,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best checkpoint → {checkpoint_path}")

    # Save tokenizer alongside model
    tokenizer.save_pretrained(str(output_path))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cognitive load detector")
    parser.add_argument("--train_path", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/detector/")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
    )
