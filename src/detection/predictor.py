from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.features.extractor import FeatureExtractor
from src.utils.text_utils import split_sentences

LABEL_NAMES = ["Low", "Medium", "High"]

# Feature weights derived from correlation analysis on OneStopEnglish
# (Higher weight = stronger positive correlation with cognitive load)
HEURISTIC_WEIGHTS = {
    "syntactic_avg_dep_depth": 0.18,
    "syntactic_clause_count": 0.12,
    "syntactic_passive_ratio": 0.08,
    "syntactic_avg_sentence_length": 0.15,
    "lexical_rare_word_ratio": 0.20,
    "lexical_subword_ratio": 0.08,
    "lexical_avg_word_length": 0.10,
    "lexical_nominalization_ratio": 0.10,
    "density_ner_density": 0.08,
    "density_concept_density": 0.10,
    "density_entity_novelty_rate": 0.06,
    "coherence_coherence_drop": 0.12,
    "coherence_coherence_std": 0.08,
}

# Normalization ranges (min, max) for each feature → map to 0–1
FEATURE_RANGES = {
    "syntactic_avg_dep_depth": (1.0, 8.0),
    "syntactic_clause_count": (0.0, 15.0),
    "syntactic_passive_ratio": (0.0, 1.0),
    "syntactic_avg_sentence_length": (5.0, 45.0),
    "lexical_rare_word_ratio": (0.0, 0.8),
    "lexical_subword_ratio": (1.0, 3.0),
    "lexical_avg_word_length": (3.0, 10.0),
    "lexical_nominalization_ratio": (0.0, 0.6),
    "density_ner_density": (0.0, 5.0),
    "density_concept_density": (0.5, 6.0),
    "density_entity_novelty_rate": (0.0, 1.0),
    "coherence_coherence_drop": (0.0, 0.5),
    "coherence_coherence_std": (0.0, 0.3),
}


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _score_to_label(score: float) -> str:
    if score < 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


class CognitiveLoadPredictor:
    def __init__(
        self,
        model_path: str = "models/detector/",
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.feature_extractor = FeatureExtractor()
        self._model = None
        self._tokenizer = None
        self._use_model = False

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._try_load_model()

    def _try_load_model(self) -> None:
        checkpoint = self.model_path / "best_model.pt"
        if not checkpoint.exists():
            return
        try:
            from transformers import AutoTokenizer

            from src.detection.classifier import CognitiveLoadModel

            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model_name = ckpt.get("model_name", "microsoft/deberta-v3-base")
            num_features = ckpt.get("num_features", 20)

            self._model = CognitiveLoadModel(
                model_name=model_name, num_features=num_features
            ).to(self.device)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._use_model = True
            print(f"Loaded trained model from {checkpoint}")
        except Exception as exc:
            print(f"Could not load model ({exc}), falling back to heuristic mode.")
            self._use_model = False

    def predict(self, text: str) -> dict:
        """Predict cognitive load for a single text.

        Returns:
            load_score: float 0–100
            load_label: "Low" | "Medium" | "High"
            confidence: float 0–1
            dimensions: dict of dimension scores
            sentence_scores: list of per-sentence scores
            method: "model" | "heuristic"
        """
        from src.utils.text_utils import truncate_text
        text = truncate_text(text, max_chars=2000)
        if self._use_model and self._model is not None:
            return self._model_predict(text)
        return self._heuristic_predict(text)

    def _model_predict(self, text: str) -> dict:
        fv = self.feature_extractor.extract(text)
        features = fv.to_list()
        target_len = 20
        if len(features) < target_len:
            features = features + [0.0] * (target_len - len(features))
        else:
            features = features[:target_len]

        encoding = self._tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        feat_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            score_pred, logits = self._model(input_ids, attention_mask, feat_tensor)

        score = float(score_pred[0].item())
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        label = LABEL_NAMES[label_idx]

        flat = fv.to_flat_dict()
        dimensions = self._compute_dimensions(flat)
        sentence_scores = self._compute_sentence_scores(text)

        return {
            "load_score": round(score, 1),
            "load_label": label,
            "confidence": round(confidence, 3),
            "dimensions": dimensions,
            "sentence_scores": sentence_scores,
            "method": "model",
        }

    def _heuristic_predict(self, text: str) -> dict:
        """Fallback using weighted feature combination to estimate load score."""
        fv = self.feature_extractor.extract(text)
        flat = fv.to_flat_dict()

        weighted_sum = 0.0
        weight_total = 0.0
        for feat_key, weight in HEURISTIC_WEIGHTS.items():
            if feat_key in flat and feat_key in FEATURE_RANGES:
                lo, hi = FEATURE_RANGES[feat_key]
                norm_val = _normalize(flat[feat_key], lo, hi)
                weighted_sum += norm_val * weight
                weight_total += weight

        if weight_total > 0:
            normalized_score = weighted_sum / weight_total
        else:
            normalized_score = 0.5

        score = normalized_score * 100.0
        label = _score_to_label(score)

        # Rough confidence based on feature completeness
        available = sum(1 for k in HEURISTIC_WEIGHTS if k in flat)
        confidence = 0.4 + 0.4 * (available / len(HEURISTIC_WEIGHTS))

        dimensions = self._compute_dimensions(flat)
        sentence_scores = self._compute_sentence_scores(text)

        return {
            "load_score": round(score, 1),
            "load_label": label,
            "confidence": round(confidence, 3),
            "dimensions": dimensions,
            "sentence_scores": sentence_scores,
            "method": "heuristic",
        }

    def _compute_dimensions(self, flat: dict[str, float]) -> dict[str, float]:
        """Aggregate per-dimension load scores (0–100 each)."""

        def dim_score(keys: list[str]) -> float:
            vals = []
            for k in keys:
                if k in flat and k in FEATURE_RANGES:
                    lo, hi = FEATURE_RANGES[k]
                    vals.append(_normalize(flat[k], lo, hi) * 100)
            return round(sum(vals) / len(vals), 1) if vals else 50.0

        return {
            "syntactic": dim_score([
                "syntactic_avg_dep_depth",
                "syntactic_clause_count",
                "syntactic_passive_ratio",
                "syntactic_avg_sentence_length",
            ]),
            "lexical": dim_score([
                "lexical_rare_word_ratio",
                "lexical_avg_word_length",
                "lexical_nominalization_ratio",
            ]),
            "density": dim_score([
                "density_ner_density",
                "density_concept_density",
                "density_entity_novelty_rate",
            ]),
            "coherence": dim_score([
                "coherence_coherence_drop",
                "coherence_coherence_std",
            ]),
        }

    def _compute_sentence_scores(self, text: str) -> list[float]:
        """Compute per-sentence heuristic load scores."""
        sentences = split_sentences(text)
        if not sentences:
            return [50.0]

        scores: list[float] = []
        for sent in sentences:
            if len(sent.strip()) < 5:
                scores.append(50.0)
                continue
            try:
                result = self._heuristic_predict(sent)
                scores.append(result["load_score"])
            except Exception:
                scores.append(50.0)
        return scores

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch prediction with tqdm."""
        return [self.predict(t) for t in tqdm(texts, desc="Predicting")]
