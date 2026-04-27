from __future__ import annotations

from dataclasses import dataclass, field

from tqdm import tqdm

from src.features.density import extract_density_features
from src.features.lexical import extract_lexical_features
from src.features.syntactic import extract_syntactic_features


@dataclass
class FeatureVector:
    syntactic: dict[str, float] = field(default_factory=dict)
    lexical: dict[str, float] = field(default_factory=dict)
    density: dict[str, float] = field(default_factory=dict)
    coherence: dict[str, float] = field(default_factory=dict)

    # Canonical key order for to_list()
    _SYNTACTIC_KEYS = [
        "max_dep_depth",
        "avg_dep_depth",
        "clause_count",
        "passive_ratio",
        "avg_sentence_length",
        "max_sentence_length",
    ]
    _LEXICAL_KEYS = [
        "rare_word_ratio",
        "subword_ratio",
        "type_token_ratio",
        "avg_word_length",
        "nominalization_ratio",
    ]
    _DENSITY_KEYS = [
        "ner_density",
        "entity_novelty_rate",
        "avg_entities_per_sentence",
        "concept_density",
        "pronoun_ratio",
    ]
    _COHERENCE_KEYS = [
        "mean_coherence",
        "min_coherence",
        "coherence_drop",
        "coherence_std",
    ]

    def to_flat_dict(self) -> dict[str, float]:
        """Merge all sub-dicts with prefixed keys."""
        result: dict[str, float] = {}
        for k, v in self.syntactic.items():
            result[f"syntactic_{k}"] = float(v)
        for k, v in self.lexical.items():
            result[f"lexical_{k}"] = float(v)
        for k, v in self.density.items():
            result[f"density_{k}"] = float(v)
        for k, v in self.coherence.items():
            result[f"coherence_{k}"] = float(v)
        return result

    def to_list(self) -> list[float]:
        """Return ordered flat list for model input."""
        values: list[float] = []
        for k in self._SYNTACTIC_KEYS:
            values.append(float(self.syntactic.get(k, 0.0)))
        for k in self._LEXICAL_KEYS:
            values.append(float(self.lexical.get(k, 0.0)))
        for k in self._DENSITY_KEYS:
            values.append(float(self.density.get(k, 0.0)))
        for k in self._COHERENCE_KEYS:
            values.append(float(self.coherence.get(k, 0.0)))
        return values


class FeatureExtractor:
    def __init__(self, hf_model_name: str = "microsoft/deberta-v3-base"):
        """Initialize tokenizer and coherence analyzer.

        Args:
            hf_model_name: HuggingFace model for subword ratio computation.
        """
        self._tokenizer = None
        self._hf_model_name = hf_model_name
        self._coherence_analyzer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name)
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def _get_coherence_analyzer(self):
        if self._coherence_analyzer is None:
            try:
                from src.embeddings.coherence import CoherenceAnalyzer
                self._coherence_analyzer = CoherenceAnalyzer()
            except Exception:
                self._coherence_analyzer = None
        return self._coherence_analyzer

    def extract(self, text: str) -> FeatureVector:
        """Run all feature extractors and return FeatureVector."""
        syntactic = extract_syntactic_features(text)
        lexical = extract_lexical_features(text, tokenizer=self._get_tokenizer())
        density = extract_density_features(text)

        analyzer = self._get_coherence_analyzer()
        # Skip coherence for short texts (< 3 sentences) — saves 2-3 seconds
        sentences = text.count('.') + text.count('!') + text.count('?')
        if analyzer is not None and sentences >= 3:
            try:
                coherence = analyzer.analyze(text)
            except Exception:
                coherence = {
                    "mean_coherence": 0.5,
                    "min_coherence": 0.5,
                    "coherence_drop": 0.0,
                    "coherence_std": 0.0,
                }
        else:
            coherence = {
                "mean_coherence": 0.5,
                "min_coherence": 0.5,
                "coherence_drop": 0.0,
                "coherence_std": 0.0,
            }

        return FeatureVector(
            syntactic=syntactic,
            lexical=lexical,
            density=density,
            coherence=coherence,
        )

    def extract_batch(self, texts: list[str]) -> list[FeatureVector]:
        """Batch extraction with tqdm progress."""
        return [self.extract(text) for text in tqdm(texts, desc="Extracting features")]
