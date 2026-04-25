from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.text_utils import split_sentences


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class CoherenceAnalyzer:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def sentence_embeddings(self, text: str) -> np.ndarray:
        """Return array of shape (n_sentences, embedding_dim)."""
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings

    def analyze(self, text: str) -> dict[str, float]:
        """Compute coherence metrics between consecutive sentences.

        Returns:
            mean_coherence: avg cosine sim between consecutive sentence pairs
            min_coherence: minimum single-step cosine sim
            coherence_drop: max single-step drop in similarity
            coherence_std: std dev of consecutive similarities
        """
        sentences = split_sentences(text)

        # Single sentence — coherence not meaningful
        if len(sentences) < 2:
            return {
                "mean_coherence": 1.0,
                "min_coherence": 1.0,
                "coherence_drop": 0.0,
                "coherence_std": 0.0,
            }

        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            sims.append(sim)

        sims_arr = np.array(sims, dtype=np.float32)

        mean_coherence = float(np.mean(sims_arr))
        min_coherence = float(np.min(sims_arr))

        # Max drop: largest decrease between consecutive similarity values
        if len(sims) >= 2:
            drops = [sims[i] - sims[i + 1] for i in range(len(sims) - 1)]
            coherence_drop = float(max(drops)) if drops else 0.0
        else:
            coherence_drop = 0.0

        coherence_std = float(np.std(sims_arr))

        return {
            "mean_coherence": mean_coherence,
            "min_coherence": min_coherence,
            "coherence_drop": coherence_drop,
            "coherence_std": coherence_std,
        }
