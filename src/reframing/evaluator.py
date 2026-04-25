from __future__ import annotations

from bert_score import score as bert_score_fn

from src.detection.predictor import CognitiveLoadPredictor
from src.utils.sari import compute_sari


class RewriteEvaluator:
    """Evaluates rewrite candidates using SARI, BERTScore, and load reduction."""

    def __init__(self, predictor: CognitiveLoadPredictor) -> None:
        self.predictor = predictor

    def evaluate(
        self,
        original: str,
        candidates: list[str],
        reference: str | None = None,
    ) -> dict:
        """Evaluate all candidates and return the best one.

        For each candidate:
        - Compute SARI (original as source, candidate as hypothesis)
        - Compute BERTScore F1
        - Re-run cognitive load predictor
        - Rank by: 0.4*SARI + 0.3*BERTScore_F1 + 0.3*(original_score - candidate_score)/100

        Returns:
            best_candidate: str
            best_index: int
            scores: list of per-candidate score dicts
            original_load_score: float
            load_delta: float (negative = load reduced)
        """
        if not candidates:
            raise ValueError("At least one candidate rewrite is required.")

        # Original load score
        orig_result = self.predictor.predict(original)
        original_load_score = orig_result["load_score"]

        # Build reference list for SARI
        refs = [reference] if reference else candidates[:1]

        # Compute BERTScore for all candidates at once
        try:
            P, R, F1 = bert_score_fn(
                candidates,
                [original] * len(candidates),
                lang="en",
                rescale_with_baseline=True,
                verbose=False,
            )
            bert_f1_scores = F1.tolist()
        except Exception:
            bert_f1_scores = [0.5] * len(candidates)

        scores: list[dict] = []
        for i, cand in enumerate(candidates):
            # SARI
            try:
                sari = compute_sari(original, cand, refs) / 100.0  # normalize to 0-1
            except Exception:
                sari = 0.0

            # BERTScore
            bert_f1 = float(bert_f1_scores[i]) if i < len(bert_f1_scores) else 0.5

            # Load score
            cand_result = self.predictor.predict(cand)
            cand_load = cand_result["load_score"]

            # Load reduction term (positive = load went down)
            load_reduction = (original_load_score - cand_load) / 100.0

            # Composite ranking score
            composite = 0.4 * sari + 0.3 * bert_f1 + 0.3 * load_reduction

            scores.append(
                {
                    "sari": round(sari, 4),
                    "bert_score": round(bert_f1, 4),
                    "load_score": round(cand_load, 1),
                    "composite": round(composite, 4),
                }
            )

        # Pick best candidate
        best_index = max(range(len(scores)), key=lambda i: scores[i]["composite"])
        best_candidate = candidates[best_index]
        best_load = scores[best_index]["load_score"]

        return {
            "best_candidate": best_candidate,
            "best_index": best_index,
            "scores": scores,
            "original_load_score": round(original_load_score, 1),
            "load_delta": round(best_load - original_load_score, 1),
        }
