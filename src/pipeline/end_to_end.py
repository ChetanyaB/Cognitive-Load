from __future__ import annotations

from src.detection.predictor import CognitiveLoadPredictor
from src.reframing.evaluator import RewriteEvaluator
from src.reframing.rewriter import TextRewriter


class CognitiveLoadPipeline:
    """End-to-end pipeline: detect cognitive load, reframe if high, evaluate."""

    def __init__(
        self,
        model_path: str = "models/detector/",
        rewrite_threshold: float = 70.0,
    ) -> None:
        self.rewrite_threshold = rewrite_threshold
        self.predictor = CognitiveLoadPredictor(model_path=model_path)
        self.rewriter = TextRewriter()
        self.evaluator = RewriteEvaluator(self.predictor)

    def run(self, text: str) -> dict:
        """Run full pipeline on a single text.

        Returns:
            original_text: str
            detection: full predictor output dict
            reframed: bool — whether reframing was triggered
            reframed_text: str | None
            reframe_scores: dict | None — {sari, bert_score, load_delta}
            final_load_score: float
            final_load_label: str
        """
        # Step 1: Detect cognitive load
        detection = self.predictor.predict(text)
        load_score = detection["load_score"]

        # Step 2: Decide whether to reframe
        if load_score <= self.rewrite_threshold:
            return {
                "original_text": text,
                "detection": detection,
                "reframed": False,
                "reframed_text": None,
                "reframe_scores": None,
                "final_load_score": detection["load_score"],
                "final_load_label": detection["load_label"],
            }

        # Step 3: Reframe
        candidates = self.rewriter.rewrite(text)

        # Step 4: Evaluate candidates
        eval_result = self.evaluator.evaluate(
            original=text,
            candidates=candidates,
            reference=None,
        )

        best_text = eval_result["best_candidate"]
        best_scores = eval_result["scores"][eval_result["best_index"]]

        # Step 5: Re-detect on best candidate
        final_detection = self.predictor.predict(best_text)

        return {
            "original_text": text,
            "detection": detection,
            "reframed": True,
            "reframed_text": best_text,
            "reframe_scores": {
                "sari": best_scores["sari"],
                "bert_score": best_scores["bert_score"],
                "load_delta": eval_result["load_delta"],
            },
            "final_load_score": final_detection["load_score"],
            "final_load_label": final_detection["load_label"],
        }

    def run_batch(self, texts: list[str]) -> list[dict]:
        """Run pipeline on multiple texts."""
        return [self.run(t) for t in texts]
