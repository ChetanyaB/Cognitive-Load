"""Test predictor heuristic mode (no trained model needed)."""
from __future__ import annotations

from src.detection.predictor import CognitiveLoadPredictor


def test_heuristic_predict_returns_valid_structure():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    result = predictor.predict("The cat sat on the mat.")
    assert 0 <= result["load_score"] <= 100
    assert result["load_label"] in ["Low", "Medium", "High"]
    assert result["method"] == "heuristic"
    assert "dimensions" in result
    assert len(result["sentence_scores"]) >= 1


def test_heuristic_low_vs_high():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    low = predictor.predict("The sun is bright today. Birds are singing.")
    high = predictor.predict(
        "The implementation of austerity measures precipitated widespread socioeconomic ramifications, "
        "disproportionately affecting marginalized demographics and exacerbating pre-existing inequalities."
    )
    assert low["load_score"] < high["load_score"]


def test_dimensions_keys():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    result = predictor.predict("A short sentence.")
    dims = result["dimensions"]
    assert "syntactic" in dims
    assert "lexical" in dims
    assert "density" in dims
    assert "coherence" in dims


def test_score_in_range():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    for text in [
        "Hello.",
        "The quick brown fox jumps over the lazy dog.",
        "Pursuant to the aforementioned contractual obligations, the indemnification clause shall remain operative.",
    ]:
        result = predictor.predict(text)
        assert 0.0 <= result["load_score"] <= 100.0, f"Out of range for: {text}"


def test_batch_predict():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    texts = ["Simple text.", "More complex text with advanced vocabulary and nested clauses."]
    results = predictor.predict_batch(texts)
    assert len(results) == 2
    for r in results:
        assert "load_score" in r
        assert "load_label" in r


def test_confidence_range():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    result = predictor.predict("Testing confidence output range.")
    assert 0.0 <= result["confidence"] <= 1.0


def test_sentence_scores_are_list_of_floats():
    predictor = CognitiveLoadPredictor(model_path="nonexistent_path/")
    result = predictor.predict("First sentence. Second sentence. Third sentence.")
    for s in result["sentence_scores"]:
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0
