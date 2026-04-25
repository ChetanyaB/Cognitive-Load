"""All tests must run with pytest and NO external API calls."""
from __future__ import annotations

import pytest

from src.features.extractor import FeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor()


def test_syntactic_features_low_complexity(extractor):
    text = "The cat sat on the mat. It was warm."
    vec = extractor.extract(text)
    assert vec.syntactic["avg_dep_depth"] < 5


def test_syntactic_features_high_complexity(extractor):
    text = (
        "The implementation of policies that disproportionately affect marginalized "
        "communities, which have historically been excluded from economic participation, "
        "necessitates comprehensive reform."
    )
    vec = extractor.extract(text)
    assert vec.syntactic["avg_dep_depth"] > 3


def test_feature_vector_has_all_keys(extractor):
    vec = extractor.extract("Hello world.")
    flat = vec.to_flat_dict()
    for key in ["syntactic_avg_dep_depth", "lexical_rare_word_ratio", "density_ner_density"]:
        assert key in flat, f"Missing key: {key}"


def test_feature_vector_all_floats(extractor):
    vec = extractor.extract("The economy grew last quarter.")
    for v in vec.to_list():
        assert isinstance(v, float), f"Non-float value: {v!r}"


def test_feature_vector_length(extractor):
    vec = extractor.extract("Short sentence.")
    assert len(vec.to_list()) == 20


def test_batch_extraction(extractor):
    texts = ["Simple text.", "The cat sat on the mat. It was a warm day."]
    results = extractor.extract_batch(texts)
    assert len(results) == 2
    for r in results:
        assert len(r.to_list()) == 20


def test_rare_word_ratio_range(extractor):
    vec = extractor.extract("The cat sat on the mat.")
    assert 0.0 <= vec.lexical["rare_word_ratio"] <= 1.0


def test_density_features_present(extractor):
    vec = extractor.extract("Apple Inc. reported record revenue in New York yesterday.")
    flat = vec.to_flat_dict()
    assert "density_ner_density" in flat
    assert flat["density_ner_density"] >= 0.0


def test_coherence_keys(extractor):
    vec = extractor.extract(
        "The dog barked. The cat ran away. Then it started to rain."
    )
    for k in ["mean_coherence", "min_coherence", "coherence_drop", "coherence_std"]:
        assert k in vec.coherence, f"Missing coherence key: {k}"
