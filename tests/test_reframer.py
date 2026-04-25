"""Test SARI implementation and prompt builder (no API calls)."""
from __future__ import annotations

from src.reframing.prompt_builder import build_prompt
from src.utils.sari import batch_sari, compute_sari, get_ngrams, tokenize


def test_tokenize_basic():
    assert tokenize("Hello world") == ["hello", "world"]


def test_tokenize_punctuation():
    result = tokenize("Hello, world! How are you?")
    assert "hello" in result
    assert "world" in result
    assert "how" in result


def test_tokenize_lowercase():
    result = tokenize("The QUICK Brown Fox")
    assert result == ["the", "quick", "brown", "fox"]


def test_get_ngrams_unigrams():
    tokens = ["the", "cat", "sat"]
    ngrams = get_ngrams(tokens, 1)
    assert ngrams[("the",)] == 1
    assert ngrams[("cat",)] == 1


def test_get_ngrams_bigrams():
    tokens = ["the", "cat", "sat"]
    ngrams = get_ngrams(tokens, 2)
    assert ngrams[("the", "cat")] == 1
    assert ngrams[("cat", "sat")] == 1


def test_get_ngrams_empty():
    ngrams = get_ngrams([], 2)
    assert len(ngrams) == 0


def test_sari_identical_texts():
    score = compute_sari("the cat sat", "the cat sat", ["the cat sat"])
    assert score > 80


def test_sari_completely_wrong():
    score = compute_sari("the cat sat on the mat", "xyz abc def ghi", ["the cat sat on the mat"])
    assert score < 30


def test_sari_returns_float():
    score = compute_sari("hello world", "hi there", ["hello world"])
    assert isinstance(score, float)


def test_sari_range():
    score = compute_sari(
        "The implementation of policies affects communities.",
        "Policies affect communities.",
        ["Policies affect many communities."],
    )
    assert 0.0 <= score <= 100.0


def test_batch_sari():
    originals = ["the cat sat", "hello world"]
    simplified = ["the cat sat", "hi there"]
    references = [["the cat sat"], ["hello world"]]
    score = batch_sari(originals, simplified, references)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_prompt_builder_returns_messages():
    msgs = build_prompt("This is a complex sentence.")
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "complex sentence" in msgs[-1]["content"]


def test_prompt_builder_has_few_shot_examples():
    msgs = build_prompt("Some text.")
    # System + 2 examples (user+assistant each) + final user = 1 + 4 + 1 = 6
    assert len(msgs) >= 5


def test_prompt_builder_system_has_rules():
    msgs = build_prompt("Any text.")
    system_content = msgs[0]["content"]
    assert "Grade 8" in system_content or "simplif" in system_content.lower()
