from __future__ import annotations

import spacy
from wordfreq import zipf_frequency

from src.features.syntactic import get_nlp

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
NOMINALIZATION_SUFFIXES = ("-tion", "-ness", "-ity", "-ment", "-ance", "-ence")


def rare_word_ratio(doc: spacy.tokens.Doc) -> float:
    """Fraction of content words with zipf_frequency < 4.0 (i.e., uncommon words)."""
    content_words = [t for t in doc if t.pos_ in CONTENT_POS and not t.is_stop and t.is_alpha]
    if not content_words:
        return 0.0
    rare = sum(1 for t in content_words if zipf_frequency(t.text.lower(), "en") < 4.0)
    return rare / len(content_words)


def subword_ratio(text: str, tokenizer) -> float:
    """Average subword tokens per word using a passed-in HuggingFace tokenizer."""
    words = text.split()
    if not words:
        return 1.0
    total_subwords = 0
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        total_subwords += max(len(encoded), 1)
    return total_subwords / len(words)


def type_token_ratio(doc: spacy.tokens.Doc) -> float:
    """Unique lemmas / total non-punct tokens."""
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return 0.0
    unique_lemmas = {t.lemma_.lower() for t in tokens}
    return len(unique_lemmas) / len(tokens)


def avg_word_length(doc: spacy.tokens.Doc) -> float:
    """Mean character length of content words."""
    content_words = [t for t in doc if t.pos_ in CONTENT_POS and t.is_alpha]
    if not content_words:
        return 0.0
    return sum(len(t.text) for t in content_words) / len(content_words)


def nominalization_ratio(doc: spacy.tokens.Doc) -> float:
    """Fraction of nouns whose lemma ends in a nominalization suffix."""
    nouns = [t for t in doc if t.pos_ == "NOUN" and t.is_alpha]
    if not nouns:
        return 0.0
    nominalizations = sum(
        1 for t in nouns if any(t.lemma_.lower().endswith(s) for s in NOMINALIZATION_SUFFIXES)
    )
    return nominalizations / len(nouns)


def extract_lexical_features(text: str, tokenizer=None) -> dict[str, float]:
    """Extract all lexical features from text."""
    nlp = get_nlp()
    doc = nlp(text)
    features = {
        "rare_word_ratio": float(rare_word_ratio(doc)),
        "type_token_ratio": float(type_token_ratio(doc)),
        "avg_word_length": float(avg_word_length(doc)),
        "nominalization_ratio": float(nominalization_ratio(doc)),
    }
    if tokenizer is not None:
        features["subword_ratio"] = float(subword_ratio(text, tokenizer))
    else:
        features["subword_ratio"] = 1.0
    return features
