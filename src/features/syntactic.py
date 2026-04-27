from __future__ import annotations

import spacy

# Load en_core_web_lg — never en_core_web_trf
_nlp: spacy.language.Language | None = None

SUBORDINATE_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}
PASSIVE_DEPS = {"auxpass", "nsubjpass"}


def get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _token_depth(token: spacy.tokens.Token) -> int:
    """Compute depth of a token in the dependency tree."""
    depth = 0
    current = token
    while current.head != current:
        current = current.head
        depth += 1
        if depth > 50:  # Guard against cycles
            break
    return depth


def max_dep_depth(doc: spacy.tokens.Doc) -> int:
    """Maximum depth of any token in the dependency tree."""
    if not doc:
        return 0
    return max(_token_depth(t) for t in doc)


def avg_dep_depth(doc: spacy.tokens.Doc) -> float:
    """Mean depth across all tokens."""
    if not doc:
        return 0.0
    depths = [_token_depth(t) for t in doc]
    return sum(depths) / len(depths)


def clause_count(doc: spacy.tokens.Doc) -> int:
    """Count tokens with subordinate clause dependency labels."""
    return sum(1 for t in doc if t.dep_ in SUBORDINATE_DEPS)


def passive_ratio(doc: spacy.tokens.Doc) -> float:
    """Fraction of sentences containing a passive construction."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    passive_sents = 0
    for sent in sentences:
        for token in sent:
            if token.dep_ in PASSIVE_DEPS:
                passive_sents += 1
                break
    return passive_sents / len(sentences)


def avg_sentence_length(doc: spacy.tokens.Doc) -> float:
    """Mean tokens per sentence (excluding punctuation and spaces)."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    lengths = [sum(1 for t in s if not t.is_punct and not t.is_space) for s in sentences]
    non_empty = [l for l in lengths if l > 0]
    if not non_empty:
        return 0.0
    return sum(non_empty) / len(non_empty)


def max_sentence_length(doc: spacy.tokens.Doc) -> int:
    """Max tokens in any sentence (excluding punctuation)."""
    sentences = list(doc.sents)
    if not sentences:
        return 0
    return max(sum(1 for t in s if not t.is_punct and not t.is_space) for s in sentences)


def extract_syntactic_features(text: str) -> dict[str, float]:
    """Run en_core_web_lg on text and return all syntactic features as a dict."""
    nlp = get_nlp()
    doc = nlp(text)
    return {
        "max_dep_depth": float(max_dep_depth(doc)),
        "avg_dep_depth": float(avg_dep_depth(doc)),
        "clause_count": float(clause_count(doc)),
        "passive_ratio": float(passive_ratio(doc)),
        "avg_sentence_length": float(avg_sentence_length(doc)),
        "max_sentence_length": float(max_sentence_length(doc)),
    }
