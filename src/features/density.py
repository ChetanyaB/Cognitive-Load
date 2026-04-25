from __future__ import annotations

import spacy

from src.features.syntactic import get_nlp


def ner_density(doc: spacy.tokens.Doc) -> float:
    """Named entities per sentence."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    return len(doc.ents) / len(sentences)


def entity_novelty_rate(doc: spacy.tokens.Doc) -> float:
    """Fraction of sentences that introduce at least one new entity."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    seen_entities: set[str] = set()
    novel_sentences = 0
    for sent in sentences:
        sent_ents = {ent.text.lower() for ent in sent.ents}
        new_ents = sent_ents - seen_entities
        if new_ents:
            novel_sentences += 1
        seen_entities |= sent_ents
    return novel_sentences / len(sentences)


def avg_entities_per_sentence(doc: spacy.tokens.Doc) -> float:
    """Average number of named entities per sentence."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    counts = []
    for sent in sentences:
        counts.append(len(list(sent.ents)))
    return sum(counts) / len(counts)


def concept_density(doc: spacy.tokens.Doc) -> float:
    """Noun chunk count per sentence (measure of information density)."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    chunks = list(doc.noun_chunks)
    return len(chunks) / len(sentences)


def pronoun_ratio(doc: spacy.tokens.Doc) -> float:
    """Pronouns / total tokens. High ratio = good anaphora, low = noun-heavy/overloaded."""
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return 0.0
    pronouns = sum(1 for t in tokens if t.pos_ == "PRON")
    return pronouns / len(tokens)


def extract_density_features(text: str) -> dict[str, float]:
    """Extract all density features from text."""
    nlp = get_nlp()
    doc = nlp(text)
    return {
        "ner_density": float(ner_density(doc)),
        "entity_novelty_rate": float(entity_novelty_rate(doc)),
        "avg_entities_per_sentence": float(avg_entities_per_sentence(doc)),
        "concept_density": float(concept_density(doc)),
        "pronoun_ratio": float(pronoun_ratio(doc)),
    }
