from __future__ import annotations

import re
import unicodedata


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex rules."""
    # Abbreviations that should not split
    abbrevs = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|fig|vol|approx|dept|est|govt|inc|corp|ltd)"
    # Replace abbreviation dots temporarily
    text = re.sub(rf"({abbrevs})\.", r"\1<DOT>", text, flags=re.IGNORECASE)
    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    # Restore abbreviation dots
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]
    return [s for s in sentences if s]


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove control chars."""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text to max_chars at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last sentence boundary
    last_punct = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_punct > max_chars // 2:
        return truncated[: last_punct + 1]
    return truncated


def count_syllables(word: str) -> int:
    """Estimate syllable count using vowel-cluster heuristic."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level."""
    sentences = split_sentences(text)
    words = re.findall(r"\b\w+\b", text)
    syllables = sum(count_syllables(w) for w in words)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)

    fk_grade = (
        0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
    )
    return max(0.0, fk_grade)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def char_count(text: str) -> int:
    return len(text)


def sentence_count(text: str) -> int:
    return len(split_sentences(text))
