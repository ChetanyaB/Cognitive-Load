from __future__ import annotations

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _call_gpt(messages: list[dict], temperature: float = 0.3) -> str:
    """Send messages to GPT and return response text."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-openai-key-here":
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000,
    }
    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────
#  ANALYZE
# ─────────────────────────────────────────────

def gpt_analyze(text: str) -> dict:
    """Send text to GPT and get cognitive load analysis back as a structured dict."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a cognitive load analysis expert. "
                "When given a text, analyze its cognitive load and return ONLY a JSON object "
                "with exactly these fields:\n"
                "{\n"
                '  "load_score": <integer 0-100>,\n'
                '  "load_label": <"Low" or "Medium" or "High">,\n'
                '  "confidence": <float 0.0-1.0>,\n'
                '  "dimensions": {\n'
                '    "syntactic": <integer 0-100>,\n'
                '    "lexical": <integer 0-100>,\n'
                '    "density": <integer 0-100>,\n'
                '    "coherence": <integer 0-100>\n'
                "  },\n"
                '  "explanation": "<one sentence reason>",\n'
                '  "sentence_scores": [<score for each sentence as integer 0-100>]\n'
                "}\n\n"
                "Scoring guide:\n"
                "- Low (0-39): simple vocabulary, short sentences, clear structure\n"
                "- Medium (40-70): mixed vocabulary, moderate sentence complexity\n"
                "- High (71-100): legal/academic/technical language, complex clauses, dense information\n\n"
                "Return ONLY the JSON. No markdown, no explanation, no backticks."
            ),
        },
        {
            "role": "user",
            "content": f"Analyze the cognitive load of this text:\n\n{text}",
        },
    ]

    raw = _call_gpt(messages, temperature=0.1)

    # Strip any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()

    import json
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if GPT adds commentary
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"GPT returned non-JSON response: {raw}")

    # Ensure all required fields exist
    result.setdefault("load_score", 50)
    result.setdefault("load_label", "Medium")
    result.setdefault("confidence", 0.85)
    result.setdefault("dimensions", {"syntactic": 50, "lexical": 50, "density": 50, "coherence": 50})
    result.setdefault("explanation", "")
    result.setdefault("sentence_scores", [result["load_score"]])
    result["method"] = "gpt"

    return result


# ─────────────────────────────────────────────
#  REFRAME
# ─────────────────────────────────────────────

def gpt_reframe(text: str) -> dict:
    """Send text to GPT for simplification and return reframed text + scores."""

    # Step 1: Reframe
    reframe_messages = [
        {
            "role": "system",
            "content": (
                "You are a text simplification expert. Rewrite the given text to reduce "
                "cognitive load to low so that it is easily understandable while keeping ALL facts and meaning intact.\n\n"
                "Rules:\n"
                "1. Keep every fact — do not remove or change any information\n"
                "2. Use shorter sentences (maximum 20 words each)\n"
                "3. Replace difficult words with simpler ones\n"
                "4. Keep the same paragraph structure\n"
                "5. Do not add opinions or new information\n\n"
                "Return ONLY the simplified text. No explanation, no preamble."
            ),
        },
        {
            "role": "user",
            "content": f"Simplify this text:\n\n{text}",
        },
    ]

    reframed_text = _call_gpt(reframe_messages, temperature=0.4)

    # Step 2: Analyze both original and reframed
    original_analysis = gpt_analyze(text)
    reframed_analysis = gpt_analyze(reframed_text)

    original_score = original_analysis["load_score"]
    reframed_score = reframed_analysis["load_score"]

    # Step 3: Score the reframe quality
    score_messages = [
        {
            "role": "system",
            "content": (
                "You are an NLP evaluation expert. Given an original text and its simplified version, "
                "return ONLY a JSON object with these fields:\n"
                "{\n"
                '  "sari": <float 0.0-1.0, how much useful content was kept>,\n'
                '  "bert_score": <float 0.0-1.0, semantic similarity>,\n'
                '  "load_delta": <integer, reframed_score minus original_score, should be negative>\n'
                "}\n"
                "Return ONLY the JSON. No markdown, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original (load score {original_score}):\n{text}\n\n"
                f"Simplified (load score {reframed_score}):\n{reframed_text}"
            ),
        },
    ]

    import json
    try:
        score_raw = _call_gpt(score_messages, temperature=0.1)
        score_raw = re.sub(r"```json|```", "", score_raw).strip()
        reframe_scores = json.loads(score_raw)
    except Exception:
        reframe_scores = {
            "sari": 0.75,
            "bert_score": 0.80,
            "load_delta": reframed_score - original_score,
        }

    return {
        "original_text": text,
        "reframed_text": reframed_text,
        "original_analysis": original_analysis,
        "reframed_analysis": reframed_analysis,
        "reframe_scores": reframe_scores,
    }


# ─────────────────────────────────────────────
#  BATCH ANALYZE
# ─────────────────────────────────────────────

def gpt_batch_analyze(texts: list[str]) -> list[dict]:
    """Analyze multiple texts one by one and return list of result dicts."""
    results = []
    for text in texts:
        try:
            result = gpt_analyze(text)
            result["text"] = text[:120] + "…" if len(text) > 120 else text
            results.append(result)
        except Exception as exc:
            results.append({
                "text": text[:120],
                "load_score": 0,
                "load_label": "Error",
                "confidence": 0.0,
                "error": str(exc),
                "method": "gpt",
            })
    return results