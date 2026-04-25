from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.reframing.prompt_builder import build_prompt

load_dotenv()

OLLAMA_DEFAULT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_API_BASE = "https://api-inference.huggingface.co/models"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
TEMPERATURES = [0.3, 0.5, 0.7]


class TextRewriter:
    """Two backends, auto-detected at init:
    1. Local Ollama (preferred, free) — detected by GET to /api/tags
    2. HuggingFace Inference API (fallback) — requires HF_TOKEN in .env
    """

    def __init__(self) -> None:
        self.hf_token: str = os.getenv("HF_TOKEN", "")
        self.model_name: str = DEFAULT_MODEL
        self.ollama_url: str = OLLAMA_DEFAULT_URL
        self.backend: str = self._detect_backend()
        print(f"TextRewriter initialized with backend: {self.backend}")

    def _detect_backend(self) -> str:
        """Try Ollama first (local, free), fall back to HF API."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return "ollama"
        except Exception:
            pass
        return "huggingface"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, text: str) -> list[str]:
        """Return list of 3 candidate rewrites at different temperatures."""
        candidates: list[str] = []
        for temp in TEMPERATURES:
            try:
                result = self.rewrite_single(text, temperature=temp)
                if result and result.strip():
                    candidates.append(result.strip())
            except Exception as exc:
                print(f"Rewrite failed at temperature {temp}: {exc}")

        if not candidates:
            # Last resort: return original text so pipeline doesn't crash
            candidates = [text]

        return candidates

    def rewrite_single(self, text: str, temperature: float = 0.5) -> str:
        """Return single rewrite at given temperature."""
        messages = build_prompt(text)
        if self.backend == "ollama":
            return self._call_ollama(messages, temperature)
        return self._call_huggingface(messages, temperature)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_ollama(self, messages: list[dict], temperature: float) -> str:
        """POST to local Ollama /api/chat endpoint."""
        # Extract model name (strip org prefix if needed for Ollama)
        ollama_model = self.model_name.split("/")[-1].lower()
        # Common Ollama model aliases
        if "mistral" in ollama_model:
            ollama_model = "mistral"

        payload: dict[str, Any] = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def _call_huggingface(self, messages: list[dict], temperature: float) -> str:
        """POST to HuggingFace Inference API."""
        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Add it to .env or start a local Ollama instance."
            )

        # Convert messages to a single prompt string for the HF API
        prompt = _messages_to_prompt(messages)

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": 512,
                "return_full_text": False,
            },
        }
        url = f"{HF_API_BASE}/{self.model_name}"
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        if isinstance(data, dict):
            return data.get("generated_text", "").strip()
        return ""


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a single Mistral-style prompt string."""
    parts: list[str] = []
    system_content = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_content = content
        elif role == "user":
            if system_content:
                parts.append(f"<s>[INST] {system_content}\n\n{content} [/INST]")
                system_content = ""
            else:
                parts.append(f"<s>[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f"{content}</s>")

    return "".join(parts)
