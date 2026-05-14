"""
FORGE — Model adapters
One interface. Any LLM. Drop in a new provider without touching eval logic.
"""

from __future__ import annotations
import os
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    text:          str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    model:         str
    raw:           Any = None          # original SDK response, if needed


# ─────────────────────────────────────────────
# Base adapter
# ─────────────────────────────────────────────

class BaseAdapter(ABC):
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature   # 0.0 = deterministic for evals

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        ...

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 2)


# ─────────────────────────────────────────────
# Anthropic — Claude
# ─────────────────────────────────────────────

class ClaudeAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models."""

    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        t0 = time.time()

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        r = client.messages.create(**kwargs)
        return ModelResponse(
            text=r.content[0].text,
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            latency_ms=self._ms(t0),
            model=self.model,
            raw=r,
        )


# ─────────────────────────────────────────────
# OpenAI — GPT / o-series
# ─────────────────────────────────────────────

class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI GPT and o-series models."""

    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        t0 = time.time()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return ModelResponse(
            text=r.choices[0].message.content or "",
            input_tokens=r.usage.prompt_tokens,
            output_tokens=r.usage.completion_tokens,
            latency_ms=self._ms(t0),
            model=self.model,
            raw=r,
        )


# ─────────────────────────────────────────────
# Google — Gemini
# ─────────────────────────────────────────────

class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models."""

    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("pip install google-generativeai")

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system or None,
        )
        t0 = time.time()
        r = model.generate_content(prompt)
        usage = getattr(r, "usage_metadata", None)

        return ModelResponse(
            text=r.text,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            latency_ms=self._ms(t0),
            model=self.model,
            raw=r,
        )


# ─────────────────────────────────────────────
# Ollama — local models (Llama 3, Mistral, etc.)
# ─────────────────────────────────────────────

class OllamaAdapter(BaseAdapter):
    """Adapter for models served locally via Ollama."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url

    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        import urllib.request, json

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }).encode()

        t0 = time.time()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        return ModelResponse(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=self._ms(t0),
            model=self.model,
            raw=data,
        )


# ─────────────────────────────────────────────
# Mock adapter — for CI/testing without API keys
# ─────────────────────────────────────────────

class MockAdapter(BaseAdapter):
    """Deterministic mock — returns a stable fake response for testing."""

    def __init__(self, model: str = "mock", responses: dict[str, str] | None = None):
        super().__init__(model)
        self._responses = responses or {}

    def complete(self, prompt: str, system: str = "") -> ModelResponse:
        key = hashlib.md5(prompt.encode()).hexdigest()[:8]
        text = self._responses.get(prompt, f"[mock response {key}]")
        return ModelResponse(
            text=text, input_tokens=len(prompt.split()),
            output_tokens=len(text.split()), latency_ms=1.0, model=self.model,
        )


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

ADAPTER_MAP = {
    "claude":  ClaudeAdapter,
    "gpt":     OpenAIAdapter,
    "gemini":  GeminiAdapter,
    "ollama":  OllamaAdapter,
    "mock":    MockAdapter,
}

def get_adapter(provider: str, model: str, **kwargs) -> BaseAdapter:
    """
    Usage:
        adapter = get_adapter("claude", "claude-sonnet-4-6")
        response = adapter.complete("What is 2+2?")
    """
    key = provider.lower()
    cls = ADAPTER_MAP.get(key)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(ADAPTER_MAP)}")
    return cls(model=model, **kwargs)
