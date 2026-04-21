"""LLM interface for benchmarks — Anthropic and OpenAI backends.

Backends (set via LLM_BACKEND env var):
  anthropic — Anthropic API (default). Set ANTHROPIC_API_KEY.
  openai    — OpenAI API. Set OPENAI_API_KEY.

Examples:
  ANTHROPIC_API_KEY=... uv run python eval_locomo.py --batch
  LLM_BACKEND=openai OPENAI_API_KEY=... LLM_MODEL=gpt-4o-mini uv run python eval_locomo.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

BACKEND = os.environ.get("LLM_BACKEND", "anthropic")
MODEL = os.environ.get("LLM_MODEL", "")

_MODEL_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}


def _effective_model() -> str:
    return MODEL or _MODEL_DEFAULTS.get(BACKEND, "")


def query(prompt: str, max_tokens: int = 300) -> str:
    """Send a prompt to the configured LLM backend."""
    if BACKEND == "anthropic":
        return _query_anthropic(prompt, max_tokens, _effective_model())
    elif BACKEND == "openai":
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ.get("OPENAI_API_KEY", "")
        return _query_openai(prompt, max_tokens, url, _effective_model(), key)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {BACKEND}. Use: anthropic, openai")


def generate_answer(question: str, context: list[dict]) -> str:
    """Generate an answer using retrieved memory context."""
    context_str = "\n".join(
        f"- [{m.get('entity', '?')}] {m.get('fact', '')}" for m in context
    )
    if not context_str.strip():
        context_str = "(no relevant memories found)"

    return query(
        f"Based on the following memories from past conversations, answer the "
        f"question concisely.\n\nMemories:\n{context_str}\n\nQuestion: {question}\n\n"
        f"Answer directly and concisely based only on the provided memories. "
        f'If the memories don\'t contain enough information, say "I don\'t have '
        f'enough information to answer this."'
    )


def judge(question: str, prediction: str, reference: str) -> dict[str, Any]:
    """Score a prediction using the configured LLM as judge."""
    text = query(
        f"You are evaluating a memory system's answer to a question.\n\n"
        f"Question: {question}\n\nReference answer: {reference}\n\n"
        f"System's answer: {prediction}\n\n"
        f"Rate the system's answer on a scale of 0.0 to 1.0:\n"
        f"- 1.0: Fully correct, contains all key information from the reference\n"
        f"- 0.75: Mostly correct, minor details missing\n"
        f"- 0.5: Partially correct, some key information present\n"
        f"- 0.25: Mostly incorrect, but contains a relevant fragment\n"
        f"- 0.0: Completely wrong or irrelevant\n\n"
        f"Respond in exactly this format:\nSCORE: <number>\nREASONING: <one sentence>",
        max_tokens=150,
    )

    score_match = re.search(r"SCORE:\s*([\d.]+)", text)
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)

    return {
        "score": float(score_match.group(1)) if score_match else 0.0,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else text[:200],
    }


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _query_anthropic(prompt: str, max_tokens: int, model: str) -> str:
    """Query the Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _query_openai(
    prompt: str, max_tokens: int, base_url: str, model: str, api_key: str,
) -> str:
    """Query the OpenAI API (or any compatible endpoint)."""
    url = f"{base_url.rstrip('/')}/chat/completions"

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if "gpt-5" in model or "o1" in model or "o3" in model:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens

    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "sift-evals/0.1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(4):
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach {base_url}: {e}") from e

    raise RuntimeError("Max retries exceeded")
