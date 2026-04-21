"""Anthropic Batch API client for benchmark evaluation.

Collects all LLM calls (answer generation + judging), submits them as a
single batch, polls for completion, and returns results keyed by custom_id.

50% cheaper than real-time API, no RPM/TPM rate limits.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")


def create_generation_request(custom_id: str, question: str, context: list[dict]) -> Request:
    """Create a batch request for answer generation."""
    context_str = "\n".join(
        f"- [{m.get('entity', '?')}] {m.get('fact', '')}" for m in context
    )
    if not context_str.strip():
        context_str = "(no relevant memories found)"

    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=DEFAULT_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"Based on the following memories from past conversations, "
                    f"answer the question concisely.\n\nMemories:\n{context_str}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer directly and concisely based only on the provided memories. "
                    f'If the memories don\'t contain enough information, say '
                    f'"I don\'t have enough information to answer this."'
                ),
            }],
        ),
    )


def create_judge_request(custom_id: str, question: str, prediction: str, reference: str) -> Request:
    """Create a batch request for LLM judging."""
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=DEFAULT_MODEL,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"You are evaluating a memory system's answer to a question.\n\n"
                    f"Question: {question}\n\nReference answer: {reference}\n\n"
                    f"System's answer: {prediction}\n\n"
                    f"Rate the system's answer on a scale of 0.0 to 1.0:\n"
                    f"- 1.0: Fully correct, contains all key information from the reference\n"
                    f"- 0.75: Mostly correct, minor details missing\n"
                    f"- 0.5: Partially correct, some key information present\n"
                    f"- 0.25: Mostly incorrect, but contains a relevant fragment\n"
                    f"- 0.0: Completely wrong or irrelevant\n\n"
                    f"Respond in exactly this format:\nSCORE: <number>\nREASONING: <one sentence>"
                ),
            }],
        ),
    )


def submit_and_wait(requests: list[Request], poll_interval: int = 10) -> dict[str, str]:
    """Submit a batch and poll until completion.

    Returns: {custom_id: response_text} for all succeeded requests.
    """
    client = anthropic.Anthropic()

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Batch {batch_id} created, waiting for completion...")

    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.expired + counts.canceled
        done = counts.succeeded + counts.errored + counts.expired + counts.canceled
        print(
            f"  Progress: {done}/{total} "
            f"(succeeded={counts.succeeded}, errored={counts.errored}, "
            f"processing={counts.processing})",
            end="\r",
        )
        if status.processing_status == "ended":
            print()
            break
        time.sleep(poll_interval)

    # Collect results
    results: dict[str, str] = {}
    succeeded = 0
    errored = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            results[result.custom_id] = text
            succeeded += 1
        else:
            errored += 1

    print(f"Batch complete: {succeeded} succeeded, {errored} failed")
    return results


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse a judge response into score + reasoning."""
    score_match = re.search(r"SCORE:\s*([\d.]+)", text)
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    return {
        "score": float(score_match.group(1)) if score_match else 0.0,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else text[:200],
    }
