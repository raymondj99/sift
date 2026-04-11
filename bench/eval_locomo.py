#!/usr/bin/env python3
"""LoCoMo benchmark evaluation for Cortex.

Downloads LoCoMo dataset (10 conversations, 1,986 QA pairs, 5 categories).
Categories: 0=single_hop, 1=multi_hop, 2=temporal_reasoning, 3=open_domain, 4=adversarial.

Usage:
    uv run python eval_locomo.py
    uv run python eval_locomo.py --limit 2
    uv run python eval_locomo.py --no-judge
    uv run python eval_locomo.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from llm import generate_answer
from scoring import score_batch, token_f1
from sift_client import SiftClient

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_PATH = Path(__file__).parent / "results" / "locomo10.json"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_NAMES = {
    0: "single_hop",
    1: "multi_hop",
    2: "temporal_reasoning",
    3: "open_domain",
    4: "adversarial",
}

SOTA = {
    "Zep": 75.1,
    "Full-context": 73.0,
    "Mem0g": 68.4,
    "Mem0": 66.9,
}


def _extract_entity_names(question: str, known_entities: list[str]) -> list[str]:
    """Extract entity names mentioned in a question.

    Matches known entity names (case-insensitive) against the question text.
    Returns the matching entity names for use as a recall filter.
    """
    q_lower = question.lower()
    return [name for name in known_entities if name.lower() in q_lower]


def load_locomo() -> list[dict]:
    """Download and cache the LoCoMo dataset."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Downloading LoCoMo from {DATA_URL}...")
    with urllib.request.urlopen(DATA_URL, timeout=30) as resp:
        data = json.loads(resp.read())

    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)
    return data


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 15]


def ingest_conversation(client: SiftClient, sample: dict):
    """Ingest a LoCoMo conversation as atomic facts.

    Stores one fact per observation (PropMem-style atomic propositions)
    for maximum retrieval precision.
    """
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")

    client.remember(
        entity=speaker_a,
        entity_type="person",
        observations=[f"{speaker_a} is a conversation participant"],
        relationships=[{"target": speaker_b, "relation": "talks_with"}],
    )

    # 1. Session summaries — highest signal per token
    for summary in sample.get("session_summary", []):
        if not summary or len(summary) < 20:
            continue
        facts = _split_sentences(summary)
        if facts:
            client.remember(entity=speaker_a, entity_type="person", observations=facts[:20])

    # 2. Event summaries
    for event_sum in sample.get("event_summary", []):
        if not event_sum or len(event_sum) < 20:
            continue
        facts = _split_sentences(event_sum)
        if facts:
            client.remember(entity=speaker_a, entity_type="person", observations=facts[:10])

    # 3. Individual dialogue turns — atomic, with date context.
    #    The turn's "speaker" field contains the actual name (e.g., "Caroline"),
    #    NOT "speaker_a"/"speaker_b".
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith(("_date_time", "_summary")):
            continue
        if not isinstance(value, list):
            continue

        session_date = conv.get(f"{key}_date_time", "")
        date_prefix = f"[{session_date}] " if session_date else ""

        for turn in value:
            text = turn.get("text", "")
            if not text or len(text) < 20:
                continue
            name = turn.get("speaker", "unknown")
            fact = f"{date_prefix}{name}: {text}"[:500]
            client.remember(entity=name, entity_type="person", observations=[fact])


def _generate_one(item: tuple) -> dict:
    """Generate an answer for one QA pair (used by ThreadPoolExecutor)."""
    question, reference, category, memories = item
    prediction = generate_answer(question, memories)
    return {
        "question": question,
        "prediction": prediction,
        "reference": reference,
        "category": category,
        "num_memories_retrieved": len(memories),
    }


def run_locomo(
    limit: int | None = None,
    use_judge: bool = True,
    dry_run: bool = False,
    use_batch: bool = False,
):
    """Run the full LoCoMo evaluation."""
    data = load_locomo()
    if limit:
        data = data[:limit]

    total_qa = sum(len(s.get("qa", [])) for s in data)
    mode = "batch" if use_batch else ("dry-run" if dry_run else "full")
    print(f"LoCoMo: {len(data)} conversations, {total_qa} QA pairs ({mode})")

    all_results: list[dict] = []
    start_time = time.time()

    for sample in data:
        conv_id = sample.get("sample_id", "unknown")
        qa_pairs = sample.get("qa", [])
        if not qa_pairs:
            continue

        index_name = f"bench-locomo-{conv_id}"
        try:
            # Phase 1: Ingest + Recall (needs MCP connection)
            with SiftClient(index=index_name) as client:
                print(f"\n[{conv_id}] Ingesting conversation...")
                ingest_conversation(client, sample)

                # Extract speaker names for entity-filtered recall
                conv = sample.get("conversation", {})
                speakers = [
                    n for n in [conv.get("speaker_a"), conv.get("speaker_b")] if n
                ]

                print(f"[{conv_id}] Recalling {len(qa_pairs)} questions...")
                recalled = []
                for qa in tqdm(qa_pairs, desc=f"  {conv_id} recall", leave=False):
                    question = qa.get("question", "")
                    reference = qa.get("answer", "")
                    cat_id = qa.get("category", -1)
                    category = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
                    if not question or not reference:
                        continue

                    # Entity-filtered recall: scope to entities mentioned in the question
                    mentioned = _extract_entity_names(question, speakers)
                    memories = client.recall(
                        question,
                        limit=20,
                        entity_names=mentioned or None,
                    )
                    recalled.append((question, str(reference), category, memories))

            # Phase 2: Generate answers
            if dry_run:
                for question, reference, category, memories in recalled:
                    prediction = " | ".join(
                        m.get("fact", "") for m in memories
                    ) or "(no memories)"
                    all_results.append({
                        "question": question, "prediction": prediction,
                        "reference": reference, "category": category,
                        "num_memories_retrieved": len(memories),
                    })
            elif use_batch:
                # Collect for batch submission later
                for question, reference, category, memories in recalled:
                    all_results.append({
                        "question": question, "prediction": "",
                        "reference": reference, "category": category,
                        "num_memories_retrieved": len(memories),
                        "_memories": memories,
                    })
            else:
                print(f"[{conv_id}] Generating {len(recalled)} answers (10 parallel)...")
                with ThreadPoolExecutor(max_workers=10) as pool:
                    futures = [pool.submit(_generate_one, r) for r in recalled]
                    for f in tqdm(as_completed(futures), total=len(futures),
                                  desc=f"  {conv_id} generate", leave=False):
                        all_results.append(f.result())

        except Exception as e:
            print(f"\n  Error on {conv_id}: {e}", file=sys.stderr)
            continue

    # Batch mode: submit all generation + judging via Anthropic Batch API
    if use_batch and all_results:
        from batch import (
            create_generation_request, create_judge_request,
            submit_and_wait, parse_judge_response,
        )

        # Step 1: Submit generation batch
        gen_requests = []
        for i, r in enumerate(all_results):
            gen_requests.append(
                create_generation_request(f"gen-{i}", r["question"], r.get("_memories", []))
            )

        print(f"\n--- Batch: generating {len(gen_requests)} answers ---")
        gen_results = submit_and_wait(gen_requests)

        for i, r in enumerate(all_results):
            r["prediction"] = gen_results.get(f"gen-{i}", "(batch failed)")
            r.pop("_memories", None)

        # Step 2: Submit judge batch
        if use_judge:
            judge_requests = []
            for i, r in enumerate(all_results):
                judge_requests.append(
                    create_judge_request(f"judge-{i}", r["question"], r["prediction"], r["reference"])
                )

            print(f"\n--- Batch: judging {len(judge_requests)} answers ---")
            judge_results = submit_and_wait(judge_requests)

            for i, r in enumerate(all_results):
                text = judge_results.get(f"judge-{i}", "SCORE: 0.0\nREASONING: batch failed")
                parsed = parse_judge_response(text)
                r["judge_score"] = parsed["score"]
                r["judge_reasoning"] = parsed["reasoning"]
                r["f1"] = token_f1(r["prediction"], r["reference"])

    elapsed = time.time() - start_time

    if use_batch:
        # Already scored inline; build the report manually
        from collections import defaultdict
        categories: dict[str, list] = defaultdict(list)
        for r in all_results:
            if "f1" not in r:
                r["f1"] = token_f1(r["prediction"], r["reference"])
            categories[r.get("category", "unknown")].append(r)

        all_f1 = [r["f1"] for r in all_results]
        all_judge = [r["judge_score"] for r in all_results if "judge_score" in r]
        per_category = {}
        for cat, entries in categories.items():
            cf = [e["f1"] for e in entries]
            cj = [e["judge_score"] for e in entries if "judge_score" in e]
            per_category[cat] = {
                "count": len(entries), "avg_f1": sum(cf)/len(cf) if cf else 0,
                "avg_judge": sum(cj)/len(cj) if cj else None,
            }
        scores = {
            "total": len(all_results),
            "avg_f1": sum(all_f1)/len(all_f1) if all_f1 else 0,
            "avg_judge": sum(all_judge)/len(all_judge) if all_judge else None,
            "per_category": per_category,
            "details": all_results,
        }
    else:
        effective_judge = use_judge and not dry_run
        print(f"\nScoring {len(all_results)} QA pairs...")
        scores = score_batch(all_results, use_judge=effective_judge)

    _print_report("LoCoMo", scores, elapsed, SOTA)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"locomo_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return scores


def _print_report(name: str, scores: dict, elapsed: float, sota: dict):
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"{name} Evaluation Results")
    print("=" * 60)
    print(f"Total questions: {scores['total']}")
    print(f"Avg Token F1:   {scores['avg_f1']:.3f}")
    if scores["avg_judge"] is not None:
        print(f"Avg Judge:      {scores['avg_judge']:.3f}")
    print(f"Time:           {elapsed:.1f}s")
    print()

    print("Per-category breakdown:")
    for cat, d in sorted(scores["per_category"].items()):
        judge_str = f"  judge: {d['avg_judge']:.3f}" if d["avg_judge"] else ""
        print(f"  {cat:<25} n={d['count']:>4}  f1: {d['avg_f1']:.3f}{judge_str}")

    print()
    print("SOTA comparison:")
    cortex_score = scores["avg_judge"] if scores["avg_judge"] else scores["avg_f1"]
    for system, score in sorted(sota.items(), key=lambda x: x[1], reverse=True):
        print(f"  {system:<20} {score:.1f}%")
    print(f"  {'Cortex':<20} {cortex_score * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for Cortex")
    parser.add_argument("--limit", type=int, help="Limit to N conversations")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls")
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API (cheaper, no rate limits)")
    args = parser.parse_args()

    run_locomo(
        limit=args.limit,
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
        use_batch=args.batch,
    )
