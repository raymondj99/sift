#!/usr/bin/env python3
"""CodingMem — novel coding assistant memory benchmark.

7 categories no existing benchmark covers: cross-session knowledge,
preference retention, temporal updates, correction learning, workflow
patterns, multi-project isolation, codebase expertise.

Usage:
    uv run python eval_codingmem.py --batch         # Anthropic batch API
    uv run python eval_codingmem.py                  # real-time API
    uv run python eval_codingmem.py --dry-run        # no LLM calls
    uv run python eval_codingmem.py --scenario cross-session-knowledge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from llm import generate_answer
from scoring import score_batch, token_f1
from sift_client import SiftClient

RESULTS_DIR = Path(__file__).parent / "results"
SCENARIOS_PATH = Path(__file__).parent / "codingmem" / "scenarios.json"


def ingest_scenario(client: SiftClient, scenario: dict):
    """Ingest a CodingMem scenario's sessions via sift_remember."""
    for session in scenario.get("sessions", []):
        session_id = session["id"]
        for event in session.get("events", []):
            event_type = event["type"]
            content = event["content"]

            if event_type == "post_compact":
                summary = content.get("compact_summary", "")
                if summary:
                    client.remember(
                        entity=f"session:{session_id}",
                        entity_type="event",
                        observations=[summary],
                    )
            elif event_type == "stop":
                message = content.get("last_assistant_message", "")
                if message and len(message) > 20:
                    client.remember(
                        entity=f"session:{session_id}",
                        entity_type="event",
                        observations=[message],
                    )
            elif event_type == "post_tool_use":
                tool_name = content.get("tool_name", "")
                tool_input = content.get("tool_input", {})
                if not isinstance(tool_input, dict):
                    continue
                file_path = tool_input.get("file_path", "")
                command = tool_input.get("command", "")

                if file_path:
                    parts = file_path.split("/")
                    short = "/".join(parts[-3:]) if len(parts) > 3 else file_path
                    client.remember(
                        entity=short,
                        entity_type="project",
                        observations=[f"{tool_name} on {short}"],
                    )
                elif command:
                    client.remember(
                        entity=f"session:{session_id}",
                        entity_type="event",
                        observations=[f"Ran: {command[:200]}"],
                    )


def run_codingmem(
    scenario_filter: str | None = None,
    use_judge: bool = True,
    dry_run: bool = False,
    use_batch: bool = False,
):
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        if not scenarios:
            print(f"No scenario matching '{scenario_filter}'", file=sys.stderr)
            sys.exit(1)

    mode = "batch" if use_batch else ("dry-run" if dry_run else "full")
    print(f"CodingMem v{data['version']}: {len(scenarios)} scenarios ({mode})\n")

    all_recalled: list[tuple] = []
    start_time = time.time()

    for scenario in tqdm(scenarios, desc="Ingesting"):
        scenario_id = scenario["id"]
        category = scenario["category"]
        index_name = f"bench-codingmem-{scenario_id}"

        try:
            with SiftClient(index=index_name) as client:
                ingest_scenario(client, scenario)
                client.consolidate()

                for qa in scenario.get("questions", []):
                    memories = client.recall(qa["question"], limit=15)
                    all_recalled.append((
                        qa["question"],
                        qa["answer"],
                        qa.get("category", category),
                        scenario_id,
                        memories,
                    ))
        except Exception as e:
            print(f"\n  Error on {scenario_id}: {e}", file=sys.stderr)

    # Generate answers
    all_results: list[dict] = []
    if dry_run:
        for question, reference, category, scenario_id, memories in all_recalled:
            prediction = " | ".join(
                m.get("fact", "") for m in memories
            ) or "(no memories)"
            all_results.append({
                "question": question, "prediction": prediction,
                "reference": reference, "category": category,
                "scenario": scenario_id, "num_memories_retrieved": len(memories),
            })
    elif use_batch:
        from batch import (
            create_generation_request, create_judge_request,
            submit_and_wait, parse_judge_response,
        )

        # Batch generation
        gen_requests = [
            create_generation_request(f"gen-{i}", q, mems)
            for i, (q, _, _, _, mems) in enumerate(all_recalled)
        ]
        print(f"\n--- Batch: generating {len(gen_requests)} answers ---")
        gen_results = submit_and_wait(gen_requests)

        for i, (question, reference, category, scenario_id, memories) in enumerate(all_recalled):
            all_results.append({
                "question": question,
                "prediction": gen_results.get(f"gen-{i}", "(batch failed)"),
                "reference": reference, "category": category,
                "scenario": scenario_id, "num_memories_retrieved": len(memories),
            })

        # Batch judging
        if use_judge:
            judge_requests = [
                create_judge_request(f"judge-{i}", r["question"], r["prediction"], r["reference"])
                for i, r in enumerate(all_results)
            ]
            print(f"\n--- Batch: judging {len(judge_requests)} answers ---")
            judge_results = submit_and_wait(judge_requests)

            for i, r in enumerate(all_results):
                text = judge_results.get(f"judge-{i}", "SCORE: 0.0\nREASONING: batch failed")
                parsed = parse_judge_response(text)
                r["judge_score"] = parsed["score"]
                r["judge_reasoning"] = parsed["reasoning"]
                r["f1"] = token_f1(r["prediction"], r["reference"])
    else:
        for question, reference, category, scenario_id, memories in all_recalled:
            prediction = generate_answer(question, memories)
            all_results.append({
                "question": question, "prediction": prediction,
                "reference": reference, "category": category,
                "scenario": scenario_id, "num_memories_retrieved": len(memories),
            })

    elapsed = time.time() - start_time

    # Score
    if use_batch and not dry_run:
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
                "count": len(entries),
                "avg_f1": sum(cf) / len(cf) if cf else 0,
                "avg_judge": sum(cj) / len(cj) if cj else None,
            }
        scores = {
            "total": len(all_results),
            "avg_f1": sum(all_f1) / len(all_f1) if all_f1 else 0,
            "avg_judge": sum(all_judge) / len(all_judge) if all_judge else None,
            "per_category": per_category,
            "details": all_results,
        }
    else:
        effective_judge = use_judge and not dry_run
        print(f"\nScoring {len(all_results)} questions...")
        scores = score_batch(all_results, use_judge=effective_judge)

    # Report
    print(f"\n{'=' * 60}")
    print("CodingMem Evaluation Results")
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
        print(f"  {cat:<25} n={d['count']:>2}  f1: {d['avg_f1']:.3f}{judge_str}")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"codingmem_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodingMem benchmark for Cortex")
    parser.add_argument("--scenario", help="Run a specific scenario by ID")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls")
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API")
    args = parser.parse_args()

    run_codingmem(
        scenario_filter=args.scenario,
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
        use_batch=args.batch,
    )
