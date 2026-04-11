"""Scoring utilities — Token F1 and LLM-as-judge via configurable backend."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm import judge as llm_judge


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (whitespace + lowercase)."""
    pred_tokens = set(str(prediction).lower().split())
    ref_tokens = set(str(reference).lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_batch(results: list[dict], use_judge: bool = True) -> dict[str, Any]:
    """Score a batch of QA results.

    Each result dict must have: question, prediction, reference, category.
    Returns aggregate metrics with per-category breakdown.
    """
    categories: dict[str, list] = defaultdict(list)
    all_f1: list[float] = []
    all_judge: list[float] = []

    # Compute Token F1 (instant, no LLM)
    for r in results:
        r["f1"] = token_f1(r["prediction"], r["reference"])
        all_f1.append(r["f1"])

    # Run LLM judge in parallel if requested
    if use_judge:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm as tqdm_bar

        def _judge(r):
            return llm_judge(r["question"], r["prediction"], r["reference"])

        with ThreadPoolExecutor(max_workers=10) as pool:
            judge_results = list(tqdm_bar(
                pool.map(_judge, results),
                total=len(results),
                desc="  Judging",
            ))

        for r, jr in zip(results, judge_results):
            r["judge_score"] = jr["score"]
            r["judge_reasoning"] = jr["reasoning"]
            all_judge.append(jr["score"])

    for r in results:
        categories[r.get("category", "unknown")].append(r)

    per_category = {}
    for cat, entries in categories.items():
        cat_f1 = [e["f1"] for e in entries]
        cat_judge = [e["judge_score"] for e in entries if "judge_score" in e]
        per_category[cat] = {
            "count": len(entries),
            "avg_f1": sum(cat_f1) / len(cat_f1) if cat_f1 else 0.0,
            "avg_judge": sum(cat_judge) / len(cat_judge) if cat_judge else None,
        }

    return {
        "total": len(results),
        "avg_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        "avg_judge": sum(all_judge) / len(all_judge) if all_judge else None,
        "per_category": per_category,
        "details": [e for entries in categories.values() for e in entries],
    }
