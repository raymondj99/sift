# Cortex Memory Evaluations

Evaluates the Cortex memory system against published baselines and a novel
coding-assistant benchmark. See [`perf/`](../perf/) for CLI-level
performance benchmarks (speed, not accuracy).

## Setup

```bash
cd evals
uv sync
```

Requires `sift` binary built with MCP feature:
```bash
cargo build --package sift-cli --features mcp
export SIFT_BIN=../target/debug/sift
```

## LLM Backend

| Backend | Env Var | Cost | Batch Support |
|---------|---------|------|---------------|
| Anthropic API | `LLM_BACKEND=anthropic` (default) | ~$0.25/1K questions | Yes (`--batch`, 50% cheaper) |
| OpenAI API | `LLM_BACKEND=openai` | Varies by model | No |

Use `--batch` with Anthropic for large benchmarks — 50% cheaper, no rate limits.

## Benchmarks

### LoCoMo (ACL 2024) — Published Baseline

1,986 QA pairs across 10 conversations, 5 categories (single-hop, multi-hop, temporal reasoning, open-domain, adversarial).

```bash
# Full benchmark (10 conversations, ~$0.75 with batch API)
ANTHROPIC_API_KEY=... uv run python eval_locomo.py --batch

# Quick test (1 conversation)
uv run python eval_locomo.py --limit 1 --batch

# Dry run (no LLM calls — tests ingestion + recall only)
uv run python eval_locomo.py --dry-run
```

**Results: Cortex 64.5% vs Zep 75.1%**

### CodingMem (Novel) — Coding Assistant Memory

20 questions across 7 categories that no existing benchmark covers: cross-session project knowledge, user preference retention, temporal knowledge updates, correction learning, tool workflow patterns, multi-project isolation, codebase expertise accumulation.

```bash
# Batch API (recommended)
ANTHROPIC_API_KEY=... uv run python eval_codingmem.py --batch

# Real-time API
ANTHROPIC_API_KEY=... uv run python eval_codingmem.py

# Dry run (no LLM calls)
uv run python eval_codingmem.py --dry-run
```

**Results: Cortex 95%**

### Internal Benchmarks (Rust)

7 benchmarks validating performance contracts and system invariants:

```bash
cargo run --package sift-cli -- bench all
```

## Architecture

```
sift_client.py   — MCP JSON-RPC client (spawns `sift mcp` over stdio)
llm.py           — LLM backend (Anthropic API or OpenAI API)
batch.py         — Anthropic Batch API for large-scale evaluation
scoring.py       — Token F1 + parallel LLM-as-judge scoring
eval_locomo.py   — LoCoMo benchmark with entity-filtered recall
eval_codingmem.py — CodingMem benchmark with episode pipeline testing
codingmem/       — CodingMem scenario definitions (7 categories)
```
