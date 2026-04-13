# Cortex — A Coding Agent's Brain

## What Cortex Is

Cortex is an automated memory system that gives AI coding agents persistent, human-like memory across sessions. It runs locally, costs nothing per query, and works without manual intervention.

**Core capabilities:**
- **Remember architecture decisions** — "We use Axum with SQLite WAL mode, layered architecture"
- **Learn user preferences** — "Uses snake_case, prefers thiserror over anyhow in libraries"
- **Accumulate codebase expertise** — "sift-memory holds a Mutex<Connection>, recall uses RRF fusion"
- **Detect workflow patterns** — "Always runs cargo test after editing, then formats before committing"
- **Handle knowledge updates** — "Migrated from Redux to Zustand" supersedes "Uses Redux Toolkit"

## What Cortex Is NOT

Cortex is not a chatbot memory system. It doesn't remember what you had for lunch or when your friend went on vacation. The architecture is optimized for the coding domain: structured entities (projects, files, tools), relation graphs (maintains, depends_on, prefers), and procedural skill extraction.

## Architecture

```
Claude Code Session
  │
  │  PostToolUse / Stop / PostCompact hooks
  │  (hot path: <100ms, zero LLM cost)
  │
  ▼
Episodes table (raw hook events)
  │
  │  Daemon timer / CLI / MCP trigger
  │  (cold path: 5-phase consolidation)
  │
  ▼
Knowledge Graph (entities → observations → relations → skills)
  │
  │  sift_recall (hybrid BM25 + vector, entity-filtered,
  │               retrieval-dependent strengthening)
  │
  ▼
Next Session (agent has persistent memory)
```

## Current State

**Implemented (Phases 1-4):**
- Episode capture via Claude Code hooks (attention filter, <100ms)
- 5-phase consolidation engine (dedup, promotion, skill extraction, decay)
- Daemon integration with periodic consolidation
- Entity resolution and spreading activation in recall
- Retrieval-dependent strengthening (Ebbinghaus spacing effect)
- Internal benchmark suite (7 invariant tests)
- CodingMem benchmark: **96.8%** across 76 questions, 7 categories
- LoCoMo baseline: **64%** (competitive with Mem0 at 66.9%, zero LLM cost)

**Not yet done:**
- Hooks not installed in real Claude Code sessions
- No agent reasoning layer (memory is stored but not proactively used)
- CodingMem is 76 questions — needs to be 500+ for publishability
- No real-world usage data

## Next Steps

### 1. Ship the Hooks

Install Cortex hooks in Claude Code and use it for real work. Validate that:
- Episode capture works on real sessions (not just synthetic benchmarks)
- Consolidation produces useful entities and observations
- Recall surfaces relevant memories in new sessions
- The system is invisible — no manual intervention needed

### 2. Build the Agent Reasoning Layer

Memory alone is passive — it waits to be queried. A real coding agent brain should:
- **Proactively surface relevant context** when a new session starts
- **Detect when the user is repeating a pattern** and suggest automation
- **Flag knowledge conflicts** ("you said React 18 last week but the codebase has React 19")
- **Build and maintain a project mental model** that evolves with the codebase

### 3. Expand CodingMem to Publishable Scale

76 questions across 7 categories is a proof of concept. A publishable benchmark needs:
- 500+ questions with difficulty gradation (easy/medium/hard per category)
- Multi-codebase scenarios (not just single-project)
- Long-horizon scenarios (10+ sessions spanning weeks of simulated time)
- Adversarial scenarios (contradictory information, outdated facts)
- Ground-truth labels validated by multiple human annotators

### 4. Real-World Validation

The ultimate test is not a benchmark — it's whether Cortex makes a coding agent measurably better at its job:
- Does the agent make fewer repeated mistakes?
- Does it apply user preferences without being reminded?
- Does it navigate familiar codebases faster in later sessions?
- Does it suggest relevant context that the user would have forgotten?

These are the metrics that matter. Benchmarks are proxies.
