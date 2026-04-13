# Cortex Agent Reasoning Layer — Implementation Plan

## The Insight

Cortex currently stores memories passively and waits to be queried. A real coding agent brain should **proactively surface context**. The research identifies two complementary mechanisms:

1. **Between sessions**: Generate `.claude/rules/` files from consolidated memory. Claude Code auto-loads these with zero retrieval overhead. Path-targeted rules mean only relevant context appears for relevant files.

2. **Within sessions**: Hook into the episode pipeline to detect patterns, corrections, and knowledge updates in real-time.

## Implementation Order

### Step 1: Install Hooks (5 min)

Add Cortex hooks to `~/.claude/settings.json`. This is prerequisite for everything else — without hooks, no episodes are captured.

**File**: `~/.claude/settings.json`

### Step 2: Generate .claude/rules/ from Memory (the big one)

New consolidation phase: after the existing 5 phases, generate path-targeted rule files from semantic-tier observations.

**How it works:**
1. After consolidation, scan semantic observations for patterns:
   - Preferences → `cortex-preferences.md`
   - Project decisions → `cortex-{project}.md` (path-targeted)
   - Corrections → `cortex-corrections.md`
   - Workflow patterns → `cortex-workflows.md`
2. Write to `~/.claude/rules/` (user-scoped) or `.claude/rules/` (project-scoped)
3. Include `globs:` frontmatter for path targeting
4. Cap at ~5K tokens total (PlugMem: more memory degrades performance when utility-per-token is low)

**Files to modify:**
- `crates/sift-memory/src/consolidation.rs` — new Phase 6: rule generation
- `crates/sift-cli/src/commands/memory.rs` — `sift memory generate-rules` command
- `crates/sift-core/src/config.rs` — add `rules_dir` config

### Step 3: Corrective Feedback Strengthening

When the agent makes a mistake (test fails, linter catches error, user corrects), strengthen the correction memory with a boosted confidence multiplier.

**Detection signals:**
- `PostToolUse` on Bash: if exit code != 0 AND previous tool was Edit → correction opportunity
- `Stop` hook: if `last_assistant_message` contains correction language ("actually", "sorry", "let me fix")
- MCP: `sift_remember` with source="correction" gets boosted confidence

**Files:**
- `crates/sift-memory/src/episodes.rs` — detect failure patterns
- `crates/sift-memory/src/lib.rs` — boosted retrieval coefficient for correction observations

### Step 4: Expand CodingMem Benchmark

Add scenarios that test PROACTIVE memory:
- Does the agent apply preferences without being asked?
- Does it avoid previously-corrected mistakes?
- Does it surface relevant context for the file being edited?
- Does it handle knowledge conflicts correctly?

**Files:**
- `bench/codingmem/scenarios.json` — new proactive scenarios
- `bench/eval_codingmem.py` — test proactive rule loading

### Step 5: Benchmark the Agent Reasoning Layer

End-to-end test: install hooks → run simulated sessions → consolidate → generate rules → verify rules contain the right content → start new session → verify agent has context.

## Verification

1. `sift memory init-hooks` → hooks installed
2. Simulated session → episodes captured → `sift memory status` shows pending
3. `sift memory consolidate` → entities + observations created
4. `sift memory generate-rules` → rule files written to `.claude/rules/`
5. New session → verify rules auto-loaded → agent has proactive context
6. `sift bench all` → 7/7 internal benchmarks pass
7. CodingMem benchmark → proactive scenarios pass
8. `cargo test --workspace` → 0 failures
