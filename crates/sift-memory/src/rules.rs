//! Generate agent instruction files from consolidated memory.
//!
//! This is the agent reasoning layer: transforms passive stored memories
//! into proactive context that coding agents auto-load at session start.
//!
//! **Universal output**: `AGENTS.md` at project root — the Linux Foundation
//! standard read by 20+ tools (Codex, Copilot, Cursor, Windsurf, Aider,
//! Cline, Amp, Devin, Zed, Gemini CLI, etc.).
//!
//! **Claude Code native**: `.claude/rules/cortex-*.md` — path-targeted
//! rule files, only generated when `.claude/` directory is detected.
//!
//! Quality over quantity: token budget is 5K total — exceeding that
//! degrades agent performance (PlugMem, Microsoft 2026).

use crate::types::RecallFilters;
use crate::MemoryStore;
use std::collections::HashSet;
use std::path::Path;
use tracing::{debug, info};

/// Result of rule generation.
#[derive(Debug, Default)]
pub struct RuleGenReport {
    pub files_written: usize,
    pub total_rules: usize,
    pub total_tokens_approx: usize,
    pub skipped_noise: usize,
    pub skipped_long: usize,
    /// Which output formats were written.
    pub formats: Vec<String>,
}

/// Maximum word count for a single rule entry.
const MAX_ENTRY_WORDS: usize = 80;

/// Total token budget across all rule files.
const MAX_TOTAL_TOKENS: usize = 5000;

/// Per-category token budget.
const MAX_CATEGORY_TOKENS: usize = MAX_TOTAL_TOKENS / 4;

/// Rule file names managed by Cortex in `.claude/rules/`.
const CLAUDE_RULE_FILES: &[&str] = &[
    "cortex-preferences.md",
    "cortex-corrections.md",
    "cortex-decisions.md",
    "cortex-workflows.md",
];

/// Markers for the Cortex-managed section in AGENTS.md.
const AGENTS_MD_START: &str = "<!-- cortex:start -->";
const AGENTS_MD_END: &str = "<!-- cortex:end -->";

// ===========================================================================
// Classified rules (intermediate representation)
// ===========================================================================

/// Classified and deduplicated rules ready for output.
///
/// This is the intermediate representation between DB queries and file writing.
/// Built once, written to multiple output formats.
struct ClassifiedRules {
    preferences: Vec<String>,
    corrections: Vec<String>,
    decisions: Vec<String>,
    workflows: Vec<String>,
    skipped_noise: usize,
    skipped_long: usize,
}

/// Classify observations from the memory store into rule categories.
fn classify_from_store(memory: &MemoryStore) -> crate::MemResult<ClassifiedRules> {
    let mut preferences = Vec::new();
    let mut corrections = Vec::new();
    let mut decisions = Vec::new();
    let mut workflows = Vec::new();
    let mut skipped_noise = 0usize;
    let mut skipped_long = 0usize;
    let mut seen: HashSet<String> = HashSet::new();

    // --- Pass 1: SQL-based classification of all observations ---
    // Scoped block so `conn` is dropped before Pass 2 (which needs the lock).
    {
        let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());

        let mut stmt = conn.prepare(
            "SELECT o.content, o.confidence, o.source, o.memory_tier,
                    e.name, e.entity_type
             FROM observations o
             JOIN entities e ON o.entity_id = e.id
             WHERE o.valid_until IS NULL
             ORDER BY
               CASE o.memory_tier
                 WHEN 'procedural' THEN 0
                 WHEN 'semantic' THEN 1
                 WHEN 'episodic' THEN 2
                 ELSE 3
               END,
               o.confidence DESC,
               o.access_count DESC,
               o.observed_at DESC",
        )?;

        let rows: Vec<ObservationRow> = stmt
            .query_map([], |row| {
                Ok(ObservationRow {
                    content: row.get(0)?,
                    confidence: row.get(1)?,
                    source: row.get(2)?,
                    _tier: row.get(3)?,
                    entity_name: row.get(4)?,
                    entity_type: row.get(5)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        for row in &rows {
            // Use a small epsilon below 0.5 to handle f32 precision drift
            // (e.g., 0.7 - 0.1 - 0.1 can yield 0.49999... in IEEE 754).
            if row.confidence < 0.49 {
                continue;
            }
            if is_noise(&row.content, &row.entity_name, &row.entity_type) {
                skipped_noise += 1;
                continue;
            }
            let word_count = row.content.split_whitespace().count();
            if word_count > MAX_ENTRY_WORDS {
                skipped_long += 1;
                continue;
            }
            if word_count < 5 {
                skipped_noise += 1;
                continue;
            }
            let norm = normalize_for_dedup(&row.content);
            if seen.contains(&norm) {
                continue;
            }

            // Strip leading bullet markers to prevent "- - text" in output.
            // Observations may have been stored with markdown bullets via
            // the PostToolUse hook capturing rule file writes.
            let clean = strip_bullet_prefix(&row.content);

            match classify(row) {
                Category::Preference => {
                    seen.insert(norm);
                    preferences.push(clean);
                }
                Category::Correction => {
                    seen.insert(norm);
                    corrections.push(clean);
                }
                Category::Decision => {
                    seen.insert(norm);
                    decisions.push(format!("[{}] {}", row.entity_name, clean));
                }
                Category::Workflow => {
                    seen.insert(norm);
                    workflows.push(clean);
                }
                // Don't add Skip to `seen` — pass 2 (semantic search)
                // can rescue these as workflow observations.
                Category::Skip => {}
            }
        }
    } // conn dropped — safe to call recall()

    // --- Pass 2: Semantic search for workflow observations ---
    // Use sift's search engine to find procedural knowledge that the
    // heuristic classifier routed to Skip or missed entirely.
    let found = search_for_workflows(memory, &mut seen);
    debug!(count = found.len(), "Pass 2: semantic workflow search");
    workflows.extend(found);

    Ok(ClassifiedRules {
        preferences,
        corrections,
        decisions,
        workflows,
        skipped_noise,
        skipped_long,
    })
}

// ===========================================================================
// Public API
// ===========================================================================

/// Generate rules to all applicable formats for the given project root.
///
/// Always writes/updates `AGENTS.md` at `project_root`.
/// If `.claude/` directory exists, also writes `.claude/rules/cortex-*.md`.
pub fn generate_all_rules(
    memory: &MemoryStore,
    project_root: &Path,
) -> crate::MemResult<RuleGenReport> {
    let rules = classify_from_store(memory)?;
    let mut report = RuleGenReport {
        skipped_noise: rules.skipped_noise,
        skipped_long: rules.skipped_long,
        ..Default::default()
    };

    // 1. Always: update AGENTS.md
    let agents_tokens = write_agents_md(project_root, &rules, &mut report)?;
    report.total_tokens_approx += agents_tokens;

    // 2. If Claude Code detected: also write .claude/rules/
    let claude_dir = project_root.join(".claude");
    if claude_dir.is_dir() {
        let rules_dir = claude_dir.join("rules");
        let claude_tokens = write_claude_rules(&rules_dir, &rules, &mut report)?;
        report.total_tokens_approx += claude_tokens;
    }

    info!(
        files = report.files_written,
        rules = report.total_rules,
        tokens = report.total_tokens_approx,
        formats = ?report.formats,
        skipped_noise = report.skipped_noise,
        skipped_long = report.skipped_long,
        "Rule generation complete"
    );

    Ok(report)
}

/// Generate rules only to `.claude/rules/` (backward-compatible API).
pub fn generate_rules(memory: &MemoryStore, rules_dir: &Path) -> crate::MemResult<RuleGenReport> {
    let rules = classify_from_store(memory)?;
    let mut report = RuleGenReport {
        skipped_noise: rules.skipped_noise,
        skipped_long: rules.skipped_long,
        ..Default::default()
    };

    let tokens = write_claude_rules(rules_dir, &rules, &mut report)?;
    report.total_tokens_approx = tokens;

    info!(
        files = report.files_written,
        rules = report.total_rules,
        tokens = report.total_tokens_approx,
        skipped_noise = report.skipped_noise,
        skipped_long = report.skipped_long,
        "Rule generation complete"
    );

    Ok(report)
}

// ===========================================================================
// AGENTS.md writer
// ===========================================================================

/// Write or update the Cortex section in AGENTS.md.
///
/// If AGENTS.md exists, replaces the `<!-- cortex:start -->...<!-- cortex:end -->`
/// section while preserving all user content. If the markers don't exist yet,
/// appends the section at the end. If the file doesn't exist, creates it with
/// just the Cortex section.
fn write_agents_md(
    project_root: &Path,
    rules: &ClassifiedRules,
    report: &mut RuleGenReport,
) -> crate::MemResult<usize> {
    let cortex_section = build_agents_md_section(rules);
    if cortex_section.is_empty() {
        return Ok(0);
    }

    let agents_path = project_root.join("AGENTS.md");
    let existing = std::fs::read_to_string(&agents_path).unwrap_or_default();

    let new_content = if existing.is_empty() {
        // No existing file — create with just Cortex section
        cortex_section.clone()
    } else if let Some((before, after)) = split_at_markers(&existing) {
        // Markers exist — replace the section
        format!("{before}{cortex_section}{after}")
    } else {
        // File exists but no markers — append
        format!("{}\n\n{}", existing.trim_end(), cortex_section)
    };

    std::fs::write(&agents_path, &new_content)
        .map_err(|e| crate::MemoryError::InvalidInput(format!("Cannot write AGENTS.md: {e}")))?;

    let tokens = cortex_section.split_whitespace().count();
    let rule_count = cortex_section
        .lines()
        .filter(|l| l.starts_with("- "))
        .count();
    report.files_written += 1;
    report.total_rules += rule_count;
    report.formats.push("AGENTS.md".to_string());

    Ok(tokens)
}

/// Build the Cortex-managed section for AGENTS.md.
fn build_agents_md_section(rules: &ClassifiedRules) -> String {
    let mut sections = Vec::new();

    let prefs = truncate_entries(&rules.preferences, MAX_CATEGORY_TOKENS);
    if !prefs.is_empty() {
        sections.push(format!(
            "### User Preferences\n\n\
             These preferences were learned from past sessions. Apply them without asking.\n\n\
             {prefs}"
        ));
    }

    let corr = truncate_entries(&rules.corrections, MAX_CATEGORY_TOKENS);
    if !corr.is_empty() {
        sections.push(format!(
            "### Corrections & Patterns to Avoid\n\n\
             These mistakes were corrected in past sessions. Do not repeat them.\n\n\
             {corr}"
        ));
    }

    let dec = truncate_entries(&rules.decisions, MAX_CATEGORY_TOKENS);
    if !dec.is_empty() {
        sections.push(format!(
            "### Project Decisions\n\n\
             Architecture and technology choices from past sessions.\n\n\
             {dec}"
        ));
    }

    let wf = truncate_entries(&rules.workflows, MAX_CATEGORY_TOKENS);
    if !wf.is_empty() {
        sections.push(format!(
            "### Workflow Patterns\n\n\
             Tool usage patterns detected across sessions.\n\n\
             {wf}"
        ));
    }

    if sections.is_empty() {
        return String::new();
    }

    format!(
        "{AGENTS_MD_START}\n\
         ## Learned Context (auto-generated by Cortex)\n\n\
         > This section is automatically maintained by [Cortex](https://github.com/raymondj99/sift).\n\
         > Do not edit manually — changes will be overwritten on next `sift memory generate-rules`.\n\n\
         {}\n\
         {AGENTS_MD_END}\n",
        sections.join("\n\n")
    )
}

/// Split file content at the cortex markers, returning (before, after).
fn split_at_markers(content: &str) -> Option<(String, String)> {
    let start = content.find(AGENTS_MD_START)?;
    let end = content.find(AGENTS_MD_END)?;
    if end <= start {
        return None;
    }
    let before = &content[..start];
    let after = &content[end + AGENTS_MD_END.len()..];
    Some((before.to_string(), after.to_string()))
}

// ===========================================================================
// .claude/rules/ writer
// ===========================================================================

/// Write classified rules to `.claude/rules/cortex-*.md` files.
fn write_claude_rules(
    rules_dir: &Path,
    rules: &ClassifiedRules,
    report: &mut RuleGenReport,
) -> crate::MemResult<usize> {
    std::fs::create_dir_all(rules_dir)
        .map_err(|e| crate::MemoryError::InvalidInput(format!("Cannot create rules dir: {e}")))?;

    // Clean stale cortex-* files from prior runs.
    for name in CLAUDE_RULE_FILES {
        let path = rules_dir.join(name);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
    }

    let mut total_tokens = 0usize;

    total_tokens += write_claude_rule_file(
        rules_dir,
        "cortex-preferences.md",
        "User Preferences",
        "These preferences were learned from past sessions. Apply them without asking.",
        &rules.preferences,
        report,
    )?;

    total_tokens += write_claude_rule_file(
        rules_dir,
        "cortex-corrections.md",
        "Corrections & Patterns to Avoid",
        "These mistakes were corrected in past sessions. Do not repeat them.",
        &rules.corrections,
        report,
    )?;

    total_tokens += write_claude_rule_file(
        rules_dir,
        "cortex-decisions.md",
        "Project Decisions",
        "Architecture and technology choices from past sessions.",
        &rules.decisions,
        report,
    )?;

    total_tokens += write_claude_rule_file(
        rules_dir,
        "cortex-workflows.md",
        "Workflow Patterns",
        "Tool usage patterns detected across sessions.",
        &rules.workflows,
        report,
    )?;

    if total_tokens > 0 {
        report.formats.push(".claude/rules/".to_string());
    }

    Ok(total_tokens)
}

/// Write a single `.claude/rules/cortex-*.md` file.
fn write_claude_rule_file(
    rules_dir: &Path,
    filename: &str,
    title: &str,
    description: &str,
    entries: &[String],
    report: &mut RuleGenReport,
) -> crate::MemResult<usize> {
    if entries.is_empty() {
        return Ok(0);
    }

    let body = truncate_entries(entries, MAX_CATEGORY_TOKENS);
    if body.is_empty() {
        return Ok(0);
    }

    let content = format!("# {title} (auto-generated by Cortex)\n\n{description}\n\n{body}\n");
    let tokens = content.split_whitespace().count();

    std::fs::write(rules_dir.join(filename), &content)
        .map_err(|e| crate::MemoryError::InvalidInput(format!("Cannot write rules: {e}")))?;

    let rule_count = body.lines().filter(|l| l.starts_with("- ")).count();
    report.files_written += 1;
    report.total_rules += rule_count;

    Ok(tokens)
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Normalize content for dedup: strip bullets, lowercase, trim.
/// Both pass 1 and pass 2 must use this so cross-pass dedup works.
fn normalize_for_dedup(s: &str) -> String {
    let stripped = strip_bullet_prefix(s);
    stripped.to_lowercase().trim().to_string()
}

/// Truncate a list of entries to fit within a token budget, formatting as bullets.
fn truncate_entries(entries: &[String], max_tokens: usize) -> String {
    let mut result = String::new();
    let mut tokens = 0usize;

    for entry in entries {
        let line = format!("- {entry}\n");
        let line_tokens = line.split_whitespace().count();
        if tokens + line_tokens > max_tokens {
            break;
        }
        result.push_str(&line);
        tokens += line_tokens;
    }

    result
}

// ===========================================================================
// Semantic workflow search
// ===========================================================================

/// Queries targeting different facets of procedural knowledge.
/// Hybrid search (vector + BM25) will match both semantically similar
/// observations and keyword hits.
const WORKFLOW_QUERIES: &[&str] = &[
    "always before after remember to verify check process steps workflow",
    "deploy release build test run first then make sure never avoid",
];

/// Minimum recall score to consider a search result as a workflow candidate.
const WORKFLOW_MIN_SCORE: f32 = 0.45;

/// Search memory for workflow observations using semantic search.
///
/// Finds procedural knowledge that the heuristic classifier missed —
/// observations like "always run X before Y" stored under entities
/// that don't match any classification pattern.
fn search_for_workflows(memory: &MemoryStore, seen: &mut HashSet<String>) -> Vec<String> {
    let filters = RecallFilters::default();
    let mut results = Vec::new();

    for query in WORKFLOW_QUERIES {
        let recalls = match memory.recall(query, 10, &filters) {
            Ok(r) => r,
            Err(_) => continue,
        };

        for r in recalls {
            if r.score < WORKFLOW_MIN_SCORE {
                continue;
            }

            // Strip leading markdown bullets — observations may have been
            // captured from rule files via the PostToolUse hook, creating
            // a feedback loop ("- Always..." → stored → re-emitted as
            // "- - Always..."). Normalize to plain text.
            let content = strip_bullet_prefix(&r.observation.content);
            let entity_type_str = r.entity_type.as_str();

            if is_noise(&content, &r.entity_name, entity_type_str) {
                continue;
            }

            let word_count = content.split_whitespace().count();
            if !(5..=MAX_ENTRY_WORDS).contains(&word_count) {
                continue;
            }

            let norm = normalize_for_dedup(&content);
            if seen.contains(&norm) {
                continue;
            }

            // Only add if the content actually reads like a workflow instruction.
            if !has_workflow_language(&content) {
                continue;
            }

            seen.insert(norm);
            results.push(content);
        }
    }

    results
}

/// Strip leading markdown bullet markers (`- `, `* `, `• `).
/// Handles nested bullets like `- - text` from feedback loops.
fn strip_bullet_prefix(s: &str) -> String {
    let mut text = s.trim_start();
    loop {
        if let Some(rest) = text.strip_prefix("- ") {
            text = rest.trim_start();
        } else if let Some(rest) = text.strip_prefix("* ") {
            text = rest.trim_start();
        } else if let Some(rest) = text.strip_prefix("• ") {
            text = rest.trim_start();
        } else {
            break;
        }
    }
    text.to_string()
}

/// Check if content contains imperative/procedural language patterns
/// that indicate a workflow instruction rather than a factual statement.
fn has_workflow_language(content: &str) -> bool {
    let lower = content.to_lowercase();

    // Imperative starts
    lower.starts_with("always ")
        || lower.starts_with("never ")
        || lower.starts_with("before ")
        || lower.starts_with("after ")
        || lower.starts_with("remember ")
        || lower.starts_with("make sure ")
        || lower.starts_with("run ")
        || lower.starts_with("check ")
        || lower.starts_with("verify ")
        || lower.starts_with("first ")
        // Procedural descriptions
        || lower.starts_with("the process ")
        || lower.starts_with("the flow ")
        || lower.starts_with("the order ")
        // Embedded workflow signals
        || lower.contains("remember to ")
        || lower.contains("don't forget ")
        || lower.contains("make sure to ")
        || lower.contains("always verify")
        || lower.contains("always check")
        || lower.contains("always run")
        || lower.contains("before releasing")
        || lower.contains("before deploying")
        || lower.contains("before committing")
        || lower.contains("before pushing")
        || lower.contains("before merging")
        || lower.contains("after every ")
}

// ===========================================================================
// Internal types
// ===========================================================================

struct ObservationRow {
    content: String,
    confidence: f32,
    source: String,
    _tier: String,
    entity_name: String,
    entity_type: String,
}

enum Category {
    Preference,
    Correction,
    Decision,
    Workflow,
    Skip,
}

// ===========================================================================
// Noise filtering
// ===========================================================================

/// Returns true if this observation is noise that should never be a rule.
fn is_noise(content: &str, entity_name: &str, entity_type: &str) -> bool {
    if content.contains('<') && content.contains('>') {
        return true;
    }

    let trimmed = content.trim();
    if trimmed.starts_with("Modified ")
        || trimmed.starts_with("Created ")
        || trimmed.starts_with("Created/wrote ")
        || trimmed.starts_with("Deleted ")
        || trimmed.starts_with("Wrote ")
        || trimmed.starts_with("Ran: ")
    {
        return true;
    }

    if entity_type == "event" || entity_name.starts_with("session:") {
        return true;
    }

    if entity_name.contains('/') || entity_name.ends_with(".rs") || entity_name.ends_with(".toml") {
        return true;
    }

    let name_lower = entity_name.to_lowercase();
    if name_lower.contains("research")
        || name_lower.contains("benchmarking")
        || name_lower.contains("failure analysis")
    {
        return true;
    }

    // Code-derivable content: file paths, struct/table definitions, module layouts.
    // An agent can read the code — no need to burn token budget on these.
    let content_lower = content.to_lowercase();
    if is_code_derivable(&content_lower) {
        return true;
    }

    // Stale planning language: completed phases, shipped priorities, done TODOs.
    if is_stale_plan(&content_lower, &name_lower) {
        return true;
    }

    false
}

/// Content that restates what's already in the code (file manifests, schema
/// field listings, struct descriptions). The agent can discover these by
/// reading the source.
fn is_code_derivable(content_lower: &str) -> bool {
    // File manifests: "New CLI subcommand at crates/sift-cli/..."
    if (content_lower.contains("crates/") || content_lower.contains("src/"))
        && (content_lower.starts_with("new ")
            || content_lower.starts_with("config extension")
            || content_lower.starts_with("enhanced recall in "))
    {
        return true;
    }

    // Schema/DDL descriptions: "ALTER TABLE ... ADD ...", "add episodes table"
    if content_lower.contains("alter table")
        || content_lower.contains("add access_count")
        || content_lower.contains("bump schema_version")
    {
        return true;
    }

    // API/protocol field listings: "stdin adds: tool_name (string), tool_input (object)..."
    if content_lower.contains("stdin adds:")
        || content_lower.contains("stdin adds ")
        || content_lower.contains("common fields:")
    {
        return true;
    }

    // Hook config format descriptions
    if content_lower.contains("hook config in settings.json:")
        || content_lower.contains("exit code 0 = success")
    {
        return true;
    }

    false
}

/// Detect observations that describe completed work still phrased as plans.
fn is_stale_plan(content_lower: &str, entity_name_lower: &str) -> bool {
    // "Implementation in 3 phases: Phase 1..." — describes future work
    if content_lower.starts_with("implementation in") && content_lower.contains("phase") {
        return true;
    }

    // "Phase 1 files" entity — implementation manifests for completed phases
    if entity_name_lower.contains("phase") && entity_name_lower.contains("files") {
        return true;
    }

    // "PLAN.md written at /Users/..." — absolute paths and completed TODOs
    if content_lower.contains("plan.md written at") {
        return true;
    }

    // "Housekeeping: clean up..." — completed housekeeping tasks
    if content_lower.starts_with("housekeeping:") {
        return true;
    }

    // "v0.1.x priority N:" — shipped version priorities
    if content_lower.contains("priority 1:")
        || content_lower.contains("priority 2:")
        || content_lower.contains("priority 3:")
    {
        // Check if it references a specific version that's likely shipped
        if content_lower.contains("v0.1.") {
            return true;
        }
    }

    // Roadmap items described as "already completed"
    if content_lower.contains("already completed") {
        return true;
    }

    false
}

// ===========================================================================
// Classification
// ===========================================================================

/// Classify an observation into a rule category.
fn classify(row: &ObservationRow) -> Category {
    let name_lower = row.entity_name.to_lowercase();
    let content_lower = row.content.to_lowercase();

    // --- Layer 0: Source-based classification (highest priority) ---
    // Procedural observations are learned procedures ("always do X",
    // "the process is Y", "remember to Z") — these are workflow rules,
    // not architecture decisions.
    if row.source == "cortex:procedural" {
        return Category::Workflow;
    }

    // --- Layer 1: Entity type (strongest signal) ---
    match row.entity_type.as_str() {
        "preference" | "person" => return Category::Preference,
        "tool" => return Category::Workflow,
        "fact" => {
            if content_lower.starts_with("don't ")
                || content_lower.starts_with("avoid ")
                || content_lower.starts_with("never ")
                || content_lower.contains("stale")
                || content_lower.contains("mistake")
            {
                return Category::Correction;
            }
            return Category::Decision;
        }
        "project" => return Category::Decision,
        _ => {}
    }

    // --- Layer 2: Entity name patterns ---
    if row.entity_type == "concept" && is_person_entity(&name_lower) {
        return Category::Preference;
    }

    if name_lower.contains("workflow") {
        return Category::Workflow;
    }

    if name_lower.contains("design")
        || name_lower.contains("architecture")
        || name_lower.contains("protocol")
        || name_lower.contains("config")
        || name_lower.contains("phase ")
        || name_lower.contains(" plan")
        || name_lower.contains("hook")
        || name_lower.contains("schema")
    {
        return Category::Decision;
    }

    // --- Layer 3: Content signals (concept entities) ---
    if row.entity_type == "concept" {
        if name_lower.contains("pattern") {
            if content_lower.starts_with("don't ")
                || content_lower.starts_with("avoid ")
                || content_lower.contains("instead")
            {
                return Category::Correction;
            }
            return Category::Decision;
        }
        if name_lower == "sift" || name_lower.starts_with("cortex") || name_lower.contains("mcp") {
            return Category::Decision;
        }
    }

    Category::Skip
}

/// Heuristic: is this entity name likely a person?
fn is_person_entity(name_lower: &str) -> bool {
    let is_single_word = !name_lower.contains(' ')
        && !name_lower.contains('/')
        && !name_lower.contains(':')
        && !name_lower.contains('.');

    let known_non_persons = ["sift", "cortex", "rust", "sqlite", "tantivy", "mcp"];

    is_single_word && !known_non_persons.contains(&name_lower)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeStore;
    use crate::types::*;

    /// Helper: set up a memory store with realistic observations.
    fn populated_store() -> MemoryStore {
        let store = MemoryStore::open_in_memory().unwrap();

        let raymond = store
            .save_entity("Raymond", crate::types::EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(
                &raymond,
                "Prefers Rust for systems programming",
                0.9,
                "test",
            )
            .unwrap();
        store
            .add_observation(
                &raymond,
                "Uses Claude Code as primary development tool",
                0.9,
                "test",
            )
            .unwrap();

        let project = store
            .save_entity("taskflow", EntityType::Project, 0.9, "test")
            .unwrap();
        store
            .add_observation(
                &project,
                "Uses Axum web framework with SQLite in WAL mode",
                0.9,
                "test",
            )
            .unwrap();

        let config = store
            .save_entity("MCP config", EntityType::Fact, 0.9, "test")
            .unwrap();
        store
            .add_observation(
                &config,
                "MCP servers must be configured at user scope in ~/.claude.json",
                0.9,
                "test",
            )
            .unwrap();

        let correction = store
            .save_entity("sift MCP config", EntityType::Fact, 0.9, "test")
            .unwrap();
        store
            .add_observation(
                &correction,
                "Don't use target/debug/sift for MCP — stale local config overrides the installed binary",
                0.9,
                "cortex:correction",
            )
            .unwrap();

        let research = store
            .save_entity(
                "Cortex research foundations",
                EntityType::Concept,
                0.9,
                "test",
            )
            .unwrap();
        store
            .add_observation(
                &research,
                "SYNAPSE uses spreading activation from cognitive science",
                0.9,
                "test",
            )
            .unwrap();

        let session = store
            .save_entity("session:abc123", EntityType::Event, 0.5, "cortex:episode")
            .unwrap();
        store
            .add_observation(
                &session,
                "User is building a memory system for sift project",
                0.8,
                "cortex:stop",
            )
            .unwrap();

        let file_ent = store
            .save_entity(
                "sift-memory/src/lib.rs",
                EntityType::Project,
                0.5,
                "cortex:episode",
            )
            .unwrap();
        store
            .add_observation(
                &file_ent,
                "Modified sift-memory/src/lib.rs",
                0.5,
                "cortex:tool",
            )
            .unwrap();

        store
    }

    // -- Claude rules tests (backward compat) --

    #[test]
    fn test_generate_rules_creates_correct_files() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        let report = generate_rules(&store, tmp.path()).unwrap();

        assert!(
            report.files_written >= 2,
            "Should write at least preferences + decisions"
        );
        assert!(report.total_rules >= 4, "Should have at least 4 rules");
        assert!(report.total_tokens_approx > 0);
    }

    #[test]
    fn test_preferences_contain_person_observations() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        generate_rules(&store, tmp.path()).unwrap();

        let prefs =
            std::fs::read_to_string(tmp.path().join("cortex-preferences.md")).unwrap_or_default();
        assert!(prefs.contains("Rust"), "Should contain Rust preference");
        assert!(
            prefs.contains("Claude Code"),
            "Should contain tool preference"
        );
    }

    #[test]
    fn test_decisions_contain_project_facts() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        generate_rules(&store, tmp.path()).unwrap();

        let decisions =
            std::fs::read_to_string(tmp.path().join("cortex-decisions.md")).unwrap_or_default();
        assert!(
            decisions.contains("Axum"),
            "Should contain framework decision"
        );
        assert!(
            decisions.contains("MCP"),
            "Should contain MCP config decision"
        );
    }

    #[test]
    fn test_corrections_contain_mistake_patterns() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        generate_rules(&store, tmp.path()).unwrap();

        let corrections =
            std::fs::read_to_string(tmp.path().join("cortex-corrections.md")).unwrap_or_default();
        assert!(
            corrections.contains("Don't use target/debug/sift"),
            "Should contain MCP correction"
        );
    }

    #[test]
    fn test_noise_is_filtered() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        let report = generate_rules(&store, tmp.path()).unwrap();

        assert!(report.skipped_noise > 0, "Should skip some noise");

        let all_content: String = CLAUDE_RULE_FILES
            .iter()
            .filter_map(|f| std::fs::read_to_string(tmp.path().join(f)).ok())
            .collect();

        assert!(
            !all_content.contains("SYNAPSE"),
            "Research notes should be filtered"
        );
        assert!(
            !all_content.contains("Modified sift-memory"),
            "File-edit records should be filtered"
        );
    }

    #[test]
    fn test_code_derivable_noise_filtered() {
        // Schema DDL descriptions
        assert!(is_noise(
            "Schema v2 migration: bump SCHEMA_VERSION to 2, add episodes table, ALTER TABLE observations ADD access_count",
            "Cortex Phase 1 files",
            "fact"
        ));

        // File manifest descriptions
        assert!(is_noise(
            "New episode ingest module at crates/sift-memory/src/episodes.rs: EpisodeStore thin SQLite wrapper",
            "Cortex Phase 1 files",
            "fact"
        ));

        // Hook stdin schema listings
        assert!(is_noise(
            "PostToolUse stdin adds: tool_name (string), tool_input (object), tool_response (object)",
            "Cortex hook stdin schemas",
            "fact"
        ));

        // But actionable facts should NOT be filtered
        assert!(!is_noise(
            "MCP servers must be configured at user scope in ~/.claude.json",
            "MCP config",
            "fact"
        ));
    }

    #[test]
    fn test_stale_plans_filtered() {
        // Completed phase implementations
        assert!(is_noise(
            "Implementation in 3 phases: Phase 1 (MVP) = schema v2 + episodes.rs",
            "Cortex automated memory system",
            "concept"
        ));

        // Shipped version priorities
        assert!(is_noise(
            "v0.1.4 priority 1: Shell completions — use clap_complete for bash/zsh/fish",
            "sift",
            "concept"
        ));

        // Already completed items
        assert!(is_noise(
            "Roadmap items already completed as of v0.1.3: parallel startup, input validation",
            "sift",
            "concept"
        ));

        // Housekeeping TODOs
        assert!(is_noise(
            "Housekeeping: clean up untracked sift-v0.1.2-aarch64-apple-darwin.tar.gz from repo root",
            "sift",
            "concept"
        ));

        // But current design decisions should NOT be filtered
        assert!(!is_noise(
            "Bi-temporal modeling separates when the agent learned it from when the fact was true",
            "sift memory design",
            "concept"
        ));
    }

    #[test]
    fn test_stale_files_are_cleaned() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("cortex-workflows.md"),
            "# Stale\n- old rule\n",
        )
        .unwrap();

        let store = populated_store();
        generate_rules(&store, tmp.path()).unwrap();

        assert!(
            !tmp.path().join("cortex-workflows.md").exists(),
            "Stale workflow file should be cleaned up"
        );
    }

    #[test]
    fn test_semantic_search_finds_workflow_observations() {
        let store = MemoryStore::open_in_memory().unwrap();

        // BM25 needs a corpus large enough for IDF to be meaningful.
        // With only 2 docs, IDF = log((N-n+0.5)/(n+0.5)) ≈ 0 and
        // all scores collapse to zero. Add filler observations.
        let filler = store
            .save_entity("general notes", EntityType::Concept, 0.9, "test")
            .unwrap();
        let filler_texts = [
            "The database uses PostgreSQL with connection pooling via pgbouncer",
            "Frontend is built with React and TypeScript strict mode enabled",
            "CI pipeline runs on GitHub Actions with parallel matrix builds",
            "Authentication uses OAuth2 with PKCE flow for single page apps",
            "Monitoring stack includes Prometheus and Grafana dashboards",
            "The caching layer uses Redis with a sixty second default TTL",
            "Logging follows structured JSON format with correlation identifiers",
            "Infrastructure is managed with Terraform and stored in the infra repo",
        ];
        for text in &filler_texts {
            store.add_observation(&filler, text, 0.9, "test").unwrap();
        }

        // Store a workflow rule under a concept entity with a multi-word
        // name that doesn't match any heuristic pattern. The SQL-based
        // classifier will route this to Skip, so only semantic search
        // (pass 2) can capture it as a workflow.
        let entity = store
            .save_entity("team practices", EntityType::Concept, 0.9, "test")
            .unwrap();
        store
            .add_observation(
                &entity,
                "Always run the integration tests before deploying to production",
                0.9,
                "sift_remember",
            )
            .unwrap();

        // Also store a non-workflow fact — should not leak into workflows
        let arch = store
            .save_entity("system layout", EntityType::Concept, 0.9, "test")
            .unwrap();
        store
            .add_observation(
                &arch,
                "The API gateway routes traffic to three backend services",
                0.9,
                "sift_remember",
            )
            .unwrap();

        let tmp = tempfile::TempDir::new().unwrap();
        generate_rules(&store, tmp.path()).unwrap();

        let workflows =
            std::fs::read_to_string(tmp.path().join("cortex-workflows.md")).unwrap_or_default();
        assert!(
            workflows.contains("integration tests"),
            "Semantic search should find workflow observation: {workflows}"
        );
        assert!(
            !workflows.contains("API gateway"),
            "Non-workflow facts should not leak into workflows"
        );
    }

    #[test]
    fn test_has_workflow_language() {
        // Positive: imperative starts
        assert!(has_workflow_language("Always run tests before pushing"));
        assert!(has_workflow_language("Never force-push to main"));
        assert!(has_workflow_language("Before releasing, update CHANGELOG"));
        assert!(has_workflow_language("Remember to bump the version number"));
        assert!(has_workflow_language("The process is: build, test, deploy"));
        assert!(has_workflow_language("Make sure the CI passes first"));
        assert!(has_workflow_language(
            "Run cargo clippy before committing code"
        ));

        // Negative: factual statements
        assert!(!has_workflow_language(
            "The system uses PostgreSQL for persistence"
        ));
        assert!(!has_workflow_language(
            "Sift is a local semantic search engine"
        ));
        assert!(!has_workflow_language("Written in Rust with async runtime"));
    }

    #[test]
    fn test_token_budget_is_respected() {
        let store = MemoryStore::open_in_memory().unwrap();
        let entity = store
            .save_entity("big-project", EntityType::Project, 1.0, "test")
            .unwrap();
        for i in 0..100 {
            store
                .add_observation(
                    &entity,
                    &format!(
                        "Architecture decision {i}: using pattern {i} for component {i} with framework {i}"
                    ),
                    0.9,
                    "test",
                )
                .unwrap();
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let report = generate_rules(&store, tmp.path()).unwrap();

        assert!(
            report.total_tokens_approx <= MAX_TOTAL_TOKENS + 100,
            "Total tokens ({}) should be within budget ({})",
            report.total_tokens_approx,
            MAX_TOTAL_TOKENS
        );
    }

    // -- AGENTS.md tests --

    #[test]
    fn test_agents_md_created_from_scratch() {
        let store = populated_store();
        let tmp = tempfile::TempDir::new().unwrap();
        let report = generate_all_rules(&store, tmp.path()).unwrap();

        assert!(
            report.formats.contains(&"AGENTS.md".to_string()),
            "Should write AGENTS.md"
        );

        let agents = std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap();
        assert!(agents.contains(AGENTS_MD_START));
        assert!(agents.contains(AGENTS_MD_END));
        assert!(agents.contains("Rust"), "Should contain preferences");
        assert!(agents.contains("Cortex"), "Should reference Cortex");
    }

    #[test]
    fn test_agents_md_preserves_user_content() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create existing AGENTS.md with user content
        std::fs::write(
            tmp.path().join("AGENTS.md"),
            "# My Project\n\nThis is my project.\n\n## Build\n\n```bash\ncargo build\n```\n",
        )
        .unwrap();

        let store = populated_store();
        generate_all_rules(&store, tmp.path()).unwrap();

        let agents = std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap();
        assert!(
            agents.contains("# My Project"),
            "Should preserve user heading"
        );
        assert!(
            agents.contains("cargo build"),
            "Should preserve user build instructions"
        );
        assert!(
            agents.contains(AGENTS_MD_START),
            "Should add Cortex section"
        );
        assert!(agents.contains("Rust"), "Should contain Cortex rules");
    }

    #[test]
    fn test_agents_md_replaces_existing_cortex_section() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create AGENTS.md with an existing Cortex section
        let content = format!(
            "# My Project\n\n{AGENTS_MD_START}\n## Old Cortex Section\n- old rule\n{AGENTS_MD_END}\n\n## Build\ncargo build\n"
        );
        std::fs::write(tmp.path().join("AGENTS.md"), &content).unwrap();

        let store = populated_store();
        generate_all_rules(&store, tmp.path()).unwrap();

        let agents = std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap();
        assert!(
            !agents.contains("old rule"),
            "Should replace old Cortex section"
        );
        assert!(agents.contains("Rust"), "Should contain new Cortex content");
        assert!(
            agents.contains("# My Project"),
            "Should preserve content before markers"
        );
        assert!(
            agents.contains("cargo build"),
            "Should preserve content after markers"
        );
    }

    #[test]
    fn test_generate_all_writes_claude_rules_when_detected() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create .claude/ directory to simulate Claude Code project
        std::fs::create_dir_all(tmp.path().join(".claude")).unwrap();

        let store = populated_store();
        let report = generate_all_rules(&store, tmp.path()).unwrap();

        assert!(
            report.formats.contains(&"AGENTS.md".to_string()),
            "Should write AGENTS.md"
        );
        assert!(
            report.formats.contains(&".claude/rules/".to_string()),
            "Should write .claude/rules/ when .claude/ exists"
        );

        // Verify Claude rules exist
        assert!(tmp
            .path()
            .join(".claude/rules/cortex-preferences.md")
            .exists());
    }

    #[test]
    fn test_generate_all_skips_claude_rules_when_no_claude_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No .claude/ directory

        let store = populated_store();
        let report = generate_all_rules(&store, tmp.path()).unwrap();

        assert!(
            report.formats.contains(&"AGENTS.md".to_string()),
            "Should always write AGENTS.md"
        );
        assert!(
            !report.formats.contains(&".claude/rules/".to_string()),
            "Should NOT write .claude/rules/ without .claude/ dir"
        );
    }

    // -- Full pipeline test --

    #[test]
    fn test_full_pipeline_ingest_to_rules() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory_for_bench().unwrap();
        let config = ConsolidationConfig::default();

        let compact = r#"{"compact_summary": "User prefers functional programming style in TypeScript. Always use const, arrow functions."}"#;
        episodes.ingest("sess1", "post_compact", compact).unwrap();

        let stop = r#"{"last_assistant_message": "I apologize for using var instead of const. The correct approach is to always use const for immutable bindings."}"#;
        episodes.ingest("sess1", "stop", stop).unwrap();

        let user = store
            .save_entity("Developer", EntityType::Person, 1.0, "sift_remember")
            .unwrap();
        store
            .add_observation(
                &user,
                "Prefers functional programming style in TypeScript",
                0.9,
                "sift_remember",
            )
            .unwrap();

        let project = store
            .save_entity("frontend-app", EntityType::Project, 0.9, "sift_remember")
            .unwrap();
        store
            .add_observation(
                &project,
                "Uses Vite for bundling, React 18 with TypeScript strict mode",
                0.9,
                "sift_remember",
            )
            .unwrap();

        let correction = store
            .save_entity(
                "TypeScript conventions",
                EntityType::Fact,
                0.9,
                "sift_remember",
            )
            .unwrap();
        store
            .add_observation(
                &correction,
                "Don't use var — always use const or let. The project enforces this via ESLint.",
                0.9,
                "cortex:correction",
            )
            .unwrap();

        let report = crate::consolidation::run_consolidation(&store, &episodes, &config).unwrap();
        assert!(
            report.episodes_processed >= 2,
            "Should process both episodes"
        );

        // Test generate_all_rules with .claude/ present
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".claude")).unwrap();

        let rule_report = generate_all_rules(&store, tmp.path()).unwrap();
        assert!(
            rule_report.files_written >= 3,
            "Should write AGENTS.md + claude rules"
        );

        // Verify AGENTS.md
        let agents = std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap();
        assert!(agents.contains("functional"));
        assert!(agents.contains("Vite"));

        // Verify Claude rules
        let prefs = std::fs::read_to_string(tmp.path().join(".claude/rules/cortex-preferences.md"))
            .unwrap_or_default();
        assert!(prefs.contains("functional"));

        // Verify correction tagging
        let conn = store.db().lock().unwrap();
        let correction_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE source = 'cortex:correction'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            correction_count >= 1,
            "Should have corrections from sift_remember (Stop extraction requires LLM)"
        );
    }
}
