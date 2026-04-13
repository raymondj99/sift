//! Generate `.claude/rules/` files from consolidated memory.
//!
//! This is the agent reasoning layer: transforms passive stored memories
//! into proactive context that Claude Code auto-loads at session start.
//!
//! Quality over quantity: only high-confidence, concise, actionable
//! observations become rules. Token budget is 5K total — exceeding that
//! degrades agent performance (PlugMem, Microsoft 2026).

use crate::MemoryStore;
use std::collections::HashSet;
use std::path::Path;
use tracing::info;

/// Result of rule generation.
#[derive(Debug, Default)]
pub struct RuleGenReport {
    pub files_written: usize,
    pub total_rules: usize,
    pub total_tokens_approx: usize,
    pub skipped_noise: usize,
    pub skipped_long: usize,
}

/// Maximum word count for a single rule entry.
const MAX_ENTRY_WORDS: usize = 80;

/// Total token budget across all rule files.
const MAX_TOTAL_TOKENS: usize = 5000;

/// Per-category token budget.
const MAX_CATEGORY_TOKENS: usize = MAX_TOTAL_TOKENS / 4;

/// Rule file names managed by Cortex (cleaned up between runs).
const RULE_FILES: &[&str] = &[
    "cortex-preferences.md",
    "cortex-corrections.md",
    "cortex-decisions.md",
    "cortex-workflows.md",
];

/// Generate rule files from the memory store's consolidated observations.
///
/// Pipeline:
/// 1. Query active observations (prefer semantic tier, skip noise)
/// 2. Classify into categories using entity name + type + content signals
/// 3. Deduplicate within each category
/// 4. Write `.md` files to the rules directory (capped at token budget)
pub fn generate_rules(memory: &MemoryStore, rules_dir: &Path) -> crate::MemResult<RuleGenReport> {
    let mut report = RuleGenReport::default();

    std::fs::create_dir_all(rules_dir)
        .map_err(|e| crate::MemoryError::InvalidInput(format!("Cannot create rules dir: {e}")))?;

    // Clean stale cortex-* files from prior runs before writing new ones.
    for name in RULE_FILES {
        let path = rules_dir.join(name);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
    }

    let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());

    let mut preferences: Vec<String> = Vec::new();
    let mut corrections: Vec<String> = Vec::new();
    let mut decisions: Vec<String> = Vec::new();
    let mut workflows: Vec<String> = Vec::new();

    let mut seen: HashSet<String> = HashSet::new();

    // Query active observations with entity metadata.
    // Order: procedural > semantic > episodic, then by confidence and access count.
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
                _source: row.get(2)?,
                _tier: row.get(3)?,
                entity_name: row.get(4)?,
                entity_type: row.get(5)?,
            })
        })?
        .filter_map(Result::ok)
        .collect();

    for row in &rows {
        // --- Quality gate ---

        if row.confidence < 0.5 {
            continue;
        }

        if is_noise(&row.content, &row.entity_name, &row.entity_type) {
            report.skipped_noise += 1;
            continue;
        }

        let word_count = row.content.split_whitespace().count();
        if word_count > MAX_ENTRY_WORDS {
            report.skipped_long += 1;
            continue;
        }
        if word_count < 5 {
            report.skipped_noise += 1;
            continue;
        }

        // Dedup by normalized content.
        let norm = row.content.to_lowercase().trim().to_string();
        if seen.contains(&norm) {
            continue;
        }
        seen.insert(norm);

        // --- Classification ---
        match classify(row) {
            Category::Preference => preferences.push(format!("- {}", row.content)),
            Category::Correction => corrections.push(format!("- {}", row.content)),
            Category::Decision => {
                decisions.push(format!("- [{}] {}", row.entity_name, row.content));
            }
            Category::Workflow => workflows.push(format!("- {}", row.content)),
            Category::Skip => {}
        }
    }

    // Collect skills as workflow rules.
    let skill_rows: Vec<(String, String, i64)> = conn
        .prepare("SELECT name, pattern, frequency FROM skills ORDER BY frequency DESC LIMIT 10")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .filter_map(Result::ok)
        .collect();

    for (_name, pattern, frequency) in &skill_rows {
        if *frequency >= 3 {
            workflows.push(format!("- {pattern} (detected {frequency} times)"));
        }
    }

    drop(stmt);
    drop(conn);

    // --- Write rule files ---
    let mut total_tokens = 0usize;

    total_tokens += write_rule_file(
        rules_dir,
        "cortex-preferences.md",
        "User Preferences",
        "These preferences were learned from past sessions. Apply them without asking.",
        &preferences,
        &mut report,
    )?;

    total_tokens += write_rule_file(
        rules_dir,
        "cortex-corrections.md",
        "Corrections & Patterns to Avoid",
        "These mistakes were corrected in past sessions. Do not repeat them.",
        &corrections,
        &mut report,
    )?;

    total_tokens += write_rule_file(
        rules_dir,
        "cortex-decisions.md",
        "Project Decisions",
        "Architecture and technology choices from past sessions.",
        &decisions,
        &mut report,
    )?;

    total_tokens += write_rule_file(
        rules_dir,
        "cortex-workflows.md",
        "Workflow Patterns",
        "Tool usage patterns detected across sessions.",
        &workflows,
        &mut report,
    )?;

    report.total_tokens_approx = total_tokens;

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

// --- Internal types ---

struct ObservationRow {
    content: String,
    confidence: f32,
    _source: String,
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

// --- Noise filtering ---

/// Returns true if this observation is noise that should never be a rule.
fn is_noise(content: &str, entity_name: &str, entity_type: &str) -> bool {
    // XML/HTML tags = conversation summaries or payload dumps.
    if content.contains('<') && content.contains('>') {
        return true;
    }

    // File-edit records from episode processing.
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

    // Session events (episode artifacts, not rules).
    if entity_type == "event" || entity_name.starts_with("session:") {
        return true;
    }

    // File-path entities (from episode processing).
    if entity_name.contains('/') || entity_name.ends_with(".rs") || entity_name.ends_with(".toml") {
        return true;
    }

    // Research notes and analysis — informational, not actionable.
    let name_lower = entity_name.to_lowercase();
    if name_lower.contains("research")
        || name_lower.contains("benchmarking")
        || name_lower.contains("failure analysis")
    {
        return true;
    }

    false
}

// --- Classification ---

/// Classify an observation into a rule category.
///
/// Uses a layered approach:
/// 1. Entity type (strongest signal — preference, tool, fact, project)
/// 2. Entity name patterns (design, config, protocol, etc.)
/// 3. Content keyword signals (for concept entities)
fn classify(row: &ObservationRow) -> Category {
    let name_lower = row.entity_name.to_lowercase();
    let content_lower = row.content.to_lowercase();

    // --- Layer 1: Entity type (strongest signal) ---
    match row.entity_type.as_str() {
        "preference" | "person" => return Category::Preference,
        "tool" => return Category::Workflow,

        // Facts: almost always decisions or corrections.
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

        // Project entities → decisions.
        "project" => return Category::Decision,

        _ => {}
    }

    // --- Layer 2: Entity name patterns (for concept/event types) ---

    // Person name heuristic — only for concept entities where the type
    // is ambiguous. Don't apply to projects, facts, etc.
    if row.entity_type == "concept" && is_person_entity(&name_lower) {
        return Category::Preference;
    }

    // Architecture/design/protocol entities → decisions.
    if name_lower.contains("design")
        || name_lower.contains("architecture")
        || name_lower.contains("protocol")
    {
        return Category::Decision;
    }

    // Config entities → decisions (infrastructure choices).
    if name_lower.contains("config") {
        return Category::Decision;
    }

    // Phase/plan entities → decisions (implementation choices).
    if name_lower.contains("phase ") || name_lower.contains(" plan") {
        return Category::Decision;
    }

    // Hook schema entities → decisions.
    if name_lower.contains("hook") || name_lower.contains("schema") {
        return Category::Decision;
    }

    // --- Layer 3: Content signals (for concept entities) ---

    if row.entity_type == "concept" {
        // Coding patterns → corrections or decisions.
        if name_lower.contains("pattern") {
            if content_lower.starts_with("don't ")
                || content_lower.starts_with("avoid ")
                || content_lower.contains("instead")
            {
                return Category::Correction;
            }
            return Category::Decision;
        }

        // Named project/tool entities stored as concept.
        if name_lower == "sift" || name_lower.starts_with("cortex") {
            return Category::Decision;
        }

        // "MCP protocol" etc.
        if name_lower.contains("mcp") {
            return Category::Decision;
        }
    }

    Category::Skip
}

/// Heuristic: is this entity name a person?
fn is_person_entity(name_lower: &str) -> bool {
    // Single capitalized word or known names. We use a simple heuristic:
    // no spaces + no special chars + not a known concept/tool name.
    let is_single_word = !name_lower.contains(' ')
        && !name_lower.contains('/')
        && !name_lower.contains(':')
        && !name_lower.contains('.');

    // Known project/tool names that could be confused with person names.
    let known_non_persons = ["sift", "cortex", "rust", "sqlite", "tantivy", "mcp"];

    is_single_word && !known_non_persons.contains(&name_lower)
}

// --- File writing ---

/// Write a single rule file if there's content. Returns token count.
fn write_rule_file(
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

    let body = truncate_rules(entries, MAX_CATEGORY_TOKENS);
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

/// Truncate a list of rules to fit within a token budget.
fn truncate_rules(rules: &[String], max_tokens: usize) -> String {
    let mut result = String::new();
    let mut tokens = 0usize;

    for rule in rules {
        let rule_tokens = rule.split_whitespace().count();
        if tokens + rule_tokens > max_tokens {
            break;
        }
        result.push_str(rule);
        result.push('\n');
        tokens += rule_tokens;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeStore;
    use crate::types::*;

    /// Helper: set up a memory store with realistic observations.
    fn populated_store() -> MemoryStore {
        let store = MemoryStore::open_in_memory().unwrap();

        // Person entity — should go to preferences
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

        // Project decision entity — should go to decisions
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

        // Fact entity (MCP config) — should go to decisions
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

        // Correction — should go to corrections (via fact entity + content)
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

        // Research entity — should be filtered as noise
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

        // Session event — should be filtered as noise
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

        // File-path entity — should be filtered as noise
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

        // Read all generated files and verify noise is absent
        let all_content: String = RULE_FILES
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
    fn test_stale_files_are_cleaned() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create a stale rule file
        std::fs::write(
            tmp.path().join("cortex-workflows.md"),
            "# Stale\n- old rule\n",
        )
        .unwrap();

        let store = populated_store();
        generate_rules(&store, tmp.path()).unwrap();

        // The stale file should be removed if no workflows were generated
        let workflows_path = tmp.path().join("cortex-workflows.md");
        assert!(
            !workflows_path.exists(),
            "Stale workflow file should be cleaned up"
        );
    }

    #[test]
    fn test_token_budget_is_respected() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Create many observations to exceed the budget
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
            report.total_tokens_approx <= MAX_TOTAL_TOKENS + 100, // small margin for headers
            "Total tokens ({}) should be within budget ({})",
            report.total_tokens_approx,
            MAX_TOTAL_TOKENS
        );
    }

    /// Full pipeline: episode ingestion + sift_remember + consolidation + rules.
    ///
    /// The rule generator primarily works on intentional memory (sift_remember)
    /// not raw episode artifacts (which are session events, correctly filtered
    /// as noise). This test simulates a realistic scenario with both.
    #[test]
    fn test_full_pipeline_ingest_to_rules() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory_for_bench().unwrap();
        let config = ConsolidationConfig::default();

        // 1. Simulate episode ingestion (raw events)
        let compact = r#"{"compact_summary": "User prefers functional programming style in TypeScript. Always use const, arrow functions."}"#;
        episodes.ingest("sess1", "post_compact", compact).unwrap();

        let stop = r#"{"last_assistant_message": "I apologize for using var instead of const. The correct approach is to always use const for immutable bindings."}"#;
        episodes.ingest("sess1", "stop", stop).unwrap();

        // 2. Simulate sift_remember (intentional memory storage)
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

        // 3. Run consolidation (processes episodes)
        let report = crate::consolidation::run_consolidation(&store, &episodes, &config).unwrap();
        assert!(
            report.episodes_processed >= 2,
            "Should process both episodes"
        );

        // 4. Generate rules
        let tmp = tempfile::TempDir::new().unwrap();
        let rule_report = generate_rules(&store, tmp.path()).unwrap();
        assert!(
            rule_report.files_written >= 2,
            "Should write preferences + decisions + corrections"
        );

        // 5. Verify rule content
        let prefs =
            std::fs::read_to_string(tmp.path().join("cortex-preferences.md")).unwrap_or_default();
        assert!(
            prefs.contains("functional"),
            "Preferences should mention functional style"
        );

        let decisions =
            std::fs::read_to_string(tmp.path().join("cortex-decisions.md")).unwrap_or_default();
        assert!(
            decisions.contains("Vite"),
            "Decisions should mention bundler"
        );

        let corrections =
            std::fs::read_to_string(tmp.path().join("cortex-corrections.md")).unwrap_or_default();
        assert!(
            corrections.contains("var"),
            "Corrections should mention var prohibition"
        );

        // 6. Verify correction tagging from Stop event
        let conn = store.db().lock().unwrap();
        let correction_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE source = 'cortex:correction'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        // At least 2: one from sift_remember, one from the Stop event
        assert!(
            correction_count >= 2,
            "Should have corrections from both sift_remember and Stop event"
        );
    }
}
