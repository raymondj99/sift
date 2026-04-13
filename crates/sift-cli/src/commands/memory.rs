//! `sift memory` subcommand — manage the Cortex automated memory system.
//!
//! Subcommands:
//! - `ingest`: Hot path — capture a hook event into the episodes table
//! - `consolidate`: Run the 5-phase consolidation pipeline
//! - `status`: Show memory system statistics
//! - `init-hooks`: Print recommended Claude Code hooks JSON

#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::Config;
use sift_memory::episodes::{self, EpisodeStore};

/// Memory directory path (alongside the search index).
fn memory_dir(config: &Config) -> anyhow::Result<std::path::PathBuf> {
    Ok(config.index_dir()?.join("memory"))
}

/// Ingest a hook event from stdin.
///
/// This is the hot path — called by Claude Code hooks on every relevant
/// tool use. It must complete in <100ms. Opens only SQLite, no search
/// engine or embedding model.
///
/// Reads JSON from stdin, applies the attention filter, and writes to
/// the episodes table if the event is worth encoding.
pub fn ingest(config: &Config, event_type: &str) -> anyhow::Result<()> {
    let dir = memory_dir(config)?;
    let store = EpisodeStore::open(&dir)?;

    // Read hook payload from stdin
    let content = std::io::read_to_string(std::io::stdin())?;
    if content.trim().is_empty() {
        // No input — nothing to ingest (hooks may fire with empty stdin)
        return Ok(());
    }

    let session_id = episodes::session_id_from_env();

    match store.ingest(&session_id, event_type, &content)? {
        Some(id) => {
            tracing::debug!("Ingested episode {id} (event={event_type})");
        }
        None => {
            tracing::debug!("Skipped event (event={event_type}, filtered by attention)");
        }
    }

    Ok(())
}

/// Show memory system status.
pub fn status(config: &Config) -> anyhow::Result<()> {
    let dir = memory_dir(config)?;

    if !dir.join("memory.db").exists() {
        println!("{} Memory system not initialized", "●".dimmed());
        println!("  Run {} to set up hooks", "sift memory init-hooks".bold());
        return Ok(());
    }

    // Single connection for all queries — no reason to open two.
    let store = EpisodeStore::open(&dir)?;
    let conn = store.connection();
    let pending = store.pending_count()?;
    let total = store.total_count()?;

    let total_entities: u64 =
        conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
    let total_observations: u64 = conn.query_row(
        "SELECT COUNT(*) FROM observations WHERE valid_until IS NULL",
        [],
        |row| row.get(0),
    )?;
    let total_relations: u64 = conn.query_row(
        "SELECT COUNT(*) FROM relations WHERE valid_until IS NULL",
        [],
        |row| row.get(0),
    )?;

    // Tier breakdown
    let mut tier_stmt = conn.prepare(
        "SELECT memory_tier, COUNT(*) FROM observations
         WHERE valid_until IS NULL GROUP BY memory_tier",
    )?;
    let tiers: Vec<(String, u64)> = tier_stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(Result::ok)
        .collect();

    let last_consolidation: Option<String> = conn
        .query_row(
            "SELECT value FROM memory_meta WHERE key = 'last_consolidation'",
            [],
            |row| row.get(0),
        )
        .ok();

    // Memory config
    let mem_config = &config.memory;

    println!("{}", "Cortex Memory System".bold());
    println!(
        "  Status:  {}",
        if mem_config.enabled {
            "enabled".green().to_string()
        } else {
            "disabled".red().to_string()
        }
    );
    println!();

    println!("{}", "Knowledge Graph".bold());
    println!("  Entities:      {total_entities}");
    println!("  Observations:  {total_observations}");
    println!("  Relations:     {total_relations}");
    println!();

    if !tiers.is_empty() {
        println!("{}", "Memory Tiers".bold());
        for (tier, count) in &tiers {
            let label = match tier.as_str() {
                "episodic" => "Episodic (raw)".to_string(),
                "semantic" => "Semantic (consolidated)".to_string(),
                "procedural" => "Procedural (skills)".to_string(),
                other => other.to_string(),
            };
            println!("  {label:<30} {count}");
        }
        println!();
    }

    println!("{}", "Episodes".bold());
    println!("  Total ingested:  {total}");
    println!(
        "  Pending:         {}",
        if pending > 0 {
            pending.to_string().yellow().to_string()
        } else {
            "0".green().to_string()
        }
    );
    if let Some(ts) = last_consolidation {
        println!("  Last consolidation: {ts}");
    } else {
        println!("  Last consolidation: {}", "never".dimmed());
    }
    println!();

    println!("{}", "Configuration".bold());
    println!(
        "  Decay rate:         {} (50% at {:.0} days)",
        mem_config.decay_rate,
        0.693 / mem_config.decay_rate // ln(2) / lambda
    );
    println!(
        "  Promotion:          {} days or {} accesses",
        mem_config.promotion_age_days, mem_config.promotion_min_access
    );
    println!(
        "  Dedup threshold:    {}",
        mem_config.semantic_dedup_threshold
    );
    println!(
        "  Skill extraction:   {}",
        if mem_config.skill_extraction {
            "enabled"
        } else {
            "disabled"
        }
    );

    Ok(())
}

/// Run the consolidation pipeline.
///
/// Opens the full MemoryStore (with search engine) and EpisodeStore,
/// runs all 5 phases (or a specific phase), and prints a report.
pub fn consolidate(config: &Config, phase: Option<&str>) -> anyhow::Result<()> {
    let dir = memory_dir(config)?;

    if !dir.join("memory.db").exists() {
        anyhow::bail!("Memory system not initialized. Run `sift memory init-hooks` first.");
    }

    if phase.is_some() {
        // TODO: per-phase execution. For now, always run all phases.
        println!(
            "{}",
            "Per-phase filtering not yet implemented, running all phases.".dimmed()
        );
    }

    let memory_store = sift_memory::MemoryStore::open(&dir)?;
    let memory_store = memory_store.with_decay_rate(config.memory.decay_rate);
    let episode_store = EpisodeStore::open(&dir)?;
    let consolidation_config = sift_memory::ConsolidationConfig::from(&config.memory);

    let start = std::time::Instant::now();
    let report = sift_memory::consolidation::run_consolidation(
        &memory_store,
        &episode_store,
        &consolidation_config,
    )?;
    let elapsed = start.elapsed();

    // Save search index changes to disk
    memory_store.save()?;

    println!("{}", "Consolidation Report".bold());
    println!("  Time:                {:.1}s", elapsed.as_secs_f64());
    println!();

    println!("{}", "Phase 1: Episode Processing".bold());
    println!("  Episodes processed:  {}", report.episodes_processed);
    println!("  Episodes skipped:    {}", report.episodes_skipped);
    println!("  Entities created:    {}", report.entities_created);
    println!("  Observations created:{}", report.observations_created);
    println!();

    println!("{}", "Phase 2: Deduplication".bold());
    println!("  Duplicates merged:   {}", report.duplicates_merged);
    println!("  Contradictions:      {}", report.contradictions_found);
    println!();

    println!("{}", "Phase 3: Promotion".bold());
    println!("  Episodic → Semantic: {}", report.promotions);
    println!();

    println!("{}", "Phase 4: Skill Extraction".bold());
    println!("  Skills created:      {}", report.skills_created);
    println!("  Skills updated:      {}", report.skills_updated);
    println!();

    println!("{}", "Phase 5: Decay & Pruning".bold());
    println!("  Observations pruned: {}", report.observations_pruned);
    println!("  Entities pruned:     {}", report.entities_pruned);

    Ok(())
}

/// Generate .claude/rules/ files from consolidated memory.
pub fn generate_rules(config: &Config) -> anyhow::Result<()> {
    let dir = memory_dir(config)?;

    if !dir.join("memory.db").exists() {
        anyhow::bail!("Memory system not initialized. Run some sessions with hooks first.");
    }

    let memory_store = sift_memory::MemoryStore::open(&dir)?;
    let cwd = std::env::current_dir().unwrap_or_default();

    // Determine project root — use git root or cwd.
    let project_root = sift_core::Config::find_project_root(&cwd).unwrap_or(cwd);

    println!(
        "{}",
        format!("Generating rules for {}", project_root.display()).bold()
    );

    let report = sift_memory::rules::generate_all_rules(&memory_store, &project_root)?;

    println!("  Files written:    {}", report.files_written);
    println!("  Total rules:      {}", report.total_rules);
    println!("  Approx tokens:    {}", report.total_tokens_approx);
    println!("  Formats:          {}", report.formats.join(", "));

    if report.files_written == 0 {
        println!(
            "\n{}",
            "No rules generated — run some sessions and consolidate first.".dimmed()
        );
    } else {
        let targets: Vec<&str> = report
            .formats
            .iter()
            .map(|f| match f.as_str() {
                "AGENTS.md" => "AGENTS.md (20+ tools)",
                ".claude/rules/" => ".claude/rules/ (Claude Code)",
                other => other,
            })
            .collect();
        println!(
            "\n{}",
            format!("Rules written to: {}", targets.join(", ")).green()
        );
    }

    Ok(())
}

/// Print recommended Claude Code hooks configuration to stdout.
pub fn init_hooks() -> anyhow::Result<()> {
    // Resolve the sift binary path
    let exe =
        std::env::current_exe().map_or_else(|_| "sift".to_string(), |p| p.display().to_string());

    let hooks_json = serde_json::json!({
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Edit|Write|MultiEdit|Bash|NotebookEdit|mcp__*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": format!("{exe} memory ingest --event post-tool-use"),
                            "timeout": 5000
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": format!("{exe} memory ingest --event stop"),
                            "timeout": 5000
                        }
                    ]
                }
            ],
            "PostCompact": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": format!("{exe} memory ingest --event post-compact"),
                            "timeout": 5000
                        }
                    ]
                }
            ]
        }
    });

    println!(
        "{}",
        "Add the following to your Claude Code settings:".bold()
    );
    println!("  File: ~/.claude/settings.json");
    println!();
    println!(
        "{}",
        serde_json::to_string_pretty(&hooks_json).unwrap_or_default()
    );
    println!();
    println!(
        "{}",
        "Or merge the \"hooks\" key into your existing settings.".dimmed()
    );

    Ok(())
}
