//! Five-phase consolidation pipeline for the Cortex memory system.
//!
//! Runs in the background (daemon timer or CLI trigger) to process raw
//! episodes into structured knowledge. No LLM calls — all extraction is
//! rule-based parsing of structured hook JSON, and all similarity uses
//! local embeddings or Jaccard text overlap.
//!
//! ## Phases
//!
//! 1. **Episode processing** — parse pending episodes into entities + observations
//! 2. **Deduplication** — exact text + embedding cosine similarity merge
//! 3. **Promotion** — episodic → semantic tier upgrade for mature observations
//! 4. **Skill extraction** — detect repeated patterns → procedural knowledge
//! 5. **Decay & pruning** — utility scoring, forgetting curve, garbage collection

use crate::episodes::EpisodeStore;
use crate::types::*;
use crate::MemoryStore;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Result of a full consolidation run across all phases.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FullConsolidationReport {
    pub episodes_processed: usize,
    pub episodes_skipped: usize,
    /// Episodes recognized but not extracted this run because the
    /// extractor wasn't applicable yet (e.g. LLM disabled, payload empty).
    /// Deferred episodes stay `Pending` and become eligible again on the
    /// next consolidation run, so the count can change between runs as
    /// LLM availability changes.
    pub episodes_deferred: usize,
    pub entities_created: usize,
    pub observations_created: usize,
    pub duplicates_merged: usize,
    pub contradictions_found: usize,
    pub promotions: usize,
    pub skills_created: usize,
    pub skills_updated: usize,
    pub observations_pruned: usize,
    pub entities_pruned: usize,
}

/// Per-episode extraction outcome.
///
/// `Done` is the success case — entities/observations were created (or the
/// extractor confirmed there was nothing to extract from a real payload).
/// `Deferred` means the extractor couldn't run yet; the episode stays
/// `Pending` so a future run with LLM available can retry.
#[derive(Debug, Clone, Copy)]
enum ExtractOutcome {
    Done {
        entities: usize,
        observations: usize,
    },
    Deferred(DeferReason),
}

#[derive(Debug, Clone, Copy)]
enum DeferReason {
    /// LLM extraction is disabled or the per-run circuit breaker has tripped.
    LlmUnavailable,
    /// The payload was missing or below the length floor for extraction.
    EmptyOrTooShort,
}

/// Run all consolidation phases in sequence.
///
/// Opens the episode store for phase 1 (episode processing), then runs
/// the remaining phases against the memory store. Writes a
/// `last_consolidation` timestamp to `memory_meta` on success.
pub fn run_consolidation(
    memory: &MemoryStore,
    episodes: &EpisodeStore,
    config: &ConsolidationConfig,
) -> crate::MemResult<FullConsolidationReport> {
    let mut report = FullConsolidationReport::default();

    // Phase 1: Process pending episodes into entities + observations
    let phase1 = phase_episode_processing(memory, episodes, config)?;
    report.episodes_processed = phase1.processed;
    report.episodes_skipped = phase1.skipped;
    report.episodes_deferred = phase1.deferred;
    report.entities_created = phase1.entities;
    report.observations_created = phase1.observations;

    // Phase 2: Deduplication (extends existing consolidate())
    let dedup = phase_deduplication(memory, config)?;
    report.duplicates_merged = dedup.duplicates_merged;
    report.contradictions_found = dedup.contradictions_found;

    // Phase 3: Episodic → Semantic promotion
    report.promotions = phase_promotion(memory, config)?;

    // Phase 4: Skill extraction
    let (created, updated) = phase_skill_extraction(memory, episodes, config)?;
    report.skills_created = created;
    report.skills_updated = updated;

    // Phase 5: Decay & pruning
    let (obs_pruned, ent_pruned) = phase_decay_and_pruning(memory, config)?;
    report.observations_pruned = obs_pruned;
    report.entities_pruned = ent_pruned;

    // Record consolidation timestamp
    {
        let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());
        let now = crate::now_secs();
        let _ = conn.execute(
            "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('last_consolidation', ?1)",
            [now.to_string()],
        );
    }

    info!(
        episodes = report.episodes_processed,
        deferred = report.episodes_deferred,
        entities = report.entities_created,
        observations = report.observations_created,
        dedup = report.duplicates_merged,
        promotions = report.promotions,
        skills = report.skills_created,
        pruned = report.observations_pruned,
        "Consolidation complete"
    );

    Ok(report)
}

#[derive(Debug, Default)]
struct EpisodePhaseStats {
    processed: usize,
    skipped: usize,
    deferred: usize,
    entities: usize,
    observations: usize,
}

// ===========================================================================
// Phase 1: Episode Processing
// ===========================================================================

/// Parse pending episodes into entities and observations.
///
/// Extraction is rule-based — no LLM calls:
/// - **PostCompact**: `compact_summary` → session entity + observation (gold)
/// - **Stop**: `last_assistant_message` → session entity + observation (silver)
/// - **PostToolUse**: tool_input file paths → file entities + action observations
///
/// Returns counts of processed/skipped/deferred episodes plus
/// entities/observations created.
fn phase_episode_processing(
    memory: &MemoryStore,
    episodes: &EpisodeStore,
    config: &ConsolidationConfig,
) -> crate::MemResult<EpisodePhaseStats> {
    let pending = episodes
        .fetch_pending(500)
        .map_err(crate::MemoryError::Sqlite)?;

    if pending.is_empty() {
        return Ok(EpisodePhaseStats::default());
    }

    let batch_id = uuid::Uuid::now_v7().to_string();
    let mut processed_ids: Vec<String> = Vec::new();
    let mut skipped_ids: Vec<String> = Vec::new();
    let mut deferred_count = 0usize;
    let mut deferred_reasons: HashMap<&'static str, usize> = HashMap::new();
    let mut entities_created = 0usize;
    let mut observations_created = 0usize;

    // Per-run LLM circuit breaker: one failure disables LLM for the rest
    // of this consolidation so a bad key or dead endpoint can't burn 500
    // serial requests × request-timeout seconds.
    let llm = LlmRuntime::new(config);

    // Group episodes by session for context
    let mut by_session: HashMap<String, Vec<&Episode>> = HashMap::new();
    for ep in &pending {
        by_session
            .entry(ep.session_id.clone())
            .or_default()
            .push(ep);
    }

    for session_episodes in by_session.values() {
        for ep in session_episodes {
            let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&ep.content) else {
                skipped_ids.push(ep.id.clone());
                continue;
            };

            let result = match ep.event_type {
                EventType::PostCompact => extract_from_post_compact(memory, ep, &parsed, &llm),
                EventType::Stop => extract_from_stop(memory, ep, &parsed, &llm),
                EventType::PostToolUse => extract_from_post_tool_use(memory, ep, &parsed),
            };

            match result {
                Ok(ExtractOutcome::Done {
                    entities,
                    observations,
                }) => {
                    entities_created += entities;
                    observations_created += observations;
                    processed_ids.push(ep.id.clone());
                }
                Ok(ExtractOutcome::Deferred(reason)) => {
                    deferred_count += 1;
                    *deferred_reasons.entry(reason.as_static_str()).or_insert(0) += 1;
                    // Leave the episode `Pending` so a later run with LLM
                    // available (or richer payload) can extract from it.
                }
                Err(e) => {
                    warn!(episode_id = %ep.id, error = %e, "Episode processing failed");
                    skipped_ids.push(ep.id.clone());
                }
            }
        }
    }

    // Mark episodes as processed/skipped. Deferred episodes stay Pending.
    if !processed_ids.is_empty() {
        episodes
            .mark_processed(&processed_ids, EpisodeState::Processed, &batch_id)
            .map_err(crate::MemoryError::Sqlite)?;
    }
    if !skipped_ids.is_empty() {
        episodes
            .mark_processed(&skipped_ids, EpisodeState::Skipped, &batch_id)
            .map_err(crate::MemoryError::Sqlite)?;
    }

    debug!(
        processed = processed_ids.len(),
        skipped = skipped_ids.len(),
        deferred = deferred_count,
        deferred_reasons = ?deferred_reasons,
        "Phase 1: episode processing"
    );

    Ok(EpisodePhaseStats {
        processed: processed_ids.len(),
        skipped: skipped_ids.len(),
        deferred: deferred_count,
        entities: entities_created,
        observations: observations_created,
    })
}

impl DeferReason {
    fn as_static_str(self) -> &'static str {
        match self {
            DeferReason::LlmUnavailable => "llm_unavailable",
            DeferReason::EmptyOrTooShort => "empty_or_too_short",
        }
    }
}

/// Extract knowledge from a PostCompact event.
///
/// The `compact_summary` is pre-distilled by the LLM — the highest value
/// signal. LLM decomposition breaks it into individually categorized
/// observations on named entities. If the LLM is unavailable, the episode
/// is skipped — heuristic extraction was removed because it produced
/// monolithic blobs on session entities that ranked poorly in recall.
fn extract_from_post_compact(
    memory: &MemoryStore,
    _ep: &Episode,
    parsed: &serde_json::Value,
    llm: &LlmRuntime<'_>,
) -> crate::MemResult<ExtractOutcome> {
    let summary = match parsed.get("compact_summary").and_then(|v| v.as_str()) {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return Ok(ExtractOutcome::Deferred(DeferReason::EmptyOrTooShort)),
    };

    match llm.try_extract(memory, summary) {
        Some(result) => result.map(|(entities, observations)| ExtractOutcome::Done {
            entities,
            observations,
        }),
        None => Ok(ExtractOutcome::Deferred(DeferReason::LlmUnavailable)),
    }
}

/// Extract knowledge from a Stop event.
///
/// The `last_assistant_message` contains the agent's final output. LLM
/// decomposition extracts categorized observations from the full message.
/// If the LLM is unavailable, the episode is skipped.
fn extract_from_stop(
    memory: &MemoryStore,
    _ep: &Episode,
    parsed: &serde_json::Value,
    llm: &LlmRuntime<'_>,
) -> crate::MemResult<ExtractOutcome> {
    let message = match parsed
        .get("last_assistant_message")
        .and_then(|v| v.as_str())
    {
        Some(s) if s.len() > 20 => s,
        _ => return Ok(ExtractOutcome::Deferred(DeferReason::EmptyOrTooShort)),
    };

    match llm.try_extract(memory, message) {
        Some(result) => result.map(|(entities, observations)| ExtractOutcome::Done {
            entities,
            observations,
        }),
        None => Ok(ExtractOutcome::Deferred(DeferReason::LlmUnavailable)),
    }
}

/// Extract knowledge from a PostToolUse event.
///
/// For write tools: extract the file path as an entity and the action as
/// an observation. This captures what files were modified and how.
fn extract_from_post_tool_use(
    memory: &MemoryStore,
    ep: &Episode,
    parsed: &serde_json::Value,
) -> crate::MemResult<ExtractOutcome> {
    let tool_name = match ep.tool_name.as_deref() {
        Some(t) => t,
        None => {
            return Ok(ExtractOutcome::Done {
                entities: 0,
                observations: 0,
            })
        }
    };

    // Extract file path from tool_input (works for Edit, Write, Read)
    let file_path = extract_file_path(parsed);
    let mut entities = 0usize;
    let mut observations = 0usize;

    if let Some(path) = file_path {
        // Resolve against existing entities to prevent fragmentation
        let short_path = shorten_path(&path);
        let (entity_id, is_new) = if let Some((id, _name)) = memory.resolve_entity(&short_path)? {
            (id, false)
        } else {
            let id = memory.save_entity(&short_path, EntityType::Project, 0.6, "cortex:episode")?;
            (id, true)
        };
        if is_new {
            entities += 1;
        }

        // Create observation about what was done
        let action = match tool_name {
            "Edit" | "MultiEdit" => format!("Modified {short_path}"),
            "Write" => format!("Created/wrote {short_path}"),
            "Bash" => {
                let cmd = extract_bash_command(parsed).unwrap_or_default();
                if cmd.is_empty() {
                    format!("Ran command in context of {short_path}")
                } else {
                    let short_cmd = if cmd.len() > 120 { &cmd[..120] } else { &cmd };
                    format!("Ran: {short_cmd}")
                }
            }
            _ => format!("{tool_name} on {short_path}"),
        };

        memory.add_observation(&entity_id, &action, 0.5, "cortex:tool")?;
        observations += 1;
    } else if tool_name == "Bash" {
        // Bash without a file path — extract the command itself
        if let Some(cmd) = extract_bash_command(parsed) {
            if cmd.len() > 15 {
                let short_cmd = if cmd.len() > 200 { &cmd[..200] } else { &cmd };
                let entity_id = memory.save_entity(
                    &format!("session:{}", &ep.session_id),
                    EntityType::Event,
                    0.5,
                    "cortex:episode",
                )?;
                entities += 1;
                memory.add_observation(
                    &entity_id,
                    &format!("Ran: {short_cmd}"),
                    0.4,
                    "cortex:tool",
                )?;
                observations += 1;
            }
        }
    }

    // --- Correction detection: Bash command failures ---
    // Check if this is a Bash tool use with error indicators in stdout.
    // Build/test failures are high-value correction opportunities.
    if tool_name == "Bash" {
        if let Some(stdout) = parsed
            .get("tool_response")
            .and_then(|r| r.get("stdout"))
            .and_then(|v| v.as_str())
        {
            if let Some(error_summary) = extract_bash_error(stdout) {
                let cmd = extract_bash_command(parsed).unwrap_or_default();
                let short_cmd = if cmd.len() > 80 { &cmd[..80] } else { &cmd };
                let entity_id = memory.save_entity(
                    &format!("session:{}", &ep.session_id),
                    EntityType::Event,
                    0.7,
                    "cortex:episode",
                )?;
                memory.add_observation(
                    &entity_id,
                    &format!("Command `{short_cmd}` failed: {error_summary}"),
                    0.8,
                    "cortex:correction",
                )?;
                observations += 1;
            }
        }
    }

    Ok(ExtractOutcome::Done {
        entities,
        observations,
    })
}

// ===========================================================================
// Phase 2: Deduplication
// ===========================================================================

/// Run deduplication — delegates to existing MemoryStore::consolidate().
fn phase_deduplication(
    memory: &MemoryStore,
    config: &ConsolidationConfig,
) -> crate::MemResult<ConsolidationReport> {
    let report = memory.consolidate_with_threshold(config.semantic_dedup_threshold)?;
    debug!(
        merged = report.duplicates_merged,
        contradictions = report.contradictions_found,
        "Phase 2: deduplication"
    );
    Ok(report)
}

// ===========================================================================
// Phase 3: Episodic → Semantic Promotion
// ===========================================================================

/// Promote mature episodic observations to the semantic tier.
///
/// Eligible: `access_count >= promotion_min_access` OR
/// `age >= promotion_age_days` with `confidence >= 0.7`.
fn phase_promotion(memory: &MemoryStore, config: &ConsolidationConfig) -> crate::MemResult<usize> {
    let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());
    let now = crate::now_secs();
    let age_threshold = now - (i64::from(config.promotion_age_days) * 86400);

    // Find episodic observations eligible for promotion
    let mut stmt = conn.prepare(
        "SELECT id, access_count, observed_at, confidence
         FROM observations
         WHERE memory_tier = 'episodic'
           AND valid_until IS NULL
           AND (access_count >= ?1 OR (observed_at <= ?2 AND confidence >= 0.7))",
    )?;

    let candidates: Vec<(String, i64, i64, f32)> = stmt
        .query_map(
            rusqlite::params![config.promotion_min_access, age_threshold],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?
        .filter_map(Result::ok)
        .collect();

    if candidates.is_empty() {
        return Ok(0);
    }

    // Promote each candidate to semantic tier
    let mut promoted = 0usize;
    for (id, _access_count, _observed_at, _confidence) in &candidates {
        conn.execute(
            "UPDATE observations SET memory_tier = 'semantic' WHERE id = ?1",
            [id],
        )?;
        promoted += 1;
    }

    debug!(promoted, "Phase 3: episodic → semantic promotion");
    Ok(promoted)
}

// ===========================================================================
// Phase 4: Skill Extraction
// ===========================================================================

/// Detect repeated tool patterns across sessions and create skills.
///
/// Scans processed episodes for recurring tool sequences (e.g., "always
/// runs tests after editing"), then creates or updates skill records.
fn phase_skill_extraction(
    memory: &MemoryStore,
    episodes: &EpisodeStore,
    config: &ConsolidationConfig,
) -> crate::MemResult<(usize, usize)> {
    if !config.skill_extraction {
        return Ok((0, 0));
    }

    let conn_ep = episodes.connection();

    // Find tool sequences across sessions: group processed PostToolUse
    // episodes by session, extract tool_name sequences
    let mut stmt = conn_ep.prepare(
        "SELECT session_id, tool_name
         FROM episodes
         WHERE processed = 1 AND event_type = 'post_tool_use' AND tool_name IS NOT NULL
         ORDER BY session_id, timestamp ASC",
    )?;

    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(crate::MemoryError::Sqlite)?
        .filter_map(Result::ok)
        .collect();

    // Build per-session tool sequences
    let mut sessions: HashMap<String, Vec<String>> = HashMap::new();
    for (session_id, tool_name) in &rows {
        sessions
            .entry(session_id.clone())
            .or_default()
            .push(tool_name.clone());
    }

    // Count 2-gram and 3-gram tool patterns across sessions
    let mut bigram_counts: HashMap<String, usize> = HashMap::new();
    let mut trigram_counts: HashMap<String, usize> = HashMap::new();

    for tools in sessions.values() {
        // Deduplicate within session to avoid one session inflating counts
        let mut seen_bigrams = std::collections::HashSet::new();
        let mut seen_trigrams = std::collections::HashSet::new();

        for window in tools.windows(2) {
            let key = format!("{}->{}", window[0], window[1]);
            if seen_bigrams.insert(key.clone()) {
                *bigram_counts.entry(key).or_insert(0) += 1;
            }
        }
        for window in tools.windows(3) {
            let key = format!("{}->{}→{}", window[0], window[1], window[2]);
            if seen_trigrams.insert(key.clone()) {
                *trigram_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let min_freq = config.skill_min_frequency as usize;
    let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());
    let now = crate::now_secs();
    let mut created = 0usize;
    let mut updated = 0usize;

    // Process patterns that meet the frequency threshold
    let patterns: Vec<(String, usize)> = trigram_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_freq)
        .chain(
            bigram_counts
                .into_iter()
                .filter(|(_, count)| *count >= min_freq),
        )
        .collect();

    for (pattern, frequency) in patterns {
        let name = format!("workflow:{pattern}");
        let confidence = (0.3 + 0.1 * frequency as f32).min(1.0);

        // Upsert: update frequency if exists, otherwise create
        let existing: Option<(String, i64)> = conn
            .query_row(
                "SELECT id, frequency FROM skills WHERE name = ?1",
                [&name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((id, old_freq)) = existing {
            let new_freq = old_freq + frequency as i64;
            let new_confidence = (0.3 + 0.1 * new_freq as f32).min(1.0);
            conn.execute(
                "UPDATE skills SET frequency = ?1, confidence = ?2, updated_at = ?3 WHERE id = ?4",
                rusqlite::params![new_freq, new_confidence, now, id],
            )?;
            updated += 1;
        } else {
            let id = uuid::Uuid::now_v7().to_string();
            conn.execute(
                "INSERT INTO skills (id, name, description, pattern, frequency, confidence, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    id,
                    name,
                    format!("Detected tool workflow: {pattern}"),
                    pattern,
                    frequency as i64,
                    confidence,
                    now,
                    now,
                ],
            )?;
            created += 1;
        }
    }

    debug!(created, updated, "Phase 4: skill extraction");
    Ok((created, updated))
}

// ===========================================================================
// Phase 5: Decay & Pruning
// ===========================================================================

/// Apply decay scoring and prune low-utility observations.
///
/// Utility formula (from PlugMem):
///   `(access_freq * confidence) / max(1, token_count / 50)`
/// where `access_freq = ln(1 + access_count)`.
///
/// Pruning rules:
/// - Episodic: utility < threshold AND age > min_days AND access_count = 0 → soft-delete
/// - Semantic: low utility → confidence -= 0.1; delete when < 0.1
/// - Procedural: never auto-delete
fn phase_decay_and_pruning(
    memory: &MemoryStore,
    config: &ConsolidationConfig,
) -> crate::MemResult<(usize, usize)> {
    let conn = memory.db().lock().unwrap_or_else(|e| e.into_inner());
    let now = crate::now_secs();
    let min_age = now - (i64::from(config.prune_min_age_days) * 86400);

    // Update utility scores for all active observations
    let mut stmt = conn.prepare(
        "SELECT id, access_count, confidence, token_count, memory_tier, observed_at, last_accessed
         FROM observations
         WHERE valid_until IS NULL",
    )?;

    struct ObsRow {
        id: String,
        access_count: i64,
        confidence: f32,
        token_count: Option<i64>,
        memory_tier: String,
        observed_at: i64,
        last_accessed: Option<i64>,
    }

    let observations: Vec<ObsRow> = stmt
        .query_map([], |row| {
            Ok(ObsRow {
                id: row.get(0)?,
                access_count: row.get(1)?,
                confidence: row.get(2)?,
                token_count: row.get(3)?,
                memory_tier: row.get(4)?,
                observed_at: row.get(5)?,
                last_accessed: row.get(6)?,
            })
        })?
        .filter_map(Result::ok)
        .collect();

    let mut pruned_obs = 0usize;

    conn.execute_batch("BEGIN")?;

    let mut update_stmt =
        conn.prepare("UPDATE observations SET utility_score = ?1 WHERE id = ?2")?;
    let mut invalidate_stmt =
        conn.prepare("UPDATE observations SET valid_until = ?1, utility_score = ?2 WHERE id = ?3")?;
    let mut decay_stmt = conn.prepare("UPDATE observations SET confidence = ?1 WHERE id = ?2")?;

    for obs in &observations {
        // Compute utility score.
        // Base confidence alone provides non-zero utility — an observation
        // doesn't need to be recalled to be valuable. Retrieval boosts it.
        let access_freq = (1.0 + obs.access_count as f64).ln();
        let tokens = obs.token_count.unwrap_or(50).max(1) as f64;
        let confidence = f64::from(obs.confidence);
        let utility = ((1.0 + access_freq) * confidence) / (tokens / 50.0).max(1.0);

        // Also factor in recency
        let last_active = obs.last_accessed.unwrap_or(obs.observed_at);
        let days_since = (now - last_active).max(0) as f64 / 86400.0;
        let recency = (-config.decay_rate * days_since).exp();
        let final_utility = utility * recency;

        update_stmt.execute(rusqlite::params![final_utility, obs.id])?;

        // Apply pruning rules based on tier
        match obs.memory_tier.as_str() {
            "episodic"
                if final_utility < f64::from(config.prune_utility_threshold)
                    && obs.observed_at < min_age
                    && obs.access_count == 0 =>
            {
                invalidate_stmt.execute(rusqlite::params![now, final_utility, obs.id])?;
                pruned_obs += 1;
            }
            "semantic" if final_utility < f64::from(config.prune_utility_threshold) => {
                let new_confidence = (obs.confidence - 0.1).max(0.0);
                if new_confidence < 0.1 {
                    invalidate_stmt.execute(rusqlite::params![now, final_utility, obs.id])?;
                    pruned_obs += 1;
                } else {
                    decay_stmt.execute(rusqlite::params![new_confidence, obs.id])?;
                }
            }
            // Procedural — never auto-delete; non-matching episodic/semantic cases fall through
            _ => {}
        }
    }

    drop(update_stmt);
    drop(invalidate_stmt);
    drop(decay_stmt);
    drop(stmt);

    conn.execute_batch("COMMIT")?;
    drop(conn);

    // Prune ghost entities left by soft-deleted observations
    let pruned_entities = memory.prune_entities()?;
    let ent_count = pruned_entities.len();

    debug!(
        observations_pruned = pruned_obs,
        entities_pruned = ent_count,
        "Phase 5: decay & pruning"
    );

    Ok((pruned_obs, ent_count))
}

// ===========================================================================
// LLM Extraction
// ===========================================================================

/// Per-consolidation-run LLM state.
///
/// Carries the config plus a circuit breaker that trips after the first
/// call failure. With up to 500 pending episodes per run, a single bad
/// API key or dead endpoint would otherwise cause 500 × request-timeout
/// seconds of wasted work.
///
/// `provider` is parsed once from the config string at construction time;
/// an invalid provider string disables LLM extraction for the entire run
/// with a single warning rather than re-failing once per episode.
struct LlmRuntime<'a> {
    config: &'a ConsolidationConfig,
    provider: Option<crate::llm::Provider>,
    disabled: std::cell::Cell<bool>,
}

impl<'a> LlmRuntime<'a> {
    fn new(config: &'a ConsolidationConfig) -> Self {
        let provider = if config.llm_extraction {
            match config
                .llm_extraction_provider
                .parse::<crate::llm::Provider>()
            {
                Ok(p) => Some(p),
                Err(e) => {
                    warn!(
                        "LLM extraction disabled: invalid provider '{}' — {}",
                        config.llm_extraction_provider, e
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            provider,
            disabled: std::cell::Cell::new(!config.llm_extraction || provider.is_none()),
        }
    }

    /// Attempt LLM decomposition for `text`.
    ///
    /// Returns `Some((entities, observations))` if the LLM produced at least
    /// one stored observation, `None` if the caller should fall back to
    /// heuristic extraction. The first failure trips the circuit breaker.
    fn try_extract(
        &self,
        memory: &MemoryStore,
        text: &str,
    ) -> Option<crate::MemResult<(usize, usize)>> {
        if self.disabled.get() {
            return None;
        }
        let provider = self.provider?;
        match crate::llm::llm_decompose(text, &self.config.llm_extraction_model, provider) {
            Ok(extracted) if !extracted.is_empty() => {
                Some(store_llm_observations(memory, &extracted))
            }
            Ok(_) => {
                debug!("LLM extraction returned no observations, falling back to heuristics");
                None
            }
            Err(e) => {
                warn!("LLM extraction failed; disabling for remainder of this run: {e}");
                self.disabled.set(true);
                None
            }
        }
    }
}

/// Store observations from LLM decomposition into the memory graph.
///
/// Dedupes entity upserts within the batch so the returned `entities`
/// counter reflects distinct entity names (not per-observation upsert
/// calls) and so the same entity can't have its type flipped by
/// conflicting categories within a single run.
fn store_llm_observations(
    memory: &MemoryStore,
    extracted: &[crate::llm::ExtractedObservation],
) -> crate::MemResult<(usize, usize)> {
    let mut entity_ids: HashMap<String, String> = HashMap::new();
    let mut observations_created = 0usize;

    for obs in extracted {
        // `parse_observations` already filters unknown categories, so this
        // should always succeed; guard with a `continue` rather than panicking
        // if the upstream filter ever drifts.
        let Some(category) = crate::llm::Category::parse(&obs.category) else {
            continue;
        };
        let confidence = category.confidence();
        let source = category.source();

        let entity_id = if let Some(id) = entity_ids.get(&obs.entity) {
            id.clone()
        } else {
            let id = memory.save_entity(&obs.entity, category.entity_type(), confidence, source)?;
            entity_ids.insert(obs.entity.clone(), id.clone());
            id
        };

        memory.add_observation(&entity_id, &obs.observation, confidence, source)?;
        observations_created += 1;
    }

    info!(
        entities = entity_ids.len(),
        observations = observations_created,
        "LLM extraction stored"
    );

    Ok((entity_ids.len(), observations_created))
}

// ===========================================================================
// Extraction Helpers
// ===========================================================================

/// Extract a file path from a PostToolUse event's tool_input.
fn extract_file_path(parsed: &serde_json::Value) -> Option<String> {
    // tool_input might be a JSON string or an object
    let input = parsed.get("tool_input")?;

    // Try as object first
    if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
        return Some(path.to_string());
    }

    // Try parsing tool_input as a JSON string containing an object
    if let Some(input_str) = input.as_str() {
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(input_str) {
            if let Some(path) = inner.get("file_path").and_then(|v| v.as_str()) {
                return Some(path.to_string());
            }
        }
    }

    None
}

/// Extract the bash command from a PostToolUse event.
fn extract_bash_command(parsed: &serde_json::Value) -> Option<String> {
    let input = parsed.get("tool_input")?;

    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        return Some(cmd.to_string());
    }

    if let Some(input_str) = input.as_str() {
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(input_str) {
            if let Some(cmd) = inner.get("command").and_then(|v| v.as_str()) {
                return Some(cmd.to_string());
            }
        }
    }

    None
}

/// Shorten a file path for use as an entity name.
///
/// Strips common prefixes and keeps the last 2-3 path components.
fn shorten_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 3 {
        return path.to_string();
    }
    // Keep last 3 components
    parts[parts.len() - 3..].join("/")
}

/// Extract a concise error summary from Bash stdout, or None if no error.
///
/// Looks for common build/test failure patterns (Rust, Node, Python, etc.)
/// and returns a truncated summary suitable for an observation.
fn extract_bash_error(stdout: &str) -> Option<String> {
    // Only check if stdout is long enough to contain meaningful errors
    // and short enough that it's not just verbose output
    if stdout.len() < 20 || stdout.len() > 50_000 {
        return None;
    }

    let lower = stdout.to_lowercase();

    // Rust compilation errors
    if lower.contains("error[e") || lower.contains("error: could not compile") {
        // Find the first "error" line
        for line in stdout.lines() {
            if line.starts_with("error") && line.len() > 10 {
                let truncated = if line.len() > 200 { &line[..200] } else { line };
                return Some(truncated.to_string());
            }
        }
    }

    // Test failures
    if lower.contains("test result: failed") || lower.contains("failures:") {
        // Extract the "test result" line
        for line in stdout.lines() {
            if line.contains("test result: FAILED") || line.contains("test result: failed") {
                return Some(line.trim().to_string());
            }
        }
    }

    // General "Exit code 1" pattern (Claude Code sometimes adds this)
    if lower.contains("exit code 1") || lower.contains("exited with code") {
        // Find the most informative error line
        for line in stdout.lines().rev() {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && trimmed.len() > 10
                && (trimmed.contains("error")
                    || trimmed.contains("Error")
                    || trimmed.contains("FAILED"))
            {
                let truncated = if trimmed.len() > 200 {
                    &trimmed[..200]
                } else {
                    trimmed
                };
                return Some(truncated.to_string());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shorten_path() {
        assert_eq!(shorten_path("/a/b/c"), "/a/b/c");
        assert_eq!(
            shorten_path("/Users/rjow/sift/crates/sift-memory/src/lib.rs"),
            "sift-memory/src/lib.rs"
        );
        assert_eq!(shorten_path("src/main.rs"), "src/main.rs");
    }

    #[test]
    fn test_extract_file_path_from_object() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{"tool_input": {"file_path": "/foo/bar.rs"}, "tool_name": "Edit"}"#,
        )
        .unwrap();
        assert_eq!(extract_file_path(&json), Some("/foo/bar.rs".to_string()));
    }

    #[test]
    fn test_extract_file_path_from_string() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{"tool_input": "{\"file_path\": \"/foo/bar.rs\"}", "tool_name": "Edit"}"#,
        )
        .unwrap();
        assert_eq!(extract_file_path(&json), Some("/foo/bar.rs".to_string()));
    }

    #[test]
    fn test_extract_file_path_missing() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"tool_input": {"command": "ls"}}"#).unwrap();
        assert_eq!(extract_file_path(&json), None);
    }

    #[test]
    fn test_extract_bash_command() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"tool_input": {"command": "cargo test"}}"#).unwrap();
        assert_eq!(extract_bash_command(&json), Some("cargo test".to_string()));
    }

    #[test]
    fn test_full_consolidation_empty() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory().unwrap();
        let config = ConsolidationConfig::default();

        let report = run_consolidation(&store, &episodes, &config).unwrap();
        assert_eq!(report.episodes_processed, 0);
        assert_eq!(report.observations_created, 0);
    }

    #[test]
    fn test_episode_processing_post_compact_without_llm() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory().unwrap();

        let content =
            r#"{"compact_summary": "User is building a Rust memory system called Cortex."}"#;
        episodes
            .ingest("sess1", EventType::PostCompact, content)
            .unwrap();

        // LLM disabled (default) — PostCompact episodes are deferred,
        // not falsely counted as processed.
        let config = ConsolidationConfig::default();
        let report = run_consolidation(&store, &episodes, &config).unwrap();

        assert_eq!(report.episodes_processed, 0);
        assert_eq!(report.episodes_deferred, 1);
        assert_eq!(report.observations_created, 0);
    }

    #[test]
    fn test_episode_processing_stop_without_llm() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"last_assistant_message": "I've finished implementing the consolidation engine. All tests pass."}"#;
        episodes.ingest("sess1", EventType::Stop, content).unwrap();

        // LLM disabled (default) — Stop episodes are deferred,
        // not falsely counted as processed.
        let config = ConsolidationConfig::default();
        let report = run_consolidation(&store, &episodes, &config).unwrap();

        assert_eq!(report.episodes_processed, 0);
        assert_eq!(report.episodes_deferred, 1);
        assert_eq!(report.observations_created, 0);
    }

    #[test]
    fn test_episode_processing_tool_use() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"tool_name": "Edit", "tool_input": {"file_path": "/Users/rjow/sift/crates/sift-memory/src/lib.rs"}, "tool_response": "ok"}"#;
        episodes
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();

        let config = ConsolidationConfig::default();
        let report = run_consolidation(&store, &episodes, &config).unwrap();

        assert_eq!(report.episodes_processed, 1);
        assert_eq!(report.entities_created, 1);
        assert_eq!(report.observations_created, 1);

        // File path should be shortened
        let entity = store.get_entity("sift-memory/src/lib.rs").unwrap();
        assert!(entity.is_some());
    }

    #[test]
    fn test_promotion() {
        let store = MemoryStore::open_in_memory().unwrap();
        let config = ConsolidationConfig {
            promotion_min_access: 1,
            ..Default::default()
        };

        // Create entity + observation, then recall to increment access_count
        let entity_id = store
            .save_entity("test-entity", EntityType::Concept, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "test fact for promotion", 1.0, "test")
            .unwrap();

        // Recall to increment access_count
        let _ = store
            .recall("test fact promotion", 10, &RecallFilters::default())
            .unwrap();

        // Run promotion
        let promoted = phase_promotion(&store, &config).unwrap();
        assert_eq!(promoted, 1);

        // Verify tier changed
        let conn = store.db().lock().unwrap();
        let tier: String = conn
            .query_row(
                "SELECT memory_tier FROM observations WHERE entity_id = ?1 AND valid_until IS NULL",
                [&entity_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tier, "semantic");
    }

    #[test]
    fn test_dedup_across_consolidation() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "duplicate fact", 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "duplicate fact", 0.8, "test")
            .unwrap();

        let episodes = EpisodeStore::open_in_memory().unwrap();
        let config = ConsolidationConfig::default();

        let report = run_consolidation(&store, &episodes, &config).unwrap();
        assert_eq!(report.duplicates_merged, 1);
    }

    #[test]
    fn test_bash_error_extraction() {
        // Rust compilation error
        let stdout = "   Compiling sift v0.1.0\nerror[E0433]: failed to resolve: use of undeclared crate\n --> src/main.rs:1:5\n\nerror: could not compile `sift`";
        let error = extract_bash_error(stdout);
        assert!(error.is_some());
        assert!(error.unwrap().contains("error"));

        // Test failure
        let stdout = "running 5 tests\ntest foo ... ok\ntest bar ... FAILED\n\ntest result: FAILED. 1 passed; 1 failed";
        let error = extract_bash_error(stdout);
        assert!(error.is_some());
        assert!(error.unwrap().contains("FAILED"));

        // Clean output — no error
        let stdout = "   Compiling sift v0.1.0\n    Finished dev [unoptimized] target(s) in 2.5s";
        assert!(extract_bash_error(stdout).is_none());

        // Too short
        assert!(extract_bash_error("ok").is_none());
    }

    #[test]
    fn test_bash_error_creates_correction_observation() {
        let store = MemoryStore::open_in_memory().unwrap();
        let episodes = EpisodeStore::open_in_memory().unwrap();

        // Bash command with compilation error in stdout
        let content = r#"{"tool_name": "Bash", "tool_input": {"command": "cargo build"}, "tool_response": {"stdout": "   Compiling sift\nerror[E0433]: failed to resolve\nerror: could not compile", "stderr": "", "interrupted": false}}"#;
        episodes
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();

        let config = ConsolidationConfig::default();
        let report = run_consolidation(&store, &episodes, &config).unwrap();

        // At minimum, the correction observation should exist.
        // ("cargo build" is short enough that the regular "Ran:" observation
        // may be skipped, but the correction always fires.)
        assert!(report.observations_created >= 1);

        // Verify correction observation exists
        let conn = store.db().lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE source = 'cortex:correction'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count >= 1);
    }
}
