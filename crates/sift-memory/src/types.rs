//! Core types for the sift-memory knowledge graph.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/// A named concept the agent knows about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier (UUIDv7, time-ordered).
    pub id: String,
    /// Human-readable name (e.g., "Raymond", "sift project").
    pub name: String,
    /// Classification of the entity.
    pub entity_type: EntityType,
    /// Unix timestamp when the entity was first created.
    pub created_at: i64,
    /// Unix timestamp when the entity was last updated.
    pub updated_at: i64,
    /// Confidence in the entity's existence (0.0–1.0).
    pub confidence: f32,
    /// Origin identifier (e.g., session ID, conversation ID).
    pub source: String,
}

/// Classification for entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Person,
    Project,
    Concept,
    Tool,
    Preference,
    Fact,
    Event,
    Location,
    Organization,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Project => "project",
            Self::Concept => "concept",
            Self::Tool => "tool",
            Self::Preference => "preference",
            Self::Fact => "fact",
            Self::Event => "event",
            Self::Location => "location",
            Self::Organization => "organization",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "person" => Some(Self::Person),
            "project" => Some(Self::Project),
            "concept" => Some(Self::Concept),
            "tool" => Some(Self::Tool),
            "preference" => Some(Self::Preference),
            "fact" => Some(Self::Fact),
            "event" => Some(Self::Event),
            "location" => Some(Self::Location),
            "organization" => Some(Self::Organization),
            _ => None,
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// An atomic fact about an entity, with temporal validity.
///
/// Uses a bi-temporal model (inspired by Zep/Graphiti):
/// - `observed_at`: when the agent learned this fact
/// - `valid_from` / `valid_until`: when the fact was/is true in the real world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Unique identifier (UUIDv7).
    pub id: String,
    /// FK to the entity this fact is about.
    pub entity_id: String,
    /// The fact content (e.g., "prefers Rust over Python").
    pub content: String,
    /// When this fact was observed/learned (Unix timestamp).
    pub observed_at: i64,
    /// When this fact became true (None = same as observed_at).
    pub valid_from: Option<i64>,
    /// When this fact stopped being true (None = still valid).
    pub valid_until: Option<i64>,
    /// Confidence in the observation (0.0–1.0).
    pub confidence: f32,
    /// Origin identifier.
    pub source: String,
    /// ID of the observation this replaces (for contradiction resolution).
    pub supersedes: Option<String>,
}

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// A directed relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Unique identifier (UUIDv7).
    pub id: String,
    /// Source entity ID.
    pub from_entity: String,
    /// Target entity ID.
    pub to_entity: String,
    /// Relationship type in active voice (e.g., "maintains", "prefers").
    pub relation_type: String,
    /// Strength/confidence of the relationship (0.0–1.0).
    pub weight: f32,
    /// Unix timestamp when created.
    pub created_at: i64,
    /// Temporal validity start.
    pub valid_from: Option<i64>,
    /// Temporal validity end (None = still valid).
    pub valid_until: Option<i64>,
    /// Origin identifier.
    pub source: String,
}

// ---------------------------------------------------------------------------
// Memory Tier (Cortex)
// ---------------------------------------------------------------------------

/// Classification of memory by consolidation stage.
///
/// Mirrors human memory science:
/// - **Episodic**: raw, session-tagged facts (short-term)
/// - **Semantic**: consolidated, deduplicated facts (long-term)
/// - **Procedural**: learned skills and patterns (expertise)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// Raw observations from individual sessions. Subject to dedup and decay.
    Episodic,
    /// Consolidated facts promoted from episodic after repeated access or aging.
    Semantic,
    /// Learned procedural knowledge (skills). Never auto-deleted.
    Procedural,
}

impl MemoryTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "procedural" => Some(Self::Procedural),
            _ => None,
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Episode (Cortex hot path)
// ---------------------------------------------------------------------------

/// A raw event captured from Claude Code hooks before consolidation.
///
/// Episodes are the staging area: hook events are written here on the hot
/// path (<100ms), then the daemon's consolidation pipeline processes them
/// into entities and observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier (UUIDv7).
    pub id: String,
    /// Claude Code session identifier.
    pub session_id: String,
    /// Hook event type: "post_tool_use", "stop", "post_compact".
    pub event_type: String,
    /// Tool name for PostToolUse events (e.g., "Edit", "Bash").
    pub tool_name: Option<String>,
    /// Raw JSON payload from hook stdin.
    pub content: String,
    /// Unix timestamp when captured.
    pub timestamp: i64,
    /// Processing state: 0=pending, 1=processed, 2=skipped.
    pub processed: i32,
    /// Consolidation batch this episode was processed in.
    pub batch_id: Option<String>,
}

/// Processing state for episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EpisodeState {
    /// Awaiting consolidation.
    Pending = 0,
    /// Successfully processed into observations.
    Processed = 1,
    /// Skipped by attention filter or error.
    Skipped = 2,
}

impl EpisodeState {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

// ---------------------------------------------------------------------------
// Skill (Cortex procedural memory)
// ---------------------------------------------------------------------------

/// A learned procedural pattern extracted from repeated behaviors.
///
/// Skills represent expertise — they're created when the consolidation
/// engine detects the same patterns recurring across multiple sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier (UUIDv7).
    pub id: String,
    /// Short descriptive name (e.g., "rust-error-handling").
    pub name: String,
    /// Human-readable description of the skill.
    pub description: String,
    /// Prescriptive knowledge — how to apply the skill.
    pub pattern: String,
    /// When this skill should be activated (optional context trigger).
    pub trigger_context: Option<String>,
    /// Number of times this pattern has been observed.
    pub frequency: i64,
    /// Confidence score: `min(1.0, 0.3 + 0.1 * frequency + 0.2 * success_rate)`.
    pub confidence: f32,
    /// Unix timestamp of first observation.
    pub created_at: i64,
    /// Unix timestamp of most recent reinforcement.
    pub updated_at: i64,
    /// Episode IDs that contributed to this skill (JSON array).
    pub source_episodes: Option<String>,
}

// ---------------------------------------------------------------------------
// Recall
// ---------------------------------------------------------------------------

/// A memory recall result combining graph data with search scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// The observation content.
    pub observation: Observation,
    /// The entity this observation belongs to.
    pub entity_name: String,
    /// The entity type.
    pub entity_type: EntityType,
    /// Search relevance score (0.0–1.0).
    pub score: f32,
}

/// Report from a consolidation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Number of duplicate observations merged.
    pub duplicates_merged: usize,
    /// Number of contradictions detected.
    pub contradictions_found: usize,
    /// IDs of observations that were superseded.
    pub superseded_ids: Vec<String>,
    /// Pairs of (observation_id, observation_id) that contradict each other.
    pub contradiction_pairs: Vec<(String, String)>,
}

/// Information about a potential knowledge conflict detected when adding
/// or consolidating observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    /// ID of the existing observation that conflicts with the new one.
    pub observation_id: String,
    /// Content of the existing (potentially outdated) observation.
    pub existing_content: String,
    /// Conflict score: high embedding similarity × low text similarity.
    /// Higher = more likely a genuine contradiction (same topic, different answer).
    pub conflict_score: f32,
    /// When the existing observation was recorded.
    pub observed_at: i64,
}

/// Filters for recall queries.
#[derive(Debug, Clone, Default)]
pub struct RecallFilters {
    /// Only return observations for this entity type.
    pub entity_type: Option<EntityType>,
    /// Only return observations valid at this timestamp (None = now).
    pub valid_at: Option<i64>,
    /// Only return observations from this source.
    pub source: Option<String>,
    /// Minimum confidence threshold.
    pub min_confidence: Option<f32>,
    /// Only return observations in this memory tier.
    pub memory_tier: Option<MemoryTier>,
    /// Only return observations belonging to these entities (case-insensitive).
    /// When set, recall is scoped to the named entities, dramatically improving
    /// precision for entity-specific questions.
    pub entity_names: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics about the memory store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entities: u64,
    pub total_observations: u64,
    pub total_relations: u64,
    pub entity_type_counts: std::collections::HashMap<String, u64>,
    pub oldest_observation: Option<i64>,
    pub newest_observation: Option<i64>,
    /// Observation counts by memory tier.
    pub tier_counts: std::collections::HashMap<String, u64>,
    /// Number of episodes pending consolidation.
    pub pending_episodes: u64,
    /// Total episodes ingested.
    pub total_episodes: u64,
    /// Unix timestamp of last consolidation run.
    pub last_consolidation: Option<i64>,
}

// ---------------------------------------------------------------------------
// Consolidation Configuration
// ---------------------------------------------------------------------------

/// Tuning parameters for the Cortex consolidation pipeline.
///
/// Loaded from `[memory]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Lambda for the Ebbinghaus forgetting curve: `exp(-lambda * days)`.
    pub decay_rate: f64,
    /// Minimum days before an episodic observation is eligible for promotion.
    pub promotion_age_days: u32,
    /// Minimum access count for promotion to semantic tier.
    pub promotion_min_access: u32,
    /// Cosine similarity threshold for semantic deduplication (0.0–1.0).
    pub semantic_dedup_threshold: f32,
    /// Utility score below which observations are eligible for pruning.
    pub prune_utility_threshold: f32,
    /// Minimum age in days before an observation can be pruned.
    pub prune_min_age_days: u32,
    /// Minimum pattern frequency to extract a skill.
    pub skill_min_frequency: u32,
    /// Whether skill extraction is enabled.
    pub skill_extraction: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,
            promotion_age_days: 3,
            promotion_min_access: 2,
            semantic_dedup_threshold: 0.92,
            prune_utility_threshold: 0.05,
            prune_min_age_days: 30,
            skill_min_frequency: 3,
            skill_extraction: true,
        }
    }
}

impl From<&sift_core::config::MemoryConfig> for ConsolidationConfig {
    fn from(mc: &sift_core::config::MemoryConfig) -> Self {
        Self {
            decay_rate: mc.decay_rate,
            promotion_age_days: mc.promotion_age_days,
            promotion_min_access: mc.promotion_min_access,
            semantic_dedup_threshold: mc.semantic_dedup_threshold,
            prune_utility_threshold: mc.prune_utility_threshold,
            prune_min_age_days: mc.prune_min_age_days,
            skill_min_frequency: mc.skill_min_frequency,
            skill_extraction: mc.skill_extraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_type_roundtrip() {
        for ty in [
            EntityType::Person,
            EntityType::Project,
            EntityType::Concept,
            EntityType::Tool,
            EntityType::Preference,
            EntityType::Fact,
            EntityType::Event,
            EntityType::Location,
            EntityType::Organization,
        ] {
            let s = ty.as_str();
            let parsed = EntityType::parse(s).unwrap();
            assert_eq!(parsed, ty);
        }
    }

    #[test]
    fn entity_type_from_str_unknown() {
        assert!(EntityType::parse("unknown").is_none());
    }

    #[test]
    fn entity_type_display() {
        assert_eq!(EntityType::Person.to_string(), "person");
        assert_eq!(EntityType::Organization.to_string(), "organization");
    }

    #[test]
    fn entity_type_serde_roundtrip() {
        let ty = EntityType::Project;
        let json = serde_json::to_string(&ty).unwrap();
        assert_eq!(json, "\"project\"");
        let parsed: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ty);
    }

    #[test]
    fn memory_tier_roundtrip() {
        for tier in [
            MemoryTier::Episodic,
            MemoryTier::Semantic,
            MemoryTier::Procedural,
        ] {
            let s = tier.as_str();
            let parsed = MemoryTier::parse(s).unwrap();
            assert_eq!(parsed, tier);
        }
    }

    #[test]
    fn memory_tier_serde_roundtrip() {
        let tier = MemoryTier::Semantic;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"semantic\"");
        let parsed: MemoryTier = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, tier);
    }

    #[test]
    fn memory_tier_unknown_returns_none() {
        assert!(MemoryTier::parse("unknown").is_none());
        assert!(MemoryTier::parse("").is_none());
    }

    #[test]
    fn episode_state_values() {
        assert_eq!(EpisodeState::Pending.as_i32(), 0);
        assert_eq!(EpisodeState::Processed.as_i32(), 1);
        assert_eq!(EpisodeState::Skipped.as_i32(), 2);
    }

    #[test]
    fn consolidation_config_defaults_are_sane() {
        let config = ConsolidationConfig::default();
        assert!(config.decay_rate > 0.0);
        assert!(config.semantic_dedup_threshold > 0.5);
        assert!(config.prune_utility_threshold < 1.0);
        assert!(config.skill_extraction);
    }

    #[test]
    fn recall_filters_default_has_no_filters() {
        let filters = RecallFilters::default();
        assert!(filters.entity_type.is_none());
        assert!(filters.valid_at.is_none());
        assert!(filters.source.is_none());
        assert!(filters.min_confidence.is_none());
        assert!(filters.memory_tier.is_none());
    }
}
