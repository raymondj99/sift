//! Core types for the sift-memory knowledge graph.

use serde::{Deserialize, Serialize};

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
}

/// Statistics about the memory store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entities: u64,
    pub total_observations: u64,
    pub total_relations: u64,
    pub entity_type_counts: std::collections::HashMap<String, u64>,
    pub oldest_observation: Option<i64>,
    pub newest_observation: Option<i64>,
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
}
