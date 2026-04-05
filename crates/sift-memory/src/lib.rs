//! Persistent knowledge graph memory for AI agents.
//!
//! `sift-memory` provides a temporal knowledge graph built on SQLite + sift's
//! hybrid search engine. Agents store entities, observations (facts with
//! temporal validity), and relations — then recall them via semantic search.
//!
//! # Architecture
//!
//! - **Graph store**: SQLite for entities, observations, relations with indexes
//! - **Vector search**: Observation embeddings in sift's vector store (HNSW/flat)
//! - **Keyword search**: Observation text in sift's FTS5/Tantivy store
//! - **Hybrid recall**: RRF fusion of vector + keyword results, filtered by
//!   temporal validity and decay-scored for relevance

pub mod schema;
pub mod types;

use rusqlite::Connection;
use sift_store::{DefaultFullTextStore, HybridSearchEngine, SimpleVectorStore};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

pub use types::*;

/// Error type for memory operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Storage error: {0}")]
    Storage(#[from] sift_core::SiftError),
    #[error("Entity not found: {0}")]
    EntityNotFound(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type MemResult<T> = std::result::Result<T, MemoryError>;

/// Current Unix timestamp in seconds.
fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Generate a new UUIDv7 (time-ordered).
fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Persistent knowledge graph memory store.
///
/// Combines a SQLite graph store with sift's hybrid search engine for
/// semantic recall over observations.
pub struct MemoryStore {
    /// SQLite connection for the knowledge graph (entities, observations, relations).
    db: Mutex<Connection>,
    /// Hybrid search engine for semantic + keyword recall over observation text.
    search: HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    /// Optional embedder for vectorizing observations.
    #[cfg(feature = "embeddings")]
    embedder: Option<std::sync::Arc<dyn sift_core::Embedder>>,
    /// Path to the memory index directory (for persistence).
    index_dir: std::path::PathBuf,
}

impl MemoryStore {
    /// Open or create a memory store at the given directory.
    ///
    /// Creates `memory.db` for the graph store and uses sift's standard
    /// vector/FTS stores in the same directory.
    pub fn open(dir: &std::path::Path) -> MemResult<Self> {
        std::fs::create_dir_all(dir).map_err(|e| {
            MemoryError::InvalidInput(format!("Cannot create memory directory: {e}"))
        })?;

        let db_path = dir.join("memory.db");
        let conn = Connection::open(&db_path)?;
        schema::init_schema(&conn)?;

        // Load persisted vector store, or create fresh if none exists.
        #[cfg(feature = "hnsw")]
        let vector_store = SimpleVectorStore::load_or_create(dir)?;
        #[cfg(not(feature = "hnsw"))]
        let vector_store = SimpleVectorStore::load_or_migrate(dir)?;

        // Migrate legacy FTS filename: older builds used `memory_bm25.json`
        // due to a feature-flag mismatch (sift-memory was compiled without
        // the fts5 feature even though Fts5Store was the actual type).
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fts_path = {
            let preferred = dir.join("memory_fts.db");
            let legacy = dir.join("memory_bm25.json");
            if !preferred.exists() && legacy.exists() {
                tracing::info!("Migrating memory FTS: memory_bm25.json -> memory_fts.db");
                if std::fs::rename(&legacy, &preferred).is_ok() {
                    // Best-effort WAL/SHM migration; harmless if they don't exist.
                    let _ = std::fs::rename(
                        dir.join("memory_bm25.json-wal"),
                        dir.join("memory_fts.db-wal"),
                    );
                    let _ = std::fs::rename(
                        dir.join("memory_bm25.json-shm"),
                        dir.join("memory_fts.db-shm"),
                    );
                    preferred
                } else {
                    tracing::warn!("FTS migration failed, using legacy path");
                    legacy
                }
            } else {
                preferred
            }
        };

        #[cfg(feature = "fulltext")]
        let fulltext_store = DefaultFullTextStore::open(&dir.join("tantivy"))?;
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fulltext_store = DefaultFullTextStore::open(&fts_path)?;
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        let fulltext_store = DefaultFullTextStore::open(&dir.join("memory_bm25.json"))?;

        let search = HybridSearchEngine::new(vector_store, fulltext_store, 0.7);

        Ok(Self {
            db: Mutex::new(conn),
            search,
            #[cfg(feature = "embeddings")]
            embedder: None,
            index_dir: dir.to_path_buf(),
        })
    }

    /// Open a memory store using an in-memory SQLite database (for testing).
    #[cfg(test)]
    fn open_in_memory() -> MemResult<Self> {
        let conn = Connection::open_in_memory()?;
        schema::init_schema(&conn)?;

        let vector_store = SimpleVectorStore::new();

        #[cfg(feature = "fulltext")]
        let fulltext_store = DefaultFullTextStore::open_in_memory()?;
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fulltext_store = DefaultFullTextStore::open_in_memory()?;
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        let fulltext_store = {
            let tmp = tempfile::tempdir().unwrap();
            DefaultFullTextStore::open(&tmp.path().join("bm25.json"))?
        };

        let search = HybridSearchEngine::new(vector_store, fulltext_store, 0.7);

        Ok(Self {
            db: Mutex::new(conn),
            search,
            #[cfg(feature = "embeddings")]
            embedder: None,
            index_dir: std::path::PathBuf::new(),
        })
    }

    /// Set the embedder for vectorizing observations.
    ///
    /// On first call (or when the vector index is stale), rebuilds the search
    /// index so existing observations get real embeddings. Skips the rebuild
    /// if the vector store already has the right number of entries.
    #[cfg(feature = "embeddings")]
    pub fn with_embedder(mut self, embedder: std::sync::Arc<dyn sift_core::Embedder>) -> Self {
        self.embedder = Some(embedder);

        // Only rebuild if the vector index is out of sync with SQLite.
        let obs_count = {
            let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
            conn.query_row(
                "SELECT COUNT(*) FROM observations WHERE valid_until IS NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as u64
        };
        let vec_count = self.search.count().unwrap_or(0);

        if obs_count != vec_count {
            tracing::info!(
                "Memory vector index stale ({vec_count} vectors, {obs_count} observations), rebuilding"
            );
            if let Err(e) = self.rebuild_search_index() {
                warn!("Failed to rebuild search index with embedder: {e}");
            }
            if let Err(e) = self.save() {
                warn!("Failed to persist rebuilt search index: {e}");
            }
        } else {
            tracing::debug!(
                "Memory vector index up to date ({vec_count} vectors), skipping rebuild"
            );
        }

        self
    }

    /// Rebuild the search index from all active observations in SQLite.
    ///
    /// Clears both vector and fulltext stores, then re-indexes every active
    /// observation with the current embedder. Only effective when the
    /// `embeddings` feature is enabled and an embedder is attached.
    #[cfg(feature = "embeddings")]
    fn rebuild_search_index(&self) -> MemResult<()> {
        let start = std::time::Instant::now();

        // Fetch observations with their entity names for prefixed indexing.
        let observations: Vec<(String, String, String)> = {
            let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
            let mut stmt = conn.prepare(
                "SELECT o.id, e.name, o.content \
                 FROM observations o \
                 JOIN entities e ON o.entity_id = e.id \
                 WHERE o.valid_until IS NULL",
            )?;
            let result: Vec<_> = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .filter_map(Result::ok)
                .collect();
            result
        };

        // Clear existing entries so we don't get duplicates.
        for (id, _, _) in &observations {
            let uri = format!("memory://observation/{id}");
            let _ = self.search.delete_by_uri(&uri);
        }

        // Re-index with "entity_name: content" text and real embeddings.
        let chunks: Vec<sift_core::EmbeddedChunk> = observations
            .iter()
            .map(|(id, entity_name, content)| {
                let search_text = format!("{entity_name}: {content}");
                let vector = self.embed_observation(&search_text);
                sift_core::EmbeddedChunk {
                    chunk: sift_core::Chunk {
                        text: search_text,
                        source_uri: format!("memory://observation/{id}"),
                        chunk_index: 0,
                        content_type: sift_core::ContentType::Text,
                        file_type: "memory".to_string(),
                        title: None,
                        language: None,
                        byte_range: None,
                    },
                    vector,
                }
            })
            .collect();

        if !chunks.is_empty() {
            self.search.insert(&chunks)?;
        }

        tracing::info!(
            "Rebuilt memory search index: {} observations in {:.1?}",
            chunks.len(),
            start.elapsed()
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Entity operations
    // -----------------------------------------------------------------------

    /// Create or update an entity. Returns the entity ID.
    ///
    /// If an entity with the same name already exists, updates it and returns
    /// the existing ID.
    pub fn save_entity(
        &self,
        name: &str,
        entity_type: EntityType,
        confidence: f32,
        source: &str,
    ) -> MemResult<String> {
        if name.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Entity name must not be empty".to_string(),
            ));
        }

        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_secs();

        // Try to find existing entity by name
        let existing: Option<String> = conn
            .query_row("SELECT id FROM entities WHERE name = ?1", [name], |row| {
                row.get(0)
            })
            .ok();

        if let Some(id) = existing {
            conn.execute(
                "UPDATE entities SET entity_type = ?1, updated_at = ?2, confidence = ?3, source = ?4
                 WHERE id = ?5",
                rusqlite::params![entity_type.as_str(), now, confidence, source, id],
            )?;
            Ok(id)
        } else {
            let id = new_id();
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, created_at, updated_at, confidence, source)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![id, name, entity_type.as_str(), now, now, confidence, source],
            )?;
            Ok(id)
        }
    }

    /// Retrieve an entity by name.
    pub fn get_entity(&self, name: &str) -> MemResult<Option<Entity>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let result = conn.query_row(
            "SELECT id, name, entity_type, created_at, updated_at, confidence, source
             FROM entities WHERE name = ?1",
            [name],
            |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: EntityType::parse(&row.get::<_, String>(2)?)
                        .unwrap_or(EntityType::Concept),
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                    confidence: row.get(5)?,
                    source: row.get(6)?,
                })
            },
        );

        match result {
            Ok(entity) => Ok(Some(entity)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Retrieve an entity by ID.
    pub fn get_entity_by_id(&self, id: &str) -> MemResult<Option<Entity>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let result = conn.query_row(
            "SELECT id, name, entity_type, created_at, updated_at, confidence, source
             FROM entities WHERE id = ?1",
            [id],
            |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: EntityType::parse(&row.get::<_, String>(2)?)
                        .unwrap_or(EntityType::Concept),
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                    confidence: row.get(5)?,
                    source: row.get(6)?,
                })
            },
        );

        match result {
            Ok(entity) => Ok(Some(entity)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List entities with optional type filter and pagination.
    ///
    /// Returns entities sorted by `updated_at` descending (most recently
    /// touched first), each annotated with its active observation count.
    pub fn list_entities(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> MemResult<Vec<(Entity, usize)>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());

        let parse_row = |row: &rusqlite::Row<'_>| {
            let entity = Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: EntityType::parse(&row.get::<_, String>(2)?)
                    .unwrap_or(EntityType::Concept),
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                confidence: row.get(5)?,
                source: row.get(6)?,
            };
            let obs_count: i64 = row.get(7)?;
            Ok((entity, obs_count as usize))
        };

        // Single query with LEFT JOIN to count active observations.
        let base = "\
            SELECT e.id, e.name, e.entity_type, e.created_at, e.updated_at, \
                   e.confidence, e.source, \
                   COUNT(o.id) AS obs_count \
            FROM entities e \
            LEFT JOIN observations o ON o.entity_id = e.id AND o.valid_until IS NULL";

        let rows: Vec<(Entity, usize)> = if let Some(ref et) = entity_type {
            let sql = format!(
                "{base} WHERE e.entity_type = ?1 \
                 GROUP BY e.id ORDER BY e.updated_at DESC LIMIT ?2 OFFSET ?3"
            );
            let mut stmt = conn.prepare(&sql)?;
            let result: Vec<_> = stmt
                .query_map(
                    rusqlite::params![et.as_str(), limit as i64, offset as i64],
                    parse_row,
                )?
                .filter_map(Result::ok)
                .collect();
            result
        } else {
            let sql = format!("{base} GROUP BY e.id ORDER BY e.updated_at DESC LIMIT ?1 OFFSET ?2");
            let mut stmt = conn.prepare(&sql)?;
            let result: Vec<_> = stmt
                .query_map(rusqlite::params![limit as i64, offset as i64], parse_row)?
                .filter_map(Result::ok)
                .collect();
            result
        };

        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Observation operations
    // -----------------------------------------------------------------------

    /// Add an observation (fact) about an entity. Returns the observation ID.
    ///
    /// The observation text is also inserted into the hybrid search engine
    /// for semantic recall.
    pub fn add_observation(
        &self,
        entity_id: &str,
        content: &str,
        confidence: f32,
        source: &str,
    ) -> MemResult<String> {
        if content.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Observation content must not be empty".to_string(),
            ));
        }

        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_secs();
        let id = new_id();

        conn.execute(
            "INSERT INTO observations (id, entity_id, content, observed_at, valid_from, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![id, entity_id, content, now, now, confidence, source],
        )?;

        // Update entity's updated_at
        conn.execute(
            "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
            rusqlite::params![now, entity_id],
        )?;

        // Look up entity name to prefix the search text so keyword search
        // for entity names (e.g. "Raymond") finds their observations.
        let entity_name: String = conn
            .query_row(
                "SELECT name FROM entities WHERE id = ?1",
                [entity_id],
                |row| row.get(0),
            )
            .unwrap_or_default();

        // Drop the lock before inserting into search (which takes its own locks)
        drop(conn);

        // Index into hybrid search for recall. The indexed text is
        // "entity_name: observation" so keyword search matches on entity names.
        let search_text = format!("{entity_name}: {content}");
        let vector = self.embed_observation(&search_text);
        let uri = format!("memory://observation/{id}");

        let chunk = sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: search_text,
                source_uri: uri,
                chunk_index: 0,
                content_type: sift_core::ContentType::Text,
                file_type: "memory".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector,
        };

        self.search.insert(&[chunk])?;

        Ok(id)
    }

    /// Get all current (non-invalidated) observations for an entity.
    pub fn get_entity_observations(&self, entity_id: &str) -> MemResult<Vec<Observation>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn.prepare(
            "SELECT id, entity_id, content, observed_at, valid_from, valid_until,
                    confidence, source, supersedes
             FROM observations
             WHERE entity_id = ?1 AND valid_until IS NULL
             ORDER BY observed_at DESC",
        )?;

        let rows = stmt
            .query_map([entity_id], |row| {
                Ok(Observation {
                    id: row.get(0)?,
                    entity_id: row.get(1)?,
                    content: row.get(2)?,
                    observed_at: row.get(3)?,
                    valid_from: row.get(4)?,
                    valid_until: row.get(5)?,
                    confidence: row.get(6)?,
                    source: row.get(7)?,
                    supersedes: row.get(8)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(rows)
    }

    /// Invalidate an observation (soft delete by setting `valid_until`).
    pub fn invalidate_observation(&self, observation_id: &str) -> MemResult<bool> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_secs();

        let updated = conn.execute(
            "UPDATE observations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
            rusqlite::params![now, observation_id],
        )?;

        // Also remove from search index
        if updated > 0 {
            drop(conn);
            let uri = format!("memory://observation/{observation_id}");
            let _ = self.search.delete_by_uri(&uri);
        }

        Ok(updated > 0)
    }

    // -----------------------------------------------------------------------
    // Relation operations
    // -----------------------------------------------------------------------

    /// Add a directed relation between two entities. Returns the relation ID.
    pub fn add_relation(
        &self,
        from_entity: &str,
        to_entity: &str,
        relation_type: &str,
        weight: f32,
        source: &str,
    ) -> MemResult<String> {
        if relation_type.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Relation type must not be empty".to_string(),
            ));
        }

        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_secs();
        let id = new_id();

        conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, weight, created_at, valid_from, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![id, from_entity, to_entity, relation_type, weight, now, now, source],
        )?;

        Ok(id)
    }

    /// Get all current relations for an entity (both outgoing and incoming).
    pub fn get_entity_relations(&self, entity_id: &str) -> MemResult<Vec<Relation>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn.prepare(
            "SELECT id, from_entity, to_entity, relation_type, weight, created_at,
                    valid_from, valid_until, source
             FROM relations
             WHERE (from_entity = ?1 OR to_entity = ?1) AND valid_until IS NULL
             ORDER BY created_at DESC",
        )?;

        let rows = stmt
            .query_map([entity_id], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    from_entity: row.get(1)?,
                    to_entity: row.get(2)?,
                    relation_type: row.get(3)?,
                    weight: row.get(4)?,
                    created_at: row.get(5)?,
                    valid_from: row.get(6)?,
                    valid_until: row.get(7)?,
                    source: row.get(8)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Recall (semantic search over observations)
    // -----------------------------------------------------------------------

    /// Recall observations by semantic + keyword search.
    ///
    /// Uses sift's hybrid search engine (RRF fusion of vector + BM25) to find
    /// relevant observations, then enriches results with entity metadata and
    /// applies temporal filtering + decay scoring.
    pub fn recall(
        &self,
        query: &str,
        top_k: usize,
        filters: &RecallFilters,
    ) -> MemResult<Vec<RecallResult>> {
        let (vector, mode) = self.embed_for_search(query);

        // Fetch extra candidates for post-filtering
        let fetch_k = top_k * 3;
        let results = self.search.search(&vector, query, fetch_k, mode)?;

        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let valid_at = filters.valid_at.unwrap_or_else(now_secs);

        let mut memory_results: Vec<RecallResult> = Vec::new();

        for r in &results {
            // Extract observation ID from URI: memory://observation/{id}
            let obs_id = match r.uri.strip_prefix("memory://observation/") {
                Some(id) => id,
                None => continue, // Skip non-memory results
            };

            // Look up the observation and its entity
            let obs_entity = conn.query_row(
                "SELECT o.id, o.entity_id, o.content, o.observed_at, o.valid_from, o.valid_until,
                        o.confidence, o.source, o.supersedes,
                        e.name, e.entity_type
                 FROM observations o
                 JOIN entities e ON o.entity_id = e.id
                 WHERE o.id = ?1",
                [obs_id],
                |row| {
                    let obs = Observation {
                        id: row.get(0)?,
                        entity_id: row.get(1)?,
                        content: row.get(2)?,
                        observed_at: row.get(3)?,
                        valid_from: row.get(4)?,
                        valid_until: row.get(5)?,
                        confidence: row.get(6)?,
                        source: row.get(7)?,
                        supersedes: row.get(8)?,
                    };
                    let entity_name: String = row.get(9)?;
                    let entity_type_str: String = row.get(10)?;
                    Ok((obs, entity_name, entity_type_str))
                },
            );

            let (obs, entity_name, entity_type_str) = match obs_entity {
                Ok(v) => v,
                Err(_) => continue, // Observation may have been deleted
            };

            // Temporal filter: skip observations not valid at the requested time
            if let Some(from) = obs.valid_from {
                if valid_at < from {
                    continue;
                }
            }
            if let Some(until) = obs.valid_until {
                if valid_at >= until {
                    continue;
                }
            }

            // Entity type filter
            let entity_type = EntityType::parse(&entity_type_str).unwrap_or(EntityType::Concept);
            if let Some(ref filter_type) = filters.entity_type {
                if entity_type != *filter_type {
                    continue;
                }
            }

            // Source filter
            if let Some(ref filter_source) = filters.source {
                if obs.source != *filter_source {
                    continue;
                }
            }

            // Confidence filter
            if let Some(min_conf) = filters.min_confidence {
                if obs.confidence < min_conf {
                    continue;
                }
            }

            // Apply decay scoring
            let days_since = (valid_at - obs.observed_at).max(0) as f64 / 86400.0;
            let recency_factor = (-0.01 * days_since).exp() as f32; // lambda = 0.01
            let score = r.score * recency_factor * obs.confidence;

            memory_results.push(RecallResult {
                observation: obs,
                entity_name,
                entity_type,
                score,
            });
        }

        // Drop the lock before potential fallback (which re-acquires it).
        drop(conn);

        // Drop results below a minimum relevance threshold. HNSW always
        // returns approximate neighbors even for unrelated queries — this
        // filter prevents garbage results from masking the entity-name fallback.
        const MIN_SCORE: f32 = 0.3;
        memory_results.retain(|r| r.score >= MIN_SCORE);

        // Sort by final score descending
        memory_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        memory_results.truncate(top_k);

        // Fallback: if search found nothing relevant, try matching query
        // terms against entity names and return those entities' observations.
        if memory_results.is_empty() {
            memory_results = self.recall_by_entity_names(query, top_k, filters)?;
        }

        Ok(memory_results)
    }

    /// Fallback recall: match query terms against entity names.
    ///
    /// When FTS5 keyword search returns no results, this scans entity names
    /// for case-insensitive matches against the query, then returns those
    /// entities' active observations. This handles queries like "tell me about
    /// Raymond" even when no observation text contains "Raymond".
    fn recall_by_entity_names(
        &self,
        query: &str,
        top_k: usize,
        filters: &RecallFilters,
    ) -> MemResult<Vec<RecallResult>> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let valid_at = filters.valid_at.unwrap_or_else(now_secs);
        let query_lower = query.to_lowercase();

        // Require at least one word >= 3 chars to avoid matching on noise
        // like "is it ok" against entity names.
        let has_meaningful_term = query_lower.split_whitespace().any(|t| t.len() >= 3);
        if !has_meaningful_term {
            return Ok(vec![]);
        }

        // Find entities whose full name appears as a substring of the query.
        let mut stmt = conn.prepare("SELECT id, name, entity_type FROM entities")?;
        let matching_entities: Vec<(String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(Result::ok)
            .filter(|(_, name, _)| {
                let name_lower = name.to_lowercase();
                // Match if the full entity name appears as a substring of the query.
                query_lower.contains(&name_lower)
            })
            .collect();

        if matching_entities.is_empty() {
            return Ok(vec![]);
        }

        // Prepare the observation query once, reuse for each entity.
        let mut obs_stmt = conn.prepare(
            "SELECT id, entity_id, content, observed_at, valid_from, valid_until,
                    confidence, source, supersedes
             FROM observations
             WHERE entity_id = ?1 AND valid_until IS NULL
             ORDER BY observed_at DESC",
        )?;

        let mut results: Vec<RecallResult> = Vec::new();

        for (entity_id, entity_name, entity_type_str) in &matching_entities {
            let entity_type = EntityType::parse(entity_type_str).unwrap_or(EntityType::Concept);

            if let Some(ref filter_type) = filters.entity_type {
                if entity_type != *filter_type {
                    continue;
                }
            }

            let observations: Vec<Observation> = obs_stmt
                .query_map([entity_id], |row| {
                    Ok(Observation {
                        id: row.get(0)?,
                        entity_id: row.get(1)?,
                        content: row.get(2)?,
                        observed_at: row.get(3)?,
                        valid_from: row.get(4)?,
                        valid_until: row.get(5)?,
                        confidence: row.get(6)?,
                        source: row.get(7)?,
                        supersedes: row.get(8)?,
                    })
                })?
                .filter_map(Result::ok)
                .collect();

            for obs in observations {
                if let Some(from) = obs.valid_from {
                    if valid_at < from {
                        continue;
                    }
                }
                if let Some(ref filter_source) = filters.source {
                    if obs.source != *filter_source {
                        continue;
                    }
                }
                if let Some(min_conf) = filters.min_confidence {
                    if obs.confidence < min_conf {
                        continue;
                    }
                }

                let days_since = (valid_at - obs.observed_at).max(0) as f64 / 86400.0;
                let recency_factor = (-0.01 * days_since).exp() as f32;
                let score = recency_factor * obs.confidence;

                results.push(RecallResult {
                    observation: obs,
                    entity_name: entity_name.clone(),
                    entity_type,
                    score,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Consolidation
    // -----------------------------------------------------------------------

    /// Run deterministic consolidation: dedup similar observations and
    /// detect contradictions.
    ///
    /// - Observations with identical content on the same entity are merged
    ///   (newer supersedes older).
    /// - Observations with very similar content (>0.95 cosine similarity by
    ///   text overlap) on the same entity are flagged as duplicates.
    pub fn consolidate(&self) -> MemResult<ConsolidationReport> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let mut report = ConsolidationReport::default();
        let now = now_secs();

        // Get all entities
        let entity_ids: Vec<String> = conn
            .prepare("SELECT id FROM entities")?
            .query_map([], |row| row.get(0))?
            .filter_map(Result::ok)
            .collect();

        for entity_id in &entity_ids {
            // Get all active observations for this entity
            let mut stmt = conn.prepare(
                "SELECT id, content, observed_at, confidence
                 FROM observations
                 WHERE entity_id = ?1 AND valid_until IS NULL
                 ORDER BY observed_at ASC",
            )?;

            let observations: Vec<(String, String, i64, f32)> = stmt
                .query_map([entity_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })?
                .filter_map(Result::ok)
                .collect();

            // Pairwise exact-text dedup
            let mut superseded: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for i in 0..observations.len() {
                if superseded.contains(&observations[i].0) {
                    continue;
                }
                for j in (i + 1)..observations.len() {
                    if superseded.contains(&observations[j].0) {
                        continue;
                    }

                    let content_i = observations[i].1.trim().to_lowercase();
                    let content_j = observations[j].1.trim().to_lowercase();

                    if content_i == content_j {
                        // Exact duplicate — supersede the older one
                        let older_id = &observations[i].0;
                        let newer_id = &observations[j].0;

                        conn.execute(
                            "UPDATE observations SET valid_until = ?1, supersedes = NULL WHERE id = ?2",
                            rusqlite::params![now, older_id],
                        )?;
                        conn.execute(
                            "UPDATE observations SET supersedes = ?1 WHERE id = ?2",
                            rusqlite::params![older_id, newer_id],
                        )?;

                        superseded.insert(older_id.clone());
                        report.duplicates_merged += 1;
                        report.superseded_ids.push(older_id.clone());
                    } else {
                        // Check for potential contradiction: similar but different
                        let similarity = text_similarity(&content_i, &content_j);
                        if similarity > 0.85 {
                            report.contradictions_found += 1;
                            report
                                .contradiction_pairs
                                .push((observations[i].0.clone(), observations[j].0.clone()));
                        }
                    }
                }
            }
        }

        // Clean up search index for superseded observations
        for id in &report.superseded_ids {
            let uri = format!("memory://observation/{id}");
            if let Err(e) = self.search.delete_by_uri(&uri) {
                warn!("Failed to remove superseded observation from search: {e}");
            }
        }

        Ok(report)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Get memory store statistics.
    pub fn stats(&self) -> MemResult<MemoryStats> {
        let conn = self.db.lock().unwrap_or_else(|e| e.into_inner());

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

        let mut type_stmt =
            conn.prepare("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")?;
        let entity_type_counts: std::collections::HashMap<String, u64> = type_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(Result::ok)
            .collect();

        let oldest: Option<i64> = conn
            .query_row(
                "SELECT MIN(observed_at) FROM observations WHERE valid_until IS NULL",
                [],
                |row| row.get(0),
            )
            .ok();

        let newest: Option<i64> = conn
            .query_row(
                "SELECT MAX(observed_at) FROM observations WHERE valid_until IS NULL",
                [],
                |row| row.get(0),
            )
            .ok();

        Ok(MemoryStats {
            total_entities,
            total_observations,
            total_relations,
            entity_type_counts,
            oldest_observation: oldest,
            newest_observation: newest,
        })
    }

    /// Persist the search stores to disk.
    pub fn save(&self) -> MemResult<()> {
        use sift_store::{FullTextStore as _, VectorIndex as _};

        if !self.index_dir.as_os_str().is_empty() {
            self.search
                .vector_store
                .save(&self.index_dir.join("vectors.bin"))?;
            self.search.fulltext_store.flush()?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Embed observation text for indexing.
    #[allow(unused_variables, clippy::unused_self)]
    fn embed_observation(&self, text: &str) -> Vec<f32> {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                let prefixed = format!("search_document: {text}");
                match embedder.embed(&prefixed) {
                    Ok(vec) => return vec,
                    Err(e) => {
                        warn!("Embedding failed for observation: {e}");
                    }
                }
            }
        }
        vec![0.0f32; 768]
    }

    /// Embed a query for search.
    #[allow(unused_variables, clippy::unused_self)]
    fn embed_for_search(&self, query: &str) -> (Vec<f32>, sift_core::SearchMode) {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                let prefixed = format!("search_query: {query}");
                match embedder.embed(&prefixed) {
                    Ok(vec) => return (vec, sift_core::SearchMode::Hybrid),
                    Err(e) => {
                        warn!("Query embedding failed: {e}. Using keyword-only.");
                    }
                }
            }
        }
        (vec![0.0f32; 768], sift_core::SearchMode::KeywordOnly)
    }
}

// ---------------------------------------------------------------------------
// Text similarity (for consolidation without embeddings)
// ---------------------------------------------------------------------------

/// Simple word-overlap Jaccard similarity for detecting near-duplicates.
fn text_similarity(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_similarity_identical() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn text_similarity_disjoint() {
        assert!((text_similarity("hello world", "foo bar") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn text_similarity_partial() {
        let sim = text_similarity("hello world foo", "hello world bar");
        assert!(sim > 0.3 && sim < 0.8); // 2/4 = 0.5
    }

    #[test]
    fn text_similarity_empty() {
        assert!((text_similarity("", "") - 1.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // MemoryStore integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn save_and_get_entity() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        assert!(!id.is_empty());

        let entity = store.get_entity("Raymond").unwrap().unwrap();
        assert_eq!(entity.name, "Raymond");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert!((entity.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn save_entity_upsert() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id1 = store
            .save_entity("sift", EntityType::Project, 0.8, "test")
            .unwrap();
        let id2 = store
            .save_entity("sift", EntityType::Tool, 1.0, "test2")
            .unwrap();

        assert_eq!(id1, id2, "Same name should return same ID");

        let entity = store.get_entity("sift").unwrap().unwrap();
        assert_eq!(
            entity.entity_type,
            EntityType::Tool,
            "Type should be updated"
        );
    }

    #[test]
    fn save_entity_rejects_empty_name() {
        let store = MemoryStore::open_in_memory().unwrap();
        let err = store.save_entity("", EntityType::Concept, 1.0, "test");
        assert!(err.is_err());
    }

    #[test]
    fn get_entity_not_found() {
        let store = MemoryStore::open_in_memory().unwrap();
        let entity = store.get_entity("nonexistent").unwrap();
        assert!(entity.is_none());
    }

    #[test]
    fn add_and_get_observations() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();

        let obs_id = store
            .add_observation(&entity_id, "prefers Rust over Python", 0.9, "test")
            .unwrap();
        assert!(!obs_id.is_empty());

        store
            .add_observation(&entity_id, "works on sift project", 1.0, "test")
            .unwrap();

        let observations = store.get_entity_observations(&entity_id).unwrap();
        assert_eq!(observations.len(), 2);
    }

    #[test]
    fn add_observation_rejects_empty_content() {
        let store = MemoryStore::open_in_memory().unwrap();
        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        let err = store.add_observation(&entity_id, "", 1.0, "test");
        assert!(err.is_err());
    }

    #[test]
    fn invalidate_observation() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        let obs_id = store
            .add_observation(&entity_id, "likes Java", 0.5, "test")
            .unwrap();

        let invalidated = store.invalidate_observation(&obs_id).unwrap();
        assert!(invalidated);

        // Should no longer appear in active observations
        let observations = store.get_entity_observations(&entity_id).unwrap();
        assert!(observations.is_empty());

        // Double-invalidate should return false
        let again = store.invalidate_observation(&obs_id).unwrap();
        assert!(!again);
    }

    #[test]
    fn add_and_get_relations() {
        let store = MemoryStore::open_in_memory().unwrap();

        let person_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        let project_id = store
            .save_entity("sift", EntityType::Project, 1.0, "test")
            .unwrap();

        let rel_id = store
            .add_relation(&person_id, &project_id, "maintains", 1.0, "test")
            .unwrap();
        assert!(!rel_id.is_empty());

        let relations = store.get_entity_relations(&person_id).unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "maintains");

        // Also found via the target entity
        let relations_to = store.get_entity_relations(&project_id).unwrap();
        assert_eq!(relations_to.len(), 1);
    }

    #[test]
    fn add_relation_rejects_empty_type() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        let err = store.add_relation(&id, &id, "", 1.0, "test");
        assert!(err.is_err());
    }

    #[test]
    fn recall_by_keyword() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "prefers Rust over Python", 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "works on sift project", 1.0, "test")
            .unwrap();

        let results = store
            .recall("Rust programming", 10, &RecallFilters::default())
            .unwrap();

        // Should find the Rust observation via keyword match
        assert!(!results.is_empty());
        assert!(results[0].observation.content.contains("Rust"));
        assert_eq!(results[0].entity_name, "Raymond");
    }

    #[test]
    fn recall_with_entity_type_filter() {
        let store = MemoryStore::open_in_memory().unwrap();

        let person_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        let project_id = store
            .save_entity("sift", EntityType::Project, 1.0, "test")
            .unwrap();

        store
            .add_observation(&person_id, "likes coding in Rust", 1.0, "test")
            .unwrap();
        store
            .add_observation(&project_id, "Rust search engine", 1.0, "test")
            .unwrap();

        let results = store
            .recall(
                "Rust",
                10,
                &RecallFilters {
                    entity_type: Some(EntityType::Person),
                    ..Default::default()
                },
            )
            .unwrap();

        // Should only find person observations
        for r in &results {
            assert_eq!(r.entity_type, EntityType::Person);
        }
    }

    #[test]
    fn recall_excludes_invalidated() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        let obs_id = store
            .add_observation(&entity_id, "temporary fact about Golang", 1.0, "test")
            .unwrap();
        store.invalidate_observation(&obs_id).unwrap();

        let results = store
            .recall("Golang", 10, &RecallFilters::default())
            .unwrap();
        assert!(
            results.is_empty(),
            "Invalidated observations should not appear in recall"
        );
    }

    #[test]
    fn consolidate_deduplicates() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();

        // Add exact duplicates (case-insensitive)
        store
            .add_observation(&entity_id, "Prefers Rust", 0.8, "session1")
            .unwrap();
        store
            .add_observation(&entity_id, "prefers rust", 1.0, "session2")
            .unwrap();

        let report = store.consolidate().unwrap();
        assert_eq!(report.duplicates_merged, 1);
        assert_eq!(report.superseded_ids.len(), 1);

        // Only one active observation should remain
        let obs = store.get_entity_observations(&entity_id).unwrap();
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn stats_basic() {
        let store = MemoryStore::open_in_memory().unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_observations, 0);
        assert_eq!(stats.total_relations, 0);

        let id = store
            .save_entity("test", EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(&id, "some fact", 1.0, "test")
            .unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_entities, 1);
        assert_eq!(stats.total_observations, 1);
        assert_eq!(*stats.entity_type_counts.get("person").unwrap(), 1);
    }

    #[test]
    fn get_entity_by_id_works() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id = store
            .save_entity("test", EntityType::Concept, 0.7, "session")
            .unwrap();

        let entity = store.get_entity_by_id(&id).unwrap().unwrap();
        assert_eq!(entity.name, "test");
        assert_eq!(entity.entity_type, EntityType::Concept);
    }

    #[test]
    fn get_entity_by_id_not_found() {
        let store = MemoryStore::open_in_memory().unwrap();
        let entity = store.get_entity_by_id("nonexistent").unwrap();
        assert!(entity.is_none());
    }
}
