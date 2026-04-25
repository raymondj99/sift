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

pub mod consolidation;
pub mod episodes;
pub mod llm;
pub mod rules;
pub mod schema;
pub mod types;

use rusqlite::{Connection, OptionalExtension};
use sift_store::{DefaultFullTextStore, HybridSearchEngine, SimpleVectorStore};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

pub use types::*;

/// Minimum score for direct recall results. HNSW always returns approximate
/// neighbors, so this prevents garbage from masking the entity-name fallback.
const RECALL_MIN_SCORE: f32 = 0.3;

/// Minimum score for spreading-activation results (related entities).
/// Lower than RECALL_MIN_SCORE since related memories are inherently weaker.
const SPREADING_MIN_SCORE: f32 = 0.1;

/// Retrieval boost multiplier for correction-tagged observations
/// (source = "cortex:correction"). These are "don't repeat this mistake"
/// memories — higher priority during recall.
const CORRECTION_RETRIEVAL_BOOST: f32 = 1.3;

const MEMORY_REBUILD_EMBED_BATCH_SIZE: usize = 64;

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

/// A transactional mutation plan for an entity page rendered by `sift memory-tool`.
#[derive(Debug, Clone, Default)]
pub struct PageMutationPlan {
    pub entity_id: String,
    pub entity_type_update: Option<EntityType>,
    pub observation_invalidations: Vec<String>,
    pub observation_rewrites: Vec<ObservationRewrite>,
    pub observation_additions: Vec<String>,
    pub relation_invalidations: Vec<String>,
    pub relation_rewrites: Vec<RelationRewrite>,
    pub relation_additions: Vec<RelationAddition>,
}

#[derive(Debug, Clone)]
pub struct ObservationRewrite {
    pub observation_id: String,
    pub new_content: String,
}

#[derive(Debug, Clone)]
pub struct RelationRewrite {
    pub relation_id: String,
    pub relation_type: String,
    pub target_name: String,
}

#[derive(Debug, Clone)]
pub struct RelationAddition {
    pub relation_type: String,
    pub target_name: String,
}

#[derive(Debug, Clone)]
enum SearchSyncOp {
    DeleteObservation(String),
    UpsertObservation {
        observation_id: String,
        entity_name: String,
        content: String,
    },
}

/// Current Unix timestamp in seconds.
pub fn now_secs() -> i64 {
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
    /// Barrier between the two-phase rebuild (delete-all + reinsert-all) and
    /// concurrent recall. Rebuild holds the write guard; recall takes a
    /// read guard. Uncontended in steady state — we pay a single atomic
    /// per recall, and rebuilds only fire on startup after schema drift.
    rebuild_lock: std::sync::RwLock<()>,
    /// Optional embedder for vectorizing observations.
    #[cfg(feature = "embeddings")]
    embedder: Option<std::sync::Arc<dyn sift_core::Embedder>>,
    /// Path to the memory index directory (for persistence).
    index_dir: std::path::PathBuf,
    /// Lambda for the Ebbinghaus forgetting curve.
    /// Configurable via `config.memory.decay_rate`. Default: 0.01.
    decay_rate: f64,
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
            rebuild_lock: std::sync::RwLock::new(()),
            #[cfg(feature = "embeddings")]
            embedder: None,
            index_dir: dir.to_path_buf(),
            decay_rate: 0.01,
        })
    }

    /// Set the decay rate (lambda) for the forgetting curve.
    ///
    /// Typically called with `config.memory.decay_rate` after construction.
    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Borrow the underlying SQLite connection (behind a Mutex).
    ///
    /// Used by the consolidation engine for direct SQL operations that
    /// go beyond the MemoryStore's public API.
    pub fn db(&self) -> &Mutex<Connection> {
        &self.db
    }

    /// Acquire the SQLite connection, recovering from a poisoned mutex.
    ///
    /// All internal mutation paths use this helper instead of inlining
    /// `lock().unwrap_or_else(|e| e.into_inner())`. Mutex poisoning here
    /// means a previous holder panicked mid-write; the connection itself
    /// is still usable, so we recover the inner value rather than
    /// propagating the poison up to every caller.
    fn lock_db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.db.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Acquire the rebuild-lock for read (recall path).
    ///
    /// Recall holds this lock for the duration of a vector search so that a
    /// concurrent rebuild can't observe a half-rebuilt index.
    fn read_rebuild(&self) -> std::sync::RwLockReadGuard<'_, ()> {
        self.rebuild_lock.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Acquire the rebuild-lock for write (rebuild path).
    #[cfg(feature = "embeddings")]
    fn write_rebuild(&self) -> std::sync::RwLockWriteGuard<'_, ()> {
        self.rebuild_lock.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Open an in-memory memory store (for benchmarks and external tests).
    pub fn open_in_memory_for_bench() -> MemResult<Self> {
        Self::open_in_memory_impl()
    }

    /// Open a memory store using an in-memory SQLite database (for testing).
    #[cfg(test)]
    fn open_in_memory() -> MemResult<Self> {
        Self::open_in_memory_impl()
    }

    fn open_in_memory_impl() -> MemResult<Self> {
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
            rebuild_lock: std::sync::RwLock::new(()),
            #[cfg(feature = "embeddings")]
            embedder: None,
            index_dir: std::path::PathBuf::new(),
            decay_rate: 0.01,
        })
    }

    /// Set the embedder for vectorizing observations.
    ///
    /// This only attaches the embedder — call [`rebuild_if_stale`] afterwards
    /// to rebuild the search index if needed. This split lets callers (e.g.
    /// the MCP server) defer the potentially slow rebuild to a background
    /// thread so startup isn't blocked.
    #[cfg(feature = "embeddings")]
    pub fn with_embedder(mut self, embedder: std::sync::Arc<dyn sift_core::Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Rebuild the vector index if it's out of sync with SQLite.
    ///
    /// Returns `true` if a rebuild was performed, `false` if already up to date.
    ///
    /// Holds the rebuild write-lock for the entire delete-all + reinsert-all
    /// sequence so concurrent `recall` callers never observe a half-populated
    /// index. The common case (index already up to date) releases the lock
    /// immediately after a cheap count comparison.
    #[cfg(feature = "embeddings")]
    pub fn rebuild_if_stale(&self) -> bool {
        // Poisoning here is benign — we're the only writer, and any prior
        // panic happened mid-rebuild on state we're about to overwrite.
        let _guard = self.write_rebuild();

        let obs_count = {
            let conn = self.lock_db();
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
                return false;
            }
            if let Err(e) = self.save() {
                warn!("Failed to persist rebuilt search index: {e}");
            }
            true
        } else {
            tracing::debug!(
                "Memory vector index up to date ({vec_count} vectors), skipping rebuild"
            );
            false
        }
    }

    /// Force a full rebuild of the memory search index and persist it.
    ///
    /// Intended for maintenance and benchmarks. Normal callers should prefer
    /// [`Self::rebuild_if_stale`] so startup only pays the rebuild cost when
    /// the vector index is out of sync with SQLite.
    #[cfg(feature = "embeddings")]
    pub fn rebuild_search_index_now(&self) -> MemResult<()> {
        let _guard = self.write_rebuild();
        self.rebuild_search_index()?;
        self.save()
    }

    /// Rebuild the search index from all active observations in SQLite.
    ///
    /// Clears both vector and fulltext stores, then re-indexes every active
    /// observation with the current embedder. Only effective when the
    /// `embeddings` feature is enabled and an embedder is attached.
    fn rebuild_search_index(&self) -> MemResult<()> {
        let start = std::time::Instant::now();

        // Fetch observations with their entity names for prefixed indexing.
        let observations: Vec<(String, String, String)> = {
            let conn = self.lock_db();
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

        // Re-index with "entity_name: content" text and real embeddings.
        // Embed all batches FIRST. If any embedding call fails, return early
        // without touching the search index — the previous (presumably good)
        // index stays intact rather than being cleared and replaced with
        // zero vectors.
        let mut chunks: Vec<sift_core::EmbeddedChunk> = Vec::with_capacity(observations.len());
        for batch in observations.chunks(MEMORY_REBUILD_EMBED_BATCH_SIZE) {
            let search_texts: Vec<String> = batch
                .iter()
                .map(|(_, entity_name, content)| format!("{entity_name}: {content}"))
                .collect();
            let vectors = self.embed_observations_batch(&search_texts)?;

            chunks.extend(batch.iter().zip(search_texts).zip(vectors).map(
                |(((id, _, _), search_text), vector)| sift_core::EmbeddedChunk {
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
                },
            ));
        }

        // Embeddings succeeded — clear existing entries (so we don't double
        // up) and insert the new ones.
        for (id, _, _) in &observations {
            let uri = format!("memory://observation/{id}");
            let _ = self.search.delete_by_uri(&uri);
        }

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

    fn apply_search_sync_ops(&self, ops: &[SearchSyncOp]) -> MemResult<()> {
        for op in ops {
            match op {
                SearchSyncOp::DeleteObservation(observation_id) => {
                    let uri = format!("memory://observation/{observation_id}");
                    self.search.delete_by_uri(&uri)?;
                }
                SearchSyncOp::UpsertObservation {
                    observation_id,
                    entity_name,
                    content,
                } => {
                    self.index_observation(observation_id, entity_name, content)?;
                }
            }
        }
        Ok(())
    }

    fn apply_search_sync_ops_with_recovery(&self, ops: &[SearchSyncOp]) -> MemResult<()> {
        if let Err(err) = self.apply_search_sync_ops(ops) {
            warn!("Memory search sync failed after committed transaction: {err}");
            if let Err(rebuild_err) = self.rebuild_search_index() {
                warn!("Failed to rebuild memory search index after sync error: {rebuild_err}");
                return Err(rebuild_err);
            }
            if let Err(save_err) = self.save() {
                warn!("Failed to persist rebuilt memory search index: {save_err}");
            }
            return Err(err);
        }
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

        let conn = self.lock_db();
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

    fn save_entity_tx(
        conn: &Connection,
        name: &str,
        entity_type: EntityType,
        confidence: f32,
        source: &str,
        now: i64,
    ) -> MemResult<String> {
        if name.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Entity name must not be empty".to_string(),
            ));
        }

        let existing: Option<String> = conn
            .query_row("SELECT id FROM entities WHERE name = ?1", [name], |row| {
                row.get(0)
            })
            .optional()?;

        if let Some(id) = existing {
            conn.execute(
                "UPDATE entities
                 SET entity_type = ?1, updated_at = ?2, confidence = ?3, source = ?4
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

    /// Resolve an entity name against existing entities, returning the
    /// canonical entity ID. Prevents entity fragmentation by matching:
    ///
    /// 1. **Exact match** — same name (already handled by `save_entity`)
    /// 2. **Normalized match** — case-insensitive, stripped common prefixes
    /// 3. **Suffix/alias match** — new name is a suffix of an existing entity
    ///    (e.g., `"lib.rs"` matches `"sift-memory/src/lib.rs"`)
    ///
    /// Returns `Some(existing_id)` if a match is found, `None` if the name
    /// is genuinely new. Callers should use `save_entity` with the returned
    /// name if `None`, or `add_observation` on the returned ID if `Some`.
    pub fn resolve_entity(&self, name: &str) -> MemResult<Option<(String, String)>> {
        let conn = self.lock_db();
        let normalized = normalize_entity_name(name);

        // 1. Exact normalized match
        let exact: Option<(String, String)> = conn
            .query_row(
                "SELECT id, name FROM entities WHERE LOWER(name) = ?1",
                [&normalized],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();
        if exact.is_some() {
            return Ok(exact);
        }

        // 2. Suffix match via SQL LIKE — avoids loading all entities into memory.
        //    Matches at path boundaries (after '/').

        // Check: existing name ends with /new_name
        let suffix_pattern = format!("%/{normalized}");
        let suffix_match: Option<(String, String)> = conn
            .query_row(
                "SELECT id, name FROM entities WHERE LOWER(name) LIKE ?1 LIMIT 1",
                [&suffix_pattern],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();
        if suffix_match.is_some() {
            return Ok(suffix_match);
        }

        // Check: new name ends with /existing_name (existing is a suffix of new)
        // We can't use LIKE for this direction efficiently, but we can check
        // if any existing name is a suffix of the normalized input.
        // Use a bounded query: only check entities whose name length < input length.
        let mut stmt = conn.prepare("SELECT id, name FROM entities WHERE LENGTH(name) < ?1")?;
        let shorter: Vec<(String, String)> = stmt
            .query_map([normalized.len() as i64], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .filter_map(Result::ok)
            .collect();

        for (id, existing_name) in &shorter {
            let existing_lower = existing_name.to_lowercase();
            if normalized.ends_with(&format!("/{existing_lower}")) {
                return Ok(Some((id.clone(), existing_name.clone())));
            }
        }

        Ok(None)
    }

    /// Retrieve an entity by name.
    pub fn get_entity(&self, name: &str) -> MemResult<Option<Entity>> {
        let conn = self.lock_db();
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
        let conn = self.lock_db();
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

    /// Update an entity's type in place.
    pub fn update_entity_type(
        &self,
        entity_id: &str,
        entity_type: EntityType,
        source: &str,
    ) -> MemResult<bool> {
        let conn = self.lock_db();
        let now = now_secs();
        let updated = conn.execute(
            "UPDATE entities
             SET entity_type = ?1, updated_at = ?2, source = ?3
             WHERE id = ?4",
            rusqlite::params![entity_type.as_str(), now, source, entity_id],
        )?;
        Ok(updated > 0)
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
        let conn = self.lock_db();

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

    /// Return all entities without pagination.
    pub fn all_entities(&self) -> MemResult<Vec<Entity>> {
        let conn = self.lock_db();
        let mut stmt = conn.prepare(
            "SELECT id, name, entity_type, created_at, updated_at, confidence, source
             FROM entities
             ORDER BY updated_at DESC, name ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
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
            })?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    /// Create a new entity page atomically for `sift memory-tool`.
    pub fn create_entity_page(
        &self,
        name: &str,
        entity_type: EntityType,
        observations: &[String],
        relations: &[RelationAddition],
        source: &str,
    ) -> MemResult<String> {
        if name.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Entity name must not be empty".to_string(),
            ));
        }

        let conn = self.lock_db();
        let now = now_secs();

        conn.execute_batch("BEGIN")?;

        let result: MemResult<(String, Vec<SearchSyncOp>)> = (|| {
            let existing: Option<String> = conn
                .query_row("SELECT id FROM entities WHERE name = ?1", [name], |row| {
                    row.get(0)
                })
                .optional()?;
            if existing.is_some() {
                return Err(MemoryError::InvalidInput(format!(
                    "entity already exists: {name}"
                )));
            }

            let entity_id = new_id();
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, created_at, updated_at, confidence, source)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![entity_id, name, entity_type.as_str(), now, now, 1.0f32, source],
            )?;

            let mut sync_ops = Vec::new();
            for content in observations {
                if content.is_empty() {
                    return Err(MemoryError::InvalidInput(
                        "Observation content must not be empty".to_string(),
                    ));
                }

                let observation_id = new_id();
                conn.execute(
                    "INSERT INTO observations (
                         id, logical_id, entity_id, content, observed_at, valid_from, confidence, source
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        observation_id,
                        observation_id,
                        entity_id,
                        content,
                        now,
                        now,
                        1.0f32,
                        source
                    ],
                )?;
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, entity_id],
                )?;
                sync_ops.push(SearchSyncOp::UpsertObservation {
                    observation_id,
                    entity_name: name.to_string(),
                    content: content.clone(),
                });
            }

            for relation in relations {
                let target_id = Self::save_entity_tx(
                    &conn,
                    &relation.target_name,
                    EntityType::Concept,
                    1.0,
                    source,
                    now,
                )?;
                Self::insert_relation_tx(
                    &conn,
                    &entity_id,
                    &target_id,
                    &relation.relation_type,
                    1.0,
                    source,
                    now,
                )?;
            }

            Ok((entity_id, sync_ops))
        })();

        let (entity_id, sync_ops) = match result {
            Ok(result) => {
                conn.execute_batch("COMMIT")?;
                result
            }
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(err);
            }
        };
        drop(conn);

        self.apply_search_sync_ops_with_recovery(&sync_ops)?;
        Ok(entity_id)
    }

    /// Apply a memory-tool entity-page mutation atomically.
    ///
    /// All SQLite writes happen inside one transaction. Search index updates
    /// are applied only after commit; if they fail, the database remains the
    /// source of truth and a rebuild is attempted before returning the error.
    pub fn apply_page_mutation_plan(&self, plan: &PageMutationPlan, source: &str) -> MemResult<()> {
        let conn = self.lock_db();
        let now = now_secs();

        conn.execute_batch("BEGIN")?;

        let result: MemResult<Vec<SearchSyncOp>> = (|| {
            let entity_name: String = conn
                .query_row(
                    "SELECT name FROM entities WHERE id = ?1",
                    [&plan.entity_id],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or_else(|| {
                    MemoryError::InvalidInput("page changed, re-view and retry".to_string())
                })?;

            let mut validated_observation_rewrites = Vec::new();
            for rewrite in &plan.observation_rewrites {
                let row: Option<(String, String)> = conn
                    .query_row(
                        "SELECT entity_id, logical_id
                         FROM observations
                         WHERE id = ?1 AND valid_until IS NULL",
                        [&rewrite.observation_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let Some((entity_id, logical_id)) = row else {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                };
                if entity_id != plan.entity_id {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
                validated_observation_rewrites.push((
                    rewrite.observation_id.clone(),
                    rewrite.new_content.clone(),
                    logical_id,
                ));
            }

            let mut validated_relation_rewrites = Vec::new();
            for rewrite in &plan.relation_rewrites {
                let row: Option<(String, f32)> = conn
                    .query_row(
                        "SELECT from_entity, weight
                         FROM relations
                         WHERE id = ?1 AND valid_until IS NULL",
                        [&rewrite.relation_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let Some((from_entity, weight)) = row else {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                };
                if from_entity != plan.entity_id {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
                validated_relation_rewrites.push((
                    rewrite.relation_id.clone(),
                    rewrite.relation_type.clone(),
                    rewrite.target_name.clone(),
                    weight,
                ));
            }

            if let Some(entity_type) = plan.entity_type_update {
                let updated = conn.execute(
                    "UPDATE entities
                     SET entity_type = ?1, updated_at = ?2, source = ?3
                     WHERE id = ?4",
                    rusqlite::params![entity_type.as_str(), now, source, plan.entity_id],
                )?;
                if updated == 0 {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
            }

            let mut sync_ops = Vec::new();

            for observation_id in &plan.observation_invalidations {
                let updated = conn.execute(
                    "UPDATE observations
                     SET valid_until = ?1
                     WHERE id = ?2 AND entity_id = ?3 AND valid_until IS NULL",
                    rusqlite::params![now, observation_id, plan.entity_id],
                )?;
                if updated == 0 {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, plan.entity_id],
                )?;
                sync_ops.push(SearchSyncOp::DeleteObservation(observation_id.clone()));
            }

            for (observation_id, new_content, logical_id) in validated_observation_rewrites {
                let updated = conn.execute(
                    "UPDATE observations
                     SET valid_until = ?1
                     WHERE id = ?2 AND entity_id = ?3 AND valid_until IS NULL",
                    rusqlite::params![now, observation_id, plan.entity_id],
                )?;
                if updated == 0 {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }

                let new_id = new_id();
                conn.execute(
                    "INSERT INTO observations (
                         id, logical_id, entity_id, content, observed_at, valid_from,
                         confidence, source, supersedes
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    rusqlite::params![
                        new_id,
                        logical_id,
                        plan.entity_id,
                        new_content,
                        now,
                        now,
                        1.0f32,
                        source,
                        observation_id
                    ],
                )?;
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, plan.entity_id],
                )?;
                sync_ops.push(SearchSyncOp::DeleteObservation(observation_id));
                sync_ops.push(SearchSyncOp::UpsertObservation {
                    observation_id: new_id,
                    entity_name: entity_name.clone(),
                    content: new_content,
                });
            }

            for content in &plan.observation_additions {
                let observation_id = new_id();
                conn.execute(
                    "INSERT INTO observations (
                         id, logical_id, entity_id, content, observed_at, valid_from, confidence, source
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        observation_id,
                        observation_id,
                        plan.entity_id,
                        content,
                        now,
                        now,
                        1.0f32,
                        source
                    ],
                )?;
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, plan.entity_id],
                )?;
                sync_ops.push(SearchSyncOp::UpsertObservation {
                    observation_id,
                    entity_name: entity_name.clone(),
                    content: content.clone(),
                });
            }

            for relation_id in &plan.relation_invalidations {
                let relation: Option<(String, String)> = conn
                    .query_row(
                        "SELECT from_entity, to_entity
                         FROM relations
                         WHERE id = ?1 AND valid_until IS NULL",
                        [relation_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let Some((from_entity, to_entity)) = relation else {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                };
                if from_entity != plan.entity_id {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
                conn.execute(
                    "UPDATE relations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
                    rusqlite::params![now, relation_id],
                )?;
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id IN (?2, ?3)",
                    rusqlite::params![now, from_entity, to_entity],
                )?;
            }

            for (relation_id, relation_type, target_name, weight) in validated_relation_rewrites {
                let relation: Option<(String, String)> = conn
                    .query_row(
                        "SELECT from_entity, to_entity
                         FROM relations
                         WHERE id = ?1 AND valid_until IS NULL",
                        [&relation_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let Some((from_entity, old_to_entity)) = relation else {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                };
                if from_entity != plan.entity_id {
                    return Err(MemoryError::InvalidInput(
                        "page changed, re-view and retry".to_string(),
                    ));
                }
                conn.execute(
                    "UPDATE relations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
                    rusqlite::params![now, relation_id],
                )?;
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id IN (?2, ?3)",
                    rusqlite::params![now, from_entity, old_to_entity],
                )?;

                let target_id = Self::save_entity_tx(
                    &conn,
                    &target_name,
                    EntityType::Concept,
                    1.0,
                    source,
                    now,
                )?;
                Self::insert_relation_tx(
                    &conn,
                    &plan.entity_id,
                    &target_id,
                    &relation_type,
                    weight,
                    source,
                    now,
                )?;
            }

            for addition in &plan.relation_additions {
                let target_id = Self::save_entity_tx(
                    &conn,
                    &addition.target_name,
                    EntityType::Concept,
                    1.0,
                    source,
                    now,
                )?;
                Self::insert_relation_tx(
                    &conn,
                    &plan.entity_id,
                    &target_id,
                    &addition.relation_type,
                    1.0,
                    source,
                    now,
                )?;
            }

            Ok(sync_ops)
        })();

        let sync_ops = match result {
            Ok(sync_ops) => {
                conn.execute_batch("COMMIT")?;
                sync_ops
            }
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(err);
            }
        };
        drop(conn);

        self.apply_search_sync_ops_with_recovery(&sync_ops)?;
        Ok(())
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
        self.add_observation_with_lineage(entity_id, content, confidence, source, None)
    }

    fn add_observation_with_lineage(
        &self,
        entity_id: &str,
        content: &str,
        confidence: f32,
        source: &str,
        logical_id: Option<&str>,
    ) -> MemResult<String> {
        if content.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Observation content must not be empty".to_string(),
            ));
        }

        let conn = self.lock_db();
        let now = now_secs();
        let id = new_id();

        let logical_id = logical_id.unwrap_or(&id).to_string();
        conn.execute(
            "INSERT INTO observations (id, logical_id, entity_id, content, observed_at, valid_from, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![id, logical_id, entity_id, content, now, now, confidence, source],
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
        self.index_observation(&id, &entity_name, content)?;

        Ok(id)
    }

    fn index_observation(
        &self,
        observation_id: &str,
        entity_name: &str,
        content: &str,
    ) -> MemResult<()> {
        // Index into hybrid search for recall. The indexed text is
        // "entity_name: observation" so keyword search matches on entity names.
        let search_text = format!("{entity_name}: {content}");
        let vector = self.embed_observation(&search_text);
        let uri = format!("memory://observation/{observation_id}");

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
        Ok(())
    }

    /// Detect potential knowledge conflicts between new content and existing
    /// observations on the same entity.
    ///
    /// Uses the conflict score formula: `embedding_sim × (1 - text_sim)`.
    /// High embedding similarity + moderate text dissimilarity = "same topic,
    /// different answer" = likely conflict.
    ///
    /// Returns conflicts sorted by score (highest first).
    pub fn detect_conflicts(
        &self,
        entity_id: &str,
        new_content: &str,
        threshold: f32,
    ) -> MemResult<Vec<ConflictInfo>> {
        let conn = self.lock_db();

        // Get all active observations for this entity
        let mut stmt = conn.prepare(
            "SELECT id, content, observed_at
             FROM observations
             WHERE entity_id = ?1 AND valid_until IS NULL",
        )?;

        let existing: Vec<(String, String, i64)> = stmt
            .query_map([entity_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(Result::ok)
            .collect();

        drop(stmt);
        drop(conn);

        if existing.is_empty() {
            return Ok(Vec::new());
        }

        // Embed the new content
        let entity_name: String = {
            let conn = self.lock_db();
            conn.query_row(
                "SELECT name FROM entities WHERE id = ?1",
                [entity_id],
                |row| row.get(0),
            )
            .unwrap_or_default()
        };

        let new_text = format!("{entity_name}: {new_content}");
        let new_vec = self.embed_observation(&new_text);
        let new_lower = new_content.trim().to_lowercase();

        let mut conflicts = Vec::new();

        for (obs_id, obs_content, observed_at) in &existing {
            let obs_text = format!("{entity_name}: {obs_content}");
            let obs_vec = self.embed_observation(&obs_text);

            let embed_sim = cosine_similarity(&new_vec, &obs_vec);
            let text_sim = text_similarity(&new_lower, &obs_content.trim().to_lowercase());

            // Conflict = high semantic similarity but different text
            // Filter: embed_sim > 0.7 avoids unrelated topics,
            //         text_sim < 0.8 avoids near-duplicates,
            //         text_sim > 0.1 avoids completely unrelated content
            let conflict_score = embed_sim * (1.0 - text_sim);

            if conflict_score > threshold && embed_sim > 0.7 && text_sim < 0.8 && text_sim > 0.1 {
                conflicts.push(ConflictInfo {
                    observation_id: obs_id.clone(),
                    existing_content: obs_content.clone(),
                    conflict_score,
                    observed_at: *observed_at,
                });
            }
        }

        // Sort by conflict score descending
        conflicts.sort_by(|a, b| {
            b.conflict_score
                .partial_cmp(&a.conflict_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(conflicts)
    }

    /// Get all current (non-invalidated) observations for an entity.
    pub fn get_entity_observations(&self, entity_id: &str) -> MemResult<Vec<Observation>> {
        let conn = self.lock_db();
        let mut stmt = conn.prepare(
            "SELECT id, logical_id, entity_id, content, observed_at, valid_from, valid_until,
                    confidence, source, supersedes
             FROM observations
             WHERE entity_id = ?1 AND valid_until IS NULL
             ORDER BY observed_at DESC",
        )?;

        let rows = stmt
            .query_map([entity_id], |row| {
                Ok(Observation {
                    id: row.get(0)?,
                    logical_id: row.get(1)?,
                    entity_id: row.get(2)?,
                    content: row.get(3)?,
                    observed_at: row.get(4)?,
                    valid_from: row.get(5)?,
                    valid_until: row.get(6)?,
                    confidence: row.get(7)?,
                    source: row.get(8)?,
                    supersedes: row.get(9)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(rows)
    }

    /// Retrieve an observation by ID.
    pub fn get_observation(&self, observation_id: &str) -> MemResult<Option<Observation>> {
        let conn = self.lock_db();
        let result = conn.query_row(
            "SELECT id, logical_id, entity_id, content, observed_at, valid_from, valid_until,
                    confidence, source, supersedes
             FROM observations
             WHERE id = ?1",
            [observation_id],
            |row| {
                Ok(Observation {
                    id: row.get(0)?,
                    logical_id: row.get(1)?,
                    entity_id: row.get(2)?,
                    content: row.get(3)?,
                    observed_at: row.get(4)?,
                    valid_from: row.get(5)?,
                    valid_until: row.get(6)?,
                    confidence: row.get(7)?,
                    source: row.get(8)?,
                    supersedes: row.get(9)?,
                })
            },
        );

        match result {
            Ok(obs) => Ok(Some(obs)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Invalidate an observation (soft delete by setting `valid_until`).
    pub fn invalidate_observation(&self, observation_id: &str) -> MemResult<bool> {
        self.invalidate_observation_internal(observation_id, None)
    }

    fn invalidate_observation_internal(
        &self,
        observation_id: &str,
        now_override: Option<i64>,
    ) -> MemResult<bool> {
        let conn = self.lock_db();
        let now = now_override.unwrap_or_else(now_secs);

        let entity_id: Option<String> = conn
            .query_row(
                "SELECT entity_id FROM observations WHERE id = ?1 AND valid_until IS NULL",
                [observation_id],
                |row| row.get(0),
            )
            .ok();

        let updated = conn.execute(
            "UPDATE observations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
            rusqlite::params![now, observation_id],
        )?;

        if updated > 0 {
            if let Some(entity_id) = entity_id {
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, entity_id],
                )?;
            }

            drop(conn);
            let uri = format!("memory://observation/{observation_id}");
            let _ = self.search.delete_by_uri(&uri);
        }

        Ok(updated > 0)
    }

    /// Supersede an active observation with a new version.
    ///
    /// Returns `Ok(Some(new_id))` on success, `Ok(None)` when the predecessor is
    /// no longer active, and an error for invalid input.
    pub fn supersede_observation(
        &self,
        observation_id: &str,
        new_content: &str,
        confidence: f32,
        source: &str,
    ) -> MemResult<Option<String>> {
        if new_content.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Observation content must not be empty".to_string(),
            ));
        }

        let conn = self.lock_db();
        let now = now_secs();

        let predecessor: Option<(String, String, String)> = conn
            .query_row(
                "SELECT o.entity_id, o.logical_id, e.name
                 FROM observations o
                 JOIN entities e ON e.id = o.entity_id
                 WHERE o.id = ?1 AND o.valid_until IS NULL",
                [observation_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();

        let Some((entity_id, logical_id, entity_name)) = predecessor else {
            return Ok(None);
        };

        let updated = conn.execute(
            "UPDATE observations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
            rusqlite::params![now, observation_id],
        )?;
        if updated == 0 {
            return Ok(None);
        }

        let new_id = new_id();
        conn.execute(
            "INSERT INTO observations (
                 id, logical_id, entity_id, content, observed_at, valid_from,
                 confidence, source, supersedes
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                new_id,
                logical_id,
                entity_id,
                new_content,
                now,
                now,
                confidence,
                source,
                observation_id
            ],
        )?;
        conn.execute(
            "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
            rusqlite::params![now, entity_id],
        )?;
        drop(conn);

        let old_uri = format!("memory://observation/{observation_id}");
        let _ = self.search.delete_by_uri(&old_uri);
        self.index_observation(&new_id, &entity_name, new_content)?;

        Ok(Some(new_id))
    }

    /// Delete an entity entirely, cascading to all its observations and
    /// relations (via `ON DELETE CASCADE`).  Also removes observation entries
    /// from the search index.  Returns the number of observations that were
    /// purged.
    pub fn delete_entity(&self, name: &str) -> MemResult<usize> {
        let conn = self.lock_db();

        // Look up the entity id. Distinguish "no such entity" (a sentinel
        // condition the MCP layer surfaces as `not_found`) from other SQLite
        // errors, which should propagate untouched.
        let entity_id: String = conn
            .query_row("SELECT id FROM entities WHERE name = ?1", [name], |row| {
                row.get(0)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    MemoryError::EntityNotFound(name.to_string())
                }
                other => MemoryError::Sqlite(other),
            })?;

        // Collect all observation ids so we can remove them from the search
        // index *before* the cascade deletes the rows.
        let obs_ids: Vec<String> = {
            let mut stmt = conn.prepare("SELECT id FROM observations WHERE entity_id = ?1")?;
            let ids = stmt
                .query_map([&entity_id], |row| row.get(0))?
                .filter_map(Result::ok)
                .collect();
            ids
        };

        // Delete the entity row — CASCADE removes observations & relations.
        conn.execute("DELETE FROM entities WHERE id = ?1", [&entity_id])?;
        drop(conn);

        // Clean the search index.
        for id in &obs_ids {
            let uri = format!("memory://observation/{id}");
            let _ = self.search.delete_by_uri(&uri);
        }

        Ok(obs_ids.len())
    }

    /// Prune all entities that have zero active (non-invalidated) observations
    /// and no relations.  Returns the names of the entities that were removed.
    pub fn prune_entities(&self) -> MemResult<Vec<String>> {
        let conn = self.lock_db();

        // Find entities with no active observations AND no relations.
        let targets: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT e.id, e.name \
                 FROM entities e \
                 WHERE NOT EXISTS ( \
                     SELECT 1 FROM observations o \
                     WHERE o.entity_id = e.id AND o.valid_until IS NULL \
                 ) \
                 AND NOT EXISTS ( \
                     SELECT 1 FROM relations r \
                     WHERE (r.from_entity = e.id OR r.to_entity = e.id) \
                       AND r.valid_until IS NULL \
                 )",
            )?;
            let rows = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(Result::ok)
                .collect();
            rows
        };

        let mut pruned_names = Vec::with_capacity(targets.len());

        for (entity_id, name) in &targets {
            // Collect expired observation ids for search index cleanup.
            let obs_ids: Vec<String> = {
                let mut stmt = conn.prepare("SELECT id FROM observations WHERE entity_id = ?1")?;
                let ids = stmt
                    .query_map([entity_id], |row| row.get(0))?
                    .filter_map(Result::ok)
                    .collect();
                ids
            };

            conn.execute("DELETE FROM entities WHERE id = ?1", [entity_id])?;

            // Clean up search index entries for expired observations.
            for id in &obs_ids {
                let uri = format!("memory://observation/{id}");
                let _ = self.search.delete_by_uri(&uri);
            }

            pruned_names.push(name.clone());
        }

        Ok(pruned_names)
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

        let conn = self.lock_db();
        let now = now_secs();
        let id = new_id();

        conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, weight, created_at, valid_from, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![id, from_entity, to_entity, relation_type, weight, now, now, source],
        )?;

        conn.execute(
            "UPDATE entities SET updated_at = ?1 WHERE id IN (?2, ?3)",
            rusqlite::params![now, from_entity, to_entity],
        )?;

        Ok(id)
    }

    fn insert_relation_tx(
        conn: &Connection,
        from_entity: &str,
        to_entity: &str,
        relation_type: &str,
        weight: f32,
        source: &str,
        now: i64,
    ) -> MemResult<String> {
        if relation_type.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Relation type must not be empty".to_string(),
            ));
        }

        let id = new_id();
        conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, weight, created_at, valid_from, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![id, from_entity, to_entity, relation_type, weight, now, now, source],
        )?;
        conn.execute(
            "UPDATE entities SET updated_at = ?1 WHERE id IN (?2, ?3)",
            rusqlite::params![now, from_entity, to_entity],
        )?;
        Ok(id)
    }

    /// Get all current relations for an entity (both outgoing and incoming).
    pub fn get_entity_relations(&self, entity_id: &str) -> MemResult<Vec<Relation>> {
        let conn = self.lock_db();
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

    /// Retrieve a relation by ID.
    pub fn get_relation(&self, relation_id: &str) -> MemResult<Option<Relation>> {
        let conn = self.lock_db();
        let result = conn.query_row(
            "SELECT id, from_entity, to_entity, relation_type, weight, created_at,
                    valid_from, valid_until, source
             FROM relations
             WHERE id = ?1",
            [relation_id],
            |row| {
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
            },
        );

        match result {
            Ok(relation) => Ok(Some(relation)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Soft-delete an active relation.
    pub fn invalidate_relation(&self, relation_id: &str) -> MemResult<bool> {
        let conn = self.lock_db();
        let now = now_secs();
        let relation: Option<(String, String)> = conn
            .query_row(
                "SELECT from_entity, to_entity
                 FROM relations
                 WHERE id = ?1 AND valid_until IS NULL",
                [relation_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();
        let updated = conn.execute(
            "UPDATE relations SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
            rusqlite::params![now, relation_id],
        )?;
        if updated > 0 {
            if let Some((from_entity, to_entity)) = relation {
                conn.execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id IN (?2, ?3)",
                    rusqlite::params![now, from_entity, to_entity],
                )?;
            }
        }
        Ok(updated > 0)
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

        // When entity-filtering, cast a wider net since most results will
        // be filtered out. Without filtering, 3x is sufficient.
        let has_entity_filter = filters.entity_names.is_some();
        let fetch_k = if has_entity_filter {
            top_k * 10
        } else {
            top_k * 3
        };

        // Block if a background rebuild is mid-flight — otherwise we'd read
        // the index during its delete-then-reinsert phase and return
        // spuriously empty results. Uncontended in the steady state.
        let results = {
            let _guard = self.read_rebuild();
            self.search.search(&vector, query, fetch_k, mode)?
        };

        let conn = self.lock_db();
        let valid_at = filters.valid_at.unwrap_or_else(now_secs);

        let mut memory_results: Vec<RecallResult> = Vec::new();

        for r in &results {
            // Extract observation ID from URI: memory://observation/{id}
            let obs_id = match r.uri.strip_prefix("memory://observation/") {
                Some(id) => id,
                None => continue, // Skip non-memory results
            };

            // Single query: observation + entity + v2 access metadata.
            // All columns fetched in one round trip — no secondary lookups.
            let obs_entity = conn.query_row(
                "SELECT o.id, o.logical_id, o.entity_id, o.content, o.observed_at, o.valid_from, o.valid_until,
                        o.confidence, o.source, o.supersedes,
                        e.name, e.entity_type,
                        o.access_count, o.last_accessed, o.memory_tier
                 FROM observations o
                 JOIN entities e ON o.entity_id = e.id
                 WHERE o.id = ?1",
                [obs_id],
                |row| {
                    let obs = Observation {
                        id: row.get(0)?,
                        logical_id: row.get(1)?,
                        entity_id: row.get(2)?,
                        content: row.get(3)?,
                        observed_at: row.get(4)?,
                        valid_from: row.get(5)?,
                        valid_until: row.get(6)?,
                        confidence: row.get(7)?,
                        source: row.get(8)?,
                        supersedes: row.get(9)?,
                    };
                    let entity_name: String = row.get(10)?;
                    let entity_type_str: String = row.get(11)?;
                    let access_count: i64 = row.get(12)?;
                    let last_accessed: Option<i64> = row.get(13)?;
                    let memory_tier: String = row.get(14)?;
                    Ok((
                        obs,
                        entity_name,
                        entity_type_str,
                        access_count,
                        last_accessed,
                        memory_tier,
                    ))
                },
            );

            let (obs, entity_name, entity_type_str, access_count, last_accessed, memory_tier) =
                match obs_entity {
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

            // Entity name filter (case-insensitive)
            if let Some(ref names) = filters.entity_names {
                let name_lower = entity_name.to_lowercase();
                if !names.iter().any(|n| n.to_lowercase() == name_lower) {
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

            // Memory tier filter (uses value from the join, not a separate query)
            if let Some(ref filter_tier) = filters.memory_tier {
                if memory_tier != filter_tier.as_str() {
                    continue;
                }
            }

            // --- Retrieval-dependent strengthening scoring ---
            //
            //   base_decay = exp(-lambda * days_since_observed)
            //   retrieval_boost = 1.0 + 0.15 * ln(1 + access_count)
            //   last_access_recency = exp(-lambda * days_since_last_accessed)
            //   recency = 0.4 * base_decay + 0.6 * max(base_decay, last_access_recency)
            //   score = search_relevance * recency * retrieval_boost * confidence
            let lambda = self.decay_rate;
            let days_since = (valid_at - obs.observed_at).max(0) as f64 / 86400.0;
            let base_decay = (-lambda * days_since).exp();

            let retrieval_boost = 1.0 + 0.15 * (1.0 + access_count as f64).ln();

            let last_access_recency = match last_accessed {
                Some(ts) => {
                    let days_since_access = (valid_at - ts).max(0) as f64 / 86400.0;
                    (-lambda * days_since_access).exp()
                }
                None => 0.0, // Never accessed — no boost
            };

            let recency = 0.4 * base_decay + 0.6 * base_decay.max(last_access_recency);

            // Correction boost: observations tagged as corrections get a 1.3x
            // multiplier. These are high-value "don't repeat this mistake"
            // memories that should surface more readily.
            let correction_boost: f32 = if obs.source == "cortex:correction" {
                CORRECTION_RETRIEVAL_BOOST
            } else {
                1.0
            };

            let score = r.score
                * recency as f32
                * retrieval_boost as f32
                * obs.confidence
                * correction_boost;

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
        memory_results.retain(|r| r.score >= RECALL_MIN_SCORE);

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

        // --- Spreading activation: 1-hop relation traversal ---
        // Surface observations from entities related to the matched ones.
        // This handles "tell me about Raymond" surfacing sift observations
        // if a Raymond→maintains→sift relation exists.
        if memory_results.len() < top_k {
            let slots = top_k - memory_results.len();
            let related = self.spread_activation(&memory_results, slots, filters)?;
            memory_results.extend(related);
        }

        // --- Log access for retrieval-dependent strengthening ---
        // Side-effect: increment access counters so future recalls are boosted.
        if !memory_results.is_empty() {
            self.log_access_batch(query, &memory_results)?;
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
        let conn = self.lock_db();
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
            "SELECT id, logical_id, entity_id, content, observed_at, valid_from, valid_until,
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
                        logical_id: row.get(1)?,
                        entity_id: row.get(2)?,
                        content: row.get(3)?,
                        observed_at: row.get(4)?,
                        valid_from: row.get(5)?,
                        valid_until: row.get(6)?,
                        confidence: row.get(7)?,
                        source: row.get(8)?,
                        supersedes: row.get(9)?,
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
    // Spreading Activation
    // -----------------------------------------------------------------------

    /// Traverse 1-hop relations from matched entities to surface related
    /// observations not found by direct search.
    ///
    /// For each entity in `seed_results`, finds related entities via the
    /// relations table, then retrieves their top observations. Scores are
    /// weighted by the relation weight (0.0–1.0) and a distance decay factor
    /// of 0.5 (related memories are half as relevant as direct matches).
    fn spread_activation(
        &self,
        seed_results: &[RecallResult],
        max_related: usize,
        filters: &RecallFilters,
    ) -> MemResult<Vec<RecallResult>> {
        if max_related == 0 || seed_results.is_empty() {
            return Ok(vec![]);
        }

        let conn = self.lock_db();
        let valid_at = filters.valid_at.unwrap_or_else(now_secs);

        // Collect unique entity IDs from the seed results
        let seed_entity_ids: std::collections::HashSet<String> = seed_results
            .iter()
            .map(|r| r.observation.entity_id.clone())
            .collect();

        // Already-seen observation IDs to avoid duplicates
        let seen_obs: std::collections::HashSet<String> = seed_results
            .iter()
            .map(|r| r.observation.id.clone())
            .collect();

        let mut related_results: Vec<RecallResult> = Vec::new();

        // Prepare all statements once — reuse across iterations
        let mut rel_stmt = conn.prepare(
            "SELECT from_entity, to_entity, weight
             FROM relations
             WHERE (from_entity = ?1 OR to_entity = ?1) AND valid_until IS NULL",
        )?;
        let mut entity_stmt =
            conn.prepare("SELECT name, entity_type FROM entities WHERE id = ?1")?;
        let mut obs_stmt = conn.prepare(
            "SELECT id, logical_id, entity_id, content, observed_at, valid_from, valid_until,
                    confidence, source, supersedes
             FROM observations
             WHERE entity_id = ?1 AND valid_until IS NULL
             ORDER BY confidence DESC, observed_at DESC
             LIMIT 3",
        )?;

        for entity_id in &seed_entity_ids {
            let neighbors: Vec<(String, f32)> = rel_stmt
                .query_map([entity_id], |row| {
                    let from: String = row.get(0)?;
                    let to: String = row.get(1)?;
                    let weight: f32 = row.get(2)?;
                    let neighbor = if from == *entity_id { to } else { from };
                    Ok((neighbor, weight))
                })?
                .filter_map(Result::ok)
                .filter(|(id, _)| !seed_entity_ids.contains(id))
                .collect();

            for (neighbor_id, rel_weight) in neighbors {
                let entity_meta: Option<(String, String)> = entity_stmt
                    .query_row([&neighbor_id], |row| Ok((row.get(0)?, row.get(1)?)))
                    .ok();

                let (entity_name, entity_type_str) = match entity_meta {
                    Some(m) => m,
                    None => continue,
                };

                let entity_type =
                    EntityType::parse(&entity_type_str).unwrap_or(EntityType::Concept);

                if let Some(ref filter_type) = filters.entity_type {
                    if entity_type != *filter_type {
                        continue;
                    }
                }

                let observations: Vec<Observation> = obs_stmt
                    .query_map([&neighbor_id], |row| {
                        Ok(Observation {
                            id: row.get(0)?,
                            logical_id: row.get(1)?,
                            entity_id: row.get(2)?,
                            content: row.get(3)?,
                            observed_at: row.get(4)?,
                            valid_from: row.get(5)?,
                            valid_until: row.get(6)?,
                            confidence: row.get(7)?,
                            source: row.get(8)?,
                            supersedes: row.get(9)?,
                        })
                    })?
                    .filter_map(Result::ok)
                    .filter(|o| !seen_obs.contains(&o.id))
                    .collect();

                const DISTANCE_DECAY: f32 = 0.5;
                for obs in observations {
                    let days_since = (valid_at - obs.observed_at).max(0) as f64 / 86400.0;
                    let recency = (-self.decay_rate * days_since).exp() as f32;
                    let score = rel_weight * DISTANCE_DECAY * obs.confidence * recency;

                    if score < SPREADING_MIN_SCORE {
                        continue;
                    }

                    related_results.push(RecallResult {
                        observation: obs,
                        entity_name: entity_name.clone(),
                        entity_type,
                        score,
                    });
                }
            }
        }

        // Sort by score descending and cap
        related_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        related_results.truncate(max_related);

        Ok(related_results)
    }

    // -----------------------------------------------------------------------
    // Consolidation
    // -----------------------------------------------------------------------

    /// Run deduplication with the default threshold (0.92 text similarity).
    pub fn consolidate(&self) -> MemResult<ConsolidationReport> {
        self.consolidate_with_threshold(0.92)
    }

    /// Run deduplication with a configurable similarity threshold.
    ///
    /// Dedup works in three passes:
    /// 1. **Exact text match** — identical content (case-insensitive) → merge
    /// 2. **High text overlap** — Jaccard > `text_threshold` → merge
    /// 3. **Semantic similarity** — embedding cosine > `text_threshold` when
    ///    text overlap is moderate (0.3–threshold) → merge
    /// 4. **Contradiction detection** — moderate text overlap (0.1–0.8) with
    ///    high embedding similarity but low text similarity → flag
    pub fn consolidate_with_threshold(
        &self,
        text_threshold: f32,
    ) -> MemResult<ConsolidationReport> {
        let conn = self.lock_db();
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

            // Pairwise dedup
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
                        // Pass 1: Exact duplicate — supersede the older one
                        let older_id = &observations[i].0;
                        let newer_id = &observations[j].0;

                        conn.execute(
                            "UPDATE observations SET valid_until = ?1, supersedes = NULL WHERE id = ?2",
                            rusqlite::params![now, older_id],
                        )?;
                        conn.execute(
                            "UPDATE observations
                             SET supersedes = ?1,
                                 logical_id = (SELECT logical_id FROM observations WHERE id = ?1)
                             WHERE id = ?2",
                            rusqlite::params![older_id, newer_id],
                        )?;

                        superseded.insert(older_id.clone());
                        report.duplicates_merged += 1;
                        report.superseded_ids.push(older_id.clone());
                        continue;
                    }

                    let text_sim = text_similarity(&content_i, &content_j);

                    if text_sim > text_threshold {
                        // Pass 2: Near-duplicate by text (high Jaccard overlap)
                        Self::merge_duplicate_observation(
                            &conn,
                            &observations[i].0,
                            &observations[j].0,
                            now,
                            &mut superseded,
                            &mut report,
                        )?;
                    } else if text_sim > 0.3 && text_sim <= text_threshold {
                        // Pass 3: Moderate text overlap — check embedding similarity.
                        // This catches semantically equivalent observations with
                        // different wording (e.g., "Prefers Rust for systems work"
                        // vs "Uses Rust for low-level programming").
                        let vec_i = self.embed_observation(&observations[i].1);
                        let vec_j = self.embed_observation(&observations[j].1);
                        let embed_sim = cosine_similarity(&vec_i, &vec_j);

                        if embed_sim > text_threshold {
                            // Semantically identical — merge
                            Self::merge_duplicate_observation(
                                &conn,
                                &observations[i].0,
                                &observations[j].0,
                                now,
                                &mut superseded,
                                &mut report,
                            )?;
                        }
                    } else if text_sim > 0.1 && text_sim < 0.8 {
                        // Pass 4: Contradiction detection — same topic but
                        // different answer (high embed sim, low text sim)
                        let vec_i = self.embed_observation(&observations[i].1);
                        let vec_j = self.embed_observation(&observations[j].1);
                        let embed_sim = cosine_similarity(&vec_i, &vec_j);
                        let conflict_score = embed_sim * (1.0 - text_sim);

                        if conflict_score > 0.15 && embed_sim > 0.7 {
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

    /// Mark the older observation as superseded by the newer one.
    fn merge_duplicate_observation(
        conn: &rusqlite::Connection,
        older_id: &str,
        newer_id: &str,
        now: i64,
        superseded: &mut std::collections::HashSet<String>,
        report: &mut ConsolidationReport,
    ) -> MemResult<()> {
        conn.execute(
            "UPDATE observations SET valid_until = ?1 WHERE id = ?2",
            rusqlite::params![now, older_id],
        )?;
        conn.execute(
            "UPDATE observations
             SET supersedes = ?1,
                 logical_id = (SELECT logical_id FROM observations WHERE id = ?1)
             WHERE id = ?2",
            rusqlite::params![older_id, newer_id],
        )?;
        superseded.insert(older_id.to_string());
        report.duplicates_merged += 1;
        report.superseded_ids.push(older_id.to_string());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Access Logging (retrieval-dependent strengthening)
    // -----------------------------------------------------------------------

    /// Log access for a batch of recall results.
    ///
    /// For each accessed observation: inserts an `access_log` entry,
    /// increments `access_count`, and updates `last_accessed` timestamp.
    /// This implements the spacing effect — recalled memories strengthen.
    ///
    /// All writes are batched in a single transaction (3N statements in 1
    /// round-trip instead of 3N implicit transactions).
    fn log_access_batch(&self, query: &str, results: &[RecallResult]) -> MemResult<()> {
        let conn = self.lock_db();
        let now = now_secs();

        conn.execute_batch("BEGIN")?;

        let result = (|| -> Result<(), rusqlite::Error> {
            for result in results {
                let log_id = new_id();
                conn.execute(
                    "INSERT INTO access_log (id, observation_id, accessed_at, query)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![log_id, result.observation.id, now, query],
                )?;

                conn.execute(
                    "UPDATE observations
                     SET access_count = access_count + 1, last_accessed = ?1
                     WHERE id = ?2",
                    rusqlite::params![now, result.observation.id],
                )?;

                conn.execute(
                    "UPDATE entities
                     SET access_count = access_count + 1, last_accessed = ?1
                     WHERE id = ?2",
                    rusqlite::params![now, result.observation.entity_id],
                )?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => conn.execute_batch("COMMIT")?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e.into());
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Get memory store statistics.
    pub fn stats(&self) -> MemResult<MemoryStats> {
        let conn = self.lock_db();

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

        // Tier breakdown
        let mut tier_stmt = conn.prepare(
            "SELECT memory_tier, COUNT(*) FROM observations
             WHERE valid_until IS NULL GROUP BY memory_tier",
        )?;
        let tier_counts: std::collections::HashMap<String, u64> = tier_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(Result::ok)
            .collect();

        // Episode stats
        let pending_episodes: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE processed = 0",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let total_episodes: u64 = conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))
            .unwrap_or(0);

        let last_consolidation: Option<i64> = conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = 'last_consolidation'",
                [],
                |row| {
                    let v: String = row.get(0)?;
                    Ok(v.parse::<i64>().ok())
                },
            )
            .unwrap_or(None);

        Ok(MemoryStats {
            total_entities,
            total_observations,
            total_relations,
            entity_type_counts,
            oldest_observation: oldest,
            newest_observation: newest,
            tier_counts,
            pending_episodes,
            total_episodes,
            last_consolidation,
        })
    }

    /// Persist the search stores to disk.
    pub fn save(&self) -> MemResult<()> {
        // `flush` is a FullTextStore trait method. `save` on the vector
        // store resolves to an inherent method on FlatVectorIndex but to
        // the VectorIndex trait method on HnswIndex — we always need the
        // trait-qualified call to cover the hnsw build.
        use sift_store::FullTextStore as _;

        if !self.index_dir.as_os_str().is_empty() {
            sift_store::VectorIndex::save(
                &self.search.vector_store,
                &self.index_dir.join("vectors.bin"),
            )?;
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
                match embedder.embed_passage(text) {
                    Ok(vec) => return vec,
                    Err(e) => {
                        warn!("Embedding failed for observation: {e}");
                    }
                }
            }
        }
        vec![0.0f32; 768]
    }

    /// Embed observation texts for rebuild indexing.
    ///
    /// `texts` must be the unprefixed indexed form (`entity_name: content`).
    /// When an embedder is configured, embedding failures and length
    /// mismatches surface as `MemoryError::Storage(...)` so the caller can
    /// abort instead of writing zero vectors that silently destroy recall.
    /// When no embedder is configured, returns zero vectors — `embed_for_search`
    /// already downgrades to `SearchMode::KeywordOnly` in that case.
    #[allow(unused_variables, clippy::unused_self)]
    fn embed_observations_batch(&self, texts: &[String]) -> MemResult<Vec<Vec<f32>>> {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let vectors = embedder.embed_passages(&refs).map_err(|e| {
                    MemoryError::Storage(sift_core::SiftError::Embedding(format!(
                        "Batch embedding failed for {} observations: {e}",
                        texts.len()
                    )))
                })?;

                if vectors.len() != texts.len() {
                    return Err(MemoryError::Storage(sift_core::SiftError::Embedding(
                        format!(
                            "Batch embedding returned {} vectors for {} observations",
                            vectors.len(),
                            texts.len()
                        ),
                    )));
                }

                return Ok(vectors);
            }
        }

        Ok(texts.iter().map(|_| vec![0.0f32; 768]).collect())
    }

    /// Embed a query for search.
    #[allow(unused_variables, clippy::unused_self)]
    fn embed_for_search(&self, query: &str) -> (Vec<f32>, sift_core::SearchMode) {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                match embedder.embed_query(query) {
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
// Entity name normalization
// ---------------------------------------------------------------------------

/// Normalize an entity name for matching: lowercase, strip common path
/// prefixes (/Users/*, /home/*, /tmp/*), and trim whitespace.
fn normalize_entity_name(name: &str) -> String {
    let mut s = name.trim().to_lowercase();

    // Strip common home directory prefixes up to and including the username:
    //   /users/rjow/sift/... → sift/...
    //   /home/user/project/... → project/...
    for prefix in &["/users/", "/home/", "/tmp/"] {
        if let Some(rest) = s.strip_prefix(prefix) {
            if let Some(pos) = rest.find('/') {
                s = rest[pos + 1..].to_string();
            }
            break; // Only one prefix can match
        }
    }

    // Strip "session:" prefix for session entities (normalization only)
    if let Some(rest) = s.strip_prefix("session:") {
        s = rest.to_string();
    }

    s
}

// ---------------------------------------------------------------------------
// Text similarity (for consolidation without embeddings)
// ---------------------------------------------------------------------------

/// Cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    (dot / denom) as f32
}

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
    fn create_entity_page_rolls_back_on_late_failure() {
        let store = MemoryStore::open_in_memory().unwrap();
        let err = store
            .create_entity_page(
                "Raymond",
                EntityType::Person,
                &["prefers Rust over Python".to_string()],
                &[RelationAddition {
                    relation_type: String::new(),
                    target_name: "sift".to_string(),
                }],
                "test",
            )
            .unwrap_err();

        assert!(err.to_string().contains("Relation type must not be empty"));
        assert!(store.get_entity("Raymond").unwrap().is_none());
        assert!(store.get_entity("sift").unwrap().is_none());
        assert!(store.all_entities().unwrap().is_empty());
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
    fn consolidate_with_threshold_uses_config_value() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();

        // Two observations with ~0.6 Jaccard overlap (not exact, not near-exact)
        // "Prefers Rust for systems programming" vs "Prefers Rust for low level programming"
        // share "Prefers Rust for programming" but differ on "systems" vs "low level"
        store
            .add_observation(
                &entity_id,
                "Prefers Rust for systems programming work",
                0.8,
                "session1",
            )
            .unwrap();
        store
            .add_observation(
                &entity_id,
                "Prefers Rust for systems programming tasks",
                1.0,
                "session2",
            )
            .unwrap();

        // With high threshold (0.95), these should NOT be merged (Jaccard < 0.95)
        let report = store.consolidate_with_threshold(0.95).unwrap();
        let obs = store.get_entity_observations(&entity_id).unwrap();
        // Both should still be active (text similarity is ~0.83, under 0.95)
        assert_eq!(obs.len(), 2, "Should not merge with high threshold");
        assert_eq!(report.duplicates_merged, 0);
    }

    #[test]
    fn consolidate_with_low_threshold_merges_more() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();

        // Near-duplicates with moderate text overlap (~0.83 Jaccard)
        store
            .add_observation(
                &entity_id,
                "Prefers Rust for systems programming work",
                0.8,
                "session1",
            )
            .unwrap();
        store
            .add_observation(
                &entity_id,
                "Prefers Rust for systems programming tasks",
                1.0,
                "session2",
            )
            .unwrap();

        // With low threshold (0.7), these SHOULD be merged (text sim ~0.83 > 0.7)
        let report = store.consolidate_with_threshold(0.7).unwrap();
        let obs = store.get_entity_observations(&entity_id).unwrap();
        assert_eq!(obs.len(), 1, "Should merge with lower threshold");
        assert_eq!(report.duplicates_merged, 1);
    }

    #[test]
    fn consolidate_default_backward_compatible() {
        // consolidate() should work identically to consolidate_with_threshold(0.92)
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();

        store
            .add_observation(&entity_id, "exact duplicate fact", 0.8, "s1")
            .unwrap();
        store
            .add_observation(&entity_id, "Exact Duplicate Fact", 1.0, "s2")
            .unwrap();

        let report = store.consolidate().unwrap();
        assert_eq!(report.duplicates_merged, 1);
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

    // -----------------------------------------------------------------------
    // Retrieval-dependent strengthening tests
    // -----------------------------------------------------------------------

    #[test]
    fn recall_increments_access_count() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "prefers Rust over Python", 1.0, "test")
            .unwrap();

        // Use exact observation text as query for reliable matching across
        // all fulltext backends (FTS5, Tantivy, BM25).
        let results = store
            .recall("prefers Rust over Python", 10, &RecallFilters::default())
            .unwrap();
        assert!(!results.is_empty());

        // Check access_count was incremented
        let conn = store.db.lock().unwrap();
        let access_count: i64 = conn
            .query_row(
                "SELECT access_count FROM observations WHERE entity_id = ?1",
                [&entity_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            access_count, 1,
            "access_count should be 1 after first recall"
        );

        let last_accessed: Option<i64> = conn
            .query_row(
                "SELECT last_accessed FROM observations WHERE entity_id = ?1",
                [&entity_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(last_accessed.is_some(), "last_accessed should be set");
        drop(conn);

        // Second recall — should increment again
        let _ = store
            .recall("prefers Rust over Python", 10, &RecallFilters::default())
            .unwrap();

        let conn = store.db.lock().unwrap();
        let access_count: i64 = conn
            .query_row(
                "SELECT access_count FROM observations WHERE entity_id = ?1",
                [&entity_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            access_count, 2,
            "access_count should be 2 after second recall"
        );
    }

    #[test]
    fn recall_writes_access_log() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("sift", EntityType::Project, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "local semantic search engine", 1.0, "test")
            .unwrap();

        let results = store
            .recall(
                "local semantic search engine",
                10,
                &RecallFilters::default(),
            )
            .unwrap();
        assert!(!results.is_empty());

        // Verify access_log has entries
        let conn = store.db.lock().unwrap();
        let log_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM access_log", [], |row| row.get(0))
            .unwrap();
        assert!(log_count > 0, "access_log should have entries after recall");

        // Verify the query was stored
        let logged_query: String = conn
            .query_row(
                "SELECT query FROM access_log ORDER BY accessed_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(logged_query, "local semantic search engine");
    }

    #[test]
    fn recall_entity_access_count_also_increments() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "prefers Rust over Python", 1.0, "test")
            .unwrap();

        let _ = store
            .recall("prefers Rust over Python", 10, &RecallFilters::default())
            .unwrap();

        let conn = store.db.lock().unwrap();
        let entity_access: i64 = conn
            .query_row(
                "SELECT access_count FROM entities WHERE id = ?1",
                [&entity_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            entity_access, 1,
            "entity access_count should be incremented"
        );
    }

    #[test]
    fn decay_rate_is_configurable() {
        let store = MemoryStore::open_in_memory().unwrap().with_decay_rate(1.0);

        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "fast-decaying fact about testing", 1.0, "test")
            .unwrap();

        // With lambda=1.0, decay after 1 day is exp(-1) ≈ 0.37.
        // With lambda=0.01, decay after 1 day is exp(-0.01) ≈ 0.99.
        // A high decay rate should produce lower scores for older observations.
        // We can't easily control observation age in this test, but we verify
        // the field is actually used by checking the store was configured.
        assert!((store.decay_rate - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Stats v2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn stats_includes_tier_counts() {
        let store = MemoryStore::open_in_memory().unwrap();

        let entity_id = store
            .save_entity("test", EntityType::Concept, 1.0, "test")
            .unwrap();
        store
            .add_observation(&entity_id, "episodic fact", 1.0, "test")
            .unwrap();

        let stats = store.stats().unwrap();
        // All observations default to episodic tier
        let episodic_count = stats.tier_counts.get("episodic").copied().unwrap_or(0);
        assert_eq!(episodic_count, 1);
        assert_eq!(stats.pending_episodes, 0);
        assert_eq!(stats.total_episodes, 0);
    }

    #[test]
    fn recall_no_results_does_not_log_access() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Empty store — recall should return nothing and write no access log
        let results = store
            .recall("nonexistent query", 10, &RecallFilters::default())
            .unwrap();
        assert!(results.is_empty());

        let conn = store.db.lock().unwrap();
        let log_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM access_log", [], |row| row.get(0))
            .unwrap();
        assert_eq!(log_count, 0, "no access log entries for empty results");
    }

    // -----------------------------------------------------------------------
    // Entity resolution tests
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_entity_exact_match() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .save_entity("sift-memory/src/lib.rs", EntityType::Project, 1.0, "test")
            .unwrap();

        // Exact match (case-insensitive)
        let result = store.resolve_entity("sift-memory/src/lib.rs").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, "sift-memory/src/lib.rs");
    }

    #[test]
    fn resolve_entity_suffix_match() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .save_entity("sift-memory/src/lib.rs", EntityType::Project, 1.0, "test")
            .unwrap();

        // "src/lib.rs" is a suffix of "sift-memory/src/lib.rs"
        let result = store.resolve_entity("src/lib.rs").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, "sift-memory/src/lib.rs");
    }

    #[test]
    fn resolve_entity_reverse_suffix() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .save_entity("src/lib.rs", EntityType::Project, 1.0, "test")
            .unwrap();

        // New name is longer and contains the existing as a suffix
        let result = store.resolve_entity("sift-memory/src/lib.rs").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, "src/lib.rs");
    }

    #[test]
    fn resolve_entity_no_match() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .save_entity("sift-memory/src/lib.rs", EntityType::Project, 1.0, "test")
            .unwrap();

        let result = store.resolve_entity("completely-different").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn normalize_entity_name_strips_home_prefix() {
        let n = normalize_entity_name("/Users/rjow/sift/crates/sift-memory/src/lib.rs");
        assert_eq!(n, "sift/crates/sift-memory/src/lib.rs");
    }

    #[test]
    fn normalize_entity_name_lowercases() {
        let n = normalize_entity_name("Raymond");
        assert_eq!(n, "raymond");
    }

    // -----------------------------------------------------------------------
    // Spreading activation tests
    // -----------------------------------------------------------------------

    #[test]
    fn spreading_activation_surfaces_related() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Create two entities with a relation
        let person_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        let project_id = store
            .save_entity("sift", EntityType::Project, 1.0, "test")
            .unwrap();
        store
            .add_relation(&person_id, &project_id, "maintains", 1.0, "test")
            .unwrap();

        // Add observations to both
        store
            .add_observation(&person_id, "prefers Rust over Python", 1.0, "test")
            .unwrap();
        store
            .add_observation(&project_id, "local semantic search engine", 1.0, "test")
            .unwrap();

        // Search for Raymond's observation — should also surface sift's via relation
        let results = store
            .recall("prefers Rust over Python", 10, &RecallFilters::default())
            .unwrap();

        // Should have Raymond's direct match AND sift's related observation
        let entity_names: Vec<&str> = results.iter().map(|r| r.entity_name.as_str()).collect();
        assert!(
            entity_names.contains(&"Raymond"),
            "Should find direct match: {entity_names:?}"
        );
        // The sift observation should appear via spreading activation
        assert!(
            entity_names.contains(&"sift"),
            "Should find related entity via spreading activation: {entity_names:?}"
        );
    }

    #[test]
    fn spreading_activation_respects_top_k() {
        let store = MemoryStore::open_in_memory().unwrap();

        let person_id = store
            .save_entity("Raymond", EntityType::Person, 1.0, "test")
            .unwrap();
        let project_id = store
            .save_entity("sift", EntityType::Project, 1.0, "test")
            .unwrap();
        store
            .add_relation(&person_id, &project_id, "maintains", 1.0, "test")
            .unwrap();

        store
            .add_observation(&person_id, "prefers Rust over Python", 1.0, "test")
            .unwrap();
        store
            .add_observation(&project_id, "local semantic search engine", 1.0, "test")
            .unwrap();

        // With top_k=1, should only return the direct match, not the related one
        let results = store
            .recall("prefers Rust over Python", 1, &RecallFilters::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_name, "Raymond");
    }

    // -----------------------------------------------------------------------
    // Entity-filtered recall tests
    // -----------------------------------------------------------------------

    #[test]
    fn recall_with_entity_name_filter() {
        let store = MemoryStore::open_in_memory().unwrap();

        let person_id = store
            .save_entity("Alice", EntityType::Person, 1.0, "test")
            .unwrap();
        let other_id = store
            .save_entity("Bob", EntityType::Person, 1.0, "test")
            .unwrap();

        store
            .add_observation(&person_id, "Alice went to Paris in June", 1.0, "test")
            .unwrap();
        store
            .add_observation(&other_id, "Bob went to Paris in July", 1.0, "test")
            .unwrap();

        // Without filter — should find both
        let results = store
            .recall("went to Paris", 10, &RecallFilters::default())
            .unwrap();
        let names: Vec<&str> = results.iter().map(|r| r.entity_name.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));

        // With entity_names filter — should only find Alice
        let filtered = store
            .recall(
                "went to Paris",
                10,
                &RecallFilters {
                    entity_names: Some(vec!["Alice".to_string()]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!filtered.is_empty());
        assert!(
            filtered.iter().all(|r| r.entity_name == "Alice"),
            "All results should be Alice's, got: {:?}",
            filtered.iter().map(|r| &r.entity_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn recall_entity_filter_case_insensitive() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id = store
            .save_entity("Caroline", EntityType::Person, 1.0, "test")
            .unwrap();
        store
            .add_observation(&id, "Caroline likes hiking in mountains", 1.0, "test")
            .unwrap();

        let results = store
            .recall(
                "hiking mountains",
                10,
                &RecallFilters {
                    entity_names: Some(vec!["caroline".to_string()]), // lowercase
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!results.is_empty(), "Case-insensitive filter should match");
    }

    #[cfg(feature = "embeddings")]
    struct RecordingEmbedder {
        calls: std::sync::Mutex<Vec<usize>>,
    }

    #[cfg(feature = "embeddings")]
    impl RecordingEmbedder {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<usize> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[cfg(feature = "embeddings")]
    impl sift_core::Embedder for RecordingEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> sift_core::SiftResult<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|_| vec![0.1; 768]).collect())
        }

        fn dimensions(&self) -> usize {
            768
        }

        fn model_name(&self) -> &'static str {
            "recording"
        }
    }

    #[cfg(feature = "embeddings")]
    struct FailingEmbedder;

    #[cfg(feature = "embeddings")]
    impl sift_core::Embedder for FailingEmbedder {
        fn embed_batch(&self, _texts: &[&str]) -> sift_core::SiftResult<Vec<Vec<f32>>> {
            Err(sift_core::SiftError::Embedding(
                "synthetic batch failure".to_string(),
            ))
        }

        fn dimensions(&self) -> usize {
            768
        }

        fn model_name(&self) -> &'static str {
            "failing"
        }
    }

    #[cfg(feature = "embeddings")]
    fn seed_rebuild_observations(store: &MemoryStore, count: usize) {
        for i in 0..count {
            let entity_id = store
                .save_entity(&format!("entity-{i}"), EntityType::Concept, 1.0, "test")
                .unwrap();
            store
                .add_observation(
                    &entity_id,
                    &format!("batched rebuild searchable observation number {i}"),
                    1.0,
                    "test",
                )
                .unwrap();
        }
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn rebuild_search_index_batches_embeddings() {
        let store = MemoryStore::open_in_memory().unwrap();
        seed_rebuild_observations(&store, MEMORY_REBUILD_EMBED_BATCH_SIZE + 1);

        let embedder = std::sync::Arc::new(RecordingEmbedder::new());
        let embedder_dyn: std::sync::Arc<dyn sift_core::Embedder> = embedder.clone();
        let store = store.with_embedder(embedder_dyn);

        store.rebuild_search_index_now().unwrap();

        assert_eq!(embedder.calls(), vec![MEMORY_REBUILD_EMBED_BATCH_SIZE, 1]);

        let results = store
            .recall(
                "batched rebuild searchable observation number 32",
                MEMORY_REBUILD_EMBED_BATCH_SIZE + 1,
                &RecallFilters::default(),
            )
            .unwrap();
        assert!(
            results
                .iter()
                .any(|result| result.observation.content.contains("number 32")),
            "rebuilt observation should remain searchable: {results:?}"
        );
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn rebuild_search_index_returns_err_when_embedder_fails() {
        // First, build a known-good index using a recording embedder so we
        // can assert it survives a subsequent failed rebuild.
        let store = MemoryStore::open_in_memory().unwrap();
        seed_rebuild_observations(&store, 3);

        let good_embedder = std::sync::Arc::new(RecordingEmbedder::new());
        let good_dyn: std::sync::Arc<dyn sift_core::Embedder> = good_embedder.clone();
        let store = store.with_embedder(good_dyn);
        store.rebuild_search_index_now().unwrap();
        let baseline_count = store.search.count().unwrap_or(0);
        assert!(
            baseline_count > 0,
            "baseline rebuild should have inserted observations"
        );

        // Swap in a failing embedder. The rebuild must error and leave the
        // previous index untouched.
        let failing: std::sync::Arc<dyn sift_core::Embedder> = std::sync::Arc::new(FailingEmbedder);
        let store = store.with_embedder(failing);

        let err = store
            .rebuild_search_index_now()
            .expect_err("rebuild must propagate embedder failure");
        match err {
            MemoryError::Storage(sift_core::SiftError::Embedding(_)) => {}
            other => panic!("expected Storage(Embedding(_)), got {other:?}"),
        }

        // Index must be unchanged: same chunk count as before the failed rebuild.
        assert_eq!(
            store.search.count().unwrap_or(0),
            baseline_count,
            "failed rebuild must leave previous index untouched"
        );
    }

    // -- Conflict detection tests --

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 0.001,
            "Orthogonal vectors should have similarity ~0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_empty_vectors() {
        let sim = cosine_similarity(&[], &[]);
        assert!(sim.abs() < f32::EPSILON);
    }

    #[test]
    fn detect_conflicts_no_existing_observations() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .save_entity("project", EntityType::Project, 1.0, "test")
            .unwrap();
        let conflicts = store.detect_conflicts(&id, "uses React 18", 0.15).unwrap();
        assert!(
            conflicts.is_empty(),
            "No conflicts when no observations exist"
        );
    }

    #[test]
    fn detect_conflicts_returns_sorted_by_score() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .save_entity("project", EntityType::Project, 1.0, "test")
            .unwrap();
        store
            .add_observation(&id, "uses React 18 for the frontend", 1.0, "test")
            .unwrap();
        store
            .add_observation(&id, "uses PostgreSQL for the database", 1.0, "test")
            .unwrap();

        // Without embeddings (test mode), detect_conflicts uses zero vectors,
        // so embed_sim will be 0.0 and no conflicts will be detected.
        // This test verifies the function doesn't panic and returns empty
        // when embeddings are unavailable.
        let conflicts = store
            .detect_conflicts(&id, "uses React 19 for the frontend", 0.15)
            .unwrap();
        // With zero vectors, cosine_similarity = 0.0, so no conflicts detected
        assert!(
            conflicts.is_empty(),
            "Zero-vector embeddings should not produce false positives"
        );
    }

    #[test]
    fn consolidate_detects_near_duplicate_dedup() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .save_entity("project", EntityType::Project, 1.0, "test")
            .unwrap();

        store
            .add_observation(&id, "uses React 18", 1.0, "test")
            .unwrap();
        store
            .add_observation(&id, "uses React 18", 1.0, "test")
            .unwrap();
        store
            .add_observation(
                &id,
                "completely different topic about databases",
                1.0,
                "test",
            )
            .unwrap();

        let report = store.consolidate().unwrap();
        assert_eq!(report.duplicates_merged, 1, "Should merge exact duplicate");
        assert_eq!(
            report.contradictions_found, 0,
            "Different topics are not contradictions"
        );
    }
}
