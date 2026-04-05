//! SQLite schema initialization for the memory knowledge graph.
//!
//! Follows the same patterns as `sift-store/src/metadata.rs`:
//! - WAL mode for concurrent reads
//! - Busy timeout for write contention
//! - Indexes on common query patterns

use rusqlite::Connection;

/// Current schema version. Bump when adding migrations.
pub const SCHEMA_VERSION: i64 = 1;

/// Initialize the memory database schema.
///
/// Creates tables if they don't exist and runs any pending migrations.
pub fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Performance pragmas (matching MetadataStore pattern)
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -4000;
         PRAGMA foreign_keys = ON;",
    )?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL,
            confidence  REAL NOT NULL DEFAULT 1.0,
            source      TEXT NOT NULL DEFAULT ''
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name
            ON entities(name);

        CREATE INDEX IF NOT EXISTS idx_entities_type
            ON entities(entity_type);

        CREATE TABLE IF NOT EXISTS observations (
            id          TEXT PRIMARY KEY,
            entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            observed_at INTEGER NOT NULL,
            valid_from  INTEGER,
            valid_until INTEGER,
            confidence  REAL NOT NULL DEFAULT 1.0,
            source      TEXT NOT NULL DEFAULT '',
            supersedes  TEXT REFERENCES observations(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_observations_entity
            ON observations(entity_id);

        CREATE INDEX IF NOT EXISTS idx_observations_validity
            ON observations(valid_from, valid_until);

        CREATE TABLE IF NOT EXISTS relations (
            id            TEXT PRIMARY KEY,
            from_entity   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            to_entity     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            weight        REAL NOT NULL DEFAULT 1.0,
            created_at    INTEGER NOT NULL,
            valid_from    INTEGER,
            valid_until   INTEGER,
            source        TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_relations_from
            ON relations(from_entity);

        CREATE INDEX IF NOT EXISTS idx_relations_to
            ON relations(to_entity);

        CREATE INDEX IF NOT EXISTS idx_relations_type
            ON relations(relation_type);

        CREATE TABLE IF NOT EXISTS memory_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;

    // Store schema version
    conn.execute(
        "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_schema_creates_tables() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();

        assert!(tables.contains(&"entities".to_string()));
        assert!(tables.contains(&"observations".to_string()));
        assert!(tables.contains(&"relations".to_string()));
        assert!(tables.contains(&"memory_meta".to_string()));
    }

    #[test]
    fn init_schema_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap(); // Should not error
    }

    #[test]
    fn schema_version_stored() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        let version: String = conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION.to_string());
    }

    #[test]
    fn foreign_keys_enforced() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Inserting an observation with a non-existent entity_id should fail
        let result = conn.execute(
            "INSERT INTO observations (id, entity_id, content, observed_at)
             VALUES ('obs1', 'nonexistent', 'test', 1000)",
            [],
        );
        assert!(result.is_err());
    }
}
