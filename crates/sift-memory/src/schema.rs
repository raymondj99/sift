//! SQLite schema initialization and migration for the memory knowledge graph.
//!
//! Follows the same patterns as `sift-store/src/metadata.rs`:
//! - WAL mode for concurrent reads
//! - Busy timeout for write contention
//! - Indexes on common query patterns
//!
//! ## Schema Versioning
//!
//! The schema version is stored in `memory_meta`. Migrations are applied
//! incrementally: each version bump has a dedicated `migrate_vN_to_vM`
//! function that runs only when upgrading from an older version.

use rusqlite::Connection;

/// Current schema version. Bump when adding migrations.
pub const SCHEMA_VERSION: i64 = 4;

/// Initialize the memory database schema.
///
/// Creates tables if they don't exist and runs any pending migrations.
/// Safe to call on both fresh databases and existing v1 databases.
pub fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Performance pragmas (matching MetadataStore pattern)
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -4000;
         PRAGMA foreign_keys = ON;",
    )?;

    // Create base tables (v1) — IF NOT EXISTS makes this safe to re-run
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
            logical_id  TEXT NOT NULL,
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

    // Check current version and apply migrations
    let current = current_schema_version(conn);

    if current < 2 {
        migrate_v1_to_v2(conn)?;
    }
    if current < 3 {
        migrate_v2_to_v3(conn)?;
    }
    if current < 4 {
        migrate_v3_to_v4(conn)?;
    }

    if has_column(conn, "observations", "logical_id")? {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_observations_logical_id
                ON observations(logical_id);",
        )?;
    }

    // Store schema version
    conn.execute(
        "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;

    Ok(())
}

/// Read the stored schema version, returning 0 if unset (fresh install).
fn current_schema_version(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT value FROM memory_meta WHERE key = 'schema_version'",
        [],
        |row| {
            let v: String = row.get(0)?;
            Ok(v.parse::<i64>().unwrap_or(0))
        },
    )
    .unwrap_or(0)
}

/// Check whether a column exists on a table using `PRAGMA table_info`.
fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, rusqlite::Error> {
    // table_info returns: cid, name, type, notnull, dflt_value, pk
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let exists = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(Result::ok)
        .any(|name| name == column);
    Ok(exists)
}

/// Add a column to a table if it doesn't already exist.
///
/// SQLite has no `ADD COLUMN IF NOT EXISTS` syntax, so we check
/// `PRAGMA table_info` first to make the migration idempotent.
fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), rusqlite::Error> {
    if !has_column(conn, table, column)? {
        conn.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {definition};"
        ))?;
    }
    Ok(())
}

/// Migrate from schema v1 to v2: add Cortex memory system tables.
///
/// Adds:
/// - `episodes` table for staging raw hook events before consolidation
/// - `access_log` table for retrieval-dependent strengthening
/// - New columns on `observations`: access_count, last_accessed, utility_score,
///   memory_tier, token_count
/// - New columns on `entities`: access_count, last_accessed
fn migrate_v1_to_v2(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Run the entire migration in a single transaction so a crash
    // mid-migration doesn't leave a partial v2 schema.
    conn.execute_batch("BEGIN")?;

    let result = migrate_v1_to_v2_inner(conn);
    match result {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    Ok(())
}

fn migrate_v1_to_v2_inner(conn: &Connection) -> Result<(), rusqlite::Error> {
    // -- New tables --

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS episodes (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            tool_name   TEXT,
            content     TEXT NOT NULL,
            timestamp   INTEGER NOT NULL,
            processed   INTEGER NOT NULL DEFAULT 0,
            batch_id    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_processed
            ON episodes(processed, timestamp);

        CREATE INDEX IF NOT EXISTS idx_episodes_session
            ON episodes(session_id);

        CREATE TABLE IF NOT EXISTS access_log (
            id              TEXT PRIMARY KEY,
            observation_id  TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
            accessed_at     INTEGER NOT NULL,
            query           TEXT,
            session_id      TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_access_observation
            ON access_log(observation_id);

        CREATE INDEX IF NOT EXISTS idx_access_accessed_at
            ON access_log(accessed_at);",
    )?;

    // -- New columns on observations --
    add_column_if_missing(
        conn,
        "observations",
        "access_count",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    add_column_if_missing(conn, "observations", "last_accessed", "INTEGER")?;
    add_column_if_missing(conn, "observations", "utility_score", "REAL")?;
    add_column_if_missing(
        conn,
        "observations",
        "memory_tier",
        "TEXT NOT NULL DEFAULT 'episodic'",
    )?;
    add_column_if_missing(conn, "observations", "token_count", "INTEGER")?;

    // -- New columns on entities --
    add_column_if_missing(
        conn,
        "entities",
        "access_count",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    add_column_if_missing(conn, "entities", "last_accessed", "INTEGER")?;

    // -- Index for tier-based queries --
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_observations_tier
            ON observations(memory_tier) WHERE valid_until IS NULL;",
    )?;

    Ok(())
}

/// Migrate from schema v2 to v3: add skills table for procedural memory.
fn migrate_v2_to_v3(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch("BEGIN")?;

    let result = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS skills (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL UNIQUE,
            description     TEXT NOT NULL,
            pattern         TEXT NOT NULL,
            trigger_context TEXT,
            frequency       INTEGER NOT NULL DEFAULT 1,
            confidence      REAL NOT NULL DEFAULT 0.5,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL,
            source_episodes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
        CREATE INDEX IF NOT EXISTS idx_skills_confidence ON skills(confidence);",
    );

    match result {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    Ok(())
}

/// Migrate from schema v3 to v4: add logical observation lineage IDs.
fn migrate_v3_to_v4(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch("BEGIN")?;

    let result = migrate_v3_to_v4_inner(conn);
    match result {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    Ok(())
}

fn migrate_v3_to_v4_inner(conn: &Connection) -> Result<(), rusqlite::Error> {
    add_column_if_missing(conn, "observations", "logical_id", "TEXT")?;
    conn.execute(
        "UPDATE observations SET logical_id = id WHERE logical_id IS NULL OR logical_id = ''",
        [],
    )?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_observations_logical_id
            ON observations(logical_id);",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_schema_creates_all_tables() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

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
        assert!(tables.contains(&"episodes".to_string()));
        assert!(tables.contains(&"access_log".to_string()));
        assert!(tables.contains(&"skills".to_string()));
    }

    #[test]
    fn init_schema_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap(); // Should not error
    }

    #[test]
    fn schema_version_stored_as_v4() {
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
        assert_eq!(version, "4");
    }

    #[test]
    fn skills_table_exists() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        conn.execute(
            "INSERT INTO skills (id, name, description, pattern, created_at, updated_at)
             VALUES ('s1', 'test-skill', 'A test', 'do the thing', 1000, 1000)",
            [],
        )
        .unwrap();

        let name: String = conn
            .query_row("SELECT name FROM skills WHERE id = 's1'", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, "test-skill");
    }

    #[test]
    fn foreign_keys_enforced() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Inserting an observation with a non-existent entity_id should fail
        let result = conn.execute(
            "INSERT INTO observations (id, logical_id, entity_id, content, observed_at)
             VALUES ('obs1', 'obs1', 'nonexistent', 'test', 1000)",
            [],
        );
        assert!(result.is_err());
    }

    #[test]
    fn v2_columns_exist_on_observations() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        assert!(has_column(&conn, "observations", "access_count").unwrap());
        assert!(has_column(&conn, "observations", "last_accessed").unwrap());
        assert!(has_column(&conn, "observations", "utility_score").unwrap());
        assert!(has_column(&conn, "observations", "memory_tier").unwrap());
        assert!(has_column(&conn, "observations", "token_count").unwrap());
        assert!(has_column(&conn, "observations", "logical_id").unwrap());
    }

    #[test]
    fn v2_columns_exist_on_entities() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        assert!(has_column(&conn, "entities", "access_count").unwrap());
        assert!(has_column(&conn, "entities", "last_accessed").unwrap());
    }

    #[test]
    fn episodes_table_structure() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Insert and query an episode to verify structure
        conn.execute(
            "INSERT INTO episodes (id, session_id, event_type, tool_name, content, timestamp)
             VALUES ('ep1', 'sess1', 'post_tool_use', 'Edit', '{\"test\": true}', 1000)",
            [],
        )
        .unwrap();

        let (processed, batch_id): (i32, Option<String>) = conn
            .query_row(
                "SELECT processed, batch_id FROM episodes WHERE id = 'ep1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(processed, 0);
        assert!(batch_id.is_none());
    }

    #[test]
    fn access_log_cascade_delete() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Create entity + observation
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
             VALUES ('e1', 'test', 'concept', 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations (id, logical_id, entity_id, content, observed_at)
             VALUES ('obs1', 'obs1', 'e1', 'test fact', 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO access_log (id, observation_id, accessed_at, query)
             VALUES ('a1', 'obs1', 2000, 'test query')",
            [],
        )
        .unwrap();

        // Delete observation — access_log entry should cascade
        conn.execute("DELETE FROM observations WHERE id = 'obs1'", [])
            .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM access_log", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn upgrade_from_v1_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();

        // Simulate a v1 database: create base tables without v2 additions
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE entities (
                 id TEXT PRIMARY KEY, name TEXT NOT NULL, entity_type TEXT NOT NULL,
                 created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                 confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL DEFAULT ''
             );
             CREATE TABLE observations (
                 id TEXT PRIMARY KEY,
                 entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                 content TEXT NOT NULL, observed_at INTEGER NOT NULL,
                 valid_from INTEGER, valid_until INTEGER,
                 confidence REAL NOT NULL DEFAULT 1.0,
                 source TEXT NOT NULL DEFAULT '',
                 supersedes TEXT REFERENCES observations(id) ON DELETE SET NULL
             );
             CREATE TABLE relations (
                 id TEXT PRIMARY KEY,
                 from_entity TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                 to_entity TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                 relation_type TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
                 created_at INTEGER NOT NULL, valid_from INTEGER, valid_until INTEGER,
                 source TEXT NOT NULL DEFAULT ''
             );
             CREATE TABLE memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO memory_meta (key, value) VALUES ('schema_version', '1');",
        )
        .unwrap();

        // Verify it's v1
        assert_eq!(current_schema_version(&conn), 1);
        assert!(!has_column(&conn, "observations", "access_count").unwrap());

        // Run init_schema — should upgrade to v4 (v1→v2→v3→v4)
        init_schema(&conn).unwrap();

        assert_eq!(current_schema_version(&conn), 4);
        assert!(has_column(&conn, "observations", "access_count").unwrap());
        assert!(has_column(&conn, "observations", "memory_tier").unwrap());
        assert!(has_column(&conn, "observations", "logical_id").unwrap());
        assert!(has_column(&conn, "entities", "access_count").unwrap());

        // Running again should be safe
        init_schema(&conn).unwrap();
        assert_eq!(current_schema_version(&conn), 4);
    }

    #[test]
    fn v2_migration_is_transactional() {
        // Verify that after a successful migration, all v2 artifacts exist.
        // The transaction ensures either all or none are applied.
        let conn = Connection::open_in_memory().unwrap();

        // Create v1 schema manually
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL, entity_type TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL DEFAULT '');
             CREATE TABLE observations (id TEXT PRIMARY KEY, entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE, content TEXT NOT NULL, observed_at INTEGER NOT NULL, valid_from INTEGER, valid_until INTEGER, confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL DEFAULT '', supersedes TEXT);
             CREATE TABLE relations (id TEXT PRIMARY KEY, from_entity TEXT NOT NULL, to_entity TEXT NOT NULL, relation_type TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0, created_at INTEGER NOT NULL, valid_from INTEGER, valid_until INTEGER, source TEXT NOT NULL DEFAULT '');
             CREATE TABLE memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO memory_meta (key, value) VALUES ('schema_version', '1');",
        )
        .unwrap();

        // Insert some v1 data
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at, updated_at) VALUES ('e1', 'test', 'concept', 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, observed_at) VALUES ('o1', 'e1', 'fact', 1000)",
            [],
        )
        .unwrap();

        // Run migration
        init_schema(&conn).unwrap();

        // All v2 artifacts must exist
        assert!(has_column(&conn, "observations", "access_count").unwrap());
        assert!(has_column(&conn, "observations", "memory_tier").unwrap());
        assert!(has_column(&conn, "observations", "logical_id").unwrap());
        assert!(has_column(&conn, "entities", "access_count").unwrap());

        // Existing data should have correct defaults
        let tier: String = conn
            .query_row(
                "SELECT memory_tier FROM observations WHERE id = 'o1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tier, "episodic");

        let access: i64 = conn
            .query_row(
                "SELECT access_count FROM observations WHERE id = 'o1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(access, 0);

        let logical_id: String = conn
            .query_row(
                "SELECT logical_id FROM observations WHERE id = 'o1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(logical_id, "o1");
    }

    #[test]
    fn memory_tier_default_value() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
             VALUES ('e1', 'test', 'concept', 1000, 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations (id, logical_id, entity_id, content, observed_at)
             VALUES ('obs1', 'obs1', 'e1', 'test fact', 1000)",
            [],
        )
        .unwrap();

        let tier: String = conn
            .query_row(
                "SELECT memory_tier FROM observations WHERE id = 'obs1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tier, "episodic");
    }
}
