//! Episode capture for the Cortex automated memory system.
//!
//! Episodes are the hot path: Claude Code hooks write raw events here via
//! `sift memory ingest`. The attention filter decides what's worth encoding
//! (write tools, session boundaries) vs. what's noise (read-only tools,
//! trivial commands).
//!
//! # Performance Contract
//!
//! The `ingest` function must complete in <100ms. It opens only SQLite —
//! no search engine, no embedding model. All intelligence is deferred to
//! the consolidation pipeline.

use crate::schema;
use crate::types::{Episode, EpisodeState, EventType};
use rusqlite::Connection;
use std::path::Path;

/// Lightweight handle for episode ingestion (hot path only).
///
/// Unlike `MemoryStore`, this opens only the SQLite database — no vector
/// store, no FTS index, no embedder. This is intentional: the ingest path
/// must be fast enough to run inside a Claude Code hook timeout (5s, but
/// we target <100ms).
pub struct EpisodeStore {
    db: Connection,
}

impl EpisodeStore {
    /// Borrow the underlying connection for direct queries.
    ///
    /// Use this instead of opening a second connection to the same DB.
    pub fn connection(&self) -> &Connection {
        &self.db
    }

    /// Open the episode store at the standard memory directory.
    pub fn open(memory_dir: &Path) -> Result<Self, rusqlite::Error> {
        std::fs::create_dir_all(memory_dir).map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Cannot create memory dir: {e}"))
        })?;

        let db_path = memory_dir.join("memory.db");
        let conn = Connection::open(&db_path)?;
        schema::init_schema(&conn)?;
        Ok(Self { db: conn })
    }

    /// Open an in-memory episode store (for benchmarks and external tests).
    pub fn open_in_memory_for_bench() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        schema::init_schema(&conn)?;
        Ok(Self { db: conn })
    }

    /// Open an in-memory episode store (for testing).
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        schema::init_schema(&conn)?;
        Ok(Self { db: conn })
    }

    /// Ingest a hook event, applying the attention filter.
    ///
    /// Returns `Some(episode_id)` if the event was stored, or `None` if
    /// the attention filter decided to skip it.
    pub fn ingest(
        &self,
        session_id: &str,
        event_type: EventType,
        content: &str,
    ) -> Result<Option<String>, rusqlite::Error> {
        let parsed: serde_json::Value =
            serde_json::from_str(content).unwrap_or(serde_json::Value::Null);

        let tool_name = parsed
            .get("tool_name")
            .and_then(|v| v.as_str())
            .map(String::from);

        // For Bash commands: extract the command string once from the parsed
        // JSON tree instead of re-parsing in the attention filter.
        let bash_command = parsed
            .get("tool_input")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| v.get("command").and_then(|c| c.as_str()).map(String::from))
            .or_else(|| {
                // tool_input might be an object, not a string
                parsed
                    .get("tool_input")
                    .and_then(|v| v.get("command"))
                    .and_then(|c| c.as_str())
                    .map(String::from)
            });

        // Apply attention filter
        let decision = attention_filter(event_type, tool_name.as_deref(), bash_command.as_deref());
        if decision == AttentionDecision::Skip {
            return Ok(None);
        }

        let id = uuid::Uuid::now_v7().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        self.db.execute(
            "INSERT INTO episodes (id, session_id, event_type, tool_name, content, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![id, session_id, event_type.as_str(), tool_name, content, now],
        )?;

        Ok(Some(id))
    }

    /// Get the count of pending (unprocessed) episodes.
    pub fn pending_count(&self) -> Result<u64, rusqlite::Error> {
        self.db.query_row(
            "SELECT COUNT(*) FROM episodes WHERE processed = ?1",
            [EpisodeState::Pending as i32],
            |row| row.get(0),
        )
    }

    /// Get the total episode count.
    pub fn total_count(&self) -> Result<u64, rusqlite::Error> {
        self.db
            .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))
    }

    /// Fetch unprocessed episodes in chronological order, up to `limit`.
    pub fn fetch_pending(&self, limit: usize) -> Result<Vec<Episode>, rusqlite::Error> {
        let mut stmt = self.db.prepare(
            "SELECT id, session_id, event_type, tool_name, content, timestamp, processed, batch_id
             FROM episodes
             WHERE processed = ?1
             ORDER BY timestamp ASC
             LIMIT ?2",
        )?;

        let rows = stmt
            .query_map(
                rusqlite::params![EpisodeState::Pending as i32, limit as i64],
                |row| {
                    let event_type_str: String = row.get(2)?;
                    let event_type = EventType::parse(&event_type_str).ok_or_else(|| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("unknown event_type {event_type_str:?}"),
                            )),
                        )
                    })?;
                    Ok(Episode {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        event_type,
                        tool_name: row.get(3)?,
                        content: row.get(4)?,
                        timestamp: row.get(5)?,
                        processed: row.get(6)?,
                        batch_id: row.get(7)?,
                    })
                },
            )?
            .filter_map(Result::ok)
            .collect();

        Ok(rows)
    }

    /// Mark episodes as processed (or skipped) by a consolidation batch.
    ///
    /// All updates are batched in a single transaction.
    pub fn mark_processed(
        &self,
        episode_ids: &[String],
        state: EpisodeState,
        batch_id: &str,
    ) -> Result<(), rusqlite::Error> {
        if episode_ids.is_empty() {
            return Ok(());
        }
        self.db.execute_batch("BEGIN")?;
        let mut stmt = self
            .db
            .prepare("UPDATE episodes SET processed = ?1, batch_id = ?2 WHERE id = ?3")?;

        for id in episode_ids {
            stmt.execute(rusqlite::params![state as i32, batch_id, id])?;
        }
        drop(stmt);
        self.db.execute_batch("COMMIT")?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attention Filter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionDecision {
    /// Worth encoding — write to episodes table.
    Encode,
    /// Noise — discard without storing.
    Skip,
}

/// Decide whether a hook event is worth encoding into an episode.
///
/// This is the core intelligence of the hot path. The rules are:
///
/// **Always encode:**
/// - `stop` events (contain `last_assistant_message`)
/// - `post_compact` events (contain `compact_summary` — pre-distilled knowledge)
///
/// **Encode if write tool:**
/// - `post_tool_use` for: Edit, Write, MultiEdit, Bash (if write command),
///   NotebookEdit, and any MCP tool (prefix `mcp__`)
///
/// **Skip (read-only noise):**
/// - `post_tool_use` for: Read, Glob, Grep, Agent, AskUser, TodoRead, TodoWrite
/// - `post_tool_use` for sift read-only MCP tools
/// - Trivial bash commands (ls, cat, pwd, echo, git status/log/diff)
///
/// `bash_command` should be the pre-extracted shell command string for Bash
/// tools (avoids double JSON parsing on the hot path).
pub fn attention_filter(
    event_type: EventType,
    tool_name: Option<&str>,
    bash_command: Option<&str>,
) -> AttentionDecision {
    match event_type {
        // Session boundaries are always valuable
        EventType::Stop | EventType::PostCompact => AttentionDecision::Encode,

        EventType::PostToolUse => {
            let tool = match tool_name {
                Some(t) => t,
                None => return AttentionDecision::Skip,
            };

            // Skip known read-only tools
            if is_read_only_tool(tool) {
                return AttentionDecision::Skip;
            }

            // Write tools: always encode
            if is_write_tool(tool) {
                // Special case: Bash — check if the command is trivial/read-only
                if tool == "Bash" {
                    if let Some(cmd) = bash_command {
                        if is_trivial_bash(cmd) {
                            return AttentionDecision::Skip;
                        }
                    }
                }
                return AttentionDecision::Encode;
            }

            // MCP tools: encode non-sift-read-only ones
            if tool.starts_with("mcp__") {
                if is_sift_read_only_mcp(tool) {
                    return AttentionDecision::Skip;
                }
                return AttentionDecision::Encode;
            }

            // Unknown tool — default to skip (conservative)
            AttentionDecision::Skip
        }
    }
}

/// Tools that only read state — never worth encoding.
fn is_read_only_tool(tool: &str) -> bool {
    matches!(
        tool,
        "Read"
            | "Glob"
            | "Grep"
            | "Agent"
            | "AskUser"
            | "AskUserQuestion"
            | "TodoRead"
            | "TaskList"
            | "TaskGet"
            | "ToolSearch"
            | "LSP"
            | "WebSearch"
            | "WebFetch"
            | "SendMessage"
    )
}

/// Tools that modify state — worth encoding.
fn is_write_tool(tool: &str) -> bool {
    matches!(
        tool,
        "Edit"
            | "Write"
            | "MultiEdit"
            | "Bash"
            | "NotebookEdit"
            | "TodoWrite"
            | "TaskCreate"
            | "TaskUpdate"
    )
}

/// Sift MCP tools that are read-only.
fn is_sift_read_only_mcp(tool: &str) -> bool {
    matches!(
        tool,
        "mcp__sift__sift_search"
            | "mcp__sift__sift_recall"
            | "mcp__sift__sift_status"
            | "mcp__sift__sift_memory_status"
            | "mcp__sift__sift_list_entities"
            | "mcp__sift__sift_list_sources"
            | "mcp__sift__sift_get_entity"
            | "mcp__sift__sift_search_skills"
    )
}

/// Check if a Bash command is trivial/read-only (not worth remembering).
///
/// Takes a plain shell command string (already extracted from JSON by the
/// caller). Errs on the side of encoding (returns false for anything
/// ambiguous — piped commands, env prefixes, etc.).
fn is_trivial_bash(command: &str) -> bool {
    let trimmed = command.trim();

    // Check first token of the command
    let first_token = trimmed.split_whitespace().next().unwrap_or("");

    matches!(
        first_token,
        "ls" | "cat"
            | "head"
            | "tail"
            | "pwd"
            | "echo"
            | "wc"
            | "file"
            | "which"
            | "where"
            | "type"
            | "env"
            | "printenv"
            | "whoami"
            | "date"
            | "df"
            | "du"
            | "uptime"
            | "uname"
            | "hostname"
            | "id"
    ) || is_trivial_git(trimmed)
}

/// Check if a command is a read-only git operation.
fn is_trivial_git(command: &str) -> bool {
    let trimmed = command.trim();
    if !trimmed.starts_with("git ") && !trimmed.starts_with("git\t") {
        return false;
    }

    // Extract the git subcommand
    let rest = trimmed.strip_prefix("git").unwrap_or("").trim();
    let subcommand = rest.split_whitespace().next().unwrap_or("");

    matches!(
        subcommand,
        "status"
            | "log"
            | "diff"
            | "show"
            | "branch"
            | "remote"
            | "tag"
            | "describe"
            | "rev-parse"
            | "ls-files"
            | "ls-tree"
            | "cat-file"
            | "rev-list"
            | "shortlog"
            | "blame"
    )
}

/// Parse a session ID from environment variables (fallback).
///
/// The primary session ID source is `$.session_id` in the JSON stdin
/// payload. This function is the fallback when the payload doesn't
/// contain a session ID. Checks `CLAUDE_CODE_SESSION_ID` and
/// `CLAUDE_SESSION_ID` env vars, then generates a UUID as last resort.
pub fn session_id_from_env() -> String {
    std::env::var("CLAUDE_CODE_SESSION_ID")
        .or_else(|_| std::env::var("CLAUDE_SESSION_ID"))
        .unwrap_or_else(|_| format!("unknown-{}", uuid::Uuid::now_v7()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Attention Filter Tests --

    #[test]
    fn stop_events_always_encode() {
        assert_eq!(
            attention_filter(EventType::Stop, None, None),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn post_compact_events_always_encode() {
        assert_eq!(
            attention_filter(EventType::PostCompact, None, None),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn write_tools_encode() {
        for tool in ["Edit", "Write", "MultiEdit", "NotebookEdit"] {
            assert_eq!(
                attention_filter(EventType::PostToolUse, Some(tool), None),
                AttentionDecision::Encode,
                "Expected {tool} to encode"
            );
        }
    }

    #[test]
    fn read_tools_skip() {
        for tool in ["Read", "Glob", "Grep", "Agent", "LSP"] {
            assert_eq!(
                attention_filter(EventType::PostToolUse, Some(tool), None),
                AttentionDecision::Skip,
                "Expected {tool} to skip"
            );
        }
    }

    #[test]
    fn bash_write_commands_encode() {
        assert_eq!(
            attention_filter(
                EventType::PostToolUse,
                Some("Bash"),
                Some("cargo build --release")
            ),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn bash_trivial_commands_skip() {
        for cmd in [
            "ls -la",
            "cat README.md",
            "pwd",
            "echo hello",
            "git status",
            "git log --oneline",
        ] {
            assert_eq!(
                attention_filter(EventType::PostToolUse, Some("Bash"), Some(cmd)),
                AttentionDecision::Skip,
                "Expected bash '{cmd}' to skip"
            );
        }
    }

    #[test]
    fn bash_git_write_commands_encode() {
        for cmd in [
            "git commit -m 'test'",
            "git push",
            "git merge main",
            "git rebase main",
        ] {
            assert_eq!(
                attention_filter(EventType::PostToolUse, Some("Bash"), Some(cmd)),
                AttentionDecision::Encode,
                "Expected bash '{cmd}' to encode"
            );
        }
    }

    #[test]
    fn sift_read_mcp_tools_skip() {
        assert_eq!(
            attention_filter(EventType::PostToolUse, Some("mcp__sift__sift_search"), None),
            AttentionDecision::Skip
        );
        assert_eq!(
            attention_filter(EventType::PostToolUse, Some("mcp__sift__sift_recall"), None),
            AttentionDecision::Skip
        );
    }

    #[test]
    fn sift_write_mcp_tools_encode() {
        assert_eq!(
            attention_filter(
                EventType::PostToolUse,
                Some("mcp__sift__sift_remember"),
                None
            ),
            AttentionDecision::Encode
        );
        assert_eq!(
            attention_filter(
                EventType::PostToolUse,
                Some("mcp__sift__sift_index_text"),
                None
            ),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn unknown_mcp_tools_encode() {
        assert_eq!(
            attention_filter(
                EventType::PostToolUse,
                Some("mcp__slack__send_message"),
                None
            ),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn no_tool_name_in_post_tool_use_skips() {
        assert_eq!(
            attention_filter(EventType::PostToolUse, None, None),
            AttentionDecision::Skip
        );
    }

    // -- EpisodeStore Tests --

    #[test]
    fn ingest_and_fetch_pending() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // Ingest a write tool event
        let content = r#"{"tool_name": "Edit", "tool_input": "{}", "tool_response": "ok"}"#;
        let id = store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();
        assert!(id.is_some());

        // Ingest a read tool event — should be skipped
        let content = r#"{"tool_name": "Read", "tool_input": "{}", "tool_response": "ok"}"#;
        let id = store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();
        assert!(id.is_none());

        assert_eq!(store.pending_count().unwrap(), 1);
        assert_eq!(store.total_count().unwrap(), 1);

        let episodes = store.fetch_pending(10).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].event_type, EventType::PostToolUse);
        assert_eq!(episodes[0].tool_name.as_deref(), Some("Edit"));
        assert_eq!(episodes[0].session_id, "sess1");
    }

    #[test]
    fn ingest_stop_event() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"last_assistant_message": "Done!", "stop_hook_active": true}"#;
        let id = store.ingest("sess1", EventType::Stop, content).unwrap();
        assert!(id.is_some());
        assert_eq!(store.pending_count().unwrap(), 1);
    }

    #[test]
    fn ingest_codex_stop_payload() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{
            "session_id": "codex-session",
            "last_assistant_message": "Implemented Codex hook parity and queued consolidation.",
            "turn_id": "turn_123",
            "stop_hook_active": true
        }"#;
        let id = store
            .ingest("codex-session", EventType::Stop, content)
            .unwrap();
        assert!(id.is_some());

        let episodes = store.fetch_pending(10).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].event_type, EventType::Stop);
        assert!(episodes[0].content.contains("last_assistant_message"));
    }

    #[test]
    fn ingest_post_compact_event() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"compact_summary": "User is building a memory system..."}"#;
        let id = store
            .ingest("sess1", EventType::PostCompact, content)
            .unwrap();
        assert!(id.is_some());
    }

    #[test]
    fn mark_processed() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"tool_name": "Edit", "tool_input": "{}"}"#;
        let id = store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap()
            .unwrap();

        store
            .mark_processed(&[id], EpisodeState::Processed, "batch-1")
            .unwrap();

        assert_eq!(store.pending_count().unwrap(), 0);
        assert_eq!(store.total_count().unwrap(), 1);
    }

    #[test]
    fn fetch_pending_respects_limit() {
        let store = EpisodeStore::open_in_memory().unwrap();

        for i in 0..5 {
            let content = format!(r#"{{"tool_name": "Edit", "tool_input": "file{i}"}}"#);
            store
                .ingest("sess1", EventType::PostToolUse, &content)
                .unwrap();
        }

        let episodes = store.fetch_pending(3).unwrap();
        assert_eq!(episodes.len(), 3);
    }

    #[test]
    fn fetch_pending_chronological_order() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // Insert in order — timestamps are auto-generated, so they should be ordered
        let content1 = r#"{"tool_name": "Edit", "tool_input": "first"}"#;
        let content2 = r#"{"tool_name": "Write", "tool_input": "second"}"#;

        store
            .ingest("sess1", EventType::PostToolUse, content1)
            .unwrap();
        store
            .ingest("sess1", EventType::PostToolUse, content2)
            .unwrap();

        let episodes = store.fetch_pending(10).unwrap();
        assert_eq!(episodes.len(), 2);
        assert!(episodes[0].timestamp <= episodes[1].timestamp);
    }

    // -----------------------------------------------------------------------
    // Full pipeline tests (JSON → extract → filter → store)
    // -----------------------------------------------------------------------

    #[test]
    fn ingest_extracts_bash_command_and_filters_trivial() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // tool_input as a JSON string (Claude Code sends it this way)
        let trivial = r#"{"tool_name": "Bash", "tool_input": "{\"command\": \"git status\"}"}"#;
        let result = store
            .ingest("sess1", EventType::PostToolUse, trivial)
            .unwrap();
        assert!(result.is_none(), "trivial bash should be filtered");

        // Non-trivial bash command should be stored
        let write = r#"{"tool_name": "Bash", "tool_input": "{\"command\": \"cargo build\"}"}"#;
        let result = store
            .ingest("sess1", EventType::PostToolUse, write)
            .unwrap();
        assert!(result.is_some(), "cargo build should be stored");
    }

    #[test]
    fn ingest_handles_tool_input_as_object() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // Some hooks send tool_input as an object, not a JSON string
        let content = r#"{"tool_name": "Bash", "tool_input": {"command": "ls -la"}}"#;
        let result = store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();
        assert!(
            result.is_none(),
            "ls should be filtered even as object input"
        );

        let content = r#"{"tool_name": "Bash", "tool_input": {"command": "make install"}}"#;
        let result = store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();
        assert!(result.is_some(), "make install should be stored");
    }

    #[test]
    fn ingest_codex_bash_post_tool_use_payload() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{
            "session_id": "codex-session",
            "tool_name": "Bash",
            "tool_input": {"command": "cargo test -p sift-cli"},
            "tool_response": {"exit_code": 0, "stdout": "ok"}
        }"#;
        let result = store
            .ingest("codex-session", EventType::PostToolUse, content)
            .unwrap();
        assert!(result.is_some(), "codex bash tool use should be stored");

        let episodes = store.fetch_pending(10).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].tool_name.as_deref(), Some("Bash"));
    }

    #[test]
    fn ingest_handles_malformed_json_gracefully() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // Completely invalid JSON — should not panic, should skip
        let result = store
            .ingest("sess1", EventType::PostToolUse, "not json")
            .unwrap();
        assert!(result.is_none(), "malformed JSON should skip gracefully");
    }

    #[test]
    fn ingest_empty_stdin_returns_ok() {
        let store = EpisodeStore::open_in_memory().unwrap();

        // Empty content with stop event — stop always encodes
        let result = store.ingest("sess1", EventType::Stop, "").unwrap();
        // Note: even empty content on a stop event will encode,
        // because attention filter only checks event_type for stop
        assert!(result.is_some());
    }

    #[test]
    fn ingest_preserves_full_content() {
        let store = EpisodeStore::open_in_memory().unwrap();

        let content = r#"{"tool_name": "Edit", "tool_input": {"file_path": "/foo/bar.rs"}, "tool_response": "success: 42 lines changed"}"#;
        store
            .ingest("sess1", EventType::PostToolUse, content)
            .unwrap();

        let episodes = store.fetch_pending(10).unwrap();
        assert_eq!(episodes.len(), 1);
        // Full JSON payload is preserved for consolidation
        assert!(episodes[0].content.contains("tool_response"));
        assert!(episodes[0].content.contains("42 lines changed"));
    }

    #[test]
    fn connection_accessor() {
        let store = EpisodeStore::open_in_memory().unwrap();
        // Verify we can use the connection directly for stat queries
        let count: i64 = store
            .connection()
            .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    // -----------------------------------------------------------------------
    // Attention filter edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn bash_no_command_encodes() {
        // Bash with no command extraction — encode by default (conservative)
        assert_eq!(
            attention_filter(EventType::PostToolUse, Some("Bash"), None),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn piped_commands_encode() {
        // Piped commands don't match simple first-token patterns
        // "grep" is not in the trivial list, so this encodes
        assert_eq!(
            attention_filter(
                EventType::PostToolUse,
                Some("Bash"),
                Some("grep -r TODO | wc -l")
            ),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn bash_with_semicolons_encodes_safely() {
        // "ls; rm -rf /" — first token is "ls;" (includes semicolon), which
        // doesn't match "ls" in the trivial list. This is correct: compound
        // commands are ambiguous, so we encode them to be safe.
        assert_eq!(
            attention_filter(EventType::PostToolUse, Some("Bash"), Some("ls; rm -rf /")),
            AttentionDecision::Encode
        );
    }

    #[test]
    fn cargo_and_make_commands_encode() {
        for cmd in [
            "cargo test",
            "cargo build --release",
            "make",
            "npm install",
            "pip install flask",
            "docker build .",
            "kubectl apply -f deploy.yaml",
        ] {
            assert_eq!(
                attention_filter(EventType::PostToolUse, Some("Bash"), Some(cmd)),
                AttentionDecision::Encode,
                "Expected '{cmd}' to encode"
            );
        }
    }
}
