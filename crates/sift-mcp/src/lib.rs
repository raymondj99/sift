//! MCP server for sift — exposes search and indexing as MCP tools.
//!
//! This crate provides a thin MCP (Model Context Protocol) layer over sift's
//! existing search pipeline. It communicates via JSON-RPC 2.0 over stdio,
//! allowing any MCP-compatible agent to search indexed content.

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use sift_core::{Config, SearchMode};
use sift_store::{DefaultFullTextStore, HybridSearchEngine, MetadataStore, SimpleVectorStore};
use std::sync::Arc;
use tracing::info;

// ---------------------------------------------------------------------------
// Tool input types
// ---------------------------------------------------------------------------

/// Input parameters for the `sift_search` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    /// Search query (natural language or keywords)
    pub query: String,
    /// Max results to return (1-50)
    pub limit: Option<i32>,
    /// Skip first N results for pagination
    pub offset: Option<i32>,
    /// Search mode: hybrid (semantic+keyword), keyword (BM25 only), vector (embedding only)
    pub mode: Option<String>,
    /// Filter results to files under this path
    pub path: Option<String>,
    /// Filter by file type (e.g., 'rs', 'md', 'pdf')
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    /// Lines of surrounding context to include (0-10)
    pub context: Option<i32>,
}

/// Input parameters for the `sift_search_skills` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchSkillsRequest {
    /// What capability you're looking for (e.g., 'pdf processing', 'code review')
    pub query: String,
    /// Detail level: metadata (~100 tokens), instructions (SKILL.md body), full (body + file listing)
    pub detail: Option<String>,
    /// Max skills to return
    pub limit: Option<i32>,
    /// Where to search: all locations, personal (~/.claude/skills), or project (.claude/skills)
    pub scope: Option<String>,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The sift MCP server. Holds shared state (search engine, metadata store)
/// and dispatches tool calls to the appropriate handler.
pub struct SiftMcpServer {
    tool_router: ToolRouter<Self>,
    config: Config,
    engine: Arc<HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>>,
    metadata: Arc<MetadataStore>,
    #[cfg(feature = "embeddings")]
    embedder: Option<Arc<dyn sift_core::Embedder>>,
}

impl Clone for SiftMcpServer {
    fn clone(&self) -> Self {
        Self {
            tool_router: self.tool_router.clone(),
            config: self.config.clone(),
            engine: self.engine.clone(),
            metadata: self.metadata.clone(),
            #[cfg(feature = "embeddings")]
            embedder: self.embedder.clone(),
        }
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for SiftMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiftMcpServer")
            .field("config_index", &self.config.index_name)
            .finish()
    }
}

#[tool_router]
impl SiftMcpServer {
    /// Create a new MCP server, opening the search engine for the given config.
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let (engine, metadata) = open_engine(&config)?;

        #[cfg(feature = "embeddings")]
        let embedder = load_embedder();

        Ok(Self {
            tool_router: Self::tool_router(),
            config,
            engine: Arc::new(engine),
            metadata: Arc::new(metadata),
            #[cfg(feature = "embeddings")]
            embedder: embedder.map(|e| Arc::new(e) as Arc<dyn sift_core::Embedder>),
        })
    }

    /// Search indexed files using hybrid semantic + keyword search.
    #[tool(
        name = "sift_search",
        description = "Search indexed files using hybrid semantic + keyword search. Returns relevant chunks with file paths, line numbers, and surrounding context. Use this when you need to find code, documentation, or any content by meaning or keywords. Supports 30+ file formats including code, markdown, PDF, Office docs, CSV, JSON, and more.",
        annotations(read_only_hint = true, open_world_hint = true)
    )]
    fn sift_search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let limit = req.limit.unwrap_or(10).clamp(1, 50) as usize;
        let offset = req.offset.unwrap_or(0).max(0) as usize;
        let context_lines = req.context.unwrap_or(2).clamp(0, 10) as usize;

        let mode = match req.mode.as_deref() {
            Some("keyword") => SearchMode::KeywordOnly,
            Some("vector") => SearchMode::VectorOnly,
            _ => SearchMode::Hybrid,
        };

        let (query_vector, effective_mode) = self.embed_query(&req.query, mode);

        // Fetch extra results to account for filtering and offset
        let fetch_k = limit + offset + 10;

        let mut results = self
            .engine
            .search(&query_vector, &req.query, fetch_k, effective_mode)
            .map_err(|e| internal_err(format!("Search failed: {e}")))?;

        // Apply file type filter
        if let Some(ref ft) = req.file_type {
            results.retain(|r| r.file_type == *ft);
        }

        // Apply path filter
        if let Some(ref path_filter) = req.path {
            results.retain(|r| {
                let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                path.starts_with(path_filter) || path.contains(path_filter)
            });
        }

        let total = results.len();
        let has_more = total > offset + limit;

        // Apply pagination
        let page: Vec<_> = results.into_iter().skip(offset).take(limit).collect();

        // Format response
        let result_items: Vec<serde_json::Value> = page
            .iter()
            .map(|r| {
                let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                let lines = format_line_range(r.byte_range, path);
                let snippet = if context_lines > 0 {
                    read_context_snippet(&r.uri, r.byte_range, context_lines)
                        .unwrap_or_else(|| truncate_text(&r.text, 200))
                } else {
                    truncate_text(&r.text, 200)
                };
                serde_json::json!({
                    "path": path,
                    "lines": lines,
                    "score": round2(r.score),
                    "type": r.file_type,
                    "snippet": snippet,
                })
            })
            .collect();

        let mode_str = match effective_mode {
            SearchMode::Hybrid => "hybrid",
            SearchMode::KeywordOnly => "keyword",
            SearchMode::VectorOnly => "vector",
        };

        let response = serde_json::json!({
            "results": result_items,
            "total": total,
            "has_more": has_more,
            "query_mode": mode_str,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Show index status.
    #[tool(
        name = "sift_status",
        description = "Show index status: number of indexed files, total chunks, storage size, file type breakdown, and source directories. Use to verify indexing is complete before searching.",
        annotations(read_only_hint = true)
    )]
    fn sift_status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let mut stats = self
            .metadata
            .stats()
            .map_err(|e| internal_err(format!("Failed to get stats: {e}")))?;

        // Calculate index size on disk
        if let Ok(index_dir) = self.config.index_dir() {
            if index_dir.exists() {
                stats.index_size_bytes = dir_size(&index_dir);
            }
        }

        let sources = self
            .metadata
            .list_sources()
            .map_err(|e| internal_err(format!("Failed to list sources: {e}")))?;

        // Extract unique directory paths
        let mut dirs: Vec<String> = sources
            .iter()
            .filter_map(|(uri, _, _)| {
                let path = uri.strip_prefix("file://")?;
                std::path::Path::new(path)
                    .parent()?
                    .to_str()
                    .map(String::from)
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        dirs.sort();

        let response = serde_json::json!({
            "total_files": stats.total_sources,
            "total_chunks": stats.total_chunks,
            "index_size_bytes": stats.index_size_bytes,
            "index_size": format_bytes(stats.index_size_bytes),
            "file_types": stats.file_type_counts,
            "source_directories": dirs,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Search for agent skills (SKILL.md files).
    #[tool(
        name = "sift_search_skills",
        description = "Search for agent skills (SKILL.md files) by name, description, or capability. Returns skill metadata (name, description) by default for minimal context usage. Use 'detail' parameter to get full skill content. Searches across ~/.claude/skills/, .claude/skills/, and any indexed directories containing SKILL.md files.",
        annotations(read_only_hint = true)
    )]
    fn sift_search_skills(
        &self,
        Parameters(req): Parameters<SearchSkillsRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let limit = req.limit.unwrap_or(5).clamp(1, 20) as usize;
        let detail = req.detail.as_deref().unwrap_or("metadata");
        let scope = req.scope.as_deref().unwrap_or("all");

        let (query_vector, mode) = self.embed_query(&req.query, SearchMode::Hybrid);

        // Search broadly, then filter to SKILL.md files
        let mut results = self
            .engine
            .search(&query_vector, &req.query, limit * 5, mode)
            .map_err(|e| internal_err(format!("Search failed: {e}")))?;

        // Filter to SKILL.md files only
        results.retain(|r| {
            let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
            path.ends_with("SKILL.md")
        });

        // Apply scope filter
        let home = std::env::var("HOME").unwrap_or_default();
        match scope {
            "personal" => {
                let prefix = format!("{home}/.claude/skills");
                results.retain(|r| {
                    let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                    path.starts_with(&prefix)
                });
            }
            "project" => {
                results.retain(|r| {
                    let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                    path.contains("/.claude/skills/")
                        && !path.starts_with(&format!("{home}/.claude"))
                });
            }
            _ => {} // "all" — no filter
        }

        // Deduplicate by file path (multiple chunks from same file)
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.uri.clone()));
        results.truncate(limit);

        // Build response based on detail level
        let skills: Vec<serde_json::Value> = results
            .iter()
            .filter_map(|r| {
                let file_path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                let content = std::fs::read_to_string(file_path).ok()?;
                let (fm, body) = sift_parsers::skill::parse_frontmatter(&content)?;

                let skill_scope = if file_path.contains(&format!("{home}/.claude/skills")) {
                    "personal"
                } else if file_path.contains("/.claude/skills/") {
                    "project"
                } else {
                    "indexed"
                };

                let mut entry = serde_json::json!({
                    "name": fm.name.unwrap_or_else(|| "unknown".into()),
                    "description": fm.description.unwrap_or_default(),
                    "path": file_path,
                    "scope": skill_scope,
                    "score": round2(r.score),
                });

                if detail == "instructions" || detail == "full" {
                    entry["frontmatter"] = serde_json::to_value(&fm.raw).unwrap_or_default();
                    entry["body"] = serde_json::Value::String(body.to_string());
                }

                if detail == "full" {
                    if let Some(parent) = std::path::Path::new(file_path).parent() {
                        let files: Vec<String> = walkdir::WalkDir::new(parent)
                            .max_depth(2)
                            .into_iter()
                            .filter_map(|e| e.ok())
                            .filter(|e| e.file_type().is_file() && e.file_name() != "SKILL.md")
                            .filter_map(|e| {
                                e.path()
                                    .strip_prefix(parent)
                                    .ok()
                                    .map(|p| p.display().to_string())
                            })
                            .collect();
                        entry["files"] = serde_json::json!(files);
                    }
                }

                Some(entry)
            })
            .collect();

        let response = serde_json::json!({
            "skills": skills,
            "total": skills.len(),
            "detail_level": detail,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }
}

#[tool_handler]
impl ServerHandler for SiftMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Sift is a universal file indexing and semantic search engine. \
                 Use sift_status to check what's indexed, sift_search to find content, \
                 and sift_search_skills to discover agent skills."
                .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// Query embedding helper
// ---------------------------------------------------------------------------

impl SiftMcpServer {
    /// Embed a query for vector/hybrid search, falling back to keyword-only
    /// if no embedding model is available.
    #[allow(unused_variables)]
    fn embed_query(&self, query: &str, mode: SearchMode) -> (Vec<f32>, SearchMode) {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                let prefixed = format!("search_query: {query}");
                match embedder.embed(&prefixed) {
                    Ok(vec) => return (vec, mode),
                    Err(e) => {
                        tracing::warn!("Embedding failed: {e}. Falling back to keyword search.");
                    }
                }
            }
        }

        // Fall back to keyword-only with a zero vector
        let fallback = match mode {
            SearchMode::VectorOnly | SearchMode::Hybrid => SearchMode::KeywordOnly,
            other @ SearchMode::KeywordOnly => other,
        };
        (vec![0.0f32; 768], fallback)
    }
}

// ---------------------------------------------------------------------------
// Engine setup (mirrors sift-cli pipeline::open_engine)
// ---------------------------------------------------------------------------

/// Open or create the hybrid search engine for the configured index.
fn open_engine(
    config: &Config,
) -> anyhow::Result<(
    HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    MetadataStore,
)> {
    config.ensure_dirs()?;
    let index_dir = config.index_dir()?;

    #[cfg(feature = "hnsw")]
    let vector_store = SimpleVectorStore::load_or_create(&index_dir)?;
    #[cfg(not(feature = "hnsw"))]
    let vector_store = SimpleVectorStore::load_or_migrate(&index_dir)?;

    #[cfg(feature = "fulltext")]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("tantivy"))?;
    #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("fts5.db"))?;
    #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("bm25.json"))?;

    #[cfg(feature = "sqlite")]
    let metadata_path = index_dir.join("metadata.db");
    #[cfg(not(feature = "sqlite"))]
    let metadata_path = index_dir.join("metadata.json");
    let metadata = MetadataStore::open(&metadata_path)?;

    let engine = HybridSearchEngine::new(vector_store, fulltext_store, config.search.hybrid_alpha);
    info!("MCP server opened index at {}", index_dir.display());

    Ok((engine, metadata))
}

/// Load the ONNX embedding model for query embedding.
#[cfg(feature = "embeddings")]
fn load_embedder() -> Option<sift_embed::OnnxEmbedder> {
    use sift_embed::{models::NOMIC_EMBED_TEXT_V2, ModelManager};

    let manager = ModelManager::new().ok()?;
    manager.init_ort_env();

    let model_def = &NOMIC_EMBED_TEXT_V2;
    if !manager.is_downloaded(model_def.name) {
        info!(
            "Embedding model not available — MCP search will use keyword-only mode. \
             Run `sift models download {}` for semantic search.",
            model_def.name
        );
        return None;
    }

    let model_dir = manager.model_dir(model_def.name);
    match sift_embed::OnnxEmbedder::load(&model_dir, model_def.name, model_def.dimensions) {
        Ok(emb) => {
            info!("Loaded embedding model: {}", model_def.name);
            Some(emb)
        }
        Err(e) => {
            tracing::warn!("Failed to load embedding model: {e}. Using keyword-only.");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Start the MCP server on stdio (stdin/stdout for JSON-RPC 2.0).
///
/// All logging goes to stderr so stdout is reserved for the MCP protocol.
pub async fn run_stdio_server(config: Config) -> anyhow::Result<()> {
    let server = SiftMcpServer::new(config)?;
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server stopped: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Create an internal MCP error.
fn internal_err(msg: String) -> rmcp::ErrorData {
    rmcp::ErrorData::new(
        rmcp::model::ErrorCode::INTERNAL_ERROR,
        msg,
        None::<serde_json::Value>,
    )
}

/// Round a float to 2 decimal places.
fn round2(x: f32) -> f32 {
    (x * 100.0).round() / 100.0
}

/// Compute the total size of a directory on disk.
fn dir_size(path: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Human-readable byte size formatting.
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Truncate text to `max_len` chars, appending "..." if truncated.
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        let end = text
            .char_indices()
            .nth(max_len)
            .map_or(text.len(), |(i, _)| i);
        format!("{}...", &text[..end])
    }
}

/// Convert a byte range to a line range string like "42-58".
fn format_line_range(byte_range: Option<(u64, u64)>, file_path: &str) -> String {
    let Some((start, end)) = byte_range else {
        return String::new();
    };
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return String::new();
    };
    let len = content.len() as u64;
    let start_line = content[..start.min(len) as usize].lines().count();
    let end_line = content[..end.min(len) as usize].lines().count();
    format!("{start_line}-{end_line}")
}

/// Read context lines around a byte range for a richer search snippet.
fn read_context_snippet(
    uri: &str,
    byte_range: Option<(u64, u64)>,
    context_lines: usize,
) -> Option<String> {
    let path = uri.strip_prefix("file://")?;
    let (start_byte, _) = byte_range?;
    let content = std::fs::read_to_string(path).ok()?;

    let lines: Vec<&str> = content.lines().collect();
    let mut offset = 0u64;
    let mut target_line = 0;
    for (i, line) in lines.iter().enumerate() {
        let line_end = offset + line.len() as u64 + 1;
        if offset <= start_byte && start_byte < line_end {
            target_line = i;
            break;
        }
        offset = line_end;
    }

    let start = target_line.saturating_sub(context_lines);
    let end = (target_line + context_lines + 1).min(lines.len());

    let snippet: Vec<String> = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4} {line}", start + i + 1))
        .collect();

    Some(snippet.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_text_short() {
        assert_eq!(truncate_text("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_text_long() {
        let result = truncate_text("hello world this is long", 5);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 9); // 5 chars + "..."
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
    }

    #[test]
    fn test_round2() {
        assert!((round2(0.876) - 0.88).abs() < f32::EPSILON);
        assert!((round2(0.5) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_internal_err() {
        let err = internal_err("test error".into());
        assert_eq!(err.message, "test error");
    }

    // -----------------------------------------------------------------------
    // format_bytes — B, KB, MB, GB ranges
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0B");
    }

    #[test]
    fn test_format_bytes_bytes_range() {
        assert_eq!(format_bytes(1), "1B");
        assert_eq!(format_bytes(1023), "1023B");
    }

    #[test]
    fn test_format_bytes_kb_range() {
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(1536), "1.5KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.0KB");
    }

    #[test]
    fn test_format_bytes_mb_range() {
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
        assert_eq!(format_bytes(5 * 1024 * 1024 + 512 * 1024), "5.5MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 - 1), "1024.0MB");
    }

    #[test]
    fn test_format_bytes_gb_range() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0GB");
    }

    // -----------------------------------------------------------------------
    // truncate_text — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncate_text_empty_string() {
        assert_eq!(truncate_text("", 10), "");
    }

    #[test]
    fn test_truncate_text_exact_limit() {
        assert_eq!(truncate_text("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_text_one_over_limit() {
        let result = truncate_text("abcdef", 5);
        assert_eq!(result, "abcde...");
    }

    #[test]
    fn test_truncate_text_zero_limit() {
        // With limit 0, everything gets truncated
        assert_eq!(truncate_text("hello", 0), "...");
    }

    #[test]
    fn test_truncate_text_unicode_multibyte() {
        // Each emoji is multiple bytes but one char. With limit=2, keep 2 chars.
        let result = truncate_text("\u{1F600}\u{1F601}\u{1F602}", 2);
        assert_eq!(result, "\u{1F600}\u{1F601}...");
    }

    #[test]
    fn test_truncate_text_unicode_exact() {
        // CJK characters: 3 chars, each 3 bytes = 9 bytes total.
        // truncate_text compares text.len() (bytes) with max_len,
        // so limit=9 means no truncation.
        let cjk = "\u{4F60}\u{597D}\u{5417}";
        assert_eq!(cjk.len(), 9); // 3 chars * 3 bytes each
        assert_eq!(truncate_text(cjk, 9), cjk);
        // But limit=2 truncates after 2 chars
        assert_eq!(truncate_text(cjk, 2), "\u{4F60}\u{597D}...");
    }

    // -----------------------------------------------------------------------
    // format_line_range — Some and None byte ranges
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_line_range_none() {
        assert_eq!(format_line_range(None, "anything"), "");
    }

    #[test]
    fn test_format_line_range_missing_file() {
        assert_eq!(
            format_line_range(Some((0, 10)), "/nonexistent/path/to/file.txt"),
            ""
        );
    }

    #[test]
    fn test_format_line_range_with_tempfile() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        // 3 lines: "aaa\nbbb\nccc\n"
        std::fs::write(&file_path, "aaa\nbbb\nccc\n").unwrap();
        let path_str = file_path.to_str().unwrap();

        // Byte 0..3 => within first line (0-indexed: line 0)
        let result = format_line_range(Some((0, 3)), path_str);
        assert_eq!(result, "0-1");

        // Byte 0..8 => spans into line 2 (0-indexed)
        let result = format_line_range(Some((0, 8)), path_str);
        assert_eq!(result, "0-2");

        // Byte range past end of file — clamped
        let result = format_line_range(Some((0, 99999)), path_str);
        assert!(!result.is_empty());
    }

    // -----------------------------------------------------------------------
    // read_context_snippet — real file and missing file
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_context_snippet_missing_file() {
        let result = read_context_snippet("file:///nonexistent/file.rs", Some((0, 10)), 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_no_byte_range() {
        let result = read_context_snippet("file:///some/file.rs", None, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_no_file_prefix() {
        // URI without "file://" prefix should return None (strip_prefix fails)
        let result = read_context_snippet("/some/path", Some((0, 5)), 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_with_tempfile() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("snippet.txt");
        let content = "line one\nline two\nline three\nline four\nline five\n";
        std::fs::write(&file_path, content).unwrap();

        let uri = format!("file://{}", file_path.display());

        // byte 0 => line 0, context_lines=1 => lines 0..2
        let snippet = read_context_snippet(&uri, Some((0, 5)), 1).unwrap();
        assert!(snippet.contains("line one"));
        assert!(snippet.contains("line two"));

        // byte offset into "line three" (byte 18), context_lines=0 => just that line
        let snippet = read_context_snippet(&uri, Some((18, 28)), 0).unwrap();
        assert!(snippet.contains("line three"));
    }

    // -----------------------------------------------------------------------
    // round2 — negative, zero, large numbers
    // -----------------------------------------------------------------------

    #[test]
    fn test_round2_zero() {
        assert!((round2(0.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_round2_negative() {
        assert!((round2(-0.876) - (-0.88)).abs() < f32::EPSILON);
        assert!((round2(-1.005) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_round2_large() {
        assert!((round2(12345.678) - 12345.68).abs() < 0.01);
    }

    #[test]
    fn test_round2_already_rounded() {
        assert!((round2(1.0) - 1.0).abs() < f32::EPSILON);
        assert!((round2(0.01) - 0.01).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // dir_size — temp directory, empty dir, nonexistent dir
    // -----------------------------------------------------------------------

    #[test]
    fn test_dir_size_with_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("b.txt"), "world!").unwrap();
        // "hello" = 5 bytes, "world!" = 6 bytes
        assert_eq!(dir_size(dir.path()), 11);
    }

    #[test]
    fn test_dir_size_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(dir_size(dir.path()), 0);
    }

    #[test]
    fn test_dir_size_nonexistent() {
        let size = dir_size(std::path::Path::new("/nonexistent/directory/xyz"));
        assert_eq!(size, 0);
    }

    #[test]
    fn test_dir_size_nested() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.path().join("top.txt"), "abc").unwrap(); // 3 bytes
        std::fs::write(sub.join("nested.txt"), "defgh").unwrap(); // 5 bytes
        assert_eq!(dir_size(dir.path()), 8);
    }

    // -----------------------------------------------------------------------
    // SearchRequest / SearchSkillsRequest deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_request_minimal() {
        let json = r#"{"query": "find something"}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "find something");
        assert!(req.limit.is_none());
        assert!(req.offset.is_none());
        assert!(req.mode.is_none());
        assert!(req.path.is_none());
        assert!(req.file_type.is_none());
        assert!(req.context.is_none());
    }

    #[test]
    fn test_search_request_full() {
        let json = r#"{
            "query": "hello",
            "limit": 20,
            "offset": 5,
            "mode": "keyword",
            "path": "/src",
            "type": "rs",
            "context": 3
        }"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "hello");
        assert_eq!(req.limit, Some(20));
        assert_eq!(req.offset, Some(5));
        assert_eq!(req.mode.as_deref(), Some("keyword"));
        assert_eq!(req.path.as_deref(), Some("/src"));
        assert_eq!(req.file_type.as_deref(), Some("rs"));
        assert_eq!(req.context, Some(3));
    }

    #[test]
    fn test_search_request_missing_query() {
        let json = r#"{"limit": 5}"#;
        let result = serde_json::from_str::<SearchRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_skills_request_minimal() {
        let json = r#"{"query": "pdf processing"}"#;
        let req: SearchSkillsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "pdf processing");
        assert!(req.detail.is_none());
        assert!(req.limit.is_none());
        assert!(req.scope.is_none());
    }

    #[test]
    fn test_search_skills_request_full() {
        let json = r#"{
            "query": "code review",
            "detail": "full",
            "limit": 10,
            "scope": "personal"
        }"#;
        let req: SearchSkillsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "code review");
        assert_eq!(req.detail.as_deref(), Some("full"));
        assert_eq!(req.limit, Some(10));
        assert_eq!(req.scope.as_deref(), Some("personal"));
    }

    #[test]
    fn test_search_skills_request_missing_query() {
        let json = r#"{"detail": "metadata"}"#;
        let result = serde_json::from_str::<SearchSkillsRequest>(json);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Server integration tests
    // -----------------------------------------------------------------------

    use std::sync::Mutex as StdMutex;

    /// Global lock to serialize tests that modify the HOME environment variable.
    static HOME_LOCK: std::sync::LazyLock<StdMutex<()>> =
        std::sync::LazyLock::new(|| StdMutex::new(()));

    /// RAII guard that restores the `HOME` environment variable on drop.
    struct HomeGuard(Option<String>);

    impl HomeGuard {
        #[allow(unsafe_code)]
        fn set(path: &std::path::Path) -> Self {
            let prev = std::env::var("HOME").ok();
            // SAFETY: we hold HOME_LOCK so no other test is reading HOME concurrently.
            unsafe { std::env::set_var("HOME", path) };
            Self(prev)
        }
    }

    impl Drop for HomeGuard {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => unsafe { std::env::set_var("HOME", v.as_str()) },
                None => unsafe { std::env::remove_var("HOME") },
            }
        }
    }

    /// Helper: create a `SiftMcpServer` backed by a temp directory.
    /// Temporarily sets `$HOME` so that `Config::index_dir()` resolves into the temp dir.
    /// Returns `(server, _tempdir, _home_guard)` — caller must keep both guards alive.
    fn make_test_server() -> (SiftMcpServer, tempfile::TempDir, HomeGuard) {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let guard = HomeGuard::set(tmp.path());
        let config = Config::default();
        let server = SiftMcpServer::new(config).expect("create server");
        (server, tmp, guard)
    }

    // -- Construction, Debug, Clone ----------------------------------------

    #[test]
    fn test_server_construction() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Verify the server was initialized with a functioning config
        let info = server.get_info();
        assert!(info.capabilities.tools.is_some());
    }

    #[test]
    fn test_server_debug() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let debug_str = format!("{:?}", server);
        assert!(debug_str.contains("SiftMcpServer"));
        assert!(debug_str.contains("config_index"));
    }

    #[test]
    fn test_server_clone() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let cloned = server.clone();
        let debug_orig = format!("{:?}", server);
        let debug_clone = format!("{:?}", cloned);
        assert_eq!(debug_orig, debug_clone);
    }

    // -- get_info ----------------------------------------------------------

    #[test]
    fn test_get_info() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let info = server.get_info();
        // ServerInfo should have instructions containing "Sift"
        let instructions = info.instructions.as_deref().unwrap_or("");
        assert!(instructions.contains("Sift"));
        assert!(instructions.contains("sift_search"));
        // Capabilities should have tools enabled
        assert!(info.capabilities.tools.is_some());
    }

    // -- sift_status -------------------------------------------------------

    #[test]
    fn test_sift_status_empty_index() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_status();
        assert!(result.is_ok());
        let call_result = result.unwrap();
        // Extract text content from the result
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["total_files"], 0);
        assert_eq!(parsed["total_chunks"], 0);
    }

    // -- sift_search -------------------------------------------------------

    #[test]
    fn test_sift_search_empty_index() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test query".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["total"], 0);
        assert_eq!(parsed["has_more"], false);
        assert!(parsed["results"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_sift_search_keyword_mode() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search(Parameters(SearchRequest {
            query: "keyword test".to_string(),
            limit: None,
            offset: None,
            mode: Some("keyword".to_string()),
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["query_mode"], "keyword");
    }

    #[test]
    fn test_sift_search_vector_mode_fallback() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Without an embedder, vector mode falls back to keyword
        let result = server.sift_search(Parameters(SearchRequest {
            query: "vector test".to_string(),
            limit: None,
            offset: None,
            mode: Some("vector".to_string()),
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        // Should fall back to keyword since no embedder is available
        assert_eq!(parsed["query_mode"], "keyword");
    }

    #[test]
    fn test_sift_search_hybrid_mode_fallback() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Hybrid also falls back to keyword when no embedder
        let result = server.sift_search(Parameters(SearchRequest {
            query: "hybrid test".to_string(),
            limit: None,
            offset: None,
            mode: Some("hybrid".to_string()),
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["query_mode"], "keyword");
    }

    #[test]
    fn test_sift_search_with_limit() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Test that limit is accepted (clamped to 1-50 range)
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: Some(100), // over max, should clamp to 50
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());

        // Also test limit below minimum
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: Some(-5), // below min, should clamp to 1
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sift_search_with_offset() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: Some(5),
            mode: None,
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());

        // Negative offset should be clamped to 0
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: Some(-3),
            mode: None,
            path: None,
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sift_search_with_file_type_filter() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: None,
            file_type: Some("rs".to_string()),
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        // With empty index, results should be empty
        assert!(parsed["results"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_sift_search_with_path_filter() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: Some("/src".to_string()),
            file_type: None,
            context: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(parsed["results"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_sift_search_with_context() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // context=0 means no surrounding lines
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: Some(0),
        }));
        assert!(result.is_ok());

        // context=10 (max)
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: Some(10),
        }));
        assert!(result.is_ok());

        // context > 10 should clamp to 10
        let result = server.sift_search(Parameters(SearchRequest {
            query: "test".to_string(),
            limit: None,
            offset: None,
            mode: None,
            path: None,
            file_type: None,
            context: Some(99),
        }));
        assert!(result.is_ok());
    }

    // -- sift_search_skills ------------------------------------------------

    #[test]
    fn test_sift_search_skills_empty_index() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "code review".to_string(),
            detail: None,
            limit: None,
            scope: None,
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["total"], 0);
        assert_eq!(parsed["detail_level"], "metadata");
        assert!(parsed["skills"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_sift_search_skills_with_scope_personal() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "test".to_string(),
            detail: None,
            limit: None,
            scope: Some("personal".to_string()),
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["total"], 0);
    }

    #[test]
    fn test_sift_search_skills_with_scope_project() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "test".to_string(),
            detail: None,
            limit: None,
            scope: Some("project".to_string()),
        }));
        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = call_result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map_or("", |t| t.text.as_str());
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(parsed["total"], 0);
    }

    #[test]
    fn test_sift_search_skills_with_scope_all() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "test".to_string(),
            detail: None,
            limit: None,
            scope: Some("all".to_string()),
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sift_search_skills_with_detail_instructions() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "test".to_string(),
            detail: Some("instructions".to_string()),
            limit: None,
            scope: None,
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sift_search_skills_with_detail_full() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        let result = server.sift_search_skills(Parameters(SearchSkillsRequest {
            query: "test".to_string(),
            detail: Some("full".to_string()),
            limit: Some(3),
            scope: None,
        }));
        assert!(result.is_ok());
    }

    // -- embed_query -------------------------------------------------------

    #[test]
    fn test_embed_query_fallback_keyword() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Without embedder, keyword mode stays as keyword
        let (vec, mode) = server.embed_query("test", SearchMode::KeywordOnly);
        assert_eq!(mode, SearchMode::KeywordOnly);
        assert_eq!(vec.len(), 768);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embed_query_fallback_hybrid() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Without embedder, hybrid falls back to keyword
        let (vec, mode) = server.embed_query("test", SearchMode::Hybrid);
        assert_eq!(mode, SearchMode::KeywordOnly);
        assert_eq!(vec.len(), 768);
    }

    #[test]
    fn test_embed_query_fallback_vector() {
        let _lock = HOME_LOCK.lock().unwrap();
        let (server, _tmp, _guard) = make_test_server();
        // Without embedder, vector-only falls back to keyword
        let (vec, mode) = server.embed_query("test", SearchMode::VectorOnly);
        assert_eq!(mode, SearchMode::KeywordOnly);
        assert_eq!(vec.len(), 768);
    }

    // -- open_engine -------------------------------------------------------

    #[test]
    fn test_open_engine_creates_stores() {
        let _lock = HOME_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = HomeGuard::set(tmp.path());
        let config = Config::default();
        let (engine, metadata) = open_engine(&config).expect("open_engine succeeds");

        // Verify the index directory was created
        let index_dir = config.index_dir().expect("index_dir");
        assert!(index_dir.exists());

        // Engine and metadata should be functional on an empty index
        let stats = metadata.stats().expect("stats");
        assert_eq!(stats.total_sources, 0);
        assert_eq!(stats.total_chunks, 0);

        // Search should return empty results
        let results = engine
            .search(&vec![0.0f32; 768], "test", 10, SearchMode::KeywordOnly)
            .expect("search");
        assert!(results.is_empty());
    }
}
