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
}
