//! MCP server for sift — exposes search and indexing as MCP tools.
//!
//! This crate provides a thin MCP (Model Context Protocol) layer over sift's
//! existing search pipeline. It communicates via JSON-RPC 2.0 over stdio,
//! allowing any MCP-compatible agent to search indexed content.
//!
//! Repeated queries are served from an LRU cache (50 entries, 60 s TTL),
//! and all tool inputs are validated with descriptive error messages.

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use sift_core::{format_bytes, Config, SearchMode};
use sift_store::{
    CachedSearchEngine, DefaultFullTextStore, FullTextStore as _, HybridSearchEngine,
    MetadataStore, SimpleVectorStore, VectorIndex as _,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

// ---------------------------------------------------------------------------
// Enum types for JSON Schema generation
// ---------------------------------------------------------------------------

/// Search mode controlling how results are ranked.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchModeParam {
    /// Combine vector similarity and BM25 keyword search (default)
    Hybrid,
    /// BM25 full-text keyword search only — no embedding model needed
    Keyword,
    /// Pure cosine similarity vector search — requires embedding model
    Vector,
}

/// Detail level for skill search results.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DetailLevel {
    /// Name and description only (~100 tokens, default)
    Metadata,
    /// SKILL.md frontmatter + body content
    Instructions,
    /// Full body + directory file listing
    Full,
}

/// Scope for skill search — where to look for SKILL.md files.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SkillScope {
    /// Search all indexed locations (default)
    All,
    /// Only ~/.claude/skills/
    Personal,
    /// Only project-level .claude/skills/
    Project,
}

/// Content type hint for indexed text.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContentTypeParam {
    /// Plain text or prose (default)
    Text,
    /// Source code
    Code,
    /// Structured data (JSON, CSV, etc.)
    Data,
}

/// Entity type for the memory knowledge graph.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum EntityTypeParam {
    /// A person (user, teammate, etc.)
    Person,
    /// A software project or repository
    Project,
    /// An abstract concept, pattern, or idea (default)
    Concept,
    /// A tool, library, or framework
    Tool,
    /// A user preference or setting
    Preference,
    /// A standalone fact or observation
    Fact,
    /// A dated event (release, incident, meeting, etc.)
    Event,
    /// A physical or network location
    Location,
    /// A company or team
    Organization,
}

impl EntityTypeParam {
    fn to_memory_type(&self) -> sift_memory::EntityType {
        match self {
            Self::Person => sift_memory::EntityType::Person,
            Self::Project => sift_memory::EntityType::Project,
            Self::Concept => sift_memory::EntityType::Concept,
            Self::Tool => sift_memory::EntityType::Tool,
            Self::Preference => sift_memory::EntityType::Preference,
            Self::Fact => sift_memory::EntityType::Fact,
            Self::Event => sift_memory::EntityType::Event,
            Self::Location => sift_memory::EntityType::Location,
            Self::Organization => sift_memory::EntityType::Organization,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool input types
// ---------------------------------------------------------------------------

/// Input parameters for the `sift_search` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    /// Search query — natural language or keywords (max 10,000 chars)
    pub query: String,
    /// Max results to return (default: 10, range: 1–50)
    pub limit: Option<i32>,
    /// Skip first N results for pagination (default: 0)
    pub offset: Option<i32>,
    /// Search mode (default: hybrid)
    pub mode: Option<SearchModeParam>,
    /// Filter results to files under this path (must not contain '..')
    pub path: Option<String>,
    /// Filter by file extension (e.g., 'rs', 'md', 'pdf')
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    /// Lines of surrounding source context to include (default: 2, range: 0–10)
    pub context: Option<i32>,
}

/// Input parameters for the `sift_search_skills` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchSkillsRequest {
    /// What capability you're looking for (e.g., 'pdf processing', 'code review'; max 10,000 chars)
    pub query: String,
    /// How much detail to return (default: metadata)
    pub detail: Option<DetailLevel>,
    /// Max skills to return (default: 5, range: 1–20)
    pub limit: Option<i32>,
    /// Where to search for SKILL.md files (default: all)
    pub scope: Option<SkillScope>,
}

/// Input parameters for the `sift_index_text` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexTextRequest {
    /// Text content to index (max 100,000 chars)
    pub text: String,
    /// URI identifier for this content (e.g., 'memory://facts/my-fact').
    /// If omitted, a unique memory:// URI is auto-generated.
    /// Must not contain '..' (path traversal).
    pub uri: Option<String>,
    /// Content type hint (default: text)
    pub content_type: Option<ContentTypeParam>,
    /// File type hint for search filtering (default: 'txt'; e.g., 'md', 'json')
    pub file_type: Option<String>,
    /// Optional title/label for this content
    pub title: Option<String>,
}

/// Input parameters for the `sift_delete` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteRequest {
    /// Exact URI of the content to delete (e.g., 'memory://agent/my-fact' or 'file:///path/to/file.txt')
    pub uri: String,
}

/// Input parameters for the `sift_list_sources` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListSourcesRequest {
    /// Filter sources to those matching this path or URI substring
    pub path: Option<String>,
    /// Max sources to return (default: 50, range: 1–500)
    pub limit: Option<i32>,
}

// ---------------------------------------------------------------------------
// Memory tool input types
// ---------------------------------------------------------------------------

/// Input parameters for the `sift_remember` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberRequest {
    /// Entity name (e.g., 'Raymond', 'sift project', 'Rust')
    pub entity: String,
    /// Entity type (default: concept)
    pub entity_type: Option<EntityTypeParam>,
    /// List of facts/observations about the entity (at least one required)
    pub observations: Vec<String>,
    /// Relationships to other entities
    pub relations: Option<Vec<RelationInput>>,
    /// Origin identifier (default: 'mcp'; e.g., session ID)
    pub source: Option<String>,
}

/// A directed relationship from one entity to another.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RelationInput {
    /// Target entity name
    pub to: String,
    /// Relationship type in active voice (e.g., 'maintains', 'prefers', 'works_on')
    #[serde(rename = "type")]
    pub relation_type: String,
}

/// Input parameters for the `sift_recall` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallRequest {
    /// What to search for in memory (natural language)
    pub query: String,
    /// Max results to return (default: 10, range: 1–50)
    pub limit: Option<i32>,
    /// Filter results to a specific entity type
    pub entity_type: Option<EntityTypeParam>,
    /// Only return memories from this source/session
    pub source: Option<String>,
    /// Minimum confidence threshold (range: 0.0–1.0)
    pub min_confidence: Option<f32>,
    /// Filter by memory tier: "episodic", "semantic", or "procedural"
    pub memory_tier: Option<String>,
    /// Only return memories about these entities (case-insensitive name match)
    pub entity_names: Option<Vec<String>>,
}

/// Input parameters for the `sift_forget` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetRequest {
    /// Observation ID to invalidate (from sift_recall results)
    pub observation_id: String,
}

/// Input parameters for the `sift_forget_entity` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetEntityRequest {
    /// Entity name to delete (exact match)
    pub entity: String,
}

/// Input parameters for the `sift_list_entities` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListEntitiesRequest {
    /// Filter by entity type (optional)
    pub entity_type: Option<EntityTypeParam>,
    /// Max entities to return (default: 20, range: 1–100)
    pub limit: Option<i32>,
    /// Skip first N entities for pagination (default: 0)
    pub offset: Option<i32>,
}

/// Input parameters for the `sift_get_entity` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetEntityRequest {
    /// Entity name to look up (exact match)
    pub entity: String,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The sift MCP server. Holds shared state (LRU-cached search engine,
/// metadata store) and dispatches tool calls to the appropriate handler.
///
/// Search results are cached (50-entry LRU, 60 s TTL) so repeated agent
/// queries avoid redundant work. All tool inputs are validated at the
/// boundary before reaching the engine.
pub struct SiftMcpServer {
    tool_router: ToolRouter<Self>,
    config: Config,
    engine: Arc<CachedSearchEngine<SimpleVectorStore, DefaultFullTextStore>>,
    metadata: Arc<MetadataStore>,
    #[cfg(feature = "embeddings")]
    embedder: Option<Arc<dyn sift_core::Embedder>>,
    memory: Option<Arc<sift_memory::MemoryStore>>,
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
            memory: self.memory.clone(),
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
        let embedder = sift_embed::load_embedder(&config, None);

        // Single-line startup banner on stderr (stdout is reserved for JSON-RPC).
        // Gives users a quick signal of whether semantic search is active without
        // having to parse RUST_LOG output.
        #[cfg(feature = "embeddings")]
        {
            use sift_core::Embedder as _;
            match embedder.as_ref() {
                Some(e) => eprintln!("sift mcp: embeddings active (model: {})", e.model_name()),
                None => eprintln!(
                    "sift mcp: keyword-only mode — run `sift models download` for semantic search"
                ),
            }
        }
        #[cfg(not(feature = "embeddings"))]
        eprintln!("sift mcp: keyword-only mode (build without `embeddings` feature)");

        // Wrap in a caching layer so repeated MCP queries hit an LRU cache.
        // 50-entry LRU with 60s TTL balances memory usage with agentic hit rates.
        let cached = CachedSearchEngine::new(engine, 50, Duration::from_secs(60));

        #[cfg(feature = "embeddings")]
        let embedder: Option<Arc<dyn sift_core::Embedder>> =
            embedder.map(|e| Arc::new(e) as Arc<dyn sift_core::Embedder>);

        let memory = {
            let memory_dir = config.index_dir().ok().map(|d| d.join("memory"));
            memory_dir.and_then(|dir| match sift_memory::MemoryStore::open(&dir) {
                Ok(store) => {
                    #[cfg(feature = "embeddings")]
                    let store = if let Some(ref emb) = embedder {
                        store.with_embedder(Arc::clone(emb))
                    } else {
                        store
                    };
                    info!("Memory store opened at {}", dir.display());
                    Some(Arc::new(store))
                }
                Err(e) => {
                    tracing::warn!("Failed to open memory store: {e}. Memory tools disabled.");
                    None
                }
            })
        };

        // Rebuild stale memory vector index in background so MCP startup
        // isn't blocked — the server can respond to initialize immediately.
        #[cfg(feature = "embeddings")]
        if let Some(ref mem) = memory {
            let mem = Arc::clone(mem);
            std::thread::spawn(move || {
                mem.rebuild_if_stale();
            });
        }

        Ok(Self {
            tool_router: Self::tool_router(),
            config,
            engine: Arc::new(cached),
            metadata: Arc::new(metadata),
            #[cfg(feature = "embeddings")]
            embedder,
            memory,
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
        // Input validation
        if req.query.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Query must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }
        if req.query.len() > 10_000 {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Query too long ({} chars, max 10000)", req.query.len()),
                None::<serde_json::Value>,
            ));
        }
        if let Some(ref path_filter) = req.path {
            if path_filter.contains("..") {
                return Err(rmcp::ErrorData::new(
                    rmcp::model::ErrorCode::INVALID_PARAMS,
                    "Path filter must not contain '..' (path traversal)".to_string(),
                    None::<serde_json::Value>,
                ));
            }
        }

        let limit = req.limit.unwrap_or(10).clamp(1, 50) as usize;
        let offset = req.offset.unwrap_or(0).max(0) as usize;
        let context_lines = req.context.unwrap_or(2).clamp(0, 10) as usize;

        let mode = match req.mode {
            Some(SearchModeParam::Keyword) => SearchMode::KeywordOnly,
            Some(SearchModeParam::Vector) => SearchMode::VectorOnly,
            Some(SearchModeParam::Hybrid) | None => SearchMode::Hybrid,
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

        // Cache file reads: multiple results from the same source file share
        // a single disk read instead of one per result.
        let mut file_cache: std::collections::HashMap<&str, Option<String>> =
            std::collections::HashMap::new();
        for r in &page {
            let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
            file_cache
                .entry(path)
                .or_insert_with(|| std::fs::read_to_string(path).ok());
        }

        let result_items: Vec<serde_json::Value> = page
            .iter()
            .map(|r| {
                let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                let file_content = file_cache.get(path).and_then(|opt| opt.as_deref());

                let lines = format_line_range(r.byte_range, file_content);
                let snippet = if context_lines > 0 {
                    read_context_snippet(r.byte_range, file_content, context_lines)
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
        // Input validation
        if req.query.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Query must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }
        if req.query.len() > 10_000 {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Query too long ({} chars, max 10000)", req.query.len()),
                None::<serde_json::Value>,
            ));
        }

        let limit = req.limit.unwrap_or(5).clamp(1, 20) as usize;

        let detail = req.detail.as_ref().map_or("metadata", |d| match d {
            DetailLevel::Metadata => "metadata",
            DetailLevel::Instructions => "instructions",
            DetailLevel::Full => "full",
        });

        let scope = req.scope.as_ref().map_or("all", |s| match s {
            SkillScope::All => "all",
            SkillScope::Personal => "personal",
            SkillScope::Project => "project",
        });

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

    /// Index arbitrary text directly into the search engine.
    #[tool(
        name = "sift_index_text",
        description = "Store text directly in the search index. Use this to persist notes, facts, agent memory, or any text content that should be searchable later. Supports custom URIs (e.g., 'memory://facts/user-preferences') for organizing stored content. Content is immediately searchable via sift_search.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_index_text(
        &self,
        Parameters(req): Parameters<IndexTextRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        if req.text.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Text must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }
        if req.text.len() > 100_000 {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Text too long ({} chars, max 100000)", req.text.len()),
                None::<serde_json::Value>,
            ));
        }
        if let Some(ref uri) = req.uri {
            if uri.contains("..") {
                return Err(rmcp::ErrorData::new(
                    rmcp::model::ErrorCode::INVALID_PARAMS,
                    "URI must not contain '..' (path traversal)".to_string(),
                    None::<serde_json::Value>,
                ));
            }
        }

        let content_type = match req.content_type {
            Some(ContentTypeParam::Code) => sift_core::ContentType::Code,
            Some(ContentTypeParam::Data) => sift_core::ContentType::Data,
            Some(ContentTypeParam::Text) | None => sift_core::ContentType::Text,
        };

        let uri = req.uri.unwrap_or_else(|| {
            format!(
                "memory://agent/{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            )
        });
        let file_type = req.file_type.unwrap_or_else(|| "txt".to_string());

        // Embed the text if an embedding model is available
        let vector = self.embed_text_for_index(&req.text);

        let chunk = sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: req.text.clone(),
                source_uri: uri.clone(),
                chunk_index: 0,
                content_type,
                file_type: file_type.clone(),
                title: req.title.clone(),
                language: None,
                byte_range: None,
            },
            vector,
        };

        self.engine
            .insert(&[chunk])
            .map_err(|e| internal_err(format!("Insert failed: {e}")))?;

        // Mirror the chunk into the metadata store so sift_status and
        // sift_list_sources reflect the new URI immediately. Without this
        // the engine and metadata stores drift apart.
        let content_hash = *blake3::hash(req.text.as_bytes()).as_bytes();
        self.metadata
            .upsert_source(
                &uri,
                &content_hash,
                req.text.len() as u64,
                &file_type,
                None,
                1,
            )
            .map_err(|e| internal_err(format!("Metadata upsert failed: {e}")))?;

        // Persist the vector store to disk
        self.save_stores();

        let response = serde_json::json!({
            "status": "indexed",
            "uri": uri,
            "content_type": content_type.to_string(),
            "file_type": file_type,
            "text_length": req.text.len(),
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Delete indexed content by URI.
    #[tool(
        name = "sift_delete",
        description = "Remove content from the search index by its URI. Use this to delete previously indexed text, notes, or memory entries. The URI must be an exact match (e.g., 'memory://agent/my-fact' or 'file:///path/to/file.txt').",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_delete(
        &self,
        Parameters(req): Parameters<DeleteRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        if req.uri.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "URI must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }

        let deleted = self
            .engine
            .delete_by_uri(&req.uri)
            .map_err(|e| internal_err(format!("Delete failed: {e}")))?;

        // Keep the metadata store in lockstep with the engine. Ignore the
        // boolean — if the URI was only in metadata or only in the engine
        // (e.g. partial prior failure) we still want to converge on "gone".
        let _ = self
            .metadata
            .remove_source(&req.uri)
            .map_err(|e| internal_err(format!("Metadata remove failed: {e}")))?;

        // Persist after deletion
        self.save_stores();

        let response = serde_json::json!({
            "status": if deleted > 0 { "deleted" } else { "not_found" },
            "uri": req.uri,
            "chunks_removed": deleted,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// List indexed sources with optional filtering.
    #[tool(
        name = "sift_list_sources",
        description = "List files and content currently in the search index. Shows URI, file type, and chunk count for each source. Use 'path' to filter to a specific directory or URI prefix. Useful for understanding what's indexed before searching.",
        annotations(read_only_hint = true)
    )]
    fn sift_list_sources(
        &self,
        Parameters(req): Parameters<ListSourcesRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let limit = req.limit.unwrap_or(50).clamp(1, 500) as usize;

        let sources = self
            .metadata
            .list_sources()
            .map_err(|e| internal_err(format!("Failed to list sources: {e}")))?;

        let mut filtered: Vec<_> = if let Some(ref path_filter) = req.path {
            sources
                .into_iter()
                .filter(|(uri, _, _)| {
                    let path = uri.strip_prefix("file://").unwrap_or(uri);
                    path.contains(path_filter.as_str()) || uri.contains(path_filter.as_str())
                })
                .collect()
        } else {
            sources
        };

        let total = filtered.len();
        filtered.truncate(limit);

        let items: Vec<serde_json::Value> = filtered
            .iter()
            .map(|(uri, file_type, chunk_count)| {
                let path = uri.strip_prefix("file://").unwrap_or(uri);
                serde_json::json!({
                    "path": path,
                    "type": file_type,
                    "chunks": chunk_count,
                })
            })
            .collect();

        let response = serde_json::json!({
            "sources": items,
            "total": total,
            "showing": items.len(),
            "has_more": total > items.len(),
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    // -----------------------------------------------------------------------
    // Memory tools (behind `memory` feature)
    // -----------------------------------------------------------------------

    /// Store facts about an entity in persistent memory.
    #[tool(
        name = "sift_remember",
        description = "Store facts about an entity in persistent memory. Creates or updates the entity and adds observations (facts) about it. Optionally creates relationships to other entities. Use this to persist knowledge across sessions — e.g., user preferences, project decisions, learned patterns.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_remember(
        &self,
        Parameters(req): Parameters<RememberRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        if req.entity.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Entity name must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }
        if req.observations.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "At least one observation is required".to_string(),
                None::<serde_json::Value>,
            ));
        }

        let entity_type = req.entity_type.as_ref().map_or(
            sift_memory::EntityType::Concept,
            EntityTypeParam::to_memory_type,
        );
        let source = req.source.as_deref().unwrap_or("mcp");

        let entity_id = memory
            .save_entity(&req.entity, entity_type, 1.0, source)
            .map_err(|e| internal_err(format!("Failed to save entity: {e}")))?;

        let mut obs_ids = Vec::new();
        let mut all_conflicts = Vec::new();
        for obs_text in &req.observations {
            if obs_text.is_empty() {
                continue;
            }

            // Detect conflicts BEFORE adding the observation
            if let Ok(conflicts) = memory.detect_conflicts(&entity_id, obs_text, 0.15) {
                for c in conflicts {
                    all_conflicts.push(serde_json::json!({
                        "observation_id": c.observation_id,
                        "existing_fact": c.existing_content,
                        "new_fact": obs_text,
                        "conflict_score": (c.conflict_score * 100.0).round() / 100.0,
                        "hint": "Consider using sift_forget to invalidate the old observation if it is outdated"
                    }));
                }
            }

            let obs_id = memory
                .add_observation(&entity_id, obs_text, 1.0, source)
                .map_err(|e| internal_err(format!("Failed to add observation: {e}")))?;
            obs_ids.push(obs_id);
        }

        let mut rel_ids = Vec::new();
        if let Some(relations) = &req.relations {
            for rel in relations {
                // Ensure target entity exists
                let target_id = memory
                    .save_entity(&rel.to, sift_memory::EntityType::Concept, 1.0, source)
                    .map_err(|e| internal_err(format!("Failed to save related entity: {e}")))?;
                let rel_id = memory
                    .add_relation(&entity_id, &target_id, &rel.relation_type, 1.0, source)
                    .map_err(|e| internal_err(format!("Failed to add relation: {e}")))?;
                rel_ids.push(rel_id);
            }
        }

        // Persist to disk
        let _ = memory.save();

        let mut response = serde_json::json!({
            "status": "remembered",
            "entity_id": entity_id,
            "entity": req.entity,
            "observations_added": obs_ids.len(),
            "relations_added": rel_ids.len(),
        });

        if !all_conflicts.is_empty() {
            response["potential_conflicts"] = serde_json::json!(all_conflicts);
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Recall facts from memory by semantic search.
    #[tool(
        name = "sift_recall",
        description = "Search persistent memory for facts about entities. Uses hybrid semantic + keyword search over stored observations. Returns facts with entity names, types, and relevance scores. Filters by entity type, source, confidence, memory tier (episodic/semantic/procedural), or specific entity names (for high-precision entity-scoped recall).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    fn sift_recall(
        &self,
        Parameters(req): Parameters<RecallRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        if req.query.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Query must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }

        let limit = req.limit.unwrap_or(10).clamp(1, 50) as usize;

        let filters = sift_memory::RecallFilters {
            entity_type: req
                .entity_type
                .as_ref()
                .map(EntityTypeParam::to_memory_type),
            source: req.source,
            min_confidence: req.min_confidence,
            memory_tier: req
                .memory_tier
                .as_deref()
                .and_then(sift_memory::MemoryTier::parse),
            entity_names: req.entity_names,
            ..Default::default()
        };

        let results = memory
            .recall(&req.query, limit, &filters)
            .map_err(|e| internal_err(format!("Recall failed: {e}")))?;

        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "entity": r.entity_name,
                    "entity_type": r.entity_type.as_str(),
                    "fact": r.observation.content,
                    "observation_id": r.observation.id,
                    "score": round2(r.score),
                    "confidence": r.observation.confidence,
                    "observed_at": r.observation.observed_at,
                    "source": r.observation.source,
                })
            })
            .collect();

        let response = serde_json::json!({
            "memories": items,
            "total": items.len(),
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Invalidate a memory observation (soft delete).
    #[tool(
        name = "sift_forget",
        description = "Invalidate a specific memory observation by ID. This is a soft delete — the observation is marked with an end timestamp but not physically removed, preserving the audit trail. Use observation_id from sift_recall results.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_forget(
        &self,
        Parameters(req): Parameters<ForgetRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        if req.observation_id.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Observation ID must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }

        let invalidated = memory
            .invalidate_observation(&req.observation_id)
            .map_err(|e| internal_err(format!("Forget failed: {e}")))?;

        let _ = memory.save();

        let response = serde_json::json!({
            "status": if invalidated { "forgotten" } else { "not_found" },
            "observation_id": req.observation_id,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Delete an entire entity and all its observations and relations.
    #[tool(
        name = "sift_forget_entity",
        description = "Delete an entire entity by name, including all its observations, relationships, and search index entries. This is a hard delete — the entity is permanently removed. Use sift_list_entities to find entity names.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_forget_entity(
        &self,
        Parameters(req): Parameters<ForgetEntityRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        if req.entity.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Entity name must not be empty".to_string(),
                None::<serde_json::Value>,
            ));
        }

        let result = memory.delete_entity(&req.entity);
        let _ = memory.save();

        let response = match result {
            Ok(obs_removed) => serde_json::json!({
                "status": "deleted",
                "entity": req.entity,
                "observations_removed": obs_removed,
            }),
            Err(sift_memory::MemoryError::EntityNotFound(_)) => serde_json::json!({
                "status": "not_found",
                "entity": req.entity,
            }),
            Err(e) => return Err(internal_err(format!("Delete failed: {e}"))),
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Remove all entities with no active observations and no relations.
    #[tool(
        name = "sift_prune",
        description = "Remove all entities that have zero active observations and no relationships. Cleans up ghost entities left behind by sift_forget. Returns the list of pruned entity names.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_prune(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        let pruned = memory
            .prune_entities()
            .map_err(|e| internal_err(format!("Prune failed: {e}")))?;

        let _ = memory.save();

        let response = serde_json::json!({
            "status": "pruned",
            "entities_removed": pruned.len(),
            "removed": pruned,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Run the Cortex consolidation pipeline.
    #[tool(
        name = "sift_consolidate",
        description = "Run the Cortex memory consolidation pipeline. Processes pending episodes into entities/observations, deduplicates, promotes episodic→semantic memories, extracts skills from repeated patterns, and prunes low-utility memories. Returns a detailed report of what changed.",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    fn sift_consolidate(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        let memory_dir = self
            .config
            .index_dir()
            .map(|d| d.join("memory"))
            .map_err(|e| internal_err(format!("Cannot determine memory dir: {e}")))?;

        let episodes = sift_memory::episodes::EpisodeStore::open(&memory_dir)
            .map_err(|e| internal_err(format!("Cannot open episode store: {e}")))?;

        let consolidation_config = sift_memory::ConsolidationConfig::from(&self.config.memory);

        let report =
            sift_memory::consolidation::run_consolidation(memory, &episodes, &consolidation_config)
                .map_err(|e| internal_err(format!("Consolidation failed: {e}")))?;

        let _ = memory.save();

        let response = serde_json::json!({
            "status": "completed",
            "episodes_processed": report.episodes_processed,
            "episodes_skipped": report.episodes_skipped,
            "episodes_deferred": report.episodes_deferred,
            "entities_created": report.entities_created,
            "observations_created": report.observations_created,
            "duplicates_merged": report.duplicates_merged,
            "contradictions_found": report.contradictions_found,
            "promotions": report.promotions,
            "skills_created": report.skills_created,
            "skills_updated": report.skills_updated,
            "observations_pruned": report.observations_pruned,
            "entities_pruned": report.entities_pruned,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Show memory store statistics.
    #[tool(
        name = "sift_memory_status",
        description = "Show memory store statistics: total entities, observations, relations, entity type breakdown, memory tier counts, episode stats, and oldest/newest memories.",
        annotations(read_only_hint = true)
    )]
    fn sift_memory_status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        let stats = memory
            .stats()
            .map_err(|e| internal_err(format!("Failed to get memory stats: {e}")))?;

        let response = serde_json::json!({
            "total_entities": stats.total_entities,
            "total_observations": stats.total_observations,
            "total_relations": stats.total_relations,
            "entity_types": stats.entity_type_counts,
            "memory_tiers": stats.tier_counts,
            "pending_episodes": stats.pending_episodes,
            "total_episodes": stats.total_episodes,
            "last_consolidation": stats.last_consolidation,
            "oldest_observation": stats.oldest_observation,
            "newest_observation": stats.newest_observation,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// List all entities stored in memory.
    #[tool(
        name = "sift_list_entities",
        description = "List all entities stored in memory, with optional type filtering and pagination. Returns entity names, types, and observation counts. Use this to browse what's in memory without needing a search query.",
        annotations(read_only_hint = true)
    )]
    fn sift_list_entities(
        &self,
        Parameters(req): Parameters<ListEntitiesRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        let limit = req.limit.unwrap_or(20).clamp(1, 100) as usize;
        let offset = req.offset.unwrap_or(0).max(0) as usize;
        let entity_type = req
            .entity_type
            .as_ref()
            .map(EntityTypeParam::to_memory_type);

        let entities = memory
            .list_entities(entity_type, limit, offset)
            .map_err(|e| internal_err(format!("Failed to list entities: {e}")))?;

        let items: Vec<serde_json::Value> = entities
            .iter()
            .map(|(e, obs_count)| {
                serde_json::json!({
                    "name": e.name,
                    "entity_type": e.entity_type.as_str(),
                    "confidence": e.confidence,
                    "observations": obs_count,
                    "source": e.source,
                    "updated_at": e.updated_at,
                })
            })
            .collect();

        let response = serde_json::json!({
            "entities": items,
            "showing": items.len(),
            "offset": offset,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).map_err(|e| internal_err(e.to_string()))?,
        )]))
    }

    /// Get all facts about a specific entity.
    #[tool(
        name = "sift_get_entity",
        description = "Get all stored facts (observations) and relationships for a named entity. Returns entity metadata plus all active observations and relations. Use when you know the entity name (from sift_list_entities or sift_recall) and want the full picture.",
        annotations(read_only_hint = true)
    )]
    fn sift_get_entity(
        &self,
        Parameters(req): Parameters<GetEntityRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let memory = self
            .memory
            .as_ref()
            .ok_or_else(|| internal_err("Memory store not available".to_string()))?;

        if req.entity.is_empty() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "Entity name must not be empty",
                None::<serde_json::Value>,
            ));
        }

        let entity = memory
            .get_entity(&req.entity)
            .map_err(|e| internal_err(format!("Failed to get entity: {e}")))?;

        let Some(entity) = entity else {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "not_found",
                    "entity": req.entity,
                }))
                .map_err(|e| internal_err(e.to_string()))?,
            )]));
        };

        let observations = memory
            .get_entity_observations(&entity.id)
            .map_err(|e| internal_err(format!("Failed to get observations: {e}")))?;
        let relations = memory
            .get_entity_relations(&entity.id)
            .map_err(|e| internal_err(format!("Failed to get relations: {e}")))?;

        // Batch-resolve all related entity names in one pass.
        let related_ids: Vec<&str> = relations
            .iter()
            .map(|r| {
                if r.from_entity == entity.id {
                    r.to_entity.as_str()
                } else {
                    r.from_entity.as_str()
                }
            })
            .collect();
        let mut name_map: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for id in &related_ids {
            if !name_map.contains_key(*id) {
                if let Ok(Some(e)) = memory.get_entity_by_id(id) {
                    name_map.insert(e.id, e.name);
                }
            }
        }

        let rel_items: Vec<serde_json::Value> = relations
            .iter()
            .filter_map(|r| {
                let target_id = if r.from_entity == entity.id {
                    &r.to_entity
                } else {
                    &r.from_entity
                };
                let target_name = name_map.get(target_id)?;
                Some(serde_json::json!({
                    "type": r.relation_type,
                    "target": target_name,
                    "direction": if r.from_entity == entity.id { "outgoing" } else { "incoming" },
                    "weight": r.weight,
                }))
            })
            .collect();

        let obs_items: Vec<serde_json::Value> = observations
            .iter()
            .map(|o| {
                serde_json::json!({
                    "observation_id": o.id,
                    "fact": o.content,
                    "confidence": o.confidence,
                    "observed_at": o.observed_at,
                    "source": o.source,
                })
            })
            .collect();

        let response = serde_json::json!({
            "entity": entity.name,
            "entity_type": entity.entity_type.as_str(),
            "confidence": entity.confidence,
            "observations": obs_items,
            "relations": rel_items,
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
            "Sift is a local semantic search engine indexing 30+ file formats.\n\n\
             SEARCH TOOLS:\n\
             - sift_status: Call first to see what's indexed (directories, file types, chunk count).\n\
             - sift_search: Hybrid semantic+keyword search. Use 'mode' (hybrid/keyword/vector), \
               'path' to scope, 'type' to filter extensions, 'context' (0-10) for surrounding lines.\n\
             - sift_search_skills: Find SKILL.md agent capabilities. Use 'detail' \
               (metadata/instructions/full) to control verbosity.\n\
             - sift_list_sources: Browse indexed files with optional path filtering.\n\n\
             WRITE TOOLS:\n\
             - sift_index_text: Store text directly in the index. \
               Supports custom URIs like 'memory://...' for organizing content.\n\
             - sift_delete: Remove content from the index by exact URI.\n\n\
             MEMORY TOOLS (entity-based knowledge graph):\n\
             - sift_remember: Store facts about named entities with types, observations, \
               and relationships. Use for user preferences, project decisions, learned patterns.\n\
             - sift_recall: Search memory by natural language. Filter by entity_type or source. \
               Returns scored facts with observation IDs.\n\
             - sift_list_entities: Browse all entities in memory with optional type filter and \
               pagination. Shows observation counts. Use to discover what's stored.\n\
             - sift_get_entity: Get all facts and relationships for a named entity. Use after \
               sift_list_entities or sift_recall to see the full picture.\n\
             - sift_forget: Soft-delete a specific observation by ID (from sift_recall results). \
               Preserves audit trail.\n\
             - sift_forget_entity: Hard-delete an entire entity and all its observations \
               and relationships.\n\
             - sift_prune: Remove all entities with zero active observations and no \
               relationships. Cleans up ghost entities left by sift_forget.\n\
             - sift_memory_status: Show memory statistics (entity/observation/relation counts).\n\n\
             WORKFLOW:\n\
             1. Start with sift_status to understand the index.\n\
             2. Use hybrid mode (default) for conceptual queries, keyword for exact matches.\n\
             3. Use sift_remember/sift_recall to persist and retrieve knowledge across sessions.\n\
             4. Use sift_index_text for raw text storage; use sift_remember for structured facts."
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
    #[allow(unused_variables, clippy::unused_self)]
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

    /// Embed text for indexing (using the "search_document:" prefix for Nomic).
    /// Returns a zero vector if no embedder is available — the text will still
    /// be keyword-searchable via FTS.
    #[allow(unused_variables, clippy::unused_self)]
    fn embed_text_for_index(&self, text: &str) -> Vec<f32> {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref embedder) = self.embedder {
                let prefixed = format!("search_document: {text}");
                match embedder.embed(&prefixed) {
                    Ok(vec) => return vec,
                    Err(e) => {
                        tracing::warn!(
                            "Embedding failed for index: {e}. Text will be keyword-only."
                        );
                    }
                }
            }
        }
        vec![0.0f32; 768]
    }

    /// Persist vector and fulltext stores to disk after mutations.
    /// Logs warnings on failure but does not propagate errors — the in-memory
    /// state is still consistent and will be saved on next success.
    fn save_stores(&self) {
        let inner = self.engine.inner();

        if let Ok(index_dir) = self.config.index_dir() {
            // Save vector store
            if let Err(e) = inner.vector_store.save(&index_dir.join("vectors.bin")) {
                tracing::warn!("Failed to persist vector store: {e}");
            }
            // Flush fulltext store
            if let Err(e) = inner.fulltext_store.flush() {
                tracing::warn!("Failed to flush fulltext store: {e}");
            }
        } else {
            tracing::warn!("Could not determine index directory for persistence");
        }
    }
}

// ---------------------------------------------------------------------------
// Engine setup (mirrors sift-cli pipeline::open_engine)
// ---------------------------------------------------------------------------

/// Open or create the hybrid search engine for the configured index.
///
/// Wraps [`sift_store::open_engine`] to map sift errors into `anyhow` for
/// the MCP entry point and to log the resolved index directory.
fn open_engine(
    config: &Config,
) -> anyhow::Result<(
    HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    MetadataStore,
)> {
    let result = sift_store::open_engine(config)?;
    if let Ok(dir) = config.index_dir() {
        info!("MCP server opened index at {}", dir.display());
    }
    Ok(result)
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
// HTTP transport (optional, behind the `http` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "http")]
pub mod http;

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
///
/// Accepts pre-read file content to avoid redundant filesystem reads.
fn format_line_range(byte_range: Option<(u64, u64)>, content: Option<&str>) -> String {
    let Some((start, end)) = byte_range else {
        return String::new();
    };
    let Some(content) = content else {
        return String::new();
    };
    let len = content.len() as u64;
    let start_line = content[..start.min(len) as usize].lines().count();
    let end_line = content[..end.min(len) as usize].lines().count();
    format!("{start_line}-{end_line}")
}

/// Read context lines around a byte range for a richer search snippet.
///
/// Accepts pre-read file content to avoid redundant filesystem reads.
fn read_context_snippet(
    byte_range: Option<(u64, u64)>,
    content: Option<&str>,
    context_lines: usize,
) -> Option<String> {
    let (start_byte, _) = byte_range?;
    let content = content?;

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
    use std::sync::Mutex;

    /// Serialize tests that mutate the HOME env var.
    static HOME_MUTEX: Mutex<()> = Mutex::new(());

    /// Run `f` with HOME pointed at `dir`, then restore.
    #[allow(unsafe_code)]
    fn with_home<F, R>(dir: &std::path::Path, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir) };
        let result = f();
        match prev {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        result
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
    fn test_internal_err() {
        let err = internal_err("test error".into());
        assert_eq!(err.message, "test error");
    }

    #[test]
    fn test_round2() {
        let r = round2(1.23456);
        assert!((r - 1.23).abs() < f32::EPSILON);
        let r = round2(0.0);
        assert!(r.abs() < f32::EPSILON);
        let r = round2(1.005);
        assert!((r - 1.01).abs() < 0.005);
    }

    #[test]
    fn test_truncate_text_short() {
        let result = truncate_text("hi", 10);
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_format_bytes_gb() {
        let result = format_bytes(2 * 1024 * 1024 * 1024);
        assert_eq!(result, "2.0GB");
    }

    #[test]
    fn test_dir_size_computes_total() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "world!!").unwrap();
        let size = dir_size(tmp.path());
        assert_eq!(size, 12); // 5 + 7
    }

    #[test]
    fn test_format_line_range_no_byte_range() {
        let result = format_line_range(None, Some("some content"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_line_range_missing_file() {
        let result = format_line_range(Some((0, 10)), None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_line_range_with_file() {
        let content = "line1\nline2\nline3\n";
        let result = format_line_range(Some((6, 11)), Some(content));
        assert!(!result.is_empty()); // Should produce "2-2" or similar
    }

    #[test]
    fn test_read_context_snippet_missing_file() {
        let result = read_context_snippet(Some((0, 10)), None, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_no_byte_range() {
        let result = read_context_snippet(None, Some("some content"), 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_no_content() {
        let result = read_context_snippet(Some((0, 10)), None, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_snippet_with_file() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let result = read_context_snippet(Some((0, 11)), Some(content), 1);
        assert!(result.is_some());
        let snippet = result.unwrap();
        assert!(snippet.contains("fn main()"));
    }

    #[test]
    fn test_server_new_and_debug() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let debug = format!("{:?}", server);
            assert!(debug.contains("SiftMcpServer"));
        });
    }

    #[test]
    fn test_server_clone() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let cloned = server.clone();
            let debug = format!("{:?}", cloned);
            assert!(debug.contains("SiftMcpServer"));
        });
    }

    #[test]
    fn test_embed_query_fallback_to_keyword() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let (vec, mode) = server.embed_query("test query", SearchMode::Hybrid);
            assert_eq!(mode, SearchMode::KeywordOnly);
            assert_eq!(vec.len(), 768);
            assert!(vec.iter().all(|&v| v == 0.0));
        });
    }

    #[test]
    fn test_embed_query_keyword_stays_keyword() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let (_, mode) = server.embed_query("test", SearchMode::KeywordOnly);
            assert_eq!(mode, SearchMode::KeywordOnly);
        });
    }

    #[test]
    fn test_sift_status_empty_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let result = server.sift_status().unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["total_files"], 0);
                assert_eq!(parsed["total_chunks"], 0);
            } else {
                panic!("Expected text content");
            }
        });
    }

    #[test]
    fn test_sift_search_empty_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchRequest {
                query: "test query".to_string(),
                limit: Some(5),
                offset: None,
                mode: Some(SearchModeParam::Keyword),
                path: None,
                file_type: None,
                context: None,
            };
            let result = server.sift_search(Parameters(req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["total"], 0);
                assert_eq!(parsed["query_mode"], "keyword");
            } else {
                panic!("Expected text content");
            }
        });
    }

    #[test]
    fn test_get_info() {
        use rmcp::ServerHandler;
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let info = server.get_info();
            assert!(info.instructions.is_some());
        });
    }

    // -----------------------------------------------------------------------
    // Input validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sift_search_rejects_empty_query() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchRequest {
                query: String::new(),
                limit: None,
                offset: None,
                mode: None,
                path: None,
                file_type: None,
                context: None,
            };
            let err = server.sift_search(Parameters(req)).unwrap_err();
            assert!(err.message.contains("must not be empty"));
        });
    }

    #[test]
    fn test_sift_search_rejects_unknown_mode_at_deserialization() {
        // With enum-typed `mode`, invalid values are rejected by serde at
        // deserialization time before the handler runs.
        let json = r#"{"query":"test","mode":"embeddding"}"#;
        let result = serde_json::from_str::<SearchRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_sift_search_accepts_explicit_hybrid() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchRequest {
                query: "test".to_string(),
                limit: Some(5),
                offset: None,
                mode: Some(SearchModeParam::Hybrid),
                path: None,
                file_type: None,
                context: None,
            };
            // Should not error — hybrid is valid (falls back to keyword w/o embedder)
            let result = server.sift_search(Parameters(req)).unwrap();
            assert!(!result.content.is_empty());
        });
    }

    #[test]
    fn test_sift_search_rejects_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchRequest {
                query: "test".to_string(),
                limit: None,
                offset: None,
                mode: Some(SearchModeParam::Keyword),
                path: Some("../../etc/passwd".to_string()),
                file_type: None,
                context: None,
            };
            let err = server.sift_search(Parameters(req)).unwrap_err();
            assert!(err.message.contains("path traversal"));
        });
    }

    #[test]
    fn test_sift_search_skills_rejects_empty_query() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchSkillsRequest {
                query: String::new(),
                detail: None,
                limit: None,
                scope: None,
            };
            let err = server.sift_search_skills(Parameters(req)).unwrap_err();
            assert!(err.message.contains("must not be empty"));
        });
    }

    #[test]
    fn test_sift_search_skills_rejects_unknown_detail_at_deserialization() {
        let json = r#"{"query":"code review","detail":"verbose"}"#;
        let result = serde_json::from_str::<SearchSkillsRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_sift_search_skills_rejects_unknown_scope_at_deserialization() {
        let json = r#"{"query":"code review","scope":"global"}"#;
        let result = serde_json::from_str::<SearchSkillsRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_sift_search_skills_valid_params() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = SearchSkillsRequest {
                query: "code review".to_string(),
                detail: Some(DetailLevel::Instructions),
                limit: Some(3),
                scope: Some(SkillScope::Personal),
            };
            // Should succeed (empty results from empty index, but no validation error)
            let result = server.sift_search_skills(Parameters(req)).unwrap();
            assert!(!result.content.is_empty());
        });
    }

    // -----------------------------------------------------------------------
    // sift_index_text tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_text_basic() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = IndexTextRequest {
                text: "The quick brown fox jumps over the lazy dog".to_string(),
                uri: Some("memory://test/fox-fact".to_string()),
                content_type: None,
                file_type: None,
                title: Some("Fox Fact".to_string()),
            };
            let result = server.sift_index_text(Parameters(req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["status"], "indexed");
                assert_eq!(parsed["uri"], "memory://test/fox-fact");
                assert_eq!(parsed["content_type"], "text");
            } else {
                panic!("Expected text content");
            }
        });
    }

    #[test]
    fn test_index_text_auto_uri() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = IndexTextRequest {
                text: "Some fact to remember".to_string(),
                uri: None,
                content_type: None,
                file_type: None,
                title: None,
            };
            let result = server.sift_index_text(Parameters(req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["status"], "indexed");
                assert!(parsed["uri"]
                    .as_str()
                    .unwrap()
                    .starts_with("memory://agent/"));
            } else {
                panic!("Expected text content");
            }
        });
    }

    #[test]
    fn test_index_text_rejects_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = IndexTextRequest {
                text: String::new(),
                uri: None,
                content_type: None,
                file_type: None,
                title: None,
            };
            let err = server.sift_index_text(Parameters(req)).unwrap_err();
            assert!(err.message.contains("must not be empty"));
        });
    }

    #[test]
    fn test_index_text_rejects_unknown_content_type_at_deserialization() {
        let json = r#"{"text":"hello","content_type":"binary"}"#;
        let result = serde_json::from_str::<IndexTextRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_text_rejects_path_traversal_uri() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = IndexTextRequest {
                text: "hello".to_string(),
                uri: Some("memory://../../etc/passwd".to_string()),
                content_type: None,
                file_type: None,
                title: None,
            };
            let err = server.sift_index_text(Parameters(req)).unwrap_err();
            assert!(err.message.contains("path traversal"));
        });
    }

    #[test]
    fn test_index_text_then_search() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();

            // Index some text
            let index_req = IndexTextRequest {
                text: "Rust is a systems programming language".to_string(),
                uri: Some("memory://test/rust-fact".to_string()),
                content_type: None,
                file_type: Some("md".to_string()),
                title: None,
            };
            server.sift_index_text(Parameters(index_req)).unwrap();

            // Search for it via keyword
            let search_req = SearchRequest {
                query: "systems programming".to_string(),
                limit: Some(5),
                offset: None,
                mode: Some(SearchModeParam::Keyword),
                path: None,
                file_type: None,
                context: None,
            };
            let result = server.sift_search(Parameters(search_req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert!(
                    parsed["total"].as_u64().unwrap() > 0,
                    "Should find indexed text"
                );
            } else {
                panic!("Expected text content");
            }
        });
    }

    // -----------------------------------------------------------------------
    // sift_delete tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_nonexistent() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = DeleteRequest {
                uri: "memory://nonexistent".to_string(),
            };
            let result = server.sift_delete(Parameters(req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["status"], "not_found");
                assert_eq!(parsed["chunks_removed"], 0);
            } else {
                panic!("Expected text content");
            }
        });
    }

    #[test]
    fn test_delete_rejects_empty_uri() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = DeleteRequest { uri: String::new() };
            let err = server.sift_delete(Parameters(req)).unwrap_err();
            assert!(err.message.contains("must not be empty"));
        });
    }

    #[test]
    fn test_index_then_delete() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();

            // Index text
            let index_req = IndexTextRequest {
                text: "temporary fact to delete".to_string(),
                uri: Some("memory://test/temp".to_string()),
                content_type: None,
                file_type: None,
                title: None,
            };
            server.sift_index_text(Parameters(index_req)).unwrap();

            // Delete it
            let del_req = DeleteRequest {
                uri: "memory://test/temp".to_string(),
            };
            let result = server.sift_delete(Parameters(del_req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["status"], "deleted");
                assert!(parsed["chunks_removed"].as_u64().unwrap() > 0);
            } else {
                panic!("Expected text content");
            }
        });
    }

    // -----------------------------------------------------------------------
    // sift_list_sources tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_sources_empty_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();
            let req = ListSourcesRequest {
                path: None,
                limit: None,
            };
            let result = server.sift_list_sources(Parameters(req)).unwrap();
            let content = &result.content[0];
            if let rmcp::model::RawContent::Text(text) = &content.raw {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["total"], 0);
                assert_eq!(parsed["showing"], 0);
            } else {
                panic!("Expected text content");
            }
        });
    }

    fn list_sources_uris(server: &SiftMcpServer) -> Vec<String> {
        let req = ListSourcesRequest {
            path: None,
            limit: Some(500),
        };
        let result = server.sift_list_sources(Parameters(req)).unwrap();
        let rmcp::model::RawContent::Text(text) = &result.content[0].raw else {
            panic!("Expected text content");
        };
        let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
        parsed["sources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|src| src["path"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn test_index_text_appears_in_list_sources_and_status() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();

            let uri = "memory://test/metadata-roundtrip".to_string();
            let req = IndexTextRequest {
                text: "Metadata roundtrip fixture content.".to_string(),
                uri: Some(uri.clone()),
                content_type: None,
                file_type: Some("md".to_string()),
                title: None,
            };
            server.sift_index_text(Parameters(req)).unwrap();

            let listed = list_sources_uris(&server);
            assert!(
                listed.contains(&uri),
                "sift_list_sources should report indexed URI; got {listed:?}"
            );

            let status = server.sift_status().unwrap();
            let rmcp::model::RawContent::Text(text) = &status.content[0].raw else {
                panic!("Expected text content");
            };
            let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
            assert!(
                parsed["total_files"].as_u64().unwrap() >= 1,
                "sift_status total_files should reflect the new URI: {parsed}"
            );
            assert!(
                parsed["total_chunks"].as_u64().unwrap() >= 1,
                "sift_status total_chunks should reflect the new chunk: {parsed}"
            );
        });
    }

    #[test]
    fn test_index_text_then_delete_clears_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let server = SiftMcpServer::new(config).unwrap();

            let uri = "memory://test/metadata-delete".to_string();
            let req = IndexTextRequest {
                text: "Soon-to-be-deleted metadata fixture.".to_string(),
                uri: Some(uri.clone()),
                content_type: None,
                file_type: None,
                title: None,
            };
            server.sift_index_text(Parameters(req)).unwrap();
            assert!(list_sources_uris(&server).contains(&uri));

            let del = DeleteRequest { uri: uri.clone() };
            server.sift_delete(Parameters(del)).unwrap();

            let listed = list_sources_uris(&server);
            assert!(
                !listed.contains(&uri),
                "sift_list_sources should not list deleted URI; got {listed:?}"
            );
        });
    }
}
