use axum::{
    error_handling::HandleErrorLayer,
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sift_core::{Embedder, IndexStats, SearchMode};
use sift_store::{DefaultFullTextStore, HybridSearchEngine, MetadataStore, SimpleVectorStore};
use std::sync::Arc;
use std::time::Duration;

pub struct AppState {
    pub engine: HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    pub metadata: MetadataStore,
    pub embedder: Option<Box<dyn Embedder>>,
    /// Path to the memory directory. Enables `/api/memory/*` endpoints when set.
    pub memory_dir: Option<std::path::PathBuf>,
    /// Shared memory store — opened once, reused across requests.
    pub memory: Option<Arc<sift_memory::MemoryStore>>,
    /// Consolidation config from user settings (not just defaults).
    pub consolidation_config: Option<sift_memory::ConsolidationConfig>,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub r#type: Option<String>,
    /// Search mode: "hybrid" (default), "vector", or "keyword"
    pub mode: Option<String>,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub total: usize,
    pub mode: String,
}

#[derive(Serialize)]
pub struct SearchResultItem {
    pub uri: String,
    pub text: String,
    pub score: f32,
    pub chunk_index: u32,
    pub content_type: String,
    pub file_type: String,
    pub title: Option<String>,
    pub byte_range: Option<(u64, u64)>,
}

#[derive(Serialize)]
pub struct SourceItem {
    pub uri: String,
    pub file_type: String,
    pub chunks: u32,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub stats: IndexStats,
    pub has_embedder: bool,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    let api_routes = Router::new()
        .route("/api/search", get(search))
        .route("/api/status", get(status))
        .route("/api/list", get(list))
        .layer(
            tower::ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|_: tower::BoxError| async {
                    StatusCode::TOO_MANY_REQUESTS
                }))
                .buffer(100)
                .rate_limit(100, Duration::from_secs(60)),
        );

    let memory_routes = Router::new()
        .route("/api/memory/status", get(memory_status))
        .route("/api/memory/consolidate", post(memory_consolidate));

    Router::new()
        .route("/health", get(health))
        .merge(api_routes)
        .merge(memory_routes)
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    // Input validation
    if params.q.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Query must not be empty".to_string(),
        ));
    }
    if params.q.len() > 10_000 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Query too long ({} chars, max 10000)", params.q.len()),
        ));
    }
    if params.limit == 0 || params.limit > 1000 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Limit must be between 1 and 1000, got {}", params.limit),
        ));
    }

    let requested_mode = match params.mode.as_deref() {
        Some("vector") => SearchMode::VectorOnly,
        Some("keyword") => SearchMode::KeywordOnly,
        _ => SearchMode::Hybrid,
    };

    // Determine effective mode based on embedder availability
    let (query_vector, effective_mode) = match (&state.embedder, requested_mode) {
        (Some(emb), mode) => {
            let vec = emb
                .embed(&format!("search_query: {}", &params.q))
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Embedding failed: {e}"),
                    )
                })?;
            (vec, mode)
        }
        (None, SearchMode::VectorOnly) => {
            return Err((
                StatusCode::BAD_REQUEST,
                "Vector search requested but no embedding model is loaded".to_string(),
            ));
        }
        (None, _) => (vec![0.0f32; 768], SearchMode::KeywordOnly),
    };

    let results = state
        .engine
        .search(&query_vector, &params.q, params.limit, effective_mode)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<SearchResultItem> = results
        .into_iter()
        .filter(|r| {
            if let Some(ref ft) = params.r#type {
                r.file_type == *ft
            } else {
                true
            }
        })
        .map(|r| SearchResultItem {
            uri: r.uri,
            text: r.text,
            score: r.score,
            chunk_index: r.chunk_index,
            content_type: r.content_type.to_string(),
            file_type: r.file_type,
            title: r.title,
            byte_range: r.byte_range,
        })
        .collect();

    let total = items.len();
    let mode_str = match effective_mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::VectorOnly => "vector",
        SearchMode::KeywordOnly => "keyword",
    };

    Ok(Json(SearchResponse {
        results: items,
        total,
        mode: mode_str.to_string(),
    }))
}

async fn list(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<SourceItem>>, (StatusCode, String)> {
    let sources = state
        .metadata
        .list_sources()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<SourceItem> = sources
        .into_iter()
        .map(|(uri, file_type, chunks)| SourceItem {
            uri,
            file_type,
            chunks,
        })
        .collect();

    Ok(Json(items))
}

async fn status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    let stats = state
        .metadata
        .stats()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(StatusResponse {
        status: "ok".to_string(),
        stats,
        has_embedder: state.embedder.is_some(),
    }))
}

// ---------------------------------------------------------------------------
// Memory endpoints
// ---------------------------------------------------------------------------

/// `GET /api/memory/status` — memory system statistics and tier breakdown.
async fn memory_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let memory = state.memory.clone().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Memory not available".into(),
    ))?;

    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let stats = memory.stats().map_err(|e| format!("Stats failed: {e}"))?;
        Ok(serde_json::json!({
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
        }))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(result))
}

/// `POST /api/memory/consolidate` — trigger on-demand consolidation.
async fn memory_consolidate(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let memory = state.memory.clone().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Memory not available".into(),
    ))?;
    let dir = state
        .memory_dir
        .clone()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Memory dir not set".into()))?;
    let config = state.consolidation_config.clone().unwrap_or_default();

    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let episodes = sift_memory::episodes::EpisodeStore::open(&dir)
            .map_err(|e| format!("Episode store failed: {e}"))?;

        let report = sift_memory::consolidation::run_consolidation(&memory, &episodes, &config)
            .map_err(|e| format!("Consolidation failed: {e}"))?;

        memory.save().map_err(|e| format!("Save failed: {e}"))?;

        Ok(serde_json::json!({
            "status": "completed",
            "episodes_processed": report.episodes_processed,
            "episodes_skipped": report.episodes_skipped,
            "entities_created": report.entities_created,
            "observations_created": report.observations_created,
            "duplicates_merged": report.duplicates_merged,
            "promotions": report.promotions,
            "skills_created": report.skills_created,
            "skills_updated": report.skills_updated,
            "observations_pruned": report.observations_pruned,
            "entities_pruned": report.entities_pruned,
        }))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use sift_store::{DefaultFullTextStore, HybridSearchEngine, MetadataStore, SimpleVectorStore};
    use tower::ServiceExt;

    struct TestHarness {
        state: Arc<AppState>,
        _dir: tempfile::TempDir,
    }

    impl TestHarness {
        fn new() -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let vector_store = SimpleVectorStore::new();
            let fts_store = DefaultFullTextStore::open(&dir.path().join("search.db")).unwrap();
            let engine = HybridSearchEngine::new(vector_store, fts_store, 0.7);
            let metadata = MetadataStore::open_in_memory().unwrap();

            let state = Arc::new(AppState {
                engine,
                metadata,
                embedder: None,
                memory_dir: None,
                memory: None,
                consolidation_config: None,
            });
            TestHarness { state, _dir: dir }
        }

        fn app(&self) -> axum::Router {
            create_router(self.state.clone())
        }
    }

    async fn get(app: axum::Router, uri: &str) -> (StatusCode, String) {
        let req = axum::http::Request::builder()
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let h = TestHarness::new();
        let (status, body) = get(h.app(), "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "ok");
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let h = TestHarness::new();
        let (status, body) = get(h.app(), "/api/status").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["status"], "ok");
        assert_eq!(resp["has_embedder"], false);
        assert_eq!(resp["stats"]["total_sources"], 0);
    }

    #[tokio::test]
    async fn test_search_keyword_mode() {
        let h = TestHarness::new();

        let chunks = vec![sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: "rust programming language systems".to_string(),
                source_uri: "file:///test.rs".to_string(),
                chunk_index: 0,
                content_type: sift_core::ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.0; 3],
        }];
        h.state.engine.insert(&chunks).unwrap();

        let (status, body) = get(h.app(), "/api/search?q=rust+programming&mode=keyword").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["mode"], "keyword");
        assert!(resp["total"].as_u64().unwrap() > 0);
        assert_eq!(resp["results"][0]["uri"], "file:///test.rs");
    }

    #[tokio::test]
    async fn test_search_empty_index() {
        let h = TestHarness::new();
        let (status, body) = get(h.app(), "/api/search?q=nothing&mode=keyword").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["total"], 0);
        assert_eq!(resp["results"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_search_vector_mode_without_embedder() {
        let h = TestHarness::new();
        let (status, _body) = get(h.app(), "/api/search?q=test&mode=vector").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_type_filter() {
        let h = TestHarness::new();

        let chunks = vec![
            sift_core::EmbeddedChunk {
                chunk: sift_core::Chunk {
                    text: "rust code here".to_string(),
                    source_uri: "file:///code.rs".to_string(),
                    chunk_index: 0,
                    content_type: sift_core::ContentType::Code,
                    file_type: "rs".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![0.0; 3],
            },
            sift_core::EmbeddedChunk {
                chunk: sift_core::Chunk {
                    text: "rust documentation here".to_string(),
                    source_uri: "file:///doc.md".to_string(),
                    chunk_index: 0,
                    content_type: sift_core::ContentType::Text,
                    file_type: "md".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![0.0; 3],
            },
        ];
        h.state.engine.insert(&chunks).unwrap();

        let (status, body) = get(h.app(), "/api/search?q=rust&mode=keyword&type=rs").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        let results = resp["results"].as_array().unwrap();
        assert!(results.iter().all(|r| r["file_type"] == "rs"));
    }

    #[tokio::test]
    async fn test_404_unknown_route() {
        let h = TestHarness::new();
        let (status, _body) = get(h.app(), "/nonexistent").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_search_default_mode_no_embedder_falls_back_to_keyword() {
        let h = TestHarness::new();

        let chunks = vec![sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: "rust fallback test".to_string(),
                source_uri: "file:///fallback.rs".to_string(),
                chunk_index: 0,
                content_type: sift_core::ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.0; 3],
        }];
        h.state.engine.insert(&chunks).unwrap();

        // Default mode (no mode param) with no embedder should fall back to keyword
        let (status, body) = get(h.app(), "/api/search?q=rust+fallback").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["mode"], "keyword");
    }

    #[tokio::test]
    async fn test_search_with_custom_limit() {
        let h = TestHarness::new();

        let chunks: Vec<sift_core::EmbeddedChunk> = (0..5)
            .map(|i| sift_core::EmbeddedChunk {
                chunk: sift_core::Chunk {
                    text: format!("rust document number {i}"),
                    source_uri: format!("file:///doc{i}.rs"),
                    chunk_index: 0,
                    content_type: sift_core::ContentType::Code,
                    file_type: "rs".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![0.0; 3],
            })
            .collect();
        h.state.engine.insert(&chunks).unwrap();

        let (status, body) = get(h.app(), "/api/search?q=rust+document&mode=keyword&limit=2").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(resp["total"].as_u64().unwrap() <= 2);
    }

    struct FakeEmbedder;

    impl sift_core::Embedder for FakeEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> sift_core::SiftResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn model_name(&self) -> &'static str {
            "fake-embedder"
        }
    }

    fn harness_with_embedder() -> TestHarness {
        let dir = tempfile::TempDir::new().unwrap();
        let vector_store = SimpleVectorStore::new();
        let fts_store = DefaultFullTextStore::open(&dir.path().join("search.db")).unwrap();
        let engine = HybridSearchEngine::new(vector_store, fts_store, 0.7);
        let metadata = MetadataStore::open_in_memory().unwrap();

        let state = Arc::new(AppState {
            engine,
            metadata,
            embedder: Some(Box::new(FakeEmbedder)),
            memory_dir: None,
            memory: None,
            consolidation_config: None,
        });
        TestHarness { state, _dir: dir }
    }

    #[tokio::test]
    async fn test_search_with_embedder_hybrid_mode() {
        let h = harness_with_embedder();

        let chunks = vec![sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: "rust vector search test".to_string(),
                source_uri: "file:///vec.rs".to_string(),
                chunk_index: 0,
                content_type: sift_core::ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![1.0, 0.0, 0.0],
        }];
        h.state.engine.insert(&chunks).unwrap();

        // With embedder, default mode should be hybrid
        let (status, body) = get(h.app(), "/api/search?q=rust+vector").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["mode"], "hybrid");
    }

    #[tokio::test]
    async fn test_search_with_embedder_vector_mode() {
        let h = harness_with_embedder();

        let chunks = vec![sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: "rust vector only search".to_string(),
                source_uri: "file:///vec.rs".to_string(),
                chunk_index: 0,
                content_type: sift_core::ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![1.0, 0.0, 0.0],
        }];
        h.state.engine.insert(&chunks).unwrap();

        let (status, body) = get(h.app(), "/api/search?q=rust&mode=vector").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["mode"], "vector");
    }

    #[tokio::test]
    async fn test_status_endpoint_with_embedder() {
        let h = harness_with_embedder();
        let (status, body) = get(h.app(), "/api/status").await;
        assert_eq!(status, StatusCode::OK);

        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["status"], "ok");
        assert_eq!(resp["has_embedder"], true);
    }

    #[tokio::test]
    async fn test_rate_limit_allows_normal_requests() {
        let h = TestHarness::new();
        // Health endpoint is not rate-limited
        let (status, body) = get(h.app(), "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "ok");

        // API endpoints are rate-limited but should pass under normal load
        let (status, body) = get(h.app(), "/api/status").await;
        assert_eq!(status, StatusCode::OK);
        let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(resp["status"], "ok");
    }
}
