use crossbeam_channel as channel;
use fs4::fs_std::FileExt;
#[cfg(feature = "fancy")]
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use sift_chunker::chunker_for_content;
#[cfg(feature = "vision")]
use sift_core::ContentType;
use sift_core::{
    CancellationToken, Chunk, Config, EmbeddedChunk, Embedder, ScanOptions, SiftResult, SourceItem,
};
#[cfg(feature = "embeddings")]
use sift_embed::{models::NOMIC_EMBED_TEXT_V2, EmbeddingCache, ModelManager, OnnxEmbedder};
use sift_parsers::ParserRegistry;
use sift_sources::{FilesystemSource, Source};
#[cfg(feature = "sqlite")]
use sift_store::TransactionGuard;
#[cfg(feature = "hnsw")]
use sift_store::VectorIndex;
use sift_store::{
    DefaultFullTextStore, FullTextStore, HybridSearchEngine, MetadataStore, SimpleVectorStore,
};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

#[cfg(feature = "vision")]
use sift_embed::{models::NOMIC_EMBED_VISION_V1_5, VisionEmbedder};

/// No-op progress indicator used when the `fancy` feature is disabled.
#[cfg(not(feature = "fancy"))]
struct NoopProgress;

#[cfg(not(feature = "fancy"))]
impl NoopProgress {
    fn inc(&self, _n: u64) {}
    fn set_message(&self, _msg: String) {}
    fn finish_and_clear(&self) {}
}

/// Open or create the hybrid search engine for an index.
pub fn open_engine(
    config: &Config,
) -> SiftResult<(
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

    Ok((engine, metadata))
}

/// Try to load the ONNX embedder. Returns `None` when the `embeddings` feature
/// is disabled or when the model is not available.
#[cfg(feature = "embeddings")]
pub fn load_embedder(model_override: Option<&str>) -> Option<OnnxEmbedder> {
    let manager = ModelManager::new().ok()?;
    manager.init_ort_env();

    // Determine which model to load
    let (model_dir, model_name, dimensions) = if let Some(name) = model_override {
        let path = std::path::Path::new(name);
        if path.is_absolute() && path.is_dir() {
            // Absolute path to a model directory
            (
                path.to_path_buf(),
                name.to_string(),
                NOMIC_EMBED_TEXT_V2.dimensions,
            )
        } else {
            // Look up by name in ~/.sift/models/
            if !manager.is_downloaded(name) {
                info!(
                    "Model '{}' not downloaded — using keyword-only mode. Run `sift models download {}` first.",
                    name, name
                );
                return None;
            }
            (
                manager.model_dir(name),
                name.to_string(),
                NOMIC_EMBED_TEXT_V2.dimensions,
            )
        }
    } else {
        let model_def = &NOMIC_EMBED_TEXT_V2;
        if !manager.is_downloaded(model_def.name) {
            info!("Embedding model not downloaded — using keyword-only mode. Run `sift models download nomic-embed-text-v2` for semantic search.");
            return None;
        }
        (
            manager.model_dir(model_def.name),
            model_def.name.to_string(),
            model_def.dimensions,
        )
    };

    match OnnxEmbedder::load(&model_dir, &model_name, dimensions) {
        Ok(embedder) => {
            info!("Loaded embedding model: {}", model_name);
            Some(embedder)
        }
        Err(e) => {
            warn!(
                "Failed to load embedding model: {}. Falling back to keyword-only.",
                e
            );
            None
        }
    }
}

/// Try to load the vision embedder for image embedding.
#[cfg(all(feature = "embeddings", feature = "vision"))]
pub fn load_vision_embedder() -> Option<VisionEmbedder> {
    let manager = ModelManager::new().ok()?;
    manager.init_ort_env();
    let model_def = &NOMIC_EMBED_VISION_V1_5;

    if !manager.is_model_file_downloaded(model_def.name) {
        debug!(
            "Vision model '{}' not downloaded — images will use text-only metadata embedding.",
            model_def.name
        );
        return None;
    }

    let model_dir = manager.model_dir(model_def.name);
    match VisionEmbedder::load(&model_dir, model_def.name, model_def.dimensions) {
        Ok(embedder) => {
            info!("Loaded vision embedding model: {}", model_def.name);
            Some(embedder)
        }
        Err(e) => {
            warn!(
                "Failed to load vision model: {}. Images will use text-only embedding.",
                e
            );
            None
        }
    }
}

/// Open the embedding cache for an index.
#[cfg(feature = "embeddings")]
pub fn open_cache(config: &Config) -> Option<EmbeddingCache> {
    #[cfg(feature = "sqlite")]
    let cache_path = config.index_dir().ok()?.join("embedding_cache.db");
    #[cfg(not(feature = "sqlite"))]
    let cache_path = config.index_dir().ok()?.join("embedding_cache.json");
    match EmbeddingCache::open(&cache_path) {
        Ok(cache) => {
            info!("Opened embedding cache ({} entries)", cache.len());
            Some(cache)
        }
        Err(e) => {
            warn!(
                "Failed to open embedding cache: {}. Proceeding without cache.",
                e
            );
            None
        }
    }
}

/// Result of the parallel read/parse/chunk phase for a single file.
struct ParsedItem {
    item: SourceItem,
    chunks: Vec<Chunk>,
    file_type: String,
}

/// A file that has been embedded and is ready for storage.
struct StoreItem {
    item: SourceItem,
    embedded: Vec<EmbeddedChunk>,
    file_type: String,
}

/// Run the full scan pipeline: discover → parse → chunk → embed → store.
///
/// Stages are connected by bounded channels so they run concurrently:
///
/// ```text
/// [Discover+Filter] → chan(64) → [Parse+Chunk (rayon)] → chan(128) → [Embed (batch)] → chan(128) → [Store]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn run_scan_pipeline(
    config: &Config,
    options: &ScanOptions,
    engine: &HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    metadata: &MetadataStore,
    embedder: Option<&dyn Embedder>,
    #[cfg(feature = "vision")] vision_embedder: Option<&VisionEmbedder>,
    token: &CancellationToken,
    quiet: bool,
) -> SiftResult<ScanStats> {
    // Acquire an advisory exclusive lock to prevent concurrent index mutations.
    let lock_path = config.index_dir()?.join(".lock");
    let lock_file = std::fs::File::create(&lock_path)?;
    lock_file.try_lock_exclusive().map_err(|_| {
        sift_core::SiftError::Storage(
            "Another sift process is using this index. If this is incorrect, remove the .lock file."
                .into(),
        )
    })?;

    let source = FilesystemSource::new();
    #[cfg(feature = "embeddings")]
    let cache = if embedder.is_some() {
        open_cache(config)
    } else {
        None
    };
    #[cfg(not(feature = "embeddings"))]
    let cache: Option<()> = None;

    let mut stats = ScanStats::default();

    // Phase 1: Discover files
    let items = source.discover(options)?;

    info!("Discovered {} files", items.len());
    stats.discovered = items.len() as u64;

    if options.dry_run {
        return Ok(stats);
    }

    // Phase 2: Filter unchanged files (batch lookup)
    let known_hashes = metadata.load_all_hashes()?;
    let mut to_process: Vec<SourceItem> = Vec::new();
    for item in items {
        match known_hashes.get(&item.uri) {
            Some(stored_hash) if stored_hash.as_slice() == item.content_hash.as_slice() => {
                stats.skipped += 1;
                debug!("Unchanged, skipping: {}", item.uri);
            }
            _ => {
                to_process.push(item);
            }
        }
    }

    if to_process.is_empty() {
        info!("All files are up to date, nothing to index");
        return Ok(stats);
    }

    let total_to_process = to_process.len();
    info!(
        "{} files to process ({} unchanged)",
        total_to_process, stats.skipped
    );

    let chunk_size = config.default.chunk_size;
    let chunk_overlap = config.default.chunk_overlap;

    // Build the rayon pool for the parse stage.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.jobs)
        .build()
        .unwrap_or_else(|_| {
            rayon::ThreadPoolBuilder::new()
                .build()
                .expect("failed to build default rayon pool")
        });

    // Atomic counters for stats gathered across threads.
    let atomic_errors = AtomicU64::new(0);
    let atomic_indexed = AtomicU64::new(0);
    let atomic_chunks = AtomicU64::new(0);
    let atomic_cache_hits = AtomicU64::new(0);

    // Progress bar (shared across the store stage).
    #[cfg(not(feature = "fancy"))]
    let _ = quiet;
    #[cfg(feature = "fancy")]
    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total_to_process as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        pb
    };
    #[cfg(not(feature = "fancy"))]
    let pb = NoopProgress;

    // --- Bounded channels between stages ---
    // discover_tx/rx: SourceItem (capacity 64)
    let (discover_tx, discover_rx) = channel::bounded::<SourceItem>(64);
    // parse_tx/rx:    ParsedItem  (capacity 128)
    let (parse_tx, parse_rx) = channel::bounded::<ParsedItem>(128);
    // embed_tx/rx:    StoreItem   (capacity 128)
    let (embed_tx, embed_rx) = channel::bounded::<StoreItem>(128);

    // Collect per-file file_type stats in the store stage (single-threaded).
    let mut file_type_map: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();

    // Use std::thread::scope so we can borrow local references across threads.
    let scope_result: Result<(), sift_core::SiftError> = std::thread::scope(|s| {
        // ---- Stage 1: Feed discovered items into the pipeline ----
        s.spawn(|| {
            for item in to_process {
                if token.is_cancelled() {
                    break;
                }
                if discover_tx.send(item).is_err() {
                    break;
                }
            }
            drop(discover_tx);
        });

        // ---- Stage 2: Parse + Chunk (rayon) ----
        s.spawn(|| {
            pool.install(|| {
                let parser_registry = ParserRegistry::new();
                discover_rx.into_iter().par_bridge().for_each(|item| {
                    if token.is_cancelled() {
                        return;
                    }
                    // Read file content
                    let content = match std::fs::read(&item.path) {
                        Ok(c) => c,
                        Err(e) => {
                            warn!("Failed to read {}: {}", item.path.display(), e);
                            atomic_errors.fetch_add(1, Ordering::Relaxed);
                            pb.inc(1);
                            return;
                        }
                    };

                    // Parse
                    let doc = match parser_registry.parse(
                        &content,
                        item.mime_type.as_deref(),
                        item.extension.as_deref(),
                    ) {
                        Ok(d) => d,
                        Err(e) => {
                            debug!("Parse error for {}: {}", item.uri, e);
                            atomic_errors.fetch_add(1, Ordering::Relaxed);
                            pb.inc(1);
                            return;
                        }
                    };

                    // Chunk
                    let chunker = chunker_for_content(doc.content_type, chunk_size, chunk_overlap);
                    let raw_chunks =
                        chunker.chunk_with_language(&doc.text, item.extension.as_deref());

                    let file_type = item.extension.as_deref().unwrap_or("unknown").to_string();

                    let chunks: Vec<Chunk> = raw_chunks
                        .iter()
                        .enumerate()
                        .map(|(i, (text, offset))| Chunk {
                            text: text.clone(),
                            source_uri: item.uri.clone(),
                            chunk_index: i as u32,
                            content_type: doc.content_type,
                            file_type: file_type.clone(),
                            title: doc.title.clone(),
                            language: doc.language.clone(),
                            byte_range: Some((*offset as u64, (*offset + text.len()) as u64)),
                        })
                        .collect();

                    if chunks.is_empty() {
                        pb.inc(1);
                        return;
                    }

                    let parsed = ParsedItem {
                        item,
                        chunks,
                        file_type,
                    };

                    // Send downstream; if the channel is closed, stop.
                    let _ = parse_tx.send(parsed);
                });
            });
            drop(parse_tx);
        });

        // ---- Stage 3: Embed (parallel) ----
        s.spawn(|| {
            parse_rx.into_iter().par_bridge().for_each(|parsed| {
                if token.is_cancelled() {
                    return;
                }

                let chunks = parsed.chunks;
                let item = parsed.item;
                let file_type = parsed.file_type;

                // Check if this is an image file that should use vision embedding
                #[cfg(feature = "vision")]
                let is_image = chunks
                    .first()
                    .is_some_and(|c| c.content_type == ContentType::Image);
                #[cfg(not(feature = "vision"))]
                let is_image = false;

                // Embed chunks
                let embedded: Vec<EmbeddedChunk> = if is_image {
                    #[cfg(feature = "vision")]
                    {
                        embed_image_chunks(&chunks, &item, vision_embedder, embedder)
                    }
                    #[cfg(not(feature = "vision"))]
                    {
                        embed_text_chunks_atomic(&chunks, embedder, &cache, &atomic_cache_hits)
                    }
                } else {
                    embed_text_chunks_atomic(&chunks, embedder, &cache, &atomic_cache_hits)
                };

                let store_item = StoreItem {
                    item,
                    embedded,
                    file_type,
                };

                let _ = embed_tx.send(store_item);
            });
            drop(embed_tx);
        });

        // ---- Stage 4: Store (runs on the scoped main thread) ----
        // TransactionGuard auto-rolls back on drop if not committed.
        #[cfg(feature = "sqlite")]
        let mut txn = TransactionGuard::begin(metadata)?;
        let batch_size: u64 = 100;
        let mut batch_count: u64 = 0;

        for store_item in embed_rx {
            if token.is_cancelled() {
                break;
            }

            pb.set_message(
                store_item
                    .item
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string(),
            );

            let chunk_count = store_item.embedded.len() as u32;

            // Delete old chunks for this URI if re-indexing
            let _ = engine.delete_by_uri(&store_item.item.uri);

            // Insert into stores
            engine.insert(&store_item.embedded)?;

            // Update metadata
            metadata.upsert_source(
                &store_item.item.uri,
                &store_item.item.content_hash,
                store_item.item.size,
                &store_item.file_type,
                store_item.item.modified_at,
                chunk_count,
            )?;

            atomic_indexed.fetch_add(1, Ordering::Relaxed);
            atomic_chunks.fetch_add(u64::from(chunk_count), Ordering::Relaxed);
            *file_type_map
                .entry(store_item.file_type.clone())
                .or_insert(0) += 1;

            pb.inc(1);

            batch_count += 1;
            if batch_count.is_multiple_of(batch_size) {
                #[cfg(feature = "sqlite")]
                txn.commit_and_reopen()?;
            }
        }

        #[cfg(feature = "sqlite")]
        txn.commit()?;

        pb.finish_and_clear();

        Ok(())
    });

    scope_result?;

    if token.is_cancelled() {
        warn!("Interrupted — partial results have been committed");
    }

    // Collect atomic counters into stats.
    stats.errors = atomic_errors.load(Ordering::Relaxed);
    stats.indexed = atomic_indexed.load(Ordering::Relaxed);
    stats.chunks = atomic_chunks.load(Ordering::Relaxed);
    stats.cache_hits = atomic_cache_hits.load(Ordering::Relaxed);
    stats.file_types = file_type_map;

    // Log cache stats
    #[cfg(feature = "embeddings")]
    if let Some(ref cache) = cache {
        let (hits, misses) = cache.stats();
        if hits + misses > 0 {
            info!(
                "Embedding cache: {} hits, {} misses ({:.0}% hit rate)",
                hits,
                misses,
                (hits as f64 / (hits + misses) as f64) * 100.0
            );
        }
    }

    // Persist stores to disk
    let vector_path = config.index_dir()?.join("vectors.bin");
    engine.vector_store.save(&vector_path)?;
    engine.fulltext_store.flush()?;

    Ok(stats)
}

/// Embed text chunks using the text embedding model (or zero-vectors if unavailable).
/// Uses an AtomicU64 for cache-hit tracking instead of &mut ScanStats so it can be
/// called from a concurrent pipeline stage.
#[cfg(feature = "embeddings")]
fn embed_text_chunks_atomic(
    chunks: &[Chunk],
    embedder: Option<&dyn Embedder>,
    cache: &Option<EmbeddingCache>,
    cache_hits: &AtomicU64,
) -> Vec<EmbeddedChunk> {
    if let Some(emb) = embedder {
        let mut all_embedded = Vec::with_capacity(chunks.len());
        for batch_chunks in chunks.chunks(32) {
            // Check cache for each chunk, collect indices that need embedding
            let mut cached_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch_chunks.len());
            let mut need_embed: Vec<usize> = Vec::new();

            for (i, chunk) in batch_chunks.iter().enumerate() {
                if let Some(ref cache) = cache {
                    if let Some(vec) = cache.get(&chunk.text) {
                        cached_vectors.push(Some(vec));
                        cache_hits.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
                cached_vectors.push(None);
                need_embed.push(i);
            }

            // Embed only the uncached chunks
            if !need_embed.is_empty() {
                let texts_to_embed: Vec<&str> = need_embed
                    .iter()
                    .map(|&i| batch_chunks[i].text.as_str())
                    .collect();

                let new_vectors = match emb.embed_batch(&texts_to_embed) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("Embedding failed for batch: {}. Using zero vectors.", e);
                        vec![vec![0.0f32; emb.dimensions()]; texts_to_embed.len()]
                    }
                };

                // Store new vectors in cache and fill in the gaps
                let mut cache_batch: Vec<(&str, &[f32])> = Vec::new();
                for (j, &idx) in need_embed.iter().enumerate() {
                    cached_vectors[idx] = Some(new_vectors[j].clone());
                    cache_batch.push((batch_chunks[idx].text.as_str(), &new_vectors[j]));
                }
                if let Some(ref cache) = cache {
                    cache.put_batch(&cache_batch);
                }
            }

            for (chunk, vector) in batch_chunks.iter().zip(cached_vectors) {
                all_embedded.push(EmbeddedChunk {
                    chunk: chunk.clone(),
                    vector: vector.unwrap_or_else(|| vec![0.0f32; emb.dimensions()]),
                });
            }
        }
        all_embedded
    } else {
        zero_vector_chunks(chunks)
    }
}

/// Embed text chunks with zero vectors (no embedding model available).
#[cfg(not(feature = "embeddings"))]
fn embed_text_chunks_atomic(
    chunks: &[Chunk],
    _embedder: Option<&dyn Embedder>,
    _cache: &Option<()>,
    _cache_hits: &AtomicU64,
) -> Vec<EmbeddedChunk> {
    zero_vector_chunks(chunks)
}

fn zero_vector_chunks(chunks: &[Chunk]) -> Vec<EmbeddedChunk> {
    chunks
        .iter()
        .map(|chunk| EmbeddedChunk {
            chunk: chunk.clone(),
            vector: vec![0.0f32; 768],
        })
        .collect()
}

/// Embed image chunks using the vision model if available, falling back to text embedding.
#[cfg(all(feature = "embeddings", feature = "vision"))]
fn embed_image_chunks(
    chunks: &[Chunk],
    item: &SourceItem,
    vision_embedder: Option<&VisionEmbedder>,
    text_embedder: Option<&dyn Embedder>,
) -> Vec<EmbeddedChunk> {
    if let Some(vision) = vision_embedder {
        // Read the raw image bytes for vision embedding
        match std::fs::read(&item.path) {
            Ok(image_bytes) => {
                match vision.embed_image(&image_bytes) {
                    Ok(vector) => {
                        // Use the vision vector for all chunks of this image
                        return chunks
                            .iter()
                            .map(|chunk| EmbeddedChunk {
                                chunk: chunk.clone(),
                                vector: vector.clone(),
                            })
                            .collect();
                    }
                    Err(e) => {
                        debug!(
                            "Vision embedding failed for {}: {}. Falling back to text.",
                            item.uri, e
                        );
                    }
                }
            }
            Err(e) => {
                debug!(
                    "Failed to re-read image {}: {}. Falling back to text.",
                    item.uri, e
                );
            }
        }
    }

    // Fallback: use text embedder on the metadata text, or zero vectors
    let dims = text_embedder.map_or(768, |e| e.dimensions());
    if let Some(emb) = text_embedder {
        chunks
            .iter()
            .map(|chunk| {
                let vector = emb
                    .embed(&chunk.text)
                    .unwrap_or_else(|_| vec![0.0f32; dims]);
                EmbeddedChunk {
                    chunk: chunk.clone(),
                    vector,
                }
            })
            .collect()
    } else {
        chunks
            .iter()
            .map(|chunk| EmbeddedChunk {
                chunk: chunk.clone(),
                vector: vec![0.0f32; dims],
            })
            .collect()
    }
}

pub use sift_core::pipeline::ScanStats;
