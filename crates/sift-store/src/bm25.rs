use crate::traits::FullTextStore;
use serde::{Deserialize, Serialize};
use sift_core::{EmbeddedChunk, SearchResult, SiftResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Pure-Rust BM25 inverted index. Zero external dependencies.
///
/// Provides real keyword search when the `fulltext` (tantivy) feature is disabled.
pub struct Bm25Store {
    inner: Mutex<Bm25Inner>,
    path: Option<std::path::PathBuf>,
}

struct Bm25Inner {
    /// term → Vec<(doc_id, term_frequency)>
    index: HashMap<String, Vec<(u32, f32)>>,
    /// doc_id → document metadata
    docs: HashMap<u32, DocMeta>,
    /// Total documents indexed
    doc_count: u32,
    /// Average document length (in tokens)
    avg_dl: f64,
    next_id: u32,
}

#[derive(Clone)]
struct DocMeta {
    uri: String,
    text: String,
    chunk_index: u32,
    content_type: sift_core::ContentType,
    file_type: String,
    title: Option<String>,
    byte_range: Option<(u64, u64)>,
    doc_len: u32,
}

/// Serializable representation of the BM25 index.
#[derive(Serialize, Deserialize)]
struct SerializableBm25 {
    doc_count: u32,
    avg_dl: f64,
    next_id: u32,
    docs: Vec<SerializableDoc>,
    index: HashMap<String, Vec<(u32, f32)>>,
}

#[derive(Serialize, Deserialize)]
struct SerializableDoc {
    id: u32,
    uri: String,
    text: String,
    chunk_index: u32,
    content_type: sift_core::ContentType,
    file_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_range: Option<(u64, u64)>,
    doc_len: u32,
}

/// BM25 parameters
const K1: f64 = 1.2;
const B: f64 = 0.75;

impl Bm25Store {
    pub fn open(path: &Path) -> SiftResult<Self> {
        let store = if path.exists() {
            let data = std::fs::read_to_string(path).map_err(sift_core::SiftError::Io)?;
            let inner = Self::deserialize(&data)?;
            Bm25Store {
                inner: Mutex::new(inner),
                path: Some(path.to_path_buf()),
            }
        } else {
            Bm25Store {
                inner: Mutex::new(Bm25Inner::new()),
                path: Some(path.to_path_buf()),
            }
        };
        Ok(store)
    }

    pub fn new() -> Self {
        Bm25Store {
            inner: Mutex::new(Bm25Inner::new()),
            path: None,
        }
    }

    fn save(&self) -> SiftResult<()> {
        if let Some(ref path) = self.path {
            let inner = self
                .inner
                .lock()
                .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
            let data = Self::serialize(&inner);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            sift_core::atomic_write(path, data.as_bytes())?;
        }
        Ok(())
    }

    /// Persist the current in-memory state to disk.
    ///
    /// Call this after a batch of insert/delete operations to avoid
    /// serializing the entire index on every mutation.
    pub fn flush(&self) -> SiftResult<()> {
        self.save()
    }

    fn serialize(inner: &Bm25Inner) -> String {
        let ser = SerializableBm25 {
            doc_count: inner.doc_count,
            avg_dl: inner.avg_dl,
            next_id: inner.next_id,
            docs: inner
                .docs
                .iter()
                .map(|(&id, meta)| SerializableDoc {
                    id,
                    uri: meta.uri.clone(),
                    text: meta.text.clone(),
                    chunk_index: meta.chunk_index,
                    content_type: meta.content_type,
                    file_type: meta.file_type.clone(),
                    title: meta.title.clone(),
                    byte_range: meta.byte_range,
                    doc_len: meta.doc_len,
                })
                .collect(),
            index: inner.index.clone(),
        };
        serde_json::to_string(&ser).expect("BM25 serialization cannot fail")
    }

    fn deserialize(data: &str) -> SiftResult<Bm25Inner> {
        let ser: SerializableBm25 =
            serde_json::from_str(data).map_err(|e| sift_core::SiftError::Storage(e.to_string()))?;

        let docs = ser
            .docs
            .into_iter()
            .map(|d| {
                (
                    d.id,
                    DocMeta {
                        uri: d.uri,
                        text: d.text,
                        chunk_index: d.chunk_index,
                        content_type: d.content_type,
                        file_type: d.file_type,
                        title: d.title,
                        byte_range: d.byte_range,
                        doc_len: d.doc_len,
                    },
                )
            })
            .collect();

        Ok(Bm25Inner {
            index: ser.index,
            docs,
            doc_count: ser.doc_count,
            avg_dl: ser.avg_dl,
            next_id: ser.next_id,
        })
    }
}

impl Bm25Inner {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            docs: HashMap::new(),
            doc_count: 0,
            avg_dl: 0.0,
            next_id: 0,
        }
    }

    fn add_doc(&mut self, meta: DocMeta) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let tokens = tokenize(&meta.text);
        let doc_len = tokens.len() as u32;
        let mut term_freqs: HashMap<&str, u32> = HashMap::new();
        for tok in &tokens {
            *term_freqs.entry(tok.as_str()).or_insert(0) += 1;
        }

        for (term, count) in term_freqs {
            let tf = count as f32;
            self.index
                .entry(term.to_string())
                .or_default()
                .push((id, tf));
        }

        // Update average document length
        let total_len = self.avg_dl * self.doc_count as f64 + doc_len as f64;
        self.doc_count += 1;
        self.avg_dl = total_len / self.doc_count as f64;

        let mut meta = meta;
        meta.doc_len = doc_len;
        self.docs.insert(id, meta);

        id
    }

    fn remove_doc(&mut self, doc_id: u32) {
        if let Some(meta) = self.docs.remove(&doc_id) {
            // Update avg_dl
            if self.doc_count > 1 {
                let total_len = self.avg_dl * self.doc_count as f64 - meta.doc_len as f64;
                self.doc_count -= 1;
                self.avg_dl = total_len / self.doc_count as f64;
            } else {
                self.doc_count = 0;
                self.avg_dl = 0.0;
            }

            // Remove from inverted index
            self.index.retain(|_, postings| {
                postings.retain(|(id, _)| *id != doc_id);
                !postings.is_empty()
            });
        }
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<(u32, f64)> {
        let query_terms = tokenize(query);
        let n = self.doc_count as f64;
        if n == 0.0 {
            return vec![];
        }

        let mut scores: HashMap<u32, f64> = HashMap::new();

        for term in &query_terms {
            if let Some(postings) = self.index.get(term.as_str()) {
                let df = postings.len() as f64;
                // IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_id, tf) in postings {
                    if let Some(meta) = self.docs.get(&doc_id) {
                        let dl = meta.doc_len as f64;
                        let tf = tf as f64;
                        // BM25 term score
                        let num = tf * (K1 + 1.0);
                        let denom = tf + K1 * (1.0 - B + B * dl / self.avg_dl);
                        let score = idf * num / denom;
                        *scores.entry(doc_id).or_insert(0.0) += score;
                    }
                }
            }
        }

        let mut results: Vec<(u32, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

impl FullTextStore for Bm25Store {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        for chunk in chunks {
            let meta = DocMeta {
                uri: chunk.chunk.source_uri.clone(),
                text: chunk.chunk.text.clone(),
                chunk_index: chunk.chunk.chunk_index,
                content_type: chunk.chunk.content_type,
                file_type: chunk.chunk.file_type.clone(),
                title: chunk.chunk.title.clone(),
                byte_range: chunk.chunk.byte_range,
                doc_len: 0, // set by add_doc
            };
            inner.add_doc(meta);
        }
        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> SiftResult<Vec<SearchResult>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        let scored = inner.search(query, top_k);

        let results = scored
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let meta = inner.docs.get(&doc_id)?;
                Some(SearchResult {
                    uri: meta.uri.clone(),
                    text: meta.text.clone(),
                    score: score as f32,
                    chunk_index: meta.chunk_index,
                    content_type: meta.content_type,
                    file_type: meta.file_type.clone(),
                    title: meta.title.clone(),
                    byte_range: meta.byte_range,
                })
            })
            .collect();

        Ok(results)
    }

    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        let ids_to_remove: Vec<u32> = inner
            .docs
            .iter()
            .filter(|(_, meta)| meta.uri == uri)
            .map(|(&id, _)| id)
            .collect();

        let count = ids_to_remove.len() as u64;
        for id in ids_to_remove {
            inner.remove_doc(id);
        }
        Ok(count)
    }

    fn flush(&self) -> SiftResult<()> {
        self.save()
    }
}

/// Simple tokenizer: lowercase, ASCII-fold, split on non-alphanumeric.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2)
        .map(|s| {
            s.chars()
                .map(|c| {
                    if c.is_ascii() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_lowercase().next().unwrap_or(c)
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::{Chunk, ContentType};

    fn make_chunk(uri: &str, text: &str) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: Chunk {
                text: text.to_string(),
                source_uri: uri.to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![],
        }
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char tokens are filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_insert_and_search() {
        let store = Bm25Store::new();
        let chunks = vec![
            make_chunk(
                "file:///payment.rs",
                "The payment processing system handles credit card transactions securely.",
            ),
            make_chunk(
                "file:///report.md",
                "Quarterly revenue projections show strong growth.",
            ),
            make_chunk(
                "file:///config.md",
                "Configure the database connection pool with max connections.",
            ),
        ];

        store.insert(&chunks).unwrap();

        let results = FullTextStore::search(&store, "payment credit card", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///payment.rs");

        let results = FullTextStore::search(&store, "quarterly revenue", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///report.md");

        let results = FullTextStore::search(&store, "database connection pool", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///config.md");
    }

    #[test]
    fn test_delete_by_uri() {
        let store = Bm25Store::new();
        store
            .insert(&[
                make_chunk("file:///a.txt", "hello world"),
                make_chunk("file:///b.txt", "goodbye world"),
            ])
            .unwrap();

        let deleted = store.delete_by_uri("file:///a.txt").unwrap();
        assert_eq!(deleted, 1);

        let results = FullTextStore::search(&store, "hello", 10).unwrap();
        assert!(results.is_empty());

        let results = FullTextStore::search(&store, "goodbye", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("bm25.json");

        // Create and populate
        {
            let store = Bm25Store::open(&path).unwrap();
            store
                .insert(&[make_chunk("file:///test.txt", "persistent search data")])
                .unwrap();
            store.flush().unwrap();
        }

        // Load and verify
        {
            let store = Bm25Store::open(&path).unwrap();
            let results = FullTextStore::search(&store, "persistent search", 10).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].uri, "file:///test.txt");
        }
    }

    #[test]
    fn test_empty_search() {
        let store = Bm25Store::new();
        let results = FullTextStore::search(&store, "anything", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scoring_favors_relevant_docs() {
        let store = Bm25Store::new();
        store
            .insert(&[
                make_chunk(
                    "file:///relevant.txt",
                    "rust programming language systems programming",
                ),
                make_chunk(
                    "file:///irrelevant.txt",
                    "cooking recipes for delicious meals",
                ),
            ])
            .unwrap();

        let results = FullTextStore::search(&store, "rust programming", 10).unwrap();
        assert_eq!(results[0].uri, "file:///relevant.txt");
    }
}
