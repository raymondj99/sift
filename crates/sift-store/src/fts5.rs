use crate::traits::FullTextStore;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;
use sift_core::{ContentType, EmbeddedChunk, SearchResult, SiftResult};

/// SQLite FTS5-backed full-text search store.
///
/// Provides BM25 keyword search using SQLite's built-in FTS5 extension.
/// Zero additional binary cost when rusqlite is already bundled.
pub struct Fts5Store {
    conn: Mutex<Connection>,
}

impl Fts5Store {
    pub fn open(path: &Path) -> SiftResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)
            .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 open error: {}", e)))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA cache_size=-8000;",
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 pragma error: {}", e)))?;

        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn open_in_memory() -> SiftResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 error: {}", e)))?;

        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn init_schema(conn: &Connection) -> SiftResult<()> {
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                 uri UNINDEXED,
                 text,
                 chunk_index UNINDEXED,
                 content_type UNINDEXED,
                 file_type UNINDEXED,
                 title,
                 tokenize = 'unicode61 remove_diacritics 2'
             );",
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 schema error: {}", e)))?;
        Ok(())
    }

    fn content_type_from_str(s: &str) -> ContentType {
        match s {
            "code" => ContentType::Code,
            "image" => ContentType::Image,
            "audio" => ContentType::Audio,
            "data" => ContentType::Data,
            _ => ContentType::Text,
        }
    }
}

impl FullTextStore for Fts5Store {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "INSERT INTO chunks_fts (uri, text, chunk_index, content_type, file_type, title)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )
            .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 prepare error: {}", e)))?;

        for chunk in chunks {
            stmt.execute(params![
                chunk.chunk.source_uri,
                chunk.chunk.text,
                chunk.chunk.chunk_index.to_string(),
                chunk.chunk.content_type.to_string(),
                chunk.chunk.file_type,
                chunk.chunk.title.as_deref().unwrap_or(""),
            ])
            .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 insert error: {}", e)))?;
        }

        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> SiftResult<Vec<SearchResult>> {
        let fts5_query = fts5_escape(query);
        if fts5_query.is_empty() {
            return Ok(vec![]);
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        // bm25() returns negative scores (lower = more relevant).
        // Negate for positive scores consistent with other stores.
        let mut stmt = conn
            .prepare(
                "SELECT uri, text, chunk_index, content_type, file_type, title, -bm25(chunks_fts)
                 FROM chunks_fts
                 WHERE chunks_fts MATCH ?1
                 ORDER BY bm25(chunks_fts)
                 LIMIT ?2",
            )
            .map_err(|e| sift_core::SiftError::Search(format!("FTS5 query error: {}", e)))?;

        let rows = stmt
            .query_map(params![fts5_query, top_k as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, f64>(6)?,
                ))
            })
            .map_err(|e| sift_core::SiftError::Search(format!("FTS5 search error: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            let (uri, text, chunk_index_str, content_type_str, file_type, title, score) =
                row.map_err(|e| sift_core::SiftError::Search(format!("FTS5 row error: {}", e)))?;

            results.push(SearchResult {
                uri,
                text,
                score: score as f32,
                chunk_index: chunk_index_str.parse().unwrap_or(0),
                content_type: Self::content_type_from_str(&content_type_str),
                file_type,
                title: if title.is_empty() { None } else { Some(title) },
                byte_range: None,
            });
        }

        Ok(results)
    }

    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let count = conn
            .execute("DELETE FROM chunks_fts WHERE uri = ?1", params![uri])
            .map_err(|e| sift_core::SiftError::Storage(format!("FTS5 delete error: {}", e)))?
            as u64;

        Ok(count)
    }
}

/// Escape a user query for FTS5 MATCH syntax.
///
/// Each word is quoted to prevent FTS5 syntax injection. Terms are joined
/// with OR for recall-oriented search (matches documents containing any term,
/// BM25 ranks documents with more matches higher).
fn fts5_escape(query: &str) -> String {
    let terms: Vec<String> = query
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(|word| {
            let escaped = word.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        })
        .collect();

    if terms.is_empty() {
        return String::new();
    }

    terms.join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::Chunk;

    fn make_embedded(uri: &str, text: &str, idx: u32) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: Chunk {
                text: text.to_string(),
                source_uri: uri.to_string(),
                chunk_index: idx,
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
    fn test_fts5_insert_and_search() {
        let store = Fts5Store::open_in_memory().unwrap();

        store
            .insert(&[
                make_embedded(
                    "file:///a.txt",
                    "the quick brown fox jumps over the lazy dog",
                    0,
                ),
                make_embedded("file:///b.txt", "rust programming language systems", 0),
            ])
            .unwrap();

        let results = store.search("quick brown fox", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///a.txt");
    }

    #[test]
    fn test_fts5_empty_search() {
        let store = Fts5Store::open_in_memory().unwrap();
        let results = store.search("nothing here", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts5_single_char_terms_filtered() {
        let store = Fts5Store::open_in_memory().unwrap();
        // Single-char query should return empty (filtered by fts5_escape)
        let results = store.search("a b c", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts5_delete() {
        let store = Fts5Store::open_in_memory().unwrap();

        store
            .insert(&[
                make_embedded("file:///a.txt", "hello world greeting", 0),
                make_embedded("file:///b.txt", "goodbye world farewell", 0),
            ])
            .unwrap();

        let deleted = store.delete_by_uri("file:///a.txt").unwrap();
        assert_eq!(deleted, 1);

        let results = store.search("hello greeting", 10).unwrap();
        assert!(results.is_empty());

        let results = store.search("goodbye farewell", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fts5_bm25_ranking() {
        let store = Fts5Store::open_in_memory().unwrap();

        store
            .insert(&[
                make_embedded(
                    "file:///relevant.txt",
                    "rust programming language systems programming",
                    0,
                ),
                make_embedded(
                    "file:///irrelevant.txt",
                    "cooking recipes for delicious meals",
                    0,
                ),
            ])
            .unwrap();

        let results = store.search("rust programming", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///relevant.txt");
    }

    #[test]
    fn test_fts5_special_characters() {
        let store = Fts5Store::open_in_memory().unwrap();

        store
            .insert(&[make_embedded(
                "file:///code.rs",
                "handling C++ templates and operator overloading",
                0,
            )])
            .unwrap();

        // Special chars in query should not cause errors
        let results = store.search("C++ templates", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fts5_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("fts5.db");

        {
            let store = Fts5Store::open(&path).unwrap();
            store
                .insert(&[make_embedded(
                    "file:///test.txt",
                    "persistent search data indexed",
                    0,
                )])
                .unwrap();
        }

        {
            let store = Fts5Store::open(&path).unwrap();
            let results = store.search("persistent search", 10).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].uri, "file:///test.txt");
        }
    }

    #[test]
    fn test_fts5_escape_function() {
        assert_eq!(fts5_escape("hello world"), "\"hello\" OR \"world\"");
        assert_eq!(fts5_escape("a b c"), ""); // single chars filtered
        assert_eq!(fts5_escape("rust"), "\"rust\"");
        assert_eq!(
            fts5_escape("hello \"world\""),
            "\"hello\" OR \"\"\"world\"\"\""
        );
        assert_eq!(fts5_escape(""), "");
        assert_eq!(fts5_escape("   "), "");
    }

    #[test]
    fn test_fts5_title_search() {
        let store = Fts5Store::open_in_memory().unwrap();

        let chunk = EmbeddedChunk {
            chunk: Chunk {
                text: "some body text content".to_string(),
                source_uri: "file:///doc.md".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "md".to_string(),
                title: Some("Architecture Overview".to_string()),
                language: None,
                byte_range: None,
            },
            vector: vec![],
        };

        store.insert(&[chunk]).unwrap();

        // Search by title content
        let results = store.search("architecture overview", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].uri, "file:///doc.md");
    }
}
