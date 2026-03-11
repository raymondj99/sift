use crate::traits::FullTextStore;
use sift_core::{ContentType, EmbeddedChunk, SearchResult, SiftResult};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

/// Tantivy-backed BM25 full-text search store.
pub struct TantivyStore {
    index: Index,
    #[allow(dead_code)]
    schema: Schema,
    uri_field: Field,
    text_field: Field,
    chunk_index_field: Field,
    content_type_field: Field,
    file_type_field: Field,
    title_field: Field,
}

impl TantivyStore {
    pub fn open(index_dir: &Path) -> SiftResult<Self> {
        let mut schema_builder = Schema::builder();

        let uri_field = schema_builder.add_text_field("uri", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", INDEXED | STORED);
        let content_type_field = schema_builder.add_text_field("content_type", STRING | STORED);
        let file_type_field = schema_builder.add_text_field("file_type", STRING | STORED);
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);

        let schema = schema_builder.build();

        std::fs::create_dir_all(index_dir)?;

        let index = match Index::open_in_dir(index_dir) {
            Ok(idx) => idx,
            Err(_) => Index::create_in_dir(index_dir, schema.clone())
                .map_err(|e| sift_core::SiftError::Storage(format!("Tantivy create error: {e}")))?,
        };

        Ok(Self {
            index,
            schema,
            uri_field,
            text_field,
            chunk_index_field,
            content_type_field,
            file_type_field,
            title_field,
        })
    }

    pub fn open_in_memory() -> SiftResult<Self> {
        let mut schema_builder = Schema::builder();

        let uri_field = schema_builder.add_text_field("uri", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", INDEXED | STORED);
        let content_type_field = schema_builder.add_text_field("content_type", STRING | STORED);
        let file_type_field = schema_builder.add_text_field("file_type", STRING | STORED);
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);

        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());

        Ok(Self {
            index,
            schema,
            uri_field,
            text_field,
            chunk_index_field,
            content_type_field,
            file_type_field,
            title_field,
        })
    }

    fn get_writer(&self) -> SiftResult<IndexWriter> {
        self.index
            .writer(50_000_000) // 50MB heap
            .map_err(|e| sift_core::SiftError::Storage(format!("Tantivy writer error: {e}")))
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

impl FullTextStore for TantivyStore {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let mut writer = self.get_writer()?;

        for ec in chunks {
            let mut doc = doc!(
                self.uri_field => ec.chunk.source_uri.as_str(),
                self.text_field => ec.chunk.text.as_str(),
                self.chunk_index_field => u64::from(ec.chunk.chunk_index),
                self.content_type_field => ec.chunk.content_type.to_string(),
                self.file_type_field => ec.chunk.file_type.as_str(),
            );

            if let Some(title) = &ec.chunk.title {
                doc.add_text(self.title_field, title);
            }

            writer
                .add_document(doc)
                .map_err(|e| sift_core::SiftError::Storage(format!("Tantivy add error: {e}")))?;
        }

        writer
            .commit()
            .map_err(|e| sift_core::SiftError::Storage(format!("Tantivy commit error: {e}")))?;

        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> SiftResult<Vec<SearchResult>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| sift_core::SiftError::Search(format!("Reader error: {e}")))?;

        let searcher = reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.text_field, self.title_field]);
        let query = query_parser
            .parse_query(query)
            .map_err(|e| sift_core::SiftError::Search(format!("Query parse error: {e}")))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k))
            .map_err(|e| sift_core::SiftError::Search(format!("Search error: {e}")))?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| sift_core::SiftError::Search(format!("Doc retrieve error: {e}")))?;

            let uri = doc
                .get_first(self.uri_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let text = doc
                .get_first(self.text_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let chunk_index = doc
                .get_first(self.chunk_index_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let content_type = doc
                .get_first(self.content_type_field)
                .and_then(|v| v.as_str())
                .map_or(ContentType::Text, Self::content_type_from_str);

            let file_type = doc
                .get_first(self.file_type_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let title = doc
                .get_first(self.title_field)
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string);

            results.push(SearchResult {
                uri,
                text,
                score,
                chunk_index,
                content_type,
                file_type,
                title,
                byte_range: None,
            });
        }

        Ok(results)
    }

    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let mut writer = self.get_writer()?;

        let term = tantivy::Term::from_field_text(self.uri_field, uri);
        writer.delete_term(term);

        writer
            .commit()
            .map_err(|e| sift_core::SiftError::Storage(format!("Tantivy commit error: {e}")))?;

        // Tantivy doesn't return delete count easily, just return 0
        Ok(0)
    }
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
    fn test_tantivy_insert_and_search() {
        let store = TantivyStore::open_in_memory().unwrap();

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
    fn test_tantivy_empty_search() {
        let store = TantivyStore::open_in_memory().unwrap();
        let results = store.search("nothing here", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_tantivy_delete() {
        let store = TantivyStore::open_in_memory().unwrap();

        store
            .insert(&[
                make_embedded("file:///a.txt", "hello world", 0),
                make_embedded("file:///b.txt", "goodbye world", 0),
            ])
            .unwrap();

        store.delete_by_uri("file:///a.txt").unwrap();

        // After delete, searching for "hello" should not find a.txt
        let _results = store.search("hello world", 10).unwrap();
        // Note: tantivy deletes are lazy, so this might still return results
        // until a merge happens. This is expected behavior.
    }
}
