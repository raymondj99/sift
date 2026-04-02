use crate::traits::Chunker;

/// Recursive character text splitter, similar to `LangChain`'s
/// `RecursiveCharacterTextSplitter`.
///
/// Splits text using a hierarchy of separators, attempting the first separator
/// first and falling back to subsequent separators for pieces that are still
/// too large.
pub struct RecursiveChunker {
    /// Target maximum chunk size in characters.
    chunk_size: usize,
    /// Overlap between consecutive chunks in characters.
    chunk_overlap: usize,
    /// Ordered list of separators to try, from coarsest to finest.
    /// Default: `["\n\n", "\n", ". ", " ", ""]`
    separators: Vec<String>,
}

impl RecursiveChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap: chunk_overlap.min(chunk_size / 2),
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
                String::new(),
            ],
        }
    }

    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    /// Core recursive splitting algorithm.
    ///
    /// Given a piece of text and a separator index, split on that separator.
    /// Any resulting piece that still exceeds `chunk_size` is recursively
    /// split using the next separator in the hierarchy. The final fallback
    /// (empty-string separator) splits character-by-character.
    fn split_recursive(&self, text: &str, sep_idx: usize) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return if text.is_empty() {
                vec![]
            } else {
                vec![text.to_string()]
            };
        }

        // If we have exhausted all separators, force-split at chunk_size
        // boundaries on character boundaries.
        if sep_idx >= self.separators.len() {
            return self.force_split(text);
        }

        let sep = &self.separators[sep_idx];

        // Empty-string separator means character-level splitting.
        if sep.is_empty() {
            return self.force_split(text);
        }

        let pieces: Vec<&str> = text.split(sep.as_str()).collect();

        // If the separator did not actually split anything, try the next one.
        if pieces.len() <= 1 {
            return self.split_recursive(text, sep_idx + 1);
        }

        // Merge small consecutive pieces back together up to chunk_size,
        // re-inserting the separator between them.
        let mut merged: Vec<String> = Vec::new();
        let mut current = String::new();

        for (i, piece) in pieces.iter().enumerate() {
            let candidate = if current.is_empty() {
                piece.to_string()
            } else {
                format!("{current}{sep}{piece}")
            };

            if candidate.len() <= self.chunk_size {
                current = candidate;
            } else {
                // Flush current if non-empty
                if !current.is_empty() {
                    merged.push(current);
                    current = String::new();
                }
                // If the individual piece itself exceeds chunk_size,
                // recursively split it with the next separator.
                if piece.len() > self.chunk_size {
                    let sub_pieces = self.split_recursive(piece, sep_idx + 1);
                    merged.extend(sub_pieces);
                } else if !piece.is_empty() {
                    current = piece.to_string();
                }
            }

            // If this is the last piece, flush.
            if i == pieces.len() - 1 && !current.is_empty() {
                merged.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            merged.push(current);
        }

        merged
    }

    /// Force-split text into chunks of at most `chunk_size` characters,
    /// respecting UTF-8 char boundaries.
    fn force_split(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            if !chunk.is_empty() {
                result.push(chunk);
            }
            start = end;
        }
        result
    }

    /// Apply chunk overlap: given a list of non-overlapping chunks,
    /// produce chunks where each chunk (after the first) starts with
    /// the last `chunk_overlap` characters of the previous chunk.
    fn apply_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if self.chunk_overlap == 0 || chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::with_capacity(chunks.len());
        result.push(chunks[0].clone());

        for i in 1..chunks.len() {
            let prev = &chunks[i - 1];
            let prev_chars: Vec<char> = prev.chars().collect();
            let overlap_start = prev_chars.len().saturating_sub(self.chunk_overlap);
            let overlap: String = prev_chars[overlap_start..].iter().collect();
            let merged = format!("{}{}", overlap, &chunks[i]);
            result.push(merged);
        }

        result
    }

    /// Compute byte offsets for each chunk by scanning through the original
    /// text once with a cursor, rather than calling `text.find()` per chunk.
    /// This is O(n) where n = text length, avoiding the previous O(n × m)
    /// substring search.
    fn compute_offsets(&self, text: &str, chunks: &[String]) -> Vec<usize> {
        if chunks.is_empty() {
            return vec![];
        }

        let mut offsets = Vec::with_capacity(chunks.len());
        let mut cursor: usize = 0;

        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 || self.chunk_overlap == 0 {
                // First chunk or no overlap: locate directly.
                if let Some(pos) = text[cursor..].find(chunk.as_str()) {
                    offsets.push(cursor + pos);
                    cursor = cursor + pos + chunk.len();
                } else {
                    offsets.push(cursor);
                }
            } else {
                // With overlap: skip the overlap prefix and locate the new content.
                // The new content starts after `chunk_overlap` chars of the overlap prefix.
                let new_byte_start: usize = chunk
                    .char_indices()
                    .nth(self.chunk_overlap)
                    .map_or(chunk.len(), |(idx, _)| idx);
                let new_content = &chunk[new_byte_start..];

                if !new_content.is_empty() {
                    if let Some(pos) = text[cursor..].find(new_content) {
                        let new_offset = cursor + pos;
                        // The chunk starts `overlap` chars before the new content.
                        let actual_offset = new_offset.saturating_sub(new_byte_start);
                        offsets.push(actual_offset);
                        cursor = new_offset;
                    } else {
                        offsets.push(cursor);
                    }
                } else {
                    offsets.push(cursor);
                }
            }
        }

        offsets
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Vec<(String, usize)> {
        if text.is_empty() {
            return vec![];
        }

        if text.len() <= self.chunk_size {
            return vec![(text.to_string(), 0)];
        }

        let raw_chunks = self.split_recursive(text, 0);

        if raw_chunks.is_empty() {
            return vec![];
        }

        let chunks = self.apply_overlap(raw_chunks);
        let offsets = self.compute_offsets(text, &chunks);

        chunks
            .into_iter()
            .zip(offsets)
            .filter(|(c, _)| !c.trim().is_empty())
            .collect()
    }

    fn name(&self) -> &'static str {
        "recursive"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunks_do_not_exceed_target_size() {
        let chunk_size = 50;
        let chunker = RecursiveChunker::new(chunk_size, 0);
        let text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph that also has text.\n\nFourth paragraph for good measure.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2, "Should produce multiple chunks");
        for (chunk_text, _) in &chunks {
            assert!(
                chunk_text.len() <= chunk_size,
                "Chunk '{}' has len {} which exceeds target {}",
                chunk_text,
                chunk_text.len(),
                chunk_size
            );
        }
    }

    #[test]
    fn test_overlap_works() {
        let chunker = RecursiveChunker::new(40, 10);
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2, "Should produce multiple chunks");

        // Check that overlap produces content shared between consecutive chunks
        for i in 1..chunks.len() {
            let prev = &chunks[i - 1].0;
            let curr = &chunks[i].0;
            // The end of the previous chunk should overlap with the start of the current chunk
            let prev_end_chars: Vec<char> = prev.chars().collect();
            let overlap_portion: String = prev_end_chars[prev_end_chars.len().saturating_sub(10)..]
                .iter()
                .collect();
            assert!(
                curr.starts_with(&overlap_portion) || curr.contains(&overlap_portion),
                "Chunk {} should contain overlap from chunk {}: overlap='{}', current='{}'",
                i,
                i - 1,
                overlap_portion,
                curr
            );
        }
    }

    #[test]
    fn test_splits_on_double_newline() {
        let chunker = RecursiveChunker::new(60, 0);
        let text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        // First chunk should contain paragraph one
        assert!(chunks[0].0.contains("Paragraph one"));
    }

    #[test]
    fn test_byte_offsets_are_within_bounds() {
        let chunker = RecursiveChunker::new(30, 0);
        let text = "Hello world.\n\nGoodbye world.\n\nThe end of the story.";
        let chunks = chunker.chunk(text);
        for (chunk_text, offset) in &chunks {
            assert!(
                *offset < text.len(),
                "Offset {} out of bounds for text len {}",
                offset,
                text.len()
            );
            assert!(!chunk_text.is_empty(), "No chunk should be empty");
        }
    }

    #[test]
    fn test_unicode_text() {
        let chunker = RecursiveChunker::new(20, 0);
        let text = "Hello world cafe\n\nGoodbye world encore";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        let all: String = chunks
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all.contains("Hello"));
        assert!(all.contains("Goodbye"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn offsets_within_bounds(text in "\\PC{1,500}", chunk_size in 10..200usize) {
                let chunker = RecursiveChunker::new(chunk_size, 0);
                let chunks = chunker.chunk(&text);
                for (_, offset) in &chunks {
                    prop_assert!(*offset <= text.len(), "Offset {} exceeds text len {}", offset, text.len());
                }
            }

            #[test]
            fn never_panics(text in "\\PC{0,1000}", chunk_size in 1..500usize, overlap in 0..100usize) {
                let chunker = RecursiveChunker::new(chunk_size, overlap);
                let _ = chunker.chunk(&text);
            }
        }
    }
}
