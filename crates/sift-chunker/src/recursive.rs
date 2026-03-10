use crate::traits::Chunker;

/// Recursive character text splitter, similar to LangChain's
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
                format!("{}{}{}", current, sep, piece)
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

    /// Compute byte offsets for each chunk within the original text.
    /// Because overlap prepends content from the previous chunk, we
    /// search for the non-overlapping part of each chunk to determine
    /// its position.
    fn compute_offsets(&self, text: &str, chunks: &[String]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(chunks.len());
        let mut search_from: usize = 0;

        for chunk in chunks {
            // For chunks with overlap, the "new" content starts after the
            // overlap portion. But the byte offset should point to where
            // this chunk's content begins in the original text.
            // We find the chunk's non-overlap content in the source.
            if self.chunk_overlap > 0 && !offsets.is_empty() {
                // The overlap portion came from the previous chunk;
                // the new content starts after `chunk_overlap` chars.
                let chunk_chars: Vec<char> = chunk.chars().collect();
                let new_start = self.chunk_overlap.min(chunk_chars.len());
                let new_content: String = chunk_chars[new_start..].iter().collect();

                if let Some(pos) = text[search_from..].find(&new_content) {
                    let offset = search_from + pos;
                    // The chunk actually starts `chunk_overlap` chars before
                    // the new content. Compute byte offset of the overlap start.
                    let byte_overlap: usize =
                        chunk_chars[..new_start].iter().map(|c| c.len_utf8()).sum();
                    let actual_offset = offset.saturating_sub(byte_overlap);
                    offsets.push(actual_offset);
                    search_from = offset;
                } else {
                    // Fallback: approximate from search_from
                    offsets.push(search_from);
                }
            } else {
                // No overlap for the first chunk; find it directly.
                if let Some(pos) = text[search_from..].find(chunk.as_str()) {
                    offsets.push(search_from + pos);
                    search_from = search_from + pos + chunk.len();
                } else {
                    // Fallback: use current search position
                    offsets.push(search_from);
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

    fn name(&self) -> &str {
        "recursive"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunker = RecursiveChunker::new(100, 0);
        assert!(chunker.chunk("").is_empty());
    }

    #[test]
    fn test_short_text_no_chunking() {
        let chunker = RecursiveChunker::new(100, 0);
        let chunks = chunker.chunk("Hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "Hello world");
        assert_eq!(chunks[0].1, 0);
    }

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
    fn test_splits_on_single_newline_when_needed() {
        let chunker = RecursiveChunker::new(30, 0);
        let text = "Line one here\nLine two here\nLine three here\nLine four here";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        // Should split on \n since no \n\n is present and lines are too long together
        let all: String = chunks
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(all.contains("Line one"));
        assert!(all.contains("Line four"));
    }

    #[test]
    fn test_splits_on_sentence_boundary() {
        let chunker = RecursiveChunker::new(40, 0);
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth one.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_splits_on_space() {
        let chunker = RecursiveChunker::new(15, 0);
        let text = "word1 word2 word3 word4 word5 word6";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        for (chunk_text, _) in &chunks {
            assert!(chunk_text.len() <= 15);
        }
    }

    #[test]
    fn test_force_splits_long_word() {
        let chunker = RecursiveChunker::new(10, 0);
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        for (chunk_text, _) in &chunks {
            assert!(
                chunk_text.len() <= 10,
                "Chunk '{}' exceeds limit",
                chunk_text
            );
        }
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
    fn test_offsets_are_non_decreasing() {
        let chunker = RecursiveChunker::new(40, 0);
        let text = "Alpha content.\n\nBravo content.\n\nCharlie content.\n\nDelta content.";
        let chunks = chunker.chunk(text);
        for i in 1..chunks.len() {
            assert!(
                chunks[i].1 >= chunks[i - 1].1,
                "Offsets should be non-decreasing: {} < {}",
                chunks[i].1,
                chunks[i - 1].1
            );
        }
    }

    #[test]
    fn test_custom_separators() {
        let chunker = RecursiveChunker::new(20, 0).with_separators(vec![
            "---".to_string(),
            " ".to_string(),
            String::new(),
        ]);
        let text = "part one---part two---part three";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].0.contains("part one"));
    }

    #[test]
    fn test_all_content_preserved_no_overlap() {
        let chunker = RecursiveChunker::new(30, 0);
        let text =
            "The quick brown fox jumps over the lazy dog and runs away fast across the field.";
        let chunks = chunker.chunk(text);
        // All words from the original should appear somewhere in the chunks
        for word in text.split_whitespace() {
            let found = chunks.iter().any(|(c, _)| c.contains(word));
            assert!(found, "Word '{}' should be in some chunk", word);
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
            fn produces_nonempty_output_for_nonempty_input(text in "[a-zA-Z0-9 ]{1,500}", chunk_size in 20..200usize) {
                let trimmed = text.trim();
                let chunker = RecursiveChunker::new(chunk_size, 0);
                let chunks = chunker.chunk(&text);
                if !trimmed.is_empty() {
                    prop_assert!(!chunks.is_empty(), "Non-empty input should produce chunks");
                }
            }

            #[test]
            fn offsets_within_bounds(text in "\\PC{1,500}", chunk_size in 10..200usize) {
                let chunker = RecursiveChunker::new(chunk_size, 0);
                let chunks = chunker.chunk(&text);
                for (_, offset) in &chunks {
                    prop_assert!(*offset <= text.len(), "Offset {} exceeds text len {}", offset, text.len());
                }
            }

            #[test]
            fn offsets_non_decreasing(text in "\\PC{1,500}", chunk_size in 10..200usize) {
                let chunker = RecursiveChunker::new(chunk_size, 0);
                let chunks = chunker.chunk(&text);
                for i in 1..chunks.len() {
                    prop_assert!(chunks[i].1 >= chunks[i - 1].1,
                        "Offsets not non-decreasing: {} < {}", chunks[i].1, chunks[i - 1].1);
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
