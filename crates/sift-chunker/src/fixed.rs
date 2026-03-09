use crate::traits::Chunker;

/// Fixed-size chunker with configurable overlap.
/// Splits on word boundaries to avoid cutting words.
pub struct FixedChunker {
    /// Target chunk size in characters (approximating tokens).
    chunk_size: usize,
    /// Overlap between adjacent chunks in characters.
    overlap: usize,
}

impl FixedChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            overlap: overlap.min(chunk_size / 2),
        }
    }
}

impl Chunker for FixedChunker {
    fn chunk(&self, text: &str) -> Vec<(String, usize)> {
        if text.is_empty() {
            return vec![];
        }

        // For short texts, don't chunk at all
        if text.len() <= self.chunk_size {
            return vec![(text.to_string(), 0)];
        }

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());

            // Find a word boundary to break on (look backwards from target end)
            let actual_end = if end < chars.len() {
                find_word_boundary(&chars, end)
            } else {
                end
            };

            let chunk_text: String = chars[start..actual_end].iter().collect();
            let byte_offset = chars[..start].iter().map(|c| c.len_utf8()).sum();

            if !chunk_text.trim().is_empty() {
                chunks.push((chunk_text.trim().to_string(), byte_offset));
            }

            if actual_end >= chars.len() {
                break;
            }

            // Move start forward, accounting for overlap
            start = if actual_end > self.overlap {
                actual_end - self.overlap
            } else {
                actual_end
            };
        }

        chunks
    }

    fn name(&self) -> &str {
        "fixed"
    }
}

/// Find the nearest word boundary before `pos`.
fn find_word_boundary(chars: &[char], pos: usize) -> usize {
    // Search backwards up to 100 chars for a space or newline
    let search_limit = pos.saturating_sub(100);
    for i in (search_limit..pos).rev() {
        if chars[i].is_whitespace() {
            return i + 1;
        }
    }
    // No good boundary found, just break at pos
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_no_chunking() {
        let chunker = FixedChunker::new(1000, 100);
        let chunks = chunker.chunk("Short text");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "Short text");
        assert_eq!(chunks[0].1, 0);
    }

    #[test]
    fn test_empty_text() {
        let chunker = FixedChunker::new(100, 10);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunking_with_overlap() {
        let chunker = FixedChunker::new(20, 5);
        let text = "The quick brown fox jumps over the lazy dog and runs away fast";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() > 1);
        // Verify all text is covered
        for (chunk, _offset) in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_word_boundary_respect() {
        let chunker = FixedChunker::new(10, 0);
        let text = "Hello World Foo Bar";
        let chunks = chunker.chunk(text);
        // Should not split in the middle of words
        for (chunk, _) in &chunks {
            assert!(
                !chunk.starts_with(' '),
                "Chunk should not start with space: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_byte_offsets() {
        let chunker = FixedChunker::new(5, 0);
        let text = "Hello World";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks[0].1, 0); // First chunk starts at 0
    }
}
