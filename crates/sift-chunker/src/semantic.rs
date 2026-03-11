use crate::traits::Chunker;

/// Semantic chunker that splits on paragraph and section boundaries.
/// Tries to keep semantically related content together.
pub struct SemanticChunker {
    max_chunk_size: usize,
    overlap: usize,
}

impl SemanticChunker {
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size: max_chunk_size.max(1),
            overlap: overlap.min(max_chunk_size / 2),
        }
    }

    /// Score how "good" a split point is. Higher = better.
    fn split_quality(text: &str, pos: usize) -> u32 {
        let bytes = text.as_bytes();

        // Check what's at/around this position
        if pos >= text.len() || !text.is_char_boundary(pos) {
            return 0;
        }

        // Double newline (paragraph break) - best split
        if pos + 1 < text.len() && bytes[pos] == b'\n' && bytes[pos + 1] == b'\n' {
            return 100;
        }

        // Markdown heading
        if pos + 2 < text.len() && bytes[pos] == b'\n' && bytes[pos + 1] == b'#' {
            return 95;
        }

        // Function/class definition (code)
        let line_start = text[..pos].rfind('\n').map_or(0, |p| p + 1);
        if !text.is_char_boundary(line_start) {
            return 0;
        }
        let line = &text[line_start..pos];
        if line.starts_with("fn ")
            || line.starts_with("def ")
            || line.starts_with("class ")
            || line.starts_with("pub fn ")
            || line.starts_with("pub struct ")
            || line.starts_with("impl ")
            || line.starts_with("func ")
            || line.starts_with("function ")
        {
            return 90;
        }

        // Single newline
        if bytes[pos] == b'\n' {
            return 50;
        }

        // Sentence end (period followed by space)
        if bytes[pos] == b'.' && pos + 1 < text.len() && bytes[pos + 1] == b' ' {
            return 40;
        }

        // Space
        if bytes[pos] == b' ' {
            return 10;
        }

        0
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Vec<(String, usize)> {
        if text.is_empty() {
            return vec![];
        }

        if text.len() <= self.max_chunk_size {
            return vec![(text.to_string(), 0)];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + self.max_chunk_size).min(text.len());
            let end = snap_to_char_boundary(text, end);

            if end >= text.len() {
                let chunk = text[start..].trim();
                if !chunk.is_empty() {
                    chunks.push((chunk.to_string(), start));
                }
                break;
            }

            // Find the best split point in the last quarter of the chunk
            let search_start = snap_to_char_boundary(text, start + (self.max_chunk_size * 3 / 4));
            let mut best_pos = end;
            let mut best_quality = 0u32;

            for pos in search_start..end {
                if !text.is_char_boundary(pos) {
                    continue;
                }
                let quality = Self::split_quality(text, pos);
                if quality > best_quality {
                    best_quality = quality;
                    best_pos = pos;
                    // Paragraph break is good enough, take it immediately
                    if quality >= 90 {
                        break;
                    }
                }
            }

            // If we found a paragraph break or heading, split after the newline(s)
            let split_at = if best_quality >= 50 {
                // Skip past the newline(s)
                let mut skip = best_pos;
                while skip < text.len() && text.as_bytes().get(skip) == Some(&b'\n') {
                    skip += 1;
                }
                snap_to_char_boundary(text, skip)
            } else {
                snap_to_char_boundary(text, best_pos)
            };

            // Guard: ensure forward progress
            if split_at <= start {
                // Force advance past current position
                let next = start + 1;
                start = snap_to_char_boundary(text, next.min(text.len()));
                continue;
            }

            let chunk = text[start..split_at].trim();
            if !chunk.is_empty() {
                chunks.push((chunk.to_string(), start));
            }

            // Apply overlap
            start = if split_at > self.overlap {
                let overlap_target = split_at - self.overlap;
                let overlap_target = snap_to_char_boundary(text, overlap_target);
                // Try to start overlap at a paragraph/line boundary
                if overlap_target < split_at {
                    text[overlap_target..split_at]
                        .find('\n')
                        .map_or(split_at, |p| overlap_target + p + 1)
                } else {
                    split_at
                }
            } else {
                split_at
            };
            start = snap_to_char_boundary(text, start);
        }

        chunks
    }

    fn name(&self) -> &'static str {
        "semantic"
    }
}

/// Snap a byte index to the nearest valid UTF-8 char boundary (rounding down).
fn snap_to_char_boundary(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return text.len();
    }
    let mut p = pos;
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text() {
        let chunker = SemanticChunker::new(1000, 0);
        let chunks = chunker.chunk("Hello world");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_paragraph_splitting() {
        let chunker = SemanticChunker::new(50, 0);
        let text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_code_splitting() {
        let chunker = SemanticChunker::new(60, 0);
        let text =
            "fn foo() {\n    println!(\"hello\");\n}\n\nfn bar() {\n    println!(\"world\");\n}\n";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty() {
        let chunker = SemanticChunker::new(100, 10);
        assert!(chunker.chunk("").is_empty());
    }

    #[test]
    fn test_offsets_increase() {
        let chunker = SemanticChunker::new(30, 0);
        let text = "Line one here.\n\nLine two here.\n\nLine three here.\n\nLine four here.";
        let chunks = chunker.chunk(text);
        if chunks.len() > 1 {
            for i in 1..chunks.len() {
                assert!(
                    chunks[i].1 >= chunks[i - 1].1,
                    "Offsets should be non-decreasing"
                );
            }
        }
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn never_panics(text in "\\PC{0,1000}", max_size in 10..500usize, overlap in 0..100usize) {
                let chunker = SemanticChunker::new(max_size, overlap);
                let _ = chunker.chunk(&text);
            }

            #[test]
            fn offsets_within_bounds(text in "\\PC{1,500}", max_size in 10..200usize) {
                let chunker = SemanticChunker::new(max_size, 0);
                let chunks = chunker.chunk(&text);
                for (_, offset) in &chunks {
                    prop_assert!(*offset <= text.len(), "Offset {} exceeds text len {}", offset, text.len());
                }
            }

            #[test]
            fn split_quality_never_panics(text in "\\PC{1,500}", pos in 0..500usize) {
                if pos < text.len() {
                    let _ = SemanticChunker::split_quality(&text, pos);
                }
            }
        }
    }
}
