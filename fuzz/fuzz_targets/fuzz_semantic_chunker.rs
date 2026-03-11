#![no_main]

use libfuzzer_sys::fuzz_target;
use sift_chunker::{Chunker, SemanticChunker};

fuzz_target!(|data: &[u8]| {
    // Only process valid UTF-8; the chunker operates on &str.
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Exercise the chunker at several representative sizes.
    for &chunk_size in &[16, 64, 256, 1024] {
        for &overlap in &[0, 8, 32] {
            let chunker = SemanticChunker::new(chunk_size, overlap);
            let chunks = chunker.chunk(text);

            // Basic sanity checks that should always hold.
            for (chunk_text, offset) in &chunks {
                assert!(!chunk_text.is_empty(), "chunks must not be empty");
                assert!(*offset <= text.len(), "offset out of bounds");
            }

            // Offsets must be non-decreasing.
            for window in chunks.windows(2) {
                assert!(
                    window[1].1 >= window[0].1,
                    "offsets must be non-decreasing"
                );
            }
        }
    }
});
