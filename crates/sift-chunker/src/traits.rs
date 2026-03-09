/// A text chunker that splits content into embedding-sized pieces.
pub trait Chunker: Send + Sync {
    /// Split text into chunks. Returns (chunk_text, byte_offset) pairs.
    fn chunk(&self, text: &str) -> Vec<(String, usize)>;

    /// Split text with a language hint (e.g. file extension like "rs", "py").
    /// Default implementation ignores the hint and calls `chunk`.
    fn chunk_with_language(&self, text: &str, _language: Option<&str>) -> Vec<(String, usize)> {
        self.chunk(text)
    }

    fn name(&self) -> &str;
}
