use sift_core::{ParsedDocument, SiftResult};

/// A parser that extracts text content from a specific file format.
pub trait Parser: Send + Sync {
    /// Returns true if this parser can handle the given MIME type or extension.
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool;

    /// Parse the file content and return extracted text + metadata.
    fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument>;

    /// Human-readable name for this parser.
    fn name(&self) -> &str;
}
