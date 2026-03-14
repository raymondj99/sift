#[cfg(feature = "archive")]
use crate::archive::ArchiveParser;
#[cfg(feature = "audio")]
use crate::audio::AudioParser;
#[cfg(feature = "data")]
use crate::data::DataParser;
#[cfg(feature = "email")]
use crate::email::EmailParser;
#[cfg(feature = "epub")]
use crate::epub::EpubParser;
use crate::notebook::NotebookParser;
#[cfg(feature = "office")]
use crate::office::OfficeParser;
#[cfg(feature = "pdf")]
use crate::pdf::PdfParser;
use crate::rtf::RtfParser;
#[cfg(feature = "spreadsheets")]
use crate::spreadsheet::SpreadsheetParser;
use crate::traits::Parser;
use crate::{code::CodeParser, image::ImageParser, text::TextParser, web::WebParser};
use sift_core::{ParsedDocument, SiftResult};
use tracing::debug;

/// Registry of all available parsers. Routes files to the correct parser
/// based on MIME type and extension.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn Parser>>,
}

impl ParserRegistry {
    pub fn new() -> Self {
        // Order matters: more specific parsers first, binary formats before text
        #[allow(unused_mut)]
        let mut parsers: Vec<Box<dyn Parser>> = vec![
            Box::new(ImageParser),
            #[cfg(feature = "pdf")]
            Box::new(PdfParser),
            #[cfg(feature = "office")]
            Box::new(OfficeParser),
            #[cfg(feature = "epub")]
            Box::new(EpubParser),
            #[cfg(feature = "spreadsheets")]
            Box::new(SpreadsheetParser),
            #[cfg(feature = "archive")]
            Box::new(ArchiveParser),
            #[cfg(feature = "email")]
            Box::new(EmailParser),
            Box::new(NotebookParser),
            Box::new(RtfParser),
            Box::new(CodeParser),
            #[cfg(feature = "data")]
            Box::new(DataParser),
            Box::new(WebParser),
            Box::new(TextParser),
        ];

        // Conditionally add audio parser before text fallback
        #[cfg(feature = "audio")]
        {
            // Insert audio parser before the text fallback (second to last)
            let text_idx = parsers.len() - 1;
            parsers.insert(text_idx, Box::new(AudioParser));
        }

        Self { parsers }
    }

    /// Find a parser for the given MIME type and extension, then parse.
    pub fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        for parser in &self.parsers {
            if parser.can_parse(mime_type, extension) {
                debug!(
                    parser = parser.name(),
                    mime = ?mime_type,
                    ext = ?extension,
                    "Parsing with"
                );
                return parser.parse(content, mime_type, extension);
            }
        }

        // Fallback: try to parse as text if it looks like valid UTF-8
        if std::str::from_utf8(content).is_ok() {
            debug!(mime = ?mime_type, ext = ?extension, "Falling back to text parser");
            return TextParser.parse(content, mime_type, extension);
        }

        Err(sift_core::SiftError::Parse {
            path: format!(
                "unknown (mime={}, ext={})",
                mime_type.unwrap_or("none"),
                extension.unwrap_or("none")
            ),
            message: "No parser found for this file type".to_string(),
        })
    }

    /// Check if any parser can handle this file.
    pub fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        self.parsers
            .iter()
            .any(|p| p.can_parse(mime_type, extension))
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::ContentType;

    #[test]
    fn test_registry_routes_text() {
        let registry = ParserRegistry::new();
        let doc = registry
            .parse(b"Hello world", Some("text/plain"), Some("txt"))
            .unwrap();
        assert_eq!(doc.content_type, ContentType::Text);
    }

    #[test]
    fn test_registry_routes_code() {
        let registry = ParserRegistry::new();
        let doc = registry
            .parse(b"fn main() {}", Some("text/x-rust"), Some("rs"))
            .unwrap();
        assert_eq!(doc.content_type, ContentType::Code);
    }

    #[test]
    fn test_registry_fallback_to_text() {
        let registry = ParserRegistry::new();
        let doc = registry.parse(b"Some plain text", None, None).unwrap();
        assert_eq!(doc.content_type, ContentType::Text);
    }

    #[test]
    fn test_registry_rejects_binary() {
        let registry = ParserRegistry::new();
        let result = registry.parse(&[0xFF, 0xFE, 0x00, 0x01], None, None);
        assert!(result.is_err());
    }
}
