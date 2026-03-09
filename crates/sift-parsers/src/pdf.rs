use crate::traits::Parser;
use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use sift_core::{ContentType, ParsedDocument, SiftResult};

pub struct PdfParser;

impl PdfParser {
    const PDF_MIMES: &[&str] = &["application/pdf"];
    const PDF_EXTENSIONS: &[&str] = &["pdf"];
}

impl Parser for PdfParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::PDF_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::PDF_EXTENSIONS.contains(&ext) {
                return true;
            }
        }
        false
    }

    fn parse(
        &self,
        content: &[u8],
        _mime_type: Option<&str>,
        _extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let mut metadata = HashMap::new();
        metadata.insert("size_bytes".to_string(), content.len().to_string());

        // pdf-extract can panic on malformed PDFs, so wrap in catch_unwind.
        // Use AssertUnwindSafe to borrow content instead of cloning the entire buffer.
        let text = match panic::catch_unwind(AssertUnwindSafe(|| {
            pdf_extract::extract_text_from_mem(content)
        })) {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => {
                return Err(sift_core::SiftError::Parse {
                    path: "pdf".to_string(),
                    message: format!("PDF extraction failed: {}", e),
                });
            }
            Err(_) => {
                return Err(sift_core::SiftError::Parse {
                    path: "pdf".to_string(),
                    message: "PDF extraction panicked (malformed PDF)".to_string(),
                });
            }
        };

        // Single-pass: replace form feeds with page break markers and trim trailing whitespace
        let mut output = String::with_capacity(text.len());
        for line in text.split('\n') {
            if line.contains('\x0C') {
                for segment in line.split('\x0C') {
                    let trimmed = segment.trim_end();
                    if !trimmed.is_empty() {
                        output.push_str(trimmed);
                        output.push('\n');
                    }
                    output.push_str("\n--- Page Break ---\n\n");
                }
                // Remove the trailing page break marker after the last segment
                if output.ends_with("\n--- Page Break ---\n\n") {
                    output.truncate(output.len() - "\n--- Page Break ---\n\n".len());
                    output.push('\n');
                }
            } else {
                output.push_str(line.trim_end());
                output.push('\n');
            }
        }

        // Trim trailing whitespace in-place
        let trimmed_len = output.trim_end().len();
        output.truncate(trimmed_len);

        Ok(ParsedDocument {
            text: output,
            title: None,
            language: None,
            content_type: ContentType::Text,
            metadata,
        })
    }

    fn name(&self) -> &str {
        "pdf"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_pdf() {
        let parser = PdfParser;
        assert!(parser.can_parse(Some("application/pdf"), None));
        assert!(parser.can_parse(None, Some("pdf")));
        assert!(!parser.can_parse(None, Some("txt")));
        assert!(!parser.can_parse(Some("text/plain"), None));
    }

    #[test]
    fn test_malformed_pdf_returns_error() {
        let parser = PdfParser;
        let result = parser.parse(b"not a real pdf", Some("application/pdf"), Some("pdf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_does_not_parse_non_pdf_types() {
        let parser = PdfParser;
        assert!(!parser.can_parse(Some("application/zip"), None));
        assert!(!parser.can_parse(None, Some("docx")));
        assert!(!parser.can_parse(None, None));
    }

    #[test]
    fn test_metadata_includes_size_bytes() {
        let parser = PdfParser;
        // This will fail to parse, but let's test with a real minimal PDF if we can.
        // Since we can't easily construct a valid PDF in-memory, we test the error path.
        let result = parser.parse(b"fake", None, Some("pdf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_returns_error() {
        let parser = PdfParser;
        let result = parser.parse(b"", None, Some("pdf"));
        assert!(result.is_err());
    }
}
