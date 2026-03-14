use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;

/// Parser for plain text, markdown, RST, org-mode files.
pub struct TextParser;

impl TextParser {
    const TEXT_MIMES: &[&str] = &["text/plain", "text/markdown", "text/x-rst", "text/x-org"];

    const TEXT_EXTENSIONS: &[&str] = &[
        "txt", "md", "markdown", "rst", "org", "log", "cfg", "ini", "conf",
    ];
}

impl Parser for TextParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::TEXT_MIMES.iter().any(|m| mime.starts_with(m)) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::TEXT_EXTENSIONS.contains(&ext) {
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
        let text = String::from_utf8_lossy(content).into_owned();

        // Extract title from first heading in markdown
        let title = text.lines().find_map(|line| {
            line.trim()
                .strip_prefix("# ")
                .map(|heading| heading.trim().to_string())
        });

        Ok(ParsedDocument {
            text,
            title,
            language: None,
            content_type: ContentType::Text,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &'static str {
        "text"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_text() {
        let parser = TextParser;
        let doc = parser
            .parse(b"Hello world", Some("text/plain"), None)
            .unwrap();
        assert_eq!(doc.text, "Hello world");
        assert_eq!(doc.content_type, ContentType::Text);
    }

    #[test]
    fn test_parse_markdown_title() {
        let parser = TextParser;
        let doc = parser
            .parse(
                b"# My Document\n\nSome content",
                Some("text/markdown"),
                None,
            )
            .unwrap();
        assert_eq!(doc.title.as_deref(), Some("My Document"));
    }
}
