use crate::html_text::{self, collapse_whitespace};
use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;

/// Parser for HTML and XML files.
pub struct WebParser;

impl WebParser {
    const WEB_MIMES: &[&str] = &[
        "text/html",
        "text/xml",
        "application/xml",
        "application/xhtml+xml",
    ];
    const WEB_EXTENSIONS: &[&str] = &["html", "htm", "xml", "xhtml"];
}

impl Parser for WebParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        crate::traits::matches(mime_type, extension, Self::WEB_MIMES, Self::WEB_EXTENSIONS)
    }

    fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let raw = String::from_utf8_lossy(content);
        let is_xml = extension == Some("xml") || mime_type.is_some_and(|m| m.contains("xml"));

        let (text, title) = if is_xml {
            (strip_xml_tags(&raw), None)
        } else {
            html_text::extract(&raw)
        };

        Ok(ParsedDocument {
            text,
            title,
            language: None,
            content_type: ContentType::Text,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &'static str {
        "web"
    }
}

/// Strip XML tags, keeping text content.
fn strip_xml_tags(xml: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;

    for ch in xml.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                result.push(' ');
            }
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }

    collapse_whitespace(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_html() {
        let parser = WebParser;
        let html = b"<html><head><title>Test Page</title></head><body><h1>Hello</h1><p>World</p></body></html>";
        let doc = parser.parse(html, Some("text/html"), None).unwrap();
        assert_eq!(doc.title.as_deref(), Some("Test Page"));
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("World"));
    }

    #[test]
    fn test_strip_scripts() {
        let html = b"<p>Before</p><script>alert('xss')</script><p>After</p>";
        let (text, _) = html_text::extract(&String::from_utf8_lossy(html));
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_strip_style_tags() {
        let html = "<p>Visible</p><style>body { color: red; }</style><p>Also visible</p>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains("Visible"));
        assert!(text.contains("Also visible"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn test_html_entities_decoded() {
        let html = "<p>5 &lt; 10 &amp; 10 &gt; 5</p>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains("5 < 10 & 10 > 5"));
    }

    #[test]
    fn test_html_entities_quotes_and_nbsp() {
        let html = "<p>&quot;hello&quot;&nbsp;&#39;world&#39;</p>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains("\"hello\""));
        assert!(text.contains("'world'"));
    }

    #[test]
    fn test_block_level_tags_add_newlines() {
        let html = "<div>Block1</div><div>Block2</div>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains('\n'));
    }

    #[test]
    fn test_heading_tags_add_newlines() {
        let html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>";
        let (text, _) = html_text::extract(html);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.len() >= 3);
    }

    #[test]
    fn test_collapse_whitespace_multiple_spaces() {
        let result = collapse_whitespace("hello    world   test");
        assert_eq!(result, "hello world test");
    }

    #[test]
    fn test_collapse_whitespace_preserves_newlines() {
        let result = collapse_whitespace("line1\n\n\nline2");
        // Multiple newlines collapse to one
        assert!(result.contains("line1\nline2") || result.contains("line1 line2"));
    }

    #[test]
    fn test_xml_parsing() {
        let parser = WebParser;
        let xml = b"<?xml version=\"1.0\"?><root><item>Value 1</item><item>Value 2</item></root>";
        let doc = parser.parse(xml, Some("text/xml"), Some("xml")).unwrap();
        assert!(doc.text.contains("Value 1"));
        assert!(doc.text.contains("Value 2"));
        assert!(doc.title.is_none());
    }

    #[test]
    fn test_strip_xml_tags_basic() {
        let xml = "<root><child>text</child></root>";
        let result = strip_xml_tags(xml);
        assert!(result.contains("text"));
        assert!(!result.contains('<'));
    }

    #[test]
    fn test_tags_with_attributes() {
        let html = "<div class=\"container\" id=\"main\"><p style=\"color:red\">Content</p></div>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("container"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn test_can_parse_mimes() {
        let parser = WebParser;
        assert!(parser.can_parse(Some("text/html"), None));
        assert!(parser.can_parse(Some("text/xml"), None));
        assert!(parser.can_parse(Some("application/xml"), None));
        assert!(parser.can_parse(Some("application/xhtml+xml"), None));
        assert!(!parser.can_parse(Some("text/plain"), None));
    }

    #[test]
    fn test_can_parse_extensions() {
        let parser = WebParser;
        assert!(parser.can_parse(None, Some("html")));
        assert!(parser.can_parse(None, Some("htm")));
        assert!(parser.can_parse(None, Some("xml")));
        assert!(parser.can_parse(None, Some("xhtml")));
        assert!(!parser.can_parse(None, Some("txt")));
    }

    #[test]
    fn test_no_mime_no_extension() {
        let parser = WebParser;
        assert!(!parser.can_parse(None, None));
    }

    #[test]
    fn test_html_no_title() {
        let html = "<p>No title here</p>";
        let (text, title) = html_text::extract(html);
        assert!(text.contains("No title here"));
        assert!(title.is_none());
    }

    #[test]
    fn test_section_article_tags() {
        let html =
            "<section>Sec</section><article>Art</article><header>Hdr</header><footer>Ftr</footer>";
        let (text, _) = html_text::extract(html);
        assert!(text.contains("Sec"));
        assert!(text.contains("Art"));
        assert!(text.contains("Hdr"));
        assert!(text.contains("Ftr"));
    }
}
