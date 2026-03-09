use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};

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
        if let Some(mime) = mime_type {
            if Self::WEB_MIMES.iter().any(|m| mime.starts_with(m)) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::WEB_EXTENSIONS.contains(&ext) {
                return true;
            }
        }
        false
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
            extract_html_text(&raw)
        };

        Ok(ParsedDocument {
            text,
            title,
            language: None,
            content_type: ContentType::Text,
            metadata: Default::default(),
        })
    }

    fn name(&self) -> &str {
        "web"
    }
}

/// Simple HTML text extraction - strips tags, extracts title.
fn extract_html_text(html: &str) -> (String, Option<String>) {
    let mut text = String::new();
    let mut title = None;
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::new();
    let mut in_title_tag = false;
    let mut title_text = String::new();

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            i += 1;
            continue;
        }

        if ch == '>' && in_tag {
            in_tag = false;
            let tag_lower = tag_name.to_lowercase();

            if tag_lower.starts_with("script") {
                in_script = true;
            } else if tag_lower.starts_with("/script") {
                in_script = false;
            } else if tag_lower.starts_with("style") {
                in_style = true;
            } else if tag_lower.starts_with("/style") {
                in_style = false;
            } else if tag_lower.starts_with("title") {
                in_title_tag = true;
            } else if tag_lower.starts_with("/title") {
                in_title_tag = false;
                title = Some(title_text.trim().to_string());
            }

            // Block-level tags get newlines
            if matches!(
                tag_lower.trim_start_matches('/'),
                "p" | "div"
                    | "br"
                    | "h1"
                    | "h2"
                    | "h3"
                    | "h4"
                    | "h5"
                    | "h6"
                    | "li"
                    | "tr"
                    | "blockquote"
                    | "pre"
                    | "hr"
                    | "section"
                    | "article"
                    | "header"
                    | "footer"
            ) {
                text.push('\n');
            }

            i += 1;
            continue;
        }

        if in_tag {
            tag_name.push(ch);
        } else if in_title_tag {
            title_text.push(ch);
        } else if !in_script && !in_style {
            text.push(ch);
        }

        i += 1;
    }

    // Decode common HTML entities
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");

    // Normalize whitespace
    let text = collapse_whitespace(&text);

    (text, title)
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

fn collapse_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_was_whitespace = false;

    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_was_whitespace {
                result.push(if ch == '\n' { '\n' } else { ' ' });
            }
            last_was_whitespace = true;
        } else {
            result.push(ch);
            last_was_whitespace = false;
        }
    }

    result.trim().to_string()
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
        let (text, _) = extract_html_text(&String::from_utf8_lossy(html));
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_parse_xml() {
        let parser = WebParser;
        let xml = b"<root><item>Hello</item><item>World</item></root>";
        let doc = parser.parse(xml, None, Some("xml")).unwrap();
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("World"));
    }
}
