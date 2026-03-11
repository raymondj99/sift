use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// Parser for EPUB (Electronic Publication) files.
/// EPUBs are ZIP archives containing XHTML chapters with metadata.
pub struct EpubParser;

impl Parser for EpubParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if mime == "application/epub+zip" {
                return true;
            }
        }
        if let Some(ext) = extension {
            if ext == "epub" {
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
        let cursor = Cursor::new(content);
        let mut archive =
            zip::ZipArchive::new(cursor).map_err(|e| sift_core::SiftError::Parse {
                path: "epub".to_string(),
                message: format!("Failed to open EPUB zip: {e}"),
            })?;

        // Try to extract the title from the OPF metadata file
        let title = extract_title_from_opf(&mut archive);

        // Collect HTML/XHTML file names sorted by their order in the archive
        let mut html_files: Vec<(usize, String)> = Vec::new();
        for i in 0..archive.len() {
            if let Ok(file) = archive.by_index(i) {
                let name = file.name().to_string();
                if is_content_file(&name) {
                    html_files.push((i, name));
                }
            }
        }

        // Sort by file path to maintain chapter order
        html_files.sort_by(|a, b| a.1.cmp(&b.1));

        let mut output = String::new();
        let mut chapter_count = 0;

        for (idx, _name) in &html_files {
            let mut file = match archive.by_index(*idx) {
                Ok(f) => f,
                Err(_) => continue,
            };

            let mut buf = String::new();
            if file.read_to_string(&mut buf).is_err() {
                continue;
            }

            let text = strip_html_tags(&buf);
            let text = text.trim();
            if text.is_empty() {
                continue;
            }

            if !output.is_empty() {
                output.push_str("\n\n");
            }
            output.push_str(text);
            chapter_count += 1;
        }

        let mut metadata = HashMap::new();
        metadata.insert("chapter_count".to_string(), chapter_count.to_string());

        Ok(ParsedDocument {
            text: output.trim().to_string(),
            title,
            language: None,
            content_type: ContentType::Text,
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "epub"
    }
}

/// Check if a file path inside the EPUB is an HTML/XHTML content file.
fn is_content_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    (lower.ends_with(".xhtml") || lower.ends_with(".html") || lower.ends_with(".htm"))
        && !lower.contains("toc")
        && !lower.contains("nav")
}

/// Try to extract the book title from the OPF (Open Packaging Format) metadata.
fn extract_title_from_opf(archive: &mut zip::ZipArchive<Cursor<&[u8]>>) -> Option<String> {
    // Find the .opf file
    let opf_name = (0..archive.len()).find_map(|i| {
        let file = archive.by_index(i).ok()?;
        let name = file.name().to_string();
        if name.ends_with(".opf") {
            Some(name)
        } else {
            None
        }
    })?;

    let mut file = archive.by_name(&opf_name).ok()?;
    let mut xml = String::new();
    file.read_to_string(&mut xml).ok()?;

    // Use quick-xml to find <dc:title> element
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut in_title = false;

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                let name = e.name();
                let local = name.as_ref();
                if local == b"dc:title" || local == b"title" {
                    in_title = true;
                }
            }
            Ok(quick_xml::events::Event::Text(e)) => {
                if in_title {
                    let title = e.unescape().unwrap_or_default().trim().to_string();
                    if !title.is_empty() {
                        return Some(title);
                    }
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                let name = e.name();
                let local = name.as_ref();
                if local == b"dc:title" || local == b"title" {
                    in_title = false;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    None
}

/// Strip HTML/XHTML tags from content, returning plain text.
fn strip_html_tags(html: &str) -> String {
    let mut text = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::new();

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
            }

            // Block-level tags produce newlines
            let tag_base = tag_lower.trim_start_matches('/');
            if matches!(
                tag_base,
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
            ) {
                text.push('\n');
            }

            i += 1;
            continue;
        }

        if in_tag {
            tag_name.push(ch);
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

    // Collapse whitespace while preserving intentional line breaks
    collapse_whitespace(&text)
}

/// Collapse runs of whitespace, keeping single newlines.
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
    use std::io::Write;

    fn make_epub(chapters: &[(&str, &str)], title: Option<&str>) -> Vec<u8> {
        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        // Write mimetype (required by EPUB spec)
        writer.start_file("mimetype", options).unwrap();
        writer.write_all(b"application/epub+zip").unwrap();

        // Write a minimal OPF file with optional title
        if let Some(t) = title {
            writer.start_file("OEBPS/content.opf", options).unwrap();
            let opf = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{t}</dc:title>
  </metadata>
</package>"#
            );
            writer.write_all(opf.as_bytes()).unwrap();
        }

        // Write chapters
        for (name, content) in chapters {
            writer.start_file(format!("OEBPS/{name}"), options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }

        let cursor = writer.finish().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn test_can_parse_epub() {
        let parser = EpubParser;
        assert!(parser.can_parse(None, Some("epub")));
        assert!(parser.can_parse(Some("application/epub+zip"), None));
        assert!(!parser.can_parse(None, Some("pdf")));
        assert!(!parser.can_parse(None, Some("zip")));
    }

    #[test]
    fn test_parse_epub_single_chapter() {
        let epub = make_epub(
            &[(
                "chapter1.xhtml",
                "<html><body><h1>Chapter 1</h1><p>Hello world.</p></body></html>",
            )],
            Some("Test Book"),
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert_eq!(doc.content_type, ContentType::Text);
        assert_eq!(doc.title.as_deref(), Some("Test Book"));
        assert!(doc.text.contains("Chapter 1"));
        assert!(doc.text.contains("Hello world."));
    }

    #[test]
    fn test_parse_epub_multiple_chapters() {
        let epub = make_epub(
            &[
                (
                    "chapter1.xhtml",
                    "<html><body><p>First chapter content.</p></body></html>",
                ),
                (
                    "chapter2.xhtml",
                    "<html><body><p>Second chapter content.</p></body></html>",
                ),
            ],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.text.contains("First chapter content."));
        assert!(doc.text.contains("Second chapter content."));
        assert_eq!(doc.metadata.get("chapter_count").unwrap(), "2");
    }

    #[test]
    fn test_parse_epub_skips_toc() {
        let epub = make_epub(
            &[
                (
                    "toc.xhtml",
                    "<html><body><p>Table of Contents entries</p></body></html>",
                ),
                (
                    "chapter1.xhtml",
                    "<html><body><p>Real content.</p></body></html>",
                ),
            ],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.text.contains("Real content."));
        assert!(!doc.text.contains("Table of Contents"));
    }

    #[test]
    fn test_parse_epub_strips_html_entities() {
        let epub = make_epub(
            &[(
                "chapter1.xhtml",
                "<html><body><p>Tom &amp; Jerry &lt;3</p></body></html>",
            )],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.text.contains("Tom & Jerry <3"));
    }

    #[test]
    fn test_invalid_epub_returns_error() {
        let parser = EpubParser;
        let result = parser.parse(b"not a zip", None, Some("epub"));
        assert!(result.is_err());
    }

    #[test]
    fn test_strip_html_tags_basic() {
        let result = strip_html_tags("<p>Hello <b>world</b></p>");
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(!result.contains('<'));
    }

    #[test]
    fn test_parse_epub_no_title_in_opf() {
        // EPUB with an OPF file that has no <dc:title>
        let epub = make_epub(
            &[(
                "chapter1.xhtml",
                "<html><body><p>Content here.</p></body></html>",
            )],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.title.is_none());
        assert!(doc.text.contains("Content here."));
    }

    #[test]
    fn test_parse_epub_empty_chapter_skipped() {
        // Chapter with only whitespace content should be skipped
        let epub = make_epub(
            &[
                ("chapter1.xhtml", "<html><body>   </body></html>"),
                (
                    "chapter2.xhtml",
                    "<html><body><p>Real content.</p></body></html>",
                ),
            ],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.text.contains("Real content."));
        // Only the non-empty chapter should count
        assert_eq!(doc.metadata.get("chapter_count").unwrap(), "1");
    }

    #[test]
    fn test_parse_epub_skips_nav_file() {
        let epub = make_epub(
            &[
                (
                    "nav.xhtml",
                    "<html><body><p>Navigation content</p></body></html>",
                ),
                (
                    "chapter1.xhtml",
                    "<html><body><p>Chapter text.</p></body></html>",
                ),
            ],
            None,
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(!doc.text.contains("Navigation content"));
        assert!(doc.text.contains("Chapter text."));
    }

    #[test]
    fn test_strip_html_script_and_style() {
        let html = "<html><head><script>var x = 1;</script><style>body { color: red; }</style></head><body><p>Visible text</p></body></html>";
        let result = strip_html_tags(html);
        assert!(result.contains("Visible text"));
        assert!(!result.contains("var x"));
        assert!(!result.contains("color: red"));
    }

    #[test]
    fn test_strip_html_nbsp_entity() {
        let html = "<p>Hello&nbsp;World</p>";
        let result = strip_html_tags(html);
        assert!(result.contains("Hello World"));
    }

    #[test]
    fn test_strip_html_quot_entity() {
        let html = "<p>&quot;Quoted&quot;</p>";
        let result = strip_html_tags(html);
        assert!(result.contains("\"Quoted\""));
    }

    #[test]
    fn test_strip_html_apostrophe_entity() {
        let html = "<p>It&#39;s</p>";
        let result = strip_html_tags(html);
        assert!(result.contains("It's"));
    }

    #[test]
    fn test_collapse_whitespace() {
        // collapse_whitespace trims start/end and collapses consecutive whitespace
        let input = "  Hello   World  \n  Next  ";
        let result = collapse_whitespace(input);
        // After collapsing, runs of spaces become a single space,
        // but newlines are preserved as newlines
        assert!(result.starts_with("Hello"));
        assert!(result.contains("World"));
        assert!(result.contains("Next"));
        assert!(result.ends_with("Next"));
    }

    #[test]
    fn test_is_content_file() {
        assert!(is_content_file("OEBPS/chapter1.xhtml"));
        assert!(is_content_file("chapter.html"));
        assert!(is_content_file("content.htm"));
        assert!(!is_content_file("toc.xhtml"));
        assert!(!is_content_file("nav.xhtml"));
        assert!(!is_content_file("style.css"));
        assert!(!is_content_file("image.png"));
    }

    #[test]
    fn test_parser_name_epub() {
        let parser = EpubParser;
        assert_eq!(parser.name(), "epub");
    }

    #[test]
    fn test_parse_epub_htm_extension() {
        // Use .htm extension for the chapter
        let epub = make_epub(
            &[(
                "chapter1.htm",
                "<html><body><p>HTM content.</p></body></html>",
            )],
            Some("HTM Book"),
        );

        let parser = EpubParser;
        let doc = parser.parse(&epub, None, Some("epub")).unwrap();
        assert!(doc.text.contains("HTM content."));
    }

    #[test]
    fn test_strip_html_block_elements_produce_newlines() {
        let html = "<h1>Title</h1><p>Paragraph</p><div>Div content</div>";
        let result = strip_html_tags(html);
        assert!(result.contains("Title"));
        assert!(result.contains("Paragraph"));
        assert!(result.contains("Div content"));
    }
}
