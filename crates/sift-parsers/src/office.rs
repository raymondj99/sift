use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::Cursor;

/// Parser for Office documents (DOCX, PPTX). Both are ZIP archives containing XML.
pub struct OfficeParser;

impl OfficeParser {
    const OFFICE_MIMES: &[&str] = &[
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ];
    const OFFICE_EXTENSIONS: &[&str] = &["docx", "pptx"];
}

impl Parser for OfficeParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::OFFICE_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::OFFICE_EXTENSIONS.contains(&ext) {
                return true;
            }
        }
        false
    }

    fn parse(
        &self,
        content: &[u8],
        _mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let ext = extension.unwrap_or("");
        let text = match ext {
            "pptx" => parse_pptx(content)?,
            _ => parse_docx(content)?,
        };

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Text,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &'static str {
        "office"
    }
}

fn parse_docx(content: &[u8]) -> SiftResult<String> {
    let cursor = Cursor::new(content);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| sift_core::SiftError::Parse {
        path: "docx".to_string(),
        message: format!("Failed to open DOCX zip: {e}"),
    })?;

    let xml = match archive.by_name("word/document.xml") {
        Ok(mut file) => {
            let mut buf = String::with_capacity(file.size() as usize);
            std::io::Read::read_to_string(&mut file, &mut buf).map_err(|e| {
                sift_core::SiftError::Parse {
                    path: "docx".to_string(),
                    message: format!("Failed to read document.xml: {e}"),
                }
            })?;
            buf
        }
        Err(e) => {
            return Err(sift_core::SiftError::Parse {
                path: "docx".to_string(),
                message: format!("Missing word/document.xml: {e}"),
            });
        }
    };

    Ok(extract_docx_text(&xml))
}

/// Extract text from DOCX XML. Collects <w:t> text nodes, uses <w:p> as paragraph breaks.
fn extract_docx_text(xml: &str) -> String {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut output = String::with_capacity(xml.len() / 3);
    let mut in_text = false;
    let mut paragraph_has_text = false;

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(ref e) | quick_xml::events::Event::Empty(ref e)) => {
                let qname = e.name();
                let name = qname.as_ref();
                if name == b"w:t" {
                    in_text = true;
                } else if name == b"w:p" {
                    if paragraph_has_text {
                        output.push('\n');
                    }
                    paragraph_has_text = false;
                } else if name == b"w:tab" {
                    output.push('\t');
                } else if name == b"w:br" {
                    output.push('\n');
                }
            }
            Ok(quick_xml::events::Event::Text(e)) => {
                if in_text {
                    let text = e.unescape().unwrap_or_default();
                    output.push_str(&text);
                    paragraph_has_text = true;
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                let qname = e.name();
                if qname.as_ref() == b"w:t" {
                    in_text = false;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    let trimmed = output.trim();
    if trimmed.len() == output.len() {
        output
    } else {
        trimmed.to_string()
    }
}

fn parse_pptx(content: &[u8]) -> SiftResult<String> {
    let cursor = Cursor::new(content);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| sift_core::SiftError::Parse {
        path: "pptx".to_string(),
        message: format!("Failed to open PPTX zip: {e}"),
    })?;

    // Collect slide filenames and sort numerically
    let mut slide_names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            let file = archive.by_index(i).ok()?;
            let name = file.name().to_string();
            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    slide_names.sort_by(|a, b| {
        let num_a = extract_slide_number(a);
        let num_b = extract_slide_number(b);
        num_a.cmp(&num_b)
    });

    let mut output = String::with_capacity(slide_names.len() * 500);
    for (i, slide_name) in slide_names.iter().enumerate() {
        if i > 0 {
            output.push_str("\n\n");
        }
        let _ = writeln!(output, "--- Slide {} ---", i + 1);

        if let Ok(mut file) = archive.by_name(slide_name) {
            let mut xml = String::new();
            if std::io::Read::read_to_string(&mut file, &mut xml).is_ok() {
                let text = extract_pptx_text(&xml);
                output.push_str(&text);
            }
        }
    }

    let trimmed_len = output.trim_end().len();
    output.truncate(trimmed_len);
    let start = output.len() - output.trim_start().len();
    if start > 0 {
        output = output[start..].to_string();
    }
    Ok(output)
}

fn extract_slide_number(name: &str) -> u32 {
    // Extract number from "ppt/slides/slide123.xml"
    name.trim_start_matches("ppt/slides/slide")
        .trim_end_matches(".xml")
        .parse()
        .unwrap_or(0)
}

/// Extract text from PPTX slide XML. Collects <a:t> text nodes.
fn extract_pptx_text(xml: &str) -> String {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut output = String::with_capacity(xml.len() / 4);
    let mut in_text = false;
    let mut last_was_paragraph = false;

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                let qname = e.name();
                let name = qname.as_ref();
                if name == b"a:t" {
                    in_text = true;
                } else if name == b"a:p" {
                    if !output.is_empty() && !last_was_paragraph {
                        output.push('\n');
                    }
                    last_was_paragraph = true;
                }
            }
            Ok(quick_xml::events::Event::Text(e)) => {
                if in_text {
                    let text = e.unescape().unwrap_or_default();
                    output.push_str(&text);
                    last_was_paragraph = false;
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                let qname = e.name();
                if qname.as_ref() == b"a:t" {
                    in_text = false;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    let trimmed = output.trim();
    if trimmed.len() == output.len() {
        output
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_office() {
        let parser = OfficeParser;
        assert!(parser.can_parse(
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            None
        ));
        assert!(parser.can_parse(None, Some("docx")));
        assert!(parser.can_parse(None, Some("pptx")));
        assert!(!parser.can_parse(None, Some("txt")));
        assert!(!parser.can_parse(None, Some("xlsx")));
    }

    #[test]
    fn test_extract_docx_text() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p><w:r><w:t>Hello World</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
            </w:body>
        </w:document>"#;
        let text = extract_docx_text(xml);
        assert!(text.contains("Hello World"));
        assert!(text.contains("Second paragraph"));
    }

    #[test]
    fn test_extract_pptx_text() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
               xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
            <p:cSld>
                <p:spTree>
                    <p:sp>
                        <p:txBody>
                            <a:p><a:r><a:t>Slide Title</a:t></a:r></a:p>
                            <a:p><a:r><a:t>Bullet point one</a:t></a:r></a:p>
                        </p:txBody>
                    </p:sp>
                </p:spTree>
            </p:cSld>
        </p:sld>"#;
        let text = extract_pptx_text(xml);
        assert!(text.contains("Slide Title"));
        assert!(text.contains("Bullet point one"));
    }

    #[test]
    fn test_invalid_zip_returns_error() {
        let parser = OfficeParser;
        let result = parser.parse(b"not a zip", None, Some("docx"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parser_name() {
        let parser = OfficeParser;
        assert_eq!(parser.name(), "office");
    }

    #[test]
    fn test_can_parse_pptx_mime() {
        let parser = OfficeParser;
        assert!(parser.can_parse(
            Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            None
        ));
    }

    #[test]
    fn test_can_parse_all_extensions() {
        let parser = OfficeParser;
        assert!(parser.can_parse(None, Some("docx")));
        assert!(parser.can_parse(None, Some("pptx")));
    }

    #[test]
    fn test_cannot_parse_non_office() {
        let parser = OfficeParser;
        assert!(!parser.can_parse(None, Some("txt")));
        assert!(!parser.can_parse(None, Some("pdf")));
        assert!(!parser.can_parse(Some("text/plain"), None));
        assert!(!parser.can_parse(None, None));
    }

    #[test]
    fn test_invalid_zip_pptx_returns_error() {
        let parser = OfficeParser;
        let result = parser.parse(b"not a zip", None, Some("pptx"));
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_docx_with_tabs_and_breaks() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p><w:r><w:t>Before tab</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>After tab</w:t></w:r></w:p>
                <w:p><w:r><w:t>Before break</w:t></w:r><w:r><w:br/></w:r><w:r><w:t>After break</w:t></w:r></w:p>
            </w:body>
        </w:document>"#;
        let text = extract_docx_text(xml);
        assert!(text.contains("Before tab"));
        assert!(text.contains("After tab"));
        assert!(text.contains("Before break"));
        assert!(text.contains("After break"));
    }

    #[test]
    fn test_extract_docx_empty_document() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
            </w:body>
        </w:document>"#;
        let text = extract_docx_text(xml);
        assert!(text.is_empty());
    }

    #[test]
    fn test_extract_pptx_empty_slide() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
               xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
        </p:sld>"#;
        let text = extract_pptx_text(xml);
        assert!(text.is_empty());
    }

    #[test]
    fn test_extract_slide_number() {
        assert_eq!(extract_slide_number("ppt/slides/slide1.xml"), 1);
        assert_eq!(extract_slide_number("ppt/slides/slide10.xml"), 10);
        assert_eq!(extract_slide_number("ppt/slides/slide123.xml"), 123);
        assert_eq!(extract_slide_number("ppt/slides/slideabc.xml"), 0);
    }

    #[test]
    fn test_parse_default_extension_is_docx() {
        // When no extension is provided, the parser defaults to DOCX parsing
        let parser = OfficeParser;
        let result = parser.parse(b"not a zip", None, None);
        assert!(result.is_err());
    }

    /// Helper: build a minimal valid DOCX (ZIP) with the given body XML.
    fn make_docx(body_xml: &str) -> Vec<u8> {
        use std::io::Write;
        let buf = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        let options = zip::write::SimpleFileOptions::default();

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(br#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#).unwrap();

        // word/document.xml
        zip.start_file("word/document.xml", options).unwrap();
        let doc = format!(
            r#"<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{}</w:body></w:document>"#,
            body_xml
        );
        zip.write_all(doc.as_bytes()).unwrap();

        zip.finish().unwrap().into_inner()
    }

    /// Helper: build a minimal valid PPTX (ZIP) with given slide XMLs.
    fn make_pptx(slides: &[&str]) -> Vec<u8> {
        use std::io::Write;
        let buf = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        let options = zip::write::SimpleFileOptions::default();

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(br#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>"#).unwrap();

        for (i, slide_xml) in slides.iter().enumerate() {
            let name = format!("ppt/slides/slide{}.xml", i + 1);
            zip.start_file(&name, options).unwrap();
            let full = format!(
                r#"<?xml version="1.0"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody>{}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#,
                slide_xml
            );
            zip.write_all(full.as_bytes()).unwrap();
        }

        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn test_parse_docx_multiple_paragraphs() {
        let parser = OfficeParser;
        let docx = make_docx(
            "<w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p>",
        );
        let doc = parser.parse(&docx, None, Some("docx")).unwrap();
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("World"));
        assert_eq!(doc.content_type, ContentType::Text);
        assert!(doc.title.is_none());
    }

    #[test]
    fn test_parse_docx_with_table() {
        let parser = OfficeParser;
        let docx = make_docx(
            "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell 1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cell 2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>",
        );
        let doc = parser.parse(&docx, None, Some("docx")).unwrap();
        assert!(doc.text.contains("Cell 1"));
        assert!(doc.text.contains("Cell 2"));
    }

    #[test]
    fn test_parse_docx_empty_body() {
        let parser = OfficeParser;
        let docx = make_docx("<w:p/>");
        let doc = parser.parse(&docx, None, Some("docx")).unwrap();
        assert!(doc.text.trim().is_empty());
    }

    #[test]
    fn test_parse_docx_with_tabs_and_breaks_via_zip() {
        let parser = OfficeParser;
        let docx = make_docx(
            "<w:p><w:r><w:t>A</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>B</w:t></w:r></w:p><w:p><w:r><w:t>C</w:t></w:r><w:r><w:br/></w:r><w:r><w:t>D</w:t></w:r></w:p>",
        );
        let doc = parser.parse(&docx, None, Some("docx")).unwrap();
        assert!(doc.text.contains("A\tB"));
        assert!(doc.text.contains("C\nD"));
    }

    #[test]
    fn test_parse_docx_missing_document_xml() {
        use std::io::Write;
        // Build a ZIP that does NOT contain word/document.xml
        let buf = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("other.xml", options).unwrap();
        zip.write_all(b"<data/>").unwrap();
        let data = zip.finish().unwrap().into_inner();

        let parser = OfficeParser;
        let result = parser.parse(&data, None, Some("docx"));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Missing word/document.xml"));
    }

    #[test]
    fn test_parse_pptx_single_slide() {
        let parser = OfficeParser;
        let pptx = make_pptx(&["<a:p><a:r><a:t>Title Slide</a:t></a:r></a:p>"]);
        let doc = parser.parse(&pptx, None, Some("pptx")).unwrap();
        assert!(doc.text.contains("--- Slide 1 ---"));
        assert!(doc.text.contains("Title Slide"));
        assert_eq!(doc.content_type, ContentType::Text);
    }

    #[test]
    fn test_parse_pptx_multiple_slides() {
        let parser = OfficeParser;
        let pptx = make_pptx(&[
            "<a:p><a:r><a:t>Slide One</a:t></a:r></a:p>",
            "<a:p><a:r><a:t>Slide Two</a:t></a:r></a:p>",
        ]);
        let doc = parser.parse(&pptx, None, Some("pptx")).unwrap();
        assert!(doc.text.contains("--- Slide 1 ---"));
        assert!(doc.text.contains("Slide One"));
        assert!(doc.text.contains("--- Slide 2 ---"));
        assert!(doc.text.contains("Slide Two"));
    }

    #[test]
    fn test_parse_pptx_empty_slides() {
        let parser = OfficeParser;
        // Build a PPTX with no slide files at all
        let pptx = make_pptx(&[]);
        let doc = parser.parse(&pptx, None, Some("pptx")).unwrap();
        assert!(doc.text.trim().is_empty());
    }

    #[test]
    fn test_parse_pptx_multiple_paragraphs_per_slide() {
        let parser = OfficeParser;
        let pptx = make_pptx(&[
            "<a:p><a:r><a:t>Title</a:t></a:r></a:p><a:p><a:r><a:t>Subtitle</a:t></a:r></a:p>",
        ]);
        let doc = parser.parse(&pptx, None, Some("pptx")).unwrap();
        assert!(doc.text.contains("Title"));
        assert!(doc.text.contains("Subtitle"));
    }
}
