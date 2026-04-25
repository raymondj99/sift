use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;

/// Parser for image files. Extracts metadata (dimensions, format, filename context)
/// into searchable text. For actual visual content search, an embedding model
/// like CLIP/Nomic Vision would generate vectors directly from pixels —
/// this parser provides the text-based fallback so images are still discoverable
/// by filename, path, and format metadata.
pub struct ImageParser;

impl ImageParser {
    const IMAGE_MIMES: &[&str] = &[
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "image/bmp",
        "image/tiff",
        "image/x-icon",
    ];

    const IMAGE_EXTENSIONS: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "tiff", "tif", "ico",
    ];
}

impl Parser for ImageParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        crate::traits::matches(
            mime_type,
            extension,
            Self::IMAGE_MIMES,
            Self::IMAGE_EXTENSIONS,
        )
    }

    fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let mut metadata = HashMap::new();
        let mut text_parts: Vec<String> = Vec::new();

        // Trust the caller-provided MIME (from the scan step) and only fall back
        // to byte sniffing when none was supplied. Stops redoing detection that
        // FilesystemSource already performed.
        let format = mime_type
            .map(str::to_owned)
            .or_else(|| infer::get(content).map(|k| k.mime_type().to_string()))
            .unwrap_or_else(|| "image/unknown".to_string());
        metadata.insert("format".to_string(), format.clone());

        let ext = extension.unwrap_or("image");
        metadata.insert("extension".to_string(), ext.to_string());
        metadata.insert("size_bytes".to_string(), content.len().to_string());

        text_parts.push(format!("Image file ({})", ext.to_uppercase()));

        // Try to read dimensions from common formats
        if let Some((w, h)) = read_dimensions(content, &format) {
            metadata.insert("width".to_string(), w.to_string());
            metadata.insert("height".to_string(), h.to_string());
            text_parts.push(format!("{w}x{h} pixels"));

            // Classify by aspect ratio / size
            if w == h {
                text_parts.push("square image".to_string());
            } else if w > h * 2 {
                text_parts.push("panoramic wide image".to_string());
            } else if h > w * 2 {
                text_parts.push("tall vertical image".to_string());
            } else if w > h {
                text_parts.push("landscape orientation".to_string());
            } else {
                text_parts.push("portrait orientation".to_string());
            }

            // Size classification
            let megapixels = (f64::from(w) * f64::from(h)) / 1_000_000.0;
            if megapixels > 8.0 {
                text_parts.push("high resolution".to_string());
            } else if megapixels < 0.1 {
                text_parts.push("thumbnail icon small".to_string());
            }
        }

        // SVG: extract text content
        if ext == "svg" || format == "image/svg+xml" {
            if let Ok(svg_text) = std::str::from_utf8(content) {
                let extracted = extract_svg_text(svg_text);
                if !extracted.is_empty() {
                    text_parts.push(extracted);
                }
            }
        }

        // Format-specific notes
        match ext {
            "png" => text_parts.push("PNG lossless image".to_string()),
            "jpg" | "jpeg" => text_parts.push("JPEG photograph image".to_string()),
            "gif" => text_parts.push("GIF image animation".to_string()),
            "webp" => text_parts.push("WebP image".to_string()),
            "svg" => text_parts.push("SVG vector graphic scalable".to_string()),
            "bmp" => text_parts.push("BMP bitmap image".to_string()),
            "ico" => text_parts.push("icon favicon".to_string()),
            _ => {}
        }

        let text = text_parts.join(". ");

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Image,
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "image"
    }
}

/// Read image dimensions from raw bytes for common formats.
///
/// `mime` must be the canonical IANA media type (e.g. `image/png`); a substring
/// match would let `application/x-pngwriter` masquerade as PNG.
fn read_dimensions(data: &[u8], mime: &str) -> Option<(u32, u32)> {
    match mime {
        "image/png" => read_png_dimensions(data),
        "image/jpeg" => read_jpeg_dimensions(data),
        "image/gif" => read_gif_dimensions(data),
        "image/webp" => read_webp_dimensions(data),
        "image/bmp" => read_bmp_dimensions(data),
        _ => None,
    }
}

/// PNG: width/height at bytes 16-23 in the IHDR chunk.
fn read_png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 || &data[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    Some((w, h))
}

/// JPEG: scan for SOF0/SOF2 marker to find dimensions.
fn read_jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        // SOF0 (0xC0) or SOF2 (0xC2) — baseline or progressive
        if (marker == 0xC0 || marker == 0xC2) && i + 9 < data.len() {
            let h = u32::from(u16::from_be_bytes([data[i + 5], data[i + 6]]));
            let w = u32::from(u16::from_be_bytes([data[i + 7], data[i + 8]]));
            return Some((w, h));
        }
        // Skip to next marker
        if i + 3 < data.len() {
            let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            i += 2 + len;
        } else {
            break;
        }
    }
    None
}

/// GIF: width/height at bytes 6-9 (little-endian).
fn read_gif_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 10 || (&data[0..3] != b"GIF") {
        return None;
    }
    let w = u32::from(u16::from_le_bytes([data[6], data[7]]));
    let h = u32::from(u16::from_le_bytes([data[8], data[9]]));
    Some((w, h))
}

/// WebP: RIFF container, VP8 chunk has dimensions.
fn read_webp_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 30 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return None;
    }
    // VP8 lossy
    if &data[12..16] == b"VP8 " && data.len() > 29 {
        // VP8 bitstream starts at offset 20, frame header at 23
        if data[23] == 0x9D && data[24] == 0x01 && data[25] == 0x2A {
            let w = u32::from(u16::from_le_bytes([data[26], data[27]])) & 0x3FFF;
            let h = u32::from(u16::from_le_bytes([data[28], data[29]])) & 0x3FFF;
            return Some((w, h));
        }
    }
    // VP8L lossless
    if &data[12..16] == b"VP8L" && data.len() > 24 {
        let b0 = u32::from(data[21]);
        let b1 = u32::from(data[22]);
        let b2 = u32::from(data[23]);
        let b3 = u32::from(data[24]);
        let w = (b0 | (b1 << 8)) & 0x3FFF;
        let h = ((b1 >> 6) | (b2 << 2) | (b3 << 10)) & 0x3FFF;
        return Some((w + 1, h + 1));
    }
    None
}

/// BMP: width/height at bytes 18-25.
fn read_bmp_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 26 || &data[0..2] != b"BM" {
        return None;
    }
    let w = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let h = u32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    Some((w, h))
}

/// Extract text content from SVG (title, desc, text elements).
fn extract_svg_text(svg: &str) -> String {
    let mut parts = Vec::new();

    // Simple regex-free extraction of <title>, <desc>, <text> content
    for tag in &["title", "desc", "text"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut search_from = 0;
        while let Some(start) = svg[search_from..].find(&open) {
            let abs_start = search_from + start;
            // Find end of opening tag
            if let Some(gt) = svg[abs_start..].find('>') {
                let content_start = abs_start + gt + 1;
                if let Some(end) = svg[content_start..].find(&close) {
                    let text = &svg[content_start..content_start + end];
                    let clean: String = text
                        .chars()
                        .filter(|c| !c.is_control() || *c == ' ')
                        .collect();
                    let clean = clean.trim();
                    if !clean.is_empty() {
                        parts.push(clean.to_string());
                    }
                    search_from = content_start + end + close.len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    parts.join(". ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_png_dimensions() {
        // Minimal valid 1x1 PNG header
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        // IHDR chunk length (13 bytes)
        png.extend_from_slice(&[0, 0, 0, 13]);
        // IHDR chunk type
        png.extend_from_slice(b"IHDR");
        // Width: 128, Height: 64
        png.extend_from_slice(&128u32.to_be_bytes());
        png.extend_from_slice(&64u32.to_be_bytes());

        let (w, h) = read_png_dimensions(&png).unwrap();
        assert_eq!(w, 128);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_parse_image_metadata() {
        let parser = ImageParser;
        // Minimal PNG-like content
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&4000u32.to_be_bytes());
        png.extend_from_slice(&3000u32.to_be_bytes());
        png.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth, color type, etc.

        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert_eq!(doc.content_type, ContentType::Image);
        assert!(doc.text.contains("4000x3000"));
        assert!(doc.text.contains("landscape"));
        assert!(doc.text.contains("high resolution"));
        assert!(doc.text.contains("PNG"));
    }

    #[test]
    fn test_parse_svg_extracts_text() {
        let parser = ImageParser;
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg"><title>Architecture Diagram</title><desc>System design overview</desc><text x="10" y="20">Database</text></svg>"#;

        let doc = parser
            .parse(svg, Some("image/svg+xml"), Some("svg"))
            .unwrap();
        assert!(doc.text.contains("Architecture Diagram"));
        assert!(doc.text.contains("System design overview"));
        assert!(doc.text.contains("Database"));
    }

    #[test]
    fn test_can_parse_image_formats() {
        let parser = ImageParser;
        assert!(parser.can_parse(Some("image/png"), None));
        assert!(parser.can_parse(Some("image/jpeg"), None));
        assert!(parser.can_parse(None, Some("jpg")));
        assert!(parser.can_parse(None, Some("gif")));
        assert!(parser.can_parse(None, Some("webp")));
        assert!(parser.can_parse(None, Some("bmp")));
        assert!(parser.can_parse(None, Some("ico")));
        assert!(!parser.can_parse(None, Some("pdf")));
    }

    #[test]
    fn test_image_parser_name() {
        assert_eq!(ImageParser.name(), "image");
    }

    #[test]
    fn test_parse_jpeg_dimensions() {
        // Build a minimal JPEG with SOF0 marker: FFD8 ... FFC0 (len)(precision)(h=64)(w=128)
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
                                         // APP0 marker (filler) — length 0x0004 means 4 bytes including the length field
        jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00]);
        // SOF0 marker with enough data: marker(2) + len(2) + precision(1) + h(2) + w(2) + padding
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x0B, 0x08]);
        jpeg.extend_from_slice(&64u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&128u16.to_be_bytes()); // width
        jpeg.push(0x00); // padding so data.len() > i+9
        let (w, h) = read_jpeg_dimensions(&jpeg).unwrap();
        assert_eq!((w, h), (128, 64));
    }

    #[test]
    fn test_parse_gif_dimensions() {
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&320u16.to_le_bytes()); // width
        gif.extend_from_slice(&240u16.to_le_bytes()); // height
        let (w, h) = read_gif_dimensions(&gif).unwrap();
        assert_eq!((w, h), (320, 240));
    }

    #[test]
    fn test_parse_bmp_dimensions() {
        let mut bmp = vec![0u8; 26];
        bmp[0] = b'B';
        bmp[1] = b'M';
        bmp[18..22].copy_from_slice(&640u32.to_le_bytes());
        bmp[22..26].copy_from_slice(&480u32.to_le_bytes());
        let (w, h) = read_bmp_dimensions(&bmp).unwrap();
        assert_eq!((w, h), (640, 480));
    }

    #[test]
    fn test_image_aspect_ratio_square() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes());
        png.extend_from_slice(&100u32.to_be_bytes());
        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("square image"));
    }

    #[test]
    fn test_image_aspect_ratio_panoramic() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&1000u32.to_be_bytes());
        png.extend_from_slice(&100u32.to_be_bytes());
        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("panoramic wide"));
    }

    #[test]
    fn test_image_aspect_ratio_tall() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes());
        png.extend_from_slice(&500u32.to_be_bytes());
        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("tall vertical"));
    }

    #[test]
    fn test_image_portrait_orientation() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes());
        png.extend_from_slice(&150u32.to_be_bytes());
        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("portrait"));
    }

    #[test]
    fn test_image_thumbnail_classification() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&16u32.to_be_bytes());
        png.extend_from_slice(&16u32.to_be_bytes());
        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("thumbnail"));
    }

    #[test]
    fn test_image_format_notes() {
        let parser = ImageParser;
        // GIF with valid dimensions
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&100u16.to_le_bytes());
        gif.extend_from_slice(&100u16.to_le_bytes());
        let doc = parser.parse(&gif, Some("image/gif"), Some("gif")).unwrap();
        assert!(doc.text.contains("GIF image animation"));

        // BMP
        let mut bmp = vec![0u8; 26];
        bmp[0] = b'B';
        bmp[1] = b'M';
        bmp[18..22].copy_from_slice(&100u32.to_le_bytes());
        bmp[22..26].copy_from_slice(&100u32.to_le_bytes());
        let doc = parser.parse(&bmp, Some("image/bmp"), Some("bmp")).unwrap();
        assert!(doc.text.contains("BMP bitmap"));
    }
}
