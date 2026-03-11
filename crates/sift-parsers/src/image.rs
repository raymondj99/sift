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
        if let Some(mime) = mime_type {
            if Self::IMAGE_MIMES.iter().any(|m| mime.starts_with(m)) {
                return true;
            }
        }
        if let Some(ext) = extension {
            let ext_lower = ext.to_lowercase();
            if Self::IMAGE_EXTENSIONS.contains(&ext_lower.as_str()) {
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
        let mut metadata = HashMap::new();
        let mut text_parts: Vec<String> = Vec::new();

        // Detect format
        let format = if let Some(kind) = infer::get(content) {
            let fmt = kind.mime_type().to_string();
            metadata.insert("format".to_string(), fmt.clone());
            fmt
        } else {
            mime_type.unwrap_or("image/unknown").to_string()
        };

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
        if ext == "svg" || format.contains("svg") {
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
fn read_dimensions(data: &[u8], mime: &str) -> Option<(u32, u32)> {
    if mime.contains("png") {
        read_png_dimensions(data)
    } else if mime.contains("jpeg") || mime.contains("jpg") {
        read_jpeg_dimensions(data)
    } else if mime.contains("gif") {
        read_gif_dimensions(data)
    } else if mime.contains("webp") {
        read_webp_dimensions(data)
    } else if mime.contains("bmp") {
        read_bmp_dimensions(data)
    } else {
        None
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
    fn test_can_parse_images() {
        let parser = ImageParser;
        assert!(parser.can_parse(Some("image/png"), None));
        assert!(parser.can_parse(Some("image/jpeg"), None));
        assert!(parser.can_parse(None, Some("jpg")));
        assert!(parser.can_parse(None, Some("svg")));
        assert!(parser.can_parse(None, Some("webp")));
        assert!(!parser.can_parse(None, Some("rs")));
        assert!(!parser.can_parse(Some("text/plain"), None));
    }

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
    fn test_parse_gif_dimensions() {
        let mut gif = b"GIF89a".to_vec();
        // Width: 320 (little-endian), Height: 240
        gif.extend_from_slice(&320u16.to_le_bytes());
        gif.extend_from_slice(&240u16.to_le_bytes());

        let (w, h) = read_gif_dimensions(&gif).unwrap();
        assert_eq!(w, 320);
        assert_eq!(h, 240);
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
    fn test_icon_classification() {
        let parser = ImageParser;
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&16u32.to_be_bytes());
        png.extend_from_slice(&16u32.to_be_bytes());
        png.extend_from_slice(&[8, 6, 0, 0, 0]);

        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("16x16"));
        assert!(doc.text.contains("square"));
        assert!(doc.text.contains("thumbnail"));
    }

    #[test]
    fn test_read_jpeg_dimensions() {
        // Construct a minimal valid JPEG with SOF0 marker
        // FF D8 = SOI, FF C0 = SOF0
        let mut jpeg = vec![0xFF, 0xD8]; // SOI marker
                                         // SOF0 marker at offset 2
        jpeg.push(0xFF);
        jpeg.push(0xC0);
        // Length of SOF0 segment (high byte, low byte)
        jpeg.push(0x00);
        jpeg.push(0x11); // 17 bytes
                         // Precision (offset 6 = i+4)
        jpeg.push(0x08);
        // Height: 480 (big-endian) at offset 7-8 = i+5, i+6
        jpeg.push(0x01);
        jpeg.push(0xE0);
        // Width: 640 (big-endian) at offset 9-10 = i+7, i+8
        jpeg.push(0x02);
        jpeg.push(0x80);
        // Pad to ensure i+9 < data.len() (need len > 11)
        jpeg.push(0x00);

        let (w, h) = read_jpeg_dimensions(&jpeg).unwrap();
        assert_eq!(w, 640);
        assert_eq!(h, 480);
    }

    #[test]
    fn test_read_jpeg_dimensions_sof2() {
        // JPEG with SOF2 (progressive) marker after an APP0 marker
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
                                         // APP0 marker at offset 2
        jpeg.push(0xFF);
        jpeg.push(0xE0); // APP0
        jpeg.push(0x00);
        jpeg.push(0x04); // length 4 (includes length bytes, so 2 payload bytes)
        jpeg.push(0x00);
        jpeg.push(0x00);
        // After skipping APP0: i = 2 + 2 + 4 = 8
        // SOF2 marker at offset 8
        jpeg.push(0xFF);
        jpeg.push(0xC2);
        jpeg.push(0x00);
        jpeg.push(0x11); // length
        jpeg.push(0x08); // precision (offset 12 = i+4)
                         // Height: 100 (offset 13-14 = i+5, i+6)
        jpeg.push(0x00);
        jpeg.push(0x64);
        // Width: 200 (offset 15-16 = i+7, i+8)
        jpeg.push(0x00);
        jpeg.push(0xC8);
        // Pad so i+9 < data.len() (need len > 17)
        jpeg.push(0x00);

        let (w, h) = read_jpeg_dimensions(&jpeg).unwrap();
        assert_eq!(w, 200);
        assert_eq!(h, 100);
    }

    #[test]
    fn test_read_jpeg_invalid() {
        // Not a JPEG
        assert!(read_jpeg_dimensions(b"not jpeg").is_none());
        // Too short
        assert!(read_jpeg_dimensions(&[0xFF]).is_none());
        // Valid SOI but no SOF marker
        assert!(read_jpeg_dimensions(&[0xFF, 0xD8, 0x00, 0x00]).is_none());
    }

    #[test]
    fn test_read_webp_dimensions() {
        // Construct minimal WebP VP8 lossy header
        let mut webp = Vec::new();
        webp.extend_from_slice(b"RIFF");
        // File size (placeholder, little-endian)
        webp.extend_from_slice(&100u32.to_le_bytes());
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(b"VP8 ");
        // Chunk size
        webp.extend_from_slice(&30u32.to_le_bytes());
        // 3 bytes before the frame tag (bytes 20-22)
        webp.push(0x00);
        webp.push(0x00);
        webp.push(0x00);
        // VP8 frame tag: 9D 01 2A
        webp.push(0x9D);
        webp.push(0x01);
        webp.push(0x2A);
        // Width: 320 (little-endian, 14-bit)
        webp.extend_from_slice(&320u16.to_le_bytes());
        // Height: 240 (little-endian, 14-bit)
        webp.extend_from_slice(&240u16.to_le_bytes());

        let (w, h) = read_webp_dimensions(&webp).unwrap();
        assert_eq!(w, 320);
        assert_eq!(h, 240);
    }

    #[test]
    fn test_read_webp_invalid() {
        // Not a WebP
        assert!(read_webp_dimensions(b"not webp").is_none());
        // Too short
        assert!(read_webp_dimensions(&[0; 10]).is_none());
        // Valid RIFF/WEBP but wrong chunk type
        let mut bad = Vec::new();
        bad.extend_from_slice(b"RIFF");
        bad.extend_from_slice(&100u32.to_le_bytes());
        bad.extend_from_slice(b"WEBP");
        bad.extend_from_slice(b"XXXX"); // Not VP8 or VP8L
        bad.extend_from_slice(&[0; 20]);
        assert!(read_webp_dimensions(&bad).is_none());
    }

    #[test]
    fn test_read_bmp_dimensions() {
        // Construct minimal BMP header
        let mut bmp = vec![0u8; 26];
        bmp[0] = b'B';
        bmp[1] = b'M';
        // Width at offset 18 (little-endian u32): 800
        bmp[18..22].copy_from_slice(&800u32.to_le_bytes());
        // Height at offset 22 (little-endian u32): 600
        bmp[22..26].copy_from_slice(&600u32.to_le_bytes());

        let (w, h) = read_bmp_dimensions(&bmp).unwrap();
        assert_eq!(w, 800);
        assert_eq!(h, 600);
    }

    #[test]
    fn test_read_bmp_invalid() {
        // Not a BMP
        assert!(read_bmp_dimensions(b"not bmp").is_none());
        // Too short
        assert!(read_bmp_dimensions(b"BM").is_none());
    }

    #[test]
    fn test_portrait_image() {
        let parser = ImageParser;
        // PNG with height > width (portrait: 100x200)
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes()); // width
        png.extend_from_slice(&200u32.to_be_bytes()); // height
        png.extend_from_slice(&[8, 6, 0, 0, 0]);

        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("portrait"));
    }

    #[test]
    fn test_panoramic_image() {
        let parser = ImageParser;
        // PNG with width > height * 2 (panoramic: 1000x400)
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&1000u32.to_be_bytes()); // width
        png.extend_from_slice(&400u32.to_be_bytes()); // height
        png.extend_from_slice(&[8, 6, 0, 0, 0]);

        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("panoramic"));
    }

    #[test]
    fn test_tall_image() {
        let parser = ImageParser;
        // PNG with height > width * 2 (tall: 100x300)
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes()); // width
        png.extend_from_slice(&300u32.to_be_bytes()); // height
        png.extend_from_slice(&[8, 6, 0, 0, 0]);

        let doc = parser.parse(&png, Some("image/png"), Some("png")).unwrap();
        assert!(doc.text.contains("tall vertical"));
    }

    #[test]
    fn test_parse_unknown_format() {
        let parser = ImageParser;
        // Non-image bytes but parsed with image MIME
        let garbage = b"this is not an image at all";
        let doc = parser
            .parse(garbage, Some("image/unknown"), Some("xyz"))
            .unwrap();
        assert_eq!(doc.content_type, ContentType::Image);
        // Should not have width/height since format detection fails
        assert!(!doc.metadata.contains_key("width"));
    }

    #[test]
    fn test_can_parse_case_insensitive() {
        let parser = ImageParser;
        assert!(parser.can_parse(None, Some("JPG")));
        assert!(parser.can_parse(None, Some("PNG")));
        assert!(parser.can_parse(None, Some("Jpeg")));
        assert!(parser.can_parse(None, Some("WEBP")));
        assert!(parser.can_parse(None, Some("BMP")));
    }

    #[test]
    fn test_webp_extension() {
        let parser = ImageParser;
        // Parse with "webp" extension - should include WebP note
        let garbage = b"not actually webp";
        let doc = parser.parse(garbage, None, Some("webp")).unwrap();
        assert!(doc.text.contains("WebP image"));
    }

    #[test]
    fn test_bmp_extension() {
        let parser = ImageParser;
        let garbage = b"not actually bmp";
        let doc = parser.parse(garbage, None, Some("bmp")).unwrap();
        assert!(doc.text.contains("BMP bitmap"));
    }

    #[test]
    fn test_ico_extension() {
        let parser = ImageParser;
        let garbage = b"not actually ico";
        let doc = parser.parse(garbage, None, Some("ico")).unwrap();
        assert!(doc.text.contains("icon favicon"));
    }

    #[test]
    fn test_jpeg_extension() {
        let parser = ImageParser;
        let garbage = b"not actually jpeg";
        let doc = parser.parse(garbage, None, Some("jpeg")).unwrap();
        assert!(doc.text.contains("JPEG photograph"));
    }

    #[test]
    fn test_gif_extension() {
        let parser = ImageParser;
        let garbage = b"not gif";
        let doc = parser.parse(garbage, None, Some("gif")).unwrap();
        assert!(doc.text.contains("GIF image animation"));
    }

    #[test]
    fn test_read_dimensions_unknown_mime() {
        // Mime type that doesn't match any known format
        assert!(read_dimensions(b"data", "application/octet-stream").is_none());
    }

    #[test]
    fn test_parser_name() {
        let parser = ImageParser;
        assert_eq!(parser.name(), "image");
    }

    #[test]
    fn test_parse_no_extension() {
        let parser = ImageParser;
        // Parse with no extension, uses default "image"
        let garbage = b"some data";
        let doc = parser.parse(garbage, Some("image/unknown"), None).unwrap();
        assert_eq!(doc.metadata.get("extension").unwrap(), "image");
    }

    #[test]
    fn test_read_png_invalid() {
        // Not a PNG
        assert!(read_png_dimensions(b"not png").is_none());
        // Too short
        assert!(read_png_dimensions(&[0x89, b'P', b'N', b'G']).is_none());
    }

    #[test]
    fn test_read_gif_invalid() {
        assert!(read_gif_dimensions(b"not gif").is_none());
        assert!(read_gif_dimensions(b"GIF89").is_none()); // too short
    }

    #[test]
    fn test_svg_extract_text_multiple_elements() {
        let svg = r#"<svg><title>Title 1</title><desc>Description</desc><text x="10" y="20">Label A</text><text x="30" y="40">Label B</text></svg>"#;
        let result = extract_svg_text(svg);
        assert!(result.contains("Title 1"));
        assert!(result.contains("Description"));
        assert!(result.contains("Label A"));
        assert!(result.contains("Label B"));
    }

    #[test]
    fn test_svg_extract_text_empty_tags() {
        let svg = r#"<svg><title></title><text x="0" y="0">   </text></svg>"#;
        let result = extract_svg_text(svg);
        // Empty and whitespace-only content should be filtered out
        assert!(result.is_empty());
    }

    #[test]
    fn test_svg_with_no_extractable_tags() {
        let svg = r#"<svg><circle cx="50" cy="50" r="40"/></svg>"#;
        let result = extract_svg_text(svg);
        assert!(result.is_empty());
    }

    #[test]
    fn test_can_parse_tiff_extensions() {
        let parser = ImageParser;
        assert!(parser.can_parse(None, Some("tiff")));
        assert!(parser.can_parse(None, Some("tif")));
        assert!(parser.can_parse(None, Some("TIF")));
    }

    #[test]
    fn test_can_parse_all_image_mimes() {
        let parser = ImageParser;
        assert!(parser.can_parse(Some("image/gif"), None));
        assert!(parser.can_parse(Some("image/webp"), None));
        assert!(parser.can_parse(Some("image/svg+xml"), None));
        assert!(parser.can_parse(Some("image/bmp"), None));
        assert!(parser.can_parse(Some("image/tiff"), None));
        assert!(parser.can_parse(Some("image/x-icon"), None));
    }

    #[test]
    fn test_read_webp_vp8l_lossless() {
        // Construct minimal WebP VP8L (lossless) header
        let mut webp = Vec::new();
        webp.extend_from_slice(b"RIFF");
        webp.extend_from_slice(&100u32.to_le_bytes());
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(b"VP8L");
        // Chunk size
        webp.extend_from_slice(&30u32.to_le_bytes());
        // Signature byte (offset 20)
        webp.push(0x2F);
        // VP8L bitstream: width and height encoded in bytes 21-24
        // width = (b0 | (b1 << 8)) & 0x3FFF, then +1
        // height = ((b1 >> 6) | (b2 << 2) | (b3 << 10)) & 0x3FFF, then +1
        // Encode width=99 (stored as 99-1=98=0x62) and height=49 (stored as 49-1=48=0x30)
        // b0 = 0x62, b1 = 0x00, so width = 0x0062 & 0x3FFF = 98, +1 = 99
        // height: (b1>>6) | (b2<<2) | (b3<<10) = 0 | (0x0C << 2) | (0 << 10) = 0x30 = 48, +1 = 49
        // For w=100 (stored 99=0x63), h=50 (stored 49=0x31):
        //   b0 = 0x63, b1 = 0x40 (b1[7:6]=01 for h low bits, b1[5:0]=0 for w high)
        //   b2 = 0x0C (h bits[9:2]), b3 = 0x00
        // Check: w = (0x63 | (0x40 << 8)) & 0x3FFF = 0x63 = 99, +1 = 100
        // Check: h = (0x40>>6) | (0x0C<<2) | (0<<10) = 1 | 48 | 0 = 49, +1 = 50
        webp.push(0x63); // b0 (offset 21)
        webp.push(0x40); // b1 (offset 22)
        webp.push(0x0C); // b2 (offset 23)
        webp.push(0x00); // b3 (offset 24)
                         // Pad to >= 30 bytes (outer length check requires data.len() >= 30)
        webp.extend_from_slice(&[0u8; 5]);

        let (w, h) = read_webp_dimensions(&webp).unwrap();
        assert_eq!(w, 100);
        assert_eq!(h, 50);
    }

    #[test]
    fn test_parse_bmp_with_dimensions() {
        let parser = ImageParser;
        // Construct minimal BMP header with dimensions
        let mut bmp = vec![0u8; 54]; // standard BMP header size
        bmp[0] = b'B';
        bmp[1] = b'M';
        bmp[18..22].copy_from_slice(&640u32.to_le_bytes());
        bmp[22..26].copy_from_slice(&480u32.to_le_bytes());

        let doc = parser.parse(&bmp, Some("image/bmp"), Some("bmp")).unwrap();
        assert!(doc.text.contains("640x480"));
        assert!(doc.text.contains("BMP bitmap"));
        assert!(doc.text.contains("landscape"));
    }

    #[test]
    fn test_svg_extension_triggers_text_extraction() {
        let parser = ImageParser;
        let svg = b"<svg><title>My SVG</title><text>Visible</text></svg>";
        let doc = parser
            .parse(svg, Some("image/svg+xml"), Some("svg"))
            .unwrap();
        assert!(doc.text.contains("My SVG"));
        assert!(doc.text.contains("Visible"));
        assert!(doc.text.contains("SVG vector graphic"));
    }
}
