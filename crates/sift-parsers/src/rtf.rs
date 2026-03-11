use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;

/// Parser for RTF (Rich Text Format) files.
/// Strips RTF control words and groups, returning clean plain text.
pub struct RtfParser;

impl Parser for RtfParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if mime == "application/rtf" || mime == "text/rtf" {
                return true;
            }
        }
        if let Some(ext) = extension {
            if ext == "rtf" {
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
        // RTF files should start with {\rtf
        if content.len() < 5 || &content[..5] != b"{\\rtf" {
            return Err(sift_core::SiftError::Parse {
                path: "rtf".to_string(),
                message: "Not a valid RTF file (missing {\\rtf header)".to_string(),
            });
        }

        let raw = String::from_utf8_lossy(content);
        let text = strip_rtf(&raw);

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Text,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &'static str {
        "rtf"
    }
}

/// Strip RTF control words, groups, and formatting to extract plain text.
fn strip_rtf(rtf: &str) -> String {
    let mut output = String::with_capacity(rtf.len() / 2);
    let chars: Vec<char> = rtf.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut depth: i32 = 0;
    // Skip content inside special destination groups like \fonttbl, \colortbl, \stylesheet, etc.
    let mut skip_depth: Option<i32> = None;

    while i < len {
        let ch = chars[i];

        match ch {
            '{' => {
                depth += 1;
                i += 1;
            }
            '}' => {
                if skip_depth == Some(depth) {
                    skip_depth = None;
                }
                depth -= 1;
                i += 1;
            }
            '\\' if i + 1 < len => {
                // Check if we are inside a skip group
                if skip_depth.is_some() {
                    // Skip until end of control word
                    i += 1;
                    while i < len
                        && chars[i] != ' '
                        && chars[i] != '\\'
                        && chars[i] != '{'
                        && chars[i] != '}'
                    {
                        i += 1;
                    }
                    // Consume trailing space after control word
                    if i < len && chars[i] == ' ' {
                        i += 1;
                    }
                    continue;
                }

                i += 1; // skip the backslash

                // Escaped literal characters
                if chars[i] == '\\' || chars[i] == '{' || chars[i] == '}' {
                    output.push(chars[i]);
                    i += 1;
                    continue;
                }

                // Unicode escape: \'XX (hex byte)
                if chars[i] == '\'' && i + 2 < len {
                    let hex = &rtf[i + 1..i + 3];
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        // Windows-1252 compatible: ASCII range maps directly
                        if byte < 128 {
                            output.push(byte as char);
                        } else {
                            // Best-effort: common Windows-1252 characters
                            output.push(decode_win1252(byte));
                        }
                    }
                    i += 3;
                    continue;
                }

                // Read the control word
                let start = i;
                while i < len && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word = &rtf[start..i];

                // Skip optional numeric parameter
                while i < len && (chars[i].is_ascii_digit() || chars[i] == '-') {
                    i += 1;
                }

                // Consume the delimiter space (if present)
                if i < len && chars[i] == ' ' {
                    i += 1;
                }

                // Handle known control words
                match word {
                    "par" | "line" => output.push('\n'),
                    "tab" => output.push('\t'),
                    // Unicode escape: \uN followed by a replacement char
                    "u" => {
                        // The numeric param was already consumed; parse it from the original
                        let param_str: String = rtf[start..i]
                            .chars()
                            .skip_while(char::is_ascii_alphabetic)
                            .take_while(|c| c.is_ascii_digit() || *c == '-')
                            .collect();
                        if let Ok(code) = param_str.parse::<i32>() {
                            let code = if code < 0 {
                                (code + 65536) as u32
                            } else {
                                code as u32
                            };
                            if let Some(c) = char::from_u32(code) {
                                output.push(c);
                            }
                        }
                        // Skip the replacement character (usually ?)
                        if i < len && chars[i] != '\\' && chars[i] != '{' && chars[i] != '}' {
                            i += 1;
                        }
                    }
                    // Skip destination groups that don't contain readable text
                    "fonttbl" | "colortbl" | "stylesheet" | "info" | "pict" | "object"
                    | "header" | "footer" | "headerl" | "headerr" | "footerl" | "footerr"
                    | "footnote" | "field" | "fldinst" | "datafield" | "themedata"
                    | "colorschememapping" | "latentstyles" | "datastore" | "xmlnstbl" => {
                        skip_depth = Some(depth);
                    }
                    // Formatting words to ignore: \b, \i, \ul, \fs, \cf, etc.
                    _ => {}
                }
            }
            _ => {
                if skip_depth.is_none() {
                    output.push(ch);
                }
                i += 1;
            }
        }
    }

    // Clean up: normalize line breaks and trim
    let output = output
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");

    // Collapse multiple blank lines into at most two newlines
    let mut result = String::with_capacity(output.len());
    let mut consecutive_newlines = 0;
    for ch in output.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                result.push(ch);
            }
        } else {
            consecutive_newlines = 0;
            result.push(ch);
        }
    }

    result.trim().to_string()
}

/// Decode a Windows-1252 byte to a Unicode character (best-effort for common chars).
fn decode_win1252(byte: u8) -> char {
    match byte {
        0x80 => '\u{20AC}', // Euro sign
        0x85 => '\u{2026}', // Ellipsis
        0x91 => '\u{2018}', // Left single quote
        0x92 => '\u{2019}', // Right single quote
        0x93 => '\u{201C}', // Left double quote
        0x94 => '\u{201D}', // Right double quote
        0x95 => '\u{2022}', // Bullet
        0x96 => '\u{2013}', // En dash
        0x97 => '\u{2014}', // Em dash
        0xA0 => '\u{00A0}', // Non-breaking space
        // Latin-1 supplement (0xA0-0xFF maps directly in Unicode)
        b @ 0xA1..=0xFF => char::from_u32(u32::from(b)).unwrap_or('?'),
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_rtf() {
        let parser = RtfParser;
        assert!(parser.can_parse(None, Some("rtf")));
        assert!(parser.can_parse(Some("application/rtf"), None));
        assert!(parser.can_parse(Some("text/rtf"), None));
        assert!(!parser.can_parse(None, Some("txt")));
        assert!(!parser.can_parse(None, Some("doc")));
    }

    #[test]
    fn test_parse_simple_rtf() {
        let rtf = br"{\rtf1\ansi Hello World}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert_eq!(doc.content_type, ContentType::Text);
        assert!(doc.text.contains("Hello World"));
    }

    #[test]
    fn test_parse_rtf_with_formatting() {
        let rtf = br"{\rtf1\ansi{\b Bold text}\par Normal text}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Bold text"));
        assert!(doc.text.contains("Normal text"));
    }

    #[test]
    fn test_parse_rtf_paragraphs() {
        let rtf = br"{\rtf1\ansi First paragraph\par Second paragraph\par Third paragraph}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("First paragraph"));
        assert!(doc.text.contains("Second paragraph"));
        assert!(doc.text.contains("Third paragraph"));
        // Paragraphs should be separated by newlines
        let lines: Vec<&str> = doc.text.lines().filter(|l| !l.is_empty()).collect();
        assert!(lines.len() >= 3);
    }

    #[test]
    fn test_parse_rtf_escaped_chars() {
        let rtf = br"{\rtf1\ansi Braces \{ and \} and backslash \\}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains('{'));
        assert!(doc.text.contains('}'));
        assert!(doc.text.contains('\\'));
    }

    #[test]
    fn test_parse_rtf_skips_fonttbl() {
        let rtf = br"{\rtf1\ansi{\fonttbl{\f0 Times New Roman;}}Hello after font table}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Hello after font table"));
        // Font table content should not appear
        assert!(!doc.text.contains("Times New Roman"));
    }

    #[test]
    fn test_invalid_rtf_returns_error() {
        let parser = RtfParser;
        let result = parser.parse(b"not rtf content", None, Some("rtf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_rtf_tabs() {
        let rtf = br"{\rtf1\ansi col1\tab col2\tab col3}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("col1\tcol2\tcol3"));
    }

    #[test]
    fn test_parser_name() {
        let parser = RtfParser;
        assert_eq!(parser.name(), "rtf");
    }

    #[test]
    fn test_parse_rtf_hex_escape_ascii() {
        // \'41 = 'A' (ASCII 0x41)
        let rtf = br"{\rtf1\ansi Letter \'41 here}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Letter A here"));
    }

    #[test]
    fn test_parse_rtf_hex_escape_win1252() {
        // \'93 = left double quote (Windows-1252)
        // \'94 = right double quote
        let rtf = br"{\rtf1\ansi \'93Hello\'94}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains('\u{201C}')); // left double quote
        assert!(doc.text.contains('\u{201D}')); // right double quote
    }

    #[test]
    fn test_parse_rtf_unicode_escape() {
        // \u8364? = Euro sign (U+20AC = 8364 decimal), ? is replacement char
        let rtf = br"{\rtf1\ansi Price: \u8364?100}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        // The euro sign should be present
        assert!(doc.text.contains('\u{20AC}'));
    }

    #[test]
    fn test_parse_rtf_line_control_word() {
        let rtf = br"{\rtf1\ansi Line one\line Line two}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        let lines: Vec<&str> = doc.text.lines().collect();
        assert!(lines.len() >= 2);
        assert!(doc.text.contains("Line one"));
        assert!(doc.text.contains("Line two"));
    }

    #[test]
    fn test_parse_rtf_skip_colortbl() {
        let rtf =
            br"{\rtf1\ansi{\colortbl;\red0\green0\blue0;\red255\green0\blue0;}After color table}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("After color table"));
        assert!(!doc.text.contains("red0"));
    }

    #[test]
    fn test_parse_rtf_skip_stylesheet() {
        let rtf = br"{\rtf1\ansi{\stylesheet{\s0 Normal;}}Visible text}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Visible text"));
        assert!(!doc.text.contains("Normal"));
    }

    #[test]
    fn test_parse_rtf_skip_info() {
        let rtf = br"{\rtf1\ansi{\info{\author John}}Document content}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Document content"));
        assert!(!doc.text.contains("John"));
    }

    #[test]
    fn test_decode_win1252_known_chars() {
        assert_eq!(decode_win1252(0x80), '\u{20AC}'); // Euro
        assert_eq!(decode_win1252(0x85), '\u{2026}'); // Ellipsis
        assert_eq!(decode_win1252(0x91), '\u{2018}'); // Left single quote
        assert_eq!(decode_win1252(0x92), '\u{2019}'); // Right single quote
        assert_eq!(decode_win1252(0x93), '\u{201C}'); // Left double quote
        assert_eq!(decode_win1252(0x94), '\u{201D}'); // Right double quote
        assert_eq!(decode_win1252(0x95), '\u{2022}'); // Bullet
        assert_eq!(decode_win1252(0x96), '\u{2013}'); // En dash
        assert_eq!(decode_win1252(0x97), '\u{2014}'); // Em dash
        assert_eq!(decode_win1252(0xA0), '\u{00A0}'); // Non-breaking space
    }

    #[test]
    fn test_decode_win1252_latin1_range() {
        // 0xA1-0xFF should map to Unicode code points directly
        assert_eq!(decode_win1252(0xA1), '\u{00A1}'); // Inverted exclamation mark
        assert_eq!(decode_win1252(0xBF), '\u{00BF}'); // Inverted question mark
        assert_eq!(decode_win1252(0xFF), '\u{00FF}'); // Latin small letter y with diaeresis
    }

    #[test]
    fn test_decode_win1252_unknown() {
        // Values in the gap (like 0x81, 0x8D, etc.) should return '?'
        assert_eq!(decode_win1252(0x81), '?');
        assert_eq!(decode_win1252(0x8D), '?');
    }

    #[test]
    fn test_parse_rtf_collapses_blank_lines() {
        // Multiple \par in a row should be collapsed
        let rtf = br"{\rtf1\ansi Hello\par\par\par\par World}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        // Count consecutive newlines - should be at most 2
        let max_consecutive = doc
            .text
            .chars()
            .fold((0u32, 0u32), |(current, max), ch| {
                if ch == '\n' {
                    let new = current + 1;
                    (new, max.max(new))
                } else {
                    (0, max)
                }
            })
            .1;
        assert!(max_consecutive <= 2);
    }

    #[test]
    fn test_parse_rtf_skip_pict() {
        let rtf = br"{\rtf1\ansi Before picture{\pict binary image data here}After picture}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains("Before picture"));
        assert!(doc.text.contains("After picture"));
        assert!(!doc.text.contains("binary image data"));
    }

    #[test]
    fn test_parse_rtf_negative_unicode() {
        // Negative unicode value: \u-4064? should map to U+F020
        // -4064 + 65536 = 61472 = 0xF020
        let rtf = br"{\rtf1\ansi \u-4064?x}";
        let parser = RtfParser;
        let doc = parser.parse(rtf, None, Some("rtf")).unwrap();
        assert!(doc.text.contains('\u{F020}'));
    }
}
