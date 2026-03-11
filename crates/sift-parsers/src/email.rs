use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Parser for email formats: EML and MBOX.
pub struct EmailParser;

const MAX_MBOX_MESSAGES: usize = 1000;

impl EmailParser {
    const EMAIL_MIMES: &[&str] = &["message/rfc822", "application/mbox"];
    const EMAIL_EXTENSIONS: &[&str] = &["eml", "mbox"];
}

impl Parser for EmailParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::EMAIL_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::EMAIL_EXTENSIONS.contains(&ext) {
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
        if ext == "mbox" {
            parse_mbox(content)
        } else {
            parse_eml(content)
        }
    }

    fn name(&self) -> &'static str {
        "email"
    }
}

fn parse_eml(content: &[u8]) -> SiftResult<ParsedDocument> {
    let message = mail_parser::MessageParser::default()
        .parse(content)
        .ok_or_else(|| sift_core::SiftError::Parse {
            path: "eml".to_string(),
            message: "Failed to parse email message".to_string(),
        })?;

    let mut metadata = HashMap::new();
    let mut parts = Vec::new();

    // Extract headers
    let subject = message.subject().unwrap_or("(no subject)").to_string();
    parts.push(format!("Subject: {subject}"));

    if let Some(from) = message.from() {
        let addrs: Vec<String> = from.iter().map(format_address).collect();
        let from_str = addrs.join(", ");
        parts.push(format!("From: {from_str}"));
        metadata.insert("from".to_string(), from_str);
    }

    if let Some(to) = message.to() {
        let addrs: Vec<String> = to.iter().map(format_address).collect();
        parts.push(format!("To: {}", addrs.join(", ")));
    }

    if let Some(date) = message.date() {
        let date_str = date.to_rfc3339();
        parts.push(format!("Date: {date_str}"));
        metadata.insert("date".to_string(), date_str);
    }

    parts.push(String::new()); // blank line separator

    // Extract body: prefer text/plain
    if let Some(body) = message.body_text(0) {
        parts.push(body.into_owned());
    } else if let Some(body) = message.body_html(0) {
        parts.push(strip_html_tags(&body));
    }

    metadata.insert("subject".to_string(), subject.clone());

    let mut text = parts.join("\n");
    let trimmed_len = text.trim().len();
    let start = text.len() - text.trim_start().len();
    text = text[start..start + trimmed_len].to_string();

    Ok(ParsedDocument {
        text,
        title: Some(subject),
        language: None,
        content_type: ContentType::Text,
        metadata,
    })
}

fn parse_mbox(content: &[u8]) -> SiftResult<ParsedDocument> {
    // Zero-copy: try UTF-8 borrow first, fall back to lossy
    let raw: Cow<str> = match std::str::from_utf8(content) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => String::from_utf8_lossy(content),
    };

    // Split MBOX on "\nFrom " boundaries (standard mbox format)
    let mut messages: Vec<&str> = Vec::new();
    let mut start = 0;

    // Handle first message (may or may not start with "From ")
    for (i, _) in raw.match_indices("\nFrom ") {
        messages.push(&raw[start..i]);
        start = i + 1; // skip the leading \n
    }
    messages.push(&raw[start..]);

    let mut output = String::with_capacity(content.len() / 2);
    let mut count = 0;

    for msg_str in &messages {
        let trimmed = msg_str.trim();
        if trimmed.is_empty() {
            continue;
        }
        count += 1;

        if count > MAX_MBOX_MESSAGES {
            let _ = write!(
                output,
                "\n--- truncated: {MAX_MBOX_MESSAGES} message limit reached ---\n"
            );
            break;
        }

        if let Some(parsed) = mail_parser::MessageParser::default().parse(trimmed.as_bytes()) {
            let subject = parsed.subject().unwrap_or("(no subject)");
            let _ = writeln!(output, "--- Message {count} ---");
            let _ = writeln!(output, "Subject: {subject}");
            if let Some(from) = parsed.from() {
                let addrs: Vec<String> = from.iter().map(format_address).collect();
                let _ = writeln!(output, "From: {}", addrs.join(", "));
            }
            if let Some(body) = parsed.body_text(0) {
                output.push_str(&body);
            }
            output.push('\n');
        }
    }

    // Cap count at the limit for metadata
    let reported_count = std::cmp::min(count, MAX_MBOX_MESSAGES);

    let mut metadata = HashMap::new();
    metadata.insert("message_count".to_string(), reported_count.to_string());

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

fn format_address(addr: &mail_parser::Addr) -> String {
    match (&addr.name, &addr.address) {
        (Some(name), Some(email)) => format!("{name} <{email}>"),
        (None, Some(email)) => email.to_string(),
        (Some(name), None) => name.to_string(),
        (None, None) => "(unknown)".to_string(),
    }
}

/// Byte-slice HTML tag stripping with single-pass whitespace cleanup.
fn strip_html_tags(html: &str) -> String {
    let bytes = html.as_bytes();
    let mut output = String::with_capacity(html.len());
    let mut i = 0;
    let mut line_start = true;
    let mut blank_line = true;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip to closing '>'
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1; // skip '>'
            }
        } else if bytes[i] == b'\n' || bytes[i] == b'\r' {
            // End of line: emit newline only if line had content
            if !blank_line {
                output.push('\n');
            }
            // Skip \r\n
            if bytes[i] == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 1;
            }
            i += 1;
            line_start = true;
            blank_line = true;
        } else if bytes[i] == b' ' || bytes[i] == b'\t' {
            if line_start {
                // Skip leading whitespace
                i += 1;
            } else {
                output.push(bytes[i] as char);
                i += 1;
            }
        } else {
            output.push(bytes[i] as char);
            line_start = false;
            blank_line = false;
            i += 1;
        }
    }

    // Trim trailing whitespace
    let trimmed_len = output.trim_end().len();
    output.truncate(trimmed_len);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_email() {
        let parser = EmailParser;
        assert!(parser.can_parse(Some("message/rfc822"), None));
        assert!(parser.can_parse(Some("application/mbox"), None));
        assert!(parser.can_parse(None, Some("eml")));
        assert!(parser.can_parse(None, Some("mbox")));
        assert!(!parser.can_parse(None, Some("txt")));
    }

    #[test]
    fn test_parse_simple_eml() {
        let parser = EmailParser;
        let eml = b"From: alice@example.com\r\nTo: bob@example.com\r\nSubject: Test Email\r\nDate: Mon, 1 Jan 2024 12:00:00 +0000\r\n\r\nHello Bob,\r\n\r\nThis is a test email.\r\n";

        let doc = parser
            .parse(eml, Some("message/rfc822"), Some("eml"))
            .unwrap();
        assert!(doc.text.contains("Subject: Test Email"));
        assert!(doc.text.contains("From: alice@example.com"));
        assert!(doc.text.contains("Hello Bob"));
        assert_eq!(doc.title, Some("Test Email".to_string()));
        assert_eq!(doc.content_type, ContentType::Text);
    }

    #[test]
    fn test_strip_html_tags() {
        let html = "<html><body><p>Hello <b>World</b></p></body></html>";
        let text = strip_html_tags(html);
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_strip_html_preserves_text_between_block_elements() {
        let html = "<div>First</div><div>Second</div><p>Third</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
        assert!(text.contains("Third"));
    }

    #[test]
    fn test_strip_html_handles_empty_input() {
        assert_eq!(strip_html_tags(""), "");
    }

    #[test]
    fn test_strip_html_passes_through_plain_text() {
        assert_eq!(strip_html_tags("no tags here"), "no tags here");
    }

    #[test]
    fn test_eml_extracts_all_headers() {
        let parser = EmailParser;
        let eml = b"From: alice@example.com\r\nTo: bob@example.com\r\nSubject: Important\r\nDate: Mon, 1 Jan 2024 12:00:00 +0000\r\n\r\nBody text.\r\n";
        let doc = parser.parse(eml, None, Some("eml")).unwrap();
        assert!(doc.text.contains("From: alice@example.com"));
        assert!(doc.text.contains("To: bob@example.com"));
        assert!(doc.text.contains("Subject: Important"));
        assert!(doc.text.contains("Date:"));
        assert!(doc.text.contains("Body text."));
        assert_eq!(doc.title, Some("Important".to_string()));
    }

    #[test]
    fn test_eml_without_body_still_parses_headers() {
        let parser = EmailParser;
        let eml = b"From: test@example.com\r\nSubject: No body\r\n\r\n";
        let doc = parser.parse(eml, None, Some("eml")).unwrap();
        assert!(doc.text.contains("Subject: No body"));
    }

    #[test]
    fn test_mbox_with_multiple_messages() {
        let parser = EmailParser;
        let mbox = b"From sender@example.com Mon Jan 01 12:00:00 2024\r\nFrom: alice@example.com\r\nSubject: First\r\n\r\nBody one.\r\n\nFrom sender@example.com Mon Jan 01 13:00:00 2024\r\nFrom: bob@example.com\r\nSubject: Second\r\n\r\nBody two.\r\n";
        let doc = parser.parse(mbox, None, Some("mbox")).unwrap();
        assert!(doc.text.contains("Message 1"));
        assert!(doc.text.contains("Message 2"));
        assert!(doc.text.contains("Subject: First"));
        assert!(doc.text.contains("Subject: Second"));
        assert_eq!(
            doc.metadata
                .get("message_count")
                .unwrap()
                .parse::<usize>()
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_mbox_valid_utf8_takes_zero_copy_path() {
        // Pure ASCII mbox — from_utf8 succeeds, no lossy copy needed.
        // We can't directly observe zero-copy, but we verify correctness.
        let parser = EmailParser;
        let ascii_mbox = b"From sender@test.com Mon Jan 01 00:00:00 2024\r\nSubject: ASCII\r\n\r\nPlain text.\r\n";
        let doc = parser.parse(ascii_mbox, None, Some("mbox")).unwrap();
        assert!(doc.text.contains("Plain text."));
    }

    #[test]
    fn test_mbox_invalid_utf8_falls_back_to_lossy() {
        let parser = EmailParser;
        let mut data =
            b"From sender@test.com Mon Jan 01 00:00:00 2024\r\nSubject: Binary\r\n\r\n".to_vec();
        data.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        let doc = parser.parse(&data, None, Some("mbox")).unwrap();
        // Should not panic — lossy path handles it
        assert!(doc.text.contains("Subject: Binary"));
    }

    #[test]
    fn test_empty_eml_returns_error() {
        let parser = EmailParser;
        let result = parser.parse(b"", None, Some("eml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_garbage_input_parses_leniently() {
        // mail_parser is permissive — garbage becomes a body-only message
        let parser = EmailParser;
        let result = parser.parse(b"this is not an email at all", None, Some("eml"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, Some("(no subject)".to_string()));
    }
}
