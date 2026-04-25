//! Shared HTML/XHTML text extraction.
//!
//! Used by the `web` and `epub` parsers. Email uses its own byte-oriented
//! stripper because its semantics around line-folding differ.

/// Extract plain text from HTML markup, returning `(body_text, title)`.
///
/// - Discards `<script>` / `<style>` blocks entirely.
/// - Maps block-level tags to single newlines.
/// - Decodes a small set of common entities (`&amp;`, `&lt;`, `&gt;`,
///   `&quot;`, `&#39;`, `&nbsp;`).
/// - Collapses runs of whitespace via [`collapse_whitespace`].
///
/// `title` is `Some` iff a `<title>...</title>` was seen.
pub(crate) fn extract(html: &str) -> (String, Option<String>) {
    let mut text = String::with_capacity(html.len() / 2);
    let mut title: Option<String> = None;
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut in_title_tag = false;
    let mut tag_name = String::new();
    let mut title_text = String::new();

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            continue;
        }

        if ch == '>' && in_tag {
            in_tag = false;
            let tag_lower = tag_name.to_ascii_lowercase();

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

            if is_block_tag(tag_lower.trim_start_matches('/')) {
                text.push('\n');
            }
            continue;
        }

        if in_tag {
            tag_name.push(ch);
        } else if in_title_tag {
            title_text.push(ch);
        } else if !in_script && !in_style {
            text.push(ch);
        }
    }

    let decoded = decode_entities(&text);
    (collapse_whitespace(&decoded), title)
}

fn is_block_tag(tag: &str) -> bool {
    matches!(
        tag,
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
    )
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Collapse runs of whitespace, preserving single newlines.
pub(crate) fn collapse_whitespace(s: &str) -> String {
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
    fn extract_strips_scripts_and_styles() {
        let html = "<p>Visible</p><script>alert('xss')</script><style>p{}</style><p>After</p>";
        let (text, _) = extract(html);
        assert!(text.contains("Visible"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("p{}"));
    }

    #[test]
    fn extract_decodes_entities() {
        let html = "<p>5 &lt; 10 &amp; 10 &gt; 5</p>";
        let (text, _) = extract(html);
        assert!(text.contains("5 < 10 & 10 > 5"));
    }

    #[test]
    fn extract_returns_title_when_present() {
        let html = "<html><title>Hello</title><body>World</body></html>";
        let (_, title) = extract(html);
        assert_eq!(title.as_deref(), Some("Hello"));
    }

    #[test]
    fn extract_returns_none_when_no_title() {
        let html = "<body>No title here</body>";
        let (_, title) = extract(html);
        assert!(title.is_none());
    }

    #[test]
    fn collapse_whitespace_preserves_single_newline() {
        assert_eq!(collapse_whitespace("a\n\n\nb"), "a\nb");
        assert_eq!(collapse_whitespace("a    b"), "a b");
        assert_eq!(collapse_whitespace("  a  "), "a");
    }
}
