use sift_core::{ParsedDocument, SiftResult};

/// A parser that extracts text content from a specific file format.
pub trait Parser: Send + Sync {
    /// Returns true if this parser can handle the given MIME type or extension.
    ///
    /// Most parsers can implement this with a one-liner via [`matches`];
    /// override only when extra logic is needed (e.g. magic-byte sniffing).
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool;

    /// Parse the file content and return extracted text + metadata.
    fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument>;

    /// Human-readable name for this parser.
    fn name(&self) -> &str;
}

/// Default `can_parse` body shared by every format-specific parser.
///
/// Match rules:
/// - `mime_type` is lowercased and stripped of any `; charset=…` parameters,
///   then matched against `mime_patterns` with `starts_with` (so an entry
///   like `"text/html"` matches `"text/html"` and `"text/html-thing"`, and
///   an entry like `"audio/"` matches every audio family member).
/// - `extension` is matched ASCII-case-insensitively against `extensions`.
///
/// Either match wins. Both `None` arguments → `false`.
pub fn matches(
    mime_type: Option<&str>,
    extension: Option<&str>,
    mime_patterns: &[&str],
    extensions: &[&str],
) -> bool {
    if let Some(mime) = mime_type {
        let mime_lower = mime.to_ascii_lowercase();
        let mime_base = mime_lower
            .split(';')
            .next()
            .unwrap_or(mime_lower.as_str())
            .trim();
        if mime_patterns.iter().any(|p| mime_base.starts_with(p)) {
            return true;
        }
    }
    if let Some(ext) = extension {
        let ext_lower = ext.to_ascii_lowercase();
        if extensions.iter().any(|e| *e == ext_lower) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn mime_exact_match() {
        assert!(matches(
            Some("application/pdf"),
            None,
            &["application/pdf"],
            &[]
        ));
    }

    #[test]
    fn mime_with_parameters() {
        assert!(matches(
            Some("text/html; charset=utf-8"),
            None,
            &["text/html"],
            &[]
        ));
    }

    #[test]
    fn mime_family_prefix() {
        assert!(matches(Some("audio/mpeg"), None, &["audio/"], &[]));
        assert!(matches(Some("audio/flac"), None, &["audio/"], &[]));
    }

    #[test]
    fn mime_uppercase_normalized() {
        assert!(matches(
            Some("APPLICATION/PDF"),
            None,
            &["application/pdf"],
            &[]
        ));
    }

    #[test]
    fn extension_case_insensitive() {
        assert!(matches(None, Some("PDF"), &[], &["pdf"]));
        assert!(matches(None, Some("Pdf"), &[], &["pdf"]));
    }

    #[test]
    fn no_match_returns_false() {
        assert!(!matches(
            Some("foo/bar"),
            Some("baz"),
            &["application/pdf"],
            &["pdf"]
        ));
    }

    #[test]
    fn both_none_returns_false() {
        assert!(!matches(None, None, &["application/pdf"], &["pdf"]));
    }
}
