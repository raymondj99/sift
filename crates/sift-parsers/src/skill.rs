//! SKILL.md frontmatter parser.
//!
//! Extracts YAML frontmatter from SKILL.md files into structured metadata.
//! The frontmatter is delimited by `---` markers at the start of the file.

use std::collections::HashMap;

/// Parsed SKILL.md frontmatter fields.
#[derive(Debug, Clone, Default)]
pub struct SkillFrontmatter {
    /// Skill name (kebab-case, max 64 chars).
    pub name: Option<String>,
    /// What it does and when to use it.
    pub description: Option<String>,
    /// License name or reference.
    pub license: Option<String>,
    /// Environment requirements.
    pub compatibility: Option<String>,
    /// Space-delimited list of pre-approved tools.
    pub allowed_tools: Option<String>,
    /// All raw key-value pairs from the frontmatter.
    pub raw: HashMap<String, String>,
}

/// Parse YAML frontmatter from a SKILL.md file's content.
///
/// Returns `Some((frontmatter, body))` where `body` is the markdown content
/// after the closing `---` delimiter. Returns `None` if no valid frontmatter
/// is found.
pub fn parse_frontmatter(content: &str) -> Option<(SkillFrontmatter, &str)> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }

    // Skip the opening `---` and any trailing whitespace on that line
    let after_open = &trimmed[3..];
    let after_open = after_open.strip_prefix('\r').unwrap_or(after_open);
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);

    // Find the closing `---`
    let close_idx = find_closing_delimiter(after_open)?;
    let yaml_block = &after_open[..close_idx];

    // Body starts after the closing `---` line
    let rest = &after_open[close_idx + 3..];
    let body = rest
        .strip_prefix('\r')
        .unwrap_or(rest)
        .strip_prefix('\n')
        .unwrap_or(rest);

    let raw = parse_yaml_kv(yaml_block);

    let fm = SkillFrontmatter {
        name: raw.get("name").cloned(),
        description: raw.get("description").cloned(),
        license: raw.get("license").cloned(),
        compatibility: raw.get("compatibility").cloned(),
        allowed_tools: raw.get("allowed-tools").cloned(),
        raw,
    };

    Some((fm, body))
}

/// Find the byte offset of the closing `---` delimiter.
/// It must appear at the start of a line.
fn find_closing_delimiter(content: &str) -> Option<usize> {
    let mut offset = 0;
    for line in content.lines() {
        if line.trim() == "---" {
            return Some(offset);
        }
        // +1 for the newline character (handle \r\n)
        offset += line.len() + 1;
        if content.as_bytes().get(offset.wrapping_sub(1)) == Some(&b'\r') {
            // Adjust if the previous byte was \r (CRLF)
        }
    }
    None
}

/// Parse simple YAML key-value pairs from a block of text.
/// Handles single-line `key: value` pairs. Ignores indented/nested lines.
fn parse_yaml_kv(block: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_key: Option<String> = None;
    let mut current_value = String::new();

    for line in block.lines() {
        // Skip empty lines and comments
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Check if this is a top-level key (not indented)
        if !line.starts_with(' ') && !line.starts_with('\t') {
            // Save previous key-value if any
            if let Some(key) = current_key.take() {
                map.insert(key, current_value.trim().to_string());
                current_value.clear();
            }

            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_string();
                let value = value.trim();

                // Handle quoted values
                let value = strip_quotes(value);

                if key.is_empty() {
                    continue;
                }

                if value.is_empty() {
                    // Could be a multi-line value or a mapping — just store key
                    current_key = Some(key);
                } else {
                    map.insert(key, value.to_string());
                }
            }
        }
        // Indented continuation lines are ignored for simplicity
    }

    // Save last key-value
    if let Some(key) = current_key.take() {
        if !current_value.trim().is_empty() {
            map.insert(key, current_value.trim().to_string());
        }
    }

    map
}

/// Strip surrounding quotes (single or double) from a value.
fn strip_quotes(s: &str) -> &str {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_frontmatter() {
        let content = "---\nname: pdf-processing\ndescription: Extract PDF text\n---\n# Body\n";
        let (fm, body) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("pdf-processing"));
        assert_eq!(fm.description.as_deref(), Some("Extract PDF text"));
        assert!(body.starts_with("# Body"));
    }

    #[test]
    fn test_parse_all_fields() {
        let content = "\
---
name: code-review
description: Review code for quality and bugs
license: MIT
compatibility: Node.js 18+
allowed-tools: Bash Read Edit
---
# Code Review Instructions
";
        let (fm, body) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("code-review"));
        assert_eq!(
            fm.description.as_deref(),
            Some("Review code for quality and bugs")
        );
        assert_eq!(fm.license.as_deref(), Some("MIT"));
        assert_eq!(fm.compatibility.as_deref(), Some("Node.js 18+"));
        assert_eq!(fm.allowed_tools.as_deref(), Some("Bash Read Edit"));
        assert!(body.starts_with("# Code Review"));
    }

    #[test]
    fn test_parse_quoted_values() {
        let content = "---\nname: \"my-skill\"\ndescription: 'A skill'\n---\nbody\n";
        let (fm, _) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("my-skill"));
        assert_eq!(fm.description.as_deref(), Some("A skill"));
    }

    #[test]
    fn test_no_frontmatter() {
        assert!(parse_frontmatter("# Just markdown\nNo frontmatter here").is_none());
    }

    #[test]
    fn test_empty_frontmatter() {
        let content = "---\n---\n# Body\n";
        let (fm, body) = parse_frontmatter(content).unwrap();
        assert!(fm.name.is_none());
        assert!(body.starts_with("# Body"));
    }

    #[test]
    fn test_frontmatter_with_comments() {
        let content = "---\nname: test\n# This is a comment\ndescription: A test skill\n---\n";
        let (fm, _) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("test"));
        assert_eq!(fm.description.as_deref(), Some("A test skill"));
    }

    #[test]
    fn test_raw_map_contains_all_keys() {
        let content = "---\nname: test\ncustom-field: custom value\n---\n";
        let (fm, _) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.raw.get("custom-field").map(String::as_str), Some("custom value"));
    }

    #[test]
    fn test_no_closing_delimiter() {
        let content = "---\nname: broken\nNo closing delimiter";
        assert!(parse_frontmatter(content).is_none());
    }

    #[test]
    fn test_whitespace_before_frontmatter() {
        let content = "  \n---\nname: test\n---\nbody";
        let (fm, _) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("test"));
    }
}
