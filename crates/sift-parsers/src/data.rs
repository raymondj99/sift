use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};

/// Parser for structured data formats: CSV, JSON, JSONL, TOML, YAML.
pub struct DataParser;

impl DataParser {
    const DATA_MIMES: &[&str] = &[
        "application/json",
        "application/jsonlines",
        "application/toml",
        "application/yaml",
        "text/csv",
    ];

    const DATA_EXTENSIONS: &[&str] = &["json", "jsonl", "toml", "yaml", "yml", "csv"];
}

impl Parser for DataParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::DATA_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::DATA_EXTENSIONS.contains(&ext) {
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
        let raw = String::from_utf8_lossy(content);
        let ext = extension.unwrap_or("");

        let text = match ext {
            "csv" => parse_csv(&raw),
            "json" => parse_json(&raw),
            "jsonl" => parse_jsonl(&raw),
            _ => raw.into_owned(),
        };

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Data,
            metadata: Default::default(),
        })
    }

    fn name(&self) -> &str {
        "data"
    }
}

/// Convert CSV to a text representation for embedding.
fn parse_csv(content: &str) -> String {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut output = String::new();

    // Clone headers so we don't hold a borrow on reader
    let headers: Option<Vec<String>> = reader
        .headers()
        .ok()
        .map(|h| h.iter().map(|s| s.to_string()).collect());

    for record in reader.records().flatten() {
        let values: Vec<&str> = record.iter().collect();
        for (i, value) in values.iter().enumerate() {
            if let Some(ref hdrs) = headers {
                if i < hdrs.len() {
                    output.push_str(&hdrs[i]);
                    output.push_str(": ");
                }
            }
            output.push_str(value);
            if i < values.len() - 1 {
                output.push_str(", ");
            }
        }
        output.push('\n');
    }

    if output.is_empty() {
        content.to_string()
    } else {
        output
    }
}

/// Flatten JSON to a readable text representation.
fn parse_json(content: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value) => flatten_json_value(&value, ""),
        Err(_) => content.to_string(),
    }
}

/// Parse JSONL: each line is a JSON object.
fn parse_jsonl(content: &str) -> String {
    let mut output = String::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            output.push_str(&flatten_json_value(&value, ""));
            output.push('\n');
        }
    }
    if output.is_empty() {
        content.to_string()
    } else {
        output
    }
}

fn flatten_json_value(value: &serde_json::Value, prefix: &str) -> String {
    let mut output = String::new();
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };
                output.push_str(&flatten_json_value(val, &new_prefix));
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                let new_prefix = format!("{}[{}]", prefix, i);
                output.push_str(&flatten_json_value(val, &new_prefix));
            }
        }
        _ => {
            if !prefix.is_empty() {
                output.push_str(prefix);
                output.push_str(": ");
            }
            output.push_str(value.to_string().trim_matches('"'));
            output.push('\n');
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv() {
        let parser = DataParser;
        let csv_data = b"name,age\nAlice,30\nBob,25\n";
        let doc = parser.parse(csv_data, None, Some("csv")).unwrap();
        assert!(doc.text.contains("name: Alice"));
        assert!(doc.text.contains("age: 30"));
        assert_eq!(doc.content_type, ContentType::Data);
    }

    #[test]
    fn test_parse_json() {
        let parser = DataParser;
        let json_data = br#"{"name":"Alice","age":30}"#;
        let doc = parser.parse(json_data, None, Some("json")).unwrap();
        assert!(doc.text.contains("name: Alice"));
        assert!(doc.text.contains("age: 30"));
    }

    #[test]
    fn test_parse_jsonl() {
        let parser = DataParser;
        let jsonl = b"{\"a\":1}\n{\"a\":2}\n";
        let doc = parser.parse(jsonl, None, Some("jsonl")).unwrap();
        assert!(doc.text.contains("a: 1"));
        assert!(doc.text.contains("a: 2"));
    }

    #[test]
    fn test_can_parse_data() {
        let parser = DataParser;
        assert!(parser.can_parse(None, Some("json")));
        assert!(parser.can_parse(None, Some("csv")));
        assert!(parser.can_parse(Some("text/csv"), None));
        assert!(!parser.can_parse(None, Some("rs")));
    }
}
