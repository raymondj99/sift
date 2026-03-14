use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Parser for Jupyter Notebook (.ipynb) files.
/// Extracts code and markdown cells with their outputs.
pub struct NotebookParser;

impl Parser for NotebookParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if mime == "application/x-ipynb+json" {
                return true;
            }
        }
        if let Some(ext) = extension {
            if ext == "ipynb" {
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
        let raw = std::str::from_utf8(content).map_err(|e| sift_core::SiftError::Parse {
            path: "ipynb".to_string(),
            message: format!("Invalid UTF-8: {e}"),
        })?;

        let notebook: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| sift_core::SiftError::Parse {
                path: "ipynb".to_string(),
                message: format!("Invalid JSON: {e}"),
            })?;

        // Detect language from kernelspec metadata
        let language = notebook
            .get("metadata")
            .and_then(|m| m.get("kernelspec"))
            .and_then(|k| k.get("language"))
            .and_then(|l| l.as_str())
            .map(std::string::ToString::to_string);

        let cells = notebook
            .get("cells")
            .and_then(|c| c.as_array())
            .ok_or_else(|| sift_core::SiftError::Parse {
                path: "ipynb".to_string(),
                message: "Missing 'cells' array".to_string(),
            })?;

        let mut output = String::new();
        let mut cell_count = 0;

        for (i, cell) in cells.iter().enumerate() {
            let cell_type = cell
                .get("cell_type")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");

            // Only extract code and markdown cells
            if cell_type != "code" && cell_type != "markdown" {
                continue;
            }

            let source = extract_source(cell);
            if source.trim().is_empty() {
                continue;
            }

            if cell_count > 0 {
                output.push('\n');
            }

            let _ = writeln!(output, "--- {} cell {} ---", cell_type, i + 1);
            output.push_str(&source);
            if !source.ends_with('\n') {
                output.push('\n');
            }

            // Include text outputs for code cells
            if cell_type == "code" {
                if let Some(outputs) = cell.get("outputs").and_then(|o| o.as_array()) {
                    let text_output = extract_text_outputs(outputs);
                    if !text_output.is_empty() {
                        let _ = writeln!(output, "--- output ---");
                        output.push_str(&text_output);
                        if !text_output.ends_with('\n') {
                            output.push('\n');
                        }
                    }
                }
            }

            cell_count += 1;
        }

        let trimmed = output.trim().to_string();

        let mut metadata = HashMap::new();
        metadata.insert("cell_count".to_string(), cell_count.to_string());

        Ok(ParsedDocument {
            text: trimmed,
            title: None,
            language,
            content_type: ContentType::Code,
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "notebook"
    }
}

/// Extract the source text from a cell. The `source` field can be a string or an array of strings.
fn extract_source(cell: &serde_json::Value) -> String {
    match cell.get("source") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(lines)) => lines
            .iter()
            .filter_map(|l| l.as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Extract text outputs from a code cell's outputs array.
fn extract_text_outputs(outputs: &[serde_json::Value]) -> String {
    let mut result = String::new();
    for output in outputs {
        let output_type = output
            .get("output_type")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let text = match output_type {
            "stream" => output.get("text"),
            "execute_result" | "display_data" => {
                output.get("data").and_then(|d| d.get("text/plain"))
            }
            _ => None,
        };

        if let Some(text_val) = text {
            match text_val {
                serde_json::Value::String(s) => result.push_str(s),
                serde_json::Value::Array(lines) => {
                    for line in lines {
                        if let Some(s) = line.as_str() {
                            result.push_str(s);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_notebook(cells_json: &str) -> String {
        format!(
            r#"{{
                "metadata": {{
                    "kernelspec": {{
                        "language": "python",
                        "display_name": "Python 3",
                        "name": "python3"
                    }}
                }},
                "cells": {cells_json},
                "nbformat": 4,
                "nbformat_minor": 5
            }}"#
        )
    }

    #[test]
    fn test_parse_code_cell() {
        let nb = make_notebook(
            r#"[{
                "cell_type": "code",
                "source": "print('hello')\n",
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "hello\n"}
                ],
                "metadata": {}
            }]"#,
        );

        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert_eq!(doc.content_type, ContentType::Code);
        assert_eq!(doc.language.as_deref(), Some("python"));
        assert!(doc.text.contains("--- code cell 1 ---"));
        assert!(doc.text.contains("print('hello')"));
        assert!(doc.text.contains("--- output ---"));
        assert!(doc.text.contains("hello"));
    }

    #[test]
    fn test_parse_markdown_cell() {
        let nb = make_notebook(
            r##"[{
                "cell_type": "markdown",
                "source": ["# Title\n", "\n", "Some description"],
                "metadata": {}
            }]"##,
        );

        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert!(doc.text.contains("--- markdown cell 1 ---"));
        assert!(doc.text.contains("# Title"));
        assert!(doc.text.contains("Some description"));
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let parser = NotebookParser;
        let result = parser.parse(b"not json", None, Some("ipynb"));
        assert!(result.is_err());
    }
}
