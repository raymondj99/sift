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
    fn test_can_parse_notebook() {
        let parser = NotebookParser;
        assert!(parser.can_parse(Some("application/x-ipynb+json"), None));
        assert!(parser.can_parse(None, Some("ipynb")));
        assert!(!parser.can_parse(None, Some("json")));
        assert!(!parser.can_parse(None, Some("py")));
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
    fn test_parse_mixed_cells() {
        let nb = make_notebook(
            r##"[
                {
                    "cell_type": "markdown",
                    "source": "# Notebook Title",
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')"],
                    "outputs": [],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": "df.head()",
                    "outputs": [
                        {"output_type": "execute_result", "data": {"text/plain": "   col1  col2\n0     1     2\n"}, "metadata": {}}
                    ],
                    "metadata": {}
                }
            ]"##,
        );

        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert!(doc.text.contains("--- markdown cell 1 ---"));
        assert!(doc.text.contains("--- code cell 2 ---"));
        assert!(doc.text.contains("--- code cell 3 ---"));
        assert!(doc.text.contains("import pandas"));
        assert!(doc.text.contains("df.head()"));
        assert!(doc.text.contains("col1"));
        assert_eq!(doc.metadata.get("cell_count").unwrap(), "3");
    }

    #[test]
    fn test_skip_empty_cells() {
        let nb = make_notebook(
            r#"[
                {"cell_type": "code", "source": "", "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": "x = 1", "outputs": [], "metadata": {}}
            ]"#,
        );

        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert_eq!(doc.metadata.get("cell_count").unwrap(), "1");
        assert!(doc.text.contains("x = 1"));
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let parser = NotebookParser;
        let result = parser.parse(b"not json", None, Some("ipynb"));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_cells_returns_error() {
        let parser = NotebookParser;
        let result = parser.parse(b"{}", None, Some("ipynb"));
        assert!(result.is_err());
    }

    #[test]
    fn test_no_kernelspec_language_is_none() {
        let nb = r#"{"metadata": {}, "cells": [{"cell_type": "code", "source": "1+1", "outputs": [], "metadata": {}}], "nbformat": 4}"#;
        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert!(doc.language.is_none());
    }

    #[test]
    fn test_skip_unknown_cell_type() {
        let nb = make_notebook(
            r#"[
                {"cell_type": "raw", "source": "raw cell content", "metadata": {}},
                {"cell_type": "code", "source": "x = 1", "outputs": [], "metadata": {}}
            ]"#,
        );
        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        // raw cells should be skipped
        assert!(!doc.text.contains("raw cell content"));
        assert!(doc.text.contains("x = 1"));
        assert_eq!(doc.metadata.get("cell_count").unwrap(), "1");
    }

    #[test]
    fn test_extract_source_array_format() {
        let cell: serde_json::Value =
            serde_json::from_str(r#"{"source": ["line 1\n", "line 2\n", "line 3"]}"#).unwrap();
        let source = extract_source(&cell);
        assert_eq!(source, "line 1\nline 2\nline 3");
    }

    #[test]
    fn test_extract_source_string_format() {
        let cell: serde_json::Value =
            serde_json::from_str(r#"{"source": "single string source"}"#).unwrap();
        let source = extract_source(&cell);
        assert_eq!(source, "single string source");
    }

    #[test]
    fn test_extract_source_missing() {
        let cell: serde_json::Value = serde_json::from_str(r#"{}"#).unwrap();
        let source = extract_source(&cell);
        assert!(source.is_empty());
    }

    #[test]
    fn test_extract_source_null() {
        let cell: serde_json::Value = serde_json::from_str(r#"{"source": null}"#).unwrap();
        let source = extract_source(&cell);
        assert!(source.is_empty());
    }

    #[test]
    fn test_extract_text_outputs_display_data() {
        let outputs: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"output_type": "display_data", "data": {"text/plain": "displayed text"}}]"#,
        )
        .unwrap();
        let result = extract_text_outputs(&outputs);
        assert_eq!(result, "displayed text");
    }

    #[test]
    fn test_extract_text_outputs_stream_array() {
        let outputs: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"output_type": "stream", "name": "stdout", "text": ["line1\n", "line2\n"]}]"#,
        )
        .unwrap();
        let result = extract_text_outputs(&outputs);
        assert_eq!(result, "line1\nline2\n");
    }

    #[test]
    fn test_extract_text_outputs_unknown_type() {
        let outputs: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"output_type": "error", "ename": "ValueError", "evalue": "bad"}]"#,
        )
        .unwrap();
        let result = extract_text_outputs(&outputs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_invalid_utf8_returns_error() {
        let parser = NotebookParser;
        let result = parser.parse(&[0xFF, 0xFE], None, Some("ipynb"));
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("UTF-8"));
    }

    #[test]
    fn test_code_cell_no_outputs_key() {
        let nb =
            make_notebook(r#"[{"cell_type": "code", "source": "print('hi')", "metadata": {}}]"#);
        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert!(doc.text.contains("print('hi')"));
        // No output section since outputs key is missing
        assert!(!doc.text.contains("--- output ---"));
    }

    #[test]
    fn test_code_cell_with_empty_output() {
        let nb = make_notebook(
            r#"[{"cell_type": "code", "source": "x = 1", "outputs": [], "metadata": {}}]"#,
        );
        let parser = NotebookParser;
        let doc = parser.parse(nb.as_bytes(), None, Some("ipynb")).unwrap();
        assert!(doc.text.contains("x = 1"));
        assert!(!doc.text.contains("--- output ---"));
    }

    #[test]
    fn test_parser_name_notebook() {
        let parser = NotebookParser;
        assert_eq!(parser.name(), "notebook");
    }
}
