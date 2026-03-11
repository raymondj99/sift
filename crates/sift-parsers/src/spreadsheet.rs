use crate::traits::Parser;
use calamine::{open_workbook_auto_from_rs, Data, Reader};
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::Cursor;

/// Parser for spreadsheet formats: XLSX, XLS, ODS.
pub struct SpreadsheetParser;

impl SpreadsheetParser {
    const SPREADSHEET_MIMES: &[&str] = &[
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.oasis.opendocument.spreadsheet",
    ];
    const SPREADSHEET_EXTENSIONS: &[&str] = &["xlsx", "xls", "ods"];
}

impl Parser for SpreadsheetParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::SPREADSHEET_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::SPREADSHEET_EXTENSIONS.contains(&ext) {
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
        let cursor = Cursor::new(content);
        let mut workbook =
            open_workbook_auto_from_rs(cursor).map_err(|e| sift_core::SiftError::Parse {
                path: "spreadsheet".to_string(),
                message: format!("Failed to open spreadsheet: {e}"),
            })?;

        let sheet_names: Vec<String> = workbook.sheet_names().clone();
        let mut metadata = HashMap::new();
        metadata.insert("sheet_count".to_string(), sheet_names.len().to_string());
        metadata.insert("size_bytes".to_string(), content.len().to_string());

        let mut output = String::with_capacity(content.len());

        for (sheet_idx, name) in sheet_names.iter().enumerate() {
            if sheet_idx > 0 {
                output.push_str("\n\n");
            }
            let _ = writeln!(output, "--- Sheet: {name} ---");

            if let Ok(range) = workbook.worksheet_range(name) {
                let mut rows = range.rows();

                // First row as headers
                let headers: Vec<String> = match rows.next() {
                    Some(row) => row.iter().map(cell_to_string).collect(),
                    None => continue,
                };

                // Remaining rows: write "header: value" directly to output
                for row in rows {
                    let mut first = true;
                    for (i, cell) in row.iter().enumerate() {
                        if is_cell_empty(cell) {
                            continue;
                        }
                        if !first {
                            output.push_str(", ");
                        }
                        first = false;
                        if i < headers.len() && !headers[i].is_empty() {
                            output.push_str(&headers[i]);
                            output.push_str(": ");
                        }
                        write_cell(&mut output, cell);
                    }
                    if !first {
                        output.push('\n');
                    }
                }
            }
        }

        let trimmed_len = output.trim_end().len();
        output.truncate(trimmed_len);

        Ok(ParsedDocument {
            text: output,
            title: None,
            language: None,
            content_type: ContentType::Data,
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "spreadsheet"
    }
}

fn is_cell_empty(cell: &Data) -> bool {
    match cell {
        Data::Empty => true,
        Data::String(s) => s.is_empty(),
        _ => false,
    }
}

/// Write a cell value directly to the output string, avoiding intermediate String allocation.
fn write_cell(output: &mut String, cell: &Data) {
    match cell {
        Data::Empty => {}
        Data::String(s) => output.push_str(s),
        Data::Int(i) => {
            let _ = write!(output, "{i}");
        }
        Data::Float(f) =>
        {
            #[allow(clippy::float_cmp)]
            if *f == f.floor() && f.abs() < 1e15 {
                let _ = write!(output, "{}", *f as i64);
            } else {
                let _ = write!(output, "{f}");
            }
        }
        Data::Bool(b) => {
            let _ = write!(output, "{b}");
        }
        Data::DateTime(dt) => {
            let _ = write!(output, "{dt}");
        }
        Data::DateTimeIso(s) => output.push_str(s),
        Data::DurationIso(s) => output.push_str(s),
        Data::Error(e) => {
            let _ = write!(output, "#ERR:{e:?}");
        }
    }
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(s) => s.clone(),
        Data::Int(i) => i.to_string(),
        Data::Float(f) =>
        {
            #[allow(clippy::float_cmp)]
            if *f == f.floor() && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => format!("{dt}"),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("#ERR:{e:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_spreadsheet() {
        let parser = SpreadsheetParser;
        assert!(parser.can_parse(
            Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            None
        ));
        assert!(parser.can_parse(None, Some("xlsx")));
        assert!(parser.can_parse(None, Some("xls")));
        assert!(parser.can_parse(None, Some("ods")));
        assert!(!parser.can_parse(None, Some("csv")));
        assert!(!parser.can_parse(None, Some("docx")));
    }

    #[test]
    fn test_invalid_spreadsheet_returns_error() {
        let parser = SpreadsheetParser;
        let result = parser.parse(b"not a spreadsheet", None, Some("xlsx"));
        assert!(result.is_err());
    }
}
