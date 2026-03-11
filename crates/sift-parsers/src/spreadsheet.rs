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

    #[test]
    fn test_is_cell_empty_variants() {
        // Empty -> true
        assert!(is_cell_empty(&Data::Empty));
        // Empty string -> true
        assert!(is_cell_empty(&Data::String(String::new())));
        // Non-empty string -> false
        assert!(!is_cell_empty(&Data::String("hello".to_string())));
        // Int -> false
        assert!(!is_cell_empty(&Data::Int(42)));
        // Float -> false
        assert!(!is_cell_empty(&Data::Float(1.23)));
        // Bool -> false
        assert!(!is_cell_empty(&Data::Bool(true)));
        assert!(!is_cell_empty(&Data::Bool(false)));
    }

    #[test]
    fn test_cell_to_string_all_variants() {
        // Empty
        assert_eq!(cell_to_string(&Data::Empty), "");
        // String
        assert_eq!(cell_to_string(&Data::String("hello".to_string())), "hello");
        // Int
        assert_eq!(cell_to_string(&Data::Int(42)), "42");
        assert_eq!(cell_to_string(&Data::Int(-7)), "-7");
        // Float (integer-like)
        assert_eq!(cell_to_string(&Data::Float(5.0)), "5");
        // Float (decimal)
        assert_eq!(cell_to_string(&Data::Float(1.23)), "1.23");
        // Bool
        assert_eq!(cell_to_string(&Data::Bool(true)), "true");
        assert_eq!(cell_to_string(&Data::Bool(false)), "false");
        // DateTimeIso
        assert_eq!(
            cell_to_string(&Data::DateTimeIso("2024-01-15T10:30:00".to_string())),
            "2024-01-15T10:30:00"
        );
        // DurationIso
        assert_eq!(
            cell_to_string(&Data::DurationIso("PT1H30M".to_string())),
            "PT1H30M"
        );
        // Error
        let err_str = cell_to_string(&Data::Error(calamine::CellErrorType::Null));
        assert!(err_str.starts_with("#ERR:"));
    }

    #[test]
    fn test_write_cell_all_variants() {
        // Empty - should write nothing
        let mut out = String::new();
        write_cell(&mut out, &Data::Empty);
        assert_eq!(out, "");

        // String
        let mut out = String::new();
        write_cell(&mut out, &Data::String("test".to_string()));
        assert_eq!(out, "test");

        // Int
        let mut out = String::new();
        write_cell(&mut out, &Data::Int(99));
        assert_eq!(out, "99");

        // Float integer-like
        let mut out = String::new();
        write_cell(&mut out, &Data::Float(7.0));
        assert_eq!(out, "7");

        // Float decimal
        let mut out = String::new();
        write_cell(&mut out, &Data::Float(1.23));
        assert_eq!(out, "1.23");

        // Bool
        let mut out = String::new();
        write_cell(&mut out, &Data::Bool(true));
        assert_eq!(out, "true");

        let mut out = String::new();
        write_cell(&mut out, &Data::Bool(false));
        assert_eq!(out, "false");

        // DateTimeIso
        let mut out = String::new();
        write_cell(&mut out, &Data::DateTimeIso("2024-01-01".to_string()));
        assert_eq!(out, "2024-01-01");

        // DurationIso
        let mut out = String::new();
        write_cell(&mut out, &Data::DurationIso("PT2H".to_string()));
        assert_eq!(out, "PT2H");

        // Error
        let mut out = String::new();
        write_cell(&mut out, &Data::Error(calamine::CellErrorType::Div0));
        assert!(out.starts_with("#ERR:"));
    }

    #[test]
    fn test_parser_name() {
        let parser = SpreadsheetParser;
        assert_eq!(parser.name(), "spreadsheet");
    }

    #[test]
    fn test_can_parse_all_mimes() {
        let parser = SpreadsheetParser;
        assert!(parser.can_parse(
            Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            None
        ));
        assert!(parser.can_parse(Some("application/vnd.ms-excel"), None));
        assert!(parser.can_parse(Some("application/vnd.oasis.opendocument.spreadsheet"), None));
    }

    #[test]
    fn test_can_parse_all_extensions() {
        let parser = SpreadsheetParser;
        assert!(parser.can_parse(None, Some("xlsx")));
        assert!(parser.can_parse(None, Some("xls")));
        assert!(parser.can_parse(None, Some("ods")));
    }

    #[test]
    fn test_cannot_parse_non_spreadsheet() {
        let parser = SpreadsheetParser;
        assert!(!parser.can_parse(None, Some("pdf")));
        assert!(!parser.can_parse(Some("text/plain"), None));
        assert!(!parser.can_parse(None, None));
    }

    /// Build a minimal valid XLSX in-memory for testing.
    /// XLSX is a ZIP with specific XML files that calamine can read.
    fn make_xlsx(rows: &[Vec<&str>]) -> Vec<u8> {
        use std::io::Write;

        let buf = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        let options = zip::write::SimpleFileOptions::default();

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"#).unwrap();

        // _rels/.rels
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"#).unwrap();

        // xl/workbook.xml
        zip.start_file("xl/workbook.xml", options).unwrap();
        zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#).unwrap();

        // Collect unique strings for shared strings table
        let mut all_strings: Vec<String> = Vec::new();
        for row in rows {
            for cell in row {
                if !all_strings.contains(&cell.to_string()) {
                    all_strings.push(cell.to_string());
                }
            }
        }

        // xl/sharedStrings.xml
        zip.start_file("xl/sharedStrings.xml", options).unwrap();
        let mut ss = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{}" uniqueCount="{}">"#,
            all_strings.len(),
            all_strings.len()
        );
        for s in &all_strings {
            let _ = write!(ss, "<si><t>{s}</t></si>");
        }
        ss.push_str("</sst>");
        zip.write_all(ss.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
        let mut sheet = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>"#,
        );
        for (row_idx, row) in rows.iter().enumerate() {
            let row_num = row_idx + 1;
            let _ = write!(sheet, "<row r=\"{row_num}\">");
            for (col_idx, cell) in row.iter().enumerate() {
                let col_letter = (b'A' + col_idx as u8) as char;
                let cell_ref = format!("{col_letter}{row_num}");
                let str_idx = all_strings.iter().position(|s| s == *cell).unwrap();
                let _ = write!(sheet, "<c r=\"{cell_ref}\" t=\"s\"><v>{str_idx}</v></c>");
            }
            sheet.push_str("</row>");
        }
        sheet.push_str("</sheetData></worksheet>");
        zip.write_all(sheet.as_bytes()).unwrap();

        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn test_parse_valid_xlsx() {
        let parser = SpreadsheetParser;
        let xlsx = make_xlsx(&[
            vec!["Name", "Age", "City"],
            vec!["Alice", "30", "NYC"],
            vec!["Bob", "25", "LA"],
        ]);
        let doc = parser.parse(&xlsx, None, Some("xlsx")).unwrap();
        assert!(doc.text.contains("Sheet1"));
        assert!(doc.text.contains("Name: Alice"));
        assert!(doc.text.contains("Age: 30"));
        assert!(doc.text.contains("City: NYC"));
        assert!(doc.text.contains("Name: Bob"));
        assert_eq!(doc.content_type, ContentType::Data);
        assert_eq!(doc.metadata.get("sheet_count").unwrap(), "1");
    }

    #[test]
    fn test_parse_xlsx_single_row_headers_only() {
        let parser = SpreadsheetParser;
        let xlsx = make_xlsx(&[vec!["Col1", "Col2"]]);
        let doc = parser.parse(&xlsx, None, Some("xlsx")).unwrap();
        // Only headers, no data rows - should still parse without error
        assert!(doc.text.contains("Sheet1"));
    }

    #[test]
    fn test_parse_xlsx_metadata_has_size_bytes() {
        let parser = SpreadsheetParser;
        let xlsx = make_xlsx(&[vec!["A"], vec!["B"]]);
        let doc = parser.parse(&xlsx, None, Some("xlsx")).unwrap();
        assert!(doc.metadata.contains_key("size_bytes"));
        let size: usize = doc.metadata.get("size_bytes").unwrap().parse().unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_cell_to_string_large_float() {
        // Float >= 1e15 takes the else branch (format!("{f}"))
        // rather than the integer cast branch
        let result = cell_to_string(&Data::Float(1e16));
        assert_eq!(result, "10000000000000000");

        // Float with decimal portion takes else branch regardless of magnitude
        let result2 = cell_to_string(&Data::Float(2.71));
        assert_eq!(result2, "2.71");
    }

    #[test]
    fn test_write_cell_large_float() {
        let mut out = String::new();
        write_cell(&mut out, &Data::Float(1e16));
        assert!(!out.is_empty());
    }

    #[test]
    fn test_write_cell_datetime_iso() {
        let mut out = String::new();
        write_cell(&mut out, &Data::DateTimeIso("2024-06-15".to_string()));
        assert_eq!(out, "2024-06-15");
    }

    #[test]
    fn test_write_cell_duration_iso() {
        let mut out = String::new();
        write_cell(&mut out, &Data::DurationIso("PT1H30M".to_string()));
        assert_eq!(out, "PT1H30M");
    }
}
