use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::{Cursor, Read};

/// Parser for archive formats: ZIP, TAR, GZ, TGZ.
pub struct ArchiveParser;

const MAX_ENTRY_SIZE: u64 = 1024 * 1024; // 1 MB per entry
const MAX_TOTAL_SIZE: usize = 10 * 1024 * 1024; // 10 MB total

impl ArchiveParser {
    const ARCHIVE_MIMES: &[&str] = &["application/zip", "application/x-tar", "application/gzip"];
    const ARCHIVE_EXTENSIONS: &[&str] = &["zip", "tar", "gz", "tgz"];
}

impl Parser for ArchiveParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::ARCHIVE_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            if Self::ARCHIVE_EXTENSIONS.contains(&ext) {
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
        let (text, entry_count) = match ext {
            "tar" => extract_tar_entries(Cursor::new(content))?,
            "tgz" => parse_tar_gz(content)?,
            "gz" => parse_gz(content)?,
            _ => parse_zip(content)?,
        };

        let mut metadata = HashMap::new();
        metadata.insert("entry_count".to_string(), entry_count.to_string());
        metadata.insert("size_bytes".to_string(), content.len().to_string());

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Text,
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "archive"
    }
}

fn parse_zip(content: &[u8]) -> SiftResult<(String, usize)> {
    let cursor = Cursor::new(content);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| sift_core::SiftError::Parse {
        path: "zip".to_string(),
        message: format!("Failed to open ZIP: {e}"),
    })?;

    let mut output = String::new();
    let mut total_bytes = 0usize;
    let mut entry_count = 0;

    for i in 0..archive.len() {
        if total_bytes >= MAX_TOTAL_SIZE {
            output.push_str("\n--- truncated: total size limit reached ---\n");
            break;
        }

        let file = match archive.by_index(i) {
            Ok(f) => f,
            Err(_) => continue,
        };

        if file.is_dir() {
            continue;
        }

        let name = file.name().to_string();

        if file.size() > MAX_ENTRY_SIZE {
            let _ = writeln!(output, "--- file: {name} (skipped: too large) ---");
            entry_count += 1;
            continue;
        }

        let remaining = MAX_TOTAL_SIZE - total_bytes;
        let cap = std::cmp::min(file.size() as usize, remaining);
        let mut buf = Vec::with_capacity(cap);
        if file
            .take(std::cmp::min(remaining as u64, MAX_ENTRY_SIZE))
            .read_to_end(&mut buf)
            .is_err()
        {
            continue;
        }

        entry_count += 1;

        if let Ok(text) = std::str::from_utf8(&buf) {
            let _ = writeln!(output, "--- file: {name} ---");
            output.push_str(text);
            if !text.ends_with('\n') {
                output.push('\n');
            }
            total_bytes += buf.len();
        } else {
            let _ = writeln!(
                output,
                "--- file: {} (binary, {} bytes) ---",
                name,
                buf.len()
            );
        }
    }

    let trimmed_len = output.trim_end().len();
    output.truncate(trimmed_len);
    Ok((output, entry_count))
}

/// Stream tar.gz: pipe `GzDecoder` directly into `tar::Archive` (no full decompression buffer).
fn parse_tar_gz(content: &[u8]) -> SiftResult<(String, usize)> {
    let decoder = flate2::read::GzDecoder::new(Cursor::new(content));
    let limited = decoder.take(MAX_TOTAL_SIZE as u64);
    extract_tar_entries(limited)
}

fn extract_tar_entries<R: Read>(reader: R) -> SiftResult<(String, usize)> {
    let mut archive = tar::Archive::new(reader);
    let mut output = String::new();
    let mut total_bytes = 0usize;
    let mut entry_count = 0;

    let entries = archive.entries().map_err(|e| sift_core::SiftError::Parse {
        path: "tar".to_string(),
        message: format!("Failed to read TAR entries: {e}"),
    })?;

    for entry in entries {
        if total_bytes >= MAX_TOTAL_SIZE {
            output.push_str("\n--- truncated: total size limit reached ---\n");
            break;
        }

        let mut entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path().map_or_else(
            |_| "(unknown)".to_string(),
            |p| p.to_string_lossy().to_string(),
        );

        let size = entry.size();
        if size > MAX_ENTRY_SIZE {
            let _ = writeln!(output, "--- file: {path} (skipped: too large) ---");
            entry_count += 1;
            continue;
        }

        // Skip directories
        if entry.header().entry_type().is_dir() {
            continue;
        }

        let remaining = MAX_TOTAL_SIZE - total_bytes;
        let cap = std::cmp::min(size as usize, remaining);
        let mut buf = Vec::with_capacity(cap);
        if entry
            .by_ref()
            .take(std::cmp::min(remaining as u64, MAX_ENTRY_SIZE))
            .read_to_end(&mut buf)
            .is_err()
        {
            continue;
        }

        entry_count += 1;

        if let Ok(text) = std::str::from_utf8(&buf) {
            let _ = writeln!(output, "--- file: {path} ---");
            output.push_str(text);
            if !text.ends_with('\n') {
                output.push('\n');
            }
            total_bytes += buf.len();
        } else {
            let _ = writeln!(
                output,
                "--- file: {} (binary, {} bytes) ---",
                path,
                buf.len()
            );
        }
    }

    let trimmed_len = output.trim_end().len();
    output.truncate(trimmed_len);
    Ok((output, entry_count))
}

fn parse_gz(content: &[u8]) -> SiftResult<(String, usize)> {
    let decoder = flate2::read::GzDecoder::new(Cursor::new(content));
    let mut decompressed = Vec::new();
    decoder
        .take(MAX_TOTAL_SIZE as u64)
        .read_to_end(&mut decompressed)
        .map_err(|e| sift_core::SiftError::Parse {
            path: "gz".to_string(),
            message: format!("Failed to decompress gzip: {e}"),
        })?;

    match String::from_utf8(decompressed) {
        Ok(text) => Ok((text, 1)),
        Err(_) => Err(sift_core::SiftError::Parse {
            path: "gz".to_string(),
            message: "Decompressed content is not valid UTF-8 text".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tar() {
        let mut builder = tar::Builder::new(Vec::new());

        let content = b"fn main() { println!(\"hello\"); }";
        let mut header = tar::Header::new_gnu();
        header.set_path("src/main.rs").unwrap();
        header.set_size(content.len() as u64);
        header.set_cksum();
        builder.append(&header, &content[..]).unwrap();

        let tar_data = builder.into_inner().unwrap();
        let parser = ArchiveParser;
        let doc = parser.parse(&tar_data, None, Some("tar")).unwrap();
        assert!(doc.text.contains("--- file: src/main.rs ---"));
        assert!(doc.text.contains("fn main()"));
    }

    #[test]
    fn test_parse_zip() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        writer.start_file("hello.txt", options).unwrap();
        writer.write_all(b"Hello from zip!").unwrap();

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("--- file: hello.txt ---"));
        assert!(doc.text.contains("Hello from zip!"));
    }
}
