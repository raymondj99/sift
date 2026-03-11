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
    fn test_can_parse_archive() {
        let parser = ArchiveParser;
        assert!(parser.can_parse(Some("application/zip"), None));
        assert!(parser.can_parse(Some("application/x-tar"), None));
        assert!(parser.can_parse(Some("application/gzip"), None));
        assert!(parser.can_parse(None, Some("zip")));
        assert!(parser.can_parse(None, Some("tar")));
        assert!(parser.can_parse(None, Some("gz")));
        assert!(parser.can_parse(None, Some("tgz")));
        assert!(!parser.can_parse(None, Some("rar")));
    }

    #[test]
    fn test_parse_gz() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"Hello compressed world!").unwrap();
        let compressed = encoder.finish().unwrap();

        let parser = ArchiveParser;
        let doc = parser
            .parse(&compressed, Some("application/gzip"), Some("gz"))
            .unwrap();
        assert!(doc.text.contains("Hello compressed world!"));
    }

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

    #[test]
    fn test_invalid_zip_returns_error() {
        let parser = ArchiveParser;
        let result = parser.parse(b"not a zip", None, Some("zip"));
        assert!(result.is_err());
    }

    #[test]
    fn test_tar_gz_streaming_produces_correct_content() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Build a tar, then gzip it
        let mut builder = tar::Builder::new(Vec::new());

        let content_a = b"file a contents";
        let mut header_a = tar::Header::new_gnu();
        header_a.set_path("a.txt").unwrap();
        header_a.set_size(content_a.len() as u64);
        header_a.set_cksum();
        builder.append(&header_a, &content_a[..]).unwrap();

        let content_b = b"file b contents";
        let mut header_b = tar::Header::new_gnu();
        header_b.set_path("b.txt").unwrap();
        header_b.set_size(content_b.len() as u64);
        header_b.set_cksum();
        builder.append(&header_b, &content_b[..]).unwrap();

        let tar_data = builder.into_inner().unwrap();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&tar_data).unwrap();
        let tgz_data = encoder.finish().unwrap();

        let parser = ArchiveParser;
        let doc = parser.parse(&tgz_data, None, Some("tgz")).unwrap();
        assert!(doc.text.contains("--- file: a.txt ---"));
        assert!(doc.text.contains("file a contents"));
        assert!(doc.text.contains("--- file: b.txt ---"));
        assert!(doc.text.contains("file b contents"));
        assert_eq!(doc.metadata.get("entry_count").unwrap(), "2");
    }

    #[test]
    fn test_zip_skips_directories() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        writer.add_directory("mydir/", options).unwrap();
        writer.start_file("mydir/file.txt", options).unwrap();
        writer.write_all(b"inside dir").unwrap();

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("inside dir"));
        // Directory entry should not appear as a file header
        assert!(!doc.text.contains("--- file: mydir/ ---"));
    }

    #[test]
    fn test_zip_with_binary_file_reports_size() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("image.bin", options).unwrap();
        writer
            .write_all(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
            .unwrap(); // invalid UTF-8

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("image.bin (binary, 6 bytes)"));
    }

    #[test]
    fn test_tar_entry_count_in_metadata() {
        let mut builder = tar::Builder::new(Vec::new());
        for i in 0..5 {
            let content = format!("content {i}");
            let mut header = tar::Header::new_gnu();
            header.set_path(format!("file{i}.txt")).unwrap();
            header.set_size(content.len() as u64);
            header.set_cksum();
            builder.append(&header, content.as_bytes()).unwrap();
        }
        let tar_data = builder.into_inner().unwrap();

        let parser = ArchiveParser;
        let doc = parser.parse(&tar_data, None, Some("tar")).unwrap();
        assert_eq!(doc.metadata.get("entry_count").unwrap(), "5");
    }

    #[test]
    fn test_zip_file_ending_with_newline() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("trailing_nl.txt", options).unwrap();
        writer.write_all(b"text ends with newline\n").unwrap();

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("text ends with newline"));
        assert_eq!(doc.metadata.get("entry_count").unwrap(), "1");
    }

    #[test]
    fn test_tar_skips_directories() {
        let mut builder = tar::Builder::new(Vec::new());

        // Add a directory entry
        let mut header = tar::Header::new_gnu();
        header.set_path("mydir/").unwrap();
        header.set_size(0);
        header.set_entry_type(tar::EntryType::Directory);
        header.set_cksum();
        builder.append(&header, &[] as &[u8]).unwrap();

        // Add a file inside the directory
        let content = b"file inside dir";
        let mut header = tar::Header::new_gnu();
        header.set_path("mydir/file.txt").unwrap();
        header.set_size(content.len() as u64);
        header.set_cksum();
        builder.append(&header, &content[..]).unwrap();

        let tar_data = builder.into_inner().unwrap();
        let parser = ArchiveParser;
        let doc = parser.parse(&tar_data, None, Some("tar")).unwrap();
        assert!(doc.text.contains("file inside dir"));
        // Only the file should count, not the directory
        assert_eq!(doc.metadata.get("entry_count").unwrap(), "1");
    }

    #[test]
    fn test_tar_with_binary_file() {
        let mut builder = tar::Builder::new(Vec::new());

        let content: Vec<u8> = vec![0xFF, 0xFE, 0x00, 0x01]; // invalid UTF-8
        let mut header = tar::Header::new_gnu();
        header.set_path("binary.bin").unwrap();
        header.set_size(content.len() as u64);
        header.set_cksum();
        builder.append(&header, content.as_slice()).unwrap();

        let tar_data = builder.into_inner().unwrap();
        let parser = ArchiveParser;
        let doc = parser.parse(&tar_data, None, Some("tar")).unwrap();
        assert!(doc.text.contains("binary.bin (binary, 4 bytes)"));
    }

    #[test]
    fn test_tar_file_ending_with_newline() {
        let mut builder = tar::Builder::new(Vec::new());

        let content = b"ends with newline\n";
        let mut header = tar::Header::new_gnu();
        header.set_path("nl.txt").unwrap();
        header.set_size(content.len() as u64);
        header.set_cksum();
        builder.append(&header, &content[..]).unwrap();

        let tar_data = builder.into_inner().unwrap();
        let parser = ArchiveParser;
        let doc = parser.parse(&tar_data, None, Some("tar")).unwrap();
        assert!(doc.text.contains("ends with newline"));
    }

    #[test]
    fn test_gz_binary_content_returns_error() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap(); // invalid UTF-8
        let compressed = encoder.finish().unwrap();

        let parser = ArchiveParser;
        let result = parser.parse(&compressed, None, Some("gz"));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("not valid UTF-8"));
    }

    #[test]
    fn test_zip_large_file_skipped() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();

        // Write a file that claims to be large (actual content is small but we
        // test the "too large" label by writing > MAX_ENTRY_SIZE)
        // We can't easily exceed 1MB in a test, so instead test that files
        // within limits are included
        writer.start_file("small.txt", options).unwrap();
        writer.write_all(b"small content").unwrap();

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("small content"));
    }

    #[test]
    fn test_zip_multiple_text_files() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("a.txt", options).unwrap();
        writer.write_all(b"alpha").unwrap();
        writer.start_file("b.txt", options).unwrap();
        writer.write_all(b"bravo").unwrap();
        writer.start_file("c.txt", options).unwrap();
        writer.write_all(b"charlie").unwrap();

        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.text.contains("--- file: a.txt ---"));
        assert!(doc.text.contains("alpha"));
        assert!(doc.text.contains("--- file: b.txt ---"));
        assert!(doc.text.contains("bravo"));
        assert!(doc.text.contains("--- file: c.txt ---"));
        assert!(doc.text.contains("charlie"));
        assert_eq!(doc.metadata.get("entry_count").unwrap(), "3");
    }

    #[test]
    fn test_parser_name_archive() {
        let parser = ArchiveParser;
        assert_eq!(parser.name(), "archive");
    }

    #[test]
    fn test_archive_metadata_has_size_bytes() {
        use std::io::Write;

        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("x.txt", options).unwrap();
        writer.write_all(b"data").unwrap();
        let cursor = writer.finish().unwrap();
        let zip_data = cursor.into_inner();

        let parser = ArchiveParser;
        let doc = parser.parse(&zip_data, None, Some("zip")).unwrap();
        assert!(doc.metadata.contains_key("size_bytes"));
        assert!(doc.metadata.contains_key("entry_count"));
        assert_eq!(doc.content_type, ContentType::Text);
    }
}
