use crate::traits::Source;
use sift_core::{ScanOptions, SiftResult, SourceItem};
use std::path::Path;
use tracing::debug;

pub struct FilesystemSource;

impl FilesystemSource {
    pub fn new() -> Self {
        Self
    }

    /// Hash a file's contents with BLAKE3.
    #[cfg(test)]
    pub(crate) fn hash_file(path: &Path) -> SiftResult<[u8; 32]> {
        let data = std::fs::read(path)?;
        Ok(*blake3::hash(&data).as_bytes())
    }

    /// MIME type from file extension alone (no I/O).
    pub(crate) fn mime_from_extension(path: &Path) -> Option<String> {
        let ext = path.extension()?.to_str()?;
        Some(
            match ext {
                "txt" => "text/plain",
                "md" | "markdown" => "text/markdown",
                "rs" => "text/x-rust",
                "py" => "text/x-python",
                "js" => "text/javascript",
                "ts" => "text/typescript",
                "jsx" | "tsx" => "text/javascript",
                "go" => "text/x-go",
                "c" | "h" => "text/x-c",
                "cpp" | "hpp" | "cc" | "cxx" => "text/x-c++",
                "java" => "text/x-java",
                "rb" => "text/x-ruby",
                "sh" | "bash" | "zsh" | "fish" => "text/x-shellscript",
                "html" | "htm" => "text/html",
                "xml" => "text/xml",
                "css" => "text/css",
                "json" => "application/json",
                "jsonl" => "application/jsonlines",
                "toml" => "application/toml",
                "yaml" | "yml" => "application/yaml",
                "csv" => "text/csv",
                "pdf" => "application/pdf",
                "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "pptx" => {
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                }
                "eml" => "message/rfc822",
                "mbox" => "application/mbox",
                "xls" => "application/vnd.ms-excel",
                "ods" => "application/vnd.oasis.opendocument.spreadsheet",
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "svg" => "image/svg+xml",
                "mp3" => "audio/mpeg",
                "wav" => "audio/wav",
                "flac" => "audio/flac",
                "zip" => "application/zip",
                "tar" => "application/x-tar",
                "gz" | "tgz" => "application/gzip",
                "rst" => "text/x-rst",
                "org" => "text/x-org",
                "cfg" | "ini" | "conf" => "text/plain",
                "log" => "text/plain",
                "sql" => "text/x-sql",
                "r" => "text/x-r",
                "swift" => "text/x-swift",
                "kt" | "kts" => "text/x-kotlin",
                "scala" => "text/x-scala",
                "zig" => "text/x-zig",
                "lua" => "text/x-lua",
                "pl" | "pm" => "text/x-perl",
                "ex" | "exs" => "text/x-elixir",
                "erl" | "hrl" => "text/x-erlang",
                "hs" => "text/x-haskell",
                "ml" | "mli" => "text/x-ocaml",
                "proto" => "text/x-protobuf",
                "tf" | "tfvars" => "text/x-terraform",
                "dockerfile" => "text/x-dockerfile",
                _ => return None,
            }
            .to_string(),
        )
    }

    /// Detect MIME type using both content-based detection and extension fallback.
    #[cfg(test)]
    fn detect_mime(path: &Path) -> Option<String> {
        // Try content-based detection first
        if let Ok(data) = std::fs::read(path) {
            if let Some(kind) = infer::get(&data) {
                return Some(kind.mime_type().to_string());
            }
        }
        // Fall back to extension
        Self::mime_from_extension(path)
    }

    /// Read a file once, returning (content_hash, content_mime).
    /// Uses the same buffer for both MIME detection (via `infer`) and BLAKE3 hashing.
    pub(crate) fn read_and_analyze(path: &Path) -> SiftResult<([u8; 32], Option<String>)> {
        let data = std::fs::read(path)?;
        let mime = infer::get(&data).map(|kind| kind.mime_type().to_string());
        let hash = *blake3::hash(&data).as_bytes();
        Ok((hash, mime))
    }
}

impl Default for FilesystemSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for FilesystemSource {
    fn discover(
        &self,
        options: &ScanOptions,
        callback: &mut dyn FnMut(SourceItem) -> SiftResult<()>,
    ) -> SiftResult<u64> {
        let mut count = 0u64;

        for scan_path in &options.paths {
            let mut builder = ignore::WalkBuilder::new(scan_path);
            builder
                .hidden(true) // skip hidden files by default
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true);

            if let Some(max_depth) = options.max_depth {
                builder.max_depth(Some(max_depth));
            }

            // Add custom ignore patterns
            let mut overrides = ignore::overrides::OverrideBuilder::new(scan_path);
            for pattern in &options.exclude_globs {
                let _ = overrides.add(&format!("!{}", pattern));
            }
            for pattern in &options.include_globs {
                let _ = overrides.add(pattern);
            }
            if let Ok(ov) = overrides.build() {
                builder.overrides(ov);
            }

            for entry in builder.build() {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        debug!("Walk error: {}", e);
                        continue;
                    }
                };

                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                // Check file size
                let metadata = match path.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let size = metadata.len();
                if let Some(max_size) = options.max_file_size {
                    if size > max_size {
                        debug!("Skipping {} (too large: {} bytes)", path.display(), size);
                        continue;
                    }
                }

                let extension = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.to_lowercase());

                // Filter by file type if specified (check extension BEFORE reading the file)
                if !options.file_types.is_empty() {
                    let ext_matches = options
                        .file_types
                        .iter()
                        .any(|ft| extension.as_deref() == Some(ft.as_str()));
                    if !ext_matches {
                        // Also check the extension-based MIME for type matching
                        let ext_mime = Self::mime_from_extension(path);
                        let mime_matches = options
                            .file_types
                            .iter()
                            .any(|ft| ext_mime.as_deref().is_some_and(|m| m.contains(ft.as_str())));
                        if !mime_matches {
                            continue;
                        }
                    }
                }

                // Single-pass: read file once for both MIME detection and hashing
                let (content_hash, content_mime) = match Self::read_and_analyze(path) {
                    Ok(result) => result,
                    Err(e) => {
                        debug!("Read error for {}: {}", path.display(), e);
                        continue;
                    }
                };

                // Determine MIME type: prefer content-based, fall back to extension
                let mime_type = content_mime.or_else(|| Self::mime_from_extension(path));

                // Skip files we can't identify unless they look like text
                if mime_type.is_none() && extension.is_none() {
                    continue;
                }

                let modified_at = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64);

                let item = SourceItem {
                    uri: format!(
                        "file://{}",
                        path.canonicalize().unwrap_or(path.to_path_buf()).display()
                    ),
                    path: path.to_path_buf(),
                    content_hash,
                    size,
                    modified_at,
                    mime_type,
                    extension,
                };

                callback(item)?;
                count += 1;
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_discover_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("hello.txt"), "hello world").unwrap();
        fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/nested.md"), "# Title").unwrap();

        let source = FilesystemSource::new();
        let options = ScanOptions {
            paths: vec![dir.path().to_path_buf()],
            ..Default::default()
        };

        let mut items = vec![];
        let count = source
            .discover(&options, &mut |item| {
                items.push(item);
                Ok(())
            })
            .unwrap();

        assert_eq!(count, 3);
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_max_file_size_filter() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("small.txt"), "hi").unwrap();
        fs::write(dir.path().join("big.txt"), "x".repeat(1000)).unwrap();

        let source = FilesystemSource::new();
        let options = ScanOptions {
            paths: vec![dir.path().to_path_buf()],
            max_file_size: Some(100),
            ..Default::default()
        };

        let mut items = vec![];
        source
            .discover(&options, &mut |item| {
                items.push(item);
                Ok(())
            })
            .unwrap();

        assert_eq!(items.len(), 1);
        assert!(items[0].uri.contains("small.txt"));
    }

    #[test]
    fn test_content_hash_deterministic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "deterministic content").unwrap();

        let h1 = FilesystemSource::hash_file(&path).unwrap();
        let h2 = FilesystemSource::hash_file(&path).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_mime_detection() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        fs::write(&path, "fn main() {}").unwrap();

        let mime = FilesystemSource::detect_mime(&path);
        assert_eq!(mime, Some("text/x-rust".to_string()));
    }

    #[test]
    fn test_read_and_analyze_returns_hash_and_mime() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();

        let (hash, _mime) = FilesystemSource::read_and_analyze(&path).unwrap();
        let expected_hash = *blake3::hash(b"hello world").as_bytes();
        assert_eq!(hash, expected_hash);
    }

    #[test]
    fn test_single_pass_matches_separate_hash() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("code.rs");
        fs::write(&path, "fn main() { println!(\"hello\"); }").unwrap();

        let separate_hash = FilesystemSource::hash_file(&path).unwrap();
        let (combined_hash, _) = FilesystemSource::read_and_analyze(&path).unwrap();
        assert_eq!(separate_hash, combined_hash);
    }

    #[test]
    fn test_mime_from_extension() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        fs::write(&path, "fn main() {}").unwrap();

        let mime = FilesystemSource::mime_from_extension(&path);
        assert_eq!(mime, Some("text/x-rust".to_string()));
    }

    #[test]
    fn test_mime_from_extension_unknown() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.xyz123");
        fs::write(&path, "unknown").unwrap();

        let mime = FilesystemSource::mime_from_extension(&path);
        assert!(mime.is_none());
    }
}
