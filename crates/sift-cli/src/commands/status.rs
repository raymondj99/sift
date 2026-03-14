use crate::{output, OutputFormat};
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, format: &OutputFormat) -> SiftResult<()> {
    let index_dir = config.index_dir()?;
    #[cfg(feature = "sqlite")]
    let metadata_path = index_dir.join("metadata.db");
    #[cfg(not(feature = "sqlite"))]
    let metadata_path = index_dir.join("metadata.json");

    if !metadata_path.exists() {
        match format {
            OutputFormat::Json => {
                println!(
                    r#"{{"status":"no_index","message":"No index found. Run `sift scan` first."}}"#
                );
            }
            OutputFormat::Csv => {
                println!("status,message");
                println!("no_index,\"No index found. Run `sift scan` first.\"");
            }
            OutputFormat::Human => {
                println!("No index found. Run `sift scan <path>` to create one.");
            }
        }
        return Ok(());
    }

    let metadata = sift_store::MetadataStore::open(&metadata_path)?;
    let mut stats = metadata.stats()?;

    // Calculate index size on disk
    if index_dir.exists() {
        stats.index_size_bytes = dir_size(&index_dir);
    }

    output::format_stats(&stats, format);

    Ok(())
}

fn dir_size(path: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{with_home, HOME_MUTEX};
    use crate::OutputFormat;
    use tempfile::TempDir;

    // When no metadata file exists, all three formats print a "no index" message and return Ok.
    #[test]
    fn test_status_no_index_human() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            assert!(run(&config, &OutputFormat::Human).is_ok());
        });
    }

    #[test]
    fn test_dir_size_with_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "world!!").unwrap();
        let size = dir_size(tmp.path());
        assert_eq!(size, 12, "total bytes should be 5 + 7 = 12");
    }
}
