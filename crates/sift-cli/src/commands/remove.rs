#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
use crate::OutputFormat;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::{Config, SiftResult};
use sift_store::FullTextStore;
#[cfg(feature = "hnsw")]
use sift_store::VectorIndex;

pub fn run(config: &Config, paths: &[String], format: &OutputFormat) -> SiftResult<()> {
    let (engine, metadata) = crate::pipeline::open_engine(config)?;

    let mut removed = 0u64;
    let mut not_found = 0u64;

    for path in paths {
        // Normalize path to URI format
        let uri = if path.starts_with("file://") {
            path.clone()
        } else {
            let abs =
                std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
            format!("file://{}", abs.display())
        };

        if metadata.remove_source(&uri)? {
            engine.delete_by_uri(&uri)?;
            removed += 1;
            if matches!(format, OutputFormat::Human) {
                println!("  {} {}", "Removed".red(), uri.dimmed());
            }
        } else {
            not_found += 1;
            if matches!(format, OutputFormat::Human) {
                println!("  {} {} (not indexed)", "Skipped".yellow(), path.dimmed());
            }
        }
    }

    // Persist stores to disk
    let vector_path = config.index_dir()?.join("vectors.bin");
    engine.vector_store.save(&vector_path)?;
    engine.fulltext_store.flush()?;

    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::json!({
                    "removed": removed,
                    "not_found": not_found,
                })
            );
        }
        OutputFormat::Csv => {
            println!("removed,not_found");
            println!("{removed},{not_found}");
        }
        OutputFormat::Human => {
            println!(
                "Removed {} source{}, {} not found",
                removed,
                if removed == 1 { "" } else { "s" },
                not_found
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{with_home, HOME_MUTEX};
    use crate::OutputFormat;
    use tempfile::TempDir;

    #[test]
    fn test_remove_not_found_human() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let paths = vec!["/nonexistent/file.txt".to_string()];
            let result = run(&config, &paths, &OutputFormat::Human);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn test_remove_already_uri_prefixed() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            // file:// prefix should be preserved as-is
            let paths = vec!["file:///some/path/that/was/not/indexed.txt".to_string()];
            let result = run(&config, &paths, &OutputFormat::Human);
            assert!(result.is_ok());
        });
    }
}
