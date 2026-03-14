use crate::{output, OutputFormat};
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, format: &OutputFormat) -> SiftResult<()> {
    #[cfg(feature = "sqlite")]
    let metadata_path = config.index_dir()?.join("metadata.db");
    #[cfg(not(feature = "sqlite"))]
    let metadata_path = config.index_dir()?.join("metadata.json");

    if !metadata_path.exists() {
        match format {
            OutputFormat::Json => println!("[]"),
            OutputFormat::Csv => println!("uri,file_type,chunks"),
            OutputFormat::Human => {
                println!("No index found. Run `sift scan <path>` to create one.");
            }
        }
        return Ok(());
    }

    let metadata = sift_store::MetadataStore::open(&metadata_path)?;
    let sources = metadata.list_sources()?;

    output::print_source_list(&sources, format);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{with_home, HOME_MUTEX};
    use crate::OutputFormat;
    use tempfile::TempDir;

    // Early-exit path: no metadata file present.
    #[test]
    fn test_list_no_index_human() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            assert!(run(&config, &OutputFormat::Human).is_ok());
        });
    }
}
