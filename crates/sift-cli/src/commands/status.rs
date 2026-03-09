use crate::{output, OutputFormat};
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, format: &OutputFormat) -> SiftResult<()> {
    let index_dir = config.index_dir();
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

    output::print_index_stats(&stats, format);

    Ok(())
}

fn dir_size(path: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}
