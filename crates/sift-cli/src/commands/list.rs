use crate::{output, OutputFormat};
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, format: &OutputFormat) -> SiftResult<()> {
    // Try daemon first
    #[cfg(unix)]
    if let Some(sources) = try_daemon_list() {
        output::print_source_list(&sources, format);
        return Ok(());
    }

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

/// Try to fetch source list from the daemon API.
#[cfg(unix)]
fn try_daemon_list() -> Option<Vec<(String, String, u32)>> {
    let body = crate::daemon_client::get("/api/list")?;
    let items: Vec<serde_json::Value> = serde_json::from_str(&body).ok()?;
    let sources = items
        .iter()
        .filter_map(|item| {
            Some((
                item.get("uri")?.as_str()?.to_string(),
                item.get("file_type")?.as_str()?.to_string(),
                item.get("chunks")?.as_u64()? as u32,
            ))
        })
        .collect();
    Some(sources)
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
