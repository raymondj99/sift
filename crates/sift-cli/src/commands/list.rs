use crate::{output, OutputFormat};
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, format: &OutputFormat) -> SiftResult<()> {
    #[cfg(feature = "sqlite")]
    let metadata_path = config.index_dir().join("metadata.db");
    #[cfg(not(feature = "sqlite"))]
    let metadata_path = config.index_dir().join("metadata.json");

    if !metadata_path.exists() {
        match format {
            OutputFormat::Json => println!("[]"),
            OutputFormat::Csv => println!("uri,file_type,chunks"),
            OutputFormat::Human => {
                println!("No index found. Run `vx scan <path>` to create one.");
            }
        }
        return Ok(());
    }

    let metadata = sift_store::MetadataStore::open(&metadata_path)?;
    let sources = metadata.list_sources()?;

    output::print_source_list(&sources, format);

    Ok(())
}
