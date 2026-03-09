use std::io::Write;
use std::path::PathBuf;
use tracing::info;

use sift_core::{Config, SiftResult};
#[cfg(feature = "hnsw")]
use sift_store::VectorIndex;

use crate::pipeline;

pub fn run(
    config: &Config,
    include_vectors: bool,
    output_path: Option<PathBuf>,
    file_type_filter: Option<String>,
) -> SiftResult<()> {
    let (engine, _metadata) = pipeline::open_engine(config)?;

    let entries = engine.vector_store.export_all()?;

    let mut writer: Box<dyn Write> = if let Some(ref path) = output_path {
        Box::new(std::fs::File::create(path).map_err(|e| {
            sift_core::SiftError::Storage(format!("Failed to create output file: {}", e))
        })?)
    } else {
        Box::new(std::io::stdout().lock())
    };

    let mut count = 0u64;
    for entry in &entries {
        // Apply file type filter
        if let Some(ref ft) = file_type_filter {
            if entry.file_type != *ft {
                continue;
            }
        }

        let mut obj = serde_json::json!({
            "uri": entry.uri,
            "text": entry.text,
            "chunk_index": entry.chunk_index,
            "content_type": entry.content_type,
            "file_type": entry.file_type,
        });

        if let Some(ref title) = entry.title {
            obj["title"] = serde_json::Value::String(title.clone());
        }

        if let Some((start, end)) = entry.byte_range {
            obj["byte_range"] = serde_json::json!([start, end]);
        }

        if include_vectors {
            obj["vector"] = serde_json::json!(entry.vector);
        }

        let line = serde_json::to_string(&obj)
            .map_err(|e| sift_core::SiftError::Storage(format!("JSON serialize error: {}", e)))?;
        writeln!(writer, "{}", line)
            .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
        count += 1;
    }

    if output_path.is_some() {
        info!(count, "Exported chunks");
    }

    Ok(())
}
