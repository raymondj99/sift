#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::SiftResult;
use sift_embed::models::{ModelManager, NOMIC_EMBED_TEXT_V2};
use tracing::info;

pub fn run(action: Option<crate::ModelsAction>) -> SiftResult<()> {
    let manager = ModelManager::new();

    match action {
        None | Some(crate::ModelsAction::List) => {
            let downloaded = manager.list_downloaded();

            println!("{}", "Available Models:".bold());
            println!();

            // Show known models
            let known = [&NOMIC_EMBED_TEXT_V2];
            for model in known {
                let status = if downloaded.contains(&model.name.to_string()) {
                    "downloaded".green().to_string()
                } else {
                    "not downloaded".dimmed().to_string()
                };

                println!(
                    "  {} ({}d, {} tokens max) [{}]",
                    model.name.cyan(),
                    model.dimensions,
                    model.max_tokens,
                    status,
                );
            }
        }

        Some(crate::ModelsAction::Download { name }) => {
            let model_def = match name.as_str() {
                "nomic-embed-text-v2" => &NOMIC_EMBED_TEXT_V2,
                _ => {
                    return Err(sift_core::SiftError::Model(format!(
                        "Unknown model: '{}'. Available: nomic-embed-text-v2",
                        name
                    )));
                }
            };

            if manager.is_downloaded(&name) {
                info!("Model '{}' is already downloaded.", name);
                return Ok(());
            }

            // Ensure ONNX Runtime is available
            if !manager.is_ort_downloaded() {
                info!("Downloading ONNX Runtime...");
                manager.download_ort()?;
                info!("ONNX Runtime downloaded.");
            }

            info!("Downloading {}...", name);
            manager.download(model_def)?;
            info!("Model '{}' downloaded successfully.", name);
        }
    }

    Ok(())
}
