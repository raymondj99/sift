#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::SiftResult;
use sift_embed::models::{all_models, downloadable_text_models, get_model, ModelManager};
use tracing::info;

pub fn run(action: Option<crate::ModelsAction>) -> SiftResult<()> {
    let manager = ModelManager::new()?;

    match action {
        None | Some(crate::ModelsAction::List) => {
            println!("{}", "Available Models:".bold());
            println!();

            for model in all_models() {
                let status = if manager.is_downloaded_model(model) {
                    "downloaded".green().to_string()
                } else if model.is_download_supported() {
                    "not downloaded".dimmed().to_string()
                } else {
                    format!("requires {}", model.runtime.as_str())
                        .yellow()
                        .to_string()
                };

                println!(
                    "  {} ({}d, {} tokens max, {}, {}) [{}]",
                    model.name.cyan(),
                    model.dimensions,
                    model.max_tokens,
                    model.runtime.as_str(),
                    model.license.as_str(),
                    status,
                );
                if !model.aliases.is_empty() {
                    println!("      aliases: {}", model.aliases.join(", ").dimmed());
                }
                println!("      {}", model.notes.dimmed());
            }
        }

        Some(crate::ModelsAction::Download { name }) => {
            let Some(model_def) = get_model(&name) else {
                let available = downloadable_text_models()
                    .map(|m| m.name)
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(sift_core::SiftError::Model(format!(
                    "Unknown model: '{name}'. Downloadable models: {available}"
                )));
            };

            if !model_def.is_download_supported() {
                let available = downloadable_text_models()
                    .map(|m| m.name)
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(sift_core::SiftError::Model(format!(
                    "Model '{}' cannot be downloaded by sift's native ONNX downloader yet (runtime: {}). {}\nDownloadable models: {}",
                    model_def.name,
                    model_def.runtime.as_str(),
                    model_def.notes,
                    available
                )));
            }

            if manager.is_downloaded_model(model_def) {
                info!("Model '{}' is already downloaded.", model_def.name);
                return Ok(());
            }

            // Ensure ONNX Runtime is available
            if !manager.is_ort_downloaded() {
                info!("Downloading ONNX Runtime...");
                manager.download_ort()?;
                info!("ONNX Runtime downloaded.");
            }

            info!("Downloading {}...", model_def.name);
            manager.download(model_def)?;
            info!("Model '{}' downloaded successfully.", model_def.name);
        }
    }

    Ok(())
}
