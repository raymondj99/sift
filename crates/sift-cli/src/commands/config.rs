#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::{Config, SiftResult};

pub fn run(config: &Config, key: Option<String>, value: Option<String>) -> SiftResult<()> {
    match (key, value) {
        (None, None) => {
            // Print full config
            let toml_str = toml::to_string_pretty(config)
                .map_err(|e| sift_core::SiftError::Config(e.to_string()))?;
            println!("{}", toml_str);
        }
        (Some(key), None) => {
            // Get a specific value
            let toml_str = toml::to_string_pretty(config)
                .map_err(|e| sift_core::SiftError::Config(e.to_string()))?;

            // Simple key lookup in TOML
            for line in toml_str.lines() {
                if line.starts_with(&key) || line.contains(&format!("\"{}\"", key)) {
                    println!("{}", line);
                    return Ok(());
                }
            }
            println!("{}: key not found", key.red());
        }
        (Some(key), Some(value)) => {
            // Set a value - update config and save
            let mut config = config.clone();
            match key.as_str() {
                "default.model" => config.default.model = value.clone(),
                "default.chunk_size" => {
                    config.default.chunk_size = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                "default.chunk_overlap" => {
                    config.default.chunk_overlap = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                "default.jobs" => {
                    config.default.jobs = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                "search.max_results" => {
                    config.search.max_results = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                "search.hybrid_alpha" => {
                    config.search.hybrid_alpha = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                "server.host" => config.server.host = value.clone(),
                "server.port" => {
                    config.server.port = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?
                }
                _ => {
                    return Err(sift_core::SiftError::Config(format!(
                        "Unknown config key: {}",
                        key
                    )));
                }
            }

            config.save()?;
            println!("{} = {}", key.green(), value);
        }
        (None, Some(_)) => {
            return Err(sift_core::SiftError::Config(
                "Must specify a key when setting a value".into(),
            ));
        }
    }

    Ok(())
}
