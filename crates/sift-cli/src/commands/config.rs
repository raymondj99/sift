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
            println!("{toml_str}");
        }
        (Some(key), None) => {
            // Get a specific value using structured lookup
            match config.get_value(&key) {
                Some(value) => println!("{} = {}", key.green(), value),
                None => println!("{}: key not found", key.red()),
            }
        }
        (Some(key), Some(value)) => {
            // Set a value - update config and save
            let mut config = config.clone();
            match key.as_str() {
                "default.model" => config.default.model.clone_from(&value),
                "default.chunk_size" => {
                    config.default.chunk_size = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                "default.chunk_overlap" => {
                    config.default.chunk_overlap = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                "default.jobs" => {
                    config.default.jobs = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                "search.max_results" => {
                    config.search.max_results = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                "search.hybrid_alpha" => {
                    config.search.hybrid_alpha = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                "server.host" => config.server.host.clone_from(&value),
                "server.port" => {
                    config.server.port = value
                        .parse()
                        .map_err(|_| sift_core::SiftError::Config("Invalid number".into()))?;
                }
                _ => {
                    return Err(sift_core::SiftError::Config(format!(
                        "Unknown config key: {key}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{with_home, HOME_MUTEX};
    use tempfile::TempDir;

    #[test]
    fn test_config_set_default_model() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            config.ensure_dirs().unwrap();
            assert!(run(
                &config,
                Some("default.model".into()),
                Some("custom-model".into())
            )
            .is_ok());
        });
    }

    #[test]
    fn test_config_set_invalid_chunk_size_errors() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let result = run(
                &config,
                Some("default.chunk_size".into()),
                Some("not_a_number".into()),
            );
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_config_set_unknown_key_errors() {
        let _lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let config = sift_core::Config::default();
            let result = run(&config, Some("unknown.key".into()), Some("value".into()));
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Unknown config key"),
                "expected 'Unknown config key' in: {err}"
            );
        });
    }

    #[test]
    fn test_config_value_without_key_errors() {
        // (None, Some(value)) is not a valid invocation
        let config = sift_core::Config::default();
        let result = run(&config, None, Some("value".into()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Must specify a key"));
    }
}
