use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "Config::default_index_name")]
    pub index_name: String,
    #[serde(default)]
    pub default: DefaultConfig,
    #[serde(default)]
    pub ignore: IgnoreConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultConfig {
    #[serde(default = "DefaultConfig::default_model")]
    pub model: String,
    #[serde(default = "DefaultConfig::default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "DefaultConfig::default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "DefaultConfig::default_max_file_size")]
    pub max_file_size: u64,
    #[serde(default)]
    pub jobs: usize, // 0 = auto
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoreConfig {
    #[serde(default = "IgnoreConfig::default_patterns")]
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "SearchConfig::default_max_results")]
    pub max_results: usize,
    #[serde(default = "SearchConfig::default_hybrid_alpha")]
    pub hybrid_alpha: f32,
    #[serde(default = "SearchConfig::default_rerank")]
    pub rerank: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "ServerConfig::default_host")]
    pub host: String,
    #[serde(default = "ServerConfig::default_port")]
    pub port: u16,
}

impl Config {
    fn default_index_name() -> String {
        "default".into()
    }

    pub fn sift_dir() -> crate::SiftResult<PathBuf> {
        Ok(dirs_home()?.join(".sift"))
    }

    pub fn index_dir(&self) -> crate::SiftResult<PathBuf> {
        Ok(Self::sift_dir()?.join("indexes").join(&self.index_name))
    }

    pub fn models_dir() -> crate::SiftResult<PathBuf> {
        Ok(Self::sift_dir()?.join("models"))
    }

    pub fn config_path() -> crate::SiftResult<PathBuf> {
        Ok(Self::sift_dir()?.join("config.toml"))
    }

    pub fn load() -> crate::SiftResult<Self> {
        Self::load_from(Self::config_path()?)
    }

    pub fn load_from(path: impl AsRef<Path>) -> crate::SiftResult<Self> {
        let path = path.as_ref();
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            toml::from_str(&content).map_err(|e| crate::SiftError::Config(e.to_string()))
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self) -> crate::SiftResult<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content =
            toml::to_string_pretty(self).map_err(|e| crate::SiftError::Config(e.to_string()))?;
        crate::atomic_write(&path, content.as_bytes())?;
        Ok(())
    }

    pub fn ensure_dirs(&self) -> crate::SiftResult<()> {
        std::fs::create_dir_all(self.index_dir()?)?;
        std::fs::create_dir_all(Self::models_dir()?)?;
        Ok(())
    }

    pub fn num_jobs(&self) -> usize {
        if self.default.jobs == 0 {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4)
        } else {
            self.default.jobs
        }
    }

    /// Look up a dotted config key (e.g. "`search.max_results`") and return its
    /// TOML-formatted value string, or `None` if the key is not recognized.
    pub fn get_value(&self, key: &str) -> Option<String> {
        match key {
            "index_name" => Some(self.index_name.clone()),
            "default.model" => Some(self.default.model.clone()),
            "default.chunk_size" => Some(self.default.chunk_size.to_string()),
            "default.chunk_overlap" => Some(self.default.chunk_overlap.to_string()),
            "default.max_file_size" => Some(self.default.max_file_size.to_string()),
            "default.jobs" => Some(self.default.jobs.to_string()),
            "search.max_results" => Some(self.search.max_results.to_string()),
            "search.hybrid_alpha" => Some(self.search.hybrid_alpha.to_string()),
            "search.rerank" => Some(self.search.rerank.to_string()),
            "server.host" => Some(self.server.host.clone()),
            "server.port" => Some(self.server.port.to_string()),
            _ => None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            index_name: Self::default_index_name(),
            default: DefaultConfig::default(),
            ignore: IgnoreConfig::default(),
            search: SearchConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

impl DefaultConfig {
    fn default_model() -> String {
        "nomic-embed-text-v2".into()
    }
    fn default_chunk_size() -> usize {
        512
    }
    fn default_chunk_overlap() -> usize {
        64
    }
    fn default_max_file_size() -> u64 {
        100 * 1024 * 1024 // 100MB
    }
}

impl Default for DefaultConfig {
    fn default() -> Self {
        Self {
            model: Self::default_model(),
            chunk_size: Self::default_chunk_size(),
            chunk_overlap: Self::default_chunk_overlap(),
            max_file_size: Self::default_max_file_size(),
            jobs: 0,
        }
    }
}

impl IgnoreConfig {
    fn default_patterns() -> Vec<String> {
        vec![
            "node_modules".into(),
            ".git".into(),
            "__pycache__".into(),
            "*.pyc".into(),
            "target/".into(),
            ".sift/".into(),
        ]
    }
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            patterns: Self::default_patterns(),
        }
    }
}

impl SearchConfig {
    fn default_max_results() -> usize {
        10
    }
    fn default_hybrid_alpha() -> f32 {
        0.7
    }
    fn default_rerank() -> bool {
        true
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: Self::default_max_results(),
            hybrid_alpha: Self::default_hybrid_alpha(),
            rerank: Self::default_rerank(),
        }
    }
}

impl ServerConfig {
    fn default_host() -> String {
        "127.0.0.1".into()
    }
    fn default_port() -> u16 {
        7820
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: Self::default_host(),
            port: Self::default_port(),
        }
    }
}

fn dirs_home() -> crate::SiftResult<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| {
            crate::SiftError::Config(
                "Cannot determine home directory. Set $HOME or $USERPROFILE.".into(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_roundtrip() {
        let config = Config::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(deserialized.default.model, config.default.model);
        assert_eq!(deserialized.server.port, config.server.port);
    }

    #[test]
    fn test_load_from_valid_toml_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
index_name = "custom"

[default]
model = "bge-m3"
chunk_size = 256
chunk_overlap = 32
max_file_size = 1024
jobs = 4

[search]
max_results = 20
hybrid_alpha = 0.5
rerank = false

[server]
host = "0.0.0.0"
port = 9000

[ignore]
patterns = ["*.log"]
"#,
        )
        .unwrap();

        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.index_name, "custom");
        assert_eq!(config.default.model, "bge-m3");
        assert_eq!(config.default.chunk_size, 256);
        assert_eq!(config.default.chunk_overlap, 32);
        assert_eq!(config.default.max_file_size, 1024);
        assert_eq!(config.default.jobs, 4);
        assert_eq!(config.search.max_results, 20);
        assert!((config.search.hybrid_alpha - 0.5).abs() < f32::EPSILON);
        assert!(!config.search.rerank);
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.ignore.patterns, vec!["*.log".to_string()]);
    }

    #[test]
    fn test_save_and_reload() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("subdir").join("config.toml");

        // Override config_path by saving to a custom location
        let mut config = Config {
            index_name: "my-index".into(),
            ..Config::default()
        };
        config.default.chunk_size = 1024;
        config.search.max_results = 50;

        // Save manually to our test path
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let content = toml::to_string_pretty(&config).unwrap();
        std::fs::write(&path, &content).unwrap();

        // Reload and verify
        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(loaded.index_name, "my-index");
        assert_eq!(loaded.default.chunk_size, 1024);
        assert_eq!(loaded.search.max_results, 50);
    }

    #[test]
    fn test_num_jobs_explicit() {
        let mut config = Config::default();
        config.default.jobs = 8;
        assert_eq!(config.num_jobs(), 8);
    }
}
