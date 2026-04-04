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
        let config = Self::load_from(Self::config_path()?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn load_from(path: impl AsRef<Path>) -> crate::SiftResult<Self> {
        let path = path.as_ref();
        let config = if path.exists() {
            let content = std::fs::read_to_string(path)?;
            toml::from_str(&content).map_err(|e| crate::SiftError::Config(e.to_string()))?
        } else {
            Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Load global config, then merge with a `.sift.toml` in `project_dir` if present.
    pub fn load_merged(project_dir: Option<&Path>) -> crate::SiftResult<Self> {
        let global = Self::load()?;
        let project_dir = match project_dir {
            Some(d) => d,
            None => return Ok(global),
        };
        let project_config_path = project_dir.join(".sift.toml");
        if !project_config_path.exists() {
            return Ok(global);
        }
        let project_content = std::fs::read_to_string(&project_config_path)?;

        // Parse as Config (with serde defaults for missing fields)
        let project: Config = toml::from_str(&project_content).map_err(|e| {
            crate::SiftError::Config(format!("{}: {e}", project_config_path.display()))
        })?;

        // Also parse as raw TOML table so we know which keys were actually specified
        let project_table: toml::Table = toml::from_str(&project_content).map_err(|e| {
            crate::SiftError::Config(format!("{}: {e}", project_config_path.display()))
        })?;

        let merged = global.merge(&project, &project_table);
        merged.validate()?;
        Ok(merged)
    }

    /// Merge values from `other` into `self`, but only for keys explicitly
    /// present in `specified` (the raw TOML table from the project config file).
    pub fn merge(&self, other: &Config, specified: &toml::Table) -> Config {
        let mut merged = self.clone();

        if specified.contains_key("index_name") {
            merged.index_name.clone_from(&other.index_name);
        }

        if let Some(default_table) = specified.get("default").and_then(|v| v.as_table()) {
            if default_table.contains_key("model") {
                merged.default.model.clone_from(&other.default.model);
            }
            if default_table.contains_key("chunk_size") {
                merged.default.chunk_size = other.default.chunk_size;
            }
            if default_table.contains_key("chunk_overlap") {
                merged.default.chunk_overlap = other.default.chunk_overlap;
            }
            if default_table.contains_key("max_file_size") {
                merged.default.max_file_size = other.default.max_file_size;
            }
            if default_table.contains_key("jobs") {
                merged.default.jobs = other.default.jobs;
            }
        }

        if let Some(search_table) = specified.get("search").and_then(|v| v.as_table()) {
            if search_table.contains_key("max_results") {
                merged.search.max_results = other.search.max_results;
            }
            if search_table.contains_key("hybrid_alpha") {
                merged.search.hybrid_alpha = other.search.hybrid_alpha;
            }
            if search_table.contains_key("rerank") {
                merged.search.rerank = other.search.rerank;
            }
        }

        if let Some(ignore_table) = specified.get("ignore").and_then(|v| v.as_table()) {
            if ignore_table.contains_key("patterns") {
                merged.ignore.patterns.clone_from(&other.ignore.patterns);
            }
        }

        if let Some(server_table) = specified.get("server").and_then(|v| v.as_table()) {
            if server_table.contains_key("host") {
                merged.server.host.clone_from(&other.server.host);
            }
            if server_table.contains_key("port") {
                merged.server.port = other.server.port;
            }
        }

        merged
    }

    /// Detect the project root by walking up from `start_dir`, looking for
    /// `.sift.toml` or `.git`.
    pub fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
        let mut dir = start_dir;
        loop {
            if dir.join(".sift.toml").exists() || dir.join(".git").exists() {
                return Some(dir.to_path_buf());
            }
            dir = dir.parent()?;
        }
    }

    /// Check semantic constraints beyond what serde deserialization provides.
    pub fn validate(&self) -> crate::SiftResult<()> {
        if self.default.chunk_size == 0 {
            return Err(crate::SiftError::Config(
                "chunk_size must be greater than 0".into(),
            ));
        }
        if self.default.chunk_overlap >= self.default.chunk_size {
            return Err(crate::SiftError::Config(format!(
                "chunk_overlap ({}) must be less than chunk_size ({})",
                self.default.chunk_overlap, self.default.chunk_size
            )));
        }
        if self.default.max_file_size == 0 {
            return Err(crate::SiftError::Config(
                "max_file_size must be greater than 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.search.hybrid_alpha) {
            return Err(crate::SiftError::Config(format!(
                "hybrid_alpha ({}) must be between 0.0 and 1.0",
                self.search.hybrid_alpha
            )));
        }
        if self.search.max_results == 0 {
            return Err(crate::SiftError::Config(
                "max_results must be greater than 0".into(),
            ));
        }
        Ok(())
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
            "ignore.patterns" => Some(
                self.ignore
                    .patterns
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
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

    #[test]
    fn test_num_jobs_zero_auto_detects() {
        let config = Config::default();
        assert_eq!(config.default.jobs, 0);
        let jobs = config.num_jobs();
        // Should auto-detect to at least 1 CPU
        assert!(jobs >= 1);
    }

    #[test]
    fn test_get_value_known_keys() {
        let config = Config::default();
        assert!(config.get_value("index_name").is_some());
        assert!(config.get_value("default.model").is_some());
        assert!(config.get_value("default.chunk_size").is_some());
        assert!(config.get_value("default.chunk_overlap").is_some());
        assert!(config.get_value("default.max_file_size").is_some());
        assert!(config.get_value("default.jobs").is_some());
        assert!(config.get_value("search.max_results").is_some());
        assert!(config.get_value("search.hybrid_alpha").is_some());
        assert!(config.get_value("search.rerank").is_some());
        assert!(config.get_value("server.host").is_some());
        assert!(config.get_value("server.port").is_some());
        assert!(config.get_value("ignore.patterns").is_some());
    }

    #[test]
    fn test_get_value_unknown_key_returns_none() {
        let config = Config::default();
        assert!(config.get_value("nonexistent.key").is_none());
        assert!(config.get_value("").is_none());
        assert!(config.get_value("default.nonexistent").is_none());
    }

    #[test]
    fn test_get_value_returns_correct_values() {
        let mut config = Config::default();
        config.default.chunk_size = 256;
        config.search.max_results = 42;
        config.server.port = 9999;

        assert_eq!(config.get_value("default.chunk_size").unwrap(), "256");
        assert_eq!(config.get_value("search.max_results").unwrap(), "42");
        assert_eq!(config.get_value("server.port").unwrap(), "9999");
    }

    #[test]
    fn test_validate_default_config_passes() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_chunk_overlap_exceeds_chunk_size() {
        let mut config = Config::default();
        config.default.chunk_overlap = 512;
        config.default.chunk_size = 256;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("chunk_overlap"));
    }

    #[test]
    fn test_validate_chunk_overlap_equals_chunk_size() {
        let mut config = Config::default();
        config.default.chunk_overlap = 256;
        config.default.chunk_size = 256;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("chunk_overlap"));
    }

    #[test]
    fn test_validate_chunk_size_zero() {
        let mut config = Config::default();
        config.default.chunk_size = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("chunk_size"));
    }

    #[test]
    fn test_validate_max_file_size_zero() {
        let mut config = Config::default();
        config.default.max_file_size = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_file_size"));
    }

    #[test]
    fn test_validate_hybrid_alpha_too_high() {
        let mut config = Config::default();
        config.search.hybrid_alpha = 1.5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("hybrid_alpha"));
    }

    #[test]
    fn test_validate_hybrid_alpha_negative() {
        let mut config = Config::default();
        config.search.hybrid_alpha = -0.1;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("hybrid_alpha"));
    }

    #[test]
    fn test_validate_hybrid_alpha_boundaries() {
        let mut config = Config::default();
        config.search.hybrid_alpha = 0.0;
        assert!(config.validate().is_ok());
        config.search.hybrid_alpha = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_max_results_zero() {
        let mut config = Config::default();
        config.search.max_results = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_results"));
    }

    #[test]
    fn test_load_from_invalid_config_values() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[default]
chunk_size = 100
chunk_overlap = 200
"#,
        )
        .unwrap();
        let result = Config::load_from(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("chunk_overlap"));
    }

    #[test]
    fn test_load_from_invalid_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is {{ not valid toml").unwrap();
        let result = Config::load_from(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_nonexistent_returns_default() {
        let config =
            Config::load_from(std::path::Path::new("/nonexistent/path/config.toml")).unwrap();
        // Missing file returns default config, not error
        assert_eq!(config.default.chunk_size, 512);
        assert_eq!(config.server.port, 7820);
    }

    #[test]
    fn test_default_config_values() {
        let config = Config::default();
        assert_eq!(config.default.chunk_size, 512);
        assert_eq!(config.default.chunk_overlap, 64);
        assert_eq!(config.default.max_file_size, 100 * 1024 * 1024);
        assert_eq!(config.default.model, "nomic-embed-text-v2");
        assert_eq!(config.search.max_results, 10);
        assert!((config.search.hybrid_alpha - 0.7).abs() < f32::EPSILON);
        assert!(config.search.rerank);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 7820);
    }

    #[test]
    fn test_ignore_patterns_formatting() {
        let mut config = Config::default();
        config.ignore.patterns = vec!["*.log".into(), "*.tmp".into()];
        let value = config.get_value("ignore.patterns").unwrap();
        assert!(value.contains("\"*.log\""));
        assert!(value.contains("\"*.tmp\""));
    }

    #[test]
    fn test_merge_only_overrides_specified_keys() {
        let global = Config::default();
        let project_toml = r#"
[default]
chunk_size = 256
"#;
        let project: Config = toml::from_str(project_toml).unwrap();
        let table: toml::Table = toml::from_str(project_toml).unwrap();
        let merged = global.merge(&project, &table);

        // chunk_size should be overridden
        assert_eq!(merged.default.chunk_size, 256);
        // Other fields should keep global defaults
        assert_eq!(merged.default.model, "nomic-embed-text-v2");
        assert_eq!(merged.default.chunk_overlap, 64);
        assert_eq!(merged.search.max_results, 10);
        assert_eq!(merged.server.port, 7820);
    }

    #[test]
    fn test_merge_multiple_sections() {
        let mut global = Config::default();
        global.default.chunk_size = 1024;
        global.search.max_results = 50;

        let project_toml = r#"
[default]
chunk_size = 128

[search]
rerank = false

[ignore]
patterns = ["vendor/"]
"#;
        let project: Config = toml::from_str(project_toml).unwrap();
        let table: toml::Table = toml::from_str(project_toml).unwrap();
        let merged = global.merge(&project, &table);

        assert_eq!(merged.default.chunk_size, 128);
        assert!(!merged.search.rerank);
        assert_eq!(merged.ignore.patterns, vec!["vendor/".to_string()]);
        // Non-specified fields keep global values
        assert_eq!(merged.search.max_results, 50);
        assert_eq!(merged.server.port, 7820);
    }

    #[test]
    fn test_load_merged_no_project_dir() {
        // load_merged with None should behave like load()
        // We can't easily test this without a HOME dir, but we can verify no panic
        let dir = tempfile::TempDir::new().unwrap();
        let result = Config::load_merged(Some(dir.path()));
        // No .sift.toml exists, so this should return global config
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_merged_with_project_config() {
        let dir = tempfile::TempDir::new().unwrap();
        let home_dir = dir.path().join("home");
        let sift_dir = home_dir.join(".sift");
        std::fs::create_dir_all(&sift_dir).unwrap();

        // Write a global config
        std::fs::write(
            sift_dir.join("config.toml"),
            r#"
[default]
chunk_size = 512
model = "nomic-embed-text-v2"

[search]
max_results = 10
"#,
        )
        .unwrap();

        // Write a project .sift.toml
        let project_dir = dir.path().join("project");
        std::fs::create_dir_all(&project_dir).unwrap();
        std::fs::write(
            project_dir.join(".sift.toml"),
            r#"
[default]
chunk_size = 256
"#,
        )
        .unwrap();

        // Use load_from for global, then merge manually (since load_merged
        // calls load() which uses the real HOME)
        let global = Config::load_from(sift_dir.join("config.toml")).unwrap();
        let project_content = std::fs::read_to_string(project_dir.join(".sift.toml")).unwrap();
        let project: Config = toml::from_str(&project_content).unwrap();
        let table: toml::Table = toml::from_str(&project_content).unwrap();
        let merged = global.merge(&project, &table);

        assert_eq!(merged.default.chunk_size, 256);
        assert_eq!(merged.default.model, "nomic-embed-text-v2");
        assert_eq!(merged.search.max_results, 10);
    }

    #[test]
    fn test_find_project_root_with_sift_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join("a").join(".sift.toml"), "").unwrap();

        let root = Config::find_project_root(&nested);
        assert_eq!(root.unwrap(), dir.path().join("a"));
    }

    #[test]
    fn test_find_project_root_with_git() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("repo").join("src");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(dir.path().join("repo").join(".git")).unwrap();

        let root = Config::find_project_root(&nested);
        assert_eq!(root.unwrap(), dir.path().join("repo"));
    }

    #[test]
    fn test_find_project_root_prefers_closest() {
        let dir = tempfile::TempDir::new().unwrap();
        let inner = dir.path().join("outer").join("inner");
        std::fs::create_dir_all(&inner).unwrap();
        // .git at outer level
        std::fs::create_dir_all(dir.path().join("outer").join(".git")).unwrap();
        // .sift.toml at inner level
        std::fs::write(inner.join(".sift.toml"), "").unwrap();

        let root = Config::find_project_root(&inner);
        assert_eq!(root.unwrap(), inner);
    }

    #[test]
    fn test_find_project_root_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("empty").join("dir");
        std::fs::create_dir_all(&nested).unwrap();

        // This may or may not find a root depending on whether there's a .git
        // above the tempdir. Just verify it doesn't panic.
        let _ = Config::find_project_root(&nested);
    }
}
