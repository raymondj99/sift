#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::SiftResult;
use std::path::Path;

const TEMPLATE: &str = r#"# Sift project configuration
# Values here override your global ~/.sift/config.toml for this project.
# Only keys you explicitly set will override — everything else inherits.

[default]
# model = "nomic-embed-text-v2"   # embedding model
# chunk_size = 512                 # tokens per chunk
# chunk_overlap = 64               # overlap between chunks
# max_file_size = 104857600        # skip files larger than this (bytes)
# jobs = 0                         # parallel workers (0 = auto-detect)

[search]
# max_results = 10                 # default result count
# hybrid_alpha = 0.7               # blend: 0.0 = keyword-only, 1.0 = vector-only
# rerank = true                    # re-rank results after fusion

[ignore]
# patterns = ["node_modules", ".git", "__pycache__", "*.pyc", "target/", ".sift/"]
"#;

pub fn run(force: bool) -> SiftResult<()> {
    let cwd = std::env::current_dir().map_err(|e| {
        sift_core::SiftError::Config(format!("Cannot determine current directory: {e}"))
    })?;
    let config_path = cwd.join(".sift.toml");

    if config_path.exists() && !force {
        return Err(sift_core::SiftError::Config(format!(
            "{} already exists (use --force to overwrite)",
            config_path.display()
        )));
    }

    std::fs::write(&config_path, TEMPLATE)?;
    println!("{} {}", "Created".green(), config_path.display());

    // Append .sift/ to .gitignore if git repo and not already ignored
    if let Some(gitignore_path) = find_gitignore(&cwd) {
        ensure_gitignore_entry(&gitignore_path, ".sift/")?;
    }

    println!(
        "\n{}",
        "Uncomment and edit values to customize sift for this project.".dimmed()
    );

    Ok(())
}

/// Walk up to find the nearest .gitignore (in a directory that has .git).
fn find_gitignore(start: &Path) -> Option<std::path::PathBuf> {
    let mut dir = start;
    loop {
        if dir.join(".git").exists() {
            return Some(dir.join(".gitignore"));
        }
        dir = dir.parent()?;
    }
}

/// Append `entry` to `.gitignore` if it's not already present.
fn ensure_gitignore_entry(path: &Path, entry: &str) -> SiftResult<()> {
    let content = if path.exists() {
        std::fs::read_to_string(path)?
    } else {
        String::new()
    };

    // Check if entry is already present (exact line match)
    if content.lines().any(|line| line.trim() == entry) {
        return Ok(());
    }

    let mut new_content = content;
    if !new_content.ends_with('\n') && !new_content.is_empty() {
        new_content.push('\n');
    }
    new_content.push_str(entry);
    new_content.push('\n');

    std::fs::write(path, new_content)?;
    #[cfg(feature = "fancy")]
    println!("{} .sift/ to {}", "Added".green(), path.display());
    #[cfg(not(feature = "fancy"))]
    println!("Added .sift/ to {}", path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::CWD_MUTEX;

    #[test]
    fn test_template_is_valid_toml() {
        // All lines are comments or empty — should parse as empty table
        let table: toml::Table = toml::from_str(TEMPLATE).unwrap();
        assert!(table.is_empty() || table.values().all(|v| v.is_table()));
    }

    #[test]
    fn test_init_creates_config_file() {
        let _lock = CWD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::TempDir::new().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let result = run(false);
        std::env::set_current_dir(&original_dir).unwrap();

        assert!(result.is_ok());
        assert!(dir.path().join(".sift.toml").exists());
    }

    #[test]
    fn test_init_refuses_overwrite_without_force() {
        let _lock = CWD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".sift.toml"), "existing").unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let result = run(false);
        std::env::set_current_dir(&original_dir).unwrap();

        assert!(result.is_err());
        // Original content preserved
        let content = std::fs::read_to_string(dir.path().join(".sift.toml")).unwrap();
        assert_eq!(content, "existing");
    }

    #[test]
    fn test_init_force_overwrites() {
        let _lock = CWD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".sift.toml"), "old").unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let result = run(true);
        std::env::set_current_dir(&original_dir).unwrap();

        assert!(result.is_ok());
        let content = std::fs::read_to_string(dir.path().join(".sift.toml")).unwrap();
        assert!(content.contains("[default]"));
    }

    #[test]
    fn test_ensure_gitignore_adds_entry() {
        let dir = tempfile::TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");
        std::fs::write(&gitignore, "/target\n").unwrap();

        ensure_gitignore_entry(&gitignore, ".sift/").unwrap();

        let content = std::fs::read_to_string(&gitignore).unwrap();
        assert!(content.contains(".sift/"));
        assert!(content.contains("/target"));
    }

    #[test]
    fn test_ensure_gitignore_no_duplicate() {
        let dir = tempfile::TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");
        std::fs::write(&gitignore, ".sift/\n").unwrap();

        ensure_gitignore_entry(&gitignore, ".sift/").unwrap();

        let content = std::fs::read_to_string(&gitignore).unwrap();
        assert_eq!(content.matches(".sift/").count(), 1);
    }

    #[test]
    fn test_ensure_gitignore_creates_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");

        ensure_gitignore_entry(&gitignore, ".sift/").unwrap();

        assert!(gitignore.exists());
        let content = std::fs::read_to_string(&gitignore).unwrap();
        assert_eq!(content.trim(), ".sift/");
    }
}
