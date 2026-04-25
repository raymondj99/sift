//! Shared command helpers.

use std::path::PathBuf;

/// Resolve a stable path to the `sift` executable that survives package
/// upgrades.
///
/// `std::env::current_exe()` on macOS Homebrew installs resolves to a
/// versioned `Cellar/sift-search/<version>/bin/sift` directory. Baking that
/// into a config (e.g. a Claude Code hooks file or an MCP server entry)
/// breaks after the next `brew upgrade` because the Cellar directory is
/// deleted.
///
/// Resolution order:
/// 1. `which("sift")` on `$PATH` — picks up the stable Homebrew prefix
///    symlink (e.g. `/opt/homebrew/bin/sift`) or `cargo install`'d binary.
/// 2. If `current_exe()` resolves under a `Cellar/` segment, walk back to
///    the sibling `bin/sift` symlink in the brew prefix
///    (`<brew>/Cellar/<formula>/<version>/...` → `<brew>/bin/sift`).
/// 3. Fall back to `current_exe()` only when neither of the above succeeds.
pub(crate) fn resolve_stable_exe() -> PathBuf {
    if let Some(p) = which("sift") {
        return p;
    }
    let current = std::env::current_exe().ok();
    if let Some(ref path) = current {
        if let Some(brew_link) = brew_prefix_sibling(path) {
            return brew_link;
        }
    }
    current.unwrap_or_else(|| PathBuf::from("sift"))
}

/// Walk up from a `Cellar/<formula>/<version>/...` path to the sibling
/// `bin/sift` symlink in the brew prefix, returning that path if it exists.
fn brew_prefix_sibling(exe: &std::path::Path) -> Option<PathBuf> {
    let components: Vec<_> = exe.components().collect();
    let cellar_idx = components.iter().position(|c| c.as_os_str() == "Cellar")?;
    // brew prefix = everything before "Cellar"
    let prefix: PathBuf = components[..cellar_idx].iter().collect();
    let candidate = prefix.join("bin").join("sift");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// `which`-style PATH lookup for a command name.
pub(crate) fn which(cmd: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{cmd}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brew_prefix_sibling_returns_none_when_not_under_cellar() {
        let p = std::path::Path::new("/usr/local/bin/sift");
        assert!(brew_prefix_sibling(p).is_none());
    }

    #[test]
    fn brew_prefix_sibling_walks_to_bin_when_symlink_exists() {
        let tmp = tempfile::TempDir::new().unwrap();
        let prefix = tmp.path();
        std::fs::create_dir_all(prefix.join("bin")).unwrap();
        std::fs::write(prefix.join("bin").join("sift"), b"#!/bin/sh\n").unwrap();
        let cellar_exe = prefix
            .join("Cellar")
            .join("sift-search")
            .join("0.1.7")
            .join("bin")
            .join("sift");
        // Note: file existence at the cellar path isn't checked, only the
        // sibling bin/sift in the prefix.
        let resolved = brew_prefix_sibling(&cellar_exe).expect("sibling should be found");
        assert_eq!(resolved, prefix.join("bin").join("sift"));
    }

    #[test]
    fn brew_prefix_sibling_returns_none_when_sibling_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let prefix = tmp.path();
        let cellar_exe = prefix
            .join("Cellar")
            .join("sift-search")
            .join("0.1.7")
            .join("bin")
            .join("sift");
        // No bin/sift was created — should return None.
        assert!(brew_prefix_sibling(&cellar_exe).is_none());
    }

    /// Serialize tests that mutate process-global env state.
    static PATH_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[allow(unsafe_code)]
    fn with_path<F, R>(new_path: &std::path::Path, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _lock = PATH_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("PATH");
        unsafe { std::env::set_var("PATH", new_path) };
        let result = f();
        match prev {
            Some(v) => unsafe { std::env::set_var("PATH", v) },
            None => unsafe { std::env::remove_var("PATH") },
        }
        result
    }

    #[test]
    #[cfg(unix)]
    fn resolve_stable_exe_prefers_path_sift_over_cellar() {
        // When `sift` is on $PATH, the stable resolver must pick it — never
        // a versioned Cellar path. This is the regression guard against
        // baking `Cellar/sift-search/<version>/bin/sift` into hooks.json.
        let tmp = tempfile::TempDir::new().unwrap();
        let bin_dir = tmp.path().join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let stable_path = bin_dir.join("sift");
        std::fs::write(&stable_path, "#!/bin/sh\n").unwrap();
        // Mark executable so `is_file()` returns true and our which()
        // picks it up.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stable_path, std::fs::Permissions::from_mode(0o755)).unwrap();

        let resolved = with_path(&bin_dir, resolve_stable_exe);

        assert_eq!(resolved, stable_path);
        let resolved_str = resolved.display().to_string();
        assert!(
            !resolved_str.contains("Cellar/"),
            "stable exe path must not contain a Cellar segment: {resolved_str}"
        );
    }
}
