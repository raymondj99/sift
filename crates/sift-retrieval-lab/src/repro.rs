//! Reproducibility envelope captured into every run JSON.
//!
//! Cheap fields only — the sift binary already pays for tracing and config
//! capture. Heavyweight fields (cpu model, ONNX provider list, etc.) are
//! deferred until a reviewer actually needs them to debug a regression.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub git_sha: Option<String>,
    pub git_dirty: bool,
    pub rustc: String,
    pub target: String,
    pub timestamp_utc: String,
}

impl Environment {
    pub fn capture() -> Self {
        Self {
            git_sha: git_sha(),
            git_dirty: git_dirty(),
            rustc: rustc_version(),
            target: env!("TARGET_TRIPLE").to_string(),
            timestamp_utc: chrono::Utc::now().to_rfc3339(),
        }
    }
}

fn git_sha() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn git_dirty() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .is_ok_and(|o| !o.stdout.is_empty())
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}
