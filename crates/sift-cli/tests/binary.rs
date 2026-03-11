use assert_cmd::{cargo_bin, Command};
use predicates::prelude::*;

fn sift_cmd() -> Command {
    Command::new(cargo_bin!("sift"))
}

#[test]
fn test_no_args_shows_help() {
    sift_cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn test_version_flag() {
    sift_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("sift"));
}

#[test]
fn test_status_fresh_index() {
    let dir = tempfile::TempDir::new().unwrap();
    let idx = format!("test-status-{}", std::process::id());
    sift_cmd()
        .args(["--index", &idx, "status"])
        .env("SIFT_DATA_DIR", dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("No index found"));
}

#[test]
fn test_scan_nonexistent_path_succeeds_with_zero() {
    let dir = tempfile::TempDir::new().unwrap();
    let idx = format!("test-scan-{}", std::process::id());
    sift_cmd()
        .args([
            "--index",
            &idx,
            "scan",
            "/nonexistent/path/that/does/not/exist",
        ])
        .env("SIFT_DATA_DIR", dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("0"));
}

#[test]
fn test_search_fresh_index() {
    let dir = tempfile::TempDir::new().unwrap();
    let idx = format!("test-search-{}", std::process::id());
    sift_cmd()
        .args(["--index", &idx, "search", "test query"])
        .env("SIFT_DATA_DIR", dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("No results found"));
}
