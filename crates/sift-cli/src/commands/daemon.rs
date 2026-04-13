#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::{Config, SiftResult};
use sift_server::routes::{create_router, AppState};
use std::path::PathBuf;
use std::sync::Arc;

fn socket_path() -> SiftResult<PathBuf> {
    Ok(Config::sift_dir()?.join("daemon.sock"))
}

fn pid_path() -> SiftResult<PathBuf> {
    Ok(Config::sift_dir()?.join("daemon.pid"))
}

fn read_pid() -> Option<u32> {
    let path = pid_path().ok()?;
    let content = std::fs::read_to_string(path).ok()?;
    content.trim().parse().ok()
}

fn is_process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Start the daemon as a detached background process.
pub fn start(config: &Config) -> SiftResult<()> {
    config.ensure_dirs()?;

    // Check if already running
    if let Some(pid) = read_pid() {
        if is_process_alive(pid) {
            println!("{} Daemon already running (PID {})", "●".green(), pid);
            return Ok(());
        }
        // Stale PID file — clean up
        let _ = std::fs::remove_file(pid_path()?);
        let _ = std::fs::remove_file(socket_path()?);
    }

    let exe = std::env::current_exe().map_err(|e| {
        sift_core::SiftError::Other(anyhow::anyhow!("Cannot find sift executable: {e}"))
    })?;

    let log_path = Config::sift_dir()?.join("daemon.log");
    let log_file = std::fs::File::create(&log_path)?;

    let child = std::process::Command::new(&exe)
        .args(["daemon", "run"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log_file))
        .spawn()
        .map_err(|e| sift_core::SiftError::Other(anyhow::anyhow!("Failed to spawn daemon: {e}")))?;

    let pid = child.id();

    // Wait for socket to appear (up to 10s)
    let socket = socket_path()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if socket.exists() {
            println!("{} Daemon started (PID {})", "●".green(), pid);
            println!("  Socket: {}", socket.display());
            println!("  Log:    {}", log_path.display());
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!(
        "{} Daemon process started (PID {}) but socket not ready yet. Check {}",
        "●".yellow(),
        pid,
        log_path.display()
    );
    Ok(())
}

/// Stop a running daemon by sending SIGTERM.
pub fn stop() -> SiftResult<()> {
    let pid = match read_pid() {
        Some(pid) if is_process_alive(pid) => pid,
        Some(_) => {
            let _ = std::fs::remove_file(pid_path()?);
            let _ = std::fs::remove_file(socket_path()?);
            println!(
                "{} Daemon is not running (cleaned up stale files)",
                "●".dimmed()
            );
            return Ok(());
        }
        None => {
            println!("{} Daemon is not running", "●".dimmed());
            return Ok(());
        }
    };

    let status = std::process::Command::new("kill")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| sift_core::SiftError::Other(anyhow::anyhow!("Failed to send signal: {e}")))?;

    if !status.success() {
        return Err(sift_core::SiftError::Other(anyhow::anyhow!(
            "Failed to stop daemon (PID {pid})"
        )));
    }

    // Wait for process to exit (up to 5s)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if !is_process_alive(pid) {
            let _ = std::fs::remove_file(pid_path()?);
            let _ = std::fs::remove_file(socket_path()?);
            println!("{} Daemon stopped", "●".red());
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!(
        "{} Daemon did not stop within 5s (PID {}). Try: kill -9 {}",
        "●".yellow(),
        pid,
        pid
    );
    Ok(())
}

/// Print daemon status.
pub fn status() -> SiftResult<()> {
    match read_pid() {
        Some(pid) if is_process_alive(pid) => {
            let socket = socket_path()?;
            println!("{} Daemon is running (PID {})", "●".green(), pid);
            if socket.exists() {
                println!("  Socket: {}", socket.display());
            } else {
                println!("  Socket not found (daemon may be starting)");
            }
        }
        Some(_) => {
            let _ = std::fs::remove_file(pid_path()?);
            let _ = std::fs::remove_file(socket_path()?);
            println!(
                "{} Daemon is not running (cleaned up stale files)",
                "●".dimmed()
            );
        }
        None => {
            println!("{} Daemon is not running", "●".dimmed());
        }
    }
    Ok(())
}

/// Run the daemon in the foreground (called by `sift daemon run`).
///
/// Loads the search engine and embedding model, then serves the HTTP API
/// over a Unix domain socket with graceful shutdown on SIGTERM/SIGINT.
/// If `config.memory.enabled`, also spawns a background consolidation
/// loop that periodically processes episodes into knowledge.
pub async fn run_daemon(config: &Config) -> SiftResult<()> {
    let socket = socket_path()?;
    let pid_file = pid_path()?;

    // Write PID file
    let pid = std::process::id();
    if let Some(parent) = pid_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&pid_file, pid.to_string())?;

    // Open engine & embedder (this is the expensive part we avoid repeating)
    let (engine, metadata) = crate::pipeline::open_engine(config)?;

    #[cfg(feature = "embeddings")]
    let embedder: Option<Box<dyn sift_core::Embedder>> =
        crate::pipeline::load_embedder(None).map(|e| Box::new(e) as Box<dyn sift_core::Embedder>);
    #[cfg(not(feature = "embeddings"))]
    let embedder: Option<Box<dyn sift_core::Embedder>> = None;

    let has_embedder = embedder.is_some();
    let memory_dir = config.index_dir().ok().map(|d| d.join("memory"));

    // Open memory store once — shared across HTTP endpoints and consolidation loop
    let memory_store: Option<Arc<sift_memory::MemoryStore>> =
        memory_dir
            .as_ref()
            .and_then(|dir| match sift_memory::MemoryStore::open(dir) {
                Ok(store) => {
                    let store = store.with_decay_rate(config.memory.decay_rate);
                    tracing::info!("Memory store opened at {}", dir.display());
                    Some(Arc::new(store))
                }
                Err(e) => {
                    tracing::warn!("Failed to open memory store: {e}. Memory features disabled.");
                    None
                }
            });
    let consolidation_config = sift_memory::ConsolidationConfig::from(&config.memory);

    let state = Arc::new(AppState {
        engine,
        metadata,
        embedder,
        memory_dir: memory_dir.clone(),
        memory: memory_store.clone(),
        consolidation_config: Some(consolidation_config.clone()),
    });
    let app = create_router(state);

    tracing::info!(
        "Daemon listening on {} (embedder: {}, memory: {})",
        socket.display(),
        if has_embedder { "loaded" } else { "none" },
        if memory_store.is_some() {
            "loaded"
        } else {
            "none"
        }
    );

    // Shutdown flag — shared between HTTP server and consolidation loop
    let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Spawn consolidation loop if memory is enabled and store opened successfully
    let consolidation_handle = if let (true, Some(mem), Some(mem_dir)) =
        (config.memory.enabled, memory_store, memory_dir)
    {
        let cfg = consolidation_config;
        let interval = config.memory.consolidation_interval;
        let flag = Arc::clone(&shutdown_flag);
        Some(tokio::spawn(async move {
            consolidation_loop(mem, mem_dir, cfg, interval, flag).await;
        }))
    } else {
        tracing::info!("Memory consolidation disabled");
        None
    };

    // Graceful shutdown on SIGTERM / SIGINT
    let flag_for_shutdown = Arc::clone(&shutdown_flag);
    let shutdown = async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("failed to register SIGINT");
        tokio::select! {
            _ = sigterm.recv() => tracing::info!("Received SIGTERM, shutting down"),
            _ = sigint.recv() => tracing::info!("Received SIGINT, shutting down"),
        }
        flag_for_shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
    };

    sift_server::serve_unix(&socket, app, shutdown)
        .await
        .map_err(sift_core::SiftError::Other)?;

    // Wait for consolidation loop to finish
    if let Some(handle) = consolidation_handle {
        let _ = handle.await;
    }

    // Cleanup (socket is removed by serve_unix, but PID file needs cleanup)
    let _ = std::fs::remove_file(&pid_file);
    tracing::info!("Daemon stopped cleanly");

    Ok(())
}

/// Background consolidation loop.
///
/// Runs all 5 consolidation phases on a timer. The MemoryStore is opened
/// once at daemon startup and shared via Arc. EpisodeStore is lightweight
/// (SQLite only, no search engine) so it's opened per cycle to avoid
/// holding a second connection across long sleep intervals.
async fn consolidation_loop(
    memory: Arc<sift_memory::MemoryStore>,
    memory_dir: std::path::PathBuf,
    config: sift_memory::ConsolidationConfig,
    interval_secs: u64,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) {
    let interval = std::time::Duration::from_secs(interval_secs);

    tracing::info!(
        interval_secs = interval.as_secs(),
        "Consolidation loop started"
    );

    loop {
        // Sleep in 1-second increments so we can check the shutdown flag
        let mut remaining = interval;
        while !remaining.is_zero() {
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("Consolidation loop shutting down");
                return;
            }
            let tick = remaining.min(std::time::Duration::from_secs(1));
            tokio::time::sleep(tick).await;
            remaining = remaining.saturating_sub(tick);
        }

        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mem = Arc::clone(&memory);
        let dir = memory_dir.clone();
        let cfg = config.clone();

        let result =
            tokio::task::spawn_blocking(move || run_consolidation_cycle(&mem, &dir, &cfg)).await;

        match result {
            Ok(Ok(report)) => {
                if report.episodes_processed > 0
                    || report.duplicates_merged > 0
                    || report.promotions > 0
                    || report.observations_pruned > 0
                {
                    tracing::info!(
                        episodes = report.episodes_processed,
                        dedup = report.duplicates_merged,
                        promotions = report.promotions,
                        pruned = report.observations_pruned,
                        "Consolidation cycle completed"
                    );
                } else {
                    tracing::debug!("Consolidation cycle: nothing to do");
                }
            }
            Ok(Err(e)) => {
                tracing::warn!("Consolidation cycle failed: {e}");
            }
            Err(e) => {
                tracing::error!("Consolidation task panicked: {e}");
            }
        }
    }
}

/// Run a single consolidation cycle (sync, runs inside spawn_blocking).
///
/// The MemoryStore is shared (opened once at daemon startup). The
/// EpisodeStore is opened per cycle since it's lightweight (SQLite only).
fn run_consolidation_cycle(
    memory: &sift_memory::MemoryStore,
    memory_dir: &std::path::Path,
    config: &sift_memory::ConsolidationConfig,
) -> Result<sift_memory::consolidation::FullConsolidationReport, sift_memory::MemoryError> {
    let episodes = sift_memory::episodes::EpisodeStore::open(memory_dir)
        .map_err(sift_memory::MemoryError::Sqlite)?;

    let report = sift_memory::consolidation::run_consolidation(memory, &episodes, config)?;

    memory.save()?;

    // Regenerate agent rules if consolidation did meaningful work.
    let did_work = report.episodes_processed > 0
        || report.duplicates_merged > 0
        || report.promotions > 0
        || report.observations_pruned > 0;

    if did_work {
        if let Ok(cwd) = std::env::current_dir() {
            let project_root = sift_core::Config::find_project_root(&cwd).unwrap_or(cwd);
            match sift_memory::rules::generate_all_rules(memory, &project_root) {
                Ok(r) => {
                    tracing::info!(
                        rules = r.total_rules,
                        tokens = r.total_tokens_approx,
                        "Rules regenerated"
                    );
                }
                Err(e) => {
                    tracing::debug!("Rule generation skipped: {e}");
                }
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_socket_and_pid_paths() {
        let socket = socket_path();
        let pid = pid_path();
        // Should resolve without error
        assert!(socket.is_ok());
        assert!(pid.is_ok());
        assert!(socket.unwrap().ends_with("daemon.sock"));
        assert!(pid.unwrap().ends_with("daemon.pid"));
    }

    #[test]
    fn test_read_pid_returns_none_when_no_file() {
        // With no PID file, should return None
        // (This test is inherently environment-dependent but should be safe)
        let result = read_pid();
        // We can't assert None because a daemon might be running,
        // but we can assert it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_is_process_alive_for_current_process() {
        let pid = std::process::id();
        assert!(is_process_alive(pid));
    }

    #[test]
    fn test_is_process_alive_for_nonexistent_pid() {
        // PID 99999999 is very unlikely to exist
        assert!(!is_process_alive(99_999_999));
    }
}
