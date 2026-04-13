use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Filesystem watch daemon that detects changes and triggers re-indexing.
pub struct WatchDaemon {
    paths: Vec<PathBuf>,
    debounce_ms: u64,
}

impl WatchDaemon {
    pub fn new(paths: Vec<PathBuf>, debounce_ms: u64) -> Self {
        Self { paths, debounce_ms }
    }

    /// Run the watch loop, calling `on_change` for each batch of changed files.
    pub fn run<F>(&self, on_change: F) -> anyhow::Result<()>
    where
        F: FnMut(Vec<PathBuf>),
    {
        self.run_inner(on_change, None)
    }

    /// Run the watch loop with a shutdown flag. The loop exits when the flag
    /// is set to `true`, after flushing any pending changes.
    pub fn run_with_shutdown<F>(
        &self,
        on_change: F,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()>
    where
        F: FnMut(Vec<PathBuf>),
    {
        self.run_inner(on_change, Some(shutdown))
    }

    fn run_inner<F>(
        &self,
        mut on_change: F,
        shutdown: Option<Arc<AtomicBool>>,
    ) -> anyhow::Result<()>
    where
        F: FnMut(Vec<PathBuf>),
    {
        let (tx, rx) = mpsc::channel::<notify::Result<Event>>();

        let mut watcher = RecommendedWatcher::new(tx, Config::default())?;

        for path in &self.paths {
            info!("Watching {} for changes...", path.display());
            watcher.watch(path, RecursiveMode::Recursive)?;
        }

        let debounce = Duration::from_millis(self.debounce_ms);
        let mut pending: Vec<PathBuf> = Vec::new();
        let mut last_event = Instant::now();

        loop {
            if shutdown.as_ref().is_some_and(|f| f.load(Ordering::Relaxed)) {
                // Flush any pending changes before exiting
                if !pending.is_empty() {
                    let batch: Vec<PathBuf> = std::mem::take(&mut pending);
                    info!("Flushing {} pending files before shutdown", batch.len());
                    on_change(batch);
                }
                info!("Watch loop shutting down");
                break;
            }

            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(event)) => {
                    for path in event.paths {
                        if path.is_file() && !pending.contains(&path) {
                            debug!("Change detected: {}", path.display());
                            pending.push(path);
                        }
                    }
                    last_event = Instant::now();
                }
                Ok(Err(e)) => {
                    warn!("Watch error: {}", e);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Check if debounce period has passed with pending changes
                    if !pending.is_empty() && last_event.elapsed() >= debounce {
                        let batch: Vec<PathBuf> = std::mem::take(&mut pending);
                        info!("Processing {} changed files", batch.len());
                        on_change(batch);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    info!("Watch channel disconnected, stopping");
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_with_nonexistent_path_returns_error() {
        let daemon = WatchDaemon::new(
            vec![PathBuf::from("/nonexistent/path/that/does/not/exist")],
            50,
        );
        let result = daemon.run(|_| {});
        assert!(
            result.is_err(),
            "watching a nonexistent path should return an error"
        );
    }

    #[test]
    fn test_shutdown_flag_stops_watch_loop() {
        let dir = tempfile::TempDir::new().unwrap();
        let daemon = WatchDaemon::new(vec![dir.path().to_path_buf()], 50);
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&flag);

        // Set shutdown flag after a short delay
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            flag_clone.store(true, Ordering::Relaxed);
        });

        // run_with_shutdown should exit after the flag is set
        let result = daemon.run_with_shutdown(|_| {}, flag);
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_creates_daemon_with_correct_config() {
        let paths = vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")];
        let daemon = WatchDaemon::new(paths.clone(), 500);
        assert_eq!(daemon.paths, paths);
        assert_eq!(daemon.debounce_ms, 500);
    }
}
