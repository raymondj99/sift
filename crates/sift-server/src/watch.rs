use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc;
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
    pub fn run<F>(&self, mut on_change: F) -> anyhow::Result<()>
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
    fn test_watch_daemon_new() {
        let paths = vec![PathBuf::from("/tmp")];
        let daemon = WatchDaemon::new(paths.clone(), 500);
        assert_eq!(daemon.paths, paths);
        assert_eq!(daemon.debounce_ms, 500);
    }

    #[test]
    fn test_watch_daemon_new_with_custom_debounce() {
        let daemon = WatchDaemon::new(vec![PathBuf::from("/a"), PathBuf::from("/b")], 1000);
        assert_eq!(daemon.paths.len(), 2);
        assert_eq!(daemon.debounce_ms, 1000);
    }

    #[test]
    fn test_watch_daemon_new_empty_paths() {
        let daemon = WatchDaemon::new(vec![], 100);
        assert!(daemon.paths.is_empty());
        assert_eq!(daemon.debounce_ms, 100);
    }
}
