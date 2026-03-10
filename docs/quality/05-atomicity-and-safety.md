# 05: Atomicity, Safety & Crash Resilience

**Category**: Robustness
**Severity**: High
**Effort**: Medium
**Crates**: sift-store, sift-core, sift-cli

## Problem

### 1. No atomic file writes

The codebase uses `std::fs::write` directly for critical data files:

| File | Location | Data at Risk |
|------|----------|-------------|
| `vectors.bin` | `flat.rs:133` | Entire vector index |
| `bm25.json` | `bm25.rs:78` | Full-text search index |
| `metadata.json` | `json_metadata.rs:33` | All source metadata |
| `config.toml` | `config.rs:98` | User configuration |

If the process crashes, is OOM-killed, or the system loses power during a `write()`, these files can be truncated or partially written. The index becomes corrupted and the user must re-scan from scratch.

SQLite (with WAL mode) handles this correctly for `metadata.db`, but the vector index and BM25 index have no protection.

**Fix**: Write-then-rename (atomic swap):

```rust
fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(data)?;
    tmp.flush()?;
    tmp.persist(path)?;
    Ok(())
}
```

`tempfile::NamedTempFile::persist` does an `fsync` + `rename`, which is atomic on all modern filesystems.

### 2. No file locking for concurrent access

Multiple `sift` processes can run against the same index directory simultaneously. There's no advisory locking to prevent:
- Two `sift scan` processes writing to the same vector index
- `sift search` reading while `sift scan` is mid-write
- `sift remove` and `sift scan` racing on metadata

SQLite's WAL mode handles concurrent metadata access, but the vector index (`vectors.bin`) and BM25 index have no concurrency control.

**Fix**: Acquire an advisory lock on a `.lock` file in the index directory before any write operation:

```rust
use std::fs::File;
use fs2::FileExt;  // or use flock directly

fn lock_index(index_dir: &Path) -> std::io::Result<File> {
    let lock_path = index_dir.join(".sift.lock");
    let f = File::create(&lock_path)?;
    f.lock_exclusive()?;  // blocks until lock acquired
    Ok(f)
}
```

Drop the lock file handle when the operation completes.

### 3. No signal handling for graceful shutdown

Ctrl-C during a scan can leave the index in an inconsistent state:
- Metadata committed (via SQLite transaction) but vector index not yet saved
- Partial BM25 index written to disk
- Channels dropped mid-flight, losing in-progress work

**Fix**: Install a signal handler that sets an atomic flag, checked in the pipeline loop:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn install_signal_handler() {
    ctrlc::set_handler(|| {
        INTERRUPTED.store(true, Ordering::SeqCst);
    }).expect("failed to set Ctrl-C handler");
}
```

The store stage checks `INTERRUPTED` before each file and performs a clean commit if interrupted. The dependency is lightweight: `ctrlc = "3"`.

### 4. `dirs_home` falls back to current directory

```rust
// config.rs:238-243
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
```

If neither `$HOME` nor `$USERPROFILE` is set, `Config::sift_dir()` returns `./.sift`. This means the index is silently created in whatever directory `sift` happens to be run from, which is almost certainly wrong.

**Fix**: Return an error:

```rust
fn dirs_home() -> SiftResult<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| SiftError::Config(
            "Cannot determine home directory. Set $HOME or $USERPROFILE.".into()
        ))
}
```

Or use the `dirs` crate which handles platform-specific home directory detection correctly (including Windows, macOS, and Linux edge cases).

### 5. Transaction safety gap in pipeline

```rust
// pipeline.rs:452-454
if let Err(e) = metadata.begin_transaction() {
    store_error = Some(e);
}
// ... items processed ...
// pipeline.rs:513-516
if store_error.is_none() {
    if let Err(e) = metadata.commit_transaction() {
        store_error = Some(e);
    }
}
```

If `begin_transaction` succeeds but the loop exits early (via `break` on store error), the transaction is left open. SQLite will roll it back when the connection closes, but this means partial work is lost silently. There should be an explicit rollback on error.

Also: the `begin_transaction` / `commit_transaction` API on `MetadataStore` re-acquires the mutex each time. If another thread somehow calls a metadata method between begin and commit, it will execute outside the transaction. The API should use a guard pattern:

```rust
pub fn transaction<F, R>(&self, f: F) -> SiftResult<R>
where F: FnOnce(&Connection) -> SiftResult<R>
{
    let conn = self.conn.lock().map_err(lock_err)?;
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match f(&conn) {
        Ok(result) => {
            conn.execute_batch("COMMIT")?;
            Ok(result)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}
```

## Priority

1. **Atomic writes** — highest priority. Data loss is unacceptable.
2. **Transaction guard** — ensures metadata consistency.
3. **Signal handling** — prevents Ctrl-C corruption.
4. **File locking** — prevents multi-process corruption.
5. **Home directory** — prevents silent misconfiguration.
