use std::io::Write;
use std::path::Path;

/// Write `data` to `path` atomically via a temporary file + rename.
///
/// The rename is atomic on POSIX, so readers will never observe a
/// partially-written file.
pub fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(data)?;
    tmp.flush()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}
