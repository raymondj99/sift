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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn atomic_write_creates_file_with_correct_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let data = b"hello world";

        atomic_write(&path, data).unwrap();

        let read_back = fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn atomic_write_overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overwrite.txt");

        atomic_write(&path, b"first").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");

        atomic_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn atomic_write_in_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        let path = sub.join("file.bin");

        let data = vec![0u8, 1, 2, 3, 255];
        atomic_write(&path, &data).unwrap();

        assert_eq!(fs::read(&path).unwrap(), data);
    }
}
