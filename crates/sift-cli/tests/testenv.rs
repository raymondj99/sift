use std::path::Path;
use tempfile::TempDir;

pub struct TestEnv {
    pub dir: TempDir,
    pub index_dir: TempDir,
}

pub struct TestEnvBuilder {
    files: Vec<(String, String)>,
    binary_files: Vec<(String, Vec<u8>)>,
}

impl Default for TestEnvBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TestEnvBuilder {
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            binary_files: Vec::new(),
        }
    }

    pub fn with_file(mut self, path: &str, content: &str) -> Self {
        self.files.push((path.to_string(), content.to_string()));
        self
    }

    pub fn with_binary_file(mut self, path: &str, data: Vec<u8>) -> Self {
        self.binary_files.push((path.to_string(), data));
        self
    }

    pub fn build(self) -> TestEnv {
        let dir = TempDir::new().unwrap();
        let index_dir = TempDir::new().unwrap();

        for (path, content) in &self.files {
            let full_path = dir.path().join(path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&full_path, content).unwrap();
        }

        for (path, data) in &self.binary_files {
            let full_path = dir.path().join(path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&full_path, data).unwrap();
        }

        TestEnv { dir, index_dir }
    }
}

impl TestEnv {
    pub fn builder() -> TestEnvBuilder {
        TestEnvBuilder::new()
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn index_path(&self) -> &Path {
        self.index_dir.path()
    }
}
