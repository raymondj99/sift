//! Source connectors for discovering files to index.
//!
//! Implements the [`Source`] trait for walking directories,
//! respecting `.gitignore` rules, and producing [`SourceItem`](sift_core::SourceItem)s
//! for the indexing pipeline.

pub mod filesystem;
pub mod traits;

pub use filesystem::FilesystemSource;
pub use traits::Source;
