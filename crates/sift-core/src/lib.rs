//! Core types, configuration, and error handling for the vx indexing pipeline.
//!
//! This crate defines the shared data types that flow through the
//! `Source → Parse → Chunk → Embed → Store → Search` pipeline:
//! [`SourceItem`], [`ParsedDocument`], [`Chunk`], [`EmbeddedChunk`],
//! and [`SearchResult`].

pub mod config;
pub mod error;
pub mod types;

pub use config::Config;
pub use error::{SiftError, SiftResult};
pub use types::*;
