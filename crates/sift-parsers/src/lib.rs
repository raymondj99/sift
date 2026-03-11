//! File format parsers with MIME-based dispatch.
//!
//! Provides a [`ParserRegistry`] that selects the correct parser for each file
//! based on its MIME type and extension. Supports 30+ formats including text,
//! code, PDF, Office documents, HTML, CSV, JSON, email, images, audio, and archives.
//!
//! Most parsers are feature-gated to minimize binary size.

pub mod error;
pub mod registry;
pub mod skill;
pub mod traits;

#[cfg(feature = "archive")]
mod archive;
#[cfg(feature = "audio")]
mod audio;
mod code;
#[cfg(feature = "data")]
mod data;
#[cfg(feature = "email")]
mod email;
#[cfg(feature = "epub")]
mod epub;
mod image;
mod notebook;
#[cfg(feature = "office")]
mod office;
#[cfg(feature = "pdf")]
mod pdf;
mod rtf;
#[cfg(feature = "spreadsheets")]
mod spreadsheet;
mod text;
mod web;

pub use error::ParseError;
pub use registry::ParserRegistry;
pub use traits::Parser;
