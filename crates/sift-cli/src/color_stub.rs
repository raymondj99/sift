//! No-op replacements for [`colored`] types used throughout the CLI.
//!
//! When the `fancy` feature is disabled, this module provides a [`Colorize`]
//! trait whose methods simply return the input string unchanged.  This keeps
//! all call-sites compiling without `#[cfg]` guards on every `println!`.

/// Placeholder for `colored::Color`.  Only the variants actually referenced
/// by the codebase are included.
#[allow(dead_code)]
pub enum Color {
    Red,
    Green,
    Yellow,
    Blue,
    Cyan,
    White,
}

#[allow(dead_code)]
pub trait Colorize {
    fn red(&self) -> &str;
    fn green(&self) -> &str;
    fn blue(&self) -> &str;
    fn cyan(&self) -> &str;
    fn yellow(&self) -> &str;
    fn white(&self) -> &str;
    fn bold(&self) -> &str;
    fn dimmed(&self) -> &str;
    fn italic(&self) -> &str;
    fn underline(&self) -> &str;
    fn bright_black(&self) -> &str;
    fn bright_white(&self) -> &str;
    fn color(&self, _color: Color) -> &str;
}

impl Colorize for str {
    fn red(&self) -> &str {
        self
    }
    fn green(&self) -> &str {
        self
    }
    fn blue(&self) -> &str {
        self
    }
    fn cyan(&self) -> &str {
        self
    }
    fn yellow(&self) -> &str {
        self
    }
    fn white(&self) -> &str {
        self
    }
    fn bold(&self) -> &str {
        self
    }
    fn dimmed(&self) -> &str {
        self
    }
    fn italic(&self) -> &str {
        self
    }
    fn underline(&self) -> &str {
        self
    }
    fn bright_black(&self) -> &str {
        self
    }
    fn bright_white(&self) -> &str {
        self
    }
    fn color(&self, _color: Color) -> &str {
        self
    }
}
