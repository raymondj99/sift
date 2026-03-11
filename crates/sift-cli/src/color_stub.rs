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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colorize_methods_are_identity() {
        let s = "hello world";
        assert_eq!(s.red(), "hello world");
        assert_eq!(s.green(), "hello world");
        assert_eq!(s.blue(), "hello world");
        assert_eq!(s.cyan(), "hello world");
        assert_eq!(s.yellow(), "hello world");
        assert_eq!(s.white(), "hello world");
        assert_eq!(s.bold(), "hello world");
        assert_eq!(s.dimmed(), "hello world");
        assert_eq!(s.italic(), "hello world");
        assert_eq!(s.underline(), "hello world");
        assert_eq!(s.bright_black(), "hello world");
        assert_eq!(s.bright_white(), "hello world");
        assert_eq!(s.color(Color::Red), "hello world");
        assert_eq!(s.color(Color::Green), "hello world");
        assert_eq!(s.color(Color::Yellow), "hello world");
        assert_eq!(s.color(Color::Blue), "hello world");
        assert_eq!(s.color(Color::Cyan), "hello world");
        assert_eq!(s.color(Color::White), "hello world");
    }

    #[test]
    fn test_colorize_empty_string() {
        let s = "";
        assert_eq!(s.bold(), "");
        assert_eq!(s.dimmed(), "");
        assert_eq!(s.red(), "");
    }

    #[test]
    fn test_colorize_chaining() {
        // Chaining should work since each method returns &str
        let s = "test";
        assert_eq!(s.bold().red().cyan(), "test");
    }
}
