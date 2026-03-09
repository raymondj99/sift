use crate::traits::Parser;
use sift_core::{ContentType, ParsedDocument, SiftResult};
use std::collections::HashMap;

/// Parser for source code files. Extracts the raw text with language metadata.
/// Tree-sitter integration can be added for AST-aware parsing.
pub struct CodeParser;

impl CodeParser {
    const CODE_MIMES: &[&str] = &[
        "text/x-rust",
        "text/x-python",
        "text/javascript",
        "text/typescript",
        "text/x-go",
        "text/x-c",
        "text/x-c++",
        "text/x-java",
        "text/x-ruby",
        "text/x-shellscript",
        "text/css",
        "text/x-sql",
        "text/x-r",
        "text/x-swift",
        "text/x-kotlin",
        "text/x-scala",
        "text/x-zig",
        "text/x-lua",
        "text/x-perl",
        "text/x-elixir",
        "text/x-erlang",
        "text/x-haskell",
        "text/x-ocaml",
        "text/x-protobuf",
        "text/x-terraform",
        "text/x-dockerfile",
    ];

    const CODE_EXTENSIONS: &[&str] = &[
        "rs",
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "go",
        "c",
        "h",
        "cpp",
        "hpp",
        "cc",
        "cxx",
        "java",
        "rb",
        "sh",
        "bash",
        "zsh",
        "fish",
        "css",
        "sql",
        "r",
        "swift",
        "kt",
        "kts",
        "scala",
        "zig",
        "lua",
        "pl",
        "pm",
        "ex",
        "exs",
        "erl",
        "hrl",
        "hs",
        "ml",
        "mli",
        "proto",
        "tf",
        "tfvars",
        "dockerfile",
        "makefile",
        "cmake",
    ];

    fn extension_to_language(ext: &str) -> Option<&'static str> {
        Some(match ext {
            "rs" => "rust",
            "py" => "python",
            "js" | "jsx" => "javascript",
            "ts" | "tsx" => "typescript",
            "go" => "go",
            "c" | "h" => "c",
            "cpp" | "hpp" | "cc" | "cxx" => "cpp",
            "java" => "java",
            "rb" => "ruby",
            "sh" | "bash" => "bash",
            "zsh" => "zsh",
            "fish" => "fish",
            "css" => "css",
            "sql" => "sql",
            "r" => "r",
            "swift" => "swift",
            "kt" | "kts" => "kotlin",
            "scala" => "scala",
            "zig" => "zig",
            "lua" => "lua",
            "pl" | "pm" => "perl",
            "ex" | "exs" => "elixir",
            "erl" | "hrl" => "erlang",
            "hs" => "haskell",
            "ml" | "mli" => "ocaml",
            "proto" => "protobuf",
            "tf" | "tfvars" => "terraform",
            _ => return None,
        })
    }
}

impl Parser for CodeParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            if Self::CODE_MIMES.contains(&mime) {
                return true;
            }
        }
        if let Some(ext) = extension {
            let ext_lower = ext.to_lowercase();
            if Self::CODE_EXTENSIONS.contains(&ext_lower.as_str()) {
                return true;
            }
        }
        false
    }

    fn parse(
        &self,
        content: &[u8],
        _mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let text = String::from_utf8_lossy(content).into_owned();

        let language = extension
            .and_then(Self::extension_to_language)
            .map(String::from);

        let mut metadata = HashMap::new();
        // Count lines as basic metadata
        let line_count = text.lines().count();
        metadata.insert("lines".to_string(), line_count.to_string());

        Ok(ParsedDocument {
            text,
            title: None,
            language,
            content_type: ContentType::Code,
            metadata,
        })
    }

    fn name(&self) -> &str {
        "code"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_code() {
        let parser = CodeParser;
        let code = b"fn main() {\n    println!(\"hello\");\n}\n";
        let doc = parser.parse(code, None, Some("rs")).unwrap();
        assert_eq!(doc.content_type, ContentType::Code);
        assert_eq!(doc.language.as_deref(), Some("rust"));
        assert_eq!(doc.metadata.get("lines").unwrap(), "3");
    }

    #[test]
    fn test_can_parse_code() {
        let parser = CodeParser;
        assert!(parser.can_parse(Some("text/x-python"), None));
        assert!(parser.can_parse(None, Some("rs")));
        assert!(parser.can_parse(None, Some("py")));
        assert!(!parser.can_parse(Some("application/pdf"), None));
    }
}
