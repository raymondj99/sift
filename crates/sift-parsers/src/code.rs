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

    fn name(&self) -> &'static str {
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

    #[test]
    fn test_extension_to_language_all() {
        // Test all extension -> language mappings
        assert_eq!(CodeParser::extension_to_language("rs"), Some("rust"));
        assert_eq!(CodeParser::extension_to_language("py"), Some("python"));
        assert_eq!(CodeParser::extension_to_language("js"), Some("javascript"));
        assert_eq!(CodeParser::extension_to_language("jsx"), Some("javascript"));
        assert_eq!(CodeParser::extension_to_language("ts"), Some("typescript"));
        assert_eq!(CodeParser::extension_to_language("tsx"), Some("typescript"));
        assert_eq!(CodeParser::extension_to_language("go"), Some("go"));
        assert_eq!(CodeParser::extension_to_language("c"), Some("c"));
        assert_eq!(CodeParser::extension_to_language("h"), Some("c"));
        assert_eq!(CodeParser::extension_to_language("cpp"), Some("cpp"));
        assert_eq!(CodeParser::extension_to_language("hpp"), Some("cpp"));
        assert_eq!(CodeParser::extension_to_language("cc"), Some("cpp"));
        assert_eq!(CodeParser::extension_to_language("cxx"), Some("cpp"));
        assert_eq!(CodeParser::extension_to_language("java"), Some("java"));
        assert_eq!(CodeParser::extension_to_language("rb"), Some("ruby"));
        assert_eq!(CodeParser::extension_to_language("sh"), Some("bash"));
        assert_eq!(CodeParser::extension_to_language("bash"), Some("bash"));
        assert_eq!(CodeParser::extension_to_language("zsh"), Some("zsh"));
        assert_eq!(CodeParser::extension_to_language("fish"), Some("fish"));
        assert_eq!(CodeParser::extension_to_language("css"), Some("css"));
        assert_eq!(CodeParser::extension_to_language("sql"), Some("sql"));
        assert_eq!(CodeParser::extension_to_language("r"), Some("r"));
        assert_eq!(CodeParser::extension_to_language("swift"), Some("swift"));
        assert_eq!(CodeParser::extension_to_language("kt"), Some("kotlin"));
        assert_eq!(CodeParser::extension_to_language("kts"), Some("kotlin"));
        assert_eq!(CodeParser::extension_to_language("scala"), Some("scala"));
        assert_eq!(CodeParser::extension_to_language("zig"), Some("zig"));
        assert_eq!(CodeParser::extension_to_language("lua"), Some("lua"));
        assert_eq!(CodeParser::extension_to_language("pl"), Some("perl"));
        assert_eq!(CodeParser::extension_to_language("pm"), Some("perl"));
        assert_eq!(CodeParser::extension_to_language("ex"), Some("elixir"));
        assert_eq!(CodeParser::extension_to_language("exs"), Some("elixir"));
        assert_eq!(CodeParser::extension_to_language("erl"), Some("erlang"));
        assert_eq!(CodeParser::extension_to_language("hrl"), Some("erlang"));
        assert_eq!(CodeParser::extension_to_language("hs"), Some("haskell"));
        assert_eq!(CodeParser::extension_to_language("ml"), Some("ocaml"));
        assert_eq!(CodeParser::extension_to_language("mli"), Some("ocaml"));
        assert_eq!(CodeParser::extension_to_language("proto"), Some("protobuf"));
        assert_eq!(CodeParser::extension_to_language("tf"), Some("terraform"));
        assert_eq!(
            CodeParser::extension_to_language("tfvars"),
            Some("terraform")
        );
    }

    #[test]
    fn test_extension_to_language_unknown() {
        assert_eq!(CodeParser::extension_to_language("xyz"), None);
        assert_eq!(CodeParser::extension_to_language("pdf"), None);
        assert_eq!(CodeParser::extension_to_language(""), None);
    }

    #[test]
    fn test_parse_python_code() {
        let parser = CodeParser;
        let code = b"def hello():\n    print('world')\n";
        let doc = parser.parse(code, None, Some("py")).unwrap();
        assert_eq!(doc.content_type, ContentType::Code);
        assert_eq!(doc.language.as_deref(), Some("python"));
        assert!(doc.text.contains("def hello()"));
    }

    #[test]
    fn test_parse_javascript_code() {
        let parser = CodeParser;
        let code = b"function greet() {\n  console.log('hi');\n}\n";
        let doc = parser.parse(code, None, Some("js")).unwrap();
        assert_eq!(doc.language.as_deref(), Some("javascript"));
    }

    #[test]
    fn test_parse_unknown_extension_code() {
        let parser = CodeParser;
        let code = b"some content\nline two\n";
        let doc = parser.parse(code, None, None).unwrap();
        assert!(doc.language.is_none());
    }

    #[test]
    fn test_parse_line_count() {
        let parser = CodeParser;
        let code = b"line1\nline2\nline3\nline4\nline5\n";
        let doc = parser.parse(code, None, Some("rs")).unwrap();
        assert_eq!(doc.metadata.get("lines").unwrap(), "5");
    }

    #[test]
    fn test_parser_name() {
        let parser = CodeParser;
        assert_eq!(parser.name(), "code");
    }

    #[test]
    fn test_can_parse_all_extensions() {
        let parser = CodeParser;
        let all_exts = [
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
        for ext in &all_exts {
            assert!(
                parser.can_parse(None, Some(ext)),
                "Expected can_parse to be true for extension: {}",
                ext
            );
        }
    }

    #[test]
    fn test_can_parse_all_mimes() {
        let parser = CodeParser;
        let all_mimes = [
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
        for mime in &all_mimes {
            assert!(
                parser.can_parse(Some(mime), None),
                "Expected can_parse to be true for mime: {}",
                mime
            );
        }
    }

    #[test]
    fn test_can_parse_case_insensitive_extension() {
        let parser = CodeParser;
        assert!(parser.can_parse(None, Some("RS")));
        assert!(parser.can_parse(None, Some("Py")));
    }

    #[test]
    fn test_cannot_parse_non_code() {
        let parser = CodeParser;
        assert!(!parser.can_parse(None, Some("pdf")));
        assert!(!parser.can_parse(None, Some("xlsx")));
        assert!(!parser.can_parse(Some("image/png"), None));
        assert!(!parser.can_parse(None, None));
    }
}
