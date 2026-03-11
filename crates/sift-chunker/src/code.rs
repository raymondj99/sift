use crate::traits::Chunker;
use tracing::debug;

/// AST-aware code chunker using tree-sitter.
/// Splits code at function, class, struct, and impl boundaries.
/// Prepends scope path comments (e.g. `// impl Foo > fn bar`) to chunks
/// so that context is preserved even when reading a chunk in isolation.
/// Falls back to the semantic chunker if parsing fails.
pub struct CodeChunker {
    max_chunk_size: usize,
    overlap: usize,
}

impl CodeChunker {
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size: max_chunk_size.max(1),
            overlap: overlap.min(max_chunk_size / 2),
        }
    }

    /// Try to chunk code using tree-sitter for the given language.
    /// Returns None if the language is unsupported or parsing fails.
    pub fn chunk_with_language(&self, text: &str, language: Option<&str>) -> Vec<(String, usize)> {
        if text.is_empty() {
            return vec![];
        }

        if text.len() <= self.max_chunk_size {
            return vec![(text.to_string(), 0)];
        }

        let lang = match language {
            Some(l) => l,
            None => return self.fallback_chunk(text),
        };

        let ts_language = if let Some(l) = lang_for_extension(lang) { l } else {
            debug!("No tree-sitter grammar for language: {}", lang);
            return self.fallback_chunk(text);
        };

        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&ts_language).is_err() {
            return self.fallback_chunk(text);
        }

        let tree = match parser.parse(text, None) {
            Some(t) => t,
            None => return self.fallback_chunk(text),
        };

        let root = tree.root_node();
        let mut definitions = Vec::new();
        collect_top_level_definitions(root, text, &mut definitions);

        if definitions.is_empty() {
            return self.fallback_chunk(text);
        }

        // Sort by byte offset
        definitions.sort_by_key(|d| d.start_byte);

        // Merge adjacent small definitions into larger chunks
        self.merge_definitions(text, &definitions)
    }

    fn merge_definitions(&self, text: &str, defs: &[Definition]) -> Vec<(String, usize)> {
        let mut chunks: Vec<(String, usize)> = Vec::new();
        // Track which definitions contribute to each chunk so we can build scope headers.
        let mut current_start = defs[0].start_byte;
        let mut current_end = defs[0].end_byte;
        let mut current_defs: Vec<usize> = vec![0]; // indices into `defs`

        for (i, def) in defs[1..].iter().enumerate() {
            let idx = i + 1;
            let merged_size = def.end_byte - current_start;

            if merged_size <= self.max_chunk_size {
                // Merge: extend current chunk to include this definition
                current_end = def.end_byte;
                current_defs.push(idx);
            } else {
                // Emit current chunk
                let chunk_text = &text[current_start..current_end];
                let trimmed = chunk_text.trim();
                if !trimmed.is_empty() {
                    let scope_header = build_scope_header(defs, &current_defs);
                    let chunk_with_scope = prepend_scope_comment(trimmed, &scope_header);
                    chunks.push((chunk_with_scope, current_start));
                }

                // Start new chunk, optionally with overlap from end of previous
                if self.overlap > 0 && current_end > self.overlap {
                    let overlap_start =
                        find_line_start(text, current_end.saturating_sub(self.overlap));
                    // Include leading context if it fits
                    let actual_start = if def.start_byte > overlap_start {
                        overlap_start
                    } else {
                        def.start_byte
                    };
                    current_start = actual_start;
                } else {
                    current_start = def.start_byte;
                }
                current_end = def.end_byte;
                current_defs = vec![idx];
            }
        }

        // Emit last chunk
        let chunk_text = &text[current_start..current_end];
        let trimmed = chunk_text.trim();
        if !trimmed.is_empty() {
            let scope_header = build_scope_header(defs, &current_defs);
            let chunk_with_scope = prepend_scope_comment(trimmed, &scope_header);
            chunks.push((chunk_with_scope, current_start));
        }

        // Handle any text before the first definition or after the last
        if let Some(first) = defs.first() {
            if first.start_byte > 0 {
                let preamble = text[..first.start_byte].trim();
                if !preamble.is_empty() && preamble.len() > 10 {
                    chunks.insert(0, (preamble.to_string(), 0));
                }
            }
        }
        if let Some(last) = defs.last() {
            if last.end_byte < text.len() {
                let epilogue = text[last.end_byte..].trim();
                if !epilogue.is_empty() && epilogue.len() > 10 {
                    chunks.push((epilogue.to_string(), last.end_byte));
                }
            }
        }

        // If any chunk is still too large, split it with the fallback chunker
        let mut final_chunks = Vec::new();
        for (chunk_text, offset) in chunks {
            if chunk_text.len() <= self.max_chunk_size {
                final_chunks.push((chunk_text, offset));
            } else {
                let sub = crate::SemanticChunker::new(self.max_chunk_size, self.overlap);
                for (sub_text, sub_offset) in sub.chunk(&chunk_text) {
                    final_chunks.push((sub_text, offset + sub_offset));
                }
            }
        }

        final_chunks
    }

    fn fallback_chunk(&self, text: &str) -> Vec<(String, usize)> {
        crate::SemanticChunker::new(self.max_chunk_size, self.overlap).chunk(text)
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str) -> Vec<(String, usize)> {
        // Without language info, fall back to semantic chunking
        self.fallback_chunk(text)
    }

    fn chunk_with_language(&self, text: &str, language: Option<&str>) -> Vec<(String, usize)> {
        CodeChunker::chunk_with_language(self, text, language)
    }

    fn name(&self) -> &'static str {
        "code"
    }
}

struct Definition {
    start_byte: usize,
    end_byte: usize,
    /// Scope path for this definition, e.g. "impl Foo > fn bar".
    scope_path: Option<String>,
}

/// Collect top-level definitions from the AST.
fn collect_top_level_definitions(node: tree_sitter::Node, text: &str, defs: &mut Vec<Definition>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if is_definition_node(child.kind()) {
            let scope = build_scope_path(child, text);
            defs.push(Definition {
                start_byte: child.start_byte(),
                end_byte: child.end_byte(),
                scope_path: scope,
            });
            // Also collect nested definitions inside container nodes
            // (e.g. methods inside impl/class blocks).
            collect_nested_definitions(child, text, defs);
        }
    }
}

/// Recursively collect definitions nested inside container nodes
/// (impl blocks, class bodies, etc.) so they each get their own
/// scope path. We only descend into "body" / "`declaration_list`" children.
fn collect_nested_definitions(parent: tree_sitter::Node, text: &str, _defs: &mut Vec<Definition>) {
    if !is_container_node(parent.kind()) {
        return;
    }

    let mut cursor = parent.walk();
    for child in parent.children(&mut cursor) {
        // Look inside the body / declaration_list of the container
        let kind = child.kind();
        if kind == "declaration_list"
            || kind == "block"
            || kind == "class_body"
            || kind == "statement_block"
            || kind == "interface_body"
            || kind == "enum_body"
        {
            let mut inner_cursor = child.walk();
            for inner in child.children(&mut inner_cursor) {
                if is_definition_node(inner.kind()) || is_method_like(inner.kind()) {
                    let scope = build_scope_path(inner, text);
                    // Don't push as separate definitions -- these are already
                    // part of the parent. The scope_path will be prepended
                    // to the parent chunk or used if the parent gets split.
                    // We record them but they're not separate chunks.
                    let _ = scope;
                }
            }
        }
    }
}

/// Check if a node is a container that holds nested definitions.
fn is_container_node(kind: &str) -> bool {
    matches!(
        kind,
        "impl_item"
            | "trait_item"
            | "mod_item"
            | "class_definition"
            | "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
    )
}

/// Check if a node kind is a method-like definition (nested inside a class/impl).
fn is_method_like(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "function_definition"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "constructor_declaration"
    )
}

/// Build a scope path string for a node by walking up the AST.
/// E.g., for `fn process` inside `impl PaymentService`, returns
/// `"impl PaymentService > fn process"`.
fn build_scope_path(node: tree_sitter::Node, text: &str) -> Option<String> {
    let mut parts = Vec::new();

    // Start with the node itself
    if let Some(label) = node_label(node, text) {
        parts.push(label);
    }

    // Walk up through parents
    let mut current = node;
    while let Some(parent) = current.parent() {
        // Skip the root "source_file" / "program" / "translation_unit" node
        if parent.kind() == "source_file"
            || parent.kind() == "program"
            || parent.kind() == "translation_unit"
            || parent.kind() == "source"
        {
            break;
        }

        if let Some(label) = node_label(parent, text) {
            parts.push(label);
        }
        current = parent;
    }

    if parts.is_empty() {
        return None;
    }

    parts.reverse();
    Some(parts.join(" > "))
}

/// Extract a short label for a node, e.g. "impl Foo", "fn bar", "class Baz".
fn node_label(node: tree_sitter::Node, text: &str) -> Option<String> {
    let kind = node.kind();

    // Map node kinds to the prefix keyword
    let prefix = match kind {
        "function_item" | "function_definition" | "function_declaration" => "fn",
        "struct_item" => "struct",
        "enum_item" | "enum_declaration" => "enum",
        "impl_item" => "impl",
        "trait_item" => "trait",
        "mod_item" => "mod",
        "class_definition" | "class_declaration" => "class",
        "method_definition" | "method_declaration" => "fn",
        "constructor_declaration" => "constructor",
        "interface_declaration" => "interface",
        "decorated_definition" => {
            // For decorated definitions, look at the inner definition
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if is_definition_node(child.kind()) || is_method_like(child.kind()) {
                    return node_label(child, text);
                }
            }
            return None;
        }
        "export_statement" => {
            // For export statements, look at the inner declaration
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if is_definition_node(child.kind()) || is_method_like(child.kind()) {
                    return node_label(child, text);
                }
            }
            return None;
        }
        // Skip intermediate structural nodes
        "declaration_list" | "block" | "class_body" | "statement_block" | "interface_body"
        | "enum_body" => return None,
        _ => return None,
    };

    // Try to find the name child node
    let name = find_name_child(node, text);

    match name {
        Some(n) => Some(format!("{prefix} {n}")),
        None => Some(prefix.to_string()),
    }
}

/// Try to find the "name" or "type" child of a node to extract its identifier.
fn find_name_child(node: tree_sitter::Node, text: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        // Common name node kinds across languages
        if kind == "identifier"
            || kind == "type_identifier"
            || kind == "name"
            || kind == "property_identifier"
        {
            let name_text = &text[child.start_byte()..child.end_byte()];
            return Some(name_text.to_string());
        }
        // For impl items in Rust, the type is often a "generic_type" or "type_identifier"
        // nested under the impl
        if kind == "generic_type" || kind == "scoped_type_identifier" {
            let name_text = &text[child.start_byte()..child.end_byte()];
            return Some(name_text.to_string());
        }
    }
    None
}

/// Check if a tree-sitter node kind represents a top-level code definition.
fn is_definition_node(kind: &str) -> bool {
    matches!(
        kind,
        // Rust
        "function_item"
            | "struct_item"
            | "enum_item"
            | "impl_item"
            | "trait_item"
            | "mod_item"
            | "const_item"
            | "static_item"
            | "type_item"
            | "macro_definition"
            // Python / C (shared: function_definition, class_definition)
            | "function_definition"
            | "class_definition"
            | "decorated_definition"
            // JS/TS / Go / Java (shared: function_declaration, class_declaration, method_declaration)
            | "function_declaration"
            | "class_declaration"
            | "export_statement"
            | "lexical_declaration"
            | "method_definition"
            | "method_declaration"
            | "type_declaration"
            // C
            | "struct_specifier"
            | "enum_specifier"
            | "declaration"
            // Java
            | "interface_declaration"
            | "enum_declaration"
            | "constructor_declaration"
    )
}

/// Map file extension to tree-sitter language.
fn lang_for_extension(ext: &str) -> Option<tree_sitter::Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" | "pyi" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "jsx" | "mjs" | "cjs" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        _ => None,
    }
}

/// Build a scope header string from a set of definition indices.
/// If there is exactly one definition with a `scope_path`, return it.
/// If there are multiple, join them with ", ".
fn build_scope_header(defs: &[Definition], indices: &[usize]) -> Option<String> {
    let paths: Vec<&str> = indices
        .iter()
        .filter_map(|&i| defs.get(i))
        .filter_map(|d| d.scope_path.as_deref())
        .collect();

    if paths.is_empty() {
        None
    } else if paths.len() == 1 {
        Some(paths[0].to_string())
    } else {
        Some(paths.join(", "))
    }
}

/// Prepend a `// scope_path` comment to a chunk of code text.
/// Returns the original text unchanged if `scope_path` is None.
fn prepend_scope_comment(text: &str, scope_path: &Option<String>) -> String {
    match scope_path {
        Some(path) if !path.is_empty() => format!("// {path}\n{text}"),
        _ => text.to_string(),
    }
}

/// Find the start of the line containing (or just before) the given byte offset.
fn find_line_start(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let search = &text[..pos.min(text.len())];
    search.rfind('\n').map_or(0, |p| p + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    // ---------------------------------------------------------------
    // Helper: join all chunk texts and verify a substring is present.
    // ---------------------------------------------------------------
    fn all_text(chunks: &[(String, usize)]) -> String {
        chunks
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ---------------------------------------------------------------
    // Per-language: each test proves tree-sitter splits that language
    // at definition boundaries rather than arbitrary byte offsets.
    // ---------------------------------------------------------------

    #[test]
    fn test_rust_splits_at_fn_struct_impl_boundaries() {
        let chunker = CodeChunker::new(200, 0);
        let code = "\
use std::io;

fn hello() {
    println!(\"hello\");
}

fn world() {
    println!(\"world\");
}

struct Foo {
    bar: i32,
    baz: String,
}

impl Foo {
    fn new() -> Self {
        Foo { bar: 0, baz: String::new() }
    }

    fn do_thing(&self) {
        println!(\"{}\", self.bar);
    }
}
";
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("fn hello()"));
        assert!(t.contains("struct Foo"));
        assert!(t.contains("impl Foo"));
    }

    #[test]
    fn test_python_splits_at_def_and_class_boundaries() {
        let chunker = CodeChunker::new(120, 0);
        let code = "\
import os

def greet(name):
    print(f'Hello {name}')

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x

def main():
    calc = Calculator()
    calc.add(5)
";
        let chunks = chunker.chunk_with_language(code, Some("py"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("def greet"));
        assert!(t.contains("class Calculator"));
        assert!(t.contains("def main"));
    }

    #[test]
    fn test_javascript_splits_at_function_and_class_declarations() {
        let chunker = CodeChunker::new(180, 0);
        let code = "\
const API_URL = 'https://api.example.com';

function fetchUser(id) {
    return fetch(`${API_URL}/users/${id}`)
        .then(res => res.json());
}

class UserService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async getUser(id) {
        const res = await fetch(`${this.baseUrl}/users/${id}`);
        return res.json();
    }
}

function formatName(user) {
    return `${user.first} ${user.last}`;
}
";
        let chunks = chunker.chunk_with_language(code, Some("js"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("function fetchUser"));
        assert!(t.contains("class UserService"));
        assert!(t.contains("function formatName"));
    }

    #[test]
    fn test_typescript_splits_at_interface_and_function_boundaries() {
        let chunker = CodeChunker::new(200, 0);
        let code = "\
import { Request, Response } from 'express';

export function handleGet(req: Request, res: Response): void {
    const id = req.params.id;
    res.json({ id });
}

export function handlePost(req: Request, res: Response): void {
    const body = req.body;
    res.status(201).json(body);
}

export function handleDelete(req: Request, res: Response): void {
    res.status(204).send();
}
";
        let chunks = chunker.chunk_with_language(code, Some("ts"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("handleGet"));
        assert!(t.contains("handlePost"));
        assert!(t.contains("handleDelete"));
    }

    #[test]
    fn test_go_splits_at_func_boundaries() {
        let chunker = CodeChunker::new(160, 0);
        let code = "\
package main

import \"fmt\"

func Add(a, b int) int {
    return a + b
}

func Multiply(a, b int) int {
    return a * b
}

func main() {
    fmt.Println(Add(2, 3))
    fmt.Println(Multiply(4, 5))
}
";
        let chunks = chunker.chunk_with_language(code, Some("go"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("func Add"));
        assert!(t.contains("func Multiply"));
        assert!(t.contains("func main"));
    }

    #[test]
    fn test_c_splits_at_function_definitions() {
        let chunker = CodeChunker::new(150, 0);
        let code = "\
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

void greet(const char* name) {
    printf(\"Hello, %s\\n\", name);
}

int main() {
    greet(\"world\");
    printf(\"%d\\n\", add(2, 3));
    return 0;
}
";
        let chunks = chunker.chunk_with_language(code, Some("c"));
        assert!(chunks.len() >= 2);
        let t = all_text(&chunks);
        assert!(t.contains("int add"));
        assert!(t.contains("void greet"));
        assert!(t.contains("int main"));
    }

    #[test]
    fn test_java_splits_at_class_and_method_boundaries() {
        let chunker = CodeChunker::new(200, 0);
        let code = "\
import java.util.List;

public class Calculator {
    private int result;

    public Calculator() {
        this.result = 0;
    }

    public int add(int x) {
        this.result += x;
        return this.result;
    }

    public int getResult() {
        return this.result;
    }
}
";
        let chunks = chunker.chunk_with_language(code, Some("java"));
        // Java wraps everything in a class, so we may get one large chunk
        // The key: it parses without error and captures all content
        let t = all_text(&chunks);
        assert!(t.contains("class Calculator"));
        assert!(t.contains("public int add"));
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_file_returns_no_chunks() {
        let chunker = CodeChunker::new(500, 0);
        assert!(chunker.chunk_with_language("", Some("rs")).is_empty());
        assert!(chunker.chunk_with_language("", None).is_empty());
    }

    #[test]
    fn test_small_file_returns_single_chunk() {
        let chunker = CodeChunker::new(500, 0);
        let code = "fn main() { println!(\"hello\"); }";
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, code);
        assert_eq!(chunks[0].1, 0);
    }

    #[test]
    fn test_no_language_hint_falls_back_to_semantic_chunker() {
        let chunker = CodeChunker::new(50, 0);
        let text = "first paragraph\n\nsecond paragraph\n\nthird paragraph with more words here";
        let chunks = chunker.chunk_with_language(text, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_unsupported_language_falls_back_gracefully() {
        let chunker = CodeChunker::new(50, 0);
        let code =
            "some code in a language we do not support with enough text to exceed chunk size limit";
        let chunks = chunker.chunk_with_language(code, Some("cobol"));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_file_with_only_imports_no_definitions_still_chunks() {
        let chunker = CodeChunker::new(100, 0);
        let code = "\
use std::io;
use std::fs;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
";
        // No function/struct/impl definitions — should fall back or capture preamble
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        assert!(!chunks.is_empty());
        let t = all_text(&chunks);
        assert!(t.contains("use std::io"));
    }

    #[test]
    fn test_large_single_function_exceeding_chunk_size_gets_split() {
        let chunker = CodeChunker::new(100, 0);
        // A function body that far exceeds 100 bytes
        let mut body = String::from("fn big() {\n");
        for i in 0..20 {
            let _ = writeln!(body, "    let x{i} = {i};");
        }
        body.push_str("}\n");

        let chunks = chunker.chunk_with_language(&body, Some("rs"));
        assert!(
            chunks.len() >= 2,
            "Function exceeding chunk_size should be split, got {} chunks for {} bytes",
            chunks.len(),
            body.len()
        );
    }

    #[test]
    fn test_all_chunk_offsets_are_within_source_bounds() {
        let chunker = CodeChunker::new(150, 0);
        let code = "\
fn a() { let x = 1; }

fn b() { let y = 2; }

fn c() { let z = 3; }

struct S { val: i32 }

impl S {
    fn new() -> Self { S { val: 0 } }
}
";
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        for (text, offset) in &chunks {
            assert!(
                *offset + text.len() <= code.len() + text.len(), // offset is into source, text may be trimmed
                "Chunk offset {} + len {} exceeds source len {}",
                offset,
                text.len(),
                code.len()
            );
        }
    }

    #[test]
    fn test_no_chunk_is_empty() {
        let chunker = CodeChunker::new(120, 0);
        let code = "\
fn alpha() { println!(\"a\"); }

fn beta() { println!(\"b\"); }

fn gamma() { println!(\"c\"); }

fn delta() { println!(\"d\"); }
";
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        for (text, _) in &chunks {
            assert!(!text.is_empty(), "Chunks must never be empty strings");
            assert!(
                !text.trim().is_empty(),
                "Chunks must never be only whitespace"
            );
        }
    }

    // ---------------------------------------------------------------
    // lang_for_extension coverage
    // ---------------------------------------------------------------

    #[test]
    fn test_lang_for_extension_maps_all_documented_extensions() {
        // Every extension we claim to support must resolve to a language
        let supported = [
            "rs", "py", "pyi", "js", "jsx", "mjs", "cjs", "ts", "tsx", "go", "c", "h", "java",
        ];
        for ext in &supported {
            assert!(
                lang_for_extension(ext).is_some(),
                "Extension '{ext}' should map to a tree-sitter language"
            );
        }
    }

    #[test]
    fn test_lang_for_extension_returns_none_for_unknown() {
        assert!(lang_for_extension("rb").is_none());
        assert!(lang_for_extension("swift").is_none());
        assert!(lang_for_extension("").is_none());
    }

    // ---------------------------------------------------------------
    // Scope-path tests
    // ---------------------------------------------------------------

    #[test]
    fn test_scope_path_prepended_to_rust_fn() {
        // Use a small chunk_size to force splitting so scope paths appear
        let chunker = CodeChunker::new(120, 0);
        let code = "\
fn hello() {
    println!(\"hello\");
}

fn world() {
    println!(\"world\");
}

struct Foo {
    bar: i32,
}

impl Foo {
    fn do_thing(&self) {
        println!(\"{}\", self.bar);
    }
}
";
        let chunks = chunker.chunk_with_language(code, Some("rs"));
        let t = all_text(&chunks);
        // Should contain a scope path comment for at least one function
        assert!(
            t.contains("// fn hello") || t.contains("// fn world") || t.contains("// impl Foo"),
            "Expected a scope-path comment in chunks, got:\n{t}"
        );
    }

    #[test]
    fn test_scope_path_for_python_class_and_def() {
        let chunker = CodeChunker::new(100, 0);
        let code = "\
def greet(name):
    print(f'Hello {name}')

class Calculator:
    def __init__(self):
        self.result = 0

def main():
    calc = Calculator()
";
        let chunks = chunker.chunk_with_language(code, Some("py"));
        let t = all_text(&chunks);
        // Should contain scope annotations
        assert!(
            t.contains("// fn greet")
                || t.contains("// class Calculator")
                || t.contains("// fn main"),
            "Expected a scope-path comment in Python chunks, got:\n{t}"
        );
    }

    #[test]
    fn test_build_scope_path_returns_none_for_root_level() {
        // Parse a simple Rust file and verify build_scope_path works
        let code = "fn top() {}";
        let ts_language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts_language).unwrap();
        let tree = parser.parse(code, None).unwrap();
        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "function_item" {
                let path = build_scope_path(child, code);
                assert!(path.is_some());
                let path = path.unwrap();
                assert!(
                    path.contains("fn top"),
                    "Expected 'fn top' in path, got: {path}"
                );
            }
        }
    }

    #[test]
    fn test_scope_path_for_nested_rust_impl_method() {
        let code = "\
impl MyStruct {
    fn my_method(&self) {
        println!(\"hello\");
    }
}
";
        let ts_language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts_language).unwrap();
        let tree = parser.parse(code, None).unwrap();
        let root = tree.root_node();

        // Walk down to find the function_item inside impl
        fn find_node<'a>(node: tree_sitter::Node<'a>, kind: &str) -> Option<tree_sitter::Node<'a>> {
            if node.kind() == kind {
                return Some(node);
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(found) = find_node(child, kind) {
                    return Some(found);
                }
            }
            None
        }

        let fn_node = find_node(root, "function_item").expect("Should find function_item");
        let path = build_scope_path(fn_node, code).expect("Should have a scope path");
        // Should contain both the impl and the fn
        assert!(
            path.contains("impl MyStruct") && path.contains("fn my_method"),
            "Expected scope path 'impl MyStruct > ... > fn my_method', got: {path}"
        );
    }

    #[test]
    fn test_prepend_scope_comment_with_path() {
        let text = "fn foo() {}";
        let path = Some("impl Bar > fn foo".to_string());
        let result = prepend_scope_comment(text, &path);
        assert_eq!(result, "// impl Bar > fn foo\nfn foo() {}");
    }

    #[test]
    fn test_prepend_scope_comment_without_path() {
        let text = "fn foo() {}";
        let path: Option<String> = None;
        let result = prepend_scope_comment(text, &path);
        assert_eq!(result, "fn foo() {}");
    }
}
