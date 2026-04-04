#![no_main]

use libfuzzer_sys::fuzz_target;
use sift_parsers::ParserRegistry;

fuzz_target!(|data: &[u8]| {
    let registry = ParserRegistry::new();

    // Fuzz HTML parsing via the registry with explicit MIME type and extension.
    let _ = registry.parse(data, Some("text/html"), Some("html"));

    // Also exercise the path where only the extension is provided.
    let _ = registry.parse(data, None, Some("html"));
});
