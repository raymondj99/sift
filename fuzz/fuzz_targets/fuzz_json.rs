#![no_main]

use libfuzzer_sys::fuzz_target;
use sift_parsers::ParserRegistry;

fuzz_target!(|data: &[u8]| {
    let registry = ParserRegistry::new();

    // Fuzz JSON parsing via the registry with explicit MIME type and extension.
    let _ = registry.parse(data, Some("application/json"), Some("json"));

    // Also exercise the path where only the extension is provided.
    let _ = registry.parse(data, None, Some("json"));
});
