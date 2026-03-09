# Windows packaging

This directory contains Windows-specific packaging files.

## Manifest

`Manifest.xml` is a Windows application manifest that:

- Enables **long path support** (paths > 260 characters) via `longPathAware`
- Sets **UTF-8** as the active code page
- Declares compatibility with Windows 7 through Windows 11

The manifest is embedded into the final binary via linker arguments in `build.rs`.
This currently only applies to MSVC builds.

## Embedding the manifest

Add this to the workspace root `build.rs`:

```rust
fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_manifest_file("pkg/windows/Manifest.xml");
        res.compile().unwrap();
    }
}
```

Or use linker arguments directly:

```rust
fn main() {
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        println!("cargo:rustc-link-arg-bin=sift=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-bin=sift=/MANIFESTINPUT:{}",
            std::path::Path::new("pkg/windows/Manifest.xml")
                .canonicalize()
                .unwrap()
                .display()
        );
    }
}
```
