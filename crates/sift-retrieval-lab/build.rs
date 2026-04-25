fn main() {
    // Cargo sets TARGET in the build-script env. Forward it to the crate
    // via a `TARGET_TRIPLE` env! lookup — used by the reproducibility
    // envelope so a Linux CI baseline is never compared to an M-series dev
    // baseline by accident.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=TARGET_TRIPLE={target}");
    println!("cargo:rerun-if-env-changed=TARGET");
}
