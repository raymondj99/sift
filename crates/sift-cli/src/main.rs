#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(not(feature = "fancy"))]
mod color_stub;
mod commands;
mod output;
mod pipeline;

#[cfg(feature = "completions")]
use clap::CommandFactory;
use clap::{Parser, Subcommand};
use tracing_subscriber::filter::LevelFilter;

/// Output format for command results.
#[derive(clap::ValueEnum, Clone, Default, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Human,
    Json,
    Csv,
}

/// Exit codes for structured error reporting.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum ExitCode {
    Success = 0,
    GeneralError = 1,
    UsageError = 2,
    ConfigError = 3,
    ModelError = 4,
    StorageError = 5,
    NoResults = 10,
}

impl ExitCode {
    fn from_error(err: &anyhow::Error) -> Self {
        if let Some(sift_err) = err.downcast_ref::<sift_core::SiftError>() {
            return Self::from_sift_error(sift_err);
        }
        ExitCode::GeneralError
    }

    fn from_sift_error(err: &sift_core::SiftError) -> Self {
        match err {
            sift_core::SiftError::Config(_) => ExitCode::ConfigError,
            sift_core::SiftError::Model(_) | sift_core::SiftError::Embedding(_) => {
                ExitCode::ModelError
            }
            sift_core::SiftError::Storage(_) => ExitCode::StorageError,
            sift_core::SiftError::Io(_) => ExitCode::StorageError,
            sift_core::SiftError::Parse { .. } => ExitCode::GeneralError,
            sift_core::SiftError::Search(_) => ExitCode::GeneralError,
            sift_core::SiftError::Source(_) => ExitCode::GeneralError,
            sift_core::SiftError::Other(_) => ExitCode::GeneralError,
        }
    }
}

/// Check whether colored output should be used, respecting NO_COLOR and TERM=dumb.
#[cfg(feature = "fancy")]
fn use_color() -> bool {
    std::env::var("NO_COLOR").is_err() && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
}

#[derive(Parser)]
#[command(
    name = "sift",
    about = "Universal data vectorizer — point at anything, search everything",
    version,
    after_help = "Examples:\n  sift scan .                              Index current directory\n  sift search \"error handling\"             Semantic search\n  sift search --type rs \"config parsing\"   Search only Rust files\n  sift status                              Show index statistics"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Use a named index (default: "default")
    #[arg(
        short,
        long,
        global = true,
        default_value = "default",
        env = "SIFT_INDEX"
    )]
    index: String,

    /// Suppress progress output
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Increase verbosity (-v for info, -vv for debug)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Output format (human, json, csv)
    #[arg(long, global = true, default_value = "human", env = "SIFT_FORMAT")]
    format: OutputFormat,

    /// Output as JSON (for piping) [alias for --format json]
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan and index data sources
    Scan {
        /// Paths to scan
        #[arg(required = true)]
        paths: Vec<std::path::PathBuf>,

        /// Maximum directory depth
        #[arg(long)]
        max_depth: Option<usize>,

        /// Skip files larger than this (bytes)
        #[arg(long)]
        max_file_size: Option<u64>,

        /// Only include files matching this glob
        #[arg(long)]
        include: Vec<String>,

        /// Exclude files matching this glob
        #[arg(long)]
        exclude: Vec<String>,

        /// Only index specific file types (e.g., pdf, rs, py)
        #[arg(short = 't', long = "type")]
        file_types: Vec<String>,

        /// Override embedding model
        #[arg(long, env = "SIFT_MODEL")]
        model: Option<String>,

        /// Show what would be indexed without actually indexing
        #[arg(long)]
        dry_run: bool,

        /// Number of parallel workers (0 = auto)
        #[arg(short, long, default_value = "0", env = "SIFT_JOBS")]
        jobs: usize,
    },

    /// Semantic search across indexed data
    Search {
        /// Search query
        #[arg(required = true)]
        query: Vec<String>,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "10")]
        max_results: usize,

        /// Filter by file type
        #[arg(short = 't', long = "type")]
        file_type: Option<String>,

        /// Filter by path pattern
        #[arg(long)]
        path: Option<String>,

        /// Minimum similarity threshold (0.0-1.0)
        #[arg(long, default_value = "0.0")]
        threshold: f32,

        /// Use only vector search (no BM25)
        #[arg(long)]
        vector_only: bool,

        /// Use only BM25 keyword search
        #[arg(long)]
        keyword_only: bool,

        /// Show surrounding context
        #[arg(short = 'c', long)]
        context: bool,

        /// Open top result in default application
        #[arg(short = 'o', long)]
        open: bool,

        /// Only show results modified after this date (ISO 8601 or relative: 7d, 2w, 3m)
        #[arg(long)]
        after: Option<String>,
    },

    /// Watch filesystem for changes and re-index
    #[cfg(feature = "serve")]
    Watch {
        /// Path to watch (default: previously scanned paths)
        path: Option<std::path::PathBuf>,

        /// Debounce interval in milliseconds
        #[arg(long, default_value = "1000")]
        debounce: u64,
    },

    /// Show index statistics and health
    Status,

    /// List indexed sources
    List,

    /// Remove a source from the index
    Remove {
        /// URI or path to remove
        #[arg(required = true)]
        paths: Vec<String>,
    },

    /// View or set configuration
    Config {
        /// Configuration key
        key: Option<String>,
        /// Configuration value (set mode)
        value: Option<String>,
    },

    /// List, download, or manage embedding models
    #[cfg(feature = "embeddings")]
    Models {
        #[command(subcommand)]
        action: Option<ModelsAction>,
    },

    /// Start HTTP API server
    #[cfg(feature = "serve")]
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(long, default_value = "7820")]
        port: u16,
    },

    /// Export index data as JSONL
    Export {
        /// Include embedding vectors in output
        #[arg(long)]
        vectors: bool,

        /// Output file (default: stdout)
        #[arg(short = 'o', long)]
        output: Option<std::path::PathBuf>,

        /// Filter by file type
        #[arg(short = 't', long = "type")]
        file_type: Option<String>,
    },

    /// Generate shell completions
    #[cfg(feature = "completions")]
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
        shell: clap_complete::Shell,
    },
}

#[cfg(feature = "embeddings")]
#[derive(Subcommand)]
enum ModelsAction {
    /// List downloaded models
    List,
    /// Download a model
    Download {
        /// Model name
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    // Resolve effective output format: --json flag overrides --format
    let format = if cli.json {
        OutputFormat::Json
    } else {
        cli.format.clone()
    };

    // Respect NO_COLOR / TERM=dumb
    #[cfg(feature = "fancy")]
    if !use_color() {
        colored::control::set_override(false);
    }

    // Set up tracing
    let level = match std::env::var("RUST_LOG").ok().as_deref() {
        Some("trace") => LevelFilter::TRACE,
        Some("debug") => LevelFilter::DEBUG,
        Some("warn") | Some("warning") => LevelFilter::WARN,
        Some("error") => LevelFilter::ERROR,
        Some(_) | None => match cli.verbose {
            0 if cli.quiet => LevelFilter::WARN,
            0 => LevelFilter::INFO,
            1 => LevelFilter::DEBUG,
            _ => LevelFilter::TRACE,
        },
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    let result = run_command(cli, &format);

    match result {
        Ok(()) => std::process::exit(ExitCode::Success as i32),
        Err(err) => {
            let code = ExitCode::from_error(&err);
            if format == OutputFormat::Json {
                let obj = serde_json::json!({
                    "error": format!("{}", err),
                    "exit_code": code as i32,
                });
                eprintln!("{}", obj);
            } else {
                eprintln!("error: {}", err);
                if std::env::args().any(|a| a == "-v" || a == "-vv" || a == "--verbose") {
                    for cause in err.chain().skip(1) {
                        eprintln!("  caused by: {}", cause);
                    }
                }
            }
            std::process::exit(code as i32);
        }
    }
}

fn run_command(cli: Cli, format: &OutputFormat) -> anyhow::Result<()> {
    let mut config = sift_core::Config::load()?;
    config.index_name = cli.index.clone();

    match cli.command {
        Commands::Scan {
            paths,
            max_depth,
            max_file_size,
            include,
            exclude,
            file_types,
            model,
            dry_run,
            jobs,
        } => {
            let options = sift_core::ScanOptions {
                paths,
                recursive: true,
                max_depth,
                max_file_size: max_file_size.or(Some(config.default.max_file_size)),
                include_globs: include,
                exclude_globs: exclude,
                file_types,
                dry_run,
                jobs: if jobs == 0 { config.num_jobs() } else { jobs },
            };
            commands::scan::run(&config, &options, model.as_deref(), format, cli.quiet)?;
        }

        Commands::Search {
            query,
            max_results,
            file_type,
            path,
            threshold,
            vector_only,
            keyword_only,
            context,
            open,
            after,
        } => {
            let mode = if vector_only {
                sift_core::SearchMode::VectorOnly
            } else if keyword_only {
                sift_core::SearchMode::KeywordOnly
            } else {
                sift_core::SearchMode::Hybrid
            };

            let after_ts = after.as_deref().map(parse_after_date).transpose()?;

            let options = sift_core::SearchOptions {
                query: query.join(" "),
                max_results,
                file_type,
                path_glob: path,
                threshold,
                mode,
                context,
                after: after_ts,
            };
            commands::search::run(&config, &options, format, open)?;
        }

        #[cfg(feature = "serve")]
        Commands::Watch { path, debounce } => {
            commands::watch::run(&config, path, debounce)?;
        }

        Commands::Status => {
            commands::status::run(&config, format)?;
        }

        Commands::List => {
            commands::list::run(&config, format)?;
        }

        Commands::Remove { paths } => {
            commands::remove::run(&config, &paths, format)?;
        }

        Commands::Config { key, value } => {
            commands::config::run(&config, key, value)?;
        }

        #[cfg(feature = "embeddings")]
        Commands::Models { action } => {
            commands::models::run(action)?;
        }

        #[cfg(feature = "serve")]
        Commands::Serve { host, port } => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| anyhow::anyhow!("Failed to start async runtime: {}", e))?;
            rt.block_on(commands::serve::run(&config, &host, port))?;
        }

        Commands::Export {
            vectors,
            output,
            file_type,
        } => {
            commands::export::run(&config, vectors, output, file_type)?;
        }

        #[cfg(feature = "completions")]
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "sift", &mut std::io::stdout());
        }
    }

    Ok(())
}

/// Parse a date string into a Unix timestamp. Supports:
/// - ISO 8601 dates: `2025-01-01`
/// - Relative durations: `7d`, `2w`, `3m`
fn parse_after_date(s: &str) -> anyhow::Result<i64> {
    let s_trimmed = s.trim();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    if let Some(num_str) = s_trimmed.strip_suffix('d') {
        let days: i64 = num_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid number in '{}'", s))?;
        return Ok(now - days * 86400);
    }
    if let Some(num_str) = s_trimmed.strip_suffix('w') {
        let weeks: i64 = num_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid number in '{}'", s))?;
        return Ok(now - weeks * 7 * 86400);
    }
    if let Some(num_str) = s_trimmed.strip_suffix('m') {
        let months: i64 = num_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid number in '{}'", s))?;
        return Ok(now - months * 30 * 86400);
    }

    // Parse ISO 8601 date: YYYY-MM-DD
    let parts: Vec<&str> = s_trimmed.split('-').collect();
    if parts.len() != 3 {
        anyhow::bail!(
            "invalid date '{}'. Use YYYY-MM-DD or relative (7d, 2w, 3m)",
            s
        );
    }
    let year: i64 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid year in '{}'", s))?;
    let month: i64 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid month in '{}'", s))?;
    let day: i64 = parts[2]
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid day in '{}'", s))?;

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || year < 1970 {
        anyhow::bail!(
            "invalid date '{}'. Use YYYY-MM-DD or relative (7d, 2w, 3m)",
            s
        );
    }

    // Days from epoch to date using the standard algorithm
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let y = if month <= 2 { year - 1 } else { year };
    let days_from_epoch =
        365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + day - 1 - 719468;

    Ok(days_from_epoch * 86400)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_after_date_iso_format() {
        let ts = parse_after_date("2025-01-01").unwrap();
        assert_eq!(ts, 1735689600);
    }

    #[test]
    fn test_parse_after_date_iso_format_midyear() {
        let ts = parse_after_date("2025-06-15").unwrap();
        assert!(ts > 1735689600);
        assert!(ts < 1767225600);
    }

    #[test]
    fn test_parse_after_date_relative_days() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let ts = parse_after_date("7d").unwrap();
        let expected = now - 7 * 86400;
        assert!((ts - expected).abs() < 2);
    }

    #[test]
    fn test_parse_after_date_relative_weeks() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let ts = parse_after_date("2w").unwrap();
        let expected = now - 2 * 7 * 86400;
        assert!((ts - expected).abs() < 2);
    }

    #[test]
    fn test_parse_after_date_relative_months() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let ts = parse_after_date("3m").unwrap();
        let expected = now - 3 * 30 * 86400;
        assert!((ts - expected).abs() < 2);
    }

    #[test]
    fn test_parse_after_date_single_day() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let ts = parse_after_date("1d").unwrap();
        let expected = now - 86400;
        assert!((ts - expected).abs() < 2);
    }

    #[test]
    fn test_parse_after_date_trims_whitespace() {
        let ts = parse_after_date("  2025-01-01  ").unwrap();
        assert_eq!(ts, 1735689600);
    }

    #[test]
    fn test_parse_after_date_invalid_format() {
        assert!(parse_after_date("not-a-date").is_err());
    }

    #[test]
    fn test_parse_after_date_invalid_number_in_relative() {
        assert!(parse_after_date("abcd").is_err());
    }

    #[test]
    fn test_parse_after_date_empty_number_before_unit() {
        assert!(parse_after_date("d").is_err());
    }

    #[test]
    fn test_parse_after_date_invalid_iso_date() {
        assert!(parse_after_date("2025-13-01").is_err());
        assert!(parse_after_date("2025-00-01").is_err());
    }
}
