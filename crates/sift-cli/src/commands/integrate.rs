//! `sift integrate` — zero-config MCP installer for AI tools.
//!
//! Detects installed AI clients (Claude Code, Cursor, Codex, Gemini CLI, etc.)
//! and writes their MCP configuration to register sift as a server. Each
//! platform is an adapter that owns its config path and merge strategy. The
//! write is idempotent: re-running the command on an already-configured host
//! is a no-op unless `--force` is set.
//!
//! Transport choice:
//! - stdio (default): spawns `sift mcp` per session. Zero setup, works offline.
//! - HTTP (via `--http <url>`): points clients at a shared `sift serve
//!   --mcp-http` endpoint. Preferred when multiple clients should share memory
//!   without each spawning its own process.

use super::memory::{build_hooks_json, codex_hooks_path, HookTarget};
#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "fancy")]
use colored::*;
use serde_json::{json, Value};
use sift_core::Config as SiftConfig;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Options supplied on the CLI for `sift integrate`.
#[allow(clippy::struct_excessive_bools)]
pub struct IntegrateOptions {
    /// Specific platform slugs to target (empty = all detected).
    pub platforms: Vec<String>,
    /// Show intended changes without writing.
    pub dry_run: bool,
    /// Overwrite existing sift entries instead of skipping.
    pub force: bool,
    /// Remove sift entries instead of adding them.
    pub uninstall: bool,
    /// Override the sift binary path (default: current exe).
    pub exe: Option<PathBuf>,
    /// If set, register sift as an HTTP MCP server at this URL instead of stdio.
    pub http_url: Option<String>,
    /// Only print detected platforms and their status; make no changes.
    pub list: bool,
}

/// What an adapter did for a single platform.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    /// Wrote a new sift entry.
    Installed,
    /// Updated an existing sift entry (contents differed).
    Updated,
    /// Entry already present and matches — no write.
    AlreadyPresent,
    /// Platform config didn't exist and we didn't create it (not detected).
    NotDetected,
    /// Removed an existing sift entry.
    Uninstalled,
    /// Nothing to uninstall (no sift entry present).
    NothingToUninstall,
    /// Dry-run — would have written.
    Planned,
    /// Adapter refused for a specific reason.
    Skipped(String),
}

/// Per-platform outcome suitable for rendering.
struct Report {
    platform: &'static str,
    config_path: PathBuf,
    action: Action,
}

/// Canonical description of the sift MCP server entry.
///
/// Each adapter translates this into the format its target platform expects.
struct SiftServerDef {
    /// Absolute path to the sift binary.
    exe: PathBuf,
    /// Args to pass to the binary (typically `["mcp"]`).
    args: Vec<String>,
    /// Env vars to set when spawning the server.
    env: BTreeMap<String, String>,
    /// If set, register over HTTP at this URL instead of stdio.
    http_url: Option<String>,
}

impl SiftServerDef {
    fn is_http(&self) -> bool {
        self.http_url.is_some()
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(opts: IntegrateOptions) -> Result<()> {
    let def = build_server_def(&opts)?;
    let home = home_dir().context(
        "cannot resolve a home directory; set HOME or USERPROFILE to use sift integrate",
    )?;

    let adapters = all_adapters();

    // Filter to requested platforms (if any).
    let targets: Vec<&Adapter> = if opts.platforms.is_empty() {
        adapters.iter().collect()
    } else {
        let mut selected = Vec::new();
        for name in &opts.platforms {
            let a = adapters
                .iter()
                .find(|a| a.slug == name.as_str())
                .ok_or_else(|| {
                    anyhow!(
                        "unknown platform '{}'. Known: {}",
                        name,
                        adapters
                            .iter()
                            .map(|a| a.slug)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            selected.push(a);
        }
        selected
    };

    // --list mode: show detection only.
    if opts.list {
        println!("{}", "Platform detection".bold());
        for a in adapters {
            let path = (a.config_path)(&home);
            let detected = (a.detect)(&home);
            let marker = if detected {
                "detected".green().to_string()
            } else {
                "not found".dimmed().to_string()
            };
            println!("  {:<14} {}  {}", a.slug, marker, path.display());
        }
        return Ok(());
    }

    // Run each adapter.
    let mut reports = Vec::new();
    for a in targets {
        let path = (a.config_path)(&home);
        let detected = (a.detect)(&home);

        // When the user explicitly named a platform, run even if not detected
        // (creates the config dir/file from scratch). Otherwise skip.
        let explicit = !opts.platforms.is_empty();
        if !detected && !explicit {
            reports.push(Report {
                platform: a.slug,
                config_path: path,
                action: Action::NotDetected,
            });
            continue;
        }

        let action = if opts.uninstall {
            (a.uninstall)(&path, &opts)
        } else {
            (a.install)(&path, &def, &opts)
        };
        let action = action.unwrap_or_else(|e| Action::Skipped(format!("error: {e}")));

        reports.push(Report {
            platform: a.slug,
            config_path: path,
            action,
        });
    }

    print_summary(&reports, &opts);
    Ok(())
}

// ---------------------------------------------------------------------------
// Server definition
// ---------------------------------------------------------------------------

fn build_server_def(opts: &IntegrateOptions) -> Result<SiftServerDef> {
    let exe = resolve_exe(opts.exe.as_deref())?;
    let mut env = BTreeMap::new();
    if let Some(ort) = resolve_ort_dylib() {
        env.insert("ORT_DYLIB_PATH".to_string(), ort.display().to_string());
    }
    Ok(SiftServerDef {
        exe,
        args: vec!["mcp".to_string()],
        env,
        http_url: opts.http_url.clone(),
    })
}

fn resolve_exe(override_: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = override_ {
        if !p.exists() {
            return Err(anyhow!("--exe path does not exist: {}", p.display()));
        }
        return Ok(p.canonicalize().unwrap_or_else(|_| p.to_path_buf()));
    }
    // Prefer an installed binary on PATH (homebrew, cargo install) so the
    // config doesn't depend on a dev target/ directory.
    if let Some(p) = which("sift") {
        return Ok(p);
    }
    // Fall back to the currently-running exe.
    let current = std::env::current_exe().context("cannot determine current executable path")?;
    Ok(current)
}

fn resolve_ort_dylib() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let home = home_dir()?;
    // Standard location used by `sift models download`.
    for rel in [
        ".sift/models/ort/libonnxruntime.dylib",
        ".sift/models/ort/libonnxruntime.so",
        ".sift/models/ort/onnxruntime.dll",
    ] {
        let p = home.join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn which(cmd: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{cmd}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

fn home_dir() -> Option<PathBuf> {
    home_dir_from_env(|key| std::env::var_os(key))
}

fn home_dir_from_env(get: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    first_non_empty_path([
        get("HOME"),
        get("USERPROFILE"),
        match (get("HOMEDRIVE"), get("HOMEPATH")) {
            (Some(drive), Some(path)) => {
                let mut joined = drive;
                joined.push(path);
                Some(joined)
            }
            _ => None,
        },
    ])
}

fn first_non_empty_path(candidates: [Option<OsString>; 3]) -> Option<PathBuf> {
    candidates
        .into_iter()
        .flatten()
        .map(PathBuf::from)
        .find(|p| !p.as_os_str().is_empty())
}

// ---------------------------------------------------------------------------
// Adapter registry
// ---------------------------------------------------------------------------

struct Adapter {
    slug: &'static str,
    /// Returns true if this platform appears to be installed.
    detect: fn(&Path) -> bool,
    /// Returns the expected config file path.
    config_path: fn(&Path) -> PathBuf,
    /// Writes (or plans) the sift entry.
    install: fn(&Path, &SiftServerDef, &IntegrateOptions) -> Result<Action>,
    /// Removes the sift entry.
    uninstall: fn(&Path, &IntegrateOptions) -> Result<Action>,
}

fn all_adapters() -> &'static [Adapter] {
    &[
        Adapter {
            slug: "claude-code",
            detect: |home| home.join(".claude.json").exists() || home.join(".claude").is_dir(),
            config_path: |home| home.join(".claude.json"),
            install: |p, d, o| install_mcp_servers_json(p, d, o, /*root_object=*/ true),
            uninstall: |p, o| uninstall_mcp_servers_json(p, o, /*root_object=*/ true),
        },
        Adapter {
            slug: "cursor",
            detect: |home| home.join(".cursor").is_dir(),
            config_path: |home| home.join(".cursor/mcp.json"),
            install: |p, d, o| install_mcp_servers_json(p, d, o, /*root_object=*/ false),
            uninstall: |p, o| uninstall_mcp_servers_json(p, o, /*root_object=*/ false),
        },
        Adapter {
            slug: "windsurf",
            detect: |home| home.join(".codeium/windsurf").is_dir(),
            config_path: |home| home.join(".codeium/windsurf/mcp_config.json"),
            install: |p, d, o| install_mcp_servers_json(p, d, o, /*root_object=*/ false),
            uninstall: |p, o| uninstall_mcp_servers_json(p, o, /*root_object=*/ false),
        },
        Adapter {
            slug: "codex",
            detect: |home| home.join(".codex").is_dir(),
            config_path: |home| home.join(".codex/config.toml"),
            install: install_codex,
            uninstall: uninstall_codex,
        },
        Adapter {
            slug: "gemini",
            detect: |home| home.join(".gemini").is_dir(),
            config_path: |home| home.join(".gemini/settings.json"),
            install: |p, d, o| install_mcp_servers_json(p, d, o, /*root_object=*/ false),
            uninstall: |p, o| uninstall_mcp_servers_json(p, o, /*root_object=*/ false),
        },
        Adapter {
            slug: "opencode",
            detect: |home| home.join(".config/opencode").is_dir(),
            config_path: |home| home.join(".config/opencode/opencode.json"),
            install: install_opencode,
            uninstall: uninstall_opencode,
        },
        Adapter {
            slug: "zed",
            detect: |home| home.join(".config/zed").is_dir(),
            config_path: |home| home.join(".config/zed/settings.json"),
            install: install_zed,
            uninstall: uninstall_zed,
        },
        Adapter {
            slug: "vscode",
            detect: detect_vscode,
            config_path: vscode_config_path,
            install: install_vscode,
            uninstall: uninstall_vscode,
        },
        Adapter {
            slug: "lm-studio",
            detect: |home| home.join(".lmstudio").is_dir(),
            config_path: |home| home.join(".lmstudio/mcp.json"),
            install: |p, d, o| install_mcp_servers_json(p, d, o, /*root_object=*/ false),
            uninstall: |p, o| uninstall_mcp_servers_json(p, o, /*root_object=*/ false),
        },
    ]
}

// ---------------------------------------------------------------------------
// Entry shapes
// ---------------------------------------------------------------------------

/// Build a `mcpServers` entry value in the standard shape most clients accept.
///
/// For stdio: `{ "command": ..., "args": [...], "env": {...} }`.
/// For HTTP:  `{ "type": "http", "url": "..." }`.
fn standard_entry(def: &SiftServerDef) -> Value {
    if let Some(url) = &def.http_url {
        return json!({ "type": "http", "url": url });
    }
    let mut entry = json!({
        "command": def.exe.display().to_string(),
        "args": def.args,
    });
    if !def.env.is_empty() {
        entry["env"] = json!(def.env);
    }
    entry
}

// ---------------------------------------------------------------------------
// Shared JSON `mcpServers` install/uninstall
// ---------------------------------------------------------------------------

/// Install into a JSON file with a top-level `mcpServers` object.
///
/// When `root_object` is true, we preserve unrelated top-level keys (used for
/// Claude Code's big `~/.claude.json`). When false, the file is just
/// `{"mcpServers": {...}}` and we create it on demand.
fn install_mcp_servers_json(
    path: &Path,
    def: &SiftServerDef,
    opts: &IntegrateOptions,
    _root_object: bool,
) -> Result<Action> {
    install_json_under_key(path, def, opts, "mcpServers")
}

fn uninstall_mcp_servers_json(
    path: &Path,
    opts: &IntegrateOptions,
    _root_object: bool,
) -> Result<Action> {
    uninstall_json_under_key(path, opts, "mcpServers")
}

/// Generic installer: read (or create) a JSON object, merge `sift` under
/// the given top-level key, write back.
fn install_json_under_key(
    path: &Path,
    def: &SiftServerDef,
    opts: &IntegrateOptions,
    top_key: &str,
) -> Result<Action> {
    let mut root = read_json_or_default(path)?;

    // Make sure the root is an object.
    if !root.is_object() {
        return Err(anyhow!(
            "{}: expected top-level JSON object, found {}",
            path.display(),
            describe_json(&root)
        ));
    }
    // Ensure `top_key` is an object (create if missing).
    let entry_name = "sift";
    let new_entry = standard_entry(def);
    let servers = root
        .as_object_mut()
        .unwrap()
        .entry(top_key.to_string())
        .or_insert_with(|| json!({}));
    if !servers.is_object() {
        return Err(anyhow!(
            "{}: '{}' must be an object, found {}",
            path.display(),
            top_key,
            describe_json(servers)
        ));
    }
    let servers_map = servers.as_object_mut().unwrap();

    let existing = servers_map.get(entry_name).cloned();
    let already_matches = existing.as_ref().is_some_and(|v| v == &new_entry);

    if already_matches {
        return Ok(Action::AlreadyPresent);
    }

    if existing.is_some() && !opts.force {
        return Ok(Action::Skipped(format!(
            "existing '{entry_name}' entry differs; re-run with --force to overwrite"
        )));
    }

    if opts.dry_run {
        return Ok(Action::Planned);
    }

    servers_map.insert(entry_name.to_string(), new_entry);

    write_json_pretty(path, &root)?;
    Ok(if existing.is_some() {
        Action::Updated
    } else {
        Action::Installed
    })
}

fn uninstall_json_under_key(path: &Path, opts: &IntegrateOptions, top_key: &str) -> Result<Action> {
    if !path.exists() {
        return Ok(Action::NothingToUninstall);
    }
    let mut root = read_json_or_default(path)?;
    let Some(servers) = root
        .as_object_mut()
        .and_then(|m| m.get_mut(top_key))
        .and_then(|v| v.as_object_mut())
    else {
        return Ok(Action::NothingToUninstall);
    };

    if servers.remove("sift").is_none() {
        return Ok(Action::NothingToUninstall);
    }

    if opts.dry_run {
        return Ok(Action::Planned);
    }

    write_json_pretty(path, &root)?;
    Ok(Action::Uninstalled)
}

fn read_json_or_default(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(&text).with_context(|| format!("parsing JSON in {}", path.display()))
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let pretty = serde_json::to_string_pretty(value)?;
    atomic_write(path, pretty.as_bytes())
}

fn describe_json(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// Codex (TOML)
// ---------------------------------------------------------------------------

fn install_codex(path: &Path, def: &SiftServerDef, opts: &IntegrateOptions) -> Result<Action> {
    let project_root = resolve_codex_project_root();
    install_codex_with_project_root(path, def, opts, &project_root)
}

fn install_codex_with_project_root(
    path: &Path,
    def: &SiftServerDef,
    opts: &IntegrateOptions,
    project_root: &Path,
) -> Result<Action> {
    if def.is_http() {
        // Codex only supports stdio MCP servers today.
        return Ok(Action::Skipped(
            "codex does not support HTTP MCP transport".into(),
        ));
    }

    let mut doc = read_toml_or_default(path)?;

    // Build desired `[mcp_servers.sift]` table.
    let mut entry = toml::value::Table::new();
    entry.insert(
        "command".into(),
        toml::Value::String(def.exe.display().to_string()),
    );
    entry.insert(
        "args".into(),
        toml::Value::Array(
            def.args
                .iter()
                .map(|s| toml::Value::String(s.clone()))
                .collect(),
        ),
    );
    if !def.env.is_empty() {
        let mut env_table = toml::value::Table::new();
        for (k, v) in &def.env {
            env_table.insert(k.clone(), toml::Value::String(v.clone()));
        }
        entry.insert("env".into(), toml::Value::Table(env_table));
    }

    // Ensure [mcp_servers] parent exists.
    let parent = doc
        .as_table_mut()
        .unwrap()
        .entry("mcp_servers")
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    let existing = parent
        .as_table_mut()
        .ok_or_else(|| anyhow!("codex config: [mcp_servers] is not a table"))?
        .get("sift")
        .cloned();
    let desired = toml::Value::Table(entry);
    let already_matches = existing.as_ref().is_some_and(|v| v == &desired);
    let had_existing_config = existing.is_some();

    if had_existing_config && !already_matches && !opts.force {
        return Ok(Action::Skipped(
            "existing [mcp_servers.sift] differs; re-run with --force".into(),
        ));
    }

    let features = doc
        .as_table_mut()
        .unwrap()
        .entry("features")
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    let features_table = features
        .as_table_mut()
        .ok_or_else(|| anyhow!("codex config: [features] is not a table"))?;
    let hooks_enabled = features_table
        .get("codex_hooks")
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);
    let features_changed = !hooks_enabled;
    if features_changed {
        features_table.insert("codex_hooks".into(), toml::Value::Boolean(true));
    }

    let config_changed = !already_matches || features_changed;
    if !already_matches {
        let parent_table = doc
            .as_table_mut()
            .unwrap()
            .get_mut("mcp_servers")
            .and_then(toml::Value::as_table_mut)
            .ok_or_else(|| anyhow!("codex config: [mcp_servers] is not a table"))?;
        parent_table.insert("sift".into(), desired);
    }

    let hooks_path = codex_hooks_path(project_root);
    let hooks_state = sync_codex_hooks(&hooks_path, def, opts, false)?;

    if !config_changed && !hooks_state.changed {
        return Ok(Action::AlreadyPresent);
    }
    if opts.dry_run {
        return Ok(Action::Planned);
    }

    if config_changed {
        let out = toml::to_string_pretty(&doc).context("serializing codex TOML")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        atomic_write(path, out.as_bytes())?;
    }

    Ok(if had_existing_config || hooks_state.had_existing {
        Action::Updated
    } else {
        Action::Installed
    })
}

fn uninstall_codex(path: &Path, opts: &IntegrateOptions) -> Result<Action> {
    let project_root = resolve_codex_project_root();
    uninstall_codex_with_project_root(path, opts, &project_root)
}

fn uninstall_codex_with_project_root(
    path: &Path,
    opts: &IntegrateOptions,
    project_root: &Path,
) -> Result<Action> {
    let hooks_path = codex_hooks_path(project_root);
    let hooks_state = sync_codex_hooks(
        &hooks_path,
        &SiftServerDef {
            exe: PathBuf::from("sift"),
            args: vec!["mcp".into()],
            env: BTreeMap::new(),
            http_url: None,
        },
        opts,
        true,
    )?;

    let mut removed_config = false;
    let mut doc = read_toml_or_default(path)?;
    if path.exists() {
        if let Some(mcp) = doc
            .as_table_mut()
            .and_then(|t| t.get_mut("mcp_servers"))
            .and_then(|v| v.as_table_mut())
        {
            removed_config = mcp.remove("sift").is_some();
            if mcp.is_empty() {
                doc.as_table_mut().unwrap().remove("mcp_servers");
            }
        }
    }

    if !removed_config && !hooks_state.changed && !hooks_state.had_existing {
        return Ok(Action::NothingToUninstall);
    }
    if opts.dry_run {
        return Ok(Action::Planned);
    }

    if removed_config {
        let out = toml::to_string_pretty(&doc)?;
        atomic_write(path, out.as_bytes())?;
    }
    Ok(Action::Uninstalled)
}

#[derive(Debug, Clone, Copy)]
struct CodexHookSyncState {
    changed: bool,
    had_existing: bool,
}

fn resolve_codex_project_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    SiftConfig::find_project_root(&cwd).unwrap_or(cwd)
}

fn sync_codex_hooks(
    path: &Path,
    def: &SiftServerDef,
    opts: &IntegrateOptions,
    remove: bool,
) -> Result<CodexHookSyncState> {
    let mut root = read_json_or_default(path)?;
    if !root.is_object() {
        return Err(anyhow!(
            "{}: expected top-level JSON object, found {}",
            path.display(),
            describe_json(&root)
        ));
    }
    let original_root = root.clone();

    let desired = build_hooks_json(HookTarget::Codex, &def.exe.display().to_string());
    let desired_hooks = desired
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("codex hooks template missing top-level 'hooks' object"))?;

    let root_obj = root.as_object_mut().unwrap();
    let hooks_value = root_obj
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}));
    if !hooks_value.is_object() {
        return Err(anyhow!(
            "{}: 'hooks' must be an object, found {}",
            path.display(),
            describe_json(hooks_value)
        ));
    }
    let hooks_obj = hooks_value.as_object_mut().unwrap();

    let mut had_existing = false;

    for (event, desired_groups_value) in desired_hooks {
        let desired_groups = desired_groups_value
            .as_array()
            .ok_or_else(|| anyhow!("codex hooks template event '{event}' is not an array"))?;

        let existing_groups = hooks_obj
            .get(event)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mut rebuilt_groups = Vec::new();
        for group in existing_groups {
            let Some(group_obj) = group.as_object() else {
                rebuilt_groups.push(group);
                continue;
            };
            let matcher = group_obj.get("matcher").and_then(Value::as_str);
            let Some(existing_hooks) = group_obj.get("hooks").and_then(Value::as_array) else {
                rebuilt_groups.push(group);
                continue;
            };

            let mut filtered_hooks = Vec::new();
            let mut removed_any = false;
            for hook in existing_hooks {
                if is_managed_codex_hook(event, matcher, hook) {
                    had_existing = true;
                    removed_any = true;
                } else {
                    filtered_hooks.push(hook.clone());
                }
            }

            if filtered_hooks.is_empty() {
                if !removed_any {
                    rebuilt_groups.push(group);
                }
                continue;
            }

            if removed_any {
                let mut new_group = group_obj.clone();
                new_group.insert("hooks".into(), Value::Array(filtered_hooks));
                rebuilt_groups.push(Value::Object(new_group));
            } else {
                rebuilt_groups.push(group);
            }
        }

        if !remove {
            for desired_group in desired_groups {
                if !rebuilt_groups.iter().any(|group| group == desired_group) {
                    rebuilt_groups.push(desired_group.clone());
                }
            }
        }

        if rebuilt_groups.is_empty() {
            hooks_obj.remove(event);
        } else {
            let new_value = Value::Array(rebuilt_groups);
            hooks_obj.insert(event.clone(), new_value);
        }
    }

    if hooks_obj.is_empty() {
        root_obj.remove("hooks");
    }

    let changed = root != original_root;

    if changed && !opts.dry_run {
        write_json_pretty(path, &root)?;
    }

    Ok(CodexHookSyncState {
        changed,
        had_existing,
    })
}

fn is_managed_codex_hook(event: &str, matcher: Option<&str>, hook: &Value) -> bool {
    let Some(hook_obj) = hook.as_object() else {
        return false;
    };
    let status = hook_obj.get("statusMessage").and_then(Value::as_str);
    let command = hook_obj
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("");
    match event {
        "PostToolUse" => {
            matcher == Some("Bash")
                && status == Some("sift: ingest Bash tool use")
                && command.contains(" memory ingest --event post-tool-use")
        }
        "Stop" => match status {
            Some("sift: ingest stop summary") => command.contains(" memory ingest --event stop"),
            Some("sift: consolidate memory") => command.contains(" memory consolidate --quiet"),
            _ => false,
        },
        _ => false,
    }
}

fn read_toml_or_default(path: &Path) -> Result<toml::Value> {
    if !path.exists() {
        return Ok(toml::Value::Table(toml::value::Table::new()));
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(toml::Value::Table(toml::value::Table::new()));
    }
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parsing TOML in {}", path.display()))?;
    Ok(value)
}

// ---------------------------------------------------------------------------
// opencode (has its own shape)
// ---------------------------------------------------------------------------

fn opencode_entry(def: &SiftServerDef) -> Value {
    if let Some(url) = &def.http_url {
        return json!({ "type": "remote", "url": url, "enabled": true });
    }
    let mut command = vec![def.exe.display().to_string()];
    command.extend(def.args.iter().cloned());
    let mut v = json!({
        "type": "local",
        "command": command,
        "enabled": true,
    });
    if !def.env.is_empty() {
        v["environment"] = json!(def.env);
    }
    v
}

fn install_opencode(path: &Path, def: &SiftServerDef, opts: &IntegrateOptions) -> Result<Action> {
    let mut root = read_json_or_default(path)?;
    if !root.is_object() {
        return Err(anyhow!(
            "{}: expected top-level JSON object",
            path.display()
        ));
    }
    // opencode uses `"mcp": { "<name>": ... }`.
    let entry = opencode_entry(def);
    let servers = root
        .as_object_mut()
        .unwrap()
        .entry("mcp".to_string())
        .or_insert_with(|| json!({}));
    if !servers.is_object() {
        return Err(anyhow!("{}: 'mcp' must be an object", path.display()));
    }
    let servers_map = servers.as_object_mut().unwrap();
    let existing = servers_map.get("sift").cloned();
    let already = existing.as_ref().is_some_and(|v| v == &entry);
    if already {
        return Ok(Action::AlreadyPresent);
    }
    if existing.is_some() && !opts.force {
        return Ok(Action::Skipped(
            "existing opencode 'sift' entry differs; re-run with --force".into(),
        ));
    }
    if opts.dry_run {
        return Ok(Action::Planned);
    }
    servers_map.insert("sift".into(), entry);
    write_json_pretty(path, &root)?;
    Ok(if existing.is_some() {
        Action::Updated
    } else {
        Action::Installed
    })
}

fn uninstall_opencode(path: &Path, opts: &IntegrateOptions) -> Result<Action> {
    uninstall_json_under_key(path, opts, "mcp")
}

// ---------------------------------------------------------------------------
// Zed (context_servers)
// ---------------------------------------------------------------------------

fn zed_entry(def: &SiftServerDef) -> Value {
    let mut cmd = json!({
        "path": def.exe.display().to_string(),
        "args": def.args,
    });
    if !def.env.is_empty() {
        cmd["env"] = json!(def.env);
    } else {
        cmd["env"] = json!({});
    }
    json!({
        "source": "custom",
        "command": cmd,
    })
}

fn install_zed(path: &Path, def: &SiftServerDef, opts: &IntegrateOptions) -> Result<Action> {
    if def.is_http() {
        return Ok(Action::Skipped(
            "zed does not support HTTP MCP transport".into(),
        ));
    }

    let mut root = read_json_or_default(path)?;
    if !root.is_object() {
        return Err(anyhow!(
            "{}: expected top-level JSON object",
            path.display()
        ));
    }
    let entry = zed_entry(def);
    let servers = root
        .as_object_mut()
        .unwrap()
        .entry("context_servers".to_string())
        .or_insert_with(|| json!({}));
    if !servers.is_object() {
        return Err(anyhow!(
            "{}: 'context_servers' must be an object",
            path.display()
        ));
    }
    let servers_map = servers.as_object_mut().unwrap();
    let existing = servers_map.get("sift").cloned();
    let already = existing.as_ref().is_some_and(|v| v == &entry);
    if already {
        return Ok(Action::AlreadyPresent);
    }
    if existing.is_some() && !opts.force {
        return Ok(Action::Skipped(
            "existing zed 'sift' context_server differs; re-run with --force".into(),
        ));
    }
    if opts.dry_run {
        return Ok(Action::Planned);
    }
    servers_map.insert("sift".into(), entry);
    write_json_pretty(path, &root)?;
    Ok(if existing.is_some() {
        Action::Updated
    } else {
        Action::Installed
    })
}

fn uninstall_zed(path: &Path, opts: &IntegrateOptions) -> Result<Action> {
    uninstall_json_under_key(path, opts, "context_servers")
}

// ---------------------------------------------------------------------------
// VS Code
// ---------------------------------------------------------------------------

fn vscode_entry(def: &SiftServerDef) -> Value {
    if let Some(url) = &def.http_url {
        return json!({ "type": "http", "url": url });
    }
    let mut v = json!({
        "type": "stdio",
        "command": def.exe.display().to_string(),
        "args": def.args,
    });
    if !def.env.is_empty() {
        v["env"] = json!(def.env);
    }
    v
}

fn vscode_config_path(home: &Path) -> PathBuf {
    // Per VS Code 1.103+, user-scope mcp.json lives in the user profile dir.
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/Code/User/mcp.json")
    } else if cfg!(target_os = "windows") {
        home.join("AppData/Roaming/Code/User/mcp.json")
    } else {
        home.join(".config/Code/User/mcp.json")
    }
}

fn detect_vscode(home: &Path) -> bool {
    let user_dir = if cfg!(target_os = "macos") {
        home.join("Library/Application Support/Code/User")
    } else if cfg!(target_os = "windows") {
        home.join("AppData/Roaming/Code/User")
    } else {
        home.join(".config/Code/User")
    };
    user_dir.is_dir()
}

fn install_vscode(path: &Path, def: &SiftServerDef, opts: &IntegrateOptions) -> Result<Action> {
    let mut root = read_json_or_default(path)?;
    if !root.is_object() {
        return Err(anyhow!(
            "{}: expected top-level JSON object",
            path.display()
        ));
    }
    // VS Code uses top-level "servers" (not "mcpServers").
    let entry = vscode_entry(def);
    let servers = root
        .as_object_mut()
        .unwrap()
        .entry("servers".to_string())
        .or_insert_with(|| json!({}));
    if !servers.is_object() {
        return Err(anyhow!("{}: 'servers' must be an object", path.display()));
    }
    let servers_map = servers.as_object_mut().unwrap();
    let existing = servers_map.get("sift").cloned();
    let already = existing.as_ref().is_some_and(|v| v == &entry);
    if already {
        return Ok(Action::AlreadyPresent);
    }
    if existing.is_some() && !opts.force {
        return Ok(Action::Skipped(
            "existing vscode 'sift' server differs; re-run with --force".into(),
        ));
    }
    if opts.dry_run {
        return Ok(Action::Planned);
    }
    servers_map.insert("sift".into(), entry);
    write_json_pretty(path, &root)?;
    Ok(if existing.is_some() {
        Action::Updated
    } else {
        Action::Installed
    })
}

fn uninstall_vscode(path: &Path, opts: &IntegrateOptions) -> Result<Action> {
    uninstall_json_under_key(path, opts, "servers")
}

// ---------------------------------------------------------------------------
// Atomic write (temp file + rename)
// ---------------------------------------------------------------------------

fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating directory {}", parent.display()))?;
    let tmp = parent.join(format!(
        ".{}.sift-integrate.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("out")
    ));
    std::fs::write(&tmp, content)
        .with_context(|| format!("writing temp file {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Summary rendering
// ---------------------------------------------------------------------------

fn print_summary(reports: &[Report], opts: &IntegrateOptions) {
    let mut any_written = false;
    for r in reports {
        let (label, colored_label) = match &r.action {
            Action::Installed => ("installed", "installed".green().to_string()),
            Action::Updated => ("updated", "updated".green().to_string()),
            Action::AlreadyPresent => ("unchanged", "unchanged".dimmed().to_string()),
            Action::NotDetected => ("not detected", "not detected".dimmed().to_string()),
            Action::Uninstalled => ("uninstalled", "uninstalled".yellow().to_string()),
            Action::NothingToUninstall => ("no entry", "no entry".dimmed().to_string()),
            Action::Planned => ("would change", "would change".cyan().to_string()),
            Action::Skipped(_) => ("skipped", "skipped".yellow().to_string()),
        };
        if matches!(
            r.action,
            Action::Installed | Action::Updated | Action::Uninstalled
        ) {
            any_written = true;
        }
        let _ = label;
        println!(
            "  {:<14} {:<14} {}",
            r.platform,
            colored_label,
            r.config_path.display()
        );
        if let Action::Skipped(reason) = &r.action {
            println!("    {}", reason.dimmed());
        }
    }
    if opts.dry_run {
        println!("\n{}", "Dry run — no files were modified.".dimmed());
    } else if any_written {
        println!(
            "\n{}",
            "Restart your AI clients to pick up the new MCP server.".bold()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_def(tmp: &Path) -> SiftServerDef {
        SiftServerDef {
            exe: tmp.join("sift-bin"),
            args: vec!["mcp".into()],
            env: {
                let mut m = BTreeMap::new();
                m.insert("ORT_DYLIB_PATH".into(), "/x/libort.dylib".into());
                m
            },
            http_url: None,
        }
    }

    fn mk_opts() -> IntegrateOptions {
        IntegrateOptions {
            platforms: vec![],
            dry_run: false,
            force: false,
            uninstall: false,
            exe: None,
            http_url: None,
            list: false,
        }
    }

    #[test]
    fn install_creates_mcp_servers_json_when_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        let def = test_def(tmp.path());

        let action = install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap();
        assert_eq!(action, Action::Installed);

        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        let entry = &parsed["mcpServers"]["sift"];
        assert_eq!(entry["args"], json!(["mcp"]));
        assert_eq!(entry["env"]["ORT_DYLIB_PATH"], json!("/x/libort.dylib"));
        assert!(entry["command"].as_str().unwrap().ends_with("sift-bin"));
    }

    #[test]
    fn install_is_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        let def = test_def(tmp.path());

        install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap();
        let action = install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap();
        assert_eq!(action, Action::AlreadyPresent);
    }

    #[test]
    fn install_preserves_unrelated_keys() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("claude.json");
        std::fs::write(
            &cfg,
            r#"{"userID": "abc", "projects": {"x": 1}, "mcpServers": {"other": {"command": "foo"}}}"#,
        )
        .unwrap();

        let def = test_def(tmp.path());
        install_mcp_servers_json(&cfg, &def, &mk_opts(), true).unwrap();

        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(parsed["userID"], json!("abc"));
        assert_eq!(parsed["projects"]["x"], json!(1));
        assert!(parsed["mcpServers"]["other"]["command"].is_string());
        assert!(parsed["mcpServers"]["sift"]["command"].is_string());
    }

    #[test]
    fn install_without_force_refuses_to_overwrite_differing_entry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        std::fs::write(
            &cfg,
            r#"{"mcpServers": {"sift": {"command": "/old/bin/sift", "args": []}}}"#,
        )
        .unwrap();

        let def = test_def(tmp.path());
        let action = install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap();
        assert!(matches!(action, Action::Skipped(_)));
        // File contents unchanged.
        let body = std::fs::read_to_string(&cfg).unwrap();
        assert!(body.contains("/old/bin/sift"));
    }

    #[test]
    fn install_with_force_updates_existing_entry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        std::fs::write(
            &cfg,
            r#"{"mcpServers": {"sift": {"command": "/old/bin/sift", "args": []}}}"#,
        )
        .unwrap();

        let def = test_def(tmp.path());
        let mut opts = mk_opts();
        opts.force = true;
        let action = install_mcp_servers_json(&cfg, &def, &opts, false).unwrap();
        assert_eq!(action, Action::Updated);
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        assert!(parsed["mcpServers"]["sift"]["command"]
            .as_str()
            .unwrap()
            .ends_with("sift-bin"));
    }

    #[test]
    fn dry_run_does_not_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        let def = test_def(tmp.path());
        let mut opts = mk_opts();
        opts.dry_run = true;
        let action = install_mcp_servers_json(&cfg, &def, &opts, false).unwrap();
        assert_eq!(action, Action::Planned);
        assert!(!cfg.exists());
    }

    #[test]
    fn zed_rejects_http() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("zed/settings.json");
        let mut def = test_def(tmp.path());
        def.http_url = Some("http://127.0.0.1:7821/mcp".into());

        let action = install_zed(&cfg, &def, &mk_opts()).unwrap();
        assert_eq!(
            action,
            Action::Skipped("zed does not support HTTP MCP transport".into())
        );
        assert!(!cfg.exists());
    }

    #[test]
    fn home_dir_prefers_home() {
        let home = home_dir_from_env(|key| match key {
            "HOME" => Some(OsString::from("/tmp/home")),
            "USERPROFILE" => Some(OsString::from("/tmp/userprofile")),
            _ => None,
        });
        assert_eq!(home, Some(PathBuf::from("/tmp/home")));
    }

    #[test]
    fn home_dir_falls_back_to_userprofile() {
        let home = home_dir_from_env(|key| match key {
            "USERPROFILE" => Some(OsString::from(r"C:\Users\ray")),
            _ => None,
        });
        assert_eq!(home, Some(PathBuf::from(r"C:\Users\ray")));
    }

    #[test]
    fn home_dir_falls_back_to_home_drive_and_path() {
        let home = home_dir_from_env(|key| match key {
            "HOMEDRIVE" => Some(OsString::from("C:")),
            "HOMEPATH" => Some(OsString::from(r"\Users\ray")),
            _ => None,
        });
        assert_eq!(home, Some(PathBuf::from(r"C:\Users\ray")));
    }

    #[test]
    fn uninstall_removes_sift_entry_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        std::fs::write(
            &cfg,
            r#"{"mcpServers": {"sift": {"command": "x"}, "other": {"command": "y"}}}"#,
        )
        .unwrap();
        let opts = mk_opts();
        let action = uninstall_mcp_servers_json(&cfg, &opts, false).unwrap();
        assert_eq!(action, Action::Uninstalled);
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        assert!(parsed["mcpServers"].get("sift").is_none());
        assert_eq!(parsed["mcpServers"]["other"]["command"], json!("y"));
    }

    #[test]
    fn uninstall_when_missing_is_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("cursor/mcp.json");
        let opts = mk_opts();
        let action = uninstall_mcp_servers_json(&cfg, &opts, false).unwrap();
        assert_eq!(action, Action::NothingToUninstall);
    }

    #[test]
    fn codex_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let def = test_def(tmp.path());
        let action = install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Installed);
        let text = std::fs::read_to_string(&cfg).unwrap();
        assert!(text.contains("[mcp_servers.sift]"));
        assert!(text.contains("[features]"));
        assert!(text.contains("codex_hooks = true"));
        assert!(text.contains("command"));
        assert!(text.contains("ORT_DYLIB_PATH"));
        let hooks_path = tmp.path().join(".codex/hooks.json");
        let hooks: Value =
            serde_json::from_str(&std::fs::read_to_string(&hooks_path).unwrap()).unwrap();
        assert_eq!(hooks["hooks"]["PostToolUse"][0]["matcher"], json!("Bash"));

        let action = install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::AlreadyPresent);

        let action = uninstall_codex_with_project_root(&cfg, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Uninstalled);
        let text = std::fs::read_to_string(&cfg).unwrap();
        assert!(!text.contains("mcp_servers"));
        let hooks: Value =
            serde_json::from_str(&std::fs::read_to_string(&hooks_path).unwrap()).unwrap();
        assert_eq!(hooks, json!({}));
    }

    #[test]
    fn codex_rejects_http() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let mut def = test_def(tmp.path());
        def.http_url = Some("http://localhost:7821/mcp".into());
        let action = install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        assert!(matches!(action, Action::Skipped(_)));
    }

    #[test]
    fn codex_preserves_other_tables() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        std::fs::write(
            &cfg,
            r#"
[profile.default]
model = "gpt-5"

[mcp_servers.other]
command = "/bin/other"
"#,
        )
        .unwrap();
        let def = test_def(tmp.path());
        install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        assert!(doc.get("profile").is_some());
        let servers = doc.get("mcp_servers").unwrap().as_table().unwrap();
        assert!(servers.contains_key("sift"));
        assert!(servers.contains_key("other"));
        let features = doc.get("features").unwrap().as_table().unwrap();
        assert_eq!(
            features.get("codex_hooks"),
            Some(&toml::Value::Boolean(true))
        );
    }

    #[test]
    fn codex_hooks_merge_and_uninstall_preserve_unrelated_handlers() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hooks_path = tmp.path().join(".codex/hooks.json");
        std::fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
        std::fs::write(
            &hooks_path,
            r#"{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "/old/sift memory ingest --event post-tool-use",
            "statusMessage": "sift: ingest Bash tool use",
            "timeout": 5
          },
          {
            "type": "command",
            "command": "/bin/other",
            "statusMessage": "other"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/old/sift memory ingest --event stop",
            "statusMessage": "sift: ingest stop summary",
            "timeout": 5
          },
          {
            "type": "command",
            "command": "/old/sift memory consolidate --quiet",
            "statusMessage": "sift: consolidate memory",
            "timeout": 30
          },
          {
            "type": "command",
            "command": "/bin/stop"
          }
        ]
      }
    ]
  }
}"#,
        )
        .unwrap();

        let cfg = tmp.path().join("config.toml");
        let def = test_def(tmp.path());
        let action = install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Updated);

        let hooks: Value =
            serde_json::from_str(&std::fs::read_to_string(&hooks_path).unwrap()).unwrap();
        let post_tool_use = hooks["hooks"]["PostToolUse"].as_array().unwrap();
        assert_eq!(post_tool_use.len(), 2);
        assert_eq!(
            post_tool_use[0]["hooks"],
            json!([{
                "type": "command",
                "command": "/bin/other",
                "statusMessage": "other"
            }])
        );
        assert_eq!(
            post_tool_use[1]["hooks"][0]["command"],
            json!(format!(
                "{} memory ingest --event post-tool-use",
                def.exe.display()
            ))
        );

        let stop = hooks["hooks"]["Stop"].as_array().unwrap();
        assert_eq!(stop.len(), 2);
        assert_eq!(
            stop[0]["hooks"],
            json!([{
                "type": "command",
                "command": "/bin/stop"
            }])
        );
        let stop_hooks = stop[1]["hooks"].as_array().unwrap();
        assert_eq!(
            stop_hooks[0]["command"],
            json!(format!("{} memory ingest --event stop", def.exe.display()))
        );
        assert_eq!(
            stop_hooks[1]["command"],
            json!(format!("{} memory consolidate --quiet", def.exe.display()))
        );

        let action = uninstall_codex_with_project_root(&cfg, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Uninstalled);
        let hooks: Value =
            serde_json::from_str(&std::fs::read_to_string(&hooks_path).unwrap()).unwrap();
        assert_eq!(
            hooks["hooks"]["PostToolUse"][0]["hooks"],
            json!([{
                "type": "command",
                "command": "/bin/other",
                "statusMessage": "other"
            }])
        );
        assert_eq!(
            hooks["hooks"]["Stop"][0]["hooks"],
            json!([{
                "type": "command",
                "command": "/bin/stop"
            }])
        );
    }

    #[test]
    fn codex_uninstall_leaves_features_table_untouched() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        std::fs::write(
            &cfg,
            r#"
[features]
codex_hooks = true
apps = true

[mcp_servers.sift]
command = "/bin/sift"
args = ["mcp"]
"#,
        )
        .unwrap();

        let action = uninstall_codex_with_project_root(&cfg, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Uninstalled);
        let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        let features = doc.get("features").unwrap().as_table().unwrap();
        assert_eq!(
            features.get("codex_hooks"),
            Some(&toml::Value::Boolean(true))
        );
        assert_eq!(features.get("apps"), Some(&toml::Value::Boolean(true)));
    }

    #[test]
    fn codex_stop_consolidation_writes_agents_md() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("config.toml");
        let def = test_def(tmp.path());
        let action = install_codex_with_project_root(&cfg, &def, &mk_opts(), tmp.path()).unwrap();
        assert_eq!(action, Action::Installed);

        let memory_dir = tmp.path().join(".sift/indexes/default/memory");
        let episodes = sift_memory::episodes::EpisodeStore::open(&memory_dir).unwrap();
        let content = r#"{
            "session_id": "codex-session",
            "last_assistant_message": "Finished the Codex hook parity implementation.",
            "stop_hook_active": true
        }"#;
        episodes.ingest("codex-session", "stop", content).unwrap();

        let store = sift_memory::MemoryStore::open(&memory_dir).unwrap();
        let report = sift_memory::consolidation::run_consolidation(
            &store,
            &episodes,
            &sift_memory::ConsolidationConfig::default(),
        )
        .unwrap();
        assert_eq!(report.episodes_processed, 1);

        let rule_report = sift_memory::rules::generate_all_rules(&store, tmp.path()).unwrap();
        assert!(rule_report.formats.contains(&"AGENTS.md".to_string()));
        assert!(tmp.path().join("AGENTS.md").exists());
    }

    #[test]
    fn opencode_uses_command_array_and_environment() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("opencode.json");
        let def = test_def(tmp.path());
        install_opencode(&cfg, &def, &mk_opts()).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        let entry = &parsed["mcp"]["sift"];
        assert_eq!(entry["type"], json!("local"));
        assert!(entry["command"].is_array());
        assert_eq!(entry["command"][1], json!("mcp"));
        assert_eq!(
            entry["environment"]["ORT_DYLIB_PATH"],
            json!("/x/libort.dylib")
        );
        assert_eq!(entry["enabled"], json!(true));
    }

    #[test]
    fn zed_uses_context_servers_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("settings.json");
        let def = test_def(tmp.path());
        install_zed(&cfg, &def, &mk_opts()).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        let entry = &parsed["context_servers"]["sift"];
        assert_eq!(entry["source"], json!("custom"));
        assert_eq!(entry["command"]["args"], json!(["mcp"]));
        assert_eq!(
            entry["command"]["env"]["ORT_DYLIB_PATH"],
            json!("/x/libort.dylib")
        );
    }

    #[test]
    fn vscode_uses_servers_key_and_stdio_type() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("mcp.json");
        let def = test_def(tmp.path());
        install_vscode(&cfg, &def, &mk_opts()).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        let entry = &parsed["servers"]["sift"];
        assert_eq!(entry["type"], json!("stdio"));
        assert_eq!(entry["args"], json!(["mcp"]));
    }

    #[test]
    fn http_url_produces_type_http_entry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("mcp.json");
        let mut def = test_def(tmp.path());
        def.http_url = Some("http://localhost:7821/mcp".into());
        install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(parsed["mcpServers"]["sift"]["type"], json!("http"));
        assert_eq!(
            parsed["mcpServers"]["sift"]["url"],
            json!("http://localhost:7821/mcp")
        );
    }

    #[test]
    fn rejects_non_object_root() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("mcp.json");
        std::fs::write(&cfg, r#"["not", "an", "object"]"#).unwrap();
        let def = test_def(tmp.path());
        let err = install_mcp_servers_json(&cfg, &def, &mk_opts(), false).unwrap_err();
        assert!(err.to_string().contains("top-level JSON object"));
    }
}
