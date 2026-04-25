//! `sift memory-tool` — Anthropic Memory Tool (`memory_20250818`) adapter.
//!
//! This is a file-shaped projection over sift's persistent memory store.
//! Entity pages under `/memories/entities/*.md` are rendered from sift memory
//! and edits are translated into entity/observation/relation mutations.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sift_core::Config;
use sift_memory::{
    Entity, EntityType, MemoryError, MemoryStore, ObservationRewrite, PageMutationPlan, Relation,
    RelationAddition, RelationRewrite,
};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const ROOT_PREFIX: &str = "/memories";
const ENTITIES_PREFIX: &str = "/memories/entities";
const TOOL_SOURCE: &str = "memory-tool";
const MIGRATION_SOURCE: &str = "memory-tool:migrate";
const LEGACY_ENTITY: &str = "legacy-memory-tool notes";

/// Parsed Memory Tool command. The `command` field on the input JSON
/// discriminates the variant.
#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Op {
    View {
        path: String,
        #[serde(default)]
        view_range: Option<[i64; 2]>,
    },
    Create {
        path: String,
        file_text: String,
    },
    StrReplace {
        path: String,
        old_str: String,
        new_str: String,
    },
    Insert {
        path: String,
        insert_line: usize,
        insert_text: String,
    },
    Delete {
        path: String,
    },
    Rename {
        old_path: String,
        new_path: String,
    },
}

enum VirtualPath {
    Root,
    EntitiesDir,
    EntityPage { slug: String, entity_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PageDoc {
    entity: String,
    entity_id: String,
    entity_type: EntityType,
    page_revision: String,
    title: String,
    observations: Vec<PageObservation>,
    relations: Vec<PageRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PageObservation {
    anchor: Option<String>,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PageRelation {
    anchor: Option<String>,
    relation_type: String,
    target: String,
}

struct MemoryTool {
    memory: MemoryStore,
}

impl MemoryTool {
    fn new(memory_dir: impl AsRef<Path>) -> Result<Self> {
        let memory = MemoryStore::open(memory_dir.as_ref()).with_context(|| {
            format!("opening memory store at {}", memory_dir.as_ref().display())
        })?;
        Ok(Self { memory })
    }

    fn exec(&self, input: &Value) -> Result<String> {
        let op: Op = serde_json::from_value(input.clone())
            .context("memory tool input did not match the memory_20250818 schema")?;
        match op {
            Op::View { path, view_range } => self.view(&path, view_range),
            Op::Create { path, file_text } => self.create(&path, &file_text),
            Op::StrReplace {
                path,
                old_str,
                new_str,
            } => self.str_replace(&path, &old_str, &new_str),
            Op::Insert {
                path,
                insert_line,
                insert_text,
            } => self.insert(&path, insert_line, &insert_text),
            Op::Delete { path } => self.delete(&path),
            Op::Rename { old_path, new_path } => self.rename(&old_path, &new_path),
        }
    }

    fn view(&self, vpath: &str, range: Option<[i64; 2]>) -> Result<String> {
        match parse_virtual_path(vpath)? {
            VirtualPath::Root => Ok("entities/\n".to_string()),
            VirtualPath::EntitiesDir => {
                let mut entries: Vec<String> = self
                    .memory
                    .all_entities()?
                    .into_iter()
                    .map(|entity| entity_page_filename(&entity))
                    .collect();
                entries.sort();
                if entries.is_empty() {
                    return Ok("(empty directory)\n".to_string());
                }
                Ok(entries.join("\n") + "\n")
            }
            VirtualPath::EntityPage { slug, entity_id } => {
                let entity = self.entity_by_path(&slug, &entity_id)?;
                let content = self.render_entity_page(&entity)?;
                if let Some([start, end]) = range {
                    view_range(&content, start, end)
                } else {
                    Ok(with_line_numbers(&content, 1))
                }
            }
        }
    }

    fn create(&self, vpath: &str, text: &str) -> Result<String> {
        let slug = parse_create_virtual_path(vpath)?;

        let parsed = parse_page(text, ParseMode::Create)?;
        if parsed.title != parsed.entity {
            bail!("page title must match frontmatter entity");
        }
        if slugify(&parsed.entity) != slug {
            bail!(
                "entity page path does not match the canonical slug for {:?}",
                parsed.entity
            );
        }
        if self.memory.get_entity(&parsed.entity)?.is_some() {
            bail!(
                "entity page already exists: {}",
                entity_page_virtual_path(
                    &self
                        .memory
                        .get_entity(&parsed.entity)?
                        .ok_or_else(|| anyhow!("entity disappeared"))?
                )
            );
        }

        let observations: Vec<String> = parsed
            .observations
            .into_iter()
            .map(|obs| obs.text)
            .collect();
        let relations: Vec<RelationAddition> = parsed
            .relations
            .into_iter()
            .map(|rel| RelationAddition {
                relation_type: rel.relation_type,
                target_name: rel.target,
            })
            .collect();

        let entity_id = match self.memory.create_entity_page(
            &parsed.entity,
            parsed.entity_type,
            &observations,
            &relations,
            TOOL_SOURCE,
        ) {
            Ok(entity_id) => entity_id,
            Err(MemoryError::InvalidInput(message))
                if message == format!("entity already exists: {}", parsed.entity) =>
            {
                let existing = self
                    .memory
                    .get_entity(&parsed.entity)?
                    .ok_or_else(|| anyhow!("entity disappeared"))?;
                bail!(
                    "entity page already exists: {}",
                    entity_page_virtual_path(&existing)
                );
            }
            Err(err) => return Err(err.into()),
        };
        self.memory.save()?;
        let entity = self
            .memory
            .get_entity_by_id(&entity_id)?
            .ok_or_else(|| anyhow!("created entity disappeared"))?;
        Ok(format!(
            "Entity page created at {}",
            entity_page_virtual_path(&entity)
        ))
    }

    fn str_replace(&self, vpath: &str, old_str: &str, new_str: &str) -> Result<String> {
        match parse_virtual_path(vpath)? {
            VirtualPath::EntityPage { slug, entity_id } => {
                let entity = self.entity_by_path(&slug, &entity_id)?;
                let current_page = self.render_entity_page(&entity)?;

                let count = current_page.matches(old_str).count();
                match count {
                    0 => bail!("page changed, re-view and retry"),
                    1 => {}
                    n => bail!("old_str matched {n} times in {vpath}; re-view and retry"),
                }

                let updated_page = current_page.replacen(old_str, new_str, 1);
                self.apply_updated_page(&entity, &current_page, &updated_page)
            }
            _ => bail!("str_replace is only supported for entity pages"),
        }
    }

    #[allow(clippy::unused_self)]
    fn insert(&self, vpath: &str, insert_line: usize, insert_text: &str) -> Result<String> {
        let _ = insert_line;
        let _ = insert_text;
        match parse_virtual_path(vpath)? {
            VirtualPath::EntityPage { .. } => {
                bail!("insert is not supported for entity pages in v1; re-view and use str_replace")
            }
            _ => bail!("insert is only supported for entity pages"),
        }
    }

    fn delete(&self, vpath: &str) -> Result<String> {
        match parse_virtual_path(vpath)? {
            VirtualPath::EntityPage { slug, entity_id } => {
                let entity = self.entity_by_path(&slug, &entity_id)?;
                self.memory.delete_entity(&entity.name)?;
                self.memory.save()?;
                Ok(format!("Deleted {vpath}"))
            }
            VirtualPath::Root | VirtualPath::EntitiesDir => {
                bail!("refusing to delete virtual memory directories")
            }
        }
    }

    #[allow(clippy::unused_self)]
    fn rename(&self, _old_vpath: &str, _new_vpath: &str) -> Result<String> {
        bail!("rename is not supported for memory-backed entity pages")
    }

    fn apply_updated_page(
        &self,
        entity: &Entity,
        current_page: &str,
        updated_page: &str,
    ) -> Result<String> {
        let current = parse_page(current_page, ParseMode::Existing)?;
        let updated = parse_page(updated_page, ParseMode::Existing)?;

        if current.page_revision != updated.page_revision {
            bail!("page changed, re-view and retry");
        }
        ensure_immutable_fields(&current, &updated)?;

        if normalized_page(&current) == normalized_page(&updated) {
            return Ok(format!(
                "No changes applied to {}",
                entity_page_virtual_path(entity)
            ));
        }

        let plan = build_mutation_plan(entity, &current, &updated)?;
        self.memory.apply_page_mutation_plan(&plan, TOOL_SOURCE)?;
        self.memory.save()?;

        Ok(format!("Updated {}", entity_page_virtual_path(entity)))
    }

    fn entity_by_path(&self, slug: &str, entity_id: &str) -> Result<Entity> {
        let entity = self
            .memory
            .get_entity_by_id(entity_id)?
            .ok_or_else(|| anyhow!("not found: /memories/entities/{slug}--{entity_id}.md"))?;
        let canonical_slug = slugify(&entity.name);
        if canonical_slug != slug {
            bail!(
                "entity page path is not canonical; re-view and use {}",
                entity_page_virtual_path(&entity)
            );
        }
        Ok(entity)
    }

    fn render_entity_page(&self, entity: &Entity) -> Result<String> {
        let mut observations = self.memory.get_entity_observations(&entity.id)?;
        observations.sort_by(|a, b| {
            a.observed_at
                .cmp(&b.observed_at)
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut relations: Vec<Relation> = self
            .memory
            .get_entity_relations(&entity.id)?
            .into_iter()
            .filter(|relation| relation.from_entity == entity.id)
            .collect();
        relations.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });

        let relation_lines: Vec<PageRelation> = relations
            .into_iter()
            .map(|relation| {
                let target = self
                    .memory
                    .get_entity_by_id(&relation.to_entity)?
                    .ok_or_else(|| {
                        anyhow!("related entity not found for relation {}", relation.id)
                    })?;
                Ok(PageRelation {
                    anchor: Some(relation.id),
                    relation_type: relation.relation_type,
                    target: target.name,
                })
            })
            .collect::<Result<_>>()?;

        let observation_lines: Vec<PageObservation> = observations
            .into_iter()
            .map(|obs| PageObservation {
                anchor: Some(obs.id),
                text: obs.content,
            })
            .collect();

        let basis = render_page_basis(entity, &observation_lines, &relation_lines);
        let revision = page_revision(&basis);
        Ok(render_page(
            entity,
            &revision,
            &observation_lines,
            &relation_lines,
        ))
    }

    fn migrate_legacy(&self, source_root: &Path) -> Result<(usize, usize)> {
        let entity_id =
            self.memory
                .save_entity(LEGACY_ENTITY, EntityType::Concept, 1.0, MIGRATION_SOURCE)?;

        let mut imported = 0usize;
        let mut skipped = 0usize;
        for entry in WalkDir::new(source_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let rel = entry
                .path()
                .strip_prefix(source_root)
                .unwrap_or_else(|_| entry.path())
                .display()
                .to_string();
            let text = std::fs::read_to_string(entry.path()).with_context(|| {
                format!("reading legacy memory file {}", entry.path().display())
            })?;
            let imported_text = format!("[legacy:{}]\n{}", rel, text.trim_end());
            let existing = self
                .memory
                .get_entity_observations(&entity_id)?
                .into_iter()
                .any(|obs| obs.source == MIGRATION_SOURCE && obs.content == imported_text);
            if existing {
                skipped += 1;
                continue;
            }
            self.memory
                .add_observation(&entity_id, &imported_text, 1.0, MIGRATION_SOURCE)?;
            imported += 1;
        }

        self.memory.save()?;
        Ok((imported, skipped))
    }
}

#[derive(Clone, Copy)]
enum ParseMode {
    Existing,
    Create,
}

fn parse_virtual_path(vpath: &str) -> Result<VirtualPath> {
    if vpath == ROOT_PREFIX {
        return Ok(VirtualPath::Root);
    }
    if vpath == ENTITIES_PREFIX {
        return Ok(VirtualPath::EntitiesDir);
    }
    if let Some(rest) = vpath.strip_prefix(&(ENTITIES_PREFIX.to_string() + "/")) {
        if let Some(name) = rest.strip_suffix(".md") {
            if !name.is_empty() && !name.contains('/') {
                if let Some((slug, entity_id)) = name.rsplit_once("--") {
                    if !slug.is_empty() && !entity_id.is_empty() {
                        return Ok(VirtualPath::EntityPage {
                            slug: slug.to_string(),
                            entity_id: entity_id.to_string(),
                        });
                    }
                }
            }
        }
    }
    bail!(
        "path must be /memories, /memories/entities, or /memories/entities/<slug>--<entity_id>.md"
    );
}

fn parse_create_virtual_path(vpath: &str) -> Result<String> {
    let rest = vpath
        .strip_prefix(&(ENTITIES_PREFIX.to_string() + "/"))
        .ok_or_else(|| anyhow!("create is only supported for /memories/entities/<slug>.md"))?;
    let slug = rest
        .strip_suffix(".md")
        .ok_or_else(|| anyhow!("create is only supported for /memories/entities/<slug>.md"))?;
    if slug.is_empty() || slug.contains('/') || slug.contains("--") {
        bail!("create is only supported for /memories/entities/<slug>.md");
    }
    Ok(slug.to_string())
}

fn entity_page_filename(entity: &Entity) -> String {
    format!("{}--{}.md", slugify(&entity.name), entity.id)
}

fn entity_page_virtual_path(entity: &Entity) -> String {
    format!("{ENTITIES_PREFIX}/{}", entity_page_filename(entity))
}

fn parse_page(text: &str, mode: ParseMode) -> Result<PageDoc> {
    let normalized = text.replace("\r\n", "\n");
    let mut lines: Vec<&str> = normalized.split('\n').collect();
    if lines.last().copied() == Some("") {
        lines.pop();
    }

    if lines.len() < 7 || lines.first().copied() != Some("---") {
        bail!("page must start with frontmatter");
    }
    let frontmatter_end = lines
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(idx, line)| (*line == "---").then_some(idx))
        .ok_or_else(|| anyhow!("frontmatter must end with ---"))?;

    let mut meta = BTreeMap::new();
    for line in &lines[1..frontmatter_end] {
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("invalid frontmatter line: {line:?}"))?;
        meta.insert(key.trim().to_string(), value.trim().to_string());
    }

    let entity = require_meta(&meta, "entity", mode)?;
    let entity_id = require_meta(&meta, "entity_id", mode)?;
    let entity_type = EntityType::parse(&require_meta(&meta, "entity_type", mode)?)
        .ok_or_else(|| anyhow!("invalid entity_type"))?;
    let page_revision = require_meta(&meta, "page_revision", mode)?;

    let mut idx = frontmatter_end + 1;
    expect_line(&lines, &mut idx, "")?;
    let title = lines
        .get(idx)
        .and_then(|line| line.strip_prefix("# "))
        .ok_or_else(|| anyhow!("page must contain a '# <entity>' title"))?
        .trim()
        .to_string();
    idx += 1;
    expect_line(&lines, &mut idx, "")?;
    expect_line(&lines, &mut idx, "## Observations")?;

    let observations = parse_observation_section(&lines, &mut idx, mode)?;
    expect_line(&lines, &mut idx, "## Relations")?;
    let relations = parse_relation_section(&lines, &mut idx, mode)?;

    if idx != lines.len() {
        bail!("unexpected trailing content in entity page");
    }

    Ok(PageDoc {
        entity,
        entity_id,
        entity_type,
        page_revision,
        title,
        observations,
        relations,
    })
}

fn parse_observation_section(
    lines: &[&str],
    idx: &mut usize,
    mode: ParseMode,
) -> Result<Vec<PageObservation>> {
    let mut items = Vec::new();
    while *idx < lines.len() {
        let line = lines[*idx];
        if line.is_empty() {
            *idx += 1;
            break;
        }
        if line == "## Relations" {
            break;
        }
        items.push(parse_observation_line(line, mode)?);
        *idx += 1;
    }
    Ok(items)
}

fn parse_relation_section(
    lines: &[&str],
    idx: &mut usize,
    mode: ParseMode,
) -> Result<Vec<PageRelation>> {
    let mut items = Vec::new();
    while *idx < lines.len() {
        let line = lines[*idx];
        if line.is_empty() {
            *idx += 1;
            continue;
        }
        items.push(parse_relation_line(line, mode)?);
        *idx += 1;
    }
    Ok(items)
}

fn parse_observation_line(line: &str, mode: ParseMode) -> Result<PageObservation> {
    let rest = line
        .trim()
        .strip_prefix("- ")
        .ok_or_else(|| anyhow!("invalid observation line: {line:?}"))?;
    let (anchor, text) = parse_anchor_and_text(rest, mode)?;
    if text.is_empty() {
        bail!("observation text must not be empty");
    }
    Ok(PageObservation { anchor, text })
}

fn parse_relation_line(line: &str, mode: ParseMode) -> Result<PageRelation> {
    let rest = line
        .trim()
        .strip_prefix("- ")
        .ok_or_else(|| anyhow!("invalid relation line: {line:?}"))?;
    let (anchor, body) = parse_anchor_and_text(rest, mode)?;
    let (relation_type, target) = body
        .split_once(" -> ")
        .ok_or_else(|| anyhow!("relation lines must use '<type> -> <target>'"))?;
    if relation_type.trim().is_empty() || target.trim().is_empty() {
        bail!("relation type and target must not be empty");
    }
    Ok(PageRelation {
        anchor,
        relation_type: relation_type.trim().to_string(),
        target: target.trim().to_string(),
    })
}

fn parse_anchor_and_text(rest: &str, mode: ParseMode) -> Result<(Option<String>, String)> {
    if let Some(without_bracket) = rest.strip_prefix('[') {
        if let Some((anchor, text)) = without_bracket.split_once("] ") {
            return Ok((Some(anchor.to_string()), text.trim().to_string()));
        }
    }
    if matches!(mode, ParseMode::Create) {
        Ok((None, rest.trim().to_string()))
    } else {
        bail!("existing entity pages must preserve anchored list items")
    }
}

fn expect_line(lines: &[&str], idx: &mut usize, expected: &str) -> Result<()> {
    let actual = lines
        .get(*idx)
        .copied()
        .ok_or_else(|| anyhow!("unexpected end of page"))?;
    if actual != expected {
        bail!("expected {:?}, found {:?}", expected, actual);
    }
    *idx += 1;
    Ok(())
}

fn require_meta(meta: &BTreeMap<String, String>, key: &str, mode: ParseMode) -> Result<String> {
    match meta.get(key) {
        Some(value) if !value.is_empty() => Ok(value.clone()),
        _ if matches!(mode, ParseMode::Create) && matches!(key, "entity_id" | "page_revision") => {
            Ok(String::new())
        }
        _ => bail!("missing required frontmatter key {key:?}"),
    }
}

fn render_page_basis(
    entity: &Entity,
    observations: &[PageObservation],
    relations: &[PageRelation],
) -> String {
    render_page_inner(entity, None, observations, relations)
}

fn render_page(
    entity: &Entity,
    revision: &str,
    observations: &[PageObservation],
    relations: &[PageRelation],
) -> String {
    render_page_inner(entity, Some(revision), observations, relations)
}

/// Shared body for `render_page` and `render_page_basis`.
///
/// `revision = None` produces the canonical "basis" form (no `page_revision`
/// line, no trailing newline) — fed back into `page_revision()` to compute the
/// content hash. `revision = Some(rev)` adds the `page_revision: <rev>` line
/// and a trailing newline; that's the form actually written to disk.
fn render_page_inner(
    entity: &Entity,
    revision: Option<&str>,
    observations: &[PageObservation],
    relations: &[PageRelation],
) -> String {
    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(out, "entity: {}", entity.name);
    let _ = writeln!(out, "entity_id: {}", entity.id);
    let _ = writeln!(out, "entity_type: {}", entity.entity_type);
    if let Some(rev) = revision {
        let _ = writeln!(out, "page_revision: {rev}");
    }
    out.push_str("---\n\n");
    let _ = writeln!(out, "# {}\n", entity.name);
    out.push_str("## Observations\n");
    for obs in observations {
        let _ = writeln!(
            out,
            "- [{}] {}",
            obs.anchor.as_deref().unwrap_or("new"),
            obs.text
        );
    }
    out.push_str("\n## Relations\n");
    for rel in relations {
        let _ = writeln!(
            out,
            "- [{}] {} -> {}",
            rel.anchor.as_deref().unwrap_or("new"),
            rel.relation_type,
            rel.target
        );
    }
    if revision.is_some() {
        out.push('\n');
    }
    out
}

fn page_revision(basis: &str) -> String {
    let hash = blake3::hash(basis.as_bytes()).to_hex().to_string();
    format!("rev_{}", &hash[..16])
}

fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

#[allow(clippy::type_complexity)]
fn normalized_page(
    page: &PageDoc,
) -> (
    String,
    String,
    EntityType,
    String,
    Vec<(Option<String>, String)>,
    Vec<(Option<String>, String, String)>,
) {
    (
        page.entity.clone(),
        page.entity_id.clone(),
        page.entity_type,
        page.title.clone(),
        page.observations
            .iter()
            .map(|obs| (obs.anchor.clone(), obs.text.clone()))
            .collect(),
        page.relations
            .iter()
            .map(|rel| {
                (
                    rel.anchor.clone(),
                    rel.relation_type.clone(),
                    rel.target.clone(),
                )
            })
            .collect(),
    )
}

fn ensure_immutable_fields(current: &PageDoc, updated: &PageDoc) -> Result<()> {
    if current.entity != updated.entity {
        bail!("entity name is read-only in v1");
    }
    if current.entity_id != updated.entity_id {
        bail!("entity_id is read-only in v1");
    }
    if current.title != updated.title {
        bail!("page title is read-only in v1");
    }
    Ok(())
}

fn build_mutation_plan(
    entity: &Entity,
    current: &PageDoc,
    updated: &PageDoc,
) -> Result<PageMutationPlan> {
    let mut plan = PageMutationPlan {
        entity_id: entity.id.clone(),
        entity_type_update: (current.entity_type != updated.entity_type)
            .then_some(updated.entity_type),
        ..PageMutationPlan::default()
    };

    build_observation_diff(&mut plan, &current.observations, &updated.observations)?;
    build_relation_diff(&mut plan, &current.relations, &updated.relations)?;
    Ok(plan)
}

fn build_observation_diff(
    plan: &mut PageMutationPlan,
    current: &[PageObservation],
    updated: &[PageObservation],
) -> Result<()> {
    let mut current_by_id = BTreeMap::new();
    for obs in current {
        current_by_id.insert(
            obs.anchor
                .clone()
                .ok_or_else(|| anyhow!("current observation missing anchor"))?,
            obs.text.clone(),
        );
    }

    let mut seen = HashSet::new();
    let mut rewrites = Vec::new();
    let mut additions = Vec::new();
    for obs in updated {
        match &obs.anchor {
            Some(id) => {
                let current_text = current_by_id
                    .get(id)
                    .ok_or_else(|| anyhow!("unknown observation anchor {id}"))?;
                if !seen.insert(id.clone()) {
                    bail!("duplicate observation anchor {id}");
                }
                if current_text != &obs.text {
                    rewrites.push((id.clone(), obs.text.clone()));
                }
            }
            None => additions.push(obs.text.clone()),
        }
    }

    let removals: Vec<String> = current_by_id
        .keys()
        .filter(|id| !seen.contains(*id))
        .cloned()
        .collect();

    if !rewrites.is_empty() && (!removals.is_empty() || !additions.is_empty()) {
        bail!("unsupported mixed observation edit; re-view and retry");
    }

    if rewrites.is_empty()
        && removals.len() == 1
        && additions.len() == 1
        && current.len() == updated.len()
    {
        rewrites.push((removals[0].clone(), additions.pop().unwrap()));
    } else if !removals.is_empty() && !additions.is_empty() {
        bail!("unsupported observation edit; re-view and retry");
    }

    if !additions.is_empty() {
        bail!("adding observation bullets via str_replace is not supported in v1");
    }

    plan.observation_invalidations.extend(removals);
    plan.observation_rewrites
        .extend(
            rewrites
                .into_iter()
                .map(|(observation_id, new_content)| ObservationRewrite {
                    observation_id,
                    new_content,
                }),
        );
    Ok(())
}

fn build_relation_diff(
    plan: &mut PageMutationPlan,
    current: &[PageRelation],
    updated: &[PageRelation],
) -> Result<()> {
    let mut current_by_id = BTreeMap::new();
    for rel in current {
        current_by_id.insert(
            rel.anchor
                .clone()
                .ok_or_else(|| anyhow!("current relation missing anchor"))?,
            (rel.relation_type.clone(), rel.target.clone()),
        );
    }

    let mut seen = HashSet::new();
    let mut rewrites = Vec::new();
    let mut additions = Vec::new();
    for rel in updated {
        match &rel.anchor {
            Some(id) => {
                let current_value = current_by_id
                    .get(id)
                    .ok_or_else(|| anyhow!("unknown relation anchor {id}"))?;
                if !seen.insert(id.clone()) {
                    bail!("duplicate relation anchor {id}");
                }
                let next_value = (rel.relation_type.clone(), rel.target.clone());
                if current_value != &next_value {
                    rewrites.push((id.clone(), next_value));
                }
            }
            None => additions.push((rel.relation_type.clone(), rel.target.clone())),
        }
    }

    let removals: Vec<String> = current_by_id
        .keys()
        .filter(|id| !seen.contains(*id))
        .cloned()
        .collect();

    if !rewrites.is_empty() && (!removals.is_empty() || !additions.is_empty()) {
        bail!("unsupported mixed relation edit; re-view and retry");
    }

    if rewrites.is_empty()
        && removals.len() == 1
        && additions.len() == 1
        && current.len() == updated.len()
    {
        rewrites.push((removals[0].clone(), additions.pop().unwrap()));
    } else if !removals.is_empty() && !additions.is_empty() {
        bail!("unsupported relation edit; re-view and retry");
    }

    if !additions.is_empty() {
        bail!("adding relation bullets via str_replace is not supported in v1");
    }

    plan.relation_invalidations.extend(removals);
    plan.relation_rewrites.extend(rewrites.into_iter().map(
        |(relation_id, (relation_type, target_name))| RelationRewrite {
            relation_id,
            relation_type,
            target_name,
        },
    ));
    Ok(())
}

fn view_range(content: &str, start: i64, end: i64) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(String::new());
    }
    if start < 1 {
        bail!("view_range start must be >= 1 (got {start})");
    }
    let start_idx = (start as usize).saturating_sub(1);
    if start_idx >= lines.len() {
        bail!(
            "view_range start {start} is past end of file (has {} lines)",
            lines.len()
        );
    }
    let end_idx = if end < 0 {
        lines.len()
    } else {
        (end as usize).min(lines.len())
    };
    if end_idx < start_idx + 1 {
        bail!("view_range end must be >= start");
    }
    Ok(with_line_numbers_iter(
        lines[start_idx..end_idx].iter().copied(),
        start as usize,
    ))
}

fn with_line_numbers(text: &str, start: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    with_line_numbers_iter(text.lines(), start)
}

fn with_line_numbers_iter<'a>(lines: impl IntoIterator<Item = &'a str>, start: usize) -> String {
    let mut out = String::new();
    for (i, line) in lines.into_iter().enumerate() {
        let num = start + i;
        let _ = writeln!(out, "{num:>6}\t{line}");
    }
    out
}

fn default_memory_dir(config: &Config, override_root: Option<PathBuf>) -> Result<PathBuf> {
    Ok(match override_root {
        Some(root) => root,
        None => config.index_dir()?.join("memory"),
    })
}

fn default_legacy_root() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".sift/memories"))
}

/// Execute a memory_20250818 tool_use from stdin.
pub fn exec(config: &Config, root: Option<PathBuf>) -> Result<()> {
    let tool = MemoryTool::new(default_memory_dir(config, root)?)?;
    let input_text = std::io::read_to_string(std::io::stdin())
        .context("reading memory-tool input from stdin")?;
    if input_text.trim().is_empty() {
        bail!("no input on stdin; expected a memory tool_use JSON object");
    }
    let input: Value = serde_json::from_str(&input_text).context("parsing memory tool_use JSON")?;
    print!("{}", tool.exec(&input)?);
    Ok(())
}

/// Print the backing memory directory used by the adapter.
pub fn path_cmd(config: &Config, root: Option<PathBuf>) -> Result<()> {
    println!("{}", default_memory_dir(config, root)?.display());
    Ok(())
}

/// Import legacy file-backed memory-tool notes into sift memory.
pub fn migrate(config: &Config, source: Option<PathBuf>) -> Result<()> {
    let source_root = source.unwrap_or(default_legacy_root()?);
    let tool = MemoryTool::new(default_memory_dir(config, None)?)?;
    let (imported, skipped) = tool.migrate_legacy(&source_root)?;
    println!(
        "Imported {imported} legacy memory files (skipped {skipped}) from {}",
        source_root.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn mk() -> (TempDir, MemoryTool) {
        let dir = TempDir::new().unwrap();
        let tool = MemoryTool::new(dir.path()).unwrap();
        (dir, tool)
    }

    fn create_page_with_relations(
        tool: &MemoryTool,
        path: &str,
        entity: &str,
        entity_type: &str,
        obs: &[&str],
        relations: &[(&str, &str)],
    ) -> String {
        let mut content = format!(
            "---\nentity: {entity}\nentity_id: \nentity_type: {entity_type}\npage_revision: \n---\n\n# {entity}\n\n## Observations\n"
        );
        for item in obs {
            let _ = writeln!(content, "- {item}");
        }
        content.push_str("\n## Relations\n");
        for (relation_type, target) in relations {
            let _ = writeln!(content, "- {relation_type} -> {target}");
        }
        tool.exec(&json!({
            "command": "create",
            "path": path,
            "file_text": content,
        }))
        .unwrap();
        canonical_path(tool, entity)
    }

    fn canonical_path(tool: &MemoryTool, entity: &str) -> String {
        let entity = tool.memory.get_entity(entity).unwrap().unwrap();
        entity_page_virtual_path(&entity)
    }

    fn raw_line<'a>(view: &'a str, needle: &str) -> &'a str {
        view.lines()
            .find(|line| line.contains(needle))
            .and_then(|line| line.split_once('\t').map(|(_, raw)| raw))
            .unwrap()
    }

    #[test]
    fn view_root_lists_entities_dir() {
        let (_dir, tool) = mk();
        let root = tool
            .exec(&json!({"command": "view", "path": "/memories"}))
            .unwrap();
        assert_eq!(root, "entities/\n");
    }

    #[test]
    fn create_and_view_entity_page() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );

        let view = tool
            .exec(&json!({"command": "view", "path": path}))
            .unwrap();
        assert!(view.contains("entity_type: person"));
        assert!(view.contains('['));
        assert!(view.contains("prefers Rust over Python"));

        let recall = tool
            .memory
            .recall("prefers Rust", 10, &sift_memory::RecallFilters::default())
            .unwrap();
        assert!(recall.iter().any(|result| result.entity_name == "Raymond"));
    }

    #[test]
    fn rewrite_observation_supersedes_old_fact() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path}))
            .unwrap();
        let old_line = current
            .lines()
            .find(|line| line.contains("prefers Rust over Python"))
            .and_then(|line| line.split_once('\t').map(|(_, raw)| raw))
            .unwrap()
            .to_string();
        let old_id = extract_anchor(&old_line).unwrap().to_string();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": old_line,
            "new_str": format!("- [{}] prefers Rust over Go", old_id),
        }))
        .unwrap();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        let active = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content, "prefers Rust over Go");
        assert_eq!(active[0].supersedes.as_deref(), Some(old_id.as_str()));
        assert_eq!(active[0].logical_id, old_id);
    }

    #[test]
    fn stale_rewrite_is_rejected() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path}))
            .unwrap();
        let old_line = raw_line(&current, "prefers Rust over Python").to_string();
        let old_id = extract_anchor(&old_line).unwrap().to_string();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        tool.memory
            .supersede_observation(&old_id, "prefers Rust over Zig", 1.0, TOOL_SOURCE)
            .unwrap();
        tool.memory.save().unwrap();

        let err = tool
            .exec(&json!({
                "command": "str_replace",
                "path": path,
                "old_str": old_line,
                "new_str": format!("- [{}] prefers Rust over Go", old_id),
            }))
            .unwrap_err();
        assert!(err.to_string().contains("re-view"));

        let active = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content, "prefers Rust over Zig");
    }

    #[test]
    fn whitespace_only_edit_is_noop() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path}))
            .unwrap();
        let old_line = raw_line(&current, "prefers Rust over Python").to_string();

        let result = tool
            .exec(&json!({
                "command": "str_replace",
                "path": path,
                "old_str": old_line.clone(),
                "new_str": format!("  {old_line}  "),
            }))
            .unwrap();
        assert!(result.contains("No changes"));
    }

    #[test]
    fn delete_entity_page_removes_memory() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        tool.exec(&json!({
            "command": "delete",
            "path": path,
        }))
        .unwrap();
        assert!(tool.memory.get_entity("Raymond").unwrap().is_none());
    }

    #[test]
    fn entity_type_edit_updates_in_place() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": raw_line(&current, "entity_type: person"),
            "new_str": "entity_type: concept",
        }))
        .unwrap();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        assert_eq!(entity.entity_type, EntityType::Concept);
    }

    #[test]
    fn insert_is_rejected_for_entity_pages() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );

        let err = tool
            .exec(&json!({
                "command": "insert",
                "path": path,
                "insert_line": 10,
                "insert_text": "- new fact",
            }))
            .unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn str_replace_cannot_add_observation_bullet() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();
        let old_line = raw_line(&current, "prefers Rust over Python").to_string();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": old_line.clone(),
            "new_str": format!("{old_line}\n- uses Homebrew on macOS"),
        }))
        .unwrap_err();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        let active = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content, "prefers Rust over Python");
    }

    #[test]
    fn str_replace_cannot_add_relation_bullet() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[("maintains", "sift")],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();
        let old_line = raw_line(&current, "maintains -> sift").to_string();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": old_line.clone(),
            "new_str": format!("{old_line}\n- collaborates -> anthropic"),
        }))
        .unwrap_err();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        let relations = tool.memory.get_entity_relations(&entity.id).unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "maintains");
    }

    #[test]
    fn unrelated_concurrent_edit_still_allows_targeted_rewrite() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();
        let old_line = raw_line(&current, "prefers Rust over Python").to_string();
        let old_id = extract_anchor(&old_line).unwrap().to_string();
        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();

        tool.memory
            .add_observation(&entity.id, "uses Homebrew on macOS", 1.0, TOOL_SOURCE)
            .unwrap();
        tool.memory.save().unwrap();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": old_line,
            "new_str": format!("- [{old_id}] prefers Rust over Go"),
        }))
        .unwrap();

        let active = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(active.len(), 2);
        assert!(active
            .iter()
            .any(|obs| obs.content == "prefers Rust over Go"));
        assert!(active
            .iter()
            .any(|obs| obs.content == "uses Homebrew on macOS"));
    }

    #[test]
    fn mixed_edit_rolls_back_entity_type_change() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python", "uses Homebrew on macOS"],
            &[],
        );
        let _ = path;
        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        let observation_id = tool
            .memory
            .get_entity_observations(&entity.id)
            .unwrap()
            .into_iter()
            .find(|obs| obs.content == "prefers Rust over Python")
            .unwrap()
            .id;

        let plan = PageMutationPlan {
            entity_id: entity.id.clone(),
            entity_type_update: Some(EntityType::Concept),
            observation_invalidations: vec![observation_id.clone()],
            observation_rewrites: vec![ObservationRewrite {
                observation_id,
                new_content: "prefers Rust over Go".to_string(),
            }],
            ..PageMutationPlan::default()
        };

        let err = tool
            .memory
            .apply_page_mutation_plan(&plan, TOOL_SOURCE)
            .unwrap_err();
        assert!(err.to_string().contains("page changed"));

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        assert_eq!(entity.entity_type, EntityType::Person);
        let active = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(active.len(), 2);
        assert!(active
            .iter()
            .any(|obs| obs.content == "prefers Rust over Python"));
    }

    #[test]
    fn slug_collisions_list_distinct_entity_pages() {
        let (_dir, tool) = mk();
        let path_a = create_page_with_relations(
            &tool,
            "/memories/entities/foo-bar.md",
            "Foo Bar",
            "concept",
            &["alpha fact"],
            &[],
        );
        let path_b = create_page_with_relations(
            &tool,
            "/memories/entities/foo-bar.md",
            "Foo-Bar",
            "concept",
            &["beta fact"],
            &[],
        );

        assert_ne!(path_a, path_b);

        let listing = tool
            .exec(&json!({"command": "view", "path": "/memories/entities"}))
            .unwrap();
        assert!(listing.contains(path_a.trim_start_matches("/memories/entities/")));
        assert!(listing.contains(path_b.trim_start_matches("/memories/entities/")));

        let view_a = tool
            .exec(&json!({"command": "view", "path": path_a}))
            .unwrap();
        let view_b = tool
            .exec(&json!({"command": "view", "path": path_b}))
            .unwrap();
        assert!(view_a.contains("alpha fact"));
        assert!(view_b.contains("beta fact"));
    }

    #[test]
    fn relation_rewrite_updates_outgoing_edge() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[("maintains", "sift")],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();
        let old_line = raw_line(&current, "maintains -> sift").to_string();
        let old_id = extract_anchor(&old_line).unwrap();

        tool.exec(&json!({
            "command": "str_replace",
            "path": path,
            "old_str": old_line,
            "new_str": format!("- [{old_id}] builds -> sift"),
        }))
        .unwrap();

        let entity = tool.memory.get_entity("Raymond").unwrap().unwrap();
        let relations = tool.memory.get_entity_relations(&entity.id).unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "builds");
        let target = tool
            .memory
            .get_entity_by_id(&relations[0].to_entity)
            .unwrap()
            .unwrap();
        assert_eq!(target.name, "sift");
    }

    #[test]
    fn malformed_anchorless_edit_is_rejected() {
        let (_dir, tool) = mk();
        let path = create_page_with_relations(
            &tool,
            "/memories/entities/raymond.md",
            "Raymond",
            "person",
            &["prefers Rust over Python"],
            &[],
        );
        let current = tool
            .exec(&json!({"command": "view", "path": path.clone()}))
            .unwrap();
        let old_line = raw_line(&current, "prefers Rust over Python").to_string();

        let err = tool
            .exec(&json!({
                "command": "str_replace",
                "path": path,
                "old_str": old_line,
                "new_str": "- prefers Rust over Go",
            }))
            .unwrap_err();
        assert!(err.to_string().contains("anchored"));
    }

    #[test]
    fn rename_is_rejected() {
        let (_dir, tool) = mk();
        let err = tool
            .exec(&json!({
                "command": "rename",
                "old_path": "/memories/entities/raymond--deadbeef.md",
                "new_path": "/memories/entities/rj.md",
            }))
            .unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn migration_is_idempotent() {
        let (_backing, tool) = mk();
        let legacy = TempDir::new().unwrap();
        std::fs::write(legacy.path().join("notes.md"), "remember this").unwrap();
        let nested = legacy.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("todo.md"), "another note").unwrap();

        let (imported, skipped) = tool.migrate_legacy(legacy.path()).unwrap();
        assert_eq!(imported, 2);
        assert_eq!(skipped, 0);

        let (imported, skipped) = tool.migrate_legacy(legacy.path()).unwrap();
        assert_eq!(imported, 0);
        assert_eq!(skipped, 2);

        let entity = tool.memory.get_entity(LEGACY_ENTITY).unwrap().unwrap();
        let observations = tool.memory.get_entity_observations(&entity.id).unwrap();
        assert_eq!(observations.len(), 2);
        assert!(observations
            .iter()
            .any(|obs| obs.content.contains("[legacy:notes.md]")));
        assert!(observations
            .iter()
            .any(|obs| obs.content.contains("[legacy:nested/todo.md]")));

        let recall = tool
            .memory
            .recall("remember this", 10, &sift_memory::RecallFilters::default())
            .unwrap();
        assert!(recall
            .iter()
            .any(|result| result.entity_name == LEGACY_ENTITY));
    }

    fn extract_anchor(line: &str) -> Option<&str> {
        line.strip_prefix("- [")
            .and_then(|rest| rest.split_once(']'))
            .map(|(id, _)| id)
    }
}
