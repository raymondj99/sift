---
name: sift-mcp
description: Local semantic search engine with hybrid search, 30+ format indexing, entity memory, and MCP integration
version: 0.1.4
tags: [search, indexing, memory, knowledge-graph, mcp]
---

# sift MCP Server

Sift is a local semantic search engine that indexes 30+ file formats and exposes search, indexing, and persistent memory via the Model Context Protocol (MCP).

## Capabilities

### Search
- **Hybrid search** — combines BM25 keyword ranking with cosine-similarity vector search via Reciprocal Rank Fusion (RRF). Scores normalized to 0–1.
- **Keyword-only** — BM25 full-text search via SQLite FTS5. No embedding model needed.
- **Vector-only** — pure cosine similarity. Requires a downloaded embedding model.
- **30+ file formats** — code, markdown, PDF, Office docs, CSV, JSON, HTML, email, images, audio, archives.
- **Context display** — return surrounding source lines around each match.
- **Filtering** — by file type, path, and modification date.

### Indexing
- **File indexing** — `sift scan <PATH>` indexes directories with incremental BLAKE3 content hashing.
- **Text indexing** — `sift_index_text` stores arbitrary text directly in the index with custom URIs.
- **Parallel pipeline** — discover → parse → chunk → embed → store, all stages concurrent.

### Entity Memory (Knowledge Graph)
- **sift_remember** — store structured facts about named entities with types, observations, and relationships.
- **sift_recall** — hybrid semantic + keyword search over memory. Entity names are indexed so searching "Raymond" finds Raymond's observations. Minimum score threshold filters irrelevant results.
- **sift_list_entities** — browse all entities with optional type filter and pagination. Returns observation counts per entity.
- **sift_get_entity** — get all observations and relationships for a named entity.
- **sift_forget** — soft-delete observations by ID, preserving audit trail.
- **sift_forget_entity** — hard-delete an entire entity and all its observations, relationships, and search index entries.
- **sift_prune** — remove all entities with zero active observations and no active relationships. Cleans up ghost entities left by sift_forget.
- **Entity types** — person, project, concept, tool, preference, fact, event, location, organization.
- **Relationships** — directed edges between entities (e.g., "Raymond" → prefers → "Rust").
- **Embeddings** — observations are embedded with nomic-embed-text-v2 for semantic recall. Index is rebuilt automatically when the embedder is first attached.

## Gathering Context Before a Task

At the start of a conversation or task, load relevant memory to inform your work:

1. **`sift_recall`** with a query describing the task — retrieves scored facts that may be relevant.
2. **`sift_list_entities`** — browse what's stored. Filter by `entity_type` (e.g., `person`, `project`, `preference`) to find relevant context quickly.
3. **`sift_get_entity`** — for any entity that looks relevant, fetch its full observations and relationships.

Do this proactively when:
- Starting a new task or conversation
- The user references prior work, decisions, or preferences
- You need to understand who the user is or how they work
- Working on a project that has been discussed before

## Persisting Insights

Whenever you produce a `★ Insight` block, store the key points in memory using `sift_remember`. Pick the most relevant entity (or create one) and add the insight as observations. This ensures non-obvious learnings from conversations are preserved across sessions rather than lost when the conversation ends.

## When to Use

- **Finding code or docs** — `sift_search` with hybrid mode for conceptual queries, keyword mode for exact symbol names.
- **Persisting knowledge** — `sift_remember` for structured facts about users, projects, decisions. `sift_recall` to retrieve them later. `sift_list_entities` and `sift_get_entity` to browse what's stored.
- **Storing raw text** — `sift_index_text` for notes, logs, or content that should be searchable.
- **Discovering capabilities** — `sift_search_skills` to find SKILL.md files describing agent skills.

## Tool Quick Reference

| Tool | Purpose | Read-only |
|------|---------|-----------|
| `sift_status` | Index statistics | Yes |
| `sift_search` | Hybrid/keyword/vector search | Yes |
| `sift_search_skills` | Find SKILL.md capabilities | Yes |
| `sift_list_sources` | Browse indexed files | Yes |
| `sift_index_text` | Store text in index | No |
| `sift_delete` | Remove content by URI | No |
| `sift_remember` | Store entity facts | No |
| `sift_recall` | Search entity memory | Yes |
| `sift_list_entities` | Browse all entities | Yes |
| `sift_get_entity` | Get entity details | Yes |
| `sift_forget` | Soft-delete observation | No |
| `sift_forget_entity` | Hard-delete entire entity | No |
| `sift_prune` | Remove ghost entities | No |
| `sift_memory_status` | Memory statistics | Yes |
