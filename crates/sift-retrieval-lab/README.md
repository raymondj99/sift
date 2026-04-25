# sift-retrieval-lab

Reproducible IR-metric harness for sift retrieval. Measures
`(encoder × matryoshka dim × RRF alpha × search-mode × corpus)` to
TREC-standard `recall@k`, `nDCG@k`, `MRR`, `MAP`, plus latency
`p50/p95/p99` and per-category breakdowns.

## What this is

A tightly-scoped Rust crate for answering one question:
*"does configuration X retrieve more relevant documents than
configuration Y, with enough statistical power to be trusted?"*
Anything that doesn't directly serve that question is deferred.

## What this is not

- **Not a runtime dependency.** Excluded from the workspace
  `default-members`. Build with `cargo build -p sift-retrieval-lab`.
- **Not coupled to `sift-memory`.** Driving retrieval through
  `MemoryStore` would entangle measured quality with the decay curve
  and retrieval-count boost multipliers. The lab depends on
  `sift-core`, `sift-embed`, and `sift-store` only — verifiable in the
  crate's `Cargo.toml`.
- **Not a sweep engine yet.** Single-cell + Phase 0 hypothesis suite +
  `research` characterization sweeps ship in v1. TOML sweep config,
  baselines (per-target committed JSON), paired bootstrap CIs / sign
  tests, drift detection, BEIR loaders, reranker trait, CSV reporters,
  and CI integration are deferred until v1 has caught one regression
  worth gating on.

## Subcommands

```text
retrieval-lab validate <corpus>         # structural validation, no model needed
retrieval-lab run --corpus … --alpha … --dim … --top-k … --mode hybrid|vector|keyword
retrieval-lab report <run.json>         # render a run JSON as Markdown
retrieval-lab phase0                    # 7-experiment hypothesis suite
retrieval-lab research                  # alpha curve · mode comparison · top-k curve
codingmem-transform                     # scenarios.json → committed corpus + qrels
```

## Quick start

```bash
# 1. Build (release recommended; ONNX dominates wall time anyway).
cargo build -p sift-retrieval-lab --release

# 2. Build the corpus from CodingMem scenarios. Output is committed.
ORT_DYLIB_PATH=$HOME/.sift/models/ort/libonnxruntime.dylib \
  ./target/release/codingmem-transform

# 3. Validate (cheap, no model required).
./target/release/retrieval-lab validate

# 4. One configuration end-to-end.
ORT_DYLIB_PATH=$HOME/.sift/models/ort/libonnxruntime.dylib \
  ./target/release/retrieval-lab run \
    --alpha 0.7 --dim 768 --top-k 10 --mode hybrid \
    --out target/retrieval-lab/baseline.json

# 5. Markdown summary of any run JSON.
./target/release/retrieval-lab report target/retrieval-lab/baseline.json

# 6. Phase 0 hypothesis suite (~3 minutes wall on M-series).
ORT_DYLIB_PATH=$HOME/.sift/models/ort/libonnxruntime.dylib \
  ./target/release/retrieval-lab phase0

# 7. Research sweeps (~5 minutes; alpha 0.0–1.0, mode comparison, top-k curve).
ORT_DYLIB_PATH=$HOME/.sift/models/ort/libonnxruntime.dylib \
  ./target/release/retrieval-lab research
```

## Phase 0 hypothesis suite

Seven experiments designed to falsify, in order, the foundational
assumptions behind the lab before any production sweep machinery
ships. Each has a hard kill-criterion:

| #  | Hypothesis                                            | Kill criterion                              |
|----|-------------------------------------------------------|---------------------------------------------|
| E1 | Qrels heuristic produces usable distributions         | >40% queries with empty qrels, or median ≥10 |
| E2 | Current sift retrieval has measurable headroom        | recall@10 ≥ 0.95 or ≤ 0.30                  |
| E3 | RRF alpha is a productive sweep axis                  | Δrecall@10 < 0.02 across {0.0, 0.3, 0.7, 1.0} |
| E4 | Matryoshka dim is a productive sweep axis             | informational only (note if Δ < 0.005)      |
| E5 | Corpus has enough queries for paired statistics       | informational only (note if CI ≥ 0.10)      |
| E6 | Aggregate metrics are stable across identical runs    | drift > 0.0005 on any of recall/nDCG/MRR/MAP |
| E7 | Latency boundary is well-controlled                   | informational only (note if p99/p50 > 10)   |

E6 was reframed during implementation: byte-exact ranking determinism
is unrealistic with multi-threaded ONNX (float reductions are
non-associative), but **aggregate metric stability** is the property
committed baselines actually need. Single-thread ONNX is the future
`--deterministic` flag.

## Architecture

| Module          | Responsibility                                                     |
|-----------------|--------------------------------------------------------------------|
| `corpus.rs`     | `Document`, `Query`, `Qrel`, `RetrievalCorpus` types. Schema-versioned. Validates on load (duplicate doc ids, dangling qrel refs, negative-query-with-qrels). Blake3 content hash for envelope pinning. |
| `transform.rs`  | `evals/codingmem/scenarios.json` → `RetrievalCorpus`. Heuristic oracle: content-word token-set overlap (≥50% of answer's content words, ≥2 distinct, word-boundary not substring). Negative-answer detection. |
| `metrics.rs`    | TREC-correct `recall_at_k`, `ndcg_at_k`, `reciprocal_rank`, `average_precision`. `2^rel - 1` numerator, `log2(rank+1)` discount. Skipped-query semantics via `Option<f64>`. |
| `latency.rs`    | `LatencyCollector` with linear-interpolated percentiles and Tukey-fence outlier reporting. Replaces the `pct * len / 100` truncation bug in `crates/sift-cli/src/commands/bench.rs:282`. |
| `runner.rs`     | Single-cell execution. Embed corpus → build `HybridSearchEngine<FlatVectorIndex, Bm25Store>` → run all queries → compute aggregate + per-category metrics. Lexicographic `doc_id` tie-break enforced after the engine returns. |
| `repro.rs`      | git SHA, dirty flag, rustc, target triple. Captured into every run JSON. |
| `error.rs`      | `RetrievalLabError` typed enum (thiserror). Library functions return `RetrievalLabResult<T>`; binaries propagate via `?` into `anyhow::Result`. |

## Latency boundary (per `PLAN-retrieval-lab.md` §4.3)

`run_cell` enforces this boundary explicitly:

- **Inside** the timed region: prefix concatenation, query
  tokenization, query embedding, matryoshka truncation, search.
- **Outside** the timed region: corpus embedding (once at index
  build), index insertion, ONNX session warmup (absorbed into
  `warmup_queries`).

`black_box` wraps the *result*, not the call, so the optimizer cannot
elide the search.

## Reproducibility envelope

Every result JSON carries:

- `schema_version` — refuse to load unknown versions
- `corpus.corpus_blake3` — content hash of canonical-JSON corpus
- `environment.git_sha` + `git_dirty` + `rustc` + `target`
- `environment.timestamp_utc`

`schema_version` and `corpus_blake3` together prevent comparing
baselines across corpus or schema changes; `target` is recorded so
baselines stay machine-pinned.

## Deferred (v1.1+)

Sweep config (TOML cross-product), per-target committed baselines,
paired bootstrap CIs, sign tests, drift detection, BEIR scifact
loader, reranker trait, content-addressed result cache, `--preset`
embedded configs, `--deterministic` flag (ort intra-threads=1), CI
workflow, top-level CSV/Parquet reporters.

Each was proposed in `PLAN-retrieval-lab.md`. Each is correct
*eventually*. None blocks the lab from producing its first numbers.
