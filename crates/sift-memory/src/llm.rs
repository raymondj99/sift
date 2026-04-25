//! LLM-powered extraction of categorized observations from session summaries.
//!
//! Decomposes compact_summary and stop messages into individually categorized
//! observations on named entities. Provider-agnostic — supports Anthropic,
//! OpenAI-compatible APIs, and local Ollama.
//!
//! Enabled via `[memory] llm_extraction = true` in config. Falls back to
//! heuristic extraction on any failure.
//!
//! # Providers
//!
//! | Provider | Env Var | API | Notes |
//! |----------|---------|-----|-------|
//! | `anthropic` | `ANTHROPIC_API_KEY` | Messages API | Default. Haiku recommended. |
//! | `openai` | `OPENAI_API_KEY` | Chat Completions | Works with any OpenAI-compatible endpoint. |
//! | `ollama` | — | `localhost:11434` | Local, free. Set `OLLAMA_HOST` to override. |

use serde::{Deserialize, Serialize};

#[cfg(feature = "llm")]
use std::time::Duration;
#[cfg(feature = "llm")]
use tracing::debug;

/// Character budget for LLM input. Applied in chars, not bytes, to avoid
/// slicing through a multi-byte UTF-8 codepoint.
#[cfg(feature = "llm")]
const MAX_INPUT_CHARS: usize = 3000;
#[cfg(feature = "llm")]
const MAX_OUTPUT_TOKENS: u32 = 2048;
/// Time to wait for the initial TCP/TLS handshake.
#[cfg(feature = "llm")]
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Time to wait for the full response body. Bounds worst-case consolidation
/// latency — one hung endpoint cannot stall the daemon indefinitely.
#[cfg(feature = "llm")]
const CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// The set of category labels we accept from the LLM. Anything else is
/// dropped by `parse_observations` to avoid storing arbitrary strings.
#[cfg(feature = "llm")]
const VALID_CATEGORIES: &[&str] = &["decision", "correction", "workflow", "preference"];

/// Shared system prompt used across all providers.
#[cfg(feature = "llm")]
const SYSTEM_PROMPT: &str = r#"You classify statements from AI coding session summaries into categories for a persistent memory system.

For each statement, return:
- category: one of decision | correction | workflow | preference | skip
- entity: the main subject (project name, tool, file, concept) — 1-4 words
- observation: a clean, concise restatement of the fact (1 sentence, max 80 words)

Category definitions:
- decision: an architecture choice, technology selection, or configuration state ("chose X", "uses Y for Z", "renamed X to Y", "X is implemented in Y")
- correction: something that was wrong/broken and was fixed, a gotcha to remember ("was broken", "silently failed", "was missing", "had stale X")
- workflow: a process rule, convention, or sequenced action to follow ("always X before Y", "run X after Y", "the process is X then Y")
- preference: user identity, personal habits, or tool preferences ("prefers X", "I like Y", "uses Z as primary tool")
- skip: ephemeral session narration with no long-term value ("I read the file", "let me check", "the file was updated")

Important:
- A sentence describing a PROBLEM or FAILURE is a correction, even if it also describes the fix.
- A sentence describing HOW something IS CONFIGURED or DESIGNED is a decision.
- A sentence describing WHAT TO DO in a SEQUENCE is a workflow.
- When in doubt between decision and correction, check: does it describe something that WENT WRONG? If yes, correction."#;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single extracted observation from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedObservation {
    pub category: String,
    pub entity: String,
    pub observation: String,
}

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    Ollama,
}

impl Provider {
    /// Canonical lowercase tag (`"anthropic"`, `"openai"`, `"ollama"`). Round
    /// trips through [`Provider::from_str`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Ollama => "ollama",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "openai" | "openai-compatible" => Ok(Self::OpenAi),
            "ollama" | "local" => Ok(Self::Ollama),
            other => Err(format!(
                "Unknown LLM provider '{other}'. Expected: anthropic, openai, ollama"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Decompose session text into categorized observations via an LLM.
///
/// The caller supplies a typed [`Provider`] (parsed once at consolidation
/// startup) so that an invalid config produces a single warning rather than
/// re-failing once per episode.
#[cfg(feature = "llm")]
pub fn llm_decompose(
    text: &str,
    model: &str,
    provider: Provider,
) -> Result<Vec<ExtractedObservation>, String> {
    if text.trim().len() < 30 {
        return Ok(Vec::new());
    }

    let input = truncate_to_chars(text, MAX_INPUT_CHARS);

    let user_msg = format!(
        "Decompose this session summary into individual facts for persistent memory. \
         Skip any ephemeral narration.\n\n{input}\n\n\
         Return a JSON array. Each object: \
         {{\"category\": \"...\", \"entity\": \"...\", \"observation\": \"...\"}}"
    );

    debug!(
        %provider,
        model,
        input_len = input.len(),
        "LLM extraction: calling API"
    );

    let agent = build_agent();
    let raw_response = match provider {
        Provider::Anthropic => call_anthropic(&agent, model, &user_msg)?,
        Provider::OpenAi => call_openai(&agent, model, &user_msg)?,
        Provider::Ollama => call_ollama(&agent, model, &user_msg)?,
    };

    parse_observations(&raw_response)
}

/// A char-boundary-safe prefix of `s` with at most `max_chars` characters.
///
/// Byte-based slicing (`&s[..n]`) panics when `n` falls inside a multi-byte
/// UTF-8 codepoint — possible whenever the agent writes emoji or CJK.
#[cfg(feature = "llm")]
fn truncate_to_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

/// Build a ureq agent with bounded connect + read timeouts.
///
/// Without these, a hung endpoint (captive portal, dead proxy) blocks
/// consolidation forever — and consolidation runs in the daemon background
/// worker with no UI feedback.
#[cfg(feature = "llm")]
fn build_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout(CALL_TIMEOUT)
        .build()
}

/// Stub when the `llm` feature is disabled.
#[cfg(not(feature = "llm"))]
pub fn llm_decompose(
    _text: &str,
    _model: &str,
    _provider: Provider,
) -> Result<Vec<ExtractedObservation>, String> {
    Err("LLM extraction requires the `llm` feature".into())
}

// ---------------------------------------------------------------------------
// Provider implementations
// ---------------------------------------------------------------------------

/// Anthropic Messages API (Claude models).
///
/// Auth: `ANTHROPIC_API_KEY` env var.
#[cfg(feature = "llm")]
fn call_anthropic(agent: &ureq::Agent, model: &str, user_msg: &str) -> Result<String, String> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

    let url =
        std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".into());

    let body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{ "role": "user", "content": user_msg }],
    });

    let resp: serde_json::Value = agent
        .post(&format!("{url}/v1/messages"))
        .set("x-api-key", &api_key)
        .set("content-type", "application/json")
        .set("anthropic-version", "2023-06-01")
        .send_json(body)
        .map_err(|e| format!("Anthropic API error: {e}"))?
        .into_json()
        .map_err(|e| format!("Anthropic response parse error: {e}"))?;

    resp["content"][0]["text"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "No text in Anthropic response".into())
}

/// OpenAI Chat Completions API (GPT models, or any compatible endpoint).
///
/// Auth: `OPENAI_API_KEY` env var.
/// Override base URL with `OPENAI_BASE_URL` (for Azure, Together, Groq, etc.).
#[cfg(feature = "llm")]
fn call_openai(agent: &ureq::Agent, model: &str, user_msg: &str) -> Result<String, String> {
    let api_key =
        std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set".to_string())?;

    let url = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());

    let body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_msg },
        ],
    });

    let resp: serde_json::Value = agent
        .post(&format!("{url}/v1/chat/completions"))
        .set("Authorization", &format!("Bearer {api_key}"))
        .set("content-type", "application/json")
        .send_json(body)
        .map_err(|e| format!("OpenAI API error: {e}"))?
        .into_json()
        .map_err(|e| format!("OpenAI response parse error: {e}"))?;

    resp["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "No content in OpenAI response".into())
}

/// Ollama local API (any model served by Ollama).
///
/// No auth required. Override host with `OLLAMA_HOST` (default: localhost:11434).
#[cfg(feature = "llm")]
fn call_ollama(agent: &ureq::Agent, model: &str, user_msg: &str) -> Result<String, String> {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());

    let body = serde_json::json!({
        "model": model,
        "stream": false,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_msg },
        ],
    });

    let resp: serde_json::Value = agent
        .post(&format!("{host}/api/chat"))
        .set("content-type", "application/json")
        .send_json(body)
        .map_err(|e| format!("Ollama API error: {e}"))?
        .into_json()
        .map_err(|e| format!("Ollama response parse error: {e}"))?;

    resp["message"]["content"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "No content in Ollama response".into())
}

// ---------------------------------------------------------------------------
// Response parsing (shared across all providers)
// ---------------------------------------------------------------------------

/// Extract structured observations from the LLM's raw text response.
///
/// The LLM is instructed to return a JSON array. Real responses may wrap
/// it in a ```json fence, add leading prose, or add trailing commentary.
/// We strip common wrappers and fall back to a bracket-scan. Unknown
/// categories are dropped to avoid storing arbitrary strings as sources.
#[cfg(feature = "llm")]
fn parse_observations(text: &str) -> Result<Vec<ExtractedObservation>, String> {
    let json_slice = extract_json_array(text).ok_or("No JSON array in response")?;

    let observations: Vec<ExtractedObservation> =
        serde_json::from_str(json_slice).map_err(|e| format!("JSON parse error: {e}"))?;

    let mut dropped_unknown = 0usize;
    let valid: Vec<ExtractedObservation> = observations
        .into_iter()
        .filter(|o| {
            if o.entity.is_empty() || o.observation.is_empty() || o.observation.len() >= 500 {
                return false;
            }
            if o.category == "skip" {
                return false;
            }
            if !VALID_CATEGORIES.contains(&o.category.as_str()) {
                dropped_unknown += 1;
                return false;
            }
            true
        })
        .collect();

    if dropped_unknown > 0 {
        debug!(
            dropped = dropped_unknown,
            "dropped observations with unknown category"
        );
    }
    debug!(extracted = valid.len(), "LLM extraction complete");
    Ok(valid)
}

/// Locate the JSON array payload inside an LLM response.
///
/// Strategy:
/// 1. If the text is wrapped in ``` / ```json fences, take the fence body.
/// 2. Otherwise, scan for the first `[` and the matching `]` at the same
///    depth. This handles arrays containing nested objects/arrays but
///    avoids being fooled by trailing `]` inside commentary.
#[cfg(feature = "llm")]
fn extract_json_array(text: &str) -> Option<&str> {
    let trimmed = strip_code_fence(text.trim());
    find_balanced_array(trimmed)
}

#[cfg(feature = "llm")]
fn strip_code_fence(s: &str) -> &str {
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    // Drop an optional language tag on the same line (e.g. ```json).
    let rest = match rest.find('\n') {
        Some(nl) => &rest[nl + 1..],
        None => rest,
    };
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// Return the largest balanced `[...]` span in `s`, or `None` if no
/// well-formed array exists.
///
/// We pick the largest because LLM responses sometimes contain incidental
/// bracketed tokens (e.g. "see [1] for details") before the real payload.
/// The actual observations array is almost always the longest balanced
/// span in the response.
#[cfg(feature = "llm")]
fn find_balanced_array(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut best: Option<(usize, usize)> = None;

    // Stack of open-bracket byte offsets. Each `]` pops the matching `[`
    // and records the closed span as a candidate.
    let mut stack: Vec<usize> = Vec::new();
    let mut in_string = false;
    let mut escape = false;

    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match b {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' => stack.push(i),
            b']' => {
                if let Some(open) = stack.pop() {
                    let span_len = i + 1 - open;
                    if best.is_none_or(|(_, len)| span_len > len) {
                        best = Some((open, span_len));
                    }
                }
            }
            _ => {}
        }
    }

    best.map(|(start, len)| &s[start..start + len])
}

// ---------------------------------------------------------------------------
// Category mapping (shared across all providers)
// ---------------------------------------------------------------------------

/// Map a category string to a confidence value.
pub fn category_confidence(category: &str) -> f32 {
    match category {
        "correction" => 0.90,
        "workflow" => 0.85,
        "preference" => 0.85,
        "decision" => 0.80,
        _ => 0.70,
    }
}

/// Map a category string to a source tag for the observation.
pub fn category_source(category: &str) -> &'static str {
    match category {
        "correction" => "cortex:correction",
        "workflow" => "cortex:procedural",
        "preference" => "cortex:preference",
        "decision" => "cortex:decision",
        _ => "cortex:llm",
    }
}

/// Map a category string to an entity type.
pub fn category_entity_type(category: &str) -> crate::types::EntityType {
    match category {
        "preference" => crate::types::EntityType::Preference,
        "workflow" => crate::types::EntityType::Concept,
        "correction" => crate::types::EntityType::Fact,
        "decision" => crate::types::EntityType::Concept,
        _ => crate::types::EntityType::Concept,
    }
}

#[cfg(all(test, feature = "llm"))]
mod tests {
    use super::*;

    #[test]
    fn truncate_to_chars_handles_multibyte_boundary() {
        // Each ✨ is 3 bytes in UTF-8. 10 sparkles = 30 bytes but only 10
        // chars, so byte-based slicing at byte 5 would panic mid-codepoint.
        let s = "✨".repeat(10);
        let got = truncate_to_chars(&s, 3);
        assert_eq!(got.chars().count(), 3);
        assert!(got.is_char_boundary(got.len()));
    }

    #[test]
    fn truncate_to_chars_short_input_unchanged() {
        assert_eq!(truncate_to_chars("hi", 100), "hi");
    }

    #[test]
    fn extract_json_array_handles_code_fence() {
        let raw =
            "```json\n[{\"category\":\"decision\",\"entity\":\"x\",\"observation\":\"y\"}]\n```";
        let slice = extract_json_array(raw).expect("should find array");
        let parsed: Vec<ExtractedObservation> = serde_json::from_str(slice).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn extract_json_array_handles_prose_wrapping() {
        // Bracketed commentary before the payload should not fool the scanner.
        let raw = "Here is the result (see [1] for details):\n\
                   [{\"category\":\"workflow\",\"entity\":\"a\",\"observation\":\"b\"}]\n\
                   End.";
        let slice = extract_json_array(raw).expect("should find array");
        let parsed: Vec<ExtractedObservation> = serde_json::from_str(slice).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn extract_json_array_balances_nested() {
        let raw = "[{\"x\":[1,2,3]},{\"y\":[]}] trailing garbage ]]]";
        let slice = extract_json_array(raw).expect("should find array");
        assert_eq!(slice, "[{\"x\":[1,2,3]},{\"y\":[]}]");
    }

    #[test]
    fn parse_observations_drops_unknown_category() {
        let raw = r#"[
            {"category":"decision","entity":"a","observation":"good"},
            {"category":"rant","entity":"b","observation":"bad"},
            {"category":"skip","entity":"c","observation":"noise"}
        ]"#;
        let got = parse_observations(raw).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].category, "decision");
    }
}
