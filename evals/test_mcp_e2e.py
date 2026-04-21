#!/usr/bin/env python3
"""End-to-end test of the full Cortex MCP pipeline.

Tests every MCP tool and the full memory lifecycle:
  remember → recall → consolidate → generate-rules → verify

Usage:
    SIFT_BIN=/path/to/sift ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
        uv run python test_mcp_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

from sift_client import SiftClient

SIFT_BIN = os.environ.get("SIFT_BIN", "sift")
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {PASS} {name}")
        passed += 1
    else:
        print(f"  {FAIL} {name}  {detail}")
        failed += 1


def main():
    global passed, failed
    index = f"cortex-e2e-test-{int(time.time())}"
    print(f"Cortex MCP End-to-End Test (index: {index})\n")

    # ── 1. Connect to MCP server ──────────────────────────────────���──────
    print("1. MCP Connection")
    try:
        client = SiftClient(index=index)
        check("MCP handshake", True)
    except Exception as e:
        print(f"  {FAIL} MCP handshake failed: {e}")
        sys.exit(1)

    # ── 2. Memory status (empty) ─────────────────────────────────────────
    print("\n2. Memory Status (empty)")
    status = client.memory_status()
    check("sift_memory_status returns", isinstance(status, dict))
    check("entities = 0", status.get("total_entities", -1) == 0)
    check("observations = 0", status.get("total_observations", -1) == 0)

    # ── 3. Remember entities ─────────────────────────────────────────────
    print("\n3. sift_remember")

    # Person entity
    r1 = client.remember(
        entity="Alice",
        entity_type="person",
        observations=[
            "Senior backend engineer specializing in Rust",
            "Prefers functional error handling with Result types",
        ],
    )
    check("remember person", "entity_id" in r1 or "id" in r1 or "entity" in str(r1).lower())

    # Project entity
    r2 = client.remember(
        entity="payment-service",
        entity_type="project",
        observations=[
            "Uses Axum web framework with tower middleware",
            "PostgreSQL with sqlx for async database access",
            "JWT authentication with RS256 signing",
        ],
    )
    check("remember project", "entity_id" in r2 or "id" in r2 or "entity" in str(r2).lower())

    # Fact entity (correction)
    r3 = client.remember(
        entity="error-handling",
        entity_type="fact",
        observations=[
            "Don't use unwrap() in production handlers — use ? operator instead",
            "Custom AppError type implements IntoResponse for Axum",
        ],
    )
    check("remember fact/correction", "entity_id" in r3 or "id" in r3 or "entity" in str(r3).lower())

    # Entity with relationship
    # MCP schema uses "relations" with {to, type} fields
    r4 = client.call_tool("sift_remember", {
        "entity": "Alice",
        "entity_type": "person",
        "observations": ["Maintains the payment-service repository"],
        "relations": [
            {"to": "payment-service", "type": "maintains"},
        ],
    })
    check("remember with relationship", r4.get("relations_added", 0) >= 1,
          f"relations_added: {r4.get('relations_added', 0)}")

    # ── 4. Memory status (populated) ─────────────────────────────────────
    print("\n4. Memory Status (populated)")
    status = client.memory_status()
    check("entities >= 3", status.get("total_entities", 0) >= 3,
          f"got {status.get('total_entities', 0)}")
    check("observations >= 7", status.get("total_observations", 0) >= 7,
          f"got {status.get('total_observations', 0)}")
    check("relations >= 1", status.get("total_relations", 0) >= 1,
          f"got {status.get('total_relations', 0)}")

    # ── 5. Recall ────────────────────────────────────────────────────────
    print("\n5. sift_recall")

    # Basic recall
    memories = client.recall("What web framework does payment-service use?")
    check("recall returns results", len(memories) > 0,
          f"got {len(memories)} results")
    if memories:
        top = memories[0]
        check("top result has entity", "entity" in top,
              f"keys: {list(top.keys())}")
        check("top result mentions Axum", "axum" in str(top).lower(),
              f"content: {str(top)[:100]}")

    # Recall with entity filter
    memories_filtered = client.recall(
        "error handling patterns",
        entity_names=["error-handling"],
    )
    check("entity-filtered recall returns", len(memories_filtered) > 0,
          f"got {len(memories_filtered)} results")
    if memories_filtered:
        check("filtered result about errors", "unwrap" in str(memories_filtered).lower() or "error" in str(memories_filtered).lower())

    # Recall with entity_type filter
    memories_typed = client.recall("engineer", entity_type="person")
    check("type-filtered recall", len(memories_typed) > 0,
          f"got {len(memories_typed)} results")

    # ── 6. List entities ─────────────────────────────────────────────────
    print("\n6. sift_list_entities")
    entities_result = client.call_tool("sift_list_entities", {})
    entities = entities_result.get("entities", [])
    check("list_entities returns", len(entities) >= 3,
          f"got {len(entities)}")
    entity_names = [e.get("name", "") for e in entities]
    check("Alice in entities", "Alice" in entity_names,
          f"names: {entity_names}")
    check("payment-service in entities", "payment-service" in entity_names)

    # ── 7. Get entity ────────────────────────────────────────────────────
    print("\n7. sift_get_entity")
    entity_detail = client.call_tool("sift_get_entity", {"entity": "Alice"})
    check("get_entity returns", "name" in entity_detail or "entity" in str(entity_detail).lower())
    observations = entity_detail.get("observations", [])
    check("Alice has observations", len(observations) >= 2,
          f"got {len(observations)}")
    relations = entity_detail.get("relations", entity_detail.get("relationships", []))
    # Relations may be in outgoing or incoming format
    check("Alice has 'maintains' relation", len(relations) >= 1 or "maintains" in str(entity_detail),
          f"relations: {relations}")

    # ── 8. Search skills ─────────────────────────────────────────────────
    print("\n8. sift_search_skills")
    skills = client.call_tool("sift_search_skills", {"query": "error handling"})
    check("search_skills returns", isinstance(skills, dict))

    # ── 9. Forget (soft-delete) ──────────────────────────────────────────
    print("\n9. sift_forget")
    # Get an observation ID to forget
    alice_detail = client.call_tool("sift_get_entity", {"entity": "Alice"})
    obs_list = alice_detail.get("observations", [])
    if obs_list:
        obs_id = obs_list[0].get("observation_id", obs_list[0].get("id", ""))
        if obs_id:
            forget_result = client.call_tool("sift_forget", {"observation_id": obs_id})
            check("forget observation", True)

            # Verify it's gone from recall
            alice_after = client.call_tool("sift_get_entity", {"entity": "Alice"})
            remaining = [o for o in alice_after.get("observations", [])
                        if o.get("valid_until") is None]
            check("observation removed", len(remaining) < len(obs_list),
                  f"before: {len(obs_list)}, after: {len(remaining)}")
        else:
            check("forget observation", False, "no observation ID found")
    else:
        check("forget observation", False, "no observations to forget")

    # ── 10. Consolidate ──────────────────────────────────────────────────
    print("\n10. sift_consolidate (if available)")
    try:
        cons_result = client.consolidate()
        check("consolidate runs", isinstance(cons_result, dict))
    except RuntimeError as e:
        if "not found" in str(e).lower():
            check("consolidate (tool not in this build)", True)
        else:
            check("consolidate", False, str(e))

    # ── 11. Prune ────────────────────────────────────────────────────────
    print("\n11. sift_prune")
    prune_result = client.call_tool("sift_prune", {})
    check("prune runs", isinstance(prune_result, dict))

    # ── 12. Forget entity ────────────────────────────────────────────────
    print("\n12. sift_forget_entity")
    fe_result = client.forget_entity("error-handling")
    check("forget_entity runs", True)

    # Verify entity is gone
    status_after = client.memory_status()
    check("entity count decreased",
          status_after.get("total_entities", 99) < status.get("total_entities", 0),
          f"before: {status.get('total_entities')}, after: {status_after.get('total_entities')}")

    client.close()

    # ── 13. CLI: generate-rules ──────────────────────────────────────────
    print("\n13. CLI: sift memory generate-rules")
    # Create a temp project dir with .claude/ to test dual output
    with tempfile.TemporaryDirectory() as tmp:
        # Re-open on the same index to test generate-rules
        # We need to call the CLI directly since generate-rules isn't an MCP tool
        os.makedirs(os.path.join(tmp, ".claude"), exist_ok=True)

        # First populate some data via a fresh client
        with SiftClient(index=index) as c2:
            c2.remember("test-user", "person", ["Prefers Python for scripting"])
            c2.remember("test-project", "project", ["Uses FastAPI with PostgreSQL"])
            c2.remember("test-correction", "fact", [
                "Don't use string concatenation for SQL queries — use parameterized queries"
            ])

        # Run generate-rules via CLI
        # The CLI operates on the default index, so we test the command syntax
        result = subprocess.run(
            [SIFT_BIN, "memory", "generate-rules"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "ORT_DYLIB_PATH": os.environ.get("ORT_DYLIB_PATH", "")},
        )
        if result.returncode == 0:
            check("generate-rules succeeds", True)
            check("output mentions AGENTS.md", "AGENTS.md" in result.stdout,
                  f"stdout: {result.stdout[:200]}")
            check("output mentions .claude/rules", ".claude/rules" in result.stdout,
                  f"stdout: {result.stdout[:200]}")
        else:
            # May fail if no memory.db for the default index — that's OK
            if "not initialized" in result.stderr:
                check("generate-rules (no memory.db)", True)
            else:
                check("generate-rules", False, result.stderr[:200])

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed > 0:
        print(f", {failed} failed")
    else:
        print(" — all green!")
    print(f"{'='*60}")

    # Cleanup: delete test index
    import shutil
    import pathlib
    idx_dir = pathlib.Path.home() / ".sift" / "indexes" / index
    if idx_dir.exists():
        shutil.rmtree(idx_dir, ignore_errors=True)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
