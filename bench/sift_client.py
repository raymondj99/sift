"""Minimal MCP client for sift — spawns `sift mcp` and talks JSON-RPC over stdio.

Usage:
    client = SiftClient()
    client.remember("Raymond", "person", ["prefers Rust over Python"])
    results = client.recall("Rust programming")
    client.close()

Or as a context manager:
    with SiftClient() as client:
        client.remember(...)
"""

from __future__ import annotations

import json
import os
import subprocess


class SiftClient:
    """MCP JSON-RPC client that communicates with `sift mcp` over stdio."""

    def __init__(self, sift_bin: str | None = None, index: str = "default"):
        self.sift_bin = sift_bin or os.environ.get("SIFT_BIN", "sift")
        self._request_id = 0
        self._proc = subprocess.Popen(
            [self.sift_bin, "--index", index, "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._initialize()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)

    # -- High-level API -------------------------------------------------------

    def remember(
        self,
        entity: str,
        entity_type: str,
        observations: list[str],
        relationships: list[dict] | None = None,
    ) -> dict:
        """Store facts about an entity."""
        args: dict = {
            "entity": entity,
            "entity_type": entity_type,
            "observations": observations,
        }
        if relationships:
            args["relationships"] = relationships
        return self.call_tool("sift_remember", args)

    def recall(
        self,
        query: str,
        limit: int = 10,
        entity_names: list[str] | None = None,
        **filters,
    ) -> list[dict]:
        """Search memory by natural language query.

        Args:
            entity_names: If set, only return memories about these entities.
                          Dramatically improves precision for entity-specific questions.
        """
        args: dict = {"query": query, "limit": limit, **filters}
        if entity_names:
            args["entity_names"] = entity_names
        result = self.call_tool("sift_recall", args)
        return result.get("memories", [])

    def consolidate(self) -> dict:
        """Run the consolidation pipeline."""
        return self.call_tool("sift_consolidate", {})

    def memory_status(self) -> dict:
        """Get memory system statistics."""
        return self.call_tool("sift_memory_status", {})

    def forget_entity(self, entity: str) -> dict:
        """Delete an entity and all its observations."""
        return self.call_tool("sift_forget_entity", {"entity": entity})

    # -- Low-level MCP protocol -----------------------------------------------

    def _initialize(self):
        """Send the MCP initialize handshake."""
        self._send_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "sift-bench", "version": "0.1.0"},
        })
        self._send_notification("notifications/initialized", {})

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the parsed result."""
        resp = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if "error" in resp:
            raise RuntimeError(f"MCP tool error: {resp['error']}")
        result = resp.get("result", {})
        for c in result.get("content", []):
            if c.get("type") == "text":
                try:
                    return json.loads(c["text"])
                except json.JSONDecodeError:
                    return {"text": c["text"]}
        return result

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._write(msg)
        return self._read_response(self._request_id)

    def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        self._write({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    def _write(self, msg: dict):
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, expected_id: int) -> dict:
        """Read lines until we get a response matching expected_id."""
        while True:
            line = self._proc.stdout.readline()
            if not line:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"sift mcp process exited unexpectedly: {stderr[:500]}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == expected_id:
                return msg


def smoke_test():
    """Quick smoke test — verifies the MCP connection works."""
    print("Connecting to sift MCP server...")
    with SiftClient() as client:
        status = client.memory_status()
        print(f"  Entities: {status.get('total_entities', '?')}")
        print(f"  Observations: {status.get('total_observations', '?')}")

        client.remember("bench-test", "concept", ["smoke test observation for benchmarking"])
        results = client.recall("smoke test benchmarking")
        found = any("smoke test" in m.get("fact", "") for m in results)
        print(f"  Remember+recall roundtrip: {'OK' if found else 'FAIL'}")

        client.forget_entity("bench-test")
        print("Smoke test passed.")


if __name__ == "__main__":
    smoke_test()
