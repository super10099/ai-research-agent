"""
Tests for src/tools/executor.py — tool dispatch and parallel execution.

Fake ToolUseBlocks are plain SimpleNamespace objects with .name/.input/.id —
executor.py only accesses those three attributes duck-typed, so there's no
need to construct a real anthropic.types.ToolUseBlock.
"""

import time
from types import SimpleNamespace

from src.tools import executor


def _tool_use(name: str, input: dict, id: str = "tu_1"):
    return SimpleNamespace(name=name, input=input, id=id)


class TestRunOne:
    async def test_unknown_tool_returns_error_result(self):
        tu = _tool_use("nonexistent_tool", {}, id="tu_x")

        result = await executor._run_one(tu)

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu_x"
        assert "unknown tool" in result["content"]
        assert "nonexistent_tool" in result["content"]

    async def test_tool_exception_is_caught_and_reported(self, monkeypatch):
        def broken_tool(**kwargs):
            raise ValueError("something went wrong")
        monkeypatch.setitem(executor._TOOL_REGISTRY, "broken_tool", broken_tool)
        tu = _tool_use("broken_tool", {}, id="tu_y")

        result = await executor._run_one(tu)

        assert "Tool execution error" in result["content"]
        assert "something went wrong" in result["content"]

    async def test_successful_tool_call(self, monkeypatch):
        def echo_tool(query: str) -> str:
            return f"echo: {query}"
        monkeypatch.setitem(executor._TOOL_REGISTRY, "echo_tool", echo_tool)
        tu = _tool_use("echo_tool", {"query": "hi"}, id="tu_z")

        result = await executor._run_one(tu)

        assert result == {"type": "tool_result", "tool_use_id": "tu_z", "content": "echo: hi"}


class TestExecuteToolsParallel:
    async def test_runs_concurrently_not_sequentially(self, monkeypatch):
        def slow_tool(**kwargs):
            time.sleep(0.2)
            return "done"
        monkeypatch.setitem(executor._TOOL_REGISTRY, "slow_tool", slow_tool)

        tool_uses = [_tool_use("slow_tool", {}, id=f"tu_{i}") for i in range(3)]

        start = time.monotonic()
        await executor.execute_tools_parallel(tool_uses)
        elapsed = time.monotonic() - start

        # Sequential would take >= 0.6s; parallel (thread pool) should be ~0.2s.
        assert elapsed < 0.4

    async def test_preserves_input_order_not_completion_order(self, monkeypatch):
        def variable_delay_tool(delay: float, label: str) -> str:
            time.sleep(delay)
            return label
        monkeypatch.setitem(executor._TOOL_REGISTRY, "variable_delay_tool", variable_delay_tool)

        tool_uses = [
            _tool_use("variable_delay_tool", {"delay": 0.15, "label": "slow"}, id="tu_a"),
            _tool_use("variable_delay_tool", {"delay": 0.0, "label": "fast"}, id="tu_b"),
        ]

        results = await executor.execute_tools_parallel(tool_uses)

        # "slow" finishes after "fast", but result order must match input order.
        assert [r["content"] for r in results] == ["slow", "fast"]
        assert [r["tool_use_id"] for r in results] == ["tu_a", "tu_b"]

    async def test_empty_list_returns_empty_list(self):
        assert await executor.execute_tools_parallel([]) == []
