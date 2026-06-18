# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA kernel agent mainline (OpenHands-style tool-use loop).

Launched by Relax agentic rollout (one process per session). It:
1. reads ``RELAX_INPUT_JSON`` (messages + metadata with the task's ``code``),
2. builds a per-session CUDA workspace and writes ``model.py``,
3. runs a ReAct loop against the Relax chat API: the model emits
   ``<tool_call>`` blocks, we execute Bash/Read/Write/Edit/Glob/Grep on the
   workspace and feed results back as tool observations,
4. runs an authoritative compile+verify+profile on the final workspace and
   writes the measurements to ``RELAX_OUTPUT_JSON`` metadata for the reward.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import APIStatusError, AsyncOpenAI

from app.tools import Workspace
from app.workspace import build_workspace, evaluate_final


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the workspace root (compile/verify/profile, ls, etc.).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read a file in the workspace (returns numbered lines).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Write/overwrite a file in the workspace (kernels/*.cu, *_binding.cpp, model_new.py).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Replace old_string with new_string in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "multi_edit", "description": "Apply multiple {old_string,new_string} edits to one file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object"}}}, "required": ["path", "edits"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "List files matching a glob pattern (e.g. kernels/**/*.cu).",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search file contents with a regex.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
]


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args or {}})
    return calls


async def run_session(messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(Path(__file__).with_name("cuda_agent_config.yaml").read_text())
    max_turns = int(config["max_turns"])
    bash_timeout = float(config.get("bash_timeout", 600.0))

    session_io = os.environ.get("RELAX_SESSION_IO_DIR", "/tmp")
    ws_path = build_workspace(os.path.join(session_io, "workspace"), metadata["code"])
    ws = Workspace(ws_path, bash_timeout=bash_timeout)

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "relax"),
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=httpx.Timeout(timeout=1200.0, connect=30.0),
    )

    stop_reason = "max_turns"
    turns = 0
    for _turn in range(max_turns):
        turns += 1
        try:
            resp = await client.chat.completions.create(model="model", messages=messages, tools=TOOLS)
        except APIStatusError as exc:
            err = {}
            try:
                err = exc.response.json().get("error", {})
            except Exception:  # noqa: BLE001
                pass
            if isinstance(err, dict) and err.get("code") == "context_length_exceeded":
                stop_reason = "finish_length"
                break
            raise

        choice = resp.choices[0]
        text = choice.message.content or ""
        messages.append({"role": "assistant", "content": text})
        if choice.finish_reason == "length":
            stop_reason = "finish_length"
            break

        tool_calls = _extract_tool_calls(text)
        if not tool_calls:
            stop_reason = "no_tool_call"
            break

        for call in tool_calls:
            observation = ws.dispatch(call["name"], call["arguments"])
            messages.append({"role": "tool", "content": observation})

    # Authoritative final evaluation (GPU): compile -> verify -> profile.
    ev = await asyncio.to_thread(evaluate_final, ws_path)
    ev.update({"num_turns": turns, "stop_reason": stop_reason})
    return {"metadata": ev}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    out = asyncio.run(run_session(payload["messages"], payload.get("metadata", {})))
    Path(args.output_json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
