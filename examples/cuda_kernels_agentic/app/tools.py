# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Minimal OpenHands-style tool layer for the CUDA agent.

All tools operate inside a single per-session workspace root. Infrastructure
files (``binding.cpp``, ``binding_registry.h``, ``utils/``, ``model.py``) are
protected: write/edit operations against them are refused, mirroring the
file-permission isolation the CUDA-Agent paper uses to prevent reward hacking.

Tools intentionally match the common coding-agent surface so the assistant can
drive a real edit/compile/verify/profile workflow:
``bash``, ``read``, ``write``, ``edit``, ``multi_edit``, ``glob``, ``grep``.
"""

from __future__ import annotations

import glob as _glob
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


MAX_OUTPUT = 6000  # chars returned to the model per tool call

PROTECTED = {"binding.cpp", "binding_registry.h", "model.py"}
PROTECTED_DIRS = {"utils"}


def _truncate(text: str) -> str:
    if text is None:
        return ""
    if len(text) <= MAX_OUTPUT:
        return text
    head = text[: MAX_OUTPUT // 2]
    tail = text[-MAX_OUTPUT // 2 :]
    return f"{head}\n... [truncated {len(text) - MAX_OUTPUT} chars] ...\n{tail}"


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: str, bash_timeout: float = 600.0) -> None:
        self.root = Path(root).resolve()
        self.bash_timeout = bash_timeout

    # --- path safety -------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ToolError(f"path escapes workspace: {rel}")
        return p

    def _check_writable(self, rel: str) -> None:
        norm = os.path.normpath(rel).lstrip("./")
        top = norm.split("/", 1)[0]
        if norm in PROTECTED or top in PROTECTED_DIRS:
            raise ToolError(f"'{rel}' is a protected infrastructure file and cannot be modified.")

    # --- tools -------------------------------------------------------------
    def bash(self, command: str) -> str:
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.bash_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[bash timeout after {self.bash_timeout}s]"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return _truncate(f"$ {command}\n{out}\n[exit code: {proc.returncode}]")

    def read(self, path: str, limit: int | None = None) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        lines = p.read_text(errors="replace").splitlines()
        if limit:
            lines = lines[:limit]
        numbered = "\n".join(f"{i + 1:6d}|{ln}" for i, ln in enumerate(lines))
        return _truncate(numbered) if numbered else "[empty file]"

    def write(self, path: str, content: str) -> str:
        self._check_writable(path)
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[wrote {len(content)} chars to {path}]"

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        self._check_writable(path)
        p = self._resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        text = p.read_text()
        count = text.count(old_string)
        if count == 0:
            raise ToolError("old_string not found")
        if count > 1 and not replace_all:
            raise ToolError(f"old_string is not unique ({count} matches); pass replace_all or add context")
        text = text.replace(old_string, new_string)
        p.write_text(text)
        return f"[edited {path} ({count} replacement(s))]"

    def multi_edit(self, path: str, edits: list[dict[str, Any]]) -> str:
        self._check_writable(path)
        p = self._resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        text = p.read_text()
        for e in edits:
            old, new = e["old_string"], e["new_string"]
            if old not in text:
                raise ToolError(f"old_string not found: {old[:60]!r}")
            text = text.replace(old, new, -1 if e.get("replace_all") else 1)
        p.write_text(text)
        return f"[applied {len(edits)} edits to {path}]"

    def glob(self, pattern: str) -> str:
        matches = _glob.glob(str(self.root / pattern), recursive=True)
        rel = sorted(os.path.relpath(m, self.root) for m in matches)
        return _truncate("\n".join(rel)) if rel else "[no matches]"

    def grep(self, pattern: str, path: str = ".") -> str:
        target = self._resolve(path)
        try:
            proc = subprocess.run(
                ["rg", "-n", "--no-heading", pattern, str(target)],
                cwd=str(self.root), capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            proc = subprocess.run(
                ["grep", "-rn", pattern, str(target)],
                cwd=str(self.root), capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "[grep timeout]"
        return _truncate(proc.stdout) if proc.stdout else "[no matches]"

    # --- dispatch ----------------------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "bash":
                return self.bash(arguments["command"])
            if name == "read":
                return self.read(arguments["path"], arguments.get("limit"))
            if name == "write":
                return self.write(arguments["path"], arguments["content"])
            if name == "edit":
                return self.edit(
                    arguments["path"], arguments["old_string"], arguments["new_string"],
                    bool(arguments.get("replace_all", False)),
                )
            if name == "multi_edit":
                return self.multi_edit(arguments["path"], arguments["edits"])
            if name == "glob":
                return self.glob(arguments["pattern"])
            if name == "grep":
                return self.grep(arguments["pattern"], arguments.get("path", "."))
            return f"[unknown tool: {name}]"
        except ToolError as exc:
            return f"[tool error: {exc}]"
        except KeyError as exc:
            return f"[tool error: missing argument {exc}]"
        except Exception as exc:  # noqa: BLE001
            return f"[tool error: {type(exc).__name__}: {exc}]"


TOOL_NAMES = {"bash", "read", "write", "edit", "multi_edit", "glob", "grep"}
