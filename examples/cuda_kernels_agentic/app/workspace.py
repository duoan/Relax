# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-session CUDA workspace: build it, protect infra, run authoritative eval.

Each agent session gets a fresh copy of ``workdir_template`` with ``model.py``
set to the task's reference module. Infrastructure (``utils/``, ``binding*.``)
is made read-only so the policy can only touch ``kernels/`` and ``model_new.py``
— matching the paper's permission isolation against reward hacking.

``evaluate_final`` compiles + verifies + profiles the final workspace state and
returns a structured dict the reward function turns into the milestone reward.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


_TEMPLATE = Path(__file__).resolve().parent.parent / "workdir_template"
_VERIFY_RE = re.compile(r"RELAX_VERIFY_JSON:\s*(\{.*\})")
_PROFILE_RE = re.compile(r"RELAX_PROFILE_JSON:\s*(\{.*\})")


def build_workspace(dest: str, task_code: str) -> str:
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(_TEMPLATE, dest_path)
    (dest_path / "model.py").write_text(task_code)
    # Protect infrastructure: read-only files, read/exec-only utils dir.
    for rel in ("binding.cpp", "binding_registry.h", "model.py"):
        try:
            os.chmod(dest_path / rel, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    for p in (dest_path / "utils").glob("*.py"):
        try:
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    return str(dest_path)


def _parse_last(regex: re.Pattern, text: str) -> dict[str, Any] | None:
    matches = regex.findall(text or "")
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def _run(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    # Isolate compile caches per workspace so concurrent load_inline / cpp_extension
    # builds (same kernel name across parallel samples) never clash, and bound nvcc
    # parallelism. Inherit the rest of the environment (CUDA, PYTHONPATH, GPU bounds).
    env = dict(os.environ)
    env["TORCH_EXTENSIONS_DIR"] = os.path.join(cwd, "torch_ext")
    env.setdefault("MAX_JOBS", "4")
    # Relax runs custom rewards in CPU-only workers (CUDA_VISIBLE_DEVICES=""), but
    # the eval needs a GPU to (a) let cpp_extension detect the arch at compile time
    # and (b) run verification / profiling. Pin a fixed arch so compilation never
    # depends on GPU visibility, and expose a physical GPU when the worker has none.
    env.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("RELAX_CUDA_ARCH", "8.0"))
    # Expose all eval GPUs to the subprocess; gpu_slot narrows CUDA_VISIBLE_DEVICES
    # to one card per slot (round-robin) so evals spread across both GPUs.
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("RELAX_EVAL_CUDA_DEVICES", "0,1")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def evaluate_final(workspace: str, compile_timeout: float = 600.0, eval_timeout: float = 600.0) -> dict[str, Any]:
    """Authoritative compile -> verify -> profile of the final workspace."""
    result: dict[str, Any] = {
        "compiled": False, "correct": False, "max_abs_err": None,
        "eager_us": None, "compile_us": None, "cuda_us": None, "error": "",
    }
    ws = Path(workspace)
    if not (ws / "model_new.py").exists():
        result["error"] = "model_new.py was never created"
        return result

    # 1) compile
    try:
        cp = _run(["bash", "utils/compile.sh"], str(ws), compile_timeout)
    except subprocess.TimeoutExpired:
        result["error"] = "compile timeout"
        return result
    if cp.returncode != 0 or not (ws / "cuda_extension.so").exists():
        result["error"] = "compile failed:\n" + (cp.stdout or "")[-1200:] + (cp.stderr or "")[-600:]
        return result
    result["compiled"] = True

    # 2) verify
    try:
        vp = _run(["python", "-m", "utils.verification"], str(ws), eval_timeout)
    except subprocess.TimeoutExpired:
        result["error"] = "verification timeout"
        return result
    vjson = _parse_last(_VERIFY_RE, vp.stdout) or {}
    result["correct"] = bool(vjson.get("correct"))
    result["max_abs_err"] = vjson.get("max_abs_err")
    if not result["correct"]:
        result["error"] = str(vjson.get("error") or vp.stdout[-800:])
        return result

    # 3) profile (only meaningful when correct)
    try:
        pp = _run(["python", "-m", "utils.profiling", "--iters", "20"], str(ws), eval_timeout)
    except subprocess.TimeoutExpired:
        result["error"] = "profiling timeout"
        return result
    pjson = _parse_last(_PROFILE_RE, pp.stdout) or {}
    result["eager_us"] = pjson.get("eager_us")
    result["compile_us"] = pjson.get("compile_us")
    result["cuda_us"] = pjson.get("cuda_us")
    return result
