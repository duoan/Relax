# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolated compile-run-and-profile sandbox for generated CUDA kernels.

The sandbox executes untrusted model-generated code in a **separate process**
with a wall-clock timeout and a capped GPU-memory fraction, so it can coexist
with colocated training on the same GPU and never takes the trainer down with
it.

When ``profile=True`` the child is launched under **Nsight Systems** (``nsys``)
to capture per-kernel GPU timings and launch counts, and the compiled ``.so`` is
inspected with **cuobjdump** for static register / shared-memory usage. Both
tools work without GPU performance-counter permissions (unlike ``ncu``), so they
run inside the unprivileged training container. The resulting ``profile`` block
is fed back to the model between agentic turns so it can optimize.

Two entry points:
- ``evaluate_kernel(code, task, ...)`` — importable; spawns the child process
  and returns a structured result dict. Call it from a thread
  (``asyncio.to_thread``) so it never blocks the rollout event loop.
- ``python -m examples.cuda_kernels.sandbox --child ...`` — the child runner.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _default_device() -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Ray sets CUDA_VISIBLE_DEVICES="" on no-GPU actors; fall back to physical 0.
    if cvd is None or cvd.strip() == "":
        return os.environ.get("RELAX_CUDA_SANDBOX_DEVICE", "0")
    return cvd.split(",")[0]


def _new_result() -> dict:
    return {
        "compiled": False,
        "correct": False,
        "error": "",
        "latency_ms": None,
        "ref_latency_ms": None,
        "speedup": None,
        "max_abs_err": None,
        "profile": None,
    }


def evaluate_kernel(
    code: str,
    task: dict,
    timeout: float = 240.0,
    mem_fraction: float = 0.1,
    do_timing: bool = False,
    profile: bool = False,
) -> dict:
    """Compile + run ``code`` against ``task`` in an isolated child process.

    With ``profile=True`` the run is wrapped in ``nsys`` and the compiled
    extension is inspected with ``cuobjdump`` to build a ``profile`` report.
    """
    result = _new_result()

    with tempfile.TemporaryDirectory(prefix="cuda_sbx_") as tmp:
        code_file = os.path.join(tmp, "candidate.py")
        task_file = os.path.join(tmp, "task.json")
        result_file = os.path.join(tmp, "result.json")
        ext_dir = os.path.join(tmp, "torch_ext")
        prof_path = os.path.join(tmp, "prof")
        with open(code_file, "w") as f:
            f.write(code)
        with open(task_file, "w") as f:
            json.dump(task, f)

        env = dict(os.environ)
        # Pin the child to a single physical GPU and isolate its build cache so
        # parallel compilations of same-named kernels never clash.
        env["CUDA_VISIBLE_DEVICES"] = _default_device()
        env["TORCH_EXTENSIONS_DIR"] = ext_dir
        env["MAX_JOBS"] = env.get("MAX_JOBS", "4")

        child_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--child",
            "--code-file", code_file,
            "--task-file", task_file,
            "--result-file", result_file,
            "--mem-fraction", str(mem_fraction),
        ]
        if do_timing or profile:
            child_cmd.append("--do-timing")

        nsys = shutil.which("nsys") if profile else None
        if nsys:
            cmd = [
                nsys, "profile",
                "-o", prof_path,
                "--force-overwrite", "true",
                "-t", "cuda",
                "--sample", "none",
                "--cpuctxsw", "none",
            ] + child_cmd
        else:
            cmd = child_cmd

        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            result["error"] = f"sandbox timeout after {timeout}s"
            return result

        if not os.path.exists(result_file):
            tail = (proc.stderr or "")[-800:]
            result["error"] = f"sandbox crashed (rc={proc.returncode}): {tail}"
            return result

        try:
            with open(result_file) as f:
                result.update(json.load(f))
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"failed to read sandbox result: {exc}"
            return result

        if profile and result.get("compiled"):
            try:
                result["profile"] = _build_profile(prof_path, ext_dir, nsys)
            except Exception as exc:  # noqa: BLE001
                result["profile"] = {"note": f"profiling failed: {exc}"}

        return result


# ---------------------------------------------------------------------------
# Profiling (parent side): parse nsys report + cuobjdump resource usage
# ---------------------------------------------------------------------------


def _build_profile(prof_path: str, ext_dir: str, nsys: str | None) -> dict:
    profile: dict = {"kernels": [], "num_launches": 0, "has_memcpy": False}
    rep = prof_path + ".nsys-rep"
    if nsys and os.path.exists(rep):
        profile.update(_parse_nsys(rep, nsys))
    profile["resources"] = _parse_cuobjdump(ext_dir)
    return profile


# The sandbox also runs torch ops (input gen, the reference expr it benchmarks
# against, the allclose error check), so the nsys trace contains torch/cuBLAS
# internal kernels. Drop them so the report shows ONLY the model's own kernels.
_TORCH_KERNEL_RE = re.compile(
    r"at::|::native::|ampere_|cutlass|cudnn|sm\d+_xmma|xmma_|cublas|cudaError|"
    r"distribution_|curand|thrust::|vectorized_elementwise|reduce_kernel|"
    r"elementwise_kernel|CatArrayBatched|index_elementwise",
    re.IGNORECASE,
)


def _parse_nsys(rep: str, nsys: str) -> dict:
    out: dict = {"kernels": [], "num_launches": 0, "has_memcpy": False}
    try:
        proc = subprocess.run(
            [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", rep],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:  # noqa: BLE001
        return out

    rows = _csv_rows(proc.stdout)
    for row in rows:
        try:
            name = row["Name"]
            avg_ns = float(row["Avg (ns)"])
            inst = int(float(row["Instances"]))
        except (KeyError, ValueError):
            continue
        if _TORCH_KERNEL_RE.search(name):
            continue  # torch/cuBLAS internal kernel, not the model's
        out["kernels"].append({"name": name, "avg_us": avg_ns / 1000.0, "instances": inst})

    # Percentages relative to the model's own kernels only (torch ops filtered).
    total_us = sum(k["avg_us"] * k["instances"] for k in out["kernels"]) or 1.0
    for k in out["kernels"]:
        k["pct"] = 100.0 * k["avg_us"] * k["instances"] / total_us
    out["kernels"].sort(key=lambda k: k["pct"], reverse=True)
    # Distinct kernels launched per solution call (instance counts are just the
    # benchmark's repeat factor, so they are not surfaced to the model).
    out["num_launches"] = len(out["kernels"])

    # Memcpy in the steady state usually means avoidable host<->device traffic.
    try:
        mem = subprocess.run(
            [nsys, "stats", "--report", "cuda_gpu_mem_time_sum", "--format", "csv", rep],
            capture_output=True, text=True, timeout=120,
        )
        out["has_memcpy"] = any("memcpy" in line.lower() for line in mem.stdout.splitlines())
    except Exception:  # noqa: BLE001
        pass
    return out


def _csv_rows(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # nsys may prepend a few "Generating/Processing ..." status lines.
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Time (%)")), None)
    if start is None:
        return []
    return list(csv.DictReader(lines[start:]))


_REG_RE = re.compile(r"REG:(\d+)")
_SHARED_RE = re.compile(r"SHARED:(\d+)")


def _parse_cuobjdump(ext_dir: str) -> dict:
    cuobjdump = shutil.which("cuobjdump")
    sos = glob.glob(os.path.join(ext_dir, "**", "*.so"), recursive=True)
    if not cuobjdump or not sos:
        return {}
    try:
        proc = subprocess.run(
            [cuobjdump, "--dump-resource-usage", sos[0]],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return {}
    regs = [int(m) for m in _REG_RE.findall(proc.stdout)]
    shared = [int(m) for m in _SHARED_RE.findall(proc.stdout)]
    res: dict = {}
    if regs:
        res["max_registers_per_thread"] = max(regs)
    if shared:
        res["max_shared_bytes_per_block"] = max(shared)
    return res


# ---------------------------------------------------------------------------
# Child runner (executed in the spawned subprocess)
# ---------------------------------------------------------------------------


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--mem-fraction", type=float, default=0.1)
    parser.add_argument("--do-timing", action="store_true")
    args = parser.parse_args()

    result = _new_result()

    def _flush(res: dict) -> None:
        with open(args.result_file, "w") as f:
            json.dump(res, f)

    try:
        import traceback

        import torch

        with open(args.code_file) as f:
            code = f.read()
        with open(args.task_file) as f:
            task = json.load(f)

        if not torch.cuda.is_available():
            result["error"] = "CUDA not available inside sandbox"
            _flush(result)
            return

        device = "cuda"
        try:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
        except Exception:  # noqa: BLE001
            pass

        # --- compile / exec the candidate code -----------------------------
        namespace: dict = {}
        try:
            exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102
        except Exception:  # noqa: BLE001
            result["error"] = "compile/exec failed:\n" + traceback.format_exc(limit=6)
            _flush(result)
            return

        entry_point = task.get("entry_point", "solution")
        fn = namespace.get(entry_point)
        if not callable(fn):
            result["error"] = f"function '{entry_point}' not defined"
            _flush(result)
            return
        result["compiled"] = True

        # --- build inputs ---------------------------------------------------
        dtype_map = {"float32": torch.float32, "float16": torch.float16}
        inputs = {}
        ordered = []
        for name, shape, dtype in task["args"]:
            t = torch.randn(*shape, dtype=dtype_map.get(dtype, torch.float32), device=device)
            inputs[name] = t
            ordered.append(t)

        # --- reference (trusted expression) --------------------------------
        ref = eval(task["reference"], {"torch": torch}, dict(inputs))  # noqa: S307

        # --- run candidate --------------------------------------------------
        try:
            out = fn(*ordered)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            result["error"] = "runtime error:\n" + traceback.format_exc(limit=6)
            _flush(result)
            return

        if not torch.is_tensor(out) or tuple(out.shape) != tuple(ref.shape):
            result["error"] = f"output shape mismatch: got {getattr(out, 'shape', None)}, want {tuple(ref.shape)}"
            _flush(result)
            return

        rtol = float(task.get("rtol", 1e-3))
        atol = float(task.get("atol", 1e-3))
        with torch.no_grad():
            result["max_abs_err"] = float((out.float() - ref.float()).abs().max().item())
        result["correct"] = bool(torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol))

        # --- optional timing ------------------------------------------------
        if args.do_timing and result["correct"]:
            result["latency_ms"] = _bench(fn, ordered)
            result["ref_latency_ms"] = _bench(lambda: eval(task["reference"], {"torch": torch}, dict(inputs)), [])
            if result["latency_ms"] and result["latency_ms"] > 0:
                result["speedup"] = result["ref_latency_ms"] / result["latency_ms"]

        _flush(result)
    except Exception:  # noqa: BLE001
        import traceback

        result["error"] = "sandbox internal error:\n" + traceback.format_exc(limit=6)
        try:
            _flush(result)
        except Exception:  # noqa: BLE001
            pass


def _bench(fn, fn_args, warmup: int = 5, iters: int = 30) -> float:
    import torch

    for _ in range(warmup):
        fn(*fn_args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*fn_args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    _child_main()
