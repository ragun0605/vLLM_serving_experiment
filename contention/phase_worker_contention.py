#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import string
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import psutil
import torch
from vllm import LLMEngine, EngineArgs, SamplingParams

try:
    from vllm.inputs import TokensPrompt
except Exception:
    TokensPrompt = None


# ============================================================
# Config
# ============================================================

PROFILE_DIR = Path(os.environ.get("VLLM_PROFILE_DIR", "./profile_logs"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_SPECS_PATH = Path(
    os.environ.get("VLLM_PROMPT_SPECS_PATH", str(PROFILE_DIR / "prompt_specs.json"))
)

ITER_CSV = PROFILE_DIR / "iteration_metrics.csv"

MODEL_NAME = os.environ.get(
    "VLLM_MODEL_NAME",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
)

NUM_REQUESTS = int(os.environ.get("VLLM_NUM_REQUESTS", "3"))
INPUT_TOKENS = int(os.environ.get("VLLM_INPUT_TOKENS", "256"))
OUTPUT_TOKENS = int(os.environ.get("VLLM_OUTPUT_TOKENS", "256"))

NUM_WARMUPS = int(os.environ.get("VLLM_NUM_WARMUPS", "1"))
WARMUP_OUTPUT_TOKENS = int(os.environ.get("VLLM_WARMUP_OUTPUT_TOKENS", "32"))

TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "2"))
EXPECTED_TP_ROWS = int(os.environ.get("VLLM_EXPECTED_TP_ROWS", str(TENSOR_PARALLEL_SIZE)))

GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))

MAX_MODEL_LEN_ENV = os.environ.get("VLLM_MAX_MODEL_LEN", "")
MAX_MODEL_LEN = int(MAX_MODEL_LEN_ENV) if MAX_MODEL_LEN_ENV.strip() else None

USE_ENFORCE_EAGER = os.environ.get("VLLM_USE_ENFORCE_EAGER", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SAMPLE_INTERVAL_SEC = float(os.environ.get("VLLM_THREAD_SAMPLE_INTERVAL_SEC", "0.001"))
ACTIVE_THREAD_CPU_EPS = float(os.environ.get("VLLM_ACTIVE_THREAD_CPU_EPS", "0.0"))
SEED = int(os.environ.get("VLLM_RANDOM_SEED", "1234"))

# ============================================================
# Scheduler-hold / min-batch barrier compatibility
# ============================================================
# These defaults intentionally match the scheduler-hold experiment files.
# The sitecustomize_scheduler_hold_v2.py patch handles scheduler-side hold
# even when VLLM_DISABLE_GPU_PROFILE_PATCH=1. This worker keeps the same
# driver-side barrier / latency-baseline behavior as phase_worker_scheduler_hold_fixed.py.
MIN_BATCH_BARRIER_MS = float(os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "0"))
MIN_BATCH_TARGET_REQUESTS = int(
    os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", str(NUM_REQUESTS))
)
EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = os.environ.get(
    "VLLM_EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY", "1"
).lower() in {"1", "true", "yes", "on"}

RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = os.environ.get(
    "VLLM_RESET_LATENCY_BASELINE_AFTER_ADD_LOOP", "1"
).lower() in {"1", "true", "yes", "on"}

# ============================================================
# CPU contention tracing config
# ============================================================

# Optional: restrict this worker process (and child vLLM workers) to a CPU set.
# Example: export VLLM_CPUSET=0-3 or VLLM_CPUSET=0,1,2,3
CPUSET_ENV = os.environ.get("VLLM_CPUSET", "").strip()

# If set, this overrides the denominator used for contention pressure.
# Otherwise len(os.sched_getaffinity(0)) is used.
CONTENTION_CORE_COUNT_ENV = os.environ.get("VLLM_CONTENTION_CORE_COUNT", "").strip()

# A step is considered contention-suspicious if runnable threads exceed available cores,
# or if CPU utilization is close to the core limit.
CONTENTION_CPU_UTIL_THRESHOLD = float(os.environ.get("VLLM_CONTENTION_CPU_UTIL_THRESHOLD", "0.85"))
CONTENTIOUS_TOP_N_THREADS = int(os.environ.get("VLLM_CONTENTION_TOP_N_THREADS", "8"))

# Per-sample thread identity tracing. This is the key output for finding
# *which* runnable/active threads competed for restricted CPU cores.
# Rows are written to active_thread_contention_detail.csv.
TRACE_THREAD_IDENTITIES = os.environ.get("VLLM_TRACE_THREAD_IDENTITIES", "1").lower() in {
    "1", "true", "yes", "on"
}



# ============================================================
# psutil helpers
# ============================================================

def safe_process(pid: int):
    try:
        return psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None


def collect_process_tree(root_pid: int) -> Dict[int, dict]:
    root = safe_process(root_pid)
    if root is None:
        return {}

    try:
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        processes = [root]

    tree = {}

    for proc in processes:
        try:
            with proc.oneshot():
                cpu_times = proc.cpu_times()
                threads = proc.threads()

                tree[proc.pid] = {
                    "pid": proc.pid,
                    "ppid": proc.ppid(),
                    "name": proc.name(),
                    "num_threads": proc.num_threads(),
                    "proc_cpu_s": float(cpu_times.user + cpu_times.system),
                    "thread_cpu_s": {
                        int(th.id): float(th.user_time + th.system_time)
                        for th in threads
                    },
                }
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue

    return tree


def flatten_thread_cpu(tree: Dict[int, dict]) -> Dict[Tuple[int, int], float]:
    flat = {}

    for pid, snap in tree.items():
        for tid, cpu_s in snap["thread_cpu_s"].items():
            flat[(pid, tid)] = cpu_s

    return flat


def tree_total_threads(tree: Dict[int, dict]) -> int:
    return sum(snap["num_threads"] for snap in tree.values())


def tree_total_proc_cpu(tree: Dict[int, dict]) -> float:
    return sum(snap["proc_cpu_s"] for snap in tree.values())


def diff_thread_cpu(before: Dict[Tuple[int, int], float],
                    after: Dict[Tuple[int, int], float],
                    exclude_keys=None):
    active_count = 0
    cpu_delta_sum = 0.0
    active_keys = set()

    if exclude_keys is None:
        exclude_keys = set()

    for key, after_cpu in after.items():
        if key in exclude_keys:
            continue

        delta = max(0.0, float(after_cpu) - float(before.get(key, 0.0)))

        if delta > ACTIVE_THREAD_CPU_EPS:
            active_count += 1
            cpu_delta_sum += delta
            active_keys.add(key)

    return active_count, cpu_delta_sum, active_keys


def _percentile(values, q: float):
    vals = sorted(float(v) for v in values if v is not None)

    if not vals:
        return 0.0

    if len(vals) == 1:
        return vals[0]

    q = min(1.0, max(0.0, float(q)))
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _effective_cpu_cores(process_cpu_delta_s, step_wall_ms):
    try:
        wall_s = float(step_wall_ms) / 1000.0
        if wall_s <= 0:
            return 0.0
        return float(process_cpu_delta_s) / wall_s
    except Exception:
        return 0.0



def _parse_cpuset_list(s: str):
    cpus = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def maybe_apply_cpu_affinity_from_env():
    if not CPUSET_ENV:
        return []
    cpus = _parse_cpuset_list(CPUSET_ENV)
    if not cpus:
        return []
    try:
        os.sched_setaffinity(0, set(cpus))
        print(f"[INFO] Applied CPU affinity from VLLM_CPUSET={CPUSET_ENV} -> {cpus}")
    except Exception as e:
        print(f"[WARN] Failed to apply CPU affinity VLLM_CPUSET={CPUSET_ENV}: {e}")
    return cpus


def get_available_core_count():
    if CONTENTION_CORE_COUNT_ENV:
        try:
            return max(1, int(CONTENTION_CORE_COUNT_ENV))
        except Exception:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)



def _read_task_wchan(pid: int, tid: int) -> str:
    """Kernel wait channel for the thread.

    For runnable threads this is often "0" or empty; for sleeping threads it can
    reveal whether the thread is blocked in futex, epoll, pipe_read, etc.
    """
    try:
        with open(f"/proc/{pid}/task/{tid}/wchan", "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""


def _read_task_last_cpu(pid: int, tid: int):
    """Return the last CPU id on which the task ran, from /proc stat if available."""
    try:
        raw = Path(f"/proc/{pid}/task/{tid}/stat").read_text(encoding="utf-8", errors="replace")
        # comm may contain spaces inside parentheses. Fields after the last ')' are stable.
        after = raw.rsplit(")", 1)[1].strip().split()
        # In proc_pid_stat, processor is field 39 (1-indexed). After removing pid+comm,
        # the state starts at field 3, so processor is after[36].
        if len(after) > 36:
            return int(after[36])
    except Exception:
        pass
    return ""


def _guess_process_role(pid: int, root_pid: int, process_name: str, cmdline: str) -> str:
    if int(pid) == int(root_pid):
        return "driver_main_process"
    c = (cmdline or "").lower()
    n = (process_name or "").lower()
    if "enginecore" in c or "engine_core" in c:
        return "vllm_engine_core"
    if "worker" in c or "vllm" in c:
        return "vllm_worker_or_child"
    if "python" in n:
        return "python_child_process"
    return "child_process"


def _python_native_thread_name_map():
    """Map native TID -> Python thread name for the current process only."""
    out = {}
    try:
        for th in threading.enumerate():
            native_id = getattr(th, "native_id", None)
            if native_id is not None:
                out[int(native_id)] = th.name
    except Exception:
        pass
    return out

def _read_task_status(pid: int, tid: int):
    """Read lightweight per-thread scheduling state from /proc.

    Returns fields useful for contention diagnosis:
    - state: R/S/D/etc. R means currently runnable/running at sample instant.
    - voluntary/nonvoluntary context switch counters.
    """
    info = {
        "name": "",
        "state": "",
        "voluntary_ctxt_switches": 0,
        "nonvoluntary_ctxt_switches": 0,
    }
    path = f"/proc/{pid}/task/{tid}/status"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    info["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("State:"):
                    # Example: "State:\tS (sleeping)" -> "S"
                    info["state"] = line.split(":", 1)[1].strip().split()[0]
                elif line.startswith("voluntary_ctxt_switches:"):
                    info["voluntary_ctxt_switches"] = int(line.rsplit(None, 1)[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    info["nonvoluntary_ctxt_switches"] = int(line.rsplit(None, 1)[1])
    except Exception:
        pass
    return info


def collect_thread_runtime_snapshot(root_pid: int) -> Dict[Tuple[int, int], dict]:
    """Collect CPU time + runnable state + context switch counters for all threads
    in the vLLM process tree. Also records thread identity metadata.
    """
    root = safe_process(root_pid)
    if root is None:
        return {}

    try:
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        processes = [root]

    out = {}
    for proc in processes:
        try:
            with proc.oneshot():
                pid = int(proc.pid)
                pname = proc.name()
                ppid = proc.ppid()
                try:
                    cmdline = " ".join(proc.cmdline())
                except Exception:
                    cmdline = ""
                process_role = _guess_process_role(pid, root_pid, pname, cmdline)
                threads = proc.threads()
            py_thread_names = _python_native_thread_name_map() if pid == os.getpid() else {}
            for th in threads:
                tid = int(th.id)
                status = _read_task_status(pid, tid)
                proc_thread_name = status.get("name", "")
                python_thread_name = py_thread_names.get(tid, "")
                # Prefer Python-level name for the current process when available,
                # but keep the kernel comm field separately.
                thread_name = python_thread_name or proc_thread_name
                out[(pid, tid)] = {
                    "pid": pid,
                    "tid": tid,
                    "ppid": ppid,
                    "process_name": pname,
                    "process_role": process_role,
                    "process_cmdline": cmdline[:240],
                    "thread_name": thread_name,
                    "kernel_thread_comm": proc_thread_name,
                    "python_thread_name": python_thread_name,
                    "state": status.get("state", ""),
                    "wchan": _read_task_wchan(pid, tid),
                    "last_cpu": _read_task_last_cpu(pid, tid),
                    "cpu_s": float(th.user_time + th.system_time),
                    "voluntary_ctxt_switches": int(status.get("voluntary_ctxt_switches", 0)),
                    "nonvoluntary_ctxt_switches": int(status.get("nonvoluntary_ctxt_switches", 0)),
                }
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue
    return out


def _snapshot_cpu_map(snapshot):
    return {k: float(v.get("cpu_s", 0.0)) for k, v in snapshot.items()}


def _ctx_switch_delta(before, after):
    vol = 0
    invol = 0
    for key, a in after.items():
        b = before.get(key, {})
        vol += max(0, int(a.get("voluntary_ctxt_switches", 0)) - int(b.get("voluntary_ctxt_switches", 0)))
        invol += max(0, int(a.get("nonvoluntary_ctxt_switches", 0)) - int(b.get("nonvoluntary_ctxt_switches", 0)))
    return vol, invol


def _top_cpu_threads(before, after, top_n=8, exclude_keys=None):
    exclude_keys = set(exclude_keys or [])
    rows = []
    for key, a in after.items():
        if key in exclude_keys:
            continue
        b = before.get(key, {})
        delta_s = max(0.0, float(a.get("cpu_s", 0.0)) - float(b.get("cpu_s", 0.0)))
        if delta_s <= ACTIVE_THREAD_CPU_EPS:
            continue
        rows.append({
            "key": key,
            "cpu_ms": delta_s * 1000.0,
            "pid": a.get("pid", key[0]),
            "tid": a.get("tid", key[1]),
            "process_name": a.get("process_name", ""),
            "thread_name": a.get("thread_name", ""),
            "state": a.get("state", ""),
            "nonvoluntary_ctx_switch_delta": max(0, int(a.get("nonvoluntary_ctxt_switches", 0)) - int(b.get("nonvoluntary_ctxt_switches", 0))),
            "voluntary_ctx_switch_delta": max(0, int(a.get("voluntary_ctxt_switches", 0)) - int(b.get("voluntary_ctxt_switches", 0))),
        })
    rows.sort(key=lambda r: r["cpu_ms"], reverse=True)
    return rows[:int(top_n)]


def _format_top_threads(rows):
    parts = []
    for r in rows:
        name = r.get("thread_name") or r.get("process_name") or "?"
        parts.append(
            f"{r['pid']}:{r['tid']}:{name}:cpu_ms={r['cpu_ms']:.3f}:ivcs={r['nonvoluntary_ctx_switch_delta']}"
        )
    return "|".join(parts)


class StepSampler:
    def __init__(self, root_pid: int, interval_sec: float):
        self.root_pid = root_pid
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.samples: List[dict] = []
        self.thread_detail_rows: List[dict] = []
        self.thread_keys_seen = set()
        self.cpu_active_thread_keys_seen = set()
        self.thread = None
        self.sampler_thread_key = None
        self._prev_snapshot = None
        self.available_cores = get_available_core_count()

    def start(self, global_step: int, initial_thread_cpu=None, initial_runtime_snapshot=None):
        self.samples = []
        self.thread_detail_rows = []
        self.thread_keys_seen = set()
        self.cpu_active_thread_keys_seen = set()
        self.sampler_thread_key = None
        self._prev_snapshot = dict(initial_runtime_snapshot or {})
        self.available_cores = get_available_core_count()
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            args=(global_step,),
            daemon=True,
            name=f"step-sampler-{global_step}",
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def _run(self, global_step: int):
        t0 = time.perf_counter()
        sample_idx = 0
        self.sampler_thread_key = (os.getpid(), threading.get_native_id())

        while not self.stop_event.is_set():
            tree = collect_process_tree(self.root_pid)
            snapshot = collect_thread_runtime_snapshot(self.root_pid)
            flat_prev = _snapshot_cpu_map(self._prev_snapshot or {})
            flat_now = _snapshot_cpu_map(snapshot)

            total_threads = tree_total_threads(tree)
            process_count = len(tree)

            for key in snapshot.keys():
                if key != self.sampler_thread_key:
                    self.thread_keys_seen.add(key)

            active_count, cpu_delta_s, active_keys = diff_thread_cpu(
                flat_prev,
                flat_now,
                exclude_keys={self.sampler_thread_key},
            )
            self.cpu_active_thread_keys_seen.update(active_keys)

            runnable_count = sum(
                1 for key, meta in snapshot.items()
                if key != self.sampler_thread_key and str(meta.get("state", "")) == "R"
            )
            runnable_over_core = max(0, int(runnable_count) - int(self.available_cores))
            active_over_core = max(0, int(active_count) - int(self.available_cores))

            vol_delta, invol_delta = _ctx_switch_delta(self._prev_snapshot or {}, snapshot)

            # Per-thread identity rows: this is what lets us answer
            # "which runnable/active thread caused contention?" rather than only
            # "which step had contention?".
            if TRACE_THREAD_IDENTITIES:
                prev_snapshot = self._prev_snapshot or {}
                for key, meta in snapshot.items():
                    if key == self.sampler_thread_key:
                        continue
                    prev = prev_snapshot.get(key, {})
                    cpu_delta_s_i = max(
                        0.0,
                        float(meta.get("cpu_s", 0.0)) - float(prev.get("cpu_s", 0.0)),
                    )
                    is_runnable = int(str(meta.get("state", "")) == "R")
                    is_active = int(cpu_delta_s_i > ACTIVE_THREAD_CPU_EPS)
                    if not (is_runnable or is_active):
                        continue
                    vol_i = max(
                        0,
                        int(meta.get("voluntary_ctxt_switches", 0))
                        - int(prev.get("voluntary_ctxt_switches", 0)),
                    )
                    invol_i = max(
                        0,
                        int(meta.get("nonvoluntary_ctxt_switches", 0))
                        - int(prev.get("nonvoluntary_ctxt_switches", 0)),
                    )
                    flags = []
                    if is_runnable:
                        flags.append("runnable_R")
                    if is_active:
                        flags.append("cpu_delta_active")
                    if invol_i > 0:
                        flags.append("involuntary_ctx_switch")
                    self.thread_detail_rows.append(
                        {
                            "global_step": global_step,
                            "sample_idx": sample_idx,
                            "t_rel_ms": (time.perf_counter() - t0) * 1000.0,
                            "available_cores": self.available_cores,
                            "sample_runnable_thread_count": runnable_count,
                            "sample_runnable_over_core": runnable_over_core,
                            "sample_active_thread_count": active_count,
                            "sample_active_over_core": active_over_core,
                            "pid": meta.get("pid", key[0]),
                            "tid": meta.get("tid", key[1]),
                            "ppid": meta.get("ppid", ""),
                            "process_name": meta.get("process_name", ""),
                            "process_role": meta.get("process_role", ""),
                            "thread_name": meta.get("thread_name", ""),
                            "kernel_thread_comm": meta.get("kernel_thread_comm", ""),
                            "python_thread_name": meta.get("python_thread_name", ""),
                            "state": meta.get("state", ""),
                            "is_runnable": is_runnable,
                            "is_active_by_cpu_delta": is_active,
                            "cpu_delta_ms_since_prev_sample": cpu_delta_s_i * 1000.0,
                            "voluntary_ctx_switch_delta_since_prev_sample": vol_i,
                            "involuntary_ctx_switch_delta_since_prev_sample": invol_i,
                            "wchan": meta.get("wchan", ""),
                            "last_cpu": meta.get("last_cpu", ""),
                            "process_cmdline": meta.get("process_cmdline", ""),
                            "reason_flags": ";".join(flags),
                        }
                    )

            self.samples.append(
                {
                    "global_step": global_step,
                    "sample_idx": sample_idx,
                    "t_rel_ms": (time.perf_counter() - t0) * 1000.0,
                    "available_cores": self.available_cores,
                    "process_count": process_count,
                    "worker_process_count": max(0, process_count - 1),
                    "total_threads": total_threads,
                    "sample_active_thread_count": active_count,
                    "sample_thread_cpu_delta_s": cpu_delta_s,
                    "sample_runnable_thread_count": runnable_count,
                    "sample_runnable_over_core": runnable_over_core,
                    "sample_active_over_core": active_over_core,
                    "sample_voluntary_ctx_switch_delta": vol_delta,
                    "sample_involuntary_ctx_switch_delta": invol_delta,
                }
            )

            self._prev_snapshot = snapshot
            sample_idx += 1
            time.sleep(self.interval_sec)


# ============================================================
# Prompt specs
# ============================================================

def safe_encode(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text)

    encoded = tokenizer(text, add_special_tokens=False)

    if isinstance(encoded, dict):
        return encoded["input_ids"]

    return encoded.input_ids


def random_word(rng: random.Random, min_len=3, max_len=10):
    n = rng.randint(min_len, max_len)
    return "".join(rng.choices(string.ascii_lowercase, k=n))


def random_text_chunk(rng: random.Random, num_words=64):
    return " ".join(random_word(rng) for _ in range(num_words))


def build_random_exact_token_ids(tokenizer, target_tokens: int, rng: random.Random):
    text = ""
    token_ids = []

    while len(token_ids) < target_tokens:
        text += random_text_chunk(rng, 64) + " "
        token_ids = safe_encode(tokenizer, text)

    token_ids = token_ids[:target_tokens]

    if len(token_ids) != target_tokens:
        raise RuntimeError(
            f"Failed to build exact token ids: got {len(token_ids)}, expected {target_tokens}"
        )

    return token_ids


def maybe_create_prompt_specs(tokenizer):
    if PROMPT_SPECS_PATH.exists():
        return

    rng = random.Random(SEED)
    specs = []

    for pair_id in range(NUM_REQUESTS):
        token_ids = build_random_exact_token_ids(
            tokenizer=tokenizer,
            target_tokens=INPUT_TOKENS,
            rng=rng,
        )

        specs.append(
            {
                "pair_id": pair_id,
                "prompt_token_ids": token_ids,
                "input_tokens": len(token_ids),
            }
        )

    PROMPT_SPECS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PROMPT_SPECS_PATH, "w", encoding="utf-8") as f:
        json.dump(specs, f)

    print(f"[INFO] prompt specs created: {PROMPT_SPECS_PATH}")


def load_prompt_specs():
    with open(PROMPT_SPECS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt_from_spec(spec):
    token_ids = list(spec["prompt_token_ids"])

    if TokensPrompt is not None:
        try:
            return TokensPrompt(prompt_token_ids=token_ids)
        except Exception:
            pass

    return {
        "prompt_token_ids": token_ids,
        "prompt": f"<pair:{spec['pair_id']},tokens:{len(token_ids)}>",
    }


# ============================================================
# General helpers
# ============================================================

def get_output_token_count(request_output) -> int:
    if not getattr(request_output, "outputs", None):
        return 0

    o = request_output.outputs[0]

    if getattr(o, "token_ids", None) is not None:
        return len(o.token_ids)

    return 0


def fmt(v, digits=6):
    if v is None:
        return ""
    return f"{float(v):.{digits}f}"


def to_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def split_ids(s):
    if s is None or s == "":
        return []

    return [x for x in str(s).split("|") if x]


def parse_scheduled_tokens_by_request(s):
    result = {}

    if s is None or s == "":
        return result

    for part in str(s).split("|"):
        if ":" not in part:
            continue

        req_id, n = part.rsplit(":", 1)

        try:
            result[req_id] = int(float(n))
        except Exception:
            result[req_id] = -1

    return result


def compute_latency_ms(
    submit_perf,
    start_processing_perf,
    first_token_perf,
    end_perf,
    output_tokens,
):
    e2e_ms = None
    queue_ms = None
    ttft_ms = None
    tpot_ms = None

    if submit_perf is not None and end_perf is not None:
        e2e_ms = max(0.0, (end_perf - submit_perf) * 1000.0)

    if submit_perf is not None and start_processing_perf is not None:
        queue_ms = max(0.0, (start_processing_perf - submit_perf) * 1000.0)

    if start_processing_perf is not None and first_token_perf is not None:
        ttft_ms = max(0.0, (first_token_perf - start_processing_perf) * 1000.0)

    tpot_token_count = max(0, int(output_tokens) - 1)

    if first_token_perf is not None and end_perf is not None and tpot_token_count > 0:
        tpot_ms = max(
            0.0,
            (end_perf - first_token_perf) * 1000.0 / tpot_token_count,
        )

    return e2e_ms, queue_ms, ttft_ms, tpot_ms, tpot_token_count


def _iter_sort_key(x):
    try:
        return (0, int(x))
    except Exception:
        return (1, str(x))



def _sleep_min_batch_barrier(phase_name: str, actual_request_count: int):
    """Sleep once after all requests are added and before first scheduling.

    This mirrors phase_worker_scheduler_hold_fixed.py.  In the scheduler-hold
    setting the driver-side sleep is normally disabled by leaving
    VLLM_MIN_BATCH_BARRIER_MS=0, but the latency baseline is still reset after
    all add_request() calls when RESET_LATENCY_BASELINE_AFTER_ADD_LOOP=1.
    """
    requested_ms = max(0.0, float(MIN_BATCH_BARRIER_MS))
    target_requests = int(MIN_BATCH_TARGET_REQUESTS)
    enabled = requested_ms > 0.0 and actual_request_count > 1

    baseline_reset_only = (
        RESET_LATENCY_BASELINE_AFTER_ADD_LOOP
        and actual_request_count > 1
        and not enabled
    )

    start_wall = time.time()
    start_perf = time.perf_counter()

    if enabled:
        print(
            f"[INFO] {phase_name}: min-batch barrier start "
            f"requested_ms={requested_ms:.3f}, "
            f"target_requests={target_requests}, "
            f"actual_requests={actual_request_count}, "
            f"exclude_from_latency={int(EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY)}"
        )
        time.sleep(requested_ms / 1000.0)

    end_perf = time.perf_counter()
    end_wall = time.time()
    observed_ms = max(0.0, (end_perf - start_perf) * 1000.0)

    if enabled:
        print(
            f"[INFO] {phase_name}: min-batch barrier done "
            f"observed_ms={observed_ms:.3f}"
        )

    exclude_or_reset = bool(
        EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY
        and (enabled or baseline_reset_only)
    )

    return {
        "min_batch_barrier_enabled": int(enabled),
        "min_batch_barrier_requested_ms": requested_ms if enabled else 0.0,
        "min_batch_barrier_observed_ms": observed_ms if enabled else 0.0,
        "min_batch_barrier_target_requests": target_requests,
        "min_batch_barrier_actual_requests": int(actual_request_count),
        "min_batch_barrier_excluded_from_latency": int(exclude_or_reset),
        "latency_baseline_reset_after_add_loop": int(baseline_reset_only),
        "min_batch_barrier_start_wall": start_wall,
        "min_batch_barrier_end_wall": end_wall,
        "min_batch_barrier_start_perf": start_perf,
        "min_batch_barrier_end_perf": end_perf,
    }


def _apply_latency_submit_baseline_after_barrier(request_state: dict, barrier_info: dict):
    """Reset request submit timestamps to barrier end when configured.

    raw_submit_* keeps the original per-request add time. submit_* is the
    latency baseline used by request-level latency functions.  This preserves
    the scheduler-hold experiment's latency semantics while this worker focuses
    on CPU thread contention.
    """
    exclude = bool(barrier_info.get("min_batch_barrier_excluded_from_latency", 0))

    for state in request_state.values():
        raw_submit_wall = state.get("raw_submit_wall", state.get("submit_wall"))
        raw_submit_perf = state.get("raw_submit_perf", state.get("submit_perf"))
        state["raw_submit_wall"] = raw_submit_wall
        state["raw_submit_perf"] = raw_submit_perf

        for k, v in barrier_info.items():
            state[k] = v

        if exclude:
            state["submit_wall"] = barrier_info["min_batch_barrier_end_wall"]
            state["submit_perf"] = barrier_info["min_batch_barrier_end_perf"]
        else:
            state["submit_wall"] = raw_submit_wall
            state["submit_perf"] = raw_submit_perf

        if raw_submit_perf is not None and state.get("submit_perf") is not None:
            state["raw_to_latency_submit_gap_ms"] = max(
                0.0,
                (state["submit_perf"] - raw_submit_perf) * 1000.0,
            )
        else:
            state["raw_to_latency_submit_gap_ms"] = None


# ============================================================
# vLLM run
# ============================================================

def build_engine():
    kwargs = dict(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    if MAX_MODEL_LEN is not None:
        kwargs["max_model_len"] = MAX_MODEL_LEN

    if USE_ENFORCE_EAGER:
        kwargs["enforce_eager"] = True

    print(f"[INFO] EngineArgs kwargs={kwargs}")

    engine_args = EngineArgs(**kwargs)
    return LLMEngine.from_engine_args(engine_args)


def run_warmup(engine, prompt_specs):
    if NUM_WARMUPS <= 0:
        return

    print(
        f"[INFO] Starting Mixed Batch Warmup "
        f"(iterations: {NUM_WARMUPS}, requests per iter: {NUM_REQUESTS})"
    )

    for warmup_iter in range(NUM_WARMUPS):
        for i in range(NUM_REQUESTS):
            spec_idx = i % len(prompt_specs)
            warmup_spec = prompt_specs[spec_idx]
            warmup_prompt = make_prompt_from_spec(warmup_spec)

            warmup_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=min(WARMUP_OUTPUT_TOKENS, OUTPUT_TOKENS),
            )

            req_id = f"warmup_iter{warmup_iter}_req{i}_{uuid.uuid4()}"
            engine.add_request(req_id, warmup_prompt, warmup_params)

        while engine.has_unfinished_requests():
            engine.step()

    print("[INFO] Mixed Batch Warmup done")


def run_phase(engine, phase_name: str, prompt_specs, measure_threads: bool):
    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=OUTPUT_TOKENS,
    )

    request_state = {}
    active_request_ids = set()
    req_decode_tracker = {}

    for spec in prompt_specs:
        pair_id = int(spec["pair_id"])
        req_id = f"{phase_name}_pair{pair_id}_{uuid.uuid4()}"
        prompt = make_prompt_from_spec(spec)

        submit_wall = time.time()
        submit_perf = time.perf_counter()

        engine.add_request(req_id, prompt, sampling_params)

        request_state[req_id] = {
            "pair_id": pair_id,
            "request_id": req_id,
            "phase": phase_name,
            "raw_submit_wall": submit_wall,
            "raw_submit_perf": submit_perf,
            "submit_wall": submit_wall,
            "submit_perf": submit_perf,
            "start_processing_wall": None,
            "start_processing_perf": None,
            "first_token_wall": None,
            "first_token_perf": None,
            "end_wall": None,
            "end_perf": None,
            "target_input_tokens": INPUT_TOKENS,
            "input_tokens": int(spec["input_tokens"]),
            "input_tokens_ok": int(int(spec["input_tokens"]) == INPUT_TOKENS),
            "target_output_tokens": OUTPUT_TOKENS,
            "output_tokens": 0,
            "output_tokens_ok": 0,
        }

        active_request_ids.add(req_id)
        req_decode_tracker[req_id] = -1

    barrier_info = _sleep_min_batch_barrier(
        phase_name=phase_name,
        actual_request_count=len(request_state),
    )
    _apply_latency_submit_baseline_after_barrier(request_state, barrier_info)

    sampler = StepSampler(root_pid=os.getpid(), interval_sec=SAMPLE_INTERVAL_SEC)

    step_rows = []
    req_detail_rows = []
    sample_rows = []
    thread_detail_rows = []
    global_step = 0

    while engine.has_unfinished_requests():
        if measure_threads:
            before_tree = collect_process_tree(os.getpid())
            before_threads = flatten_thread_cpu(before_tree)
            before_runtime_snapshot = collect_thread_runtime_snapshot(os.getpid())
            before_proc_cpu = tree_total_proc_cpu(before_tree)
            sampler.start(
                global_step,
                initial_thread_cpu=before_threads,
                initial_runtime_snapshot=before_runtime_snapshot,
            )
        else:
            before_tree = {}
            before_threads = {}
            before_runtime_snapshot = {}
            before_proc_cpu = 0.0

        step_start_wall = time.time()
        step_start_perf = time.perf_counter()

        request_outputs = engine.step()

        step_end_perf = time.perf_counter()
        step_end_wall = time.time()

        if measure_threads:
            sampler.stop()

            after_tree = collect_process_tree(os.getpid())
            after_threads = flatten_thread_cpu(after_tree)
            after_runtime_snapshot = collect_thread_runtime_snapshot(os.getpid())
            after_proc_cpu = tree_total_proc_cpu(after_tree)

            exclude_keys = (
                {sampler.sampler_thread_key}
                if sampler.sampler_thread_key is not None
                else set()
            )

            (
                active_thread_count_edge,
                thread_cpu_delta_s_edge,
                edge_active_keys,
            ) = diff_thread_cpu(
                before_threads,
                after_threads,
                exclude_keys=exclude_keys,
            )

            process_cpu_delta_s = max(0.0, after_proc_cpu - before_proc_cpu)

            samples = list(sampler.samples)

            if samples:
                sample_rows.extend(samples)
                thread_detail_rows.extend(list(sampler.thread_detail_rows))

                total_threads_mean = (
                    sum(s["total_threads"] for s in samples) / len(samples)
                )
                total_threads_max = max(s["total_threads"] for s in samples)
                total_threads_min = min(s["total_threads"] for s in samples)
                process_count_max = max(s["process_count"] for s in samples)
                worker_process_count_max = max(
                    s["worker_process_count"] for s in samples
                )
                thread_sample_count = len(samples)

                sample_active_counts = [
                    float(s.get("sample_active_thread_count", 0.0))
                    for s in samples
                ]
                sample_cpu_deltas = [
                    float(s.get("sample_thread_cpu_delta_s", 0.0))
                    for s in samples
                ]
                sample_active_thread_count_mean = _mean(sample_active_counts)
                sample_active_thread_count_max = max(sample_active_counts)
                sample_active_thread_count_p90 = _percentile(sample_active_counts, 0.90)
                sample_thread_cpu_delta_s_sum = sum(sample_cpu_deltas)
                sample_active_thread_count_union = len(
                    sampler.cpu_active_thread_keys_seen
                )

                sample_runnable_counts = [
                    float(s.get("sample_runnable_thread_count", 0.0))
                    for s in samples
                ]
                sample_runnable_over_core_values = [
                    float(s.get("sample_runnable_over_core", 0.0))
                    for s in samples
                ]
                sample_active_over_core_values = [
                    float(s.get("sample_active_over_core", 0.0))
                    for s in samples
                ]
                sample_invol_ctx_switch_values = [
                    float(s.get("sample_involuntary_ctx_switch_delta", 0.0))
                    for s in samples
                ]
                sample_vol_ctx_switch_values = [
                    float(s.get("sample_voluntary_ctx_switch_delta", 0.0))
                    for s in samples
                ]

                step_runnable_thread_count_mean = _mean(sample_runnable_counts)
                step_runnable_thread_count_p90 = _percentile(sample_runnable_counts, 0.90)
                step_runnable_thread_count_max = max(sample_runnable_counts)

                step_runnable_over_core_mean = _mean(sample_runnable_over_core_values)
                step_runnable_over_core_p90 = _percentile(sample_runnable_over_core_values, 0.90)
                step_runnable_over_core_max = max(sample_runnable_over_core_values)

                step_active_over_core_mean = _mean(sample_active_over_core_values)
                step_active_over_core_p90 = _percentile(sample_active_over_core_values, 0.90)
                step_active_over_core_max = max(sample_active_over_core_values)

                step_sample_involuntary_ctx_switch_delta_sum = sum(sample_invol_ctx_switch_values)
                step_sample_voluntary_ctx_switch_delta_sum = sum(sample_vol_ctx_switch_values)
            else:
                total_threads_mean = float(tree_total_threads(after_tree))
                total_threads_max = int(tree_total_threads(after_tree))
                total_threads_min = int(tree_total_threads(after_tree))
                process_count_max = len(after_tree)
                worker_process_count_max = max(0, len(after_tree) - 1)
                thread_sample_count = 0

                sample_active_thread_count_mean = 0.0
                sample_active_thread_count_max = 0.0
                sample_active_thread_count_p90 = 0.0
                sample_thread_cpu_delta_s_sum = 0.0
                sample_active_thread_count_union = 0
                step_runnable_thread_count_mean = 0.0
                step_runnable_thread_count_p90 = 0.0
                step_runnable_thread_count_max = 0.0
                step_runnable_over_core_mean = 0.0
                step_runnable_over_core_p90 = 0.0
                step_runnable_over_core_max = 0.0
                step_active_over_core_mean = 0.0
                step_active_over_core_p90 = 0.0
                step_active_over_core_max = 0.0
                step_sample_involuntary_ctx_switch_delta_sum = 0.0
                step_sample_voluntary_ctx_switch_delta_sum = 0.0

            transient_thread_count_seen = len(
                sampler.thread_keys_seen
                - set(before_threads.keys())
                - set(after_threads.keys())
            )

            active_thread_count_union = len(
                set(edge_active_keys) | set(sampler.cpu_active_thread_keys_seen)
            )

            # Backward-compatible 대표 active thread count.
            # 기존 before/after edge 방식과 sample p90/union 방식을 함께 보고,
            # 너무 짧은 step에서 한쪽만 0으로 떨어지는 문제를 완화한다.
            active_thread_count = max(
                float(active_thread_count_edge),
                float(sample_active_thread_count_p90),
            )
            thread_cpu_delta_s = max(
                float(thread_cpu_delta_s_edge),
                float(sample_thread_cpu_delta_s_sum),
            )
            step_effective_cpu_cores = _effective_cpu_cores(
                process_cpu_delta_s,
                (step_end_perf - step_start_perf) * 1000.0,
            )
            available_cores = get_available_core_count()
            step_cpu_utilization_ratio = (
                float(step_effective_cpu_cores) / float(available_cores)
                if available_cores > 0
                else 0.0
            )
            step_contention_flag = int(
                float(step_runnable_over_core_p90) > 0.0
                or float(step_active_over_core_p90) > 0.0
                or float(step_cpu_utilization_ratio) >= CONTENTION_CPU_UTIL_THRESHOLD
            )
            step_voluntary_ctx_switch_delta, step_involuntary_ctx_switch_delta = _ctx_switch_delta(
                before_runtime_snapshot,
                after_runtime_snapshot,
            )
            top_threads = _top_cpu_threads(
                before_runtime_snapshot,
                after_runtime_snapshot,
                top_n=CONTENTIOUS_TOP_N_THREADS,
                exclude_keys=exclude_keys,
            )
            step_top_cpu_threads = _format_top_threads(top_threads)
        else:
            after_tree = {}
            active_thread_count = ""
            thread_cpu_delta_s = ""
            process_cpu_delta_s = ""
            total_threads_mean = ""
            total_threads_max = ""
            total_threads_min = ""
            process_count_max = ""
            worker_process_count_max = ""
            transient_thread_count_seen = ""
            thread_sample_count = ""
            active_thread_count_edge = ""
            thread_cpu_delta_s_edge = ""
            sample_active_thread_count_mean = ""
            sample_active_thread_count_max = ""
            sample_active_thread_count_p90 = ""
            sample_active_thread_count_union = ""
            sample_thread_cpu_delta_s_sum = ""
            active_thread_count_union = ""
            step_effective_cpu_cores = ""
            available_cores = ""
            step_cpu_utilization_ratio = ""
            step_contention_flag = ""
            step_runnable_thread_count_mean = ""
            step_runnable_thread_count_p90 = ""
            step_runnable_thread_count_max = ""
            step_runnable_over_core_mean = ""
            step_runnable_over_core_p90 = ""
            step_runnable_over_core_max = ""
            step_active_over_core_mean = ""
            step_active_over_core_p90 = ""
            step_active_over_core_max = ""
            step_voluntary_ctx_switch_delta = ""
            step_involuntary_ctx_switch_delta = ""
            step_sample_involuntary_ctx_switch_delta_sum = ""
            step_sample_voluntary_ctx_switch_delta_sum = ""
            step_top_cpu_threads = ""

        step_wall_ms = (step_end_perf - step_start_perf) * 1000.0

        returned_ids = []
        finished_ids_now = []
        unfinished_ids_now = []

        for out in request_outputs:
            req_id = out.request_id
            returned_ids.append(req_id)
            out_tok = get_output_token_count(out)

            if req_id in request_state:
                if request_state[req_id]["start_processing_perf"] is None:
                    request_state[req_id]["start_processing_wall"] = step_start_wall
                    request_state[req_id]["start_processing_perf"] = step_start_perf

                request_state[req_id]["output_tokens"] = out_tok

                if request_state[req_id]["first_token_perf"] is None and out_tok > 0:
                    request_state[req_id]["first_token_wall"] = step_end_wall
                    request_state[req_id]["first_token_perf"] = step_end_perf

                if getattr(out, "finished", False):
                    request_state[req_id]["end_wall"] = step_end_wall
                    request_state[req_id]["end_perf"] = step_end_perf
                    request_state[req_id]["output_tokens_ok"] = int(
                        out_tok == OUTPUT_TOKENS
                    )
                    active_request_ids.discard(req_id)
                    finished_ids_now.append(req_id)
                else:
                    unfinished_ids_now.append(req_id)

        step_rows.append(
            {
                "phase": phase_name,
                "global_step": global_step,
                "min_batch_barrier_enabled": barrier_info.get("min_batch_barrier_enabled", 0),
                "min_batch_barrier_requested_ms": f"{float(barrier_info.get('min_batch_barrier_requested_ms', 0.0)):.3f}",
                "min_batch_barrier_observed_ms": f"{float(barrier_info.get('min_batch_barrier_observed_ms', 0.0)):.3f}",
                "min_batch_barrier_target_requests": barrier_info.get("min_batch_barrier_target_requests", ""),
                "min_batch_barrier_actual_requests": barrier_info.get("min_batch_barrier_actual_requests", ""),
                "min_batch_barrier_excluded_from_latency": barrier_info.get("min_batch_barrier_excluded_from_latency", 0),
                "latency_baseline_reset_after_add_loop": barrier_info.get("latency_baseline_reset_after_add_loop", 0),
                "step_start_wall": f"{step_start_wall:.9f}",
                "step_end_wall": f"{step_end_wall:.9f}",
                "step_wall_ms": f"{step_wall_ms:.3f}",
                "returned_request_ids": "|".join(returned_ids),
                "finished_request_ids_now": "|".join(finished_ids_now),
                "unfinished_request_ids_now": "|".join(unfinished_ids_now),
                "active_request_ids_after_step": "|".join(sorted(active_request_ids)),
                "returned_request_count": len(returned_ids),
                "active_request_count_after_step": len(active_request_ids),
                "process_count_after_step": len(after_tree),
                "process_count_max_in_step": process_count_max,
                "worker_process_count_max_in_step": worker_process_count_max,
                "total_threads_min_in_step": total_threads_min,
                "total_threads_mean_in_step": total_threads_mean,
                "total_threads_max_in_step": total_threads_max,
                "step_active_thread_count": active_thread_count,
                "step_active_thread_count_edge": active_thread_count_edge,
                "step_active_thread_count_sample_mean": sample_active_thread_count_mean,
                "step_active_thread_count_sample_p90": sample_active_thread_count_p90,
                "step_active_thread_count_sample_max": sample_active_thread_count_max,
                "step_active_thread_count_sample_union": sample_active_thread_count_union,
                "step_active_thread_count_union": active_thread_count_union,
                "step_thread_cpu_delta_s": thread_cpu_delta_s,
                "step_thread_cpu_delta_s_edge": thread_cpu_delta_s_edge,
                "step_sample_thread_cpu_delta_s_sum": sample_thread_cpu_delta_s_sum,
                "process_cpu_delta_s": process_cpu_delta_s,
                "step_effective_cpu_cores": step_effective_cpu_cores,
                "available_cores": available_cores,
                "step_cpu_utilization_ratio": step_cpu_utilization_ratio,
                "step_contention_flag": step_contention_flag,
                "step_runnable_thread_count_mean": step_runnable_thread_count_mean,
                "step_runnable_thread_count_p90": step_runnable_thread_count_p90,
                "step_runnable_thread_count_max": step_runnable_thread_count_max,
                "step_runnable_over_core_mean": step_runnable_over_core_mean,
                "step_runnable_over_core_p90": step_runnable_over_core_p90,
                "step_runnable_over_core_max": step_runnable_over_core_max,
                "step_active_over_core_mean": step_active_over_core_mean,
                "step_active_over_core_p90": step_active_over_core_p90,
                "step_active_over_core_max": step_active_over_core_max,
                "step_voluntary_ctx_switch_delta": step_voluntary_ctx_switch_delta,
                "step_involuntary_ctx_switch_delta": step_involuntary_ctx_switch_delta,
                "step_sample_voluntary_ctx_switch_delta_sum": step_sample_voluntary_ctx_switch_delta_sum,
                "step_sample_involuntary_ctx_switch_delta_sum": step_sample_involuntary_ctx_switch_delta_sum,
                "step_top_cpu_threads": step_top_cpu_threads,
                "transient_thread_count_seen": transient_thread_count_seen,
                "thread_sample_count": thread_sample_count,
            }
        )

        for req_id in returned_ids:
            if req_id not in req_decode_tracker:
                continue

            if req_decode_tracker[req_id] < 0:
                request_stage = "Prefill"
                request_decode_iter_idx = -1
                req_decode_tracker[req_id] = 0
            else:
                request_stage = "Decode"
                request_decode_iter_idx = req_decode_tracker[req_id]
                req_decode_tracker[req_id] += 1

            pair_id = request_state[req_id]["pair_id"]

            req_detail_rows.append(
                {
                    "phase": phase_name,
                    "pair_id": pair_id,
                    "request_id": req_id,
                    "request_stage": request_stage,
                    "request_decode_iter_idx": request_decode_iter_idx,
                    "global_step": global_step,
                    "shared_min_batch_barrier_enabled": barrier_info.get("min_batch_barrier_enabled", 0),
                    "shared_min_batch_barrier_observed_ms": barrier_info.get("min_batch_barrier_observed_ms", 0.0),
                    "shared_min_batch_barrier_excluded_from_latency": barrier_info.get("min_batch_barrier_excluded_from_latency", 0),
                    "shared_latency_baseline_reset_after_add_loop": barrier_info.get("latency_baseline_reset_after_add_loop", 0),
                    "shared_returned_request_ids": "|".join(returned_ids),
                    "shared_finished_request_ids_now": "|".join(finished_ids_now),
                    "shared_active_request_ids_after_step": "|".join(
                        sorted(active_request_ids)
                    ),
                    "shared_returned_request_count": len(returned_ids),
                    "shared_active_request_count_after_step": len(active_request_ids),
                    "shared_step_wall_ms": step_wall_ms,
                    "shared_step_active_thread_count": active_thread_count,
                    "shared_step_active_thread_count_edge": active_thread_count_edge,
                    "shared_step_active_thread_count_sample_mean": sample_active_thread_count_mean,
                    "shared_step_active_thread_count_sample_p90": sample_active_thread_count_p90,
                    "shared_step_active_thread_count_sample_max": sample_active_thread_count_max,
                    "shared_step_active_thread_count_sample_union": sample_active_thread_count_union,
                    "shared_step_active_thread_count_union": active_thread_count_union,
                    "shared_step_thread_cpu_delta_s": thread_cpu_delta_s,
                    "shared_step_thread_cpu_delta_s_edge": thread_cpu_delta_s_edge,
                    "shared_step_sample_thread_cpu_delta_s_sum": sample_thread_cpu_delta_s_sum,
                    "shared_process_cpu_delta_s": process_cpu_delta_s,
                    "shared_step_effective_cpu_cores": step_effective_cpu_cores,
                    "shared_available_cores": available_cores,
                    "shared_step_cpu_utilization_ratio": step_cpu_utilization_ratio,
                    "shared_step_contention_flag": step_contention_flag,
                    "shared_step_runnable_thread_count_mean": step_runnable_thread_count_mean,
                    "shared_step_runnable_thread_count_p90": step_runnable_thread_count_p90,
                    "shared_step_runnable_thread_count_max": step_runnable_thread_count_max,
                    "shared_step_runnable_over_core_mean": step_runnable_over_core_mean,
                    "shared_step_runnable_over_core_p90": step_runnable_over_core_p90,
                    "shared_step_runnable_over_core_max": step_runnable_over_core_max,
                    "shared_step_active_over_core_mean": step_active_over_core_mean,
                    "shared_step_active_over_core_p90": step_active_over_core_p90,
                    "shared_step_active_over_core_max": step_active_over_core_max,
                    "shared_step_voluntary_ctx_switch_delta": step_voluntary_ctx_switch_delta,
                    "shared_step_involuntary_ctx_switch_delta": step_involuntary_ctx_switch_delta,
                    "shared_step_top_cpu_threads": step_top_cpu_threads,
                    "shared_step_total_threads_mean": total_threads_mean,
                    "shared_step_total_threads_max": total_threads_max,
                }
            )

        if global_step % 50 == 0:
            print(
                f"[{phase_name}] step={global_step}, "
                f"returned={len(returned_ids)}, active_thread={active_thread_count}"
            )

        global_step += 1

    return request_state, step_rows, req_detail_rows, sample_rows, thread_detail_rows


# ============================================================
# GPU iteration parsing
# ============================================================

def read_csv_dicts(path: Path):
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rank_key(row):
    rank = str(row.get("rank", ""))
    pid = str(row.get("process_pid", ""))

    if rank != "":
        return f"rank:{rank}"

    if pid != "":
        return f"pid:{pid}"

    return "unknown"


def _make_logical_iter_row(req_id, logical_idx, occs):
    rank_keys = sorted(set(o["rank_key"] for o in occs))
    iter_idx_list = sorted(set(o["raw_iter_idx"] for o in occs), key=_iter_sort_key)
    sources = sorted(set(o["request_id_source"] for o in occs))

    scheduled_candidates = [
        o["scheduled_tokens_for_req"]
        for o in occs
        if o["scheduled_tokens_for_req"] is not None
    ]

    scheduled_tokens_for_req = (
        max(scheduled_candidates) if scheduled_candidates else None
    )

    start_values = [o["start_wall"] for o in occs]
    end_values = [o["end_wall"] for o in occs]

    worker_rows = len(occs)
    distinct_rank_keys = len(rank_keys)

    worker_rows_ok = int(
        worker_rows == EXPECTED_TP_ROWS
        and distinct_rank_keys == EXPECTED_TP_ROWS
    )

    iter_wall_ms_max = max(float(o["iter_wall_ms"]) for o in occs)
    iter_gpu_ms_max = max(float(o["iter_gpu_ms"]) for o in occs)

    # 핵심:
    # non-GPU는 wall-max - gpu-max로 재계산하지 않고,
    # sitecustomize.py가 같은 rank 안에서 기록한 iter_non_gpu_wall_ms의 rank별 max를 사용한다.
    # 이렇게 해야 서로 다른 rank의 wall max와 gpu max가 섞이는 문제를 줄일 수 있다.
    iter_non_gpu_wall_ms_max = max(
        float(
            o.get(
                "iter_non_gpu_wall_ms",
                max(0.0, float(o["iter_wall_ms"]) - float(o["iter_gpu_ms"])),
            )
        )
        for o in occs
    )

    return {
        "request_id": req_id,
        "logical_idx": logical_idx,
        "scheduled_tokens_for_req": scheduled_tokens_for_req,
        "iter_gpu_ms_max": iter_gpu_ms_max,
        "iter_wall_ms_max": iter_wall_ms_max,
        "iter_non_gpu_wall_ms_max": iter_non_gpu_wall_ms_max,
        "worker_rows": worker_rows,
        "worker_rows_ok": worker_rows_ok,
        "rank_key_list": "|".join(rank_keys),
        "iter_idx_list": "|".join(iter_idx_list),
        "request_id_sources": "|".join(sources),
        "rank_start_skew_ms": (max(start_values) - min(start_values)) * 1000.0,
        "rank_end_skew_ms": (max(end_values) - min(end_values)) * 1000.0,
    }


def build_logical_iters_by_request():
    iter_rows = read_csv_dicts(ITER_CSV)
    occurrences_by_request = defaultdict(list)

    for row in iter_rows:
        source = str(row.get("request_id_source", ""))

        if source == "finished_req_ids":
            continue

        token_map = parse_scheduled_tokens_by_request(
            row.get("scheduled_tokens_by_request", "")
        )

        if token_map:
            req_ids = list(token_map.keys())
        else:
            req_ids = split_ids(row.get("request_ids", ""))

        for req_id in req_ids:
            if req_id.startswith("warmup_"):
                continue

            start = to_float(row.get("iter_start_wall"), default=None)
            end = to_float(row.get("iter_end_wall"), default=None)

            if start is None or end is None:
                continue

            occ = {
                "request_id": req_id,
                "raw_iter_idx": str(row.get("iter_idx", "")),
                "start_wall": start,
                "end_wall": end,
                "rank": str(row.get("rank", "")),
                "rank_key": rank_key(row),
                "request_id_source": source,
                "scheduled_tokens_for_req": token_map.get(req_id, None),
                "iter_wall_ms": to_float(row.get("iter_wall_ms"), default=0.0),
                "iter_gpu_ms": to_float(row.get("iter_gpu_ms"), default=0.0),
                "iter_non_gpu_wall_ms": to_float(
                    row.get("iter_non_gpu_wall_ms"),
                    default=max(
                        0.0,
                        to_float(row.get("iter_wall_ms"), default=0.0)
                        - to_float(row.get("iter_gpu_ms"), default=0.0),
                    ),
                ),
            }

            occurrences_by_request[req_id].append(occ)

    by_request = defaultdict(list)

    for req_id, occurrences in occurrences_by_request.items():
        buckets = defaultdict(list)

        for occ in occurrences:
            buckets[occ["raw_iter_idx"]].append(occ)

        logical_groups = []

        for raw_iter_idx in sorted(buckets.keys(), key=_iter_sort_key):
            bucket = sorted(
                buckets[raw_iter_idx],
                key=lambda x: (x["start_wall"], x["rank_key"]),
            )

            cur = []
            seen_ranks = set()

            for occ in bucket:
                if occ["rank_key"] in seen_ranks or len(cur) >= EXPECTED_TP_ROWS:
                    if cur:
                        logical_groups.append(cur)

                    cur = []
                    seen_ranks = set()

                cur.append(occ)
                seen_ranks.add(occ["rank_key"])

            if cur:
                logical_groups.append(cur)

        logical_groups.sort(
            key=lambda group: (
                _iter_sort_key(group[0]["raw_iter_idx"]),
                min(o["start_wall"] for o in group),
            )
        )

        for logical_idx, occs in enumerate(logical_groups):
            by_request[req_id].append(
                _make_logical_iter_row(req_id, logical_idx, occs)
            )

    return by_request


def split_prefill_decode_iters(logical_iters):
    if not logical_iters:
        return [], [], "none"

    has_token_info = any(
        it.get("scheduled_tokens_for_req") is not None
        and it.get("scheduled_tokens_for_req") >= 0
        for it in logical_iters
    )

    if has_token_info:
        prefill = []
        decode = []

        for it in logical_iters:
            n = it.get("scheduled_tokens_for_req")

            if n is None or n < 0:
                decode.append(it)
            elif n > 1:
                prefill.append(it)
            else:
                decode.append(it)

        if not prefill:
            return (
                logical_iters[:1],
                logical_iters[1:],
                "fallback_first_iter_no_prefill_detected",
            )

        return prefill, decode, "scheduled_tokens"

    return logical_iters[:1], logical_iters[1:], "fallback_first_iter"


# ============================================================
# Summaries
# ============================================================

def build_phase_summary(request_state, df_detail: pd.DataFrame, include_gpu: bool):
    logical_by_req = build_logical_iters_by_request() if include_gpu else {}
    rows = []

    for req_id, state in request_state.items():
        pair_id = state["pair_id"]
        output_tokens = int(state.get("output_tokens", 0))

        (
            e2e_ms,
            queue_ms,
            request_observed_ttft_ms,
            request_observed_tpot_ms,
            tpot_token_count,
        ) = compute_latency_ms(
            submit_perf=state.get("submit_perf"),
            start_processing_perf=state.get("start_processing_perf"),
            first_token_perf=state.get("first_token_perf"),
            end_perf=state.get("end_perf"),
            output_tokens=output_tokens,
        )

        grp = (
            df_detail[df_detail["request_id"] == req_id]
            if not df_detail.empty
            else pd.DataFrame()
        )

        prefill_detail = (
            grp[grp["request_stage"] == "Prefill"]
            if not grp.empty
            else pd.DataFrame()
        )

        decode_detail = (
            grp[grp["request_stage"] == "Decode"]
            if not grp.empty
            else pd.DataFrame()
        )

        row = {
            "pair_id": pair_id,
            "request_id": req_id,
            "phase": state["phase"],
            "submit_wall": fmt(state.get("submit_wall"), 9),
            "start_processing_wall": fmt(state.get("start_processing_wall"), 9),
            "first_token_wall": fmt(state.get("first_token_wall"), 9),
            "end_wall": fmt(state.get("end_wall"), 9),

            "target_input_tokens": state.get("target_input_tokens", ""),
            "input_tokens": state.get("input_tokens", ""),
            "input_tokens_ok": state.get("input_tokens_ok", ""),

            "target_output_tokens": state.get("target_output_tokens", ""),
            "output_tokens": output_tokens,
            "output_tokens_ok": state.get("output_tokens_ok", ""),

            "e2e_ms": fmt(e2e_ms, 3),
            "queue_ms": fmt(queue_ms, 3),

            # active phase에서는 request-observed 값을 사용.
            # gpu phase에서는 아래 include_gpu 블록에서 iteration-derived 값으로 덮어쓴다.
            "ttft_ms": fmt(request_observed_ttft_ms, 3),
            "tpot_ms": fmt(request_observed_tpot_ms, 6),
            "tpot_token_count": tpot_token_count,
        }

        if not prefill_detail.empty:
            row.update(
                {
                    "prefill_active_thread_count": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail["shared_step_active_thread_count"],
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_active_thread_count_edge": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail.get(
                                    "shared_step_active_thread_count_edge",
                                    prefill_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_active_thread_count_sample_p90": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail.get(
                                    "shared_step_active_thread_count_sample_p90",
                                    prefill_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_active_thread_count_union": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail.get(
                                    "shared_step_active_thread_count_union",
                                    prefill_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_effective_cpu_cores": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail.get(
                                    "shared_step_effective_cpu_cores",
                                    pd.Series([], dtype=float),
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_available_cores": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_available_cores", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "prefill_cpu_utilization_ratio": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_step_cpu_utilization_ratio", pd.Series([], dtype=float)), errors="coerce").mean()),
                        6,
                    ),
                    "prefill_contention_flag": int(
                        pd.to_numeric(prefill_detail.get("shared_step_contention_flag", pd.Series([], dtype=float)), errors="coerce").fillna(0).max()
                    ),
                    "prefill_runnable_thread_count_p90": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_step_runnable_thread_count_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "prefill_runnable_over_core_p90": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_step_runnable_over_core_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "prefill_active_over_core_p90": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_step_active_over_core_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "prefill_involuntary_ctx_switch_delta": fmt(
                        float(pd.to_numeric(prefill_detail.get("shared_step_involuntary_ctx_switch_delta", pd.Series([], dtype=float)), errors="coerce").sum()),
                        3,
                    ),
                    "prefill_top_cpu_threads": str(
                        prefill_detail.get("shared_step_top_cpu_threads", pd.Series([""])).iloc[0]
                    ),
                    "prefill_total_threads_mean": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail["shared_step_total_threads_mean"],
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "prefill_process_cpu_delta_s": fmt(
                        float(
                            pd.to_numeric(
                                prefill_detail["shared_process_cpu_delta_s"],
                                errors="coerce",
                            ).mean()
                        ),
                        9,
                    ),
                }
            )
        else:
            row.update(
                {
                    "prefill_active_thread_count": "",
                    "prefill_active_thread_count_edge": "",
                    "prefill_active_thread_count_sample_p90": "",
                    "prefill_active_thread_count_union": "",
                    "prefill_effective_cpu_cores": "",
                    "prefill_available_cores": "",
                    "prefill_cpu_utilization_ratio": "",
                    "prefill_contention_flag": "",
                    "prefill_runnable_thread_count_p90": "",
                    "prefill_runnable_over_core_p90": "",
                    "prefill_active_over_core_p90": "",
                    "prefill_involuntary_ctx_switch_delta": "",
                    "prefill_top_cpu_threads": "",
                    "prefill_total_threads_mean": "",
                    "prefill_process_cpu_delta_s": "",
                }
            )

        if not decode_detail.empty:
            row.update(
                {
                    "decode_active_thread_count_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail["shared_step_active_thread_count"],
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_active_thread_count_edge_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_active_thread_count_edge",
                                    decode_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_active_thread_count_sample_p90_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_active_thread_count_sample_p90",
                                    decode_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_active_thread_count_sample_max_peak": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_active_thread_count_sample_max",
                                    decode_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).max()
                        ),
                        3,
                    ),
                    "decode_active_thread_count_union_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_active_thread_count_union",
                                    decode_detail["shared_step_active_thread_count"],
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_effective_cpu_cores_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_effective_cpu_cores",
                                    pd.Series([], dtype=float),
                                ),
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_effective_cpu_cores_p90": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_effective_cpu_cores",
                                    pd.Series([], dtype=float),
                                ),
                                errors="coerce",
                            ).quantile(0.90)
                        ),
                        3,
                    ),
                    "decode_effective_cpu_cores_max": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail.get(
                                    "shared_step_effective_cpu_cores",
                                    pd.Series([], dtype=float),
                                ),
                                errors="coerce",
                            ).max()
                        ),
                        3,
                    ),
                    "decode_available_cores": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_available_cores", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "decode_cpu_utilization_ratio_avg": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_cpu_utilization_ratio", pd.Series([], dtype=float)), errors="coerce").mean()),
                        6,
                    ),
                    "decode_cpu_utilization_ratio_p90": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_cpu_utilization_ratio", pd.Series([], dtype=float)), errors="coerce").quantile(0.90)),
                        6,
                    ),
                    "decode_contention_step_count": int(
                        pd.to_numeric(decode_detail.get("shared_step_contention_flag", pd.Series([], dtype=float)), errors="coerce").fillna(0).sum()
                    ),
                    "decode_contention_step_ratio": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_contention_flag", pd.Series([], dtype=float)), errors="coerce").fillna(0).mean()),
                        6,
                    ),
                    "decode_runnable_thread_count_p90_avg": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_runnable_thread_count_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "decode_runnable_thread_count_max_peak": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_runnable_thread_count_max", pd.Series([], dtype=float)), errors="coerce").max()),
                        3,
                    ),
                    "decode_runnable_over_core_p90_avg": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_runnable_over_core_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "decode_runnable_over_core_max_peak": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_runnable_over_core_max", pd.Series([], dtype=float)), errors="coerce").max()),
                        3,
                    ),
                    "decode_active_over_core_p90_avg": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_active_over_core_p90", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "decode_involuntary_ctx_switch_delta_total": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_involuntary_ctx_switch_delta", pd.Series([], dtype=float)), errors="coerce").sum()),
                        3,
                    ),
                    "decode_involuntary_ctx_switch_delta_avg": fmt(
                        float(pd.to_numeric(decode_detail.get("shared_step_involuntary_ctx_switch_delta", pd.Series([], dtype=float)), errors="coerce").mean()),
                        3,
                    ),
                    "decode_top_cpu_threads_first_contention": str(
                        (
                            decode_detail.get("shared_step_top_cpu_threads", pd.Series([""]))
                            .replace("", pd.NA)
                            .dropna()
                            .head(1)
                            .iloc[0]
                        )
                        if (
                            len(decode_detail) > 0
                            and "shared_step_top_cpu_threads" in decode_detail.columns
                            and not decode_detail["shared_step_top_cpu_threads"].replace("", pd.NA).dropna().empty
                        )
                        else ""
                    ),
                    "decode_total_threads_mean_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail["shared_step_total_threads_mean"],
                                errors="coerce",
                            ).mean()
                        ),
                        3,
                    ),
                    "decode_total_threads_max_peak": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail["shared_step_total_threads_max"],
                                errors="coerce",
                            ).max()
                        ),
                        3,
                    ),
                    "decode_process_cpu_delta_s_avg": fmt(
                        float(
                            pd.to_numeric(
                                decode_detail["shared_process_cpu_delta_s"],
                                errors="coerce",
                            ).mean()
                        ),
                        9,
                    ),
                    "num_decode_steps": int(len(decode_detail)),
                    "decode_steps_expected": max(0, output_tokens - 1),
                    "decode_steps_match": int(
                        len(decode_detail) == max(0, output_tokens - 1)
                    ),
                }
            )
        else:
            row.update(
                {
                    "decode_active_thread_count_avg": "",
                    "decode_active_thread_count_edge_avg": "",
                    "decode_active_thread_count_sample_p90_avg": "",
                    "decode_active_thread_count_sample_max_peak": "",
                    "decode_active_thread_count_union_avg": "",
                    "decode_effective_cpu_cores_avg": "",
                    "decode_effective_cpu_cores_p90": "",
                    "decode_effective_cpu_cores_max": "",
                    "decode_available_cores": "",
                    "decode_cpu_utilization_ratio_avg": "",
                    "decode_cpu_utilization_ratio_p90": "",
                    "decode_contention_step_count": "",
                    "decode_contention_step_ratio": "",
                    "decode_runnable_thread_count_p90_avg": "",
                    "decode_runnable_thread_count_max_peak": "",
                    "decode_runnable_over_core_p90_avg": "",
                    "decode_runnable_over_core_max_peak": "",
                    "decode_active_over_core_p90_avg": "",
                    "decode_involuntary_ctx_switch_delta_total": "",
                    "decode_involuntary_ctx_switch_delta_avg": "",
                    "decode_top_cpu_threads_first_contention": "",
                    "decode_total_threads_mean_avg": "",
                    "decode_total_threads_max_peak": "",
                    "decode_process_cpu_delta_s_avg": "",
                    "num_decode_steps": 0,
                    "decode_steps_expected": max(0, output_tokens - 1),
                    "decode_steps_match": 0,
                }
            )

        if include_gpu:
            logical_iters = logical_by_req.get(req_id, [])
            gpu_prefill, gpu_decode, method = split_prefill_decode_iters(logical_iters)

            e2ft_request_observed_ms = None

            if (
                state.get("submit_perf") is not None
                and state.get("first_token_perf") is not None
            ):
                e2ft_request_observed_ms = max(
                    0.0,
                    (state["first_token_perf"] - state["submit_perf"]) * 1000.0,
                )

            # ============================================================
            # TTFT: prefill logical iteration 기준
            # ============================================================

            ttft_prefill_wall_ms = (
                sum(float(it["iter_wall_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            ttft_gpu_ms = (
                sum(float(it["iter_gpu_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            ttft_prefill_non_gpu_wall_ms = (
                sum(float(it["iter_non_gpu_wall_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            ttft_gpu_exceeds_prefill_wall = 0

            if ttft_prefill_wall_ms is not None and ttft_gpu_ms is not None:
                ttft_gpu_exceeds_prefill_wall = int(
                    ttft_gpu_ms > ttft_prefill_wall_ms + 1e-3
                )

            ttft_ms_for_report = ttft_prefill_wall_ms
            ttft_non_gpu = ttft_prefill_non_gpu_wall_ms

            # ============================================================
            # TPOT: decode logical iteration 기준
            # ============================================================

            tpot_request_observed_ms = request_observed_tpot_ms

            decode_wall_total_ms = (
                sum(float(it["iter_wall_ms_max"]) for it in gpu_decode)
                if gpu_decode
                else None
            )

            decode_gpu_total_ms = (
                sum(float(it["iter_gpu_ms_max"]) for it in gpu_decode)
                if gpu_decode
                else None
            )

            decode_non_gpu_wall_total_ms = (
                sum(float(it["iter_non_gpu_wall_ms_max"]) for it in gpu_decode)
                if gpu_decode
                else None
            )

            tpot_decode_wall_ms = None
            tpot_gpu_ms = None
            tpot_non_gpu = None

            if tpot_token_count > 0 and decode_wall_total_ms is not None:
                tpot_decode_wall_ms = decode_wall_total_ms / tpot_token_count

            if tpot_token_count > 0 and decode_gpu_total_ms is not None:
                tpot_gpu_ms = decode_gpu_total_ms / tpot_token_count

            if tpot_token_count > 0 and decode_non_gpu_wall_total_ms is not None:
                tpot_non_gpu = decode_non_gpu_wall_total_ms / tpot_token_count

            tpot_gpu_exceeds_decode_wall = 0

            if decode_wall_total_ms is not None and decode_gpu_total_ms is not None:
                tpot_gpu_exceeds_decode_wall = int(
                    decode_gpu_total_ms > decode_wall_total_ms + 1e-3
                )

            tpot_ms_for_report = tpot_decode_wall_ms

            # ============================================================
            # E2E: 기존 request-level e2e와 GPU total을 함께 둔다.
            # 추가로 iteration-derived e2e도 diagnostic으로 기록한다.
            # ============================================================

            e2e_gpu_ms = (
                sum(float(it["iter_gpu_ms_max"]) for it in logical_iters)
                if logical_iters
                else None
            )

            e2e_iteration_wall_ms = (
                sum(float(it["iter_wall_ms_max"]) for it in logical_iters)
                if logical_iters
                else None
            )

            e2e_iteration_non_gpu_wall_ms = (
                sum(float(it["iter_non_gpu_wall_ms_max"]) for it in logical_iters)
                if logical_iters
                else None
            )

            e2e_non_gpu = (
                None
                if e2e_ms is None or e2e_gpu_ms is None
                else max(0.0, e2e_ms - e2e_gpu_ms)
            )

            e2e_ratio = (
                None
                if e2e_ms is None or not e2e_ms or e2e_non_gpu is None
                else e2e_non_gpu / e2e_ms
            )

            ttft_ratio = (
                None
                if (
                    ttft_ms_for_report is None
                    or not ttft_ms_for_report
                    or ttft_non_gpu is None
                )
                else ttft_non_gpu / ttft_ms_for_report
            )

            tpot_ratio = (
                None
                if (
                    tpot_ms_for_report is None
                    or not tpot_ms_for_report
                    or tpot_non_gpu is None
                )
                else tpot_non_gpu / tpot_ms_for_report
            )

            worker_rows_values = [it["worker_rows"] for it in logical_iters]
            worker_ok_values = [it["worker_rows_ok"] for it in logical_iters]
            rank_start_skews = [it["rank_start_skew_ms"] for it in logical_iters]
            rank_end_skews = [it["rank_end_skew_ms"] for it in logical_iters]

            row.update(
                {
                    # request-level diagnostic
                    "ttft_request_observed_ms": fmt(request_observed_ttft_ms, 3),
                    "e2ft_request_observed_ms": fmt(e2ft_request_observed_ms, 3),
                    "tpot_request_observed_ms": fmt(tpot_request_observed_ms, 6),

                    # TTFT iteration-derived
                    "ttft_prefill_wall_ms": fmt(ttft_prefill_wall_ms, 3),
                    "ttft_prefill_non_gpu_wall_ms": fmt(
                        ttft_prefill_non_gpu_wall_ms,
                        3,
                    ),
                    "ttft_gpu_exceeds_prefill_wall": ttft_gpu_exceeds_prefill_wall,

                    # 기존 ttft_ms 컬럼을 phase B에서는 iteration-derived 값으로 덮어씀
                    "ttft_ms": fmt(ttft_ms_for_report, 3),

                    # TPOT iteration-derived
                    "tpot_decode_wall_total_ms": fmt(decode_wall_total_ms, 3),
                    "tpot_decode_gpu_total_ms": fmt(decode_gpu_total_ms, 3),
                    "tpot_decode_non_gpu_wall_total_ms": fmt(
                        decode_non_gpu_wall_total_ms,
                        3,
                    ),
                    "tpot_gpu_exceeds_decode_wall": tpot_gpu_exceeds_decode_wall,

                    # 기존 tpot_ms 컬럼을 phase B에서는 iteration-derived 값으로 덮어씀
                    "tpot_ms": fmt(tpot_ms_for_report, 6),

                    # E2E diagnostic
                    "e2e_iteration_wall_ms": fmt(e2e_iteration_wall_ms, 3),
                    "e2e_iteration_non_gpu_wall_ms": fmt(
                        e2e_iteration_non_gpu_wall_ms,
                        3,
                    ),

                    # iteration count / grouping diagnostic
                    "gpu_num_logical_iterations": len(logical_iters),
                    "gpu_num_prefill_iters": len(gpu_prefill),
                    "gpu_num_decode_iters": len(gpu_decode),
                    "gpu_decode_iters_expected": max(0, output_tokens - 1),
                    "gpu_decode_iters_match": int(
                        len(gpu_decode) == max(0, output_tokens - 1)
                    ),
                    "gpu_stage_classification_method": method,
                    "all_iters_have_expected_tp_rows": (
                        int(all(worker_ok_values)) if worker_ok_values else 0
                    ),
                    "min_worker_rows_per_iter": (
                        min(worker_rows_values) if worker_rows_values else ""
                    ),
                    "max_worker_rows_per_iter": (
                        max(worker_rows_values) if worker_rows_values else ""
                    ),
                    "max_rank_start_skew_ms": (
                        fmt(max(rank_start_skews), 3) if rank_start_skews else ""
                    ),
                    "max_rank_end_skew_ms": (
                        fmt(max(rank_end_skews), 3) if rank_end_skews else ""
                    ),

                    # GPU / non-GPU metrics
                    "e2e_gpu_ms": fmt(e2e_gpu_ms, 3),
                    "ttft_gpu_ms": fmt(ttft_gpu_ms, 3),
                    "tpot_gpu_total_ms": fmt(decode_gpu_total_ms, 3),
                    "tpot_gpu_ms": fmt(tpot_gpu_ms, 6),

                    "e2e_non_gpu_wall_ms": fmt(e2e_non_gpu, 3),
                    "ttft_non_gpu_wall_ms": fmt(ttft_non_gpu, 3),
                    "tpot_non_gpu_wall_ms": fmt(tpot_non_gpu, 6),

                    "e2e_non_gpu_wall_ratio": fmt(e2e_ratio, 6),
                    "ttft_non_gpu_wall_ratio": fmt(ttft_ratio, 6),
                    "tpot_non_gpu_wall_ratio": fmt(tpot_ratio, 6),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_phase_outputs(
    phase,
    request_state,
    step_rows,
    detail_rows,
    sample_rows,
    thread_detail_rows,
    include_gpu,
):
    df_step = pd.DataFrame(step_rows)
    df_detail = pd.DataFrame(detail_rows)
    df_samples = pd.DataFrame(sample_rows)
    df_thread_details = pd.DataFrame(thread_detail_rows)

    df_summary = build_phase_summary(
        request_state=request_state,
        df_detail=df_detail,
        include_gpu=include_gpu,
    )

    df_step.to_csv(PROFILE_DIR / f"{phase}_step_metrics.csv", index=False)
    df_detail.to_csv(PROFILE_DIR / f"{phase}_request_step_detail.csv", index=False)
    df_samples.to_csv(PROFILE_DIR / f"{phase}_thread_samples.csv", index=False)
    df_thread_details.to_csv(PROFILE_DIR / f"{phase}_thread_contention_detail.csv", index=False)
    df_summary.to_csv(PROFILE_DIR / f"{phase}_request_metrics_measured.csv", index=False)

    print(f"[INFO] saved phase outputs under {PROFILE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["active", "gpu"])
    args = parser.parse_args()

    phase = args.phase

    random.seed(SEED)
    torch.manual_seed(SEED)

    applied_cpus = maybe_apply_cpu_affinity_from_env()
    print(f"[INFO] phase={phase}")
    print(f"[INFO] profile_dir={PROFILE_DIR}")
    print(f"[INFO] effective_cpu_affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}")
    print(f"[INFO] contention_available_cores={get_available_core_count()}")
    print(f"[INFO] prompt_specs={PROMPT_SPECS_PATH}")
    print(f"[INFO] ITER_CSV={ITER_CSV}")
    print(
        f"[INFO] min_batch_barrier_ms={MIN_BATCH_BARRIER_MS}, "
        f"target_requests={MIN_BATCH_TARGET_REQUESTS}, "
        f"exclude_from_latency={int(EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY)}, "
        f"reset_baseline_after_add_loop={int(RESET_LATENCY_BASELINE_AFTER_ADD_LOOP)}"
    )

    engine = build_engine()
    tokenizer = engine.get_tokenizer()

    maybe_create_prompt_specs(tokenizer)
    prompt_specs = load_prompt_specs()

    run_warmup(engine, prompt_specs)

    measure_threads = phase == "active"
    include_gpu = phase == "gpu"

    request_state, step_rows, detail_rows, sample_rows, thread_detail_rows = run_phase(
        engine=engine,
        phase_name=phase,
        prompt_specs=prompt_specs,
        measure_threads=measure_threads,
    )

    if include_gpu:
        time.sleep(0.2)

    save_phase_outputs(
        phase=phase,
        request_state=request_state,
        step_rows=step_rows,
        detail_rows=detail_rows,
        sample_rows=sample_rows,
        thread_detail_rows=thread_detail_rows,
        include_gpu=include_gpu,
    )


if __name__ == "__main__":
    main()