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
# Min-batch barrier
# ============================================================
# Goal:
#   Add all NUM_REQUESTS to the vLLM engine first, then wait a short
#   deterministic interval before the first engine.step().  This gives the
#   multiprocess EngineCore side enough time to receive all requests before
#   the scheduler builds the first prefill batch.
#
# Measurement rule:
#   By default, the barrier is excluded from request-level latency.  We keep
#   raw submit timestamps for diagnostics, but reset submit_wall/submit_perf to
#   the barrier end time so e2e/queue/TTFT/TPOT are not inflated by the
#   artificial batching delay.
MIN_BATCH_BARRIER_MS = float(os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "0"))
MIN_BATCH_TARGET_REQUESTS = int(
    os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", str(NUM_REQUESTS))
)
EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = os.environ.get(
    "VLLM_EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY", "1"
).lower() in {"1", "true", "yes", "on"}

# With scheduler-side hold, the driver-side sleep barrier should usually be 0.
# Still, for fair request-level latency, we normally reset submit_* to the
# moment after all requests have been added. This removes the artificial
# sequential add_request loop gap from request 0/1 while preserving the actual
# model-serving interval after the batch is ready.
RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = os.environ.get(
    "VLLM_RESET_LATENCY_BASELINE_AFTER_ADD_LOOP", "1"
).lower() in {"1", "true", "yes", "on"}


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


class StepSampler:
    def __init__(self, root_pid: int, interval_sec: float):
        self.root_pid = root_pid
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.samples: List[dict] = []
        self.thread_keys_seen = set()
        self.cpu_active_thread_keys_seen = set()
        self.thread = None
        self.sampler_thread_key = None
        self._prev_thread_cpu = None

    def start(self, global_step: int, initial_thread_cpu=None):
        self.samples = []
        self.thread_keys_seen = set()
        self.cpu_active_thread_keys_seen = set()
        self.sampler_thread_key = None
        self._prev_thread_cpu = dict(initial_thread_cpu or {})
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
            flat = flatten_thread_cpu(tree)
            total_threads = tree_total_threads(tree)
            process_count = len(tree)

            for pid, snap in tree.items():
                for tid in snap["thread_cpu_s"]:
                    key = (pid, tid)
                    if key != self.sampler_thread_key:
                        self.thread_keys_seen.add(key)

            active_count, cpu_delta_s, active_keys = diff_thread_cpu(
                self._prev_thread_cpu or {},
                flat,
                exclude_keys={self.sampler_thread_key},
            )
            self.cpu_active_thread_keys_seen.update(active_keys)
            self._prev_thread_cpu = flat

            self.samples.append(
                {
                    "global_step": global_step,
                    "sample_idx": sample_idx,
                    "t_rel_ms": (time.perf_counter() - t0) * 1000.0,
                    "process_count": process_count,
                    "worker_process_count": max(0, process_count - 1),
                    "total_threads": total_threads,
                    "sample_active_thread_count": active_count,
                    "sample_thread_cpu_delta_s": cpu_delta_s,
                }
            )

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

    With scheduler-side min-batch hold, the driver-side sleep is usually
    disabled by setting VLLM_MIN_BATCH_BARRIER_MS=0.  We still use this
    helper to create a single diagnostic object and, when requested, reset
    request-level submit_* timestamps to the point after all add_request()
    calls have completed.

    Important: baseline_reset_only must be defined locally.  Older generated
    code referenced it without assignment, which caused NameError during
    warmup before any measurement started.
    """
    requested_ms = max(0.0, float(MIN_BATCH_BARRIER_MS))
    target_requests = int(MIN_BATCH_TARGET_REQUESTS)
    enabled = requested_ms > 0.0 and actual_request_count > 1

    # Reset latency baseline after the add_request loop even when the
    # driver-side sleep barrier is disabled.  This removes the small artificial
    # sequential request-insertion gap from request-level e2e/queue metrics.
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
    latency baseline consumed by compute_latency_ms().  Therefore TTFT/e2e/queue
    remain comparable with older runs, while raw_* diagnostics still show the
    artificial waiting window.
    """
    exclude = bool(barrier_info.get("min_batch_barrier_excluded_from_latency", 0))

    for state in request_state.values():
        raw_submit_wall = state.get("raw_submit_wall", state.get("submit_wall"))
        raw_submit_perf = state.get("raw_submit_perf", state.get("submit_perf"))
        state["raw_submit_wall"] = raw_submit_wall
        state["raw_submit_perf"] = raw_submit_perf

        for k, v in barrier_info.items():
            if k.endswith("_perf"):
                # perf_counter values are process-local diagnostics only.
                state[k] = v
            else:
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

        _sleep_min_batch_barrier(
            phase_name=f"warmup_iter{warmup_iter}",
            actual_request_count=NUM_REQUESTS,
        )

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

        raw_submit_wall = time.time()
        raw_submit_perf = time.perf_counter()

        add_start_wall = time.time()
        add_start_perf = time.perf_counter()
        engine.add_request(req_id, prompt, sampling_params)
        add_end_perf = time.perf_counter()
        add_end_wall = time.time()

        request_state[req_id] = {
            "pair_id": pair_id,
            "request_id": req_id,
            "phase": phase_name,
            "raw_submit_wall": raw_submit_wall,
            "raw_submit_perf": raw_submit_perf,
            # submit_* is the latency baseline. It is overwritten after the
            # min-batch barrier when the barrier is excluded from latency.
            "submit_wall": raw_submit_wall,
            "submit_perf": raw_submit_perf,
            "add_start_wall": add_start_wall,
            "add_start_perf": add_start_perf,
            "add_end_wall": add_end_wall,
            "add_end_perf": add_end_perf,
            "add_wall_ms": max(0.0, (add_end_perf - add_start_perf) * 1000.0),
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
    global_step = 0

    while engine.has_unfinished_requests():
        if measure_threads:
            before_tree = collect_process_tree(os.getpid())
            before_threads = flatten_thread_cpu(before_tree)
            before_proc_cpu = tree_total_proc_cpu(before_tree)
            sampler.start(global_step, initial_thread_cpu=before_threads)
        else:
            before_tree = {}
            before_threads = {}
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

    return request_state, step_rows, req_detail_rows, sample_rows


# ============================================================
# GPU iteration parsing
# ============================================================

def read_csv_dicts(path: Path):
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_first_non_warmup_min_batch_info():
    """Return diagnostics from sitecustomize's first non-warmup iteration.

    In TP=2 the first non-warmup iteration is logged once per rank.  We keep
    min/max num_requests and all-rank ok so the final CSV can quickly show
    whether the first measured prefill actually contained all target requests.
    """
    rows = [
        r for r in read_csv_dicts(ITER_CSV)
        if str(r.get("is_first_non_warmup_iter", "")) == "1"
    ]

    if not rows:
        return {}

    def to_int_field(row, field, default=None):
        try:
            v = row.get(field, "")
            if v == "" or v is None:
                return default
            return int(float(v))
        except Exception:
            return default

    num_requests = [
        v for v in (to_int_field(r, "num_requests", None) for r in rows)
        if v is not None
    ]
    targets = [
        v for v in (
            to_int_field(r, "first_non_warmup_min_batch_target", None)
            for r in rows
        )
        if v is not None
    ]
    ok_values = [
        v for v in (
            to_int_field(r, "first_non_warmup_min_batch_ok", None)
            for r in rows
        )
        if v is not None
    ]

    return {
        "first_non_warmup_iter_rows": len(rows),
        "first_non_warmup_num_requests_min": (
            min(num_requests) if num_requests else ""
        ),
        "first_non_warmup_num_requests_max": (
            max(num_requests) if num_requests else ""
        ),
        "first_non_warmup_min_batch_target": (
            max(targets) if targets else ""
        ),
        "first_non_warmup_min_batch_ok_all_ranks": (
            int(all(v == 1 for v in ok_values)) if ok_values else ""
        ),
    }


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
    first_batch_info = (
        read_first_non_warmup_min_batch_info() if include_gpu else {}
    )
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

        raw_e2e_ms = None
        raw_e2ft_ms = None
        raw_queue_ms = None

        if state.get("raw_submit_perf") is not None and state.get("end_perf") is not None:
            raw_e2e_ms = max(
                0.0,
                (state["end_perf"] - state["raw_submit_perf"]) * 1000.0,
            )

        if (
            state.get("raw_submit_perf") is not None
            and state.get("first_token_perf") is not None
        ):
            raw_e2ft_ms = max(
                0.0,
                (state["first_token_perf"] - state["raw_submit_perf"]) * 1000.0,
            )

        if (
            state.get("raw_submit_perf") is not None
            and state.get("start_processing_perf") is not None
        ):
            raw_queue_ms = max(
                0.0,
                (state["start_processing_perf"] - state["raw_submit_perf"]) * 1000.0,
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
            "raw_submit_wall": fmt(state.get("raw_submit_wall"), 9),
            "submit_wall": fmt(state.get("submit_wall"), 9),
            "add_wall_ms": fmt(state.get("add_wall_ms"), 6),
            "raw_to_latency_submit_gap_ms": fmt(
                state.get("raw_to_latency_submit_gap_ms"),
                3,
            ),
            "min_batch_barrier_enabled": state.get("min_batch_barrier_enabled", 0),
            "min_batch_barrier_requested_ms": fmt(
                state.get("min_batch_barrier_requested_ms", 0.0),
                3,
            ),
            "min_batch_barrier_observed_ms": fmt(
                state.get("min_batch_barrier_observed_ms", 0.0),
                3,
            ),
            "min_batch_barrier_target_requests": state.get(
                "min_batch_barrier_target_requests",
                "",
            ),
            "min_batch_barrier_actual_requests": state.get(
                "min_batch_barrier_actual_requests",
                "",
            ),
            "min_batch_barrier_excluded_from_latency": state.get(
                "min_batch_barrier_excluded_from_latency",
                0,
            ),
            "latency_baseline_reset_after_add_loop": state.get(
                "latency_baseline_reset_after_add_loop",
                0,
            ),
            "first_non_warmup_iter_rows": first_batch_info.get(
                "first_non_warmup_iter_rows",
                "",
            ),
            "first_non_warmup_num_requests_min": first_batch_info.get(
                "first_non_warmup_num_requests_min",
                "",
            ),
            "first_non_warmup_num_requests_max": first_batch_info.get(
                "first_non_warmup_num_requests_max",
                "",
            ),
            "first_non_warmup_min_batch_target": first_batch_info.get(
                "first_non_warmup_min_batch_target",
                "",
            ),
            "first_non_warmup_min_batch_ok_all_ranks": first_batch_info.get(
                "first_non_warmup_min_batch_ok_all_ranks",
                "",
            ),
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
            "raw_e2e_ms_including_barrier": fmt(raw_e2e_ms, 3),
            "raw_queue_ms_including_barrier": fmt(raw_queue_ms, 3),
            "raw_e2ft_ms_including_barrier": fmt(raw_e2ft_ms, 3),

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


            prefill_time_ms = (
                sum(float(it["iter_wall_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            prefill_gpu_ms = (
                sum(float(it["iter_gpu_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            prefill_non_gpu_wall_ms = (
                sum(float(it["iter_non_gpu_wall_ms_max"]) for it in gpu_prefill)
                if gpu_prefill
                else None
            )

            prefill_gpu_exceeds_wall = 0

            if prefill_time_ms is not None and prefill_gpu_ms is not None:
                prefill_gpu_exceeds_wall = int(
                    prefill_gpu_ms > prefill_time_ms + 1e-3
                )

            prefill_non_gpu_wall_ratio = (
                None
                if (
                    prefill_time_ms is None
                    or not prefill_time_ms
                    or prefill_non_gpu_wall_ms is None
                )
                else prefill_non_gpu_wall_ms / prefill_time_ms
            )

        
            ttft_non_gpu = (
                None
                if actual_ttft_ms is None or ttft_gpu_ms is None
                else max(0.0, actual_ttft_ms - ttft_gpu_ms)
            )

            

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
                    "ttft_actual_ms": fmt(actual_ttft_ms, 3),
                    "ttft_actual_raw_including_barrier_ms": fmt(raw_e2ft_ms, 3),
                    "ttft_request_observed_ms": fmt(request_observed_ttft_ms, 3),
                    "ttft_compute_window_ms": fmt(ttft_compute_window_ms, 3),
                    "e2ft_request_observed_ms": fmt(e2ft_request_observed_ms, 3),
                    "tpot_request_observed_ms": fmt(tpot_request_observed_ms, 6),

                    # Prefill iteration-derived values.
                    "prefill_time_ms": fmt(prefill_time_ms, 3),
                    "prefill_gpu_ms": fmt(prefill_gpu_ms, 3),
                    "prefill_non_gpu_wall_ms": fmt(prefill_non_gpu_wall_ms, 3),
                    "prefill_non_gpu_wall_ratio": fmt(prefill_non_gpu_wall_ratio, 6),

                    # Backward-compatible diagnostic names.
                    "ttft_prefill_wall_ms": fmt(prefill_time_ms, 3),
                    "ttft_prefill_non_gpu_wall_ms": fmt(
                        prefill_non_gpu_wall_ms,
                        3,
                    ),
                    "ttft_gpu_exceeds_prefill_wall": prefill_gpu_exceeds_wall,
                    "prefill_gpu_exceeds_wall": prefill_gpu_exceeds_wall,
                    "ttft_ms": fmt(ttft_ms_for_report, 3),

                    # TPOT iteration-derived
                    "tpot_decode_wall_total_ms": fmt(decode_wall_total_ms, 3),
                    "tpot_decode_gpu_total_ms": fmt(decode_gpu_total_ms, 3),
                    "tpot_decode_non_gpu_wall_total_ms": fmt(
                        decode_non_gpu_wall_total_ms,
                        3,
                    ),
                    "tpot_gpu_exceeds_decode_wall": tpot_gpu_exceeds_decode_wall,

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
    include_gpu,
):
    df_step = pd.DataFrame(step_rows)
    df_detail = pd.DataFrame(detail_rows)
    df_samples = pd.DataFrame(sample_rows)

    df_summary = build_phase_summary(
        request_state=request_state,
        df_detail=df_detail,
        include_gpu=include_gpu,
    )

    df_step.to_csv(PROFILE_DIR / f"{phase}_step_metrics.csv", index=False)
    df_detail.to_csv(PROFILE_DIR / f"{phase}_request_step_detail.csv", index=False)
    df_samples.to_csv(PROFILE_DIR / f"{phase}_thread_samples.csv", index=False)
    df_summary.to_csv(PROFILE_DIR / f"{phase}_request_metrics_measured.csv", index=False)

    print(f"[INFO] saved phase outputs under {PROFILE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["active", "gpu"])
    args = parser.parse_args()

    phase = args.phase

    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[INFO] phase={phase}")
    print(f"[INFO] profile_dir={PROFILE_DIR}")
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

    request_state, step_rows, detail_rows, sample_rows = run_phase(
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
        include_gpu=include_gpu,
    )


if __name__ == "__main__":
    main()
