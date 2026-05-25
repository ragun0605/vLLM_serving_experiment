#!/usr/bin/env python3
"""
sitecustomize.py for two-phase vLLM profiling.

This version contains two independent patches:
  1) Scheduler-side first main batch hold.
     - This is enabled even when VLLM_DISABLE_GPU_PROFILE_PATCH=1.
     - It prevents gpu_pair0/active_pair0 from entering the scheduler queue
       alone before pair1/pair2 arrive.
  2) GPUModelRunner.execute_model timing/attribution patch.
     - This is disabled only when VLLM_DISABLE_GPU_PROFILE_PATCH=1.
"""

import csv
import os
import threading
import time
from pathlib import Path


PROFILE_DIR = Path(os.environ.get("VLLM_PROFILE_DIR", "./profile_logs"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

ITER_CSV = PROFILE_DIR / "iteration_metrics.csv"
SCHED_HOLD_CSV = PROFILE_DIR / "scheduler_min_batch_hold_metrics.csv"
CSV_LOCK_FILE = PROFILE_DIR / ".iteration_metrics.lock"

_LOCK = threading.Lock()

# ============================================================
# Config
# ============================================================

DISABLE_GPU_PROFILE_PATCH = os.environ.get("VLLM_DISABLE_GPU_PROFILE_PATCH", "0").lower() in {
    "1", "true", "yes", "on"
}

MIN_BATCH_TARGET_REQUESTS = int(
    os.environ.get(
        "VLLM_MIN_BATCH_TARGET_REQUESTS",
        os.environ.get("VLLM_NUM_REQUESTS", "3"),
    )
)

# Scheduler-side first-batch hold.
ENABLE_SCHEDULER_MIN_BATCH_HOLD = os.environ.get(
    "VLLM_ENABLE_SCHEDULER_MIN_BATCH_HOLD", "1"
).lower() in {"1", "true", "yes", "on"}

SCHEDULER_MIN_BATCH_TARGET_REQUESTS = int(
    os.environ.get(
        "VLLM_SCHEDULER_MIN_BATCH_TARGET_REQUESTS",
        os.environ.get(
            "VLLM_MIN_BATCH_TARGET_REQUESTS",
            os.environ.get("VLLM_NUM_REQUESTS", "3"),
        ),
    )
)

SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS = float(
    os.environ.get(
        "VLLM_SCHEDULER_MIN_BATCH_HOLD_MS",
        os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "200"),
    )
)

SCHEDULER_HOLD_INCLUDE_WARMUP = os.environ.get(
    "VLLM_SCHEDULER_HOLD_INCLUDE_WARMUP", "0"
).lower() in {"1", "true", "yes", "on"}

SCHEDULER_HOLD_PREFIXES = tuple(
    p.strip()
    for p in os.environ.get(
        "VLLM_SCHEDULER_HOLD_REQUEST_PREFIXES",
        "active_pair,gpu_pair",
    ).split(",")
    if p.strip()
)

_FIRST_NON_WARMUP_ITER_SEEN = {"v": False}

ITER_HEADER = [
    "iter_idx",
    "iter_start_wall",
    "iter_end_wall",
    "iter_wall_ms",
    "iter_gpu_ms",
    "iter_non_gpu_wall_ms",
    "request_ids",
    "request_id_source",
    "scheduled_tokens_by_request",
    "num_requests",
    "num_warmup_requests",
    "is_warmup",
    "is_first_non_warmup_iter",
    "first_non_warmup_min_batch_target",
    "first_non_warmup_min_batch_ok",
    "process_pid",
    "process_ppid",
    "rank",
    "local_rank",
]

SCHED_HOLD_HEADER = [
    "event",
    "event_wall",
    "event_perf",
    "process_pid",
    "process_ppid",
    "scheduler_class",
    "scheduler_obj_id",
    "request_id",
    "buffer_request_ids",
    "buffer_size",
    "target_requests",
    "timeout_ms",
    "elapsed_ms",
    "reason",
    "patch_path",
]


def _file_locked_write(fn):
    with _LOCK:
        try:
            import fcntl

            with open(CSV_LOCK_FILE, "a+", encoding="utf-8") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    return fn()
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except Exception:
            return fn()


def _append_csv_row(path: Path, header, row):
    def _write():
        need_header = (not path.exists()) or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(header)
            w.writerow(row)

    _file_locked_write(_write)


def _append_iter_row(row):
    _append_csv_row(ITER_CSV, ITER_HEADER, row)


def _append_sched_hold_row(
    event,
    scheduler_obj,
    request_id="",
    buffer_items=None,
    reason="",
    elapsed_ms="",
    patch_path="",
):
    try:
        buffer_items = buffer_items or []
        buffer_ids = [_request_id_from_item(item) for item in buffer_items]
        cls_name = scheduler_obj.__class__.__module__ + "." + scheduler_obj.__class__.__name__
        row = [
            event,
            f"{time.time():.9f}",
            f"{time.perf_counter():.9f}",
            os.getpid(),
            os.getppid(),
            cls_name,
            id(scheduler_obj),
            request_id,
            "|".join(buffer_ids),
            len(buffer_ids),
            SCHEDULER_MIN_BATCH_TARGET_REQUESTS,
            SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS,
            elapsed_ms if elapsed_ms != "" else "",
            reason,
            patch_path,
        ]
        _append_csv_row(SCHED_HOLD_CSV, SCHED_HOLD_HEADER, row)
    except Exception as e:
        # Do not let diagnostics break serving.
        print(f"[sitecustomize] scheduler hold metric write skipped: {e}")


# ============================================================
# Shared request helpers
# ============================================================


def _request_id_from_item(item):
    """Return request id from a Request object or a buffered tuple."""
    try:
        if isinstance(item, tuple) and item:
            item = item[0]
        for attr in ("request_id", "req_id", "id"):
            v = getattr(item, attr, None)
            if v is not None:
                return str(v)
        # Some wrapper objects may keep the request under .request.
        inner = getattr(item, "request", None)
        if inner is not None and inner is not item:
            return _request_id_from_item(inner)
    except Exception:
        pass
    return str(item)


def _is_warmup_request_id(req_id: str) -> bool:
    return str(req_id).startswith("warmup_")


def _is_hold_candidate_request_id(req_id: str) -> bool:
    req_id = str(req_id)
    if _is_warmup_request_id(req_id):
        return bool(SCHEDULER_HOLD_INCLUDE_WARMUP)
    if not SCHEDULER_HOLD_PREFIXES:
        return True
    return req_id.startswith(SCHEDULER_HOLD_PREFIXES)


def _rank():
    return os.environ.get("RANK", os.environ.get("VLLM_RANK", ""))


def _local_rank():
    return os.environ.get("LOCAL_RANK", os.environ.get("VLLM_LOCAL_RANK", ""))


# ============================================================
# Scheduler-side min batch hold patch
# ============================================================


def _patch_scheduler_min_batch_hold():
    if not ENABLE_SCHEDULER_MIN_BATCH_HOLD:
        print("[sitecustomize] scheduler min-batch hold disabled")
        return False

    if SCHEDULER_MIN_BATCH_TARGET_REQUESTS <= 1:
        print(
            "[sitecustomize] scheduler min-batch hold skipped: "
            f"target={SCHEDULER_MIN_BATCH_TARGET_REQUESTS}"
        )
        return False

    candidates = [
        "vllm.v1.core.sched.scheduler",
        "vllm.core.scheduler",
    ]

    last_err = None
    for module_name in candidates:
        try:
            import importlib

            mod = importlib.import_module(module_name)
            Scheduler = getattr(mod, "Scheduler", None)
            if Scheduler is None:
                continue

            if getattr(Scheduler, "_min_batch_hold_patch_done", False):
                return True

            orig_add_request = getattr(Scheduler, "add_request", None)
            if orig_add_request is None:
                continue

            orig_schedule = getattr(Scheduler, "schedule", None)

            def _state(self):
                st = getattr(self, "_min_batch_hold_state", None)
                if st is None:
                    st = {
                        "buffer": [],
                        "start_perf": None,
                        "closed": False,
                        "flush_count": 0,
                    }
                    setattr(self, "_min_batch_hold_state", st)
                return st

            def _elapsed_ms(st):
                if st.get("start_perf") is None:
                    return 0.0
                return (time.perf_counter() - st["start_perf"]) * 1000.0

            def _flush(self, st, reason: str):
                buf = list(st.get("buffer") or [])
                if not buf:
                    return
                elapsed = _elapsed_ms(st)
                # Close before replaying to avoid recursively holding the replayed requests.
                st["closed"] = True
                st["buffer"] = []
                st["flush_count"] = int(st.get("flush_count", 0)) + 1
                _append_sched_hold_row(
                    "flush_before_replay",
                    self,
                    request_id="",
                    buffer_items=buf,
                    reason=reason,
                    elapsed_ms=f"{elapsed:.3f}",
                    patch_path=module_name,
                )
                for request, args, kwargs in buf:
                    orig_add_request(self, request, *args, **kwargs)
                _append_sched_hold_row(
                    "flush_after_replay",
                    self,
                    request_id="",
                    buffer_items=buf,
                    reason=reason,
                    elapsed_ms=f"{elapsed:.3f}",
                    patch_path=module_name,
                )

            def add_request_wrapped(self, request, *args, **kwargs):
                st = _state(self)
                req_id = _request_id_from_item(request)

                if st.get("closed") or not _is_hold_candidate_request_id(req_id):
                    return orig_add_request(self, request, *args, **kwargs)

                if st.get("start_perf") is None:
                    st["start_perf"] = time.perf_counter()
                    _append_sched_hold_row(
                        "start",
                        self,
                        request_id=req_id,
                        buffer_items=[],
                        reason="first_candidate",
                        elapsed_ms="0.000",
                        patch_path=module_name,
                    )

                st["buffer"].append((request, args, kwargs))
                elapsed = _elapsed_ms(st)
                _append_sched_hold_row(
                    "hold",
                    self,
                    request_id=req_id,
                    buffer_items=st["buffer"],
                    reason="waiting_for_target",
                    elapsed_ms=f"{elapsed:.3f}",
                    patch_path=module_name,
                )

                if len(st["buffer"]) >= SCHEDULER_MIN_BATCH_TARGET_REQUESTS:
                    _flush(self, st, "target_reached")
                elif (
                    SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS >= 0
                    and elapsed >= SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS
                ):
                    _flush(self, st, "timeout_on_add_request")

                return None

            setattr(Scheduler, "add_request", add_request_wrapped)

            if orig_schedule is not None:
                def schedule_wrapped(self, *args, **kwargs):
                    st = _state(self)
                    if (
                        not st.get("closed")
                        and st.get("buffer")
                        and SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS >= 0
                        and _elapsed_ms(st) >= SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS
                    ):
                        _flush(self, st, "timeout_on_schedule")
                    return orig_schedule(self, *args, **kwargs)

                setattr(Scheduler, "schedule", schedule_wrapped)

            setattr(Scheduler, "_min_batch_hold_patch_done", True)
            print(
                "[sitecustomize] Scheduler.add_request patched for first-batch hold "
                f"module={module_name}, target={SCHEDULER_MIN_BATCH_TARGET_REQUESTS}, "
                f"timeout_ms={SCHEDULER_MIN_BATCH_HOLD_TIMEOUT_MS}, "
                f"prefixes={SCHEDULER_HOLD_PREFIXES}, include_warmup={SCHEDULER_HOLD_INCLUDE_WARMUP}"
            )
            return True
        except Exception as e:
            last_err = e
            continue

    print(f"[sitecustomize] Scheduler.add_request hold patch skipped: {last_err}")
    return False


# ============================================================
# GPU iteration timing patch
# ============================================================


def _extract_request_ids_and_scheduled_tokens(scheduler_output):
    """
    실제 scheduled된 request만 attribution 대상으로 사용합니다.
    finished_req_ids는 종료 bookkeeping에 가까운 경우가 있으므로
    GPU time attribution에 사용하지 않습니다.
    """
    ids = set()
    token_map = {}

    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)

    if num_scheduled_tokens is not None:
        try:
            for req_id, n_tokens in num_scheduled_tokens.items():
                req_id_s = str(req_id)
                ids.add(req_id_s)
                try:
                    token_map[req_id_s] = int(n_tokens)
                except Exception:
                    token_map[req_id_s] = -1

            if ids:
                return sorted(ids), token_map, "num_scheduled_tokens"
        except Exception:
            pass

    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if scheduled_new_reqs is not None:
        try:
            for new_req_data in scheduled_new_reqs:
                req_id = getattr(new_req_data, "req_id", None)
                if req_id is not None:
                    ids.add(str(req_id))
            if ids:
                return sorted(ids), token_map, "scheduled_new_reqs_fallback"
        except Exception:
            pass

    scheduled_cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if scheduled_cached_reqs is not None:
        try:
            req_ids = getattr(scheduled_cached_reqs, "req_ids", None)
            if req_ids is not None:
                ids.update(str(req_id) for req_id in req_ids)
            if ids:
                return sorted(ids), token_map, "scheduled_cached_reqs_fallback"
        except Exception:
            pass

    return [], {}, "none"


def _format_scheduled_tokens_by_request(token_map):
    if not token_map:
        return ""

    return "|".join(
        f"{req_id}:{token_map[req_id]}"
        for req_id in sorted(token_map.keys())
    )


def _count_warmup_request_ids(req_ids):
    return sum(1 for req_id in req_ids if str(req_id).startswith("warmup_"))


def _patch_gpu_model_runner():
    if DISABLE_GPU_PROFILE_PATCH:
        print("[sitecustomize] GPU profile patch disabled by VLLM_DISABLE_GPU_PROFILE_PATCH")
        return False

    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:
        print(f"[sitecustomize] GPUModelRunner patch skipped: {e}")
        return False

    if getattr(GPUModelRunner, "_two_phase_gpu_profile_patch_done", False):
        return True

    orig = GPUModelRunner.execute_model
    counter = {"v": 0}

    def wrapped(self, scheduler_output, *args, **kwargs):
        iter_idx = counter["v"]
        counter["v"] += 1

        req_ids, scheduled_token_map, req_id_source = (
            _extract_request_ids_and_scheduled_tokens(scheduler_output)
        )

        num_warmup_requests = _count_warmup_request_ids(req_ids)
        is_warmup = 1 if (
            len(req_ids) > 0 and num_warmup_requests == len(req_ids)
        ) else 0

        is_first_non_warmup_iter = 0
        first_non_warmup_min_batch_ok = ""
        if len(req_ids) > 0 and not is_warmup and not _FIRST_NON_WARMUP_ITER_SEEN["v"]:
            _FIRST_NON_WARMUP_ITER_SEEN["v"] = True
            is_first_non_warmup_iter = 1
            first_non_warmup_min_batch_ok = int(
                len(req_ids) >= MIN_BATCH_TARGET_REQUESTS
            )

        process_pid = os.getpid()
        process_ppid = os.getppid()
        rank = _rank()
        local_rank = _local_rank()

        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_wall = time.time()
        start_perf = time.perf_counter()

        start_evt.record()
        out = orig(self, scheduler_output, *args, **kwargs)
        end_evt.record()
        torch.cuda.synchronize()

        end_perf = time.perf_counter()
        end_wall = time.time()

        gpu_ms = float(start_evt.elapsed_time(end_evt))
        wall_ms = float((end_perf - start_perf) * 1000.0)
        non_gpu_wall_ms = max(0.0, wall_ms - gpu_ms)

        row = [
            iter_idx,
            f"{start_wall:.9f}",
            f"{end_wall:.9f}",
            f"{wall_ms:.3f}",
            f"{gpu_ms:.3f}",
            f"{non_gpu_wall_ms:.3f}",
            "|".join(req_ids),
            req_id_source,
            _format_scheduled_tokens_by_request(scheduled_token_map),
            len(req_ids),
            num_warmup_requests,
            is_warmup,
            is_first_non_warmup_iter,
            MIN_BATCH_TARGET_REQUESTS,
            first_non_warmup_min_batch_ok,
            process_pid,
            process_ppid,
            rank,
            local_rank,
        ]

        _append_iter_row(row)
        return out

    GPUModelRunner.execute_model = wrapped
    GPUModelRunner._two_phase_gpu_profile_patch_done = True

    print("[sitecustomize] GPUModelRunner.execute_model patched for High-Precision GPU timing")
    return True


# Scheduler hold must be attempted regardless of GPU profile patch enable/disable.
_patch_scheduler_min_batch_hold()
_patch_gpu_model_runner()
