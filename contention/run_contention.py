#!/usr/bin/env python3
"""
Run a scheduler-hold-compatible contention-focused vLLM experiment.

This runner keeps the same scheduler hold / min-batch environment as
run5_scheduler_hold.py, but executes only the active phase and outputs the
runnable/active thread identities observed during contention-suspicious
engine.step() calls.

Main outputs:
  BASE_PROFILE_DIR/contention_runnable_thread_details.csv
  BASE_PROFILE_DIR/contention_runnable_thread_summary_by_tid.csv
  BASE_PROFILE_DIR/contention_runnable_thread_summary_by_type.csv

Use with:
  export VLLM_PHASE_WORKER=./phase_worker_scheduler_hold_runnable_thread_identity.py
  export VLLM_CPUSET=0-3
  export VLLM_CONTENTION_CORE_COUNT=4
  python run5_scheduler_hold_contention_identity.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


BASE_PROFILE_DIR = Path(os.environ.get("VLLM_BASE_PROFILE_DIR", "./scheduler_hold_runnable_thread_identity_logs"))
WORKER_SCRIPT = Path(os.environ.get("VLLM_PHASE_WORKER", "./phase_worker_scheduler_hold_runnable_thread_identity.py")).resolve()

# The scheduler-hold patch must be importable as a module named sitecustomize.
# This runner copies sitecustomize_scheduler_hold_v2.py into a private import
# directory as sitecustomize.py and prepends that directory to PYTHONPATH, so
# it does not depend on manually renaming the uploaded file.
SITECUSTOMIZE_SOURCE = Path(
    os.environ.get("VLLM_SCHEDULER_HOLD_SITECUSTOMIZE", "./sitecustomize_scheduler_hold_v2.py")
).resolve()
RESET = os.environ.get("VLLM_RESET_PROFILE_FILES", "1").lower() not in {"0", "false", "no"}
NUM_REPEATS = int(os.environ.get("VLLM_NUM_REPEATS", "5"))
PROMPT_SPECS_PATH = BASE_PROFILE_DIR / "prompt_specs.json"

# Save every step's runnable/active threads, not just contention-suspicious steps.
SAVE_ALL_STEPS = os.environ.get("VLLM_CONTENTION_SAVE_ALL_STEPS", "0").lower() in {"1", "true", "yes", "on"}
KEEP_RAW = os.environ.get("VLLM_KEEP_CONTENTION_RAW", "0").lower() in {"1", "true", "yes", "on"}
CPU_UTIL_THRESHOLD = float(os.environ.get("VLLM_CONTENTION_CPU_UTIL_THRESHOLD", "0.85"))

# Match run5_scheduler_hold.py defaults unless the user overrides them.
MIN_BATCH_BARRIER_MS = os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "0")
MIN_BATCH_TARGET_REQUESTS = os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", "")
EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = os.environ.get("VLLM_EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY", "1")
RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = os.environ.get("VLLM_RESET_LATENCY_BASELINE_AFTER_ADD_LOOP", "1")
ENABLE_SCHEDULER_MIN_BATCH_HOLD = os.environ.get("VLLM_ENABLE_SCHEDULER_MIN_BATCH_HOLD", "1")
SCHEDULER_MIN_BATCH_HOLD_MS = os.environ.get(
    "VLLM_SCHEDULER_MIN_BATCH_HOLD_MS",
    os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "200"),
)
SCHEDULER_MIN_BATCH_TARGET_REQUESTS = os.environ.get(
    "VLLM_SCHEDULER_MIN_BATCH_TARGET_REQUESTS",
    os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", ""),
)


def _prepend_pythonpath(env: Dict[str, str], paths: Iterable[Path]):
    old_pythonpath = env.get("PYTHONPATH", "")
    unique = []
    for p in paths:
        p = str(Path(p).resolve())
        if p not in unique:
            unique.append(p)
    env["PYTHONPATH"] = os.pathsep.join(unique + ([old_pythonpath] if old_pythonpath else []))


def _prepare_sitecustomize_import_dir() -> Path:
    import_dir = BASE_PROFILE_DIR / "_scheduler_hold_sitecustomize_import"
    import_dir.mkdir(parents=True, exist_ok=True)
    dst = import_dir / "sitecustomize.py"
    if SITECUSTOMIZE_SOURCE.exists():
        shutil.copy2(SITECUSTOMIZE_SOURCE, dst)
        print(f"[INFO] scheduler-hold sitecustomize prepared: {dst}")
    elif dst.exists():
        print(f"[WARN] sitecustomize source missing but existing copy will be used: {dst}")
    else:
        print(f"[WARN] scheduler-hold sitecustomize source not found: {SITECUSTOMIZE_SOURCE}")
        print("[WARN] The run will proceed, but scheduler-side hold patch may not be loaded.")
    return import_dir


def run_active_phase_subprocess(repeat_id: int, profile_dir: Path, sitecustomize_import_dir: Path):
    env = os.environ.copy()
    env["VLLM_PROFILE_DIR"] = str(profile_dir)
    env["VLLM_PROMPT_SPECS_PATH"] = str(PROMPT_SPECS_PATH)
    env["VLLM_REPEAT_ID"] = str(repeat_id)
    env["VLLM_DISABLE_GPU_PROFILE_PATCH"] = "1"
    env["VLLM_TRACE_THREAD_IDENTITIES"] = "1"

    # Ensure scheduler-hold settings match the base run5/phase_worker environment.
    env.setdefault("VLLM_ENABLE_SCHEDULER_MIN_BATCH_HOLD", ENABLE_SCHEDULER_MIN_BATCH_HOLD)
    env.setdefault("VLLM_MIN_BATCH_BARRIER_MS", MIN_BATCH_BARRIER_MS)
    if MIN_BATCH_TARGET_REQUESTS:
        env.setdefault("VLLM_MIN_BATCH_TARGET_REQUESTS", MIN_BATCH_TARGET_REQUESTS)
    env.setdefault("VLLM_EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY", EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY)
    env.setdefault("VLLM_RESET_LATENCY_BASELINE_AFTER_ADD_LOOP", RESET_LATENCY_BASELINE_AFTER_ADD_LOOP)
    env.setdefault("VLLM_SCHEDULER_MIN_BATCH_HOLD_MS", SCHEDULER_MIN_BATCH_HOLD_MS)
    if SCHEDULER_MIN_BATCH_TARGET_REQUESTS:
        env.setdefault("VLLM_SCHEDULER_MIN_BATCH_TARGET_REQUESTS", SCHEDULER_MIN_BATCH_TARGET_REQUESTS)

    _prepend_pythonpath(
        env,
        [sitecustomize_import_dir, Path.cwd().resolve(), WORKER_SCRIPT.parent.resolve()],
    )

    cmd = [sys.executable, str(WORKER_SCRIPT), "--phase", "active"]

    print("\n" + "=" * 100)
    print(f"[RUN] repeat_id={repeat_id}")
    print("[RUN] phase=active/runnable-thread-identity")
    print(f"[RUN] profile_dir={profile_dir}")
    print(f"[RUN] worker_script={WORKER_SCRIPT}")
    print(f"[RUN] prompt_specs={PROMPT_SPECS_PATH}")
    print(f"[RUN] save_all_steps={SAVE_ALL_STEPS}")
    print(f"[RUN] scheduler_hold_sitecustomize={SITECUSTOMIZE_SOURCE}")
    print(f"[RUN] scheduler_hold_enabled={env.get('VLLM_ENABLE_SCHEDULER_MIN_BATCH_HOLD')}")
    print(f"[RUN] scheduler_hold_ms={env.get('VLLM_SCHEDULER_MIN_BATCH_HOLD_MS')}")
    print(f"[RUN] scheduler_hold_target={env.get('VLLM_SCHEDULER_MIN_BATCH_TARGET_REQUESTS', 'NUM_REQUESTS')}")
    print(f"[RUN] driver_min_batch_barrier_ms={env.get('VLLM_MIN_BATCH_BARRIER_MS')}")
    print("=" * 100)

    subprocess.run(cmd, env=env, check=True)


def _safe_numeric(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _as_int_str_values(values) -> List[str]:
    out = []
    for v in values:
        try:
            if pd.isna(v):
                continue
            out.append(str(int(float(v))))
        except Exception:
            s = str(v).strip()
            if s:
                out.append(s)
    return sorted(set(out), key=lambda x: int(x) if x.isdigit() else x)


def _first_non_empty(values) -> str:
    for v in values:
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _build_step_location_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty or "global_step" not in detail.columns:
        return pd.DataFrame()

    detail = detail.copy()
    detail["pair_id_num"] = pd.to_numeric(detail.get("pair_id", pd.Series([], dtype=object)), errors="coerce")
    detail["decode_iter_num"] = pd.to_numeric(detail.get("request_decode_iter_idx", pd.Series([], dtype=object)), errors="coerce")

    rows = []
    for global_step, g in detail.groupby("global_step", dropna=False):
        stages = [str(x) for x in g.get("request_stage", pd.Series([], dtype=str)).dropna().tolist()]
        stage_set = sorted(set(stages))
        if not stage_set:
            step_stage = "NoReturnedRequest"
        elif len(stage_set) == 1:
            step_stage = stage_set[0]
        else:
            step_stage = "Mixed(" + "+".join(stage_set) + ")"

        pair_ids = _as_int_str_values(g.get("pair_id_num", pd.Series([], dtype=object)).tolist())

        decode_iters = []
        for v in g.get("decode_iter_num", pd.Series([], dtype=object)).tolist():
            try:
                if pd.notna(v) and int(v) >= 0:
                    decode_iters.append(int(v))
            except Exception:
                pass
        decode_iters = sorted(set(decode_iters))

        if step_stage == "Prefill":
            logical_location = "Prefill"
        elif step_stage == "Decode":
            if not decode_iters:
                logical_location = "Decode[unknown]"
            elif len(decode_iters) == 1:
                logical_location = f"Decode[{decode_iters[0]}]"
            else:
                logical_location = f"Decode[{min(decode_iters)}-{max(decode_iters)}]"
        else:
            logical_location = step_stage

        loc_parts = []
        for _, r in g.sort_values(["pair_id_num", "request_id"], na_position="last").iterrows():
            try:
                pair = str(int(r.get("pair_id_num"))) if pd.notna(r.get("pair_id_num")) else "?"
            except Exception:
                pair = "?"
            stage = str(r.get("request_stage", ""))
            dec = r.get("decode_iter_num", pd.NA)
            if stage == "Decode":
                try:
                    stage_s = f"Decode[{int(dec)}]" if pd.notna(dec) else "Decode[unknown]"
                except Exception:
                    stage_s = "Decode[unknown]"
            else:
                stage_s = stage or "Unknown"
            req_id = str(r.get("request_id", ""))[:48]
            loc_parts.append(f"pair{pair}:{stage_s}:req={req_id}")

        rows.append({
            "global_step": global_step,
            "step_stage": step_stage,
            "logical_location": logical_location,
            "pair_ids_in_step": "|".join(pair_ids),
            "decode_iter_min": min(decode_iters) if decode_iters else "",
            "decode_iter_max": max(decode_iters) if decode_iters else "",
            "decode_iter_indices": "|".join(str(x) for x in decode_iters),
            "request_location_detail": " | ".join(loc_parts),
            "detail_returned_request_ids": _first_non_empty(g.get("shared_returned_request_ids", pd.Series([], dtype=str)).tolist()),
            "detail_finished_request_ids_now": _first_non_empty(g.get("shared_finished_request_ids_now", pd.Series([], dtype=str)).tolist()),
            "detail_active_request_ids_after_step": _first_non_empty(g.get("shared_active_request_ids_after_step", pd.Series([], dtype=str)).tolist()),
        })
    return pd.DataFrame(rows)


def _derive_contention_reasons(row: pd.Series) -> str:
    reasons = []
    def val(name, default=0.0):
        try:
            return float(row.get(name, default))
        except Exception:
            return default
    if val("step_runnable_over_core_p90") > 0:
        reasons.append("runnable_p90_over_core")
    if val("step_runnable_over_core_max") > 0:
        reasons.append("runnable_max_over_core")
    if val("step_active_over_core_p90") > 0:
        reasons.append("active_p90_over_core")
    if val("step_active_over_core_max") > 0:
        reasons.append("active_max_over_core")
    if val("step_cpu_utilization_ratio") >= CPU_UTIL_THRESHOLD:
        reasons.append(f"cpu_util_ge_{CPU_UTIL_THRESHOLD:g}")
    if val("step_involuntary_ctx_switch_delta") > 0:
        reasons.append("involuntary_context_switch")
    return ";".join(reasons) if reasons else "no_strong_contention_signal"


def _build_event_steps(step: pd.DataFrame, loc: pd.DataFrame) -> pd.DataFrame:
    if not loc.empty:
        step = step.merge(loc, on="global_step", how="left")
    for col in ["step_stage", "logical_location", "pair_ids_in_step", "decode_iter_min", "decode_iter_max", "decode_iter_indices", "request_location_detail"]:
        if col not in step.columns:
            step[col] = ""
        step[col] = step[col].fillna("")
    for col in [
        "step_contention_flag", "step_cpu_utilization_ratio", "step_runnable_over_core_p90",
        "step_runnable_over_core_max", "step_active_over_core_p90", "step_active_over_core_max",
        "step_involuntary_ctx_switch_delta", "step_effective_cpu_cores", "available_cores", "step_wall_ms",
        "step_runnable_thread_count_p90", "step_runnable_thread_count_max",
        "step_active_thread_count_sample_p90", "step_active_thread_count_sample_max",
    ]:
        if col in step.columns:
            step[col] = pd.to_numeric(step[col], errors="coerce")
    step["contention_reasons"] = step.apply(_derive_contention_reasons, axis=1)
    mask = pd.Series(False, index=step.index)
    if "step_contention_flag" in step.columns:
        mask |= _safe_numeric(step, "step_contention_flag") > 0
    mask |= _safe_numeric(step, "step_runnable_over_core_p90") > 0
    mask |= _safe_numeric(step, "step_active_over_core_p90") > 0
    mask |= _safe_numeric(step, "step_cpu_utilization_ratio") >= CPU_UTIL_THRESHOLD
    if SAVE_ALL_STEPS:
        events = step.copy()
        events["event_filter_note"] = "all_steps_saved"
    else:
        events = step[mask].copy()
        if events.empty:
            events = step.copy()
            events["event_filter_note"] = "no_contention_event_detected_showing_all_steps"
        else:
            events["event_filter_note"] = "contention_suspected"
    return events


def _mode_string(s: pd.Series) -> str:
    vals = [str(x) for x in s.dropna().tolist() if str(x).strip() and str(x).lower() != "nan"]
    if not vals:
        return ""
    return pd.Series(vals).mode().iloc[0]


def _join_unique(values, limit=12) -> str:
    vals = []
    for x in values:
        sx = str(x).strip()
        if sx and sx.lower() != "nan" and sx not in vals:
            vals.append(sx)
    return "|".join(vals[:limit])


def build_thread_identity_outputs_for_repeat(repeat_id: int, repeat_dir: Path):
    active_dir = repeat_dir / "active"
    step_csv = active_dir / "active_step_metrics.csv"
    req_detail_csv = active_dir / "active_request_step_detail.csv"
    thread_detail_csv = active_dir / "active_thread_contention_detail.csv"

    if not step_csv.exists():
        raise FileNotFoundError(f"Missing step metrics: {step_csv}")
    if not thread_detail_csv.exists():
        raise FileNotFoundError(f"Missing thread identity details: {thread_detail_csv}")

    step = pd.read_csv(step_csv)
    req_detail = pd.read_csv(req_detail_csv) if req_detail_csv.exists() else pd.DataFrame()
    loc = _build_step_location_table(req_detail)
    events = _build_event_steps(step, loc)

    detail = pd.read_csv(thread_detail_csv)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Restrict to contention steps by default. This retains every runnable/active
    # thread sample inside those steps, not just top-N strings.
    event_steps = set(pd.to_numeric(events["global_step"], errors="coerce").dropna().astype(int).tolist())
    detail["global_step_num"] = pd.to_numeric(detail["global_step"], errors="coerce")
    if not SAVE_ALL_STEPS:
        detail = detail[detail["global_step_num"].astype("Int64").isin(event_steps)].copy()

    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Merge concrete request/iteration location and step-level pressure metrics.
    merge_cols = [
        "global_step", "logical_location", "step_stage", "pair_ids_in_step",
        "decode_iter_min", "decode_iter_max", "decode_iter_indices", "request_location_detail",
    ]
    if not loc.empty:
        detail = detail.merge(loc[merge_cols], on="global_step", how="left")
    else:
        for c in merge_cols[1:]:
            detail[c] = ""

    step_keep = [
        "global_step", "min_batch_barrier_enabled", "min_batch_barrier_observed_ms",
        "min_batch_barrier_actual_requests", "latency_baseline_reset_after_add_loop",
        "step_wall_ms", "available_cores", "step_effective_cpu_cores",
        "step_cpu_utilization_ratio", "step_runnable_thread_count_p90", "step_runnable_thread_count_max",
        "step_runnable_over_core_p90", "step_runnable_over_core_max", "step_active_over_core_p90",
        "step_active_over_core_max", "step_involuntary_ctx_switch_delta", "step_voluntary_ctx_switch_delta",
    ]
    step_keep = [c for c in step_keep if c in events.columns]
    detail = detail.merge(events[step_keep + ["contention_reasons", "event_filter_note"]], on="global_step", how="left")

    detail.insert(0, "repeat_id", repeat_id)

    # Stable numeric columns.
    for c in [
        "is_runnable", "is_active_by_cpu_delta", "cpu_delta_ms_since_prev_sample",
        "involuntary_ctx_switch_delta_since_prev_sample", "voluntary_ctx_switch_delta_since_prev_sample",
        "sample_runnable_thread_count", "sample_runnable_over_core", "sample_active_thread_count", "sample_active_over_core",
    ]:
        if c in detail.columns:
            detail[c] = pd.to_numeric(detail[c], errors="coerce").fillna(0)

    # Most useful column order.
    detail_cols = [
        "repeat_id", "global_step", "sample_idx", "t_rel_ms", "logical_location", "step_stage",
        "pair_ids_in_step", "decode_iter_indices", "request_location_detail",
        "available_cores_x", "available_cores_y", "sample_runnable_thread_count", "sample_runnable_over_core",
        "sample_active_thread_count", "sample_active_over_core", "step_cpu_utilization_ratio",
        "step_runnable_over_core_p90", "step_active_over_core_p90", "contention_reasons",
        "pid", "tid", "ppid", "process_role", "process_name", "thread_name", "kernel_thread_comm",
        "python_thread_name", "state", "is_runnable", "is_active_by_cpu_delta",
        "cpu_delta_ms_since_prev_sample", "involuntary_ctx_switch_delta_since_prev_sample",
        "voluntary_ctx_switch_delta_since_prev_sample", "wchan", "last_cpu", "reason_flags", "process_cmdline",
    ]
    # Resolve duplicate available_cores column names after merge.
    if "available_cores_x" in detail.columns and "available_cores" not in detail.columns:
        detail["available_cores"] = detail["available_cores_x"]
    if "available_cores_y" in detail.columns:
        detail["step_available_cores"] = detail["available_cores_y"]
    detail_cols = [c for c in detail_cols if c in detail.columns]
    extra_cols = [c for c in detail.columns if c not in detail_cols and c != "global_step_num"]
    detail = detail[detail_cols + extra_cols]
    detail = detail.sort_values(["repeat_id", "global_step", "sample_idx", "is_runnable", "cpu_delta_ms_since_prev_sample"], ascending=[True, True, True, False, False])

    # Per concrete TID summary.
    tid_group = [
        "repeat_id", "pid", "tid", "process_role", "process_name", "thread_name", "kernel_thread_comm", "python_thread_name",
    ]
    tid_group = [c for c in tid_group if c in detail.columns]
    by_tid = detail.groupby(tid_group, dropna=False).agg(
        first_global_step=("global_step", "min"),
        last_global_step=("global_step", "max"),
        unique_step_count=("global_step", pd.Series.nunique),
        sample_rows=("sample_idx", "count"),
        runnable_samples=("is_runnable", "sum"),
        active_samples=("is_active_by_cpu_delta", "sum"),
        total_cpu_delta_ms=("cpu_delta_ms_since_prev_sample", "sum"),
        max_cpu_delta_ms_per_sample=("cpu_delta_ms_since_prev_sample", "max"),
        involuntary_ctx_switch_delta_sum=("involuntary_ctx_switch_delta_since_prev_sample", "sum"),
        voluntary_ctx_switch_delta_sum=("voluntary_ctx_switch_delta_since_prev_sample", "sum"),
        common_wchan=("wchan", _mode_string),
        locations=("logical_location", _join_unique),
        reason_flags=("reason_flags", _join_unique),
        process_cmdline=("process_cmdline", _first_non_empty),
    ).reset_index()
    by_tid = by_tid.sort_values(["runnable_samples", "total_cpu_delta_ms", "involuntary_ctx_switch_delta_sum"], ascending=False)

    # Cross-repeat/type summary: useful because PIDs/TIDs change every repeat.
    type_group = ["logical_location", "process_role", "process_name", "thread_name", "kernel_thread_comm", "python_thread_name"]
    type_group = [c for c in type_group if c in detail.columns]
    by_type = detail.groupby(type_group, dropna=False).agg(
        repeat_count=("repeat_id", pd.Series.nunique),
        unique_step_count=("global_step", pd.Series.nunique),
        sample_rows=("sample_idx", "count"),
        runnable_samples=("is_runnable", "sum"),
        active_samples=("is_active_by_cpu_delta", "sum"),
        total_cpu_delta_ms=("cpu_delta_ms_since_prev_sample", "sum"),
        max_cpu_delta_ms_per_sample=("cpu_delta_ms_since_prev_sample", "max"),
        involuntary_ctx_switch_delta_sum=("involuntary_ctx_switch_delta_since_prev_sample", "sum"),
        voluntary_ctx_switch_delta_sum=("voluntary_ctx_switch_delta_since_prev_sample", "sum"),
        common_wchan=("wchan", _mode_string),
        example_pid_tid=("tid", lambda s: str(s.iloc[0]) if len(s) else ""),
        reason_flags=("reason_flags", _join_unique),
    ).reset_index()
    by_type = by_type.sort_values(["runnable_samples", "total_cpu_delta_ms", "involuntary_ctx_switch_delta_sum"], ascending=False)

    # Per-repeat saves for quick inspection if repeat dirs are kept.
    detail_path = repeat_dir / "contention_runnable_thread_details.csv"
    by_tid_path = repeat_dir / "contention_runnable_thread_summary_by_tid.csv"
    detail.to_csv(detail_path, index=False)
    by_tid.to_csv(by_tid_path, index=False)
    print(f"[DONE] repeat thread detail saved: {detail_path}")

    return detail, by_tid, by_type


def _print_preview(df: pd.DataFrame):
    if df.empty:
        print("[INFO] No runnable/active thread rows captured.")
        return
    cols = [
        "repeat_id", "global_step", "sample_idx", "logical_location", "sample_runnable_over_core",
        "pid", "tid", "process_role", "thread_name", "kernel_thread_comm", "state", "is_runnable",
        "cpu_delta_ms_since_prev_sample", "involuntary_ctx_switch_delta_since_prev_sample", "wchan", "reason_flags",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 160)
    print("[RUNNABLE / ACTIVE THREAD PREVIEW]")
    print(df[cols].head(40).to_string(index=False))
    print("=" * 160)


def main():
    if RESET and BASE_PROFILE_DIR.exists():
        shutil.rmtree(BASE_PROFILE_DIR)
    BASE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("[CONFIG]")
    print(f"BASE_PROFILE_DIR = {BASE_PROFILE_DIR}")
    print(f"WORKER_SCRIPT    = {WORKER_SCRIPT}")
    print(f"PROMPT_SPECS     = {PROMPT_SPECS_PATH}")
    print(f"NUM_REPEATS      = {NUM_REPEATS}")
    print(f"SAVE_ALL_STEPS   = {SAVE_ALL_STEPS}")
    print(f"KEEP_RAW         = {KEEP_RAW}")
    print(f"SITECUSTOMIZE_SOURCE = {SITECUSTOMIZE_SOURCE}")
    print(f"MIN_BATCH_BARRIER_MS = {MIN_BATCH_BARRIER_MS}")
    print(f"MIN_BATCH_TARGET_REQUESTS = {MIN_BATCH_TARGET_REQUESTS or 'NUM_REQUESTS'}")
    print(f"EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = {EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY}")
    print(f"RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = {RESET_LATENCY_BASELINE_AFTER_ADD_LOOP}")
    print(f"ENABLE_SCHEDULER_MIN_BATCH_HOLD = {ENABLE_SCHEDULER_MIN_BATCH_HOLD}")
    print(f"SCHEDULER_MIN_BATCH_TARGET_REQUESTS = {SCHEDULER_MIN_BATCH_TARGET_REQUESTS or 'MIN_BATCH_TARGET_REQUESTS/NUM_REQUESTS'}")
    print(f"SCHEDULER_MIN_BATCH_HOLD_MS = {SCHEDULER_MIN_BATCH_HOLD_MS}")
    print("=" * 100)

    sitecustomize_import_dir = _prepare_sitecustomize_import_dir()

    all_details = []
    all_by_tid = []
    all_by_type = []

    for repeat_id in range(NUM_REPEATS):
        repeat_dir = BASE_PROFILE_DIR / f"repeat_{repeat_id:03d}"
        active_dir = repeat_dir / "active"
        active_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[REPEAT START] {repeat_id + 1}/{NUM_REPEATS}")
        print(f"[REPEAT DIR]   {repeat_dir}")
        print("#" * 100)

        run_active_phase_subprocess(
            repeat_id=repeat_id,
            profile_dir=active_dir,
            sitecustomize_import_dir=sitecustomize_import_dir,
        )
        detail, by_tid, by_type = build_thread_identity_outputs_for_repeat(repeat_id, repeat_dir)
        if not detail.empty:
            all_details.append(detail)
        if not by_tid.empty:
            all_by_tid.append(by_tid)
        if not by_type.empty:
            all_by_type.append(by_type)

        if not KEEP_RAW:
            # Keep only the compact per-repeat thread identity outputs.
            keep_files = []
            keep_names = [
                "contention_runnable_thread_details.csv",
                "contention_runnable_thread_summary_by_tid.csv",
                "active/scheduler_min_batch_hold_metrics.csv",
            ]
            for idx, name in enumerate(keep_names):
                p = repeat_dir / name
                if p.exists():
                    final = repeat_dir / Path(name).name
                    tmp = repeat_dir / f".__keep_{idx}_{Path(name).name}.tmp"
                    shutil.copy2(p, tmp)
                    keep_files.append((tmp, final))
            if active_dir.exists():
                shutil.rmtree(active_dir)
            for tmp, final in keep_files:
                shutil.move(str(tmp), str(final))

    final_detail = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    final_by_tid = pd.concat(all_by_tid, ignore_index=True) if all_by_tid else pd.DataFrame()
    final_by_type_raw = pd.concat(all_by_type, ignore_index=True) if all_by_type else pd.DataFrame()

    detail_path = BASE_PROFILE_DIR / "contention_runnable_thread_details.csv"
    by_tid_path = BASE_PROFILE_DIR / "contention_runnable_thread_summary_by_tid.csv"
    by_type_path = BASE_PROFILE_DIR / "contention_runnable_thread_summary_by_type.csv"

    final_detail.to_csv(detail_path, index=False)
    final_by_tid.to_csv(by_tid_path, index=False)

    # Re-aggregate type summary across repeats because per-repeat type summaries are partial.
    if not final_detail.empty:
        type_group = ["logical_location", "process_role", "process_name", "thread_name", "kernel_thread_comm", "python_thread_name"]
        type_group = [c for c in type_group if c in final_detail.columns]
        final_by_type = final_detail.groupby(type_group, dropna=False).agg(
            repeat_count=("repeat_id", pd.Series.nunique),
            unique_step_count=("global_step", pd.Series.nunique),
            sample_rows=("sample_idx", "count"),
            runnable_samples=("is_runnable", "sum"),
            active_samples=("is_active_by_cpu_delta", "sum"),
            total_cpu_delta_ms=("cpu_delta_ms_since_prev_sample", "sum"),
            max_cpu_delta_ms_per_sample=("cpu_delta_ms_since_prev_sample", "max"),
            involuntary_ctx_switch_delta_sum=("involuntary_ctx_switch_delta_since_prev_sample", "sum"),
            voluntary_ctx_switch_delta_sum=("voluntary_ctx_switch_delta_since_prev_sample", "sum"),
            common_wchan=("wchan", _mode_string),
            locations=("logical_location", _join_unique),
            reason_flags=("reason_flags", _join_unique),
        ).reset_index().sort_values(["runnable_samples", "total_cpu_delta_ms", "involuntary_ctx_switch_delta_sum"], ascending=False)
    else:
        final_by_type = pd.DataFrame()
    final_by_type.to_csv(by_type_path, index=False)

    print("\n" + "=" * 120)
    print("[DONE] Runnable thread identity outputs saved")
    print(f"  - {detail_path}")
    print(f"  - {by_tid_path}")
    print(f"  - {by_type_path}")
    print("=" * 120)
    _print_preview(final_detail)


if __name__ == "__main__":
    main()
