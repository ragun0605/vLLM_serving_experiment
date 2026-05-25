#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Config & Paths
# ============================================================

BASE_PROFILE_DIR = Path(
    os.environ.get("VLLM_BASE_PROFILE_DIR", "./two_phase_profile_logs")
)

WORKER_SCRIPT = Path(
    os.environ.get("VLLM_PHASE_WORKER", "./phase_worker.py")
).resolve()

RESET = os.environ.get("VLLM_RESET_PROFILE_FILES", "1").lower() not in {
    "0",
    "false",
    "no",
}

NUM_REPEATS = int(os.environ.get("VLLM_NUM_REPEATS", "5"))
MIN_BATCH_BARRIER_MS = os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "0")
MIN_BATCH_TARGET_REQUESTS = os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", "")
EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = os.environ.get(
    "VLLM_EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY",
    "1",
)
RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = os.environ.get(
    "VLLM_RESET_LATENCY_BASELINE_AFTER_ADD_LOOP",
    "1",
)
ENABLE_SCHEDULER_MIN_BATCH_HOLD = os.environ.get(
    "VLLM_ENABLE_SCHEDULER_MIN_BATCH_HOLD",
    "1",
)
SCHEDULER_MIN_BATCH_HOLD_MS = os.environ.get(
    "VLLM_SCHEDULER_MIN_BATCH_HOLD_MS",
    os.environ.get("VLLM_MIN_BATCH_BARRIER_MS", "200"),
)
SCHEDULER_MIN_BATCH_TARGET_REQUESTS = os.environ.get(
    "VLLM_SCHEDULER_MIN_BATCH_TARGET_REQUESTS",
    os.environ.get("VLLM_MIN_BATCH_TARGET_REQUESTS", ""),
)


PROMPT_SPECS_PATH = BASE_PROFILE_DIR / "prompt_specs.json"


# ============================================================
# Requested output columns
# ============================================================
REQUEST_OUTPUT_COLUMNS = [
    # scenario / request identity
    "repeat_id",
    "pair_id",
    "active_request_id",
    "gpu_request_id",

    # Min-batch barrier diagnostics
    "min_batch_barrier_enabled_phase_b",
    "min_batch_barrier_requested_ms_phase_b",
    "min_batch_barrier_observed_ms_phase_b",
    "min_batch_barrier_target_requests_phase_b",
    "min_batch_barrier_actual_requests_phase_b",
    "min_batch_barrier_excluded_from_latency_phase_b",
    "latency_baseline_reset_after_add_loop_phase_b",
    "first_non_warmup_iter_rows",
    "first_non_warmup_num_requests_min",
    "first_non_warmup_num_requests_max",
    "first_non_warmup_min_batch_target",
    "first_non_warmup_min_batch_ok_all_ranks",
    "raw_to_latency_submit_gap_ms_phase_b",
    "raw_e2e_ms_including_barrier_phase_b",
    "raw_queue_ms_including_barrier_phase_b",
    "raw_e2ft_ms_including_barrier_phase_b",
    "ttft_actual_raw_including_barrier_ms_phase_b",

    # Phase B request-level latency
    "e2e_ms_phase_b",
    "queue_ms_phase_b",
    "ttft_ms_phase_b",
    "tpot_ms_phase_b",
    "tpot_token_count",

    # Phase B TTFT diagnostics
    "ttft_actual_ms_phase_b",
    "ttft_compute_window_ms_phase_b",
    "ttft_request_observed_ms_phase_b",
    "e2ft_request_observed_ms_phase_b",
    "tpot_request_observed_ms_phase_b",

    # Prefill time
    "prefill_time_ms_phase_b",
    "prefill_gpu_ms_phase_b",
    "prefill_non_gpu_wall_ms",
    "prefill_non_gpu_wall_ratio",
    "prefill_non_gpu_wall_percent",

    # Non-GPU time / ratio for E2E, TTFT, TPOT
    "e2e_non_gpu_wall_ms",
    "e2e_non_gpu_wall_ratio",
    "e2e_non_gpu_wall_percent",
    "ttft_non_gpu_wall_ms",
    "ttft_non_gpu_wall_ratio",
    "ttft_non_gpu_wall_percent",
    "tpot_non_gpu_wall_ms",
    "tpot_non_gpu_wall_ratio",
    "tpot_non_gpu_wall_percent",

    # Token sanity checks
    "target_input_tokens",
    "input_tokens",
    "input_tokens_ok",
    "target_output_tokens",
    "output_tokens",
    "output_tokens_ok",

    # GPU contribution diagnostics
    "e2e_gpu_ms",
    "ttft_gpu_ms",
    "tpot_gpu_ms",
    "tpot_gpu_total_ms",

    # Stage-level active thread metrics from Phase A
    "prefill_active_thread_count",
    "prefill_active_thread_count_edge",
    "prefill_active_thread_count_sample_p90",
    "prefill_active_thread_count_union",
    "prefill_effective_cpu_cores",
    "decode_active_thread_count_avg",
    "decode_active_thread_count_edge_avg",
    "decode_active_thread_count_sample_p90_avg",
    "decode_active_thread_count_sample_max_peak",
    "decode_active_thread_count_union_avg",
    "decode_effective_cpu_cores_avg",
    "decode_effective_cpu_cores_p90",
    "decode_effective_cpu_cores_max",

    # Optional stage-level total-thread diagnostics
    "prefill_total_threads_mean",
    "decode_total_threads_mean_avg",
    "decode_total_threads_max_peak",

    # Optional process CPU diagnostics from Phase A
    "prefill_process_cpu_delta_s",
    "decode_process_cpu_delta_s_avg",
    "active_decode_process_cpu_delta_total_s_est",
    "active_total_process_cpu_delta_s_est",
    "active_total_process_cpu_core_ms_est",

    # Iteration / stage sanity checks
    "active_num_decode_steps",
    "active_decode_steps_expected",
    "active_decode_steps_match",
    "gpu_num_prefill_iters",
    "gpu_num_decode_iters",
    "gpu_decode_iters_expected",
    "gpu_decode_iters_match",
    "gpu_stage_classification_method",
    "all_iters_have_expected_tp_rows",
]

REQUEST_SUMMARY_METRICS = [
    "e2e_ms_phase_b",
    "ttft_ms_phase_b",
    "tpot_ms_phase_b",
    "prefill_time_ms_phase_b",
    "ttft_compute_window_ms_phase_b",
    "e2e_non_gpu_wall_ms",
    "ttft_non_gpu_wall_ms",
    "tpot_non_gpu_wall_ms",
    "prefill_non_gpu_wall_ms",
    "e2e_non_gpu_wall_ratio",
    "ttft_non_gpu_wall_ratio",
    "tpot_non_gpu_wall_ratio",
    "prefill_non_gpu_wall_ratio",
    "queue_ms_phase_b",
    "min_batch_barrier_observed_ms_phase_b",
    "first_non_warmup_num_requests_max",
    "first_non_warmup_min_batch_ok_all_ranks",
    "raw_to_latency_submit_gap_ms_phase_b",
    "raw_e2e_ms_including_barrier_phase_b",
    "raw_e2ft_ms_including_barrier_phase_b",
    "prefill_active_thread_count",
    "prefill_active_thread_count_edge",
    "prefill_active_thread_count_sample_p90",
    "prefill_active_thread_count_union",
    "prefill_effective_cpu_cores",
    "decode_active_thread_count_avg",
    "decode_active_thread_count_edge_avg",
    "decode_active_thread_count_sample_p90_avg",
    "decode_active_thread_count_sample_max_peak",
    "decode_active_thread_count_union_avg",
    "decode_effective_cpu_cores_avg",
    "decode_effective_cpu_cores_p90",
    "decode_effective_cpu_cores_max",
    "prefill_total_threads_mean",
    "decode_total_threads_mean_avg",
    "decode_total_threads_max_peak",
]




def _existing_columns(df: pd.DataFrame, columns):
    return [c for c in columns if c in df.columns]


def _add_percent_columns(df: pd.DataFrame):
    """Add human-readable percent columns next to numeric ratio columns."""
    percent_specs = [
        ("e2e_non_gpu_wall_ratio", "e2e_non_gpu_wall_percent"),
        ("ttft_non_gpu_wall_ratio", "ttft_non_gpu_wall_percent"),
        ("tpot_non_gpu_wall_ratio", "tpot_non_gpu_wall_percent"),
        ("prefill_non_gpu_wall_ratio", "prefill_non_gpu_wall_percent"),

        # Backward-compatible names used by older summaries.
        ("e2e_non_gpu_wall_ratio", "non_gpu_ratio_e2e_percent"),
        ("ttft_non_gpu_wall_ratio", "non_gpu_ratio_ttft_percent"),
        ("tpot_non_gpu_wall_ratio", "non_gpu_ratio_tpot_percent"),

        # Optional / older broad naming.
        ("e2e_broad_non_gpu_wall_ratio", "broad_non_gpu_ratio_e2e_percent"),
        ("ttft_broad_non_gpu_wall_ratio", "broad_non_gpu_ratio_ttft_percent"),
        ("tpot_broad_non_gpu_wall_ratio", "broad_non_gpu_ratio_tpot_percent"),
        (
            "ttft_compute_window_non_gpu_wall_ratio",
            "compute_window_non_gpu_ratio_ttft_percent",
        ),
        (
            "e2e_outside_model_runner_wall_ratio",
            "outside_model_runner_ratio_e2e_percent",
        ),
        ("e2e_model_runner_wall_ratio", "model_runner_ratio_e2e_percent"),
        ("e2e_iteration_gpu_wall_ratio", "iteration_gpu_ratio_percent"),
        ("e2e_model_runner_residual_ratio", "model_runner_residual_ratio_percent"),
    ]

    for ratio_col, percent_col in percent_specs:
        if ratio_col in df.columns:
            df[percent_col] = (
                pd.to_numeric(df[ratio_col], errors="coerce") * 100.0
            ).round(1).astype("string") + "%"

    return df


def select_request_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return compact request-level output requested for the final CSV."""
    keep = _existing_columns(df, REQUEST_OUTPUT_COLUMNS)
    return df[keep].copy()

# ============================================================
# Utility
# ============================================================

def _prepend_pythonpath(env, paths):
    old_pythonpath = env.get("PYTHONPATH", "")
    unique_paths = []

    for p in paths:
        p = str(Path(p).resolve())
        if p not in unique_paths:
            unique_paths.append(p)

    if old_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(unique_paths + [old_pythonpath])
    else:
        env["PYTHONPATH"] = os.pathsep.join(unique_paths)


def run_subprocess(
    phase: str,
    profile_dir: Path,
    disable_patch: bool,
    repeat_id: int,
):
    env = os.environ.copy()

    env["VLLM_PROFILE_DIR"] = str(profile_dir)
    env["VLLM_PROMPT_SPECS_PATH"] = str(PROMPT_SPECS_PATH)
    env["VLLM_DISABLE_GPU_PROFILE_PATCH"] = "1" if disable_patch else "0"
    env["VLLM_REPEAT_ID"] = str(repeat_id)

    _prepend_pythonpath(
        env,
        paths=[
            Path.cwd().resolve(),
            WORKER_SCRIPT.parent.resolve(),
        ],
    )

    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--phase",
        phase,
    ]

    print("\n" + "=" * 100)
    print(f"[RUN] repeat_id={repeat_id}")
    print(f"[RUN] phase={phase}")
    print(f"[RUN] profile_dir={profile_dir}")
    print(f"[RUN] disable_patch={disable_patch}")
    print(f"[RUN] worker_script={WORKER_SCRIPT}")
    print(f"[RUN] prompt_specs={PROMPT_SPECS_PATH}")
    print("=" * 100)

    subprocess.run(cmd, env=env, check=True)


# ============================================================
# One repeat merge
# ============================================================

def merge_one_repeat_results(
    repeat_id: int,
    repeat_dir: Path,
):
    active_dir = repeat_dir / "active"
    gpu_dir = repeat_dir / "gpu"

    active_csv = active_dir / "active_request_metrics_measured.csv"
    gpu_csv = gpu_dir / "gpu_request_metrics_measured.csv"

    if not active_csv.exists():
        raise FileNotFoundError(f"Missing active summary: {active_csv}")

    if not gpu_csv.exists():
        raise FileNotFoundError(f"Missing gpu summary: {gpu_csv}")

    active = pd.read_csv(active_csv)
    gpu = pd.read_csv(gpu_csv)

    active_keep = [
        "pair_id",
        "request_id",

        "min_batch_barrier_enabled",
        "min_batch_barrier_requested_ms",
        "min_batch_barrier_observed_ms",
        "min_batch_barrier_target_requests",
        "min_batch_barrier_actual_requests",
        "min_batch_barrier_excluded_from_latency",
        "raw_to_latency_submit_gap_ms",
        "raw_e2e_ms_including_barrier",
        "raw_queue_ms_including_barrier",
        "raw_e2ft_ms_including_barrier",

        "prefill_active_thread_count",
        "prefill_active_thread_count_edge",
        "prefill_active_thread_count_sample_p90",
        "prefill_active_thread_count_union",
        "prefill_effective_cpu_cores",
        "decode_active_thread_count_avg",
        "decode_active_thread_count_edge_avg",
        "decode_active_thread_count_sample_p90_avg",
        "decode_active_thread_count_sample_max_peak",
        "decode_active_thread_count_union_avg",
        "decode_effective_cpu_cores_avg",
        "decode_effective_cpu_cores_p90",
        "decode_effective_cpu_cores_max",
        "prefill_total_threads_mean",
        "decode_total_threads_mean_avg",
        "decode_total_threads_max_peak",
        "prefill_process_cpu_delta_s",
        "decode_process_cpu_delta_s_avg",

        "num_decode_steps",
        "decode_steps_expected",
        "decode_steps_match",
    ]

    gpu_keep = [
        "pair_id",
        "request_id",

        # min-batch barrier diagnostics
        "min_batch_barrier_enabled",
        "min_batch_barrier_requested_ms",
        "min_batch_barrier_observed_ms",
        "min_batch_barrier_target_requests",
        "min_batch_barrier_actual_requests",
        "min_batch_barrier_excluded_from_latency",
        "first_non_warmup_iter_rows",
        "first_non_warmup_num_requests_min",
        "first_non_warmup_num_requests_max",
        "first_non_warmup_min_batch_target",
        "first_non_warmup_min_batch_ok_all_ranks",
        "raw_to_latency_submit_gap_ms",
        "raw_e2e_ms_including_barrier",
        "raw_queue_ms_including_barrier",
        "raw_e2ft_ms_including_barrier",

        # request-level latency
        "e2e_ms",
        "queue_ms",
        "ttft_ms",
        "tpot_ms",
        "tpot_token_count",

        # actual TTFT / prefill diagnostics
        "ttft_actual_ms",
        "ttft_actual_raw_including_barrier_ms",
        "ttft_compute_window_ms",
        "prefill_time_ms",
        "prefill_gpu_ms",
        "prefill_non_gpu_wall_ms",
        "prefill_non_gpu_wall_ratio",

        # request-level diagnostic
        "ttft_worker_ms",
        "tpot_worker_ms",
        "e2e_worker_ms",
        "ttft_request_observed_ms",
        "e2ft_request_observed_ms",
        "tpot_request_observed_ms",

        # token checks
        "target_input_tokens",
        "input_tokens",
        "input_tokens_ok",
        "target_output_tokens",
        "output_tokens",
        "output_tokens_ok",

        # GPU contribution
        "e2e_gpu_ms",
        "ttft_gpu_ms",
        "tpot_gpu_total_ms",
        "tpot_gpu_ms",

        # broad non-GPU
        "e2e_broad_non_gpu_wall_ms",
        "ttft_broad_non_gpu_wall_ms",
        "ttft_compute_window_non_gpu_wall_ms",
        "tpot_broad_non_gpu_wall_ms",

        "e2e_broad_non_gpu_wall_ratio",
        "ttft_broad_non_gpu_wall_ratio",
        "ttft_compute_window_non_gpu_wall_ratio",
        "tpot_broad_non_gpu_wall_ratio",

        # compatibility non-GPU names
        "e2e_non_gpu_wall_ms",
        "ttft_non_gpu_wall_ms",
        "tpot_non_gpu_wall_ms",
        "e2e_non_gpu_wall_ratio",
        "ttft_non_gpu_wall_ratio",
        "tpot_non_gpu_wall_ratio",

        # model-runner residual diagnostic
        "e2e_iteration_wall_ms",
        "e2e_model_runner_residual_ms",
        "ttft_model_runner_wall_ms",
        "ttft_model_runner_residual_ms",
        "tpot_model_runner_wall_ms",
        "tpot_model_runner_residual_ms",

        # compatibility diagnostic
        "e2e_iteration_non_gpu_wall_ms",
        "ttft_prefill_wall_ms",
        "ttft_prefill_non_gpu_wall_ms",
        "tpot_decode_wall_total_ms",
        "tpot_decode_gpu_total_ms",
        "tpot_decode_non_gpu_wall_total_ms",

        # outside model-runner diagnostic
        "e2e_outside_model_runner_wall_ms",
        "e2e_outside_model_runner_wall_ms_raw",
        "e2e_outside_model_runner_wall_ratio",
        "e2e_model_runner_wall_ratio",
        "e2e_iteration_gpu_wall_ratio",
        "e2e_model_runner_residual_ratio",
        "e2e_iteration_residual_non_gpu_wall_ratio",

        # sanity flags
        "e2e_gpu_exceeds_worker_e2e",
        "ttft_gpu_exceeds_worker_ttft",
        "tpot_gpu_exceeds_worker_tpot",
        "e2e_model_runner_exceeds_e2e",
        "ttft_gpu_exceeds_prefill_wall",
        "tpot_gpu_exceeds_decode_wall",

        # iteration / grouping
        "gpu_num_logical_iterations",
        "gpu_num_prefill_iters",
        "gpu_num_decode_iters",
        "gpu_decode_iters_expected",
        "gpu_decode_iters_match",
        "gpu_stage_classification_method",
        "all_iters_have_expected_tp_rows",
        "min_worker_rows_per_iter",
        "max_worker_rows_per_iter",
        "max_rank_start_skew_ms",
        "max_rank_end_skew_ms",
    ]

    active_keep = [c for c in active_keep if c in active.columns]
    gpu_keep = [c for c in gpu_keep if c in gpu.columns]

    active_small = active[active_keep].rename(
        columns={
            "request_id": "active_request_id",
            "min_batch_barrier_enabled": "min_batch_barrier_enabled_phase_a",
            "min_batch_barrier_requested_ms": "min_batch_barrier_requested_ms_phase_a",
            "min_batch_barrier_observed_ms": "min_batch_barrier_observed_ms_phase_a",
            "min_batch_barrier_target_requests": "min_batch_barrier_target_requests_phase_a",
            "min_batch_barrier_actual_requests": "min_batch_barrier_actual_requests_phase_a",
            "min_batch_barrier_excluded_from_latency": "min_batch_barrier_excluded_from_latency_phase_a",
            "raw_to_latency_submit_gap_ms": "raw_to_latency_submit_gap_ms_phase_a",
            "raw_e2e_ms_including_barrier": "raw_e2e_ms_including_barrier_phase_a",
            "raw_queue_ms_including_barrier": "raw_queue_ms_including_barrier_phase_a",
            "raw_e2ft_ms_including_barrier": "raw_e2ft_ms_including_barrier_phase_a",
            "num_decode_steps": "active_num_decode_steps",
            "decode_steps_expected": "active_decode_steps_expected",
            "decode_steps_match": "active_decode_steps_match",
        }
    )

    gpu_small = gpu[gpu_keep].rename(
        columns={
            "request_id": "gpu_request_id",
            "min_batch_barrier_enabled": "min_batch_barrier_enabled_phase_b",
            "min_batch_barrier_requested_ms": "min_batch_barrier_requested_ms_phase_b",
            "min_batch_barrier_observed_ms": "min_batch_barrier_observed_ms_phase_b",
            "min_batch_barrier_target_requests": "min_batch_barrier_target_requests_phase_b",
            "min_batch_barrier_actual_requests": "min_batch_barrier_actual_requests_phase_b",
            "min_batch_barrier_excluded_from_latency": "min_batch_barrier_excluded_from_latency_phase_b",
            "raw_to_latency_submit_gap_ms": "raw_to_latency_submit_gap_ms_phase_b",
            "raw_e2e_ms_including_barrier": "raw_e2e_ms_including_barrier_phase_b",
            "raw_queue_ms_including_barrier": "raw_queue_ms_including_barrier_phase_b",
            "raw_e2ft_ms_including_barrier": "raw_e2ft_ms_including_barrier_phase_b",
            "e2e_ms": "e2e_ms_phase_b",
            "queue_ms": "queue_ms_phase_b",
            "ttft_ms": "ttft_ms_phase_b",
            "tpot_ms": "tpot_ms_phase_b",
            "ttft_actual_ms": "ttft_actual_ms_phase_b",
            "ttft_actual_raw_including_barrier_ms": "ttft_actual_raw_including_barrier_ms_phase_b",
            "ttft_compute_window_ms": "ttft_compute_window_ms_phase_b",
            "prefill_time_ms": "prefill_time_ms_phase_b",
            "prefill_gpu_ms": "prefill_gpu_ms_phase_b",

            "ttft_request_observed_ms": "ttft_request_observed_ms_phase_b",
            "e2ft_request_observed_ms": "e2ft_request_observed_ms_phase_b",
            "tpot_request_observed_ms": "tpot_request_observed_ms_phase_b",
        }
    )

    final = active_small.merge(
        gpu_small,
        on="pair_id",
        how="inner",
    )

    final.insert(0, "repeat_id", repeat_id)

    # ============================================================
    # Active phase process CPU/core-time diagnostic
    # ============================================================

    if (
        "prefill_process_cpu_delta_s" in final.columns
        and "decode_process_cpu_delta_s_avg" in final.columns
        and "active_num_decode_steps" in final.columns
    ):
        prefill_cpu_s = pd.to_numeric(
            final["prefill_process_cpu_delta_s"],
            errors="coerce",
        )

        decode_cpu_s_avg = pd.to_numeric(
            final["decode_process_cpu_delta_s_avg"],
            errors="coerce",
        )

        decode_steps = pd.to_numeric(
            final["active_num_decode_steps"],
            errors="coerce",
        )

        final["active_decode_process_cpu_delta_total_s_est"] = (
            decode_cpu_s_avg * decode_steps
        )

        final["active_total_process_cpu_delta_s_est"] = (
            prefill_cpu_s + final["active_decode_process_cpu_delta_total_s_est"]
        )

        final["active_total_process_cpu_core_ms_est"] = (
            final["active_total_process_cpu_delta_s_est"] * 1000.0
        )

    # ============================================================
    # Percent columns + requested compact final output
    # ============================================================

    final = _add_percent_columns(final)

    # 진단용 전체 컬럼 보존
    final_full_path = repeat_dir / "final_paired_request_metrics_full.csv"
    final.to_csv(final_full_path, index=False)

    # 사용자가 요청한 핵심 컬럼 중심의 최종 request-level CSV
    final_compact = select_request_output_columns(final)
    final_path = repeat_dir / "final_paired_request_metrics.csv"
    final_compact.to_csv(final_path, index=False)

    print(f"[DONE] repeat_id={repeat_id} compact final saved: {final_path}")
    print(f"[DONE] repeat_id={repeat_id} full final saved:    {final_full_path}")

    return final_compact


# ============================================================
# Repeated summary
# ============================================================

def _numeric_series(df: pd.DataFrame, col: str):
    return pd.to_numeric(df[col], errors="coerce").dropna()


def summarize_numeric_columns(
    df: pd.DataFrame,
    columns,
    group_cols=None,
):
    rows = []

    if group_cols is None:
        grouped = [((), df)]
        group_cols = []
    else:
        grouped = list(df.groupby(group_cols, dropna=False))

    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = {
            group_cols[i]: group_key[i]
            for i in range(len(group_cols))
        }

        for col in columns:
            if col not in group_df.columns:
                continue

            s = _numeric_series(group_df, col)

            if s.empty:
                continue

            row = dict(group_info)
            row.update(
                {
                    "metric": col,
                    "count": int(s.count()),
                    "avg": float(s.mean()),
                    "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                    "min": float(s.min()),
                    "median": float(s.median()),
                    "t90": float(s.quantile(0.90)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


def build_overall_average_metrics(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 scenario/repeat에서 얻은 모든 request row를 하나로 모아 평균을 낸다.
    ratio는 두 가지를 모두 남긴다.
      - *_ratio_avg: request별 ratio의 단순 평균
      - *_ratio_from_avg: 평균 non-GPU time / 평균 latency
    """

    def mean_col(col: str):
        if col not in all_df.columns:
            return None
        s = pd.to_numeric(all_df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.mean())

    def ratio_from_means(non_gpu_col: str, total_col: str):
        non_gpu = mean_col(non_gpu_col)
        total = mean_col(total_col)
        if non_gpu is None or total is None or total == 0:
            return None
        return float(non_gpu / total)

    row = {
        "num_rows": int(len(all_df)),
        "num_repeats": int(all_df["repeat_id"].nunique()) if "repeat_id" in all_df.columns else "",
        "num_pair_ids": int(all_df["pair_id"].nunique()) if "pair_id" in all_df.columns else "",

        # Average Phase B latency
        "e2e_ms_phase_b_avg": mean_col("e2e_ms_phase_b"),
        "ttft_ms_phase_b_avg": mean_col("ttft_ms_phase_b"),
        "tpot_ms_phase_b_avg": mean_col("tpot_ms_phase_b"),
        "prefill_time_ms_phase_b_avg": mean_col("prefill_time_ms_phase_b"),
        "ttft_compute_window_ms_phase_b_avg": mean_col("ttft_compute_window_ms_phase_b"),
        "queue_ms_phase_b_avg": mean_col("queue_ms_phase_b"),
        "min_batch_barrier_observed_ms_phase_b_avg": mean_col("min_batch_barrier_observed_ms_phase_b"),
        "raw_to_latency_submit_gap_ms_phase_b_avg": mean_col("raw_to_latency_submit_gap_ms_phase_b"),
        "raw_e2e_ms_including_barrier_phase_b_avg": mean_col("raw_e2e_ms_including_barrier_phase_b"),
        "raw_e2ft_ms_including_barrier_phase_b_avg": mean_col("raw_e2ft_ms_including_barrier_phase_b"),

        # Average non-GPU time
        "e2e_non_gpu_wall_ms_avg": mean_col("e2e_non_gpu_wall_ms"),
        "ttft_non_gpu_wall_ms_avg": mean_col("ttft_non_gpu_wall_ms"),
        "tpot_non_gpu_wall_ms_avg": mean_col("tpot_non_gpu_wall_ms"),
        "prefill_non_gpu_wall_ms_avg": mean_col("prefill_non_gpu_wall_ms"),

        # Mean of per-request ratios
        "e2e_non_gpu_wall_ratio_avg": mean_col("e2e_non_gpu_wall_ratio"),
        "ttft_non_gpu_wall_ratio_avg": mean_col("ttft_non_gpu_wall_ratio"),
        "tpot_non_gpu_wall_ratio_avg": mean_col("tpot_non_gpu_wall_ratio"),
        "prefill_non_gpu_wall_ratio_avg": mean_col("prefill_non_gpu_wall_ratio"),

        # Ratio derived from averaged times
        "e2e_non_gpu_wall_ratio_from_avg": ratio_from_means(
            "e2e_non_gpu_wall_ms",
            "e2e_ms_phase_b",
        ),
        "ttft_non_gpu_wall_ratio_from_avg": ratio_from_means(
            "ttft_non_gpu_wall_ms",
            "ttft_ms_phase_b",
        ),
        "tpot_non_gpu_wall_ratio_from_avg": ratio_from_means(
            "tpot_non_gpu_wall_ms",
            "tpot_ms_phase_b",
        ),
        "prefill_non_gpu_wall_ratio_from_avg": ratio_from_means(
            "prefill_non_gpu_wall_ms",
            "prefill_time_ms_phase_b",
        ),

        # Average stage-level thread counts
        "prefill_active_thread_count_avg": mean_col("prefill_active_thread_count"),
        "decode_active_thread_count_avg": mean_col("decode_active_thread_count_avg"),
        "prefill_total_threads_mean_avg": mean_col("prefill_total_threads_mean"),
        "decode_total_threads_mean_avg": mean_col("decode_total_threads_mean_avg"),
        "decode_total_threads_max_peak_avg": mean_col("decode_total_threads_max_peak"),
    }

    out = pd.DataFrame([row])

    # Human-readable percent columns for ratios.
    for ratio_col in [
        "e2e_non_gpu_wall_ratio_avg",
        "ttft_non_gpu_wall_ratio_avg",
        "tpot_non_gpu_wall_ratio_avg",
        "prefill_non_gpu_wall_ratio_avg",
        "e2e_non_gpu_wall_ratio_from_avg",
        "ttft_non_gpu_wall_ratio_from_avg",
        "tpot_non_gpu_wall_ratio_from_avg",
        "prefill_non_gpu_wall_ratio_from_avg",
    ]:
        if ratio_col in out.columns:
            out[ratio_col.replace("ratio", "percent")] = (
                pd.to_numeric(out[ratio_col], errors="coerce") * 100.0
            ).round(1).astype("string") + "%"

    return out


def build_repeat_summaries(all_df: pd.DataFrame):
    # 요청한 핵심 metric만 반복 통계 대상으로 사용한다.
    summary_metrics = _existing_columns(all_df, REQUEST_SUMMARY_METRICS)

    # 모든 request row를 대상으로 한 metric별 전체 분포 요약
    overall = summarize_numeric_columns(
        all_df,
        columns=summary_metrics,
        group_cols=None,
    )

    # pair_id별, 즉 동일 prompt/request 위치별 5회 반복 결과의 avg/median/t90/max
    by_pair = summarize_numeric_columns(
        all_df,
        columns=summary_metrics,
        group_cols=["pair_id"],
    )

    # repeat/scenario별 요약. 특정 repeat만 튀었는지 확인할 때 사용한다.
    by_repeat = summarize_numeric_columns(
        all_df,
        columns=summary_metrics,
        group_cols=["repeat_id"],
    )

    # 모든 scenario/repeat/request를 하나로 모은 평균 행.
    overall_average = build_overall_average_metrics(all_df)

    overall_path = BASE_PROFILE_DIR / "repeat_summary_overall.csv"
    by_pair_path = BASE_PROFILE_DIR / "repeat_summary_by_pair.csv"
    by_repeat_path = BASE_PROFILE_DIR / "repeat_summary_by_repeat.csv"
    overall_avg_path = BASE_PROFILE_DIR / "overall_request_average_metrics.csv"

    overall.to_csv(overall_path, index=False)
    by_pair.to_csv(by_pair_path, index=False)
    by_repeat.to_csv(by_repeat_path, index=False)
    overall_average.to_csv(overall_avg_path, index=False)

    print("\n" + "=" * 120)
    print("[DONE] Requested repeated-run summaries saved")
    print(f"  - {overall_path}")
    print(f"  - {by_pair_path}        # pair_id별 5회 반복 avg/median/t90/max")
    print(f"  - {by_repeat_path}")
    print(f"  - {overall_avg_path}    # 모든 request 평균 및 평균 기반 ratio")
    print("=" * 120)

    return overall, by_pair, by_repeat, overall_average

def print_key_summary(overall: pd.DataFrame, overall_average: pd.DataFrame = None):
    key_metrics = [
        "e2e_ms_phase_b",
        "ttft_ms_phase_b",
        "tpot_ms_phase_b",
        "prefill_time_ms_phase_b",
        "ttft_compute_window_ms_phase_b",
        "e2e_non_gpu_wall_ms",
        "ttft_non_gpu_wall_ms",
        "tpot_non_gpu_wall_ms",
        "e2e_non_gpu_wall_ratio",
        "ttft_non_gpu_wall_ratio",
        "tpot_non_gpu_wall_ratio",
        "queue_ms_phase_b",
        "min_batch_barrier_observed_ms_phase_b",
        "raw_to_latency_submit_gap_ms_phase_b",
        "prefill_active_thread_count",
        "decode_active_thread_count_avg",
    ]

    show = overall[overall["metric"].isin(key_metrics)].copy()

    if not show.empty:
        preferred_cols = [
            "metric",
            "count",
            "avg",
            "std",
            "median",
            "t90",
            "min",
            "max",
        ]
        show = show[[c for c in preferred_cols if c in show.columns]]

        print("\n" + "=" * 140)
        print("[KEY SUMMARY] all requests across all repeats: avg / median / t90 / max")
        print("-" * 140)
        print(show.to_string(index=False))
        print("=" * 140)

    if overall_average is not None and not overall_average.empty:
        print("\n" + "=" * 140)
        print("[OVERALL AVERAGE ROW] mean latency, mean non-GPU time, and ratio from means")
        print("-" * 140)
        print(overall_average.to_string(index=False))
        print("=" * 140)


# ============================================================
# Main
# ============================================================

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
    print(f"RESET            = {RESET}")
    print(f"MIN_BATCH_BARRIER_MS = {MIN_BATCH_BARRIER_MS}")
    print(f"MIN_BATCH_TARGET_REQUESTS = {MIN_BATCH_TARGET_REQUESTS or 'NUM_REQUESTS'}")
    print(f"EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY = {EXCLUDE_MIN_BATCH_BARRIER_FROM_LATENCY}")
    print(f"RESET_LATENCY_BASELINE_AFTER_ADD_LOOP = {RESET_LATENCY_BASELINE_AFTER_ADD_LOOP}")
    print(f"ENABLE_SCHEDULER_MIN_BATCH_HOLD = {ENABLE_SCHEDULER_MIN_BATCH_HOLD}")
    print(f"SCHEDULER_MIN_BATCH_TARGET_REQUESTS = {SCHEDULER_MIN_BATCH_TARGET_REQUESTS or 'MIN_BATCH_TARGET_REQUESTS/NUM_REQUESTS'}")
    print(f"SCHEDULER_MIN_BATCH_HOLD_MS = {SCHEDULER_MIN_BATCH_HOLD_MS}")
    print("=" * 100)

    all_repeats = []

    for repeat_id in range(NUM_REPEATS):
        repeat_dir = BASE_PROFILE_DIR / f"repeat_{repeat_id:03d}"
        active_dir = repeat_dir / "active"
        gpu_dir = repeat_dir / "gpu"

        active_dir.mkdir(parents=True, exist_ok=True)
        gpu_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[REPEAT START] {repeat_id + 1}/{NUM_REPEATS}")
        print(f"[REPEAT DIR]   {repeat_dir}")
        print("#" * 100)

        run_subprocess(
            phase="active",
            profile_dir=active_dir,
            disable_patch=True,
            repeat_id=repeat_id,
        )

        run_subprocess(
            phase="gpu",
            profile_dir=gpu_dir,
            disable_patch=False,
            repeat_id=repeat_id,
        )

        final = merge_one_repeat_results(
            repeat_id=repeat_id,
            repeat_dir=repeat_dir,
        )

        all_repeats.append(final)

    if not all_repeats:
        raise RuntimeError("No repeat results were generated.")

    all_df = pd.concat(all_repeats, ignore_index=True)

    all_path = BASE_PROFILE_DIR / "all_repeats_paired_request_metrics.csv"
    all_df.to_csv(all_path, index=False)

    print("\n" + "=" * 120)
    print("[DONE] All repeat rows saved")
    print(f"  - {all_path}")
    print("=" * 120)

    overall, by_pair, by_repeat, overall_average = build_repeat_summaries(all_df)

    print_key_summary(overall, overall_average)

if __name__ == "__main__":
    main()
