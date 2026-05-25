# vLLM Serving Experiment: CPU Core Impact on LLM Serving Performance

This repository contains the experiment code and result files for analyzing how CPU core availability affects LLM serving performance in a vLLM-based inference environment.

The study focuses on the following research question:

> How does restricting the number of available CPU cores affect request-level latency and non-GPU residual overhead in vLLM serving?

The experiments use **vLLM 0.11.2** serving **Llama 3.1 8B Instruct** with **Tensor Parallelism (TP=2)** on two NVIDIA RTX 3090 GPUs. CPU core availability is restricted using Linux `taskset`, and the experiments measure request-level latency, CUDA-event-based GPU execution time, non-GPU residual overhead, active CPU thread activity, and potential thread contention events.

---

## 1. Repository Structure

The repository is organized into two main experiment groups:

```text
vLLM_serving_experiment/
├── README.md
├── core_check/
│   ├── run.py
│   ├── phase_worker.py
│   ├── sitecustomize.py
│   ├── 2core/
│   │   ├── all_repeats_paired_request_metrics.csv
│   │   ├── overall_request_average_metrics.csv
│   │   ├── prompt_specs.json
│   │   ├── repeat_summary_by_pair.csv
│   │   ├── repeat_summary_by_repeat.csv
│   │   └── repeat_summary_overall.csv
│   ├── 4core/
│   ├── 8core/
│   ├── 16core/
│   └── 32core/
└── contention/
    ├── run_contention.py
    ├── phase_worker_contention.py
    ├── sitecustomize.py
    └── 2core_contention/
        ├── contention_runnable_thread_details.csv
        ├── contention_runnable_thread_summary_by_tid.csv
        ├── contention_runnable_thread_summary_by_type.csv
        ├── cpu_thread_contention_scenarios.csv
        └── prompt_specs.json
```

### Main directories

- `core_check/`  
  Contains the main CPU core sweep experiment code and results.  
  The evaluated CPU core configurations are `2core`, `4core`, `8core`, `16core`, and `32core`.

- `contention/`  
  Contains the supplementary 2-core contention analysis code and results.  
  This analysis traces potential CPU thread contention events under the most resource-constrained condition.

---

## 2. Experiment Overview

Modern LLM serving systems are often optimized around GPU-side bottlenecks such as batching, KV cache management, and attention kernel efficiency. However, the CPU also performs critical control-plane work, including:

- request scheduling,
- runtime coordination,
- request state management,
- KV cache metadata management,
- worker orchestration,
- and CPU-side preparation around GPU execution.

This repository provides a controlled offline vLLM measurement framework that separates:

- request-level latency metrics,
- iteration-level GPU execution time,
- non-GPU residual overhead,
- active CPU thread activity,
- and potential thread contention events.

The main goal is to characterize whether vLLM serving requires full CPU core allocation, or whether similar serving performance can be maintained with fewer CPU cores under a fixed small-batch offline workload.

---

## 3. Experimental Setup

### 3.1 Hardware

| Component | Specification |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 3090 |
| Tensor parallelism | TP = 2 |
| CPU | Intel Xeon Gold 5218 @ 2.30 GHz |
| Physical CPU cores | 32 |
| Threads per core | 1 |
| CPU core configurations | 2, 4, 8, 16, 32 |

### 3.2 Software

| Component | Version / Setting |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| LLM serving framework | vLLM 0.11.2 |
| Model | Llama 3.1 8B Instruct |
| GPU memory utilization | 0.9 |
| Maximum model length | 2048 |
| Execution mode | Offline vLLM engine |
| Request insertion | `engine.add_request()` |
| Generation loop | repeated `engine.step()` |

The experiment directly invokes the offline vLLM engine instead of using the OpenAI-compatible online API server. Therefore, HTTP parsing, network communication, client-server serialization, and API server overhead are intentionally excluded.

This controlled setup isolates the behavior of the vLLM engine itself, especially scheduling, request state management, worker coordination, GPU execution, and non-GPU residual overhead.

---

## 4. Workload Configuration

Each measured run uses one fixed small batch consisting of three requests.

| Parameter | Value |
|---|---|
| Number of measured requests per run | 3 |
| Input length per request | 256 tokens |
| Output length per request | 256 tokens |
| Sampling temperature | 0.0 |
| EOS handling | `ignore_eos=True` |
| Repeats per CPU core condition | 10 |
| Request-level rows per CPU core condition | 30 |
| Warmup requests | 3 |
| Warmup input length | 256 tokens |
| Warmup output length | 32 tokens |

Each measured request is forced to generate exactly 256 output tokens. Therefore, TPOT is computed over 255 output-token intervals:

```text
TPOT = (t_end - t_first) / (N_out - 1)

N_out = 256
```

Warmup requests and warmup iterations are excluded from the final latency, GPU time, and non-GPU residual calculations.

---

## 5. CPU Core Control

CPU core availability is restricted using Linux `taskset`.

Example commands:

```bash
taskset -c 0-1  python core_check/run.py
taskset -c 0-3  python core_check/run.py
taskset -c 0-7  python core_check/run.py
taskset -c 0-15 python core_check/run.py
taskset -c 0-31 python core_check/run.py
```

The five CPU core configurations are:

```text
2 cores
4 cores
8 cores
16 cores
32 cores
```

The 32-core condition is used as the full-core reference baseline.

Note that `taskset` restricts CPU affinity for the target process and its child processes, but it does not fully isolate the system from operating-system activity or background processes. Therefore, the results should be interpreted as controlled affinity-based measurements rather than fully isolated bare-metal measurements.

---

## 6. Main CPU Core Sweep

The main CPU core sweep is located in:

```text
core_check/
```

### 6.1 Main files

```text
core_check/run.py
core_check/phase_worker.py
core_check/sitecustomize.py
```

### 6.2 File roles

#### `run.py`

`run.py` orchestrates the main experiment.

It is responsible for:

- running each CPU core condition,
- launching active and GPU measurement phases,
- repeating each condition,
- collecting raw outputs,
- merging request-level and iteration-level metrics,
- and generating summary CSV files.

#### `phase_worker.py`

`phase_worker.py` executes the offline vLLM engine and collects request-level metrics.

It records:

- request submission time,
- first token time,
- request completion time,
- E2E latency,
- TTFT,
- TPOT,
- input token count,
- output token count,
- active CPU thread activity,
- and process-tree CPU activity.

#### `sitecustomize.py`

`sitecustomize.py` is automatically imported by Python and patches vLLM internals inside the vLLM worker processes.

It provides two key functions:

1. Scheduler-side min-batch hold
2. CUDA-event-based GPU timing instrumentation

The GPU timing patch wraps:

```text
GPUModelRunner.execute_model()
```

For each vLLM iteration, it records:

- iteration index,
- wall-clock start/end time,
- CUDA-event-based GPU time,
- scheduled request IDs,
- per-request scheduled token counts,
- tensor-parallel rank information,
- and prefill/decode classification.

---

## 7. Scheduler-Side Min-Batch Hold

The experiment uses a fixed batch size of three measured requests. However, vLLM’s offline engine receives requests through sequential calls to:

```python
engine.add_request(...)
```

Without additional control, the scheduler may begin processing before all three requests are enqueued. This can cause the first measured prefill iteration to include only one or two requests instead of the intended three-request batch.

To avoid this artifact, the scheduler-side min-batch hold buffers incoming measured requests until all three requests are available. The requests are then replayed into the original scheduler queue together.

The artificial hold delay is excluded from E2E and TTFT measurements by resetting the latency baseline after request insertion is complete.

---

## 8. Main Result Files

Each CPU core result directory contains the following files:

```text
all_repeats_paired_request_metrics.csv
overall_request_average_metrics.csv
prompt_specs.json
repeat_summary_by_pair.csv
repeat_summary_by_repeat.csv
repeat_summary_overall.csv
```

### 8.1 `all_repeats_paired_request_metrics.csv`

This file contains request-level rows across all repeats.

Typical fields include:

- core count,
- repeat index,
- request ID or pair ID,
- input token count,
- output token count,
- E2E latency,
- TTFT,
- TPOT,
- GPU time,
- non-GPU residual time,
- non-GPU residual ratio,
- active CPU thread metrics.

### 8.2 `overall_request_average_metrics.csv`

This file summarizes request-level averages for a CPU core condition.

It is useful for comparing the overall behavior of each core configuration.

### 8.3 `prompt_specs.json`

This file stores the prompt specifications used for the experiment.

The same prompt specification structure is reused to reduce prompt-induced variance across repeated runs.

### 8.4 `repeat_summary_by_pair.csv`

This file summarizes metrics by request pair or stable request position.

### 8.5 `repeat_summary_by_repeat.csv`

This file summarizes metrics by repeat.

This is useful when analyzing run-to-run variability.

### 8.6 `repeat_summary_overall.csv`

This file provides an overall summary across repeats.

---

## 9. Metrics

### 9.1 Request-level latency

For each request:

```text
E2E latency = t_end - t_submit
TTFT        = t_first - t_submit
TPOT        = (t_end - t_first) / (N_out - 1)
```

where:

```text
N_out = 256
```

### 9.2 GPU time

GPU execution time is measured using CUDA events inserted around:

```text
GPUModelRunner.execute_model()
```

For TP=2, each logical vLLM iteration produces one timing row per tensor-parallel rank. Since the two ranks execute in parallel and synchronize at the logical iteration boundary, rank times are aggregated using max-over-ranks rather than sum-over-ranks.

For logical iteration `i`:

```text
G_i = max(G_i,rank0, G_i,rank1)
```

This reflects the wall-clock completion time of the TP=2 logical iteration.

### 9.3 Non-GPU residual time

The non-GPU residual is defined as:

```text
non-GPU residual time = wall-clock latency - CUDA-event-based GPU time
```

This residual includes all wall-clock components not captured by CUDA event timing, such as:

- request scheduling,
- Python runtime overhead,
- KV cache metadata management,
- inter-process coordination,
- synchronization,
- waiting time,
- OS scheduling delay,
- and other non-GPU runtime overheads.

Important note:

> The non-GPU residual is not a direct measurement of CPU execution time. It should not be interpreted as pure CPU time.

### 9.4 Active CPU thread count

Active CPU threads are measured using `psutil` by sampling the process tree and checking per-thread CPU time deltas.

A thread is considered active if its CPU time increases during the measurement window.

The sampler thread itself is excluded from active thread counting.

---

## 10. Key Results

### 10.1 Request-level latency

| CPU cores | E2E mean (ms) | TTFT mean (ms) | TPOT mean (ms) |
|---:|---:|---:|---:|
| 2  | 4403.8 | 26.4  | 14.443 |
| 4  | 3276.4 | 29.1  | 11.524 |
| 8  | 3296.1 | 77.5  | 11.522 |
| 16 | 3311.0 | 86.2  | 11.524 |
| 32 | 3340.4 | 132.6 | 11.618 |

Main observations:

- The 4-, 8-, 16-, and 32-core conditions show similar E2E latency and TPOT.
- The 2-core condition shows a clear increase in E2E latency and TPOT.
- Compared with the 32-core baseline, the 2-core condition increases E2E latency by approximately 31.8%.
- Compared with the 32-core baseline, the 2-core condition increases TPOT by approximately 24.3%.
- TTFT shows a different pattern from E2E and TPOT, with higher variability in the 8-, 16-, and 32-core conditions. Therefore, TTFT should be interpreted as a secondary observation rather than the main conclusion.

### 10.2 Non-GPU residual overhead

| CPU cores | E2E residual mean (ms) | E2E residual ratio | TPOT residual mean (ms) | TPOT residual ratio |
|---:|---:|---:|---:|---:|
| 2  | 1118.9 | 25.4% | 1.844 | 12.7% |
| 4  | 328.8  | 10.0% | 0.039 | 0.3% |
| 8  | 349.2  | 10.6% | 0.039 | 0.3% |
| 16 | 363.5  | 11.0% | 0.040 | 0.3% |
| 32 | 369.4  | 11.1% | 0.043 | 0.4% |

Main observations:

- The 4–32-core conditions maintain an E2E non-GPU residual ratio of approximately 10–11%.
- The 2-core condition increases the E2E non-GPU residual ratio to 25.4%.
- The 4–32-core conditions maintain a TPOT non-GPU residual ratio of approximately 0.3–0.4%.
- The 2-core condition increases the TPOT non-GPU residual ratio to approximately 12.7%.

### 10.3 Active CPU thread activity

| CPU cores | Prefill active threads mean | Decode active threads mean |
|---:|---:|---:|
| 2  | 4.0 | 3.208 |
| 4  | 4.0 | 3.260 |
| 8  | 4.7 | 3.511 |
| 16 | 5.0 | 3.378 |
| 32 | 4.9 | 3.366 |

Main observations:

- Prefill-stage active thread count is approximately 4–5 across all CPU core conditions.
- Decode-stage active thread count is approximately 3.2–3.5 across all CPU core conditions.
- Under the 2-core condition, the average number of active decode-stage threads exceeds the number of allocated CPU cores, which motivates the supplementary contention analysis.

---

## 11. Supplementary 2-Core Contention Analysis

The supplementary contention analysis is located in:

```text
contention/
```

### 11.1 Main files

```text
contention/run_contention.py
contention/phase_worker_contention.py
contention/sitecustomize.py
```

### 11.2 Result directory

```text
contention/2core_contention/
```

This directory contains:

```text
contention_runnable_thread_details.csv
contention_runnable_thread_summary_by_tid.csv
contention_runnable_thread_summary_by_type.csv
cpu_thread_contention_scenarios.csv
prompt_specs.json
```

### 11.3 Purpose

The 2-core condition is the most resource-constrained condition in the CPU core sweep. The contention analysis checks whether more vLLM-related CPU threads are active than the available CPU cores during each iteration.

This analysis is used as supplementary evidence for interpreting the 2-core performance degradation.

### 11.4 Definition of potential contention

A potential contention event is defined as:

```text
active vLLM-related threads > allocated CPU cores
```

For the 2-core condition, an iteration is marked as a potential contention event if more than two vLLM-related threads consume CPU time within the sampling window.

Important note:

> This is a thread-level proxy. It is not a direct hardware-counter measurement of CPU contention, run-queue stalls, or context-switch pressure.

### 11.5 Main contention finding

The dominant potential contention group occurs in the decode stage and includes:

```text
TP0 worker
+ TP1 worker
+ driver main scheduling/orchestration
```

This group accounts for approximately 89.34% of summarized potential contention events in the 2-core condition.

This suggests that the 2-core degradation is mainly associated with decode-stage CPU-side activity and coordination rather than prefill-stage activity.

---

## 12. How to Run

### 12.1 Environment setup

Create and activate a Python environment with vLLM 0.11.2.

```bash
python3 -m venv venv-vllm-0.11.2
source venv-vllm-0.11.2/bin/activate

pip install --upgrade pip
pip install vllm==0.11.2
pip install psutil pandas numpy transformers
```

Depending on the CUDA and PyTorch installation, additional GPU-specific packages may be required.

### 12.2 Set the model path

Set the model name or local model path.

```bash
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
```

For a local model:

```bash
export MODEL_NAME="/path/to/Llama-3.1-8B-Instruct"
```

### 12.3 Run the main CPU core sweep

The instrumentation depends on `sitecustomize.py`, so the `core_check/` directory should be visible in `PYTHONPATH`.

```bash
export PYTHONPATH="$PWD/core_check:$PYTHONPATH"
```

Run each core condition using `taskset`.

```bash
taskset -c 0-1  python core_check/run.py
taskset -c 0-3  python core_check/run.py
taskset -c 0-7  python core_check/run.py
taskset -c 0-15 python core_check/run.py
taskset -c 0-31 python core_check/run.py
```

The exact command-line arguments may depend on the current version of `run.py`.

### 12.4 Run the 2-core contention analysis

The contention experiment uses its own `sitecustomize.py`, so the `contention/` directory should be placed in `PYTHONPATH`.

```bash
export PYTHONPATH="$PWD/contention:$PYTHONPATH"
taskset -c 0-1 python contention/run_contention.py
```

The contention analysis should use the same model, hardware, TP setting, input length, output length, and batch size as the main CPU core sweep.

---

## 13. Interpretation

The results suggest that, under the fixed small-batch offline workload used in this study, vLLM serving does not necessarily require full 32-core CPU allocation to maintain similar request-level performance.

In the evaluated setting:

- 4–32 CPU cores show similar E2E latency and TPOT.
- 2 CPU cores produce clear degradation in E2E latency, TPOT, and non-GPU residual overhead.
- Decode-stage potential contention events are frequently observed in the 2-core setting.
- The observed degradation in the 2-core condition is consistent with increased non-GPU residual overhead and decode-stage thread activity.

However, these results should not be generalized to all LLM serving deployments. The experiment uses a controlled offline setting, a fixed small-batch workload, a single model, a single hardware configuration, and a specific vLLM version.

---

## 14. Limitations

This repository reports a controlled measurement study with several important limitations.

### 14.1 Offline engine only

The experiment directly invokes the offline vLLM engine. It does not include HTTP parsing, network communication, client-server serialization, or API server overhead.

### 14.2 Fixed small-batch workload

The workload uses three requests per run, each with 256 input tokens and 256 output tokens. The results may differ under high-concurrency online traffic, variable sequence lengths, different request arrival patterns, or different batch sizes.

### 14.3 Single model and hardware configuration

The experiment uses Llama 3.1 8B Instruct, vLLM 0.11.2, two RTX 3090 GPUs, and TP=2. Results may differ for larger models, different GPUs, different TP degrees, or newer vLLM versions.

### 14.4 Non-GPU residual is not direct CPU time

Non-GPU residual time is computed by subtracting CUDA-event GPU execution time from request-level wall-clock latency. It includes scheduling, synchronization, waiting time, Python runtime overhead, OS scheduling delay, and other non-GPU components.

Therefore, it should not be interpreted as pure CPU execution time.

### 14.5 Potential contention is a proxy

Potential contention events are based on active thread counts exceeding allocated CPU cores. They are not direct measurements from hardware performance counters.

### 14.6 CPU affinity is not full isolation

`taskset` restricts CPU affinity but does not fully isolate the system from OS noise or background activity.

### 14.7 Limited repeat count

Each CPU core condition is repeated 10 times. This is useful for stability checking, but larger repeat counts would improve statistical confidence.

---

## 15. Related Background

This experiment is motivated by the observation that LLM serving systems are often optimized around GPU memory efficiency, batching, and kernel execution, while CPU-side control-plane work is less frequently isolated as an independent experimental variable.

vLLM improves LLM serving throughput through PagedAttention and block-based KV cache management. This repository complements that direction by focusing on CPU core availability and its effect on request-level latency and non-GPU residual overhead.

---

## 16. Citation

If you use this repository or refer to the experimental results, please cite the accompanying report or paper draft:

```bibtex
@misc{kim2026cpuvllmserving,
  title  = {Characterizing the Performance Impact of CPU Resources in LLM Serving},
  author = {Dongguk Kim},
  year   = {2026},
  note   = {vLLM 0.11.2, Llama 3.1 8B Instruct, CPU core sweep study}
}
```

For vLLM and PagedAttention, please refer to:

```bibtex
@inproceedings{kwon2023efficient,
  title     = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author    = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle = {Proceedings of the ACM Symposium on Operating Systems Principles},
  year      = {2023}
}
```

---

## 17. License

This repository is intended for academic and research use.

Please check the licenses of all third-party dependencies, including vLLM, PyTorch, Hugging Face Transformers, and the model weights used in the experiment.

---

## 18. Contact

```text
Dongguk Kim
Department of Electrical and Electronic Engineering
Korea University
```
