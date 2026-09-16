---
name: max-throughput
description: >
  Use when running compute-heavy work: model training, fine-tuning,
  evaluation, benchmarks, simulations, data preprocessing, dataset
  generation, compilation, test suites, or any long-running batch job,
  or when a job is slower than expected and needs performance tuning.
  Detect available CPU/GPU/memory resources, parallelize aggressively,
  then profile the running job to find the true bottleneck (data
  pipeline, CPU decode, GPU compute, VRAM, I/O) and tune batch size,
  DataLoader workers, pre-encoding, and precision accordingly to
  minimize wall-clock runtime.
risk: safe
source: https://github.com/Promethe-us/max-throughput
---

# Max Throughput

The default objective is **minimum wall-clock completion time**, not minimum
resource consumption. Leaving GPUs or CPU cores idle while runnable work
exists is a bug, not a conservative choice. When a job is slow, **measure
first, then tune the measured bottleneck** - never guess knobs.

## When to Use This Skill

- Use when launching model training, fine-tuning, evaluation, benchmarks,
  simulations, data preprocessing, or dataset generation.
- Use when running builds, test suites, or any long-running batch job.
- Use when a job is slower than expected and needs performance tuning
  (batch size, DataLoader workers, pre-encoding, precision).

## Workflow

```
Task received
   |
   v
Resource discovery  (scripts/inspect_resources.py)
   |
   v
Build a work DAG: which subtasks are independent?
   |
   v
Dispatch concurrently
   +-- agent-level: parallel subagents for independent tasks
   +-- GPU-level:   DDP / torchrun / one experiment per free GPU
   +-- CPU-level:   multiprocessing, pytest-xdist, make -j$(nproc)
   |
   v
Profile & verify  (scripts/quick_triage.py while the job runs)
   |
   v
Bottleneck class identified?
   +-- data pipeline / CPU decode  -> num_workers, prefetch, pre-encode
   +-- GPU compute                 -> batch size, AMP, torch.compile
   +-- VRAM                        -> grad accumulation / checkpointing
   +-- under-occupied              -> pack more jobs / DDP
   |
   v
Apply ONE knob, re-measure, iterate
```

## 0. Always inspect the machine first

Before launching any long-running compute, run the bundled resource probe
from this skill's directory:

```bash
python scripts/inspect_resources.py          # human-readable summary
python scripts/inspect_resources.py --json   # machine-readable
```

Linux/macOS alternative: `bash scripts/inspect_resources.sh`

Never assume "1 GPU, 4 cores". Read the actual numbers and use them to
size every launch below.

## 1. Agent-level parallelism

- Dispatch independent investigations, file edits, reviews, downloads, and
  experiments to parallel subagents / concurrent tool calls.
- Never serialize two independent long-running tasks without a stated
  reason (shared writable state, GPU memory limits, license servers).
- Batch independent tool calls in a single turn instead of one by one.

## 2. GPU workloads

- Detect free GPUs (`nvidia-smi --query-gpu=index,memory.free --format=csv`).
- Single training job, multi-GPU capable: prefer
  `torchrun --standalone --nproc_per_node=<N> train.py` (DDP) over
  `python train.py`. Use FSDP/deepspeed zero-3 when the model does not fit.
- Multiple independent jobs (sweeps, seeds, evals): pack one job per free
  GPU with explicit `CUDA_VISIBLE_DEVICES=<i>` for each.
- Small models on large GPUs: co-locate several jobs per GPU until
  VRAM/compute is saturated, instead of one-model-one-GPU.
- Data loading is often the real bottleneck: set DataLoader
  `num_workers` based on measurement (see section 5), enable
  `pin_memory=True`, and use `persistent_workers=True` for long runs.
- Do not leave a GPU idle while a runnable independent GPU task exists.

## 3. CPU workloads

- Builds: `cmake --build . -j$(nproc)` / `make -j$(nproc)` /
  `cargo build -j$(nproc)` when supported.
- Tests: `pytest -n auto` (pytest-xdist) when tests are independent;
  `nextest -j` for Rust; shard suites that cannot share state.
- Data preprocessing: shard the input (by file or line ranges) and process
  shards concurrently with `multiprocessing`, `joblib`, GNU `parallel`,
  or `xargs -P`. Target ~75-90% of logical cores.
- Compression/archives, image conversion, dataset downloads: parallelize
  per-file (`parallel -j$(nproc) < tool`, `aria2c -x16` for downloads).

## 4. Avoid blind oversubscription

Parallelism must be safe to actually be faster:

- Budget RAM: total worker memory must fit in available RAM; check with the
  resource probe before choosing worker counts.
- Multi-process + intra-op threads multiply: when launching N workers set
  `OMP_NUM_THREADS=1` (or `= total_cores / N`) and likewise
  `MKL_NUM_THREADS`, otherwise N processes each spawn N threads and thrash.
- DDP already occupies all GPUs: do not additionally pin experiments onto
  the same GPUs unless VRAM clearly allows it.
- Respect external limits: cluster schedulers, license servers, rate
  limits, and shared filesystems can make "more parallel" slower.

## 5. Profile first, then tune the real bottleneck

After launching, verify utilization, then find the ROOT CAUSE before
touching any knob:

```bash
# while the job runs in its steady phase:
python scripts/quick_triage.py --duration 60
```

Triage signals (details: `references/profiling.md`):

| Signal | Real bottleneck | First knobs to try |
|---|---|---|
| GPU util spiky (many samples < 50%) AND CPU high | data pipeline CPU-bound (decode/tokenize/augment) | raise `num_workers`/`prefetch_factor`; if still short, **pre-encode** the dataset |
| GPU util low, CPU low, disk I/O saturated | storage-bound | local NVMe cache, sharded/tar format (WebDataset), prefetch, pre-encode |
| GPU util ~100% steady, still slow | compute-bound | larger batch (VRAM permitting), AMP/bf16, `torch.compile`, `channels_last`, fused optimizer |
| GPU util low, everything idle, tiny model | kernel-launch / occupancy bound | much larger batch, CUDA graphs, or pack more jobs on the GPU |
| OOM / VRAM at 100% | memory-bound | gradient accumulation, gradient checkpointing, smaller micro-batch |
| Step time gaps between kernels in trace | Python / sync / H2D overhead | `pin_memory` + `non_blocking=True`, `persistent_workers`, remove per-step syncs |

Tuning rules (details: `references/tuning-playbook.md`):

- Change ONE knob at a time and re-measure throughput (items/s), never two.
- Batch size: find the VRAM ceiling by doubling until OOM then back off ~20%;
  scale LR with global batch and add warmup; use gradient accumulation to
  reach the target global batch when VRAM-limited.
- `num_workers`: benchmark the DataLoader ALONE (iterate without training)
  and compare with GPU consumption rate; tune until dataloader-only
  throughput exceeds consumption with margin. Windows: workers need the
  `if __name__ == "__main__":` guard.
- Pre-encode / pre-tokenize when decode is the measured hotspot and
  workers cannot keep up: convert once to tensors / pre-tokenized ids /
  WebDataset shards. Costs disk space and pipeline flexibility - do it
  only when profiling says decode dominates.
- Re-measure after every change; stop when gains drop below ~5%.

Correcting under-utilization is part of the task, not an optional
improvement.

## 6. Decision summary

| Situation | Default action |
|---|---|
| Independent experiments/seeds/evals | one per free GPU, run concurrently |
| Single multi-GPU-capable training | torchrun/DDP across all free GPUs |
| Independent test files | `pytest -n auto` |
| Compilation | `-j$(nproc)` |
| Large preprocessing | shard + multiprocessing / GNU parallel |
| Independent research/edits | parallel subagents |
| Anything long-running | profile with quick_triage.py, then tune the measured bottleneck |
| Slow despite high GPU util | efficiency knobs: batch size, AMP, torch.compile |
| Slow with spiky/low GPU util | data pipeline: workers, prefetch, pre-encode |
