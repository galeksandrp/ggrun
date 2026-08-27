# FreeToken development lane

> Separate, parked engineering lane. It is not part of the active ggrun core
> roadmap and is not a dependency of the next ggrun release.

Status: engine integration parked after the initial adapter. Comparison
protocol preparation resumed without changing that boundary, 2026-08-24.

## Boundary

The existing `ggrun freetoken` command is an experimental process/API adapter.
FreeToken remains its own inference runtime, Python environment, model-format
stack, calibration cache, and control plane.

This lane must not:

- route FreeToken through ggrun's GGUF tensor planner, llama.cpp tune cache,
  allocation probe, or OOM re-planner;
- translate FreeToken flags into llama.cpp flags or the reverse;
- vendor FreeToken's Torch/CUDA dependency stack or runtime kernels;
- make FreeToken installation a normal ggrun install/update dependency;
- claim multi-GPU, general GGUF, or cross-platform support before the upstream
  runtime provides and ggrun independently verifies it.

## Completed foundation

- explicit `ggrun freetoken` command;
- separate binary/version handshake;
- strict one-physical-NVIDIA-GPU isolation;
- occupied-port protection and structured loading/serving health gate;
- `/v1/models` verification and process-tree cleanup;
- native FreeToken flags passed only after `--`;
- documented separation between `ggrun detect --bandwidth` and
  `ft bench bw`.

## Parked backlog

If this lane is resumed, work proceeds in this order:

1. Run the controlled [real-agent runtime comparison](agentic-runtime-comparison-plan.md)
   on an equivalent base model, context, sampling, task suite, and concurrency.
   Record verified task outcomes as well as load, TTFT, prefill, decode, queue,
   RAM, VRAM, cache reuse, cancellation, and stability.
2. Add read-only ingestion of FreeToken health, stats, request history, and cache
   geometry without changing the upstream server.
3. Consider explicit cache-pool resize commands only after request draining,
   maintenance-state handling, rollback behavior, and a post-rebuild canary are
   proven.
4. Independently prototype semantic tool-call anchors in a maintained
   llama.cpp/ik backend branch; do not copy runtime code into the launcher.
5. Consider an engine recommendation only after repeated measurements establish
   a stable decision boundary.

## Resume gate

Do not activate this lane until:

- the active ggrun core roadmap reaches its next verified patch release;
- FreeToken is installed in an isolated environment with a supported local
  checkpoint;
- the machine is available for an idle calibration and controlled comparison;
- the upstream version and hardware limitations are re-checked;
- the operator explicitly chooses to resume the lane.

Until then, the adapter remains available for manual experiments but receives
only safety or compatibility fixes.
