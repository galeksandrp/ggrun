> Internal engineering design note — not end-user documentation. See [Architecture](architecture.md) or [Usage](usage.md) for user-facing docs.

# Compute preflight: fastest stable placement without paying for failed loads

Status: allocation preflight, promotion canaries, measured runtime-growth
accounting, and post-health CUDA-OOM recovery implemented. Documentation
reconciled 2026-08-24; the release-matrix long-context validation remains
pending on idle hardware.

## Why

ggrun's goal is to find the fastest stable whole-layer plan for big MoE serving.
Maximum VRAM fill is not the objective when it reduces prefill or parallel throughput.
Two failure
classes broke that promise (DeepSeek-V4-Flash 146GB, 3×GPU, 2026-07-06/07):

1. **Load-time CUDA OOM.** ggrun's Go-side placement math disagreed with the backend's
   actual allocation and the launch died at `cudaMalloc` — after paying up to 15 minutes
   of `--no-mmap` model load per attempt on HDD.
2. **Runtime CUDA OOM after a healthy load.** The server loaded, passed the health
   check, served small requests — then a real ~30k-token prompt needed an extra
   1000 MiB on the tightest GPU at the 8192-token prefill/checkpoint step and the
   server died. ggrun had cached that placement as "good" because it only ever
   validated startup.

Both must be caught *before* the expensive load, or at worst recovered automatically —
and never by guessed fixed margins (workspace rule: placement math derives from real or
measured values only; fixed-MB cushions break at scale).

## What llama.cpp already provides (source-verified)

- `common_get_device_memory_data()` (`common/fit.cpp`) loads model+context with
  `mparams.no_alloc = true`: tensors get dummy buffers, the scheduler builds the real
  startup graphs, and `llama_get_memory_breakdown()` reports planned per-device
  model/context/compute bytes. No VRAM is committed; runs in ~1s even for a 146GB model.
- CLI frontend: `llama-fit-params --fit-print on <args>` prints per-device MiB rows:
  `CUDA0 <model> <context> <compute>` (+ `Host` row). Built from `tools/fit-params/`
  (target `llama-fit-params`, not built by default).
- Limit: startup reserve covers the synthetic prompt-processing and token-generation
  graphs. Real requests go through `process_ubatch` → `ggml_backend_sched_alloc_graph`,
  which can re-reserve a *larger* graph later (llama-server creates prompt-cache
  checkpoints every 8192 tokens by default — the exact crash point observed). So the
  no-alloc preflight is necessary but not sufficient for runtime stability.

Measured accuracy (DeepSeek-V4 @ ctx 1M, q8_0 KV, parallel 4, mainline b9859):
preflight per-GPU totals matched post-load `nvidia-smi` within ~40-300 MiB of the
measured values (plus the separately-probed ~680 MB CUDA context overhead per GPU).

## Stage 1 — implemented

`go/cmd/ggrun/preflight.go` + gate in `startLaunchWithCUDAOOMRecovery`:

- `findFitParamsBin`: looks for `llama-fit-params` next to the resolved server binary
  (backend build dir), then `.bin/`, then PATH. It never pairs a custom fork with an
  unrelated mainline oracle from PATH.
- Before every launch attempt (including OOM re-plans), ggrun runs the fit-print with
  the memory-shaping subset of the real args (`-m/-c/-b/-ub/-ctk/-ctv/-np/-ngl/-ts/
  -sm/-ot/--n-cpu-moe/-fa/-mg`) and `CUDA_DEVICE_ORDER=PCI_BUS_ID` (same device
  numbering contract as the real launch — mainline's default enumeration is
  fastest-first, which put a 15.6GB buffer on a 12GB card when launched without it).
- Per CUDA device: `model+context+compute + measured CUDA overhead` vs free VRAM.
  A deficit feeds `ReplanAfterOOM` with the exact measured overshoot — same machinery
  as startup OOM recovery, but at ~1s instead of a full load. Capped at 3 preflight
  re-plans. Oracle failure falls back to the contained generic probe; it never silently
  skips the memory gate.
- Placement caches (`.place`) are now written **only after a healthy load** (main.go
  success branch, recovery.handleHealthy), and overwritten on success after a derate.
  Previously OOM re-plans persisted never-loaded plans, which poisoned later launches.

## Stage 2 — functional and workload promotion canaries (implemented)

HTTP health is no longer treated as proof that a placement is reusable.
`verifyAndActivateLaunch` creates a lifecycle profile and advances it through
allocation, load, functional, cache, performance, and active states:

- The first real workload is a bounded cold/append/older-branch/replay cache
  canary. It requires a non-empty bounded answer from every call, deterministic
  replay, and measured prefix restoration. Its generated prefix crosses at
  least two 512-token checkpoint boundaries on ordinary tokenizers, exercising
  graph creation and checkpoint state rather than merely listing the model.
- A functional failure rejects the profile and stops the launch. A backend that
  answers but cannot prove deterministic cache restoration may remain available
  in a visibly degraded state, but its placement and full launch config are not
  promoted.
- Claude Code profiles additionally exercise the real Anthropic gateway. When a
  separate worker/reviewer is seated, its `local-fast` route receives its own
  canary.
- Only an active profile persists the MoE `.place` entry, verified launch
  configuration, and explicitly screened calibration result. A later explicit
  screen can reuse only the exact model/backend/hardware/policy identity that
  passed. Automatic optimizer replay additionally requires the versioned agent
  screen plus branch/replay, functional/gateway, lifecycle, and clean-relaunch
  validation level.
- On Linux, after the functional canary has forced the first real allocations,
  the server cgroup is resized from the pre-launch ceiling to measured
  non-reclaimable host memory plus configured headroom, while preserving the
  planned host-footprint floor.

The original design proposed running a full per-slot long-context sweep and
deriving GPU growth from `nvidia-smi` deltas during every promotion. The
implemented promotion canary is deliberately bounded; it does not claim to
cover the largest checkpoint, vision, speculation, or maximum fan-out shape.
Those cases belong to the release hardware matrix. Runtime VRAM growth is
instead learned from backend allocation evidence after the model is known to
be serving, as described below, so model/KV bytes are not accidentally recorded
twice as generic growth.

## Stage 3 — runtime OOM recording and recovery (implemented)

Post-health crashes are now distinguished from load failures by a serving
boundary in the backend log (`model loaded` or ggrun's passed-health marker).
An allocation failure before that boundary remains startup placement evidence;
only a failure after it can become runtime-growth evidence.

- A parseable post-health `cudaMalloc` failure records the device and failed
  allocation for the exact model, context, ubatch, KV, backend, GPU-set, and
  parallel signature. CUDA VMM failures that omit the allocation size use one
  GGUF-derived routed-expert layer as the smallest placement-changing reserve.
  When no expert ledger exists, the fallback is explicitly tagged estimated
  and contained to that runtime key so a later measurement can replace it.
- Preflight consumes the recorded per-device growth in addition to the
  backend's model/context/compute rows and measured CUDA overhead. A cold
  context/ubatch key may carry only the largest non-estimated observation for
  the same model, backend, GPU set, and parallel count; estimated or
  cross-parallel evidence is rejected.
- The failed lifecycle profile, `.place` entry, calibration decisions, and
  verified full config are revoked before re-planning. Re-planning skips the
  stale placement cache, shares the launch-wide rejected-argv set, and refuses
  to restart an argv identical to one that already crashed.
- A normal serving launch retries at most two recognized runtime CUDA OOMs.
  Every replacement server must pass health and all promotion canaries again
  before it becomes active. An unrecognized crash or exhausted recovery budget
  fails visibly instead of entering a restart loop.
- In Claude Code mode the terminal belongs to the Claude client, so ggrun does
  not replace its backend invisibly during the session. A health monitor and
  the client-exit path record a matching post-health OOM for the next launch,
  guarded by a log fingerprint so the same crash is not counted twice.
- `ggrun probe-reset <model> ...` removes learned runtime growth for one exact
  signature when an operator intentionally wants a clean measurement.

This recovery is the safety net for runtime shapes the bounded promotion canary
does not cover. The pending real-hardware release matrix still has to prove a
long prompt crossing a context-checkpoint boundary followed by concurrent
decode; implementation is not a substitute for that release evidence.

## Backend-neutral allocation firewall — implemented

Backends without a matching fit oracle are measured without modifying the fork:

- `native/memguard/libggrun-memguard.so` is injected with `LD_PRELOAD` and
  intercepts CUDA runtime/driver device, managed, async, VMM, pinned-host, and
  `mlock` allocation paths. Per-visible-device limits come from current free VRAM
  minus separately measured CUDA context overhead.
- Every event is JSONL. The authoritative device total is the allocator peak;
  backend log labels are optional and any difference is retained as
  `unaccounted_bytes`, never discarded.
- The backend runs in a fresh systemd cgroup v2 scope with `MemoryHigh`,
  `MemoryMax`, no swap, `memory.oom.group=1`, and process-group cleanup. The
  kernel's `memory.peak` and `oom_kill` counters are read before scope removal.
- A denied CUDA allocation returns CUDA OOM to the backend process and feeds
  `active + requested - limit` into the existing re-planner. A host cgroup OOM
  fails closed and is not retried with a larger implicit allowance.
- Verified plans are keyed by backend build, model identity, hardware, and exact
  memory-shaping argv under `memory-probes/`. Incomplete coverage is not cached.
- Production placement sets `RequireMeasuredBuffers`: cold compute/host buffer
  formulas no longer decide fit. The contained probe measures the candidate,
  records exact evidence, and the existing bounded loop recomputes until argv is
  stable before a serving launch is allowed.

Use `ggrun memory-probe <model> --json` to run this loop and stop before serving.
Backends without an advertised allocation-only dry-run require interactive consent,
or `--allow-live-memory-probe` for unattended use.

## Non-goals

- No unverified per-arch memory formulas as final authority. Cold-cache estimates
  may use measured architecture fallbacks, but the backend preflight remains the
  oracle and replaces them with exact rows before load.
- No fixed safety margins. Every reserve must trace to a probe, a fit-print row, or a
  parsed backend log line.
