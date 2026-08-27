# Workflow capacity plan

> Proposed core feature. This design uses user-authorized idle hardware to
> shorten multi-agent Claude Code and Ultracode workflows without weakening the
> foreground model or silently taking ownership of additional devices.

Status: accepted for development planning, 2026-08-24. Implementation starts
after the current patch-release gate unless a release-blocking dependency is
discovered during the routing spike.

## Decision

This should be a native ggrun feature. `--parallel` alone is not the feature:
adding sequence slots divides one server's context and compute, and it cannot
make a saturated GPU produce more decode bandwidth. The useful abstraction is
a **workflow capacity planner** that can evaluate three different ways to use
explicitly offered hardware:

1. add safe sequence slots to the primary server;
2. start another instance of the primary model;
3. start one or more smaller worker models on separate devices.

`auto` should measure the eligible alternatives and select the one that reduces
end-to-end workflow time while preserving foreground responsiveness and the
required model quality. It must not assume that more slots, tensor-splitting a
dense model, or a smaller worker is always better.

The initial implementation should favor independent servers on independent
GPUs. They avoid cross-device synchronization, isolate failures and queues, and
let the primary model keep its existing placement. A multi-GPU primary replica
remains a benchmark candidate, not a default: PCIe traffic and the slowest GPU
can make it worse than separate small workers.

## Production evidence from the current run

A 2026-08-24 snapshot of the live Qwen3.8-27B Claude Code workload confirms
that capacity, rather than single-request speed, is the immediate problem:

- router admission was one active request, one queued request, limit one;
- 2,884 successful requests across 127 conversations accumulated 170.5
  request-hours in the admission queue, versus 15.1 hours of prefill and 55.2
  hours of streamed decode;
- the latest 100 records had queue p50 21.8 seconds and p95 593 seconds;
- prompt-cache reuse was already 74.1%, so discarding fairness to chase a small
  reuse gain would optimize the wrong term;
- the active 3090 Ti was saturated while the 4070 and 3060 were idle;
- the backend pass-cost estimator projects large batching gains, but those
  projections are not measurements and the run has reached 145k context, so
  splitting today's context allocation into more slots is not automatically
  valid.

The exact runtime A/B protocol and artifact contract live in the separate
[real-agent comparison plan](agentic-runtime-comparison-plan.md). The core
capacity feature uses those workload shapes without depending on FreeToken.

## Product contract

- Additional hardware is opt-in and explicitly scoped by the user. ggrun never
  claims every idle accelerator merely because it is visible.
- "Idle" means available at planning time and still within normal runtime
  occupancy checks. A device becoming busy must fail closed or drain; ggrun
  must not evict an unrelated process.
- The foreground conversation and its authoritative synthesis stay on the
  primary model. A smaller model never receives them through a heuristic.
- Background work is moved only through a deterministic route marker supplied
  by Claude Code or the Workflow integration. Prompt-text classification is not
  an acceptable routing contract.
- Explicit user choices win. In particular, an explicit `--parallel` value is
  not rewritten by `auto`.
- The primary model keeps an admission reservation so a fan-out cannot make the
  user's next foreground turn wait behind all background agents.
- Every process is represented in placement, memory accounting, status,
  shutdown, resume compatibility, and support output.
- Failure of a worker or replica cannot terminate the primary server. A reroute
  is visible and obeys the destination model's context and quality policy.

## Proposed interface

Names are provisional until the CLI spike, but the behavior should be explicit:

```text
--workflow-capacity off|auto|slots|replica|worker
--workflow-gpus <selectors>
--workflow-model <gguf-or-catalog-id>
--workflow-max-workers <n>
```

- `off` is the initial default and preserves today's behavior.
- `auto` benchmarks and selects among strategies that fit the granted devices.
- `slots` changes only primary-server concurrency and remains subject to
  per-slot context and functional canaries.
- `replica` serves the same checkpoint from a separate backend and therefore
  preserves primary-model capability for routed agents.
- `worker` uses a smaller checkpoint only for work types whose route explicitly
  permits it.
- `--workflow-gpus` is a grant, separate from the primary placement selector.
  It accepts stable ggrun device selectors; no omitted GPU is consumed.
- `--workflow-model` pins the worker checkpoint. Without it, ggrun may recommend
  a compatible installed or catalog model but must show the choice before first
  download or launch.

The TUI should later expose the same policy as "Workflow capacity", list the
granted devices and planned roles, and preview additional VRAM/RAM use before
launch. Configuration persistence should store stable selectors, not transient
CUDA ordinals alone.

## Capacity strategies

### Primary slots

Use additional slots only when the checkpoint still has the required context
per slot, exact memory probes pass, and a concurrent canary improves aggregate
work rather than merely interleaving one saturated decode path. This is the
lowest-memory option, but it shares compute, KV capacity, and failure fate with
the foreground server.

The planner must compare at least:

- context available to each request;
- foreground time to first token and decode rate;
- aggregate completed agent work per minute;
- queue time, cache reuse, VRAM/RAM peak, and stability.

### Same-model replicas

A replica loads the exact primary checkpoint on one or more granted GPUs and is
registered as a separate backend pool. Requests are assigned by health,
capacity, and conversation affinity so an agent stays with its KV cache.

This is the preferred quality-preserving expansion when the model fits and the
measured workflow completion time wins. Dense tensor-split replicas are allowed
only as measured candidates; ggrun must not infer that two mismatched GPUs are
faster together.

### Smaller worker pools

One or more independent worker backends can provide the largest throughput gain
when the primary checkpoint does not fit on the granted devices. Suitable jobs
include explicitly typed search, file discovery, summarization, mechanical
transformation, first-pass review, and other tasks whose workflow declares that
a worker model is acceptable. Planning, difficult code changes, final
synthesis, and ambiguous tasks remain on the primary model.

The current `local-fast` worker/reviewer establishes the useful primitives:
explicit alias routing, a separately launched backend, a placement companion,
measured VRAM, and independent failure handling. The new pool must generalize
those primitives instead of overloading the security reviewer:

- the reviewer retains its own priority and fail-safe behavior;
- worker replicas have individual health and concurrency limits;
- dispatch is least-loaded among compatible healthy workers;
- an agent is sticky to one backend for cache locality;
- worker overflow either queues or visibly falls back according to policy;
- a smaller worker never silently impersonates a same-model replica.

CPU workers are a later explicit opt-in, not an automatic use of "idle"
hardware. Host-offloaded concurrent decode can reduce total throughput enough
to make the workflow slower.

## Routing contract

The gateway already distinguishes `local` from the exact `local-fast` alias and
does not inspect prompt wording. Workflow capacity needs the same standard for
individual agents.

The first engineering spike must determine which stable identity the Workflow
runtime exposes at the request boundary. In preference order:

1. a supported per-agent model/tier option inserted by the existing PreToolUse
   hook;
2. a hook-created workflow/run/agent route manifest joined to stable request
   metadata at the gateway;
3. existing Claude Code model-tier aliases for only the requests Claude Code
   already marks as cheap.

If no stable marker exists, arbitrary Workflow agents remain on the primary
model. ggrun must not guess from labels, phases, token counts, or prompt text.
The hook rewrite must remain idempotent for inline, saved, named, and resumed
workflows, just as the current `stallMs` rewrite is.

Internally, the router grows from one primary proxy plus one companion proxy to
typed backend pools:

```text
foreground/main  -> primary backend, reserved admission
same-model agent -> healthy primary-model replica, then policy fallback
worker-safe agent -> compatible worker pool, then queue or visible fallback
review/classifier -> reviewer lane, unchanged safety semantics
```

Each pool reports active, queued, limit, health, model identity, context,
device placement, and cache-affinity decisions through `/ggrun/router`, the
Claude status line, and `ggrun support`.

## Planning and calibration

Planning occurs after the primary placement is proven. The remaining inventory
is the intersection of user-granted workflow devices and devices that are
actually available; it is not simply `all GPUs - primary GPU`.

Every candidate uses the existing evidence rules:

- exact checkpoint and backend identity;
- measured or backend-reported memory rather than a fixed safety cushion;
- load, functional, context, and concurrent-work canaries;
- placement-ledger reservations before any process starts;
- cgroup/process-tree accounting and clean rollback on partial startup;
- cached results keyed by model, backend build, hardware topology, context, KV
  type, concurrency, and route policy.

The calibration workload must include a serial baseline and representative
fan-out with foreground traffic. Selection is lexicographic:

1. reject any candidate that violates context, correctness, isolation, or
   stability gates;
2. reject a candidate whose foreground regression exceeds measured noise or a
   user-configured latency budget;
3. among the survivors, minimize workflow makespan; then prefer lower memory
   and power cost when results are equivalent.

A cached plan is advisory only. Runtime occupancy changes still trigger a
fresh availability check, and a failed worker is drained without poisoning the
verified primary placement.

## Delivery sequence

### Phase 0: observe and define the route

- capture redacted request metadata for a small inline Workflow, a saved
  Ultracode workflow, and a resumed run;
- identify a stable per-agent marker or prove that the Workflow hook can add
  one;
- add pool-shaped status data without changing routing;
- record serial workflow and foreground-latency baselines.

Exit gate: routing is deterministic across retries and resume, and no prompt
heuristic is required.

### Phase 1: explicit worker mode

- generalize the current companion reservation and process lifecycle into a
  worker pool;
- route only explicitly worker-safe agents;
- reserve primary foreground admission;
- add health draining, sticky dispatch, metrics, shutdown, and tests;
- support user-pinned worker model and GPU selectors.

Exit gate: two simultaneous worker-safe agents can run on granted idle hardware
while a foreground request remains responsive; worker crash and overflow paths
are visible and safe.

### Phase 2: same-model replicas

- add exact-checkpoint replica planning and independent backend processes;
- implement affinity-aware load balancing and fallback to the verified primary;
- benchmark single-GPU and, where plausible, tensor-split replica candidates.

Exit gate: model identity and context are identical across the replica route,
cache affinity survives multi-turn agents, and aggregate workflow time beats the
serial baseline without a meaningful foreground regression.

### Phase 3: calibrated auto mode

- evaluate slots, replica, worker, and mixed candidates on the granted devices;
- persist the winning evidence under the full hardware/runtime fingerprint;
- surface the chosen topology and rejected alternatives in dry-run, TUI, and
  support output.

Exit gate: `auto` reproduces its choice from valid evidence, invalidates it on a
relevant change, and falls back to the verified single-primary launch when no
candidate wins.

## Acceptance matrix

- opt-in and device-scope tests prove ungranted GPUs are untouched;
- occupied-device and mid-run contention tests fail closed without eviction;
- parent-turn priority is preserved under maximum agent fan-out;
- required per-agent context is never reduced silently;
- worker-safe, replica-required, foreground, and reviewer routes cannot cross;
- sticky multi-turn agents demonstrate cache reuse;
- backend crash, OOM, startup failure, and context overflow have tested drain
  and fallback behavior;
- resume rejects an incompatible pool/model/topology unless explicitly forced;
- process shutdown leaves no worker, replica, router, or cgroup behind;
- status and support data explain where every request went and why;
- benchmark evidence compares workflow makespan as well as single-request
  tokens per second.

## Reference-rig hypothesis

On the current three-GPU development rig, the first core hypothesis is to leave
the primary 27B model and its full context on the RTX 3090 Ti and test a second
exact-model replica across the explicitly granted RTX 4070 and RTX 3060. That
preserves model capability and avoids the smaller-model specialization policy
question. It may still lose because of PCIe synchronization and the slower GPU,
so primary slots and alternate placements remain measured candidates. Smaller
workers stay a later explicit-policy experiment, not the assumed solution.

## Non-goals

- becoming a cluster or distributed-workflow orchestrator;
- taking unlisted GPUs automatically;
- routing by prompt-content heuristics;
- replacing Claude Code or Ultracode's workflow scheduler;
- weakening the existing Auto security-review lane;
- making the separate FreeToken experiment a dependency of this core feature.
