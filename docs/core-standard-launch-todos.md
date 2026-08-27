# Core standard-launch optimizer TODOs

> Active core tracker for ordinary `ggrun` launches from either the TUI or
> direct CLI. This is the GGUF/llama.cpp lane; it does not depend on FreeToken,
> AI Tune, or multi-model orchestration.

Status: active implementation, 2026-08-27.

## Implementation snapshot

The 2026-08-27 working tree contains the first core implementation and its
synthetic regressions. The repo-local candidate is rebuilt and passes 1,367 Go
tests, the full race run, vet, formatting, shell/Python regressions, ShellCheck,
and the Linux arm64, Darwin arm64, and Windows amd64 cross-builds. Checkboxes
below remain open under the tracking rule at the end of this document until the
work is committed and claims that depend on real hardware have preserved live
evidence.

| Work | Implementation state | Remaining acceptance |
|---|---|---|
| KVFIT-1 through KVFIT-7 | Implemented in the shared placement-backed automatic-context resolver, with exact adjacent-boundary and scope evidence | Commit plus live long-context/canary proof |
| KVFIT-8 | Synthetic coverage exists for selected/current-free GPUs, companion reservations, CPU-only, dense/MoE/recurrent shapes, explicit quality/max, and odd KV geometry | Finish the named public matrix and no-fit-oracle fixtures |
| KVFIT-9 | Boundary evidence is persisted and verified-config reuse is scoped | Capture load, cache, agent, and adjacent-rejection evidence on real hardware |
| PERF-1, PERF-3, PERF-5, PERF-9, UX-3 | Implemented: standard launch owns bounded search; candidates are complete placements; scope includes policy/capabilities; batch pairs preserve explicit intent | Commit and multi-hardware proof |
| PERF-2, PERF-6, PERF-7 | Implemented as baseline plus one calculated finalist, with a measured prefill pilot, identical budget-scaled cold+append scenarios, two samples, concurrent generation, mixed foreground traffic, lifecycle gates, delayed promotion, and a reusable baseline-won result | Add longer branch/replay and long-context hardware acceptance; quantify noise on public hardware |
| PERF-10 | Automatic 1/2/4 slot neighbors are implemented with useful per-slot context and full re-placement | Add capability-specific unified/partitioned KV A/B |
| PERF-4, PERF-8, PERF-12, UX-2 | Not complete | Implement measured headroom continuation, broader optional knobs, and bounded control UX |
| PERF-11 | Partially implemented in the rebuilt candidate: MoE topology candidates now include each feasible sole-backbone owner, every candidate is a full placement recompute, and sustained device imbalance can redirect the one live finalist by compute role | Run the intentional roomy hardware comparison, then add only capability-proven row/peer candidates |
| UX-1 | Implemented for launch, dry-run, dry-run JSON, TUI config screen, and `ggrun status` | Support-expert status remains the NanoBeige controller; launch inspect is `ggrun status` |
| ROOMY-1, ROOMY-2, ROOMY-3, ROOMY-4, ROOMY-6 | Implemented in source: exact residual slack classifies roomy; tight live-tests only the proven shape; topology ranking prefers a fitting fastest single GPU; batch/ubatch/slots are full recomputes; winner/baseline-won/boundary persist | Commit plus live roomy dense/MoE/recurrent proof |
| ROOMY-5 | Partially implemented in the rebuilt candidate: PCI-keyed SM sampling spans the complete agent trial; sustained idle devices become a typed bottleneck and can select one fully recomputed MoE backbone-owner finalist | Capture a matched live DeepSeek-class baseline/finalist comparison, then add process CPU, peer/host traffic, and queue telemetry only where it changes finalist selection |
| Exact-argv admission and long-load UX | Implemented and rebuilt: a recomputed argv must receive its own allocation evidence, lateral MoE split churn retains the exact proven placement, and 64+ GiB models warn before loading | Install for a future launch, then repeat the live MoE case after the current server is intentionally stopped |
| MMAP-1, MMAP-2, MMAP-4 | Implemented: production/preflight/recovery/daemon share capability-aware reclaim policy; unknown/anonymous loaders fail closed; mmap remains last-resort and consent-gated | Commit plus live resident/mmap/anonymous cases |
| MMAP-3 | Host ledger now separates exact reclaimable expert bytes from non-reclaimable runtime, KV, embeddings, and checkpoint reserve | Audit remaining backend-reported buffers/page tables/companions against live cgroup data |
| MMAP-5 through MMAP-7 | Not complete | Requires the storage/workload and real too-large-model acceptance window |
| PIN/P3 | Deliberately not started | Blocked on appropriate hardware/model artifacts as specified below |

## Live evidence: the same model in two different residency classes

### Current ~217 GB host: roomy/performance evidence, 2026-08-26

The current 146 GiB-class Q3 XL launch is **not** representative tight-fit
evidence. The same checkpoint fits this server comfortably: its guarded probe
peaked at about 121.3 GiB of a 196.7 GiB host limit, and the live host still has
about 106 GiB available. Recovery to a known-safe argv does not change that
classification; recovery history and current residual capacity are separate
facts.

The serving plan has 1,048,576 total context, two 524,288-token slots, BF16 GPU
KV, `128/64`, seven GPU expert layers, 36 CPU-expert layers, and no mmap. Its
layer split is `0.23/0.67/0.09`: the 3090 Ti holds most bytes, but storage is not
the performance objective.

The first production argv changed an exact-probed `0.27/0.65/0.09` tensor split
to an unprobed `0.26/0.64/0.10` split. CUDA2 then failed a 617.26 MiB graph
allocation after 4m26s; health reported the failed attempt at 4m36s. The source
fix now binds allocation evidence to the exact argv, re-verifies genuinely
denser replans, and retains the proven split when only a lateral split changes.

The cache canary processed 6,347 cold tokens at 24.16 tok/s and restored 6,343
tokens on a strict append. The backend then explicitly forced full processing
for the older branch because recurrent/SWA state was unavailable, restoring
zero tokens; ggrun correctly left the profile degraded instead of promoting it.
The first real Claude request is a cold 76,433-token prompt at about 20.4 tok/s.
That is roughly 62–65 minutes of prefill. A live utilization sample makes the
immediate bottleneck more specific: the server holds about 95 GiB RSS but has
no disk reads or major faults, consumes only about one CPU core, saturates the
RTX 4070 at 94–100% SM, and leaves the 3090 Ti mostly at 0–6% and the 3060
mostly idle. Repeated live snapshots still show the RTX 4070 at 95–100% SM while
the 3090 Ti and RTX 3060 wait. The recovered `ubatch=64` may be conservative,
but the directly observed defect is serial/imbalanced layer ownership. The
source frontier now includes a fully recomputed sole-backbone-owner hypothesis
for every feasible GPU and scores sparse MoE topology by ordinary-layer role,
not just stored bytes. The running installed binary predates that change, so
this observation identifies the roomy performance miss; it does not validate
the fix.

The bottleneck is phase-dependent rather than one universal utilization
number. A later compute snapshot on 2026-08-27 showed the main process using
about 8.5 CPU cores while GPU SM was roughly 17% / 16% / 0%. With 36 CPU MoE
layers, that is consistent with a CPU-expert-limited phase. The calculated
frontier already prices CPU-expert bandwidth and prefers a feasible denser GPU
expert pack; the repeated live workflow still decides whether that, a larger
microbatch, more useful slots, or an owner topology actually wins. The current
source does not yet record process CPU/peer/queue telemetry as typed promotion
evidence, so that remains part of ROOMY-5 rather than being implied by GPU SM.

`parallel=2` is therefore not yet proven faster. It provides a second slot and
halves guaranteed context to 524,288 tokens per agent, but the observed request
uses one slot while the other is idle. A matched p1/p2 workload comparison must
wait for an intentional hardware window; do not reload the live model merely to
manufacture that number.

### Historical 128 GB host: fit/stability evidence, 2026-06-22

The earlier reference run is the representative constrained-fit case. On the
3090 Ti 24 GB + RTX 3060 12 GB + RTX 4070 12 GB + 128 GB RAM rig,
DeepSeek-V4-Flash UD-IQ4_XS (~128 GiB) ran at 1M context and parallel 4 with the
3090 Ti owning the dense path and the two smaller/slow-link cards storing three
complete expert layers each. It completed a 60,020-token request plus three
concurrent requests without OOM, restart, truncation, or health failure;
measured prefill was 29.05 tok/s for the main request and decode was 5.88 tok/s.
That preserved run validates the stable fit path under pressure. It must not be
relabelled as evidence that the current roomy host is optimized. Full details
remain in [launch-performance.md](launch-performance.md#deepseek-v4-long-context-service-stress).

## Product contract

The standard launch has two jobs:

1. **Fit:** run the requested model stably with useful agent context and the
   user's quality constraints on the hardware actually available.
2. **Go fast:** when it fits, use measured spare capacity to minimize real
   agent-workflow wall time instead of leaving safe performance unused.

These are phases of one core engine, not optional `ai-tune` or `--calibrate`
product modes. TUI and CLI must resolve to the same decision and evidence.

“Mid-size” is not a model-file threshold. The core must classify the *resolved
launch* from exact evidence:

- **Roomy resident:** the requested context, KV quality, companions, and model
  are resident with enough measured VRAM/RAM headroom to test at least one
  materially larger batch/ubatch or a faster legal placement. This is the next
  primary optimization path. ggrun should actively spend the headroom to
  minimize agent-workflow time.
- **Tight resident:** the launch is resident but one or more devices or host
  ledgers are near their verified boundary. Preserve the known-good shape and
  admit only monotonic, contained experiments; unused capacity on another GPU
  does not by itself authorize a riskier split.
- **Non-resident:** normal resident placement cannot satisfy the constraints;
  enter the mmap/offload recovery ladder visibly and optimize only inside the
  residency tier that actually survives.

Model parameter count, quant name, or summed nameplate VRAM must never choose
the class. The same checkpoint can be roomy on one machine and tight on
another, and companions or an explicit context can change the answer.

On a cold model/backend/hardware identity, “best” means the best safe estimate
available without delaying launch indefinitely. After bounded measurement, it
means the fastest **measured and fully validated** explored candidate. ggrun
must not claim a theoretical global optimum it has not measured.

The objective is lexicographic:

1. preserve correctness, explicit choices, minimum per-agent context, and
   required KV quality;
2. reject OOMs, hangs, cache corruption, unstable relaunches, and unacceptable
   foreground regressions;
3. minimize representative agent-turn time and workflow makespan;
4. use lower memory, disk I/O, and power as tie-breakers inside noise.

Raw prefill or decode speed is diagnostic, not the objective. Queue time, TTFT,
prefix reuse, tool-turn latency, foreground progress, and correct completed
tasks belong in the score.

## Residency and optimization ladder

```text
constraints + capabilities
          |
          v
stable placement baseline
          |
  exact allocation + residual headroom
          |
   +------+------+----------------+
   |             |                |
roomy resident  tight resident   no resident fit
   |             |                |
   v             v                v
spend headroom  preserve proven  ordinary file-backed mmap, only
on speed        shape; bounded   where CPU experts are reclaimable
                experiments
   |             |                |
   +------+------+----------------+
          |
validate and cache the winner
          |
          v
selective mmap + per-expert mlock only for a supported MoE
that still exceeds VRAM + usable RAM
```

The final box is experimental. It is not the backend's global `--mlock`:
locking an oversized whole mapping defeats reclaim and prevents fit. The idea
locks only workload-hot expert ranges and leaves the cold tail mmap-backed.

## What is already in place

- [x] TUI and direct CLI converge on the same launch/placement engine.
- [x] Hardware, GGUF, backend capabilities, companion reservations, and user
  flags feed placement.
- [x] Stable single/multi-GPU dense, MoE offload, dense CPU offload, and
  CPU-only strategies exist.
- [x] Exact allocation preflight, bounded OOM re-planning, runtime-growth
  evidence, functional/cache canaries, and verified-config reuse exist.
- [x] KV measurements reject known parallel and `swa-full` poisoning cases.
- [x] The first `ctx=fit` defects were fixed: Claude mode no longer turns fit
  into model maximum, and RAM is no longer counted as unbounded GPU KV space.
- [x] Ordinary mmap planning distinguishes resident CPU-expert memory from a
  reclaimable working set, has a consent gate, and has a reclaim-band design.
- [ ] The current core has not yet proven the fastest standard launch across
  ggrun's public hardware/model/backend matrix.
- [ ] KV fit and mmap recovery still need the end-to-end gates below.

## P0 — finish KV fit

`ctx=fit` must be a placement result, not aggregate-memory arithmetic bolted on
before placement.

- [ ] **KVFIT-1 — define fit.** Choose the largest useful automatic context in
  the fastest feasible residency tier under the selected model, KV quality,
  slots, backend, and user-granted devices. Never count RAM as GPU KV capacity
  or silently lower an explicit number/`ctx=max`.
- [ ] **KVFIT-2 — use effective inventory.** Size against current free memory
  on selected devices after backend/CUDA overhead, compute buffers,
  reviewer/draft companions, runtime growth, and other claims—not summed
  nameplate VRAM across every detected GPU.
- [ ] **KVFIT-3 — make it joint.** Recompute context, KV type/placement,
  `swa-full`, checkpoints, batch/ubatch, slots, speculation, tensor split, and
  expert placement as one candidate. No late overlay may invalidate the ledger.
- [ ] **KVFIT-4 — find the boundary.** Within each legal KV/residency class,
  retain the largest context granule exact admission accepts plus evidence for
  the adjacent rejected granule. Do not snap to powers of two.
- [ ] **KVFIT-5 — preserve semantics.** Forbidden KV types do not exist.
  Quality changes require explicit policy. Preserve user devices, parallel,
  and backend unless a visible recovery rule permits a change.
- [ ] **KVFIT-6 — unify entry points.** TUI, CLI, Claude defaults, dry-run,
  memory-probe, recovery, daemon/reload, and verified restore use one resolver
  and display one effective answer.
- [ ] **KVFIT-7 — scope evidence.** Key exact shards, backend build/capabilities,
  hardware policy, context, slots, KV/SWA, batch pair, speculation, companions,
  and planner schema. Reject evidence that cannot be normalized safely.
- [ ] **KVFIT-8 — regression matrix.** Cover resident and context-reduced dense,
  heterogeneous multi-GPU, CPU-only, MoE expert/KV competition, iSWA/recurrent,
  odd KV head dimensions, explicit max, occupied devices, and no fit oracle.
- [ ] **KVFIT-9 — live proof.** Record that the chosen context passes load and
  agent/cache canaries and that the next granule fails admission or crosses a
  visibly slower residency tier.

Exit: TUI, CLI, dry-run, and support agree; the chosen launch survives the
long-context gate; automatic fit never pushes a resident KV cache to host
silently.

## P1 — make resident standard launch converge on fastest

### Roomy resident fast path — next implementation priority

This is the common model-fits-easily case and the cleanest place to finish the
second product promise. It comes before more specialization for extreme
tight-fit models.

- [ ] **ROOMY-1 — evidence-based admission.** After exact allocation, compute
  residual VRAM/RAM and identify which batch/ubatch, KV, slot, and placement
  changes have real headroom. Do not use a fixed model-size or free-VRAM cutoff.
- [ ] **ROOMY-2 — fastest baseline.** Compare single fastest-device residency
  against multi-device placement when both fit; do not pay peer/scheduling
  overhead merely to fill every GPU. Keep context and quality fixed.
- [ ] **ROOMY-3 — spend graph headroom.** Raise ubatch and logical batch through
  complete placement recomputes until exact admission reaches the boundary;
  live-test only the best calculated finalist against the stable baseline.
- [ ] **ROOMY-4 — workload-owned slots.** Search 1/2/4 slots only when parallel
  was not explicit, preserve useful context per agent, and rank end-to-end
  workflow makespan plus foreground latency—not aggregate tok/s alone.
- [ ] **ROOMY-5 — observe device balance.** Sample per-device utilization,
  service time, CPU use, peer/host traffic, and queueing during the bounded
  workload. A full card that does no work is storage, not a successful speed
  plan. Feed that evidence into the next topology candidate.
- [ ] **ROOMY-6 — finite promotion.** Persist winner, baseline-won, rejected
  finalist, and the explored headroom boundary under the exact profile scope so
  later launches start immediately and do not repeat settled work.

Exit: on representative comfortably resident dense, MoE, and recurrent models,
the standard launch either promotes a faster validated configuration or records
that the stable baseline won. No explicit choice changes, and no run searches
again without a scope change, reset, or maintenance request.

### Controller and evidence

- [ ] **PERF-1 — core ownership.** Build a standard-launch optimizer shared by
  TUI/CLI. Reuse benchmark/calibration primitives without requiring legacy
  `--ai-tune`, `--tune`, or explicit `--calibrate`.
- [ ] **PERF-2 — agent workload.** Version a privacy-safe workload with cold
  long prefill, cached append, branch/replay, short tool turns, concurrent
  decode, and foreground traffic during fan-out.
- [ ] **PERF-3 — performance profile.** Key exact model/quant/shards, backend
  binary/build/capabilities, OS/driver, CPU/RAM/GPU/topology, selected devices,
  context/KV/SWA, slots/KV mode, batch pair, placement, speculation, workload,
  and explicit-intent bits.
- [ ] **PERF-4 — measured headroom.** Derive residual VRAM, RAM, compute, queue,
  foreground, and I/O headroom. Free VRAM permits a test; it does not prove a
  faster result.
- [ ] **PERF-5 — feasible candidates only.** Every candidate passes full
  placement and exact allocation/lifecycle admission. Do not patch argv after
  accounting.
- [ ] **PERF-6 — bounded search.** Screen neighbors of the stable baseline,
  fully run finalists, repeat the winner, and stop inside noise or budget.
- [ ] **PERF-7 — conservative promotion.** Require correctness, cache reuse,
  agent turns, foreground progress, long-context stability, clean relaunch, and
  exact identity. OOM/canary failure revokes the winner.
- [ ] **PERF-8 — reuse and continue.** Start immediately from a valid winner.
  Otherwise start the safe estimate and bound launch-time work; continue deeper
  search only in an explicit maintenance/idle window.

### Candidate dimensions

| Dimension | Gain | Cost/coupling | Treatment |
|---|---|---|---|
| logical batch `-b` | more prompt work per scheduler pass | long prefill may starve decode | agent fairness workload |
| microbatch `-ub` | occupancy and prompt speed | graph VRAM, MoE placement, checkpoints | full placement + preflight |
| `--parallel` | less queueing, more aggregate work | latency, KV/sequence state, per-agent context | measure makespan + foreground |
| unified/partitioned KV | dynamic slots or prefix sharing | independent prefixes may lose; semantics vary | capability-detect and A/B |
| device/layer/expert placement | faster resident compute | buffer entry fees and host traffic | exact device ledger/topology |
| CPU threads/affinity | faster host experts | contention/oversubscription | mixed-load measurement |
| flash/backend kernels | memory or attention speed | shape/correctness support | capability + canary gate |
| speculation | fewer target steps | draft memory and acceptance overhead | optional measured candidate |
| mmap/offload | makes an impossible model run | faults, disk bandwidth, tail latency | recovery, not resident tuning |

- [ ] **PERF-9 — coupled batch search.** Search legal `(batch, ubatch)` pairs,
  keep `ubatch <= batch`, recompute placement/checkpoints, protect explicit
  sides, and score prompt speed plus decode fairness.
- [ ] **PERF-10 — parallel/KV search.** Search workload-appropriate slots with
  real unified/partitioned semantics. Report total and guaranteed per-agent
  context.
- [ ] **PERF-11 — topology search.** Compare legal single/multi-device plans
  using measured bandwidth and compute buffers. No reference-rig GPU names or
  fixed VRAM thresholds in policy.
- [ ] **PERF-12 — optional knobs last.** Threads, affinity, flash, fork kernels,
  and speculation enter only after memory-shaping core is stable.

### User-visible behavior

- [ ] **UX-1 — explain it.** TUI, launch, dry-run, status, and support show
  baseline/winner state, context per agent, batch pair, slots/KV mode,
  placement/residency, evidence age, rejected finalists, and bottleneck.
- [ ] **UX-2 — bounded control.** Provide off/estimate/converge, finite budget,
  re-evaluation, and reset without making legacy commands the normal path.
- [ ] **UX-3 — explicit intent wins.** Model, quant, context, KV, devices,
  backend, batch, ubatch, parallel, mmap, and speculation remain constraints.
- [ ] **UX-4 — no routine consent noise.** Measurement inside an authorized
  residency tier does not repeatedly ask. Disk-backed residency, live probes,
  new devices, or quality relaxation stay explicit/fail-closed.

Exit: the promoted standard config beats the stable baseline outside noise on
agent wall time, preserves foreground/cache gates, survives relaunch, and is
reused without re-searching.

## P2 — harden ordinary mmap as final generic fit fallback

This is the existing file-backed CPU-expert path when non-reclaimable working
state fits but resident CPU expert bytes do not. It follows every normal
resident plan and precedes selective pinning.

- [ ] **MMAP-1 — re-audit current main.** Production, preflight, daemon, and
  recovery must share `memory.high` reclaim and whole-host `memory.max` rules.
- [ ] **MMAP-2 — prove pageability.** Gate on observed/backend capability that
  CPU experts remain file-backed, not only a backend name. Anonymous CUDA-host
  expert buffers are ineligible.
- [ ] **MMAP-3 — exact host ledger.** Count dense/shared/router tensors, CPU KV,
  checkpoints, graphs, pinned buffers, page tables, companions, page cache,
  and reclaimable experts exactly once.
- [ ] **MMAP-4 — last resort only.** Prefer a stable resident placement. First
  disk-backed transition is visible; unattended use needs explicit policy.
- [ ] **MMAP-5 — real workload tax.** Measure narrow/spread, cold/warm cache,
  long prefill, decode, and concurrent agents; record backend major faults,
  disk I/O/latency, reclaim, TTFT, and workflow wall time.
- [ ] **MMAP-6 — storage viability.** Detect measured storage behavior and warn
  or refuse a cold path that cannot meet the declared profile. No fixed mmap
  penalty assumption.
- [ ] **MMAP-7 — live acceptance.** Re-run the too-large-for-resident-RAM case
  that exposed the reclaim bug and an anonymous-expert refusal case.

Exit: accepted mmap survives spread traffic without cgroup OOM; anonymous
experts are refused; a resident winner is never silently replaced by mmap.

## P3 — selective mmap + mlock for models beyond RAM + VRAM

This imports the development-lab disk-MoE memory idea without making its
planner/worker routing a core dependency. It applies only to expert-major MoE
layouts whose loader exposes file-backed expert ranges. Dense, row-interleaved,
and anonymous CUDA-host layouts are ineligible.

It is blocked on appropriate hardware/model artifacts. Do not implement or
benchmark the large-model experiment on the 128 GB daily driver just to close a
checkbox.

### Phase 0 — cheap stop/go evidence

- [ ] **PIN-0.1 — capable route.** Prove exact backend/model/quant support,
  expert mapping, and pageability; fail closed for anonymous expert buffers.
- [ ] **PIN-0.2 — contained profile.** One short-context slot, small ubatch,
  capped reasoning/output, mmap + reclaim band, and no global `--mlock`.
- [ ] **PIN-0.3 — stop/go corpus.** Run ten privacy-safe agent planning briefs;
  record cold/warm TTFT, decode, disk reads, major faults, reclaim, correctness,
  warmup, and an expert-spread variant.
- [ ] **PIN-0.4 — stop if compute loses.** Below 0.5 warm tok/s, stop. Treat
  0.5–0.8 as research-only; build pinning only with a credible >=0.8 tok/s path.

### Phase 1 — calibrated pin-set

- [ ] **PIN-1.1 — per-expert ranges.** Extend GGUF metadata to
  `(layer, expert_id) -> [offset,length]`, validate shards/layout, and reject
  row-interleaving.
- [ ] **PIN-1.2 — activation evidence.** Trace bounded `{layer, expert_id}` data
  over a privacy-safe 64–256-turn agent corpus, keyed by model/quant/backend/
  workload/schema.
- [ ] **PIN-1.3 — residency ledger.** Reserve OS/runtime, dense/shared/router/
  embedding, KV, checkpoints, scratch, companions, VRAM weights, and an unlocked
  fault window before filling RAM with hot ranges.
- [ ] **PIN-1.4 — range control.** Mmap the original GGUF; prefault and `mlock`
  hot ranges; `madvise(MADV_COLD)` the tail; cleanly unlock on rollback/exit.
  Do not split or prune the model in version one.
- [ ] **PIN-1.5 — domain shift.** Record hit/miss, disk, TTFT, and turn wall
  time. Rebuild only on sustained drift; unseen experts remain on disk.
- [ ] **PIN-1.6 — same-corpus A/B.** Beat ordinary mmap outside noise. A miss
  rate >=25% rejects the pin profile.

### Phase 2 — product gate

- [ ] **PIN-2.1 — routing stays optional.** Prove single-model residency first.
  A planner+worker consumer needs separate routing/quality validation.
- [ ] **PIN-2.2 — no short-bakeoff auto selection.** Require a full hardware,
  model, and workload profile plus explicit authorization.
- [ ] **PIN-2.3 — TUI eligibility.** Require warm 2k-brief TTFT <15 s, warm
  decode >=1.0 tok/s, stable reclaim, and an end-to-end agent win.
- [ ] **PIN-2.4 — fail closed.** Bad evidence, unsupported layout/backend,
  inadequate fault-window RAM, failed `mlock`, disk stall, OOM, or miss storm
  falls back to ordinary mmap or a resident smaller model—never expert deletion
  or silent thrashing.

Exit: a model exceeding usable RAM+VRAM serves a bounded declared agent role
with every expert reachable, stable reclaim, and measured benefit. Until then
this is blocked/experimental and outside “automatic best.”

## P4 — public hardware and release proof

- [ ] **PORT-1 — capability policy.** Generate candidates from detected CUDA,
  ROCm/HIP, Vulkan, Metal, and CPU/backend semantics; unsupported knobs vanish.
- [ ] **PORT-2 — portable fixtures.** Use synthetic profiles and recorded
  evidence from multiple hardware shapes. Reference-rig data tests rules; it
  does not become a rule.
- [ ] **PORT-3 — hardware matrix.** Validate resident dense, heterogeneous
  multi-GPU, offloaded MoE, recurrent/SWA, parallel-agent, and mmap cases; add
  AMD/Metal/Windows evidence when hardware exists.
- [ ] **PORT-4 — fault injection.** Cover stale profiles, flag changes, busy
  devices, partial shards, no oracle, crash/OOM, disk stalls, and interrupted
  relaunch.
- [ ] **PORT-5 — release contract.** Document cold estimate vs converged winner,
  per-agent context, inspection/reset, and all last-resort warnings.

## Tracking rules

- Close a box only with implementation commit, focused regression test, and
  preserved hardware evidence for fit/performance claims.
- Convert reference-rig observations into capability/measurement rules before
  making defaults.
- A successful load alone is neither stability nor performance acceptance.
- Never weaken quality, useful context, foreground response, cache reuse, or an
  explicit setting silently to win.
- Replicas/worker pools stay in the workflow-capacity plan. FreeToken stays in
  its separate lane.
- AI Tune is legacy/optional. Manual calibration can remain diagnostic, but
  standard launch owns the final fit and performance decision.
