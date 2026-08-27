# Standard-launch optimizer theory and handoff

> Durable engineering record for the ordinary `ggrun` TUI and direct-CLI
> launch path. This document describes the GGUF/llama.cpp core only. FreeToken,
> fork discovery, replicas, and specialist-model orchestration are separate
> development lanes.

Status: source correction committed and canonical binary rebuilt, 2026-08-27.
Live performance acceptance still requires the next intentional model window.

## The contract in one sentence

Resolve the safest useful launch first, classify its measured remaining room,
then spend only proven room on one bounded performance experiment and keep a
configuration only when the same agent workload measures it faster.

There is one standard launch, not a user-facing "fit mode" and "fast mode".
Internally it has two ordered stages:

1. **Fit and stability.** Preserve explicit device, context, KV-quality, slot,
   batch, and mmap choices. Produce a complete placement, prove the exact argv
   where possible, recover monotonically if it does not fit, and reach a healthy
   serving baseline.
2. **Performance.** Only after the baseline has allocation evidence, classify
   it as roomy, tight, or non-resident. Calculate legal complete alternatives,
   select at most one automatic finalist, admit that exact argv without a
   recovery rewrite, and compare it with the baseline on identical agent work.

A performance hypothesis must never become the fit baseline merely because its
name or topology looks fast. In particular, `moe-owner-N` is a challenger
coordinate, not an automatic default-placement policy.

## Controller state machine

```text
request + explicit constraints + hardware + GGUF + backend identity
                              |
                              v
                    complete fit calculation
                              |
                   contained exact preflight
                              |
              +---------------+----------------+
              |                                |
          does not fit                       fits
              |                                |
   monotonic recovery ladder          healthy baseline load
   (recompute every argv)                       |
              |                       exact allocation ledger
              +---------------+----------------+
                              |
                              v
         classify resolved launch, not model file size
             |                 |                 |
           roomy              tight          non-resident
             |                 |                 |
   broader resident      same proven shape;   serve stable
   candidate frontier    denser GPU expert     fallback;
                         pack is monotonic      no auto reload
             |                 |
             +--------+--------+
                      |
            static phase-aware ranking
                      |
          one finalist with predicted gain
                      |
      exact-argv admission (no recovery rewrite)
                      |
       identical repeated agent-workload A/B
                      |
       correctness + cache + lifecycle gates
                      |
        promote winner or persist baseline-won
```

The static model is a search reducer. It is not proof that a configuration is
fast. The live agent workload is the performance authority.

## Residency classes

Residency is a property of the fully resolved launch on the hardware available
at that moment. It is not a parameter-count or quant-name class.

- **Roomy resident:** the baseline is resident and its exact device/host ledger
  leaves enough room for at least one materially different, legal candidate.
  Topology, physical microbatch, logical batch, and useful slot count may enter
  the bounded performance frontier.
- **Tight resident:** the baseline is resident but exact evidence does not
  authorize a topology reshuffle. Keep its allocation-proven shape. A
  same-shape batch/ubatch/slot neighbor or a denser GPU-expert pack may be tried
  only when its complete ledger remains feasible.
- **Non-resident:** ordinary resident placement cannot satisfy the constraints,
  so CPU/disk-backed mmap or another last-resort tier is required. Automatic
  launch serves the stable fallback and does not add another potentially
  minutes-long model reload for a low-confidence speed guess.

Recovery history and residency are separate. A launch can recover from one bad
argv and still be roomy after the safe argv is measured. Conversely, idle VRAM
on one card does not make a globally constrained launch roomy.

## llama.cpp controls: what they actually change

| Control | Backend meaning | ggrun consequence |
|---|---|---|
| `-c / --ctx-size` | Total server context capacity. With multiple slots the useful guaranteed capacity per active agent is approximately total context divided by slots. | Context is a quality/workload constraint and a major KV-memory input. It is not free capacity for parallelism. |
| `-b / --batch-size` | Maximum logical batch submitted to decode scheduling. | It caps how much prompt work the scheduler may assemble; raising it alone does not create a larger physical CUDA graph when ubatch is unchanged. Always require `batch >= ubatch`. |
| `-ub / --ubatch-size` | Maximum physical microbatch used to split a logical batch. | It directly changes graph/scratch allocation and often prefill occupancy. Larger can speed prompt ingest, but may consume VRAM that could otherwise hold KV or MoE experts. |
| `-np / --parallel` | Number of server slots/sequences. The server continuously combines eligible work into a shared batch. | It helps only when independent requests are runnable. It increases slot/KV state and reduces guaranteed context per agent; a single active request does not become twice as fast just because two slots exist. |
| continuous batching | The server builds one shared batch from prompt and decode tokens across slots and calls the model decode path when the batch is ready. | Batch, ubatch, parallel, request arrival shape, and prefix reuse must be measured together. Static token/s from an isolated prompt is insufficient. |
| `--tensor-split` with `--split-mode layer` | Assigns consecutive layer ranges to devices. Those layer ranges form a serial model path; it is not generic data parallelism. | Putting ordinary layers on every GPU can add slow-device and boundary costs. A device that merely stores routed experts is a different role from an ordinary-layer owner. |
| `--split-mode row` | Tensor-parallel row splitting where supported. | Do not generate it until the backend/device topology proves support and peer behavior. It needs a separate cost/admission model from layer split. |
| `-ot` tensor overrides | Places selected tensors (for ggrun, commonly routed experts) on named devices/CPU. | Expert storage is also expert execution placement. Current llama.cpp disables pipeline parallel when tensor overrides exist, so an MoE `-ot` plan must not be priced as an overlapped pipeline. |
| `--mmap` / `--no-mmap` | File-backed mapping versus resident loading behavior, subject to backend implementation. | mmap is a fit/reclaim policy, not a normal speed lever. Exact pageability observation must agree with the planned host ledger. |
| `--mlock` | Requests locking mapped pages in RAM. | Global mlock is not a solution for a model larger than usable RAM. Selective hot-expert pinning would require backend capability and remains a later last-resort experiment. |

The authoritative public descriptions are llama.cpp's
[server options and parallel-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[server scheduling notes](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md),
[multi-GPU documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md),
and [speed-bench instructions](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md).
The latter explicitly sweeps ubatch and requires batch to be at least ubatch.

The checked-out backend currently reinforces two crucial details:

- `.src/llama.cpp/src/llama-context.cpp` clamps `n_ubatch <= n_batch` and
  splits a logical batch into physical ubatches.
- Pipeline parallel requires multiple devices, full layer offload, layer split,
  offloaded K/Q/V, and no tensor overrides. Because ggrun's sparse-MoE placement
  uses `-ot`, owner candidates must be evaluated as serial layer paths plus
  routed expert work, not as overlapped pipeline stages.

Recheck these source invariants when the backend identity changes; ggrun caches
and measurements are backend-build scoped.

## Phase model

An agent workflow alternates between phases with different bottlenecks:

- **Cold prefill:** matrix-heavy prompt ingestion. Larger physical batches can
  raise arithmetic intensity and amortize weight reads, but graph memory,
  quantized kernels, MoE routing, and CPU experts can move the knee.
- **Cached append:** small prompt processing after prefix restoration. Cache-hit
  correctness and restore amount often matter more than headline cold-prefill
  throughput.
- **Decode:** usually low arithmetic intensity at small concurrency. Weight
  service and KV traffic tend to make memory bandwidth dominant. Multiple live
  sequences can share a weight pass, but contention and slot state are real.
- **Queueing/tool turns:** agent requests arrive, pause for tools, branch, and
  resume. Useful parallel width is workload concurrency, not all hardware the
  user owns.

This is why one utilization snapshot cannot name a universal bottleneck.
GPU-SM, memory-controller, host CPU, host-bandwidth, peer traffic, queue depth,
and active slots have to be associated with the phase being timed.

The broader inference literature supports this shape:

- [FlashAttention](https://arxiv.org/abs/2205.14135) shows why IO movement is a
  first-class performance term.
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) motivates
  iteration-level scheduling across requests.
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  shows how chunked prefill can reduce decode stalls.
- [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) explains why KV-memory
  fragmentation and sharing influence serving capacity.

These papers inform candidate generation and telemetry; they do not prove that
an upstream llama.cpp build implements the same scheduler or kernels.

## Sparse-MoE ordering model

For a model with `E` routed experts and `k` selected experts per token, a
bounded uniform-routing prior for the fraction of an expert tensor touched by
`B` independent tokens is:

```text
p_touch(B) = 1 - (1 - k/E)^B
```

This is only an expected ordering prior. Real routing can be skewed, kernels may
fuse work, and experts can be reused within a batch.

The working-tree cost estimate separates:

```text
T_backbone_gpu = sum_i(backbone_bytes_i / gpu_vram_bandwidth_i)

T_expert_gpu =
    sum_i(p_touch(B) * routed_expert_bytes_i / gpu_vram_bandwidth_i)

T_expert_cpu =
    p_touch(B) * cpu_routed_expert_bytes / host_memory_bandwidth

T_transfer =
    layer-boundary activations / link_bandwidth
  + remote-expert activation round trips / link_bandwidth
  + CPU-expert activation round trips / PCIe bandwidth
```

Important corrections encoded by this model:

- GPU-resident experts are not free just because they are sparse.
- CPU experts are host-memory-bandwidth work. Do not model all CPU-resident
  expert weights as if the complete weights stream over PCIe each token;
  activation transfers are a separate term.
- Ordinary layer split is a serial sum. An idle expert-storage GPU is not
  automatically evidence of bad balance.
- A saturated ordinary-layer owner can be the expected best topology if every
  alternative moves the backbone to a slower device.
- For prefill, ubatch provides a bounded arithmetic-intensity prior, while the
  larger expert-touch fraction is priced separately.

Relevant MoE systems work includes
[Fiddler](https://arxiv.org/abs/2402.07033) for CPU/GPU expert orchestration,
[MoE-Infinity](https://arxiv.org/abs/2401.14361) for activation-aware expert
prefetch/cache, and [PowerInfer](https://arxiv.org/abs/2312.12456) for hot/cold
residency. They are design references, not features ggrun can claim from
llama.cpp today.

## Candidate and promotion rules

Every candidate is a complete `placement.Compute` result. No performance
controller may patch `-b`, `-ub`, `--parallel`, `--tensor-split`, KV
placement, or `-ot` onto an old ledger.

Candidate generation must obey:

- Explicit user values are constraints. An automatic coordinate may move; an
  explicit coordinate may not.
- Context and KV quality remain fixed during ordinary performance A/B.
- `ubatch <= batch` is invariant before display, accounting, caching, and argv
  emission.
- Slot candidates must preserve the configured minimum useful per-agent
  context.
- Topology candidates are performance hypotheses only in the roomy lane.
- Tight candidates retain the exact proven shape, except a denser GPU-expert
  pack is a monotonic use of VRAM when its complete ledger fits.
- Non-resident automatic launch does not reload a giant model for speculative
  performance search.
- A candidate known infeasible from exact evidence is rejected. A conservative
  estimate may be challenged only by contained admission.

Automatic selection must:

1. Rank by predicted agent cost, never by candidate-name prefix.
2. Prefer exact allocation evidence only inside the estimate/noise tie.
3. Use phase-tagged utilization to explain a challenger, not to override a
   predicted regression.
4. Start the challenger with exact-argv admission. A preflight miss, startup
   OOM, compatibility rewrite, speculation rewrite, companion rejection, or
   mmap contradiction rejects that candidate rather than entering fit recovery.
5. Measure baseline and finalist with the same prompt bytes, concurrency,
   cache state, repetitions, and time budget.
6. Require a material measured improvement plus functional, cache, gateway,
   lifecycle, and clean-relaunch gates before automatic promotion.
7. Persist either the winner or a baseline-won result under the full scope so
   ordinary launch is finite.

The objective is agent turn time and workflow makespan. Prompt/decode token
rates, TTFT, queue time, cache restoration, and utilization are explanatory
components.

## Data plan

### Static inventory

Record once per hardware/backend identity:

| Data | Purpose |
|---|---|
| GPU physical ID, model, current free/total VRAM | Stable cache identity and allocation ceiling |
| measured VRAM bandwidth | Price ordinary and expert weight service |
| PCIe negotiated width/generation and directional bandwidth | Price activation boundaries and detect degraded links |
| peer-access/P2P matrix and ACS/IOMMU state | Gate future row/tensor-parallel candidates |
| host usable/free RAM and cgroup limit | Exact resident/mmap containment |
| measured host-memory bandwidth and NUMA affinity | Price CPU experts |
| CPU cores/topology | Detect CPU-expert and scheduler limits |
| backend commit, build flags, capability probe | Invalidate assumptions when llama.cpp changes |

NVIDIA's [CUDA best-practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
and [multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
are the references for minimizing host/device movement and validating P2P
topology rather than assuming nameplate PCIe behavior.

### Exact allocation evidence per candidate

Key by model shards/hash, exact backend build/capabilities, selected physical
GPU set/order, tensor split and overrides, context, KV K/V types, SWA mode,
batch, ubatch, slots, speculation/companions, mmap policy, and planner schema.

Persist:

- Per GPU: model, KV/context, graph/compute, runtime growth, other overhead,
  free-at-plan, required, residual slack, and role.
- Host: resident model/expert bytes, KV, graph/runtime/checkpoint reserve,
  reclaimable file-backed bytes, non-reclaimable bytes, and residual slack.
- Evidence level: static estimate, no-allocation plan, live allocated, runtime
  observed, or rejected boundary.
- Exact argv identity and rejection reason. Never transfer proof to an argv with
  a changed split, ubatch, slot count, KV mode, companion, or backend.

### Phase-tagged live telemetry

Collect at a modest sampling interval over the complete agent screen:

- phase: cold prefill, cached append, decode, queue/tool idle, load;
- per GPU: SM, memory-controller activity, VRAM used, power/clock throttling;
- process: CPU utilization by core/NUMA, RSS, major faults, disk bytes;
- interconnect: host-device and peer bytes when available;
- server: active/busy slots, queued requests, prompt tokens, generated tokens;
- cache: submitted, restored, recomputed, and branch/replay outcome.

Do not add telemetry merely because it is available. A signal earns a place in
automatic selection only after a test shows it changes a finalist decision
correctly.

### Agent-workload result

For each sample persist:

- exact candidate identity and allocation evidence;
- identical prompt bytes/tokens and requested concurrency;
- cold prefill, cached append, decode, TTFT, total turn time, and makespan;
- per-lane progress/fairness and foreground stall;
- functional output result, cache correctness, gateway health, clean stop,
  memory return, restart, and long-context canary;
- sample count, dispersion/noise, winner margin, and rejection cause.

The workload must include real agent properties: a large reusable system/project
prefix, appended turns, concurrent generation when requested, and branch/replay
behavior.

## Implementation map and current state

| Area | File / functions | 2026-08-27 state |
|---|---|---|
| Stable placement | `go/pkg/placement/placement.go`: `Compute`, `buildMoEOffload`, ubatch fit ladder | Existing fit-first baseline retained. `MoESplitOwnerGPU` is calibration-only; nil does not force an owner policy. |
| Candidate generation | `go/pkg/placement/calibrate.go`: `CalibrationCandidates`, batch/slot/topology recomputes | Bounded complete candidates exist, including one sole-backbone-owner hypothesis per feasible MoE GPU. |
| Ledger/classification | `go/pkg/placement/optimizer.go`: `BuildResourceLedger`, `AnalyzeCandidateFrontier`, `TightLiveCandidates` | Roomy/tight/non-resident states and exact-ledger gating exist. Complete guarded GPU/host peaks carry a placement hash, so backend-unlabelled model bytes remain exact only for the argv shape that produced them. Host slack for a topology that adds CPU experts is charged explicitly. |
| Static ranking | `EstimateStrategyCost` and MoE helper functions | Working tree prices backbone, GPU experts, CPU experts, and activation transfers separately. Owner names no longer receive priority. |
| Utilization signal | `AnalyzeDeviceBalance`, `SelectDeviceBalanceFinalist` | Idle MoE expert storage is ignored as a false balance peer. A telemetry-directed topology must also predict at least a small cost reduction. |
| Automatic controller | `go/cmd/ggrun/calibrate.go`: `automaticCalibrationFinalistPlan`, `runCalibration` | Baseline plus one finalist; identical bounded agent workload; measured promotion/baseline-won persistence. The old owner-name shortcut is removed. |
| Exact challenger admission | `go/cmd/ggrun/main.go`: `startLaunchExactAdmission` | Working tree rejects all currently identified argv-rewrite paths rather than recovering a challenger into a different candidate. |
| Persistence | `go/pkg/placement/calibrate.go` | Schema must be bumped whenever candidate/scoring/evidence semantics change; placement-bound aggregate evidence uses calibration schema 16. |
| Tests | repository validation | Focused four-package run: 889 tests. Full and race runs: 1,400 tests each. Build, vet, Windows vet, formatting, ShellCheck, Python/shell suites, and Linux ARM64, Darwin ARM64, and Windows AMD64 cross-builds pass. |

The working tree also contains a separate Grok/user fork-discovery lane. Do not
rewrite or discard `forksearch`, backend-selection, or TUI changes while
finishing this optimizer lane.

## Hardware evidence available now

The active Qwen3.8 Flash Next qwen4exp run is a stable baseline, not a settled
performance winner. It serves one explicit slot at ctx 262,144, batch/ubatch
`2048/256`, Q8 K/V, 21 GPU and 27 CPU expert layers, and no mmap. Aggregate
backend counters currently show about 117.44 prompt tok/s and 11.11 decode
tok/s; a completed 104k-context turn decoded at 10.61 tok/s and the following
turn was near 9.9 tok/s. Prompt work drove the RTX 4070 to about 79% SM while
the other devices were lighter; decode samples were low-to-moderate on all
three. That observation identifies phase imbalance but does not select a
different topology or slot count.

Its schema-15 record had zero exact frontier candidates even though guarded
peaks existed: this backend reported its GPU model allocation inside
`unaccounted`, while the legacy exactness gate required labelled model-share
rows. It chose `ubatch-2048` from a low-confidence estimate, could not admit it,
and restored `2048/256`. The working tree fixes the evidence model, preserves
the full host cgroup peak and full guarded breakdown across later KV-only
observations, and advances calibration schema 16. The next intentional relaunch
must show how the corrected exact ledger classifies the launch before another
performance claim is made.

The older DeepSeek-class runs answer two other questions:

Two runs of the same broad model class answer different questions:

- On the historical 128 GB host, the fit path was genuinely tight. The preserved
  1M-context, parallel-4 run completed one 60,020-token request plus three
  concurrent requests without OOM/restart/truncation. It measured about
  29.05 prompt tok/s and 5.88 decode tok/s. That is stability evidence.
- On the current roughly 217 GB host, a 146 GiB-class Q3 XL configuration is
  roomy despite a prior recovery. The observed launch used two slots, 1,048,576
  total context, BF16 GPU KV, batch/ubatch 128/64, seven GPU expert layers,
  36 CPU-expert layers, no mmap, and an approximately
  `0.23/0.67/0.09` layer split.
- During one 76,433-token cold prefill, CUDA0 (RTX 4070) repeatedly reached
  roughly 95-100% SM while the 3090 Ti and 3060 were mostly idle. Because
  `-ot` disables pipeline parallel, that is evidence of a serial
  ordinary-layer phase, not proof that every idle storage GPU should receive
  backbone work.
- A later sample used about 8.5 CPU cores while GPU SM was roughly
  17%/16%/0%, consistent with a different CPU-expert-limited phase.
- A cache canary processed 6,347 cold tokens at 24.16 tok/s and restored 6,343
  tokens on a strict append. A later branch could not restore recurrent/SWA
  state and was correctly not promoted.

Not proven yet:

- parallel 2 is faster than parallel 1 for the actual requested concurrency;
- ubatch 64, 128, 256, 512, or higher is the prefill knee;
- the 3090 Ti sole-backbone topology wins end to end;
- moving more/fewer experts to GPU wins after accounting for host bandwidth;
- the current working-tree finalist survives exact allocation and beats the
  baseline on identical agent work.

Do not turn the first utilization sample into any of those claims.

## Known risks that remain

1. The MoE touch model assumes independent uniform routing and uses an ordinal
   ubatch gain. It needs measured calibration, not more confident constants.
2. Link cost currently uses negotiated PCIe ceilings when direct measurements
   are absent. It does not yet have a complete directional P2P matrix.
3. Prefill transfer and kernel-compute terms are deliberately approximate.
   Phase telemetry is not yet rich enough to fit a hardware-specific roofline.
4. Device-balance sampling currently has GPU SM/memory data but not typed
   process-CPU, host-bandwidth, peer-traffic, or queue evidence in finalist
   selection.
5. A complete public matrix still needs dense, sparse MoE, recurrent/SSM,
   single-GPU, heterogeneous multi-GPU, tight, roomy, and non-resident hardware.
6. A failed sole automatic finalist currently settles the scoped baseline even
   when the static estimate was low-confidence. Placement-bound evidence makes
   the next selection better grounded, but admission-only fallback ordering and
   bounded re-evaluation policy still need an explicit design.

## Ordered next work

### P0 — make the correction safe to hand off

- [x] Restore the stable fit baseline; keep forced owner placement
  calibration-only.
- [x] Remove candidate-name priority from static and automatic selection.
- [x] Price GPU expert work separately from backbone, CPU experts, and
  activation transfers.
- [x] Stop treating idle expert storage as an ordinary-layer balance defect.
- [x] Make known challenger mutation/recovery paths fail exact admission.
- [x] Add focused placement/controller regressions; 889 focused tests pass.
- [x] Bind complete guarded allocations to their exact placement and retain
  unlabelled device/cgroup peaks without transferring them to another split.
- [x] Bump calibration schema to 16 so decisions produced by superseded
  scoring or exact-evidence behavior cannot be reused.
- [x] Add a generic exact-argv identity invariant and regression test.
- [x] Add branch-level tests for each exact-admission mutation class.
- [x] Run full Go tests, race tests, vet, formatting checks, shell/Python tests,
  ShellCheck, and supported cross-builds.
- [x] Rebuild and atomically sync the canonical `/home/mik/.local/bin/ggrun`.

### P1 — measure the current roomy MoE case

Run only when the hardware/model window is intentionally available:

1. Preserve the exact known-good baseline argv and cache identity.
2. Capture static inventory and an idle resource baseline.
3. Load the baseline once; capture exact model/context/graph/host ledgers.
4. Run the identical agent screen twice while sampling phase-tagged telemetry.
5. Let corrected static ranking choose exactly one finalist. Likely hypotheses
   include larger ubatch, denser GPU experts, a different useful slot count, or
   the 3090 Ti owner topology; do not preselect one by name.
6. Exact-admit and load only that finalist. Any rewrite or recovery rejects it.
7. Run the same two workload samples and lifecycle/cache gates.
8. Promote only for at least the configured material improvement outside noise;
   otherwise restore and persist baseline-won.
9. Preserve logs, exact argv, scope key, allocation rows, workload JSON, and
   telemetry together.

### P2 — turn measurement into a general optimizer

- Fit hardware-specific ubatch knees from successful exact scopes while keeping
  conservative priors for unseen public hardware.
- Add process-CPU and host-bandwidth evidence to distinguish GPU-backbone from
  CPU-expert bottlenecks.
- Add queue/busy-slot evidence before allowing slot width to outrank batch or
  topology.
- Measure P2P before generating row/tensor-parallel candidates.
- Expand the acceptance matrix and publish anonymized evidence fixtures.
- Finish ordinary mmap last-resort acceptance. Keep selective expert
  mmap/mlock parked until a supported backend and too-large-model window exist.

## Resume checklist

From the repository root:

```text
rtk git status --short --branch
rtk git diff --stat
```

Then inspect this document, `docs/core-standard-launch-todos.md`, and the
targeted diffs. Do not reset the dirty tree: it contains both the optimizer work
and a user-owned fork lane.

Validation commands must be run separately:

```text
cd go
rtk gofmt -w <only files intentionally changed>
rtk go test ./pkg/placement ./cmd/ggrun -count=1
rtk go test ./... -count=1
rtk go test -race ./... -count=1
rtk go vet ./...
```

Use the repository's documented validation/build scripts for the remaining
shell/Python/cross-platform checks. Never claim live optimality from the static
suite, and never replace the canonical binary until the full validation gate
passes.
