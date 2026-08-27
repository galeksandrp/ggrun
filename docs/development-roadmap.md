# ggrun core development roadmap

> Active engineering plan. This lane covers ggrun's GGUF/llama.cpp product
> only. External inference-engine experiments are tracked separately and are
> not release dependencies for this plan.

Status: active, 2026-08-27.

## Current baseline

- The last pushed baseline is `eb19f46` (`v3.2.8-18-geb19f46`).
- The 15 findings from the August update/TUI/CLI adversarial audit are closed.
  The last six interrupted fixes landed in `a9cc4b3`; they are not future work.
- Runtime graph-growth measurement, the functional/cache/workload promotion
  canaries, and post-health CUDA-OOM recovery are implemented. The
  `compute-preflight-plan.md` design note now describes the shipped behavior
  and keeps the long-context hardware case as an explicit release-matrix gate.
- The currently running DeepSeek launch uses a previously installed binary.
  The newer repo-local candidate can test each feasible sole-backbone owner and
  use sustained per-device imbalance to choose the one finalist. It is built
  and source-validated, but it has not replaced the live process or installed
  command.
- Do not use the current ~217 GB-RAM DeepSeek run as tight-fit acceptance. It is
  the roomy/performance case and currently exposes a saturated RTX 4070 while
  the larger 3090 Ti mostly waits. The preserved 128 GB, 1M-context,
  parallel-4 DeepSeek stress run remains the constrained-fit/stability
  evidence.
- A model server is currently using the machine. Hardware bandwidth must be
  calibrated only after the server is intentionally stopped and the GPUs are
  idle.

## Prepared hardware-window handoff

The installed command deliberately remains the binary used by the current live
session. The 2026-08-27 repo-local candidate is built at both `go/ggrun` and
`ggrun`; both copies have SHA-256
`95366d446cbfa2a80889c0d3dcb7a20e6d21bb630bff79b5c0b08531ae2317cf`.
Installing that candidate later affects future launches only and does not
replace a launcher or backend process already loaded. Once the operator opens
an idle hardware window, run:

```bash
scripts/core-hardware-window.sh check
scripts/core-hardware-window.sh plan
scripts/core-hardware-window.sh bandwidth
```

`check` was rerun against this candidate and refuses, as intended, because the
live ggrun, reviewer, and main llama-server still own the hardware. The script
never stops a process. Once idle,
`bandwidth` records the candidate/hash, dirty source patch, CPU/RAM/PCI/GPU
fingerprint, backend versions, full bandwidth detection, and the cached
follow-up under `.benchmarks/core-readiness/<UTC stamp>/`.

Source validation for this candidate passes 1,367 Go tests and the same 1,367
under the race detector, native and Windows vet, formatting and diff checks, 24
Python tests, all shell regression scripts, error-level ShellCheck, Bash syntax,
and Linux arm64, Darwin arm64, and Windows amd64 cross-builds. This is source
readiness, not a substitute for the idle real-hardware matrix.

The production-shaped bottleneck evidence and the separate FreeToken A/B
contract are preserved in the
[real-agent runtime comparison plan](agentic-runtime-comparison-plan.md).

The accepted next core milestone is tracked in
[Core standard-launch optimizer TODOs](core-standard-launch-todos.md). It keeps
the two product promises—fit the requested model safely, then converge on the
fastest validated agentic configuration—in the shared TUI/direct-CLI path. It
also records KV-fit completion, ordinary mmap recovery, and the selective
mmap/mlock experiment as a hardware-gated last resort.

## Active sequence

### 1. Repair and install the current developer build — completed 2026-08-24

Build the launcher from the current checkout and replace the older installed
command without rebuilding or replacing a working backend. Repair the local
`.bin/ggrun` developer link as part of the same operation; it currently points
to a missing repository-root binary.

Acceptance gate:

- `ggrun version` identifies a build containing the current core commits;
- `ggrun detect --bandwidth` is present;
- the installed command and `.bin/ggrun` both execute;
- the already-running server is not stopped or disturbed by the install;
- a scratch-home smoke test covers help, an unknown command, non-interactive
  invocation, and benchmark error handling.

Result:

- the tested candidate, repository-root binary, `.bin/ggrun`, and installed
  command are byte-identical;
- the previous installed launcher is recoverable at
  `~/.local/bin/ggrun.bak-before-eb19f46`;
- the existing ggrun parent and llama-server retained their original PIDs, and
  the server continued to return `{"status":"ok"}`;
- all acceptance checks above passed without rebuilding or replacing a backend.

### 2. Record idle hardware bandwidth

After the current server is stopped by the operator, run the bandwidth probe on
an otherwise idle host. The production profile must never be learned while a
model owns VRAM or is generating.

Acceptance gate:

- the profile fingerprint matches the exact CPU, RAM, GPU, and PCI layout;
- host memory and pinned H2D/D2H measurements are present for every NVIDIA GPU;
- a second detection run consumes the profile without re-measuring;
- stale, partial, corrupt, or hardware-mismatched profiles still fail closed;
- representative dry runs show the measured topology source in their placement
  evidence.

### 3. Run the real-hardware core regression matrix

Exercise the new code as a user would, not only through unit tests. Preserve
machine-readable results under `.benchmarks/` with the model identity, backend
build, hardware fingerprint, exact argv, context, concurrency, and timestamps.

Required cases:

1. A fully resident dense model on the fastest suitable GPU.
2. A heterogeneous multi-GPU MoE placement, including an offloaded-expert path
   and the measured 512/1024/2048 ubatch candidates.
3. A hybrid/recurrent or SWA model through cold prompt, append, older branch,
   and identical replay so checkpoint reuse is proven.
4. A parallel hybrid/recurrent agent launch exercising the valid `128/128`,
   `256/128`, `256/256`, `512/256`, and `512/512` batch/ubatch ladder.
   Preserve repeated cold-ingest-plus-cache-append workflow wall time,
   append-only diagnostics, reused/new tokens, aggregate prefill, aggregate
   decode, mixed foreground throughput, exact allocation admission, and every
   screened result. Do not mark a winner automatic-eligible until the full
   validation contract and clean relaunch pass.
5. A long prompt crossing at least one context-checkpoint boundary, followed by
   concurrent decode, to exercise runtime graph growth and post-health recovery.
6. A small odd-head-dimension GGUF proving automatic KV selection cannot choose
   an incompatible block-quantized cache type.
7. CLI journeys for invalid GPU selection, invalid model, occupied port,
   unknown command, bare benchmark, and non-TTY startup.

Acceptance gate:

- no unexplained OOM, restart loop, hang, or silent flag loss;
- a recovered placement is persisted only after its canary passes;
- cache append/branch/replay counters meet the existing canary thresholds;
- benchmark results distinguish load time, TTFT, prefill, decode, RAM, VRAM,
  cache reuse, and stability;
- any failure becomes a minimal regression test before its fix lands.

### 4. Reconcile engineering documentation and release state

Update the preflight design note to describe the implemented canary and
post-health recovery paths. Add an Unreleased changelog section covering every
post-v3.2.8 core commit, and verify that user documentation matches the actual
CLI and installer behavior.

Acceptance gate:

- no implemented feature is still labelled merely planned;
- changelog entries trace to the post-v3.2.8 commits;
- help, README, usage, troubleshooting, and installer examples agree;
- generated/recommendation data is current and reproducible.

Progress, 2026-08-24:

- the preflight design note now describes the implemented promotion canaries,
  measured runtime-growth accounting, and post-health recovery behavior;
- an Unreleased changelog section traces every post-v3.2.8 commit;
- the remaining help/README/usage/troubleshooting/install and generated-data
  reconciliation stays open until the real-hardware matrix establishes the
  final release behavior.

### 5. Cut and verify the next patch release

Only tag after the real-hardware matrix passes. The release gate includes the
ordinary Go/shell/Python suites, race tests, vet, formatting, cross-builds, the
release packaging jobs, and an upgrade test from the installed v3.2.8 build.

Acceptance gate:

- all release assets and `SHA256SUMS` are published successfully;
- a clean install and an in-place update both select a backend that starts;
- a failed update preserves the previous launcher, backend binary, source
  checkout, and user configuration;
- the installed version and repository tag agree;
- the release is exercised once through launch, health, canary, request, and
  graceful shutdown on the primary Linux/NVIDIA rig.

## After the patch release

Keep the next work ordered rather than starting all of it at once:

- first execute the accepted
  [core standard-launch optimizer](core-standard-launch-todos.md): finish KV
  fit, then prioritize the roomy-resident path for models that fit comfortably.
  Establish convergence there across fastest-device versus multi-device
  placement, batch/ubatch, automatic parallel, and KV before spending more
  optimization effort on extreme tight-fit models. Tight resident launches
  keep their conservative exact-proof path; ordinary mmap remains the final
  generic fit fallback;
- keep selective per-expert mmap/mlock pinning behind its hardware and evidence
  gates; it is only for supported MoEs that still exceed usable RAM plus VRAM,
  and is not part of automatic best until its acceptance matrix passes;
- only after the standard single-backend core is complete, implement the
  proposed [Workflow capacity planner](workflow-capacity-plan.md): use only
  user-granted idle devices, establish a deterministic per-agent route, then
  measure primary slots, same-model replicas, and smaller worker pools against
  end-to-end workflow time;
- expose hardware/profile evidence and verified launch state more clearly in
  the TUI and support report;
- refresh published placement benchmarks on the three-GPU reference rig;
- validate Metal on real Apple hardware, AMD/Vulkan on real AMD hardware, and
  Windows CUDA update/install paths;
- expand backend/model coverage only when a real checkpoint and repeatable
  regression fixture are available.

## Lane rules

- Core placement remains GGUF- and backend-evidence-driven.
- “Roomy,” “tight,” and “non-resident” describe the exact resolved launch on
  this hardware, not a parameter-count, quant, or model-file-size category.
- Verified spare capacity should be spent on performance for a roomy launch;
  tight-fit recovery rules must not become the default behavior for models that
  fit comfortably.
- No fixed memory cushion may replace a measured or backend-reported value.
- A cache entry is trusted only for the exact model/runtime/hardware identity
  encoded by its schema.
- A successful load alone is not a verified configuration; the functional and
  workload canaries remain mandatory.
- Workflow capacity cannot claim an ungranted device, route by prompt
  heuristics, or trade away foreground responsiveness without visible measured
  evidence.
- Mmap is a last-resort residency tier, not a free speed optimization. Global
  `--mlock` cannot make an oversized mapping fit; selective pinning must prove
  file-backed expert ranges, keep an unlocked fault window, and pass its
  hardware/workload gates.
- External-engine work cannot block, redefine, or silently change this release
  sequence.
