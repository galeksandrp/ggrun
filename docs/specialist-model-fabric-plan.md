# Specialist model fabric plan

> Exploratory orchestration idea above the
> [Workflow capacity planner](workflow-capacity-plan.md). A large primary model,
> ideally a capable MoE when that is the best verified fit, plans and synthesizes
> work while ggrun executes typed tasks on a local fleet of smaller specialists,
> fast generalists, replicas, or additional sequence slots.

Status: parked idea, 2026-08-24. This draft did not fully capture the intended
product direction and must not drive implementation or release work. Revisit it
with the user from first principles only after the current ggrun core roadmap is
finished.

## Vision

The primary model should act as the local lead engineer: understand the user's
goal, break it into tasks, decide which tasks may be delegated, inspect the
results, and produce the authoritative answer. ggrun should act as the model
fabric underneath it: place model processes, advertise proven capabilities,
route typed requests, enforce priority and isolation, and measure which topology
actually completes the workflow fastest.

```text
                         foreground conversation
                                    |
                                    v
                    large primary / MoE orchestrator
                         plan, judge, synthesize
                                    |
                         typed delegation envelope
                                    |
                                    v
                  ggrun capability and capacity router
                 /              |              |       \
                v               v              v        v
       same-model replicas  fast generalists  specialists  reviewer
        authoritative tier   many short jobs   code/vision/  safety lane
                                              research/etc.
                 \              |              /
                  +------ results + provenance -+
                                    |
                                    v
                    primary verification/synthesis
```

This is broader than merely increasing `--parallel`. The fabric can use:

- several instances of the same model for quality-preserving concurrency;
- several instances of a smaller fast model for high-throughput mechanical
  work;
- additional slots inside one backend when shared weights and batching win;
- different specialist checkpoints for tasks where their measured capability
  is better than a generalist's;
- a mixed topology when it beats every single-strategy candidate.

## Responsibility boundary

The large model or Workflow owns **semantic orchestration**:

- decomposition and dependencies;
- whether a task is safe to delegate;
- the capability and quality required;
- review of uncertain or consequential output;
- final synthesis and user-facing decisions.

ggrun owns **execution orchestration**:

- verified model and capability inventory;
- exact GPU/RAM placement and process lifecycle;
- backend pools, replicas, slots, queues, health, and cache affinity;
- deterministic routing from declared requirements;
- admission control, foreground reservation, fallback, and observability;
- calibration from workload evidence.

This boundary matters. ggrun must not read an arbitrary prompt and decide that a
smaller model is probably good enough. The primary model explicitly delegates a
typed task; ggrun chooses only among backends proven compatible with that type.

## Typed delegation contract

Workflow agents need a small machine-readable task envelope. The exact wire
format depends on what Claude Code and the Workflow runtime expose, but the
contract needs at least:

```text
task class          search | summarize | code-edit | test | vision | review | ...
required abilities  tools, modality, language/domain, structured output
quality tier        primary-required | replica-required | worker-safe
context requirement minimum input/output window
parallel safety     independent | ordered | exclusive
affinity            workflow, agent, repository, and conversation identity
verification        primary, specialist-pair, deterministic-check, or none
fallback policy     queue, same-tier fallback, primary fallback, or fail
```

The existing Workflow PreToolUse hook is the natural place to preserve or add
this metadata if the runtime supports it. The gateway must receive a stable
marker that survives inline and named workflows, retries, and resume. Until
that exists, only Claude Code's existing explicit model-tier aliases may use a
smaller pool.

Delegation metadata is policy, not trusted content from a worker. A worker
cannot promote itself to an authoritative tier or request broader tools.

## Capability registry

Each backend pool exposes a manifest derived from configuration and proven by
canaries. Model names alone are not evidence of capability.

The registry records:

- exact model artifact, checksum, quantization, template, and backend build;
- role and quality tier approved by the user;
- modalities, tool-call format, languages, and task classes;
- usable context and output limits at the selected concurrency;
- device placement, memory footprint, slots, and replica count;
- measured prefill, decode, workflow-task throughput, and queue latency;
- health, recent failures, and task-specific canary results;
- whether outputs require primary or paired verification.

Capabilities can come from a shipped catalog, local configuration, or a prior
measurement, but claimed abilities remain unavailable until the corresponding
canary passes for the exact model/runtime identity.

## Pool types

### Primary orchestrator

The primary is the strongest verified model for planning and synthesis. A large
MoE is an attractive orchestrator when its active-parameter speed, context,
tool use, and reasoning pass the workload canaries, but the architecture does
not hard-code MoE as a requirement.

Foreground turns, ambiguous tasks, cross-task reasoning, consequential code
decisions, and final synthesis stay here. At least one admission lane is
reserved for the foreground conversation.

### Same-model replicas

Replicas run the exact primary checkpoint and template. They absorb
authoritative-tier agents without a quality downgrade. ggrun distributes new
agents to the least-loaded healthy replica and keeps each multi-turn agent
sticky to its chosen backend for KV reuse.

A replica may occupy one GPU or a verified multi-GPU placement. Tensor splitting
across mismatched idle GPUs is a measured candidate, never an automatic choice.

### Fast generalist pools

A fast generalist handles many short, explicitly worker-safe tasks. It may be
scaled in two independent dimensions:

- **replicas:** load the model on several GPUs for true simultaneous decode and
  failure isolation;
- **slots:** share one copy of the weights among several sequences when context,
  KV memory, batching, and aggregate throughput still pass.

The planner benchmarks replica count and per-replica slot count together.
Blindly maximizing either can reduce per-agent context or saturate one GPU.

### Specialist pools

Specialists are small models selected for demonstrated capability, not simply
size. Candidate roles include:

- repository search and retrieval planning;
- code editing in a particular language or framework;
- test generation and failure triage;
- long-context extraction and summarization;
- vision, OCR, or document understanding;
- structured data conversion;
- first-pass code review or critique.

This list is extensible. A deployment need not install every role, and a fast
generalist may cover several roles better than separate tiny checkpoints.
Calibration decides whether specialization earns its memory and operational
cost.

### Reviewer and verifier lanes

The existing Auto security reviewer remains isolated from ordinary worker
capacity. Correctness verification may use deterministic tools, a second
specialist, or the primary model, but it must not weaken the security lane or
allow two copies of the same weak answer to masquerade as independent evidence.

## Scheduling policy

Routing first filters by hard requirements: quality tier, capabilities,
modality, context, tool format, user policy, and health. It then chooses among
eligible backends using measured queue delay, throughput, cache affinity,
memory pressure, and the task's latency goal.

Priority is explicit:

1. foreground primary turn;
2. required security or correctness verification;
3. dependency-unblocking workflow tasks;
4. ordinary parallel workers;
5. speculative, shadow, or calibration work.

Independent Workflow tasks may execute concurrently. Dependency order remains
the Workflow orchestrator's responsibility. Backpressure queues work instead of
starting unaccounted processes, and lower-priority jobs cannot consume the
foreground reservation.

## Verification and trust

Small models are accelerators, not authorities. Every returned result carries
provenance: model identity, backend, task class, timing, route and fallback
history, and verification state.

The primary model receives only the context necessary to judge the result and
can request a retry at a stronger tier. Deterministic outputs should be checked
with deterministic tools where possible—for example parsing, compilation, or
tests—before spending another model call.

Tool and data access follow least privilege:

- a worker receives the smallest useful prompt and repository scope;
- credentials and unrelated conversation history are not copied by default;
- write-capable tasks require the Workflow's existing permission policy;
- worker output is treated as untrusted input until its verification contract
  is satisfied.

## Configuration direction

The final interface should support both an automatic policy and a declarative
pool configuration. A conceptual configuration is:

```yaml
fabric:
  enabled: true
  orchestrator: primary
  foreground_reserve: 1
  pools:
    - name: fast
      model: <installed-or-catalog-model>
      devices: [<user-granted-device>]
      replicas: auto
      parallel: auto
      capabilities: [search, summarize, mechanical-edit]
      quality: worker-safe
    - name: primary-replica
      model: primary
      devices: [<user-granted-device-set>]
      replicas: auto
      quality: replica-required
```

This is a behavior sketch, not a committed file schema. The TUI should build
the same configuration, preview the process topology and memory budget, and
show which capabilities are measured versus merely configured.

## Adaptive selection

The planner optimizes end-to-end job completion, not isolated tokens per
second. Its evidence set includes:

- foreground TTFT and decode regression;
- complete Workflow makespan and critical-path time;
- task success and verification rate by capability;
- queue delay, cache hits, retries, fallbacks, and failed delegations;
- aggregate GPU utilization, VRAM/RAM peaks, PCIe cost, power, and stability.

The fabric may learn that one fast generalist is better than several
specialists, that replicas beat slots, or that a particular task must remain on
the primary. Those decisions are cached only for the exact model, backend,
hardware, concurrency, capability policy, and benchmark identity.

Runtime learning adjusts routing weights within user policy; it does not
silently approve a new model capability or lower a task's quality tier. Shadow
comparisons are opt-in and run only on spare capacity.

## Delivery sequence

### Layer 1: capacity substrate

Implement the [Workflow capacity planner](workflow-capacity-plan.md): explicit
device grants, backend pools, placement reservations, foreground priority,
health draining, affinity, metrics, and exact routing markers.

### Layer 2: homogeneous fast pool

Run multiple replicas and calibrated slots of one fast generalist. This proves
pool scheduling and gives immediate value for high-fan-out workflows without
the complexity of heterogeneous capability selection.

Exit gate: the same workflow completes faster than the serial baseline, the
foreground latency budget holds, and resume/cache affinity are correct.

### Layer 3: primary-model replicas

Add exact primary replicas for authoritative parallel agents and mix them with
worker pools under one scheduler.

Exit gate: authoritative agents never cross into worker-safe pools, and mixed
fan-out improves critical-path time without starving primary synthesis.

### Layer 4: capability-aware specialists

Add the registry, typed task envelopes, per-capability canaries, least-privilege
context, and at least two genuinely different worker roles.

Exit gate: routes are explainable and deterministic, each specialist beats the
fast-generalist control on its approved task class, and incorrect results
escalate through the declared verification policy.

### Layer 5: evidence-driven auto fabric

Calibrate topology, replica count, slots, task routing, and fallbacks from real
workflows. Persist and explain the winning plan, invalidating it on relevant
model/runtime/hardware changes.

Exit gate: automatic selection is reproducible, every route has provenance, and
the verified single-primary configuration remains the safe fallback.

## Reference-rig hypothesis

For the current three-GPU rig, the first topology to measure is:

- RTX 3090 Ti: large primary/orchestrator;
- RTX 4070: stronger fast generalist with enough context for substantial
  worker tasks;
- RTX 3060: another fast replica or a narrower specialist/reviewer role.

An alternative is a same-model primary replica across the two smaller GPUs.
The fabric should choose it only if its quality-preserving concurrency offsets
cross-GPU synchronization and produces a better workflow critical path. Exact
models, quantizations, slots, and roles remain calibration outputs.

## Acceptance gates

- no backend uses a device outside the user's explicit grant;
- the primary remains usable and prioritized under maximum fan-out;
- task envelopes survive inline, named, resumed, and retried workflows;
- capability claims are bound to exact artifacts and passing canaries;
- foreground, authoritative-replica, worker-safe, and security-review traffic
  cannot cross tiers;
- slots never reduce usable task context below the declared requirement;
- affinity and cache reuse work across multi-turn agents;
- worker errors, bad output, OOM, or disappearance trigger the declared visible
  fallback without killing the primary;
- deterministic verification and primary review are recorded in result
  provenance;
- dry-run, status, TUI, and support output explain the full topology and every
  routing decision;
- a representative Ultracode matrix proves improvement in total makespan and
  task success, not only synthetic throughput.

## Non-goals

- letting ggrun invent semantic task classes from prompt text;
- turning every visible model or GPU on automatically;
- training or fine-tuning models as part of routing;
- replacing Claude Code/Ultracode's workflow dependency graph;
- treating majority vote among weak replicas as guaranteed correctness;
- making the separate FreeToken lane a dependency of the fabric.
