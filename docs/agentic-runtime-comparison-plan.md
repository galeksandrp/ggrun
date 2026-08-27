# Real-agent runtime comparison plan

> Measurement lane for ggrun/llama.cpp versus FreeToken. This is not a
> FreeToken integration plan and does not block the active ggrun core release.

Status: protocol preparation active, hardware execution waiting for an idle
machine and a compatible FreeToken checkpoint, 2026-08-24.

## Question being answered

The useful question is not which engine wins a short synthetic decode. It is:

> On the same machine, how quickly and reliably can the same coding agent
> finish the same verified work while preserving foreground responsiveness?

FreeToken's native decode benchmark is still a useful control, but it measures
one warmed AIME request. The primary comparison must run real tool-using agent
loops, long histories, cacheable follow-up turns, fan-out, cancellation, and a
repository test oracle.

## Live ggrun baseline

The currently running Qwen3.8-27B Q4_K_XL Claude Code session already supplies
a production-shaped baseline. A read-only snapshot on 2026-08-24 found:

- one main-model admission slot active, one request queued, and a hard limit of
  one;
- 3,813 recorded requests across 127 conversation keys: 2,884 successful, 926
  cancelled before first byte, and three HTTP 400 responses;
- for successful requests, queue p50 5.04 seconds, p95 1,059 seconds, and max
  2,077 seconds over the whole run; the latest 100 records had queue p50 21.8
  seconds and p95 593 seconds;
- 170.5 accumulated successful-request hours waiting for admission versus 15.1
  hours in prefill and 55.2 hours in streamed decode;
- 74.1% of prompt tokens reused from cache, about 795 evaluated prompt tokens/s,
  and 27.9 output tokens/s over completed requests;
- a ten-second sample of the active roughly 57k-token turn delivered about
  59.9 output tokens/s while the RTX 3090 Ti held 86-91% SM utilization, 75-79%
  memory utilization, and roughly 445 W; the RTX 4070 and RTX 3060 remained
  idle;
- the backend's pass-cost estimator reported a 97.4% fixed share and projected
  27.4, 101.4, and 184.7 aggregate tokens/s at batch sizes 1, 4, and 8. Those are
  hypotheses from observed pass cost, not measured concurrent results.

The present bottleneck is therefore **admission capacity during agent fan-out**.
Once admitted, the main request saturates its assigned GPU; it is not waiting
on the CPU, a backend queue, or model loading. The router holds the excess work,
which is why llama.cpp itself reports zero deferred requests while ggrun reports
a queue.

This investigation also found that the router summary counted every
pre-first-byte cancellation as decode time. The source now excludes aborted and
failed requests from model-work throughput while retaining their queue and
cancellation totals. The live process still runs the older binary and is not
modified by that fix.

## Two comparisons, not one misleading score

### Model-controlled product comparison

Use the same base model lineage and the nearest quality-equivalent formats each
engine actually supports. ggrun consumes GGUF; FreeToken primarily consumes HF
safetensors or FTW. A GGUF Q4 versus NVFP4/FP8 result is therefore a product
comparison, not a pure kernel comparison, and the quantization difference must
stay in the result manifest.

The first candidate is Qwen3.8-27B: the ggrun Q4_K_XL checkpoint is installed,
and the current FreeToken source contains Qwen3.8-27B NVFP4 handling. That
FreeToken path must pass a load and quality canary before it is accepted because
it is not yet listed in the upstream supported-model table. The conservative
fallback is to acquire both Qwen3.6-27B variants documented by each engine.

### Best-supported-system comparison

Run each system's recommended model format and measured runtime policy on the
same explicitly granted hardware and power envelope. This answers what a user
can achieve, but it must be reported separately because model and quantization
quality can differ.

## Controlled variables

Every paired run pins and records:

- base model, checkpoint revision, quantization, tokenizer, chat template, and
  reasoning/tool parser;
- agent CLI and Ultracode workflow revision;
- repository fixture commit, task text, initial files, and test command;
- context and output limits, sampling values, thinking/effort setting, tool
  permissions, network policy, and maximum agent fan-out;
- physical GPU grant, CPU affinity/threads, RAM policy, power limits, driver,
  CUDA, backend build, and engine commit;
- cold or warm state, including model page cache, kernel cache, prefix cache,
  and expert-cache calibration.

Each task starts from a disposable worktree of the same fixture commit. The
agent may edit only that worktree. A run is invalid if it reuses another run's
repository mutations, changes the oracle, silently falls back to a cloud model,
or uses hardware outside the manifest.

## Workload suite

1. **Short serial repair** — diagnose and fix one localized defect, run the
   supplied tests, and explain the cause. This establishes cold TTFT, tool-call
   correctness, and baseline task quality.
2. **Cross-file implementation** — implement a bounded feature touching source,
   tests, and documentation. This exercises ordinary multi-turn coding rather
   than a one-response benchmark.
3. **Long-history branch and replay** — inspect a large fixture, make an edit,
   receive a follow-up that preserves the prefix, then an older-branch follow-up
   and an identical replay. This measures prefix/state reuse and recomputation.
4. **Tool-call rewrite loop** — make the agent issue a tool call whose echoed
   result changes representation before the next turn. This is the direct test
   for FreeToken's semantic tool-call anchor and ggrun's backend checkpoint
   behavior.
5. **Ultracode fan-out** — run a fixed eight-agent workflow plus periodic
   foreground probes. Record workflow makespan, foreground TTFT p50/p95/max,
   per-agent completion, fairness, cancellation, and cache locality.
6. **Cancellation and recovery** — cancel queued, prefill, and decode requests;
   then immediately submit a foreground repair. No orphan process, leaked slot,
   corrupted cache, or prolonged maintenance state is allowed.

The suite uses repository tests and machine-checkable task assertions as the
primary oracle. An LLM judge may be supplementary, never the only pass/fail
signal. Run at least one warm-up and three measured repetitions per condition;
report every result plus median and tail latency rather than only the best run.

## Metrics and evidence

Primary outcome metrics:

- task pass rate, repository test result, invalid/malformed tool calls, harmful
  or out-of-scope edits, and human intervention count;
- end-to-end workflow makespan and useful successful tasks per hour;
- foreground TTFT p50/p95/max under fan-out.

Runtime diagnostics:

- request queue/admission time where the engine exposes it, TTFT, streamed
  decode duration, prompt/output/cache-read tokens, cache-hit share, and
  cancelled/failed request phase;
- cold/warm load time, prefill and aggregate decode rate, active concurrency,
  expert-cache policy/misses where available, and cache rebuild/eviction events;
- peak VRAM/RAM, GPU utilization/power, CPU utilization, PCIe traffic, crashes,
  OOM recovery, and clean shutdown.

A client-side collector is the comparable source for end-to-end timing because
FreeToken's request ring reports duration and TTFT but not a separate per-request
queue interval, while ggrun's router does. Engine-native statistics remain
valuable supplementary evidence and must not be forced into false equivalence.

Store each run under:

```text
.benchmarks/agentic-ab/<UTC stamp>/
  manifest.json
  task-results.jsonl
  client-events.jsonl
  engine-requests.jsonl
  hardware.csv
  server.log
  agent.log
  patch.diff
  oracle.log
  summary.json
```

The manifest includes exact argv and environment keys but redacts secret values.
No benchmark claim is made without its artifact directory.

## Hardware-window matrix

Run the single-engine controls first on the RTX 3090 Ti, one engine at a time:

1. ggrun current verified placement, one slot;
2. FreeToken one request with its validated automatic policy;
3. each engine at measured concurrency candidates 2 and 4 while retaining the
   task's full per-agent context;
4. FreeToken offload versus hybrid only after `ft bench bw` has produced a
   fingerprint-matching profile.

FreeToken's semantic tool-call checkpoint is opt-in in the inspected source
(`--enable-special-token-ckpt`, default false). Run product defaults first, then
repeat the tool-call rewrite case with that flag as a named feature-on variant;
do not silently enable it only for the favorable comparison.

Then test ggrun's core workflow-capacity hypotheses separately:

1. additional primary slots only if full per-agent context and foreground
   latency remain valid;
2. a second exact-model replica across the user-granted RTX 4070 and RTX 3060;
3. alternate multi-GPU placement only when its measured makespan beats the
   independent primary-plus-replica topology.

Do not fold those multi-GPU ggrun results into the one-GPU engine comparison.
They answer the separate product question of how well ggrun uses the whole
granted rig.

## Readiness and remaining gates

Ready now:

- live ggrun request evidence and corrected summary accounting;
- a stable A/B contract, workload categories, metrics, and artifact schema;
- ggrun's experimental single-GPU FreeToken process/API adapter;
- a pinned upstream FreeToken inspection point,
  `bd372b630a028e3faa51f4ab0ef6a98c2f2de501`.

Still required before execution:

- create and version the disposable repository fixtures and exact Ultracode
  workflow inputs;
- implement the client-side event collector and manifest writer;
- install FreeToken in an isolated environment and record its exact version;
- acquire or convert the chosen HF/FTW checkpoint and pass load, output-quality,
  tool-parser, and context canaries;
- stop the current model intentionally, prove the host is idle, record bandwidth
  profiles, and run the guarded matrix.

FreeToken engine changes, cache resizing, and semantic-anchor ports remain in
the separate parked development lane. Measurements may inform ggrun core, but
they do not silently activate that integration work.
