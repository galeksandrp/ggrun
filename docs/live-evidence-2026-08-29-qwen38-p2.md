# Qwen3.8 Flash Next Q3 parallel-2 live evidence — 2026-08-29

This is a privacy-safe measurement record for the ordinary ggrun Claude Code
launch. It retains counts, timings, placement, and utilization only; no prompt,
response, tool input, or process environment is copied here.

## Identity and launch

- Evidence cutoff: `2026-08-29T23:51:36Z`.
- Source revision: `40423d3407edbfece9700f1d5a54ea527b6b990a`.
- Live controller binary SHA-256:
  `75eeee2d5dfd4e96ba84cec36637c0a108e06922e72c307c1fe2cb6c495b7218`.
- Model: `Qwen3.8-Flash-Next-UD-Q3_K_XL`, qwen4exp backend.
- Resolved shape: 262,144 total context, two 131,072-token slots,
  `batch=128`, `ubatch=64`, GPU `q5_1` KV, 24 CPU-MoE layers,
  `tensor-split=0.29,0.61,0.10`, `no-mmap`, CRAM 12,800 MiB, 16 recurrent
  checkpoints.
- The user explicitly supplied `parallel=2`. Automatic candidate generation
  therefore preserved p2 and did not run a controlled p1 challenger.
- Main and dedicated Qwen3.5-2B reviewer health endpoints remained healthy.

## Calibration evidence

The bounded p2 baseline measured:

| Metric | Result |
|---|---:|
| Two-lane workload makespan | 64.10 s |
| Aggregate decode reported by the bounded sample | 22.9 tok/s |
| Aggregate prefill reported by the bounded sample | 62.5 tok/s |
| Mixed foreground rate | 1.8 tok/s |
| Slowest of two lanes | 64.15 s |

The small bounded sample did not represent the later long-context fan-out. It
was retained only as an admission-level baseline; it did not prove p2 faster
than p1.

The calculated `batch=8192,ubatch=8192` finalist failed exact admission with a
2,304 MiB CUDA0 deficit. `batch=4096,ubatch=4096` also failed exact admission,
with a 7,089 MiB CUDA0 deficit. Recovery correctly refused to mutate either
challenger into a different configuration, but the persisted calibration record
currently says only `unavailable`; the typed failure detail is present in the
controller output and should also be persisted.

## Real agent-workload request evidence

The router JSONL contained 140 completed records at the cutoff:

| Route/outcome | Count |
|---|---:|
| Main-model requests | 120 |
| Main-model HTTP 200 | 48 |
| Main-model aborted | 72 |
| Reviewer accepted | 1 |
| Reviewer rejected as `invalid-verdict` | 19 |
| Aborts at 55–70 seconds | 19 |
| Aborts at 590–610 seconds | 53 |

For the 48 successful main-model requests:

| Distribution | p50 | p95 |
|---|---:|---:|
| Router queue | 5 ms | 579,408 ms |
| Time to first byte | 59,765 ms | 611,210 ms |
| Total request time | 609,143 ms | 2,098,223 ms |

The router aggregate at the cutoff reported 68 successful and 72 aborted
requests, a maximum/mean queue of 599,999/305,479 ms, 34,011 output tokens,
1,885,084 cache-read tokens, and 1.38 aggregate output tok/s over recorded decode
time. Aborted requests accumulated 32,154,990 ms in the router queue but only
811,169 ms in service. This proves queueing, rather than backend crashes, caused
most lost work.

The two timeout classes have different causes:

1. The reviewer produced a fast answer, but the exact observed text was
   `<block>yes` while the parser accepts only one complete
   `<block>yes</block>` or `<block>no</block>` token. Each rejected verdict fell
   back to the saturated main model; the caller then abandoned it around 60 s.
2. Fan-out requests that could not acquire either main slot were abandoned at
   almost exactly 600 s. The live controller's claim that extra agents queue
   "without timing out" is therefore false for this Claude Code path even
   though ggrun injects its maximum timeout settings.

## Long mixed-phase behavior

During the aligned live samples both slots were active on approximately
38k–62k-token prompts. The status path reported about 57.8 prefill tok/s and
0.4 decode tok/s. Direct server timing during simultaneous prefill and decode
showed the prefill near 58–61 tok/s while the competing decode fell through
roughly 1.2 tok/s. This is the harmful case the bounded launch sample missed.

For comparison only—not a controlled single-coordinate A/B—the immediately
preceding p1 Qwen3.8 Q3 evidence on this machine reported about 132–134 prefill
tok/s, 12.8–16.4 decode tok/s, 15.8–17.5 mixed foreground tok/s, and roughly
48.8 s makespan. Its placement also differed (`batch=2048`, `ubatch=256`, 22
CPU-MoE layers), so it is strong operational evidence that the current p2 shape
is worse, not final causal attribution to parallel width alone.

## Aligned hardware sample

A 30-second sample during the harmful mixed phase showed:

- GPU0 (RTX 4070, PCIe Gen3 x16): typically 89–98% reported SM, roughly
  10–12.5 GB/s PCIe RX and 1.1–1.5 GB/s TX, about 58 W.
- GPU1 (RTX 3090 Ti, PCIe Gen3 x16): usually 0–16% SM with brief 30–35% samples,
  little PCIe traffic, about 118–130 W.
- GPU2 (RTX 3060, PCIe Gen3 x8 active): usually 0–13% SM with brief 23–30%
  samples, little PCIe traffic, about 43–46 W.
- Main process: 134.1% average CPU, 63.6 GiB RSS, no disk I/O, no major/minor
  faults during the sample.
- Host: normally 93–96% CPU idle, no swap activity, about 134 GiB available
  RAM. It is a single-socket, single-NUMA-node i9-10940X system; all 65,088 MiB
  reported by `numastat` for the main process was local to node 0.

This corrects the earlier single-phase diagnosis. Host-resident experts still
create the shared-memory/CPU leg, but in this concurrent prefill phase the
observable choke point moves to the GPU0/PCIe path: PCIe Gen3 x16 receives
roughly 10–12.5 GB/s while the other GPUs and most CPU cores wait. The optimizer
must classify bottlenecks by phase; neither "DDR only" nor "GPU compute only"
is an adequate model.

Direct memory-controller/cache counters could not be collected because the
host has `perf_event_paranoid=4` and the session lacks `CAP_PERFMON`. No claim
of exact DDR channel saturation is made from this run.

## Consequences for the core scheduler

Do not promote p2 from capacity arithmetic or a small two-lane sample. The next
implementation must preserve two separate concepts: allocated slots and active
compute lanes.

1. Record per-active-slot phase (`cold prefill`, `append`, `decode`), new versus
   cached prompt tokens, short-window rates, queue age/deadline, and per-device
   PCIe/SM utilization.
2. On a measured host-offloaded MoE, do not admit a large cold prefill beside a
   latency-sensitive decode when the observed decode-regression guard is
   breached.
3. Measure decode+decode and cache-hot append+decode separately. Permit two
   compute lanes only where same-workload makespan improves and foreground
   latency remains inside its guard.
4. Keep reviewer/utility work off the main queue. Make the reviewer emit or
   normalize one canonical verdict without weakening fail-closed semantics.
5. Treat incoming request deadlines as scheduler evidence. Do not promise
   indefinite queueing until the real Claude/Workflow 600 s cancellation path
   is removed or kept alive correctly.
6. Persist typed finalist-admission failures so an `unavailable` result is
   explainable and safely reusable.

Hot-expert caching remains a later capacity improvement. Phase-aware admission
is required even with hot experts because cold prefill, PCIe concentration, and
foreground latency do not disappear automatically.
