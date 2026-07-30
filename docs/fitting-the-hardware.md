# Fitting the hardware

ggrun's job is to find the fastest *stable* configuration for agentic work on the
hardware actually present, by measuring rather than guessing. This document records
the principles that fall out of that goal, and the measurements behind them.

Everything below was measured on one rig — RTX 3090 Ti 24GB (x16), RTX 3060 12GB
(x1), RTX 4070 12GB (x4), 128GB RAM — running Laguna-S-2.1-UD-Q4_K_XL against Claude
Code. Read the numbers as one data point and the *principles* as the durable part.

## The objective is turn time, not throughput

Prefill and decode tok/s are not the goal. The goal is wall time for one agentic
turn, and optimising the components separately gets the answer wrong.

| config | prefill | decode | reuse | turn |
|---|---:|---:|---:|---:|
| no swa-full, moe 23 | **153.5** | 11.15 | 0% | 174 s |
| swa-full, KV on CPU, moe 22 | 127.4 | 5.75 | 91% | 117 s |
| swa-full, KV on GPU, moe 30 | 112.2 | **12.16** | 87% | **54 s** |

The config with the best prefill *and* the best decode-per-token at the time was the
slowest overall. Prefix reuse moved a 18,400-token prefill to ~1,700, which is worth
more than every throughput gain available. A planner that optimises tok/s picks the
174 s row.

**Principle: rank configurations by modelled turn time, with reuse in the model.**

## Prefix reuse is binary, not gradual

On an interleaved sliding-window model the server either resumes from a prior prompt
or re-processes all of it. There is no partial credit: when no context checkpoint
qualifies, `n_past` is set to 0 and a 91% prefix match is discarded whole.

Measured: 0.0% reuse across 229,583 tokens without `--swa-full`, 82–91% with it.
Checkpoints *were* being created (9–26 per run) and were rejected every time for
overshooting the resume point.

`--swa-full` avoids the reset by construction: it sets `size_swa = size_base`, so the
windowed cache never wraps, `seq_pos_min` stays 0, and the guard that triggers the
reset never fires. This is llama.cpp server behaviour, not model-specific — it
applies to every iSWA architecture (gemma2, gemma3, cohere2, exaone4, llama4, plamo3).

**Principle: on iSWA models, reuse is a placement input, not a runtime accident.**
The flag that enables it must be visible to the planner, because it multiplies KV by
4x–8x depending on architecture.

## KV and experts compete for the same VRAM

`--swa-full` gives every layer full-depth KV instead of only the non-windowed ones.
On Laguna that is 48 layers instead of 12 — a 4x multiplier that lands entirely on
the split owner, because all attention layers live there.

```
ctx 262144, q4_0, swa-full   KV = 13,824 MiB   ->  --n-cpu-moe 30
ctx 131072, q4_0, swa-full   KV =  6,912 MiB   ->  ~5 expert layers recovered
ctx 262144, q4_0, no swa     KV =  3,456 MiB   ->  ~7 recovered, reuse lost
```

Every MiB of KV is a MiB unavailable to experts, and expert layers are what keep
decode on the GPU. Context size is therefore not a free parameter: it is priced in
expert layers.

**Principle: context size, KV quality and swa-full are one joint decision with
expert placement, not four independent knobs.**

## The bottleneck moves, and low utilisation is the tell

With KV on CPU, the split owner ran at 72–87% SM and was the constraint. With KV on
GPU and 30 of 48 expert layers pushed to CPU, the picture inverted:

```
GPU0  sm 24%   GPU1  sm 2%   GPU2  sm 2%   CPU 49% of 8 threads
```

Nothing is saturated. That signature means serialisation: the GPU waits while the
CPU computes its expert layers and vice versa. Neither device can exceed roughly its
share of the layer split.

**Principle: when no device is saturated, the fix is rebalancing layers, not a
faster device.** Utilisation below ~60% everywhere is a placement bug, not a hardware
limit.

## PCIe bandwidth is rarely the constraint; compute buffers are

In `--split-mode layer` a layer's KV lives on the GPU that owns it and that GPU
computes its attention, so KV never crosses the bus. Only activations cross, at layer
boundaries:

```
prefill (ub 512):  512 tok x ~8 KB x 2 boundaries ~= 8 MB
                   x4 gen3 (~3.5 GB/s) -> 2.3 ms against 4.56 s of compute = 0.05%
decode:            1 tok x ~8 KB x 2 ~= 16 KB -> 5 us against 82 ms/token = 0.006%
```

Even the x1 card costs ~0.2%. What actually makes a second compute GPU expensive is
the compute buffer: a split owner reserves ~4,267 MiB against ~99 MiB for an
expert-only GPU. On a 12 GB card that entry fee costs about three expert layers, and
promoting the 4070 to hold a third of the KV nets **-3 expert layers on GPU overall**.

**Principle: weight multi-GPU decisions by compute-buffer cost, not link width.**
ggrun already models this (`expertOnlyComputeReserveMB`); the point is that link
width is the wrong first-order term.

## Measurement beats estimation, and measurements can be poisoned

ggrun's estimate for Laguna's KV was 9,238 MiB against an actual 13,864 — a 4.6 GB
error feeding every placement decision. Reading the geometry back from the backend's
own load-time output fixed it exactly.

But a measurement is only valid under the conditions that produced it. A `--swa-full`
launch prints every cache at the same depth, so the layer split it reports is
degenerate (`48,0,0`), and the bytes-per-token rate it yields (55,296) is 4x the
plain-launch truth (13,864). Recorded blindly, that poisons every subsequent plain
launch.

**Principle: record a measurement with the conditions it was taken under, and refuse
to record one whose conditions erase the thing being measured.**
