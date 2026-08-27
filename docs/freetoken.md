# Experimental FreeToken adapter

ggrun has a deliberately narrow adapter for
[FreeToken](https://github.com/FlashML-org/FreeToken), an Apache-2.0 native
MoE serving engine. It is an opt-in comparison path, not another llama.cpp
flag dialect.

Further integration work is deliberately parked in a
[separate development lane](freetoken-development-lane.md). The active
[ggrun core roadmap](development-roadmap.md) has no FreeToken dependency.

Install FreeToken in its own Python environment first:

```bash
uv pip install "freetoken[accel]"
ft --version
```

For FreeToken's `hybrid` CPU/GPU expert policy, also run its own real-kernel
calibration once:

```bash
ft bench bw
```

That cache belongs to FreeToken and is intentionally separate from `ggrun
detect --bandwidth`: ggrun measures topology ceilings for its placement model,
while FreeToken measures its expert kernels to choose its runtime policy.

Then select exactly one physical NVIDIA GPU and launch a checkpoint directory,
FTW directory, or Hugging Face model ID:

```bash
ggrun freetoken Qwen/Qwen3.6-35B-A3B --gpu 1
```

On a host with only one NVIDIA GPU, `--gpu` may be omitted. On a multi-GPU
host it is required. ggrun sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES` so FreeToken sees only that card.

The small set of mapped options stays explicit:

```bash
ggrun freetoken Qwen/Qwen3.6-35B-A3B \
  --gpu 1 --ctx 65536 --parallel 1 --moe-backend auto
```

Native `ft serve` options go after `--`:

```bash
ggrun freetoken Qwen/Qwen3.6-35B-A3B --gpu 1 \
  -- --memory-ratio 0.86 --moe-cache-auto
```

Use `--dry-run` to print the exact isolated environment and argv without
starting a server. `--ft-bin` selects an environment-specific `ft` executable;
`FREETOKEN_BIN` provides the same default.

## What the adapter does

- verifies that the executable identifies itself as FreeToken;
- requires one detected NVIDIA GPU and hides every other device;
- refuses an occupied port instead of accepting another server's health check;
- waits for FreeToken's structured `/health` state to reach
  `maintenance=serving` and then verifies `/v1/models`;
- owns the whole frontend/worker process tree and stops it on shutdown;
- exposes FreeToken's native OpenAI and Anthropic endpoints unchanged.

## What it intentionally does not do

- It does not feed a FreeToken launch through ggrun's GGUF tensor planner,
  llama.cpp tune cache, allocation probe, or OOM replan. Those flags describe a
  different runtime and would be false precision here.
- It does not enable tensor parallelism or ggrun multi-GPU placement. Tensor-
  parallel passthrough flags are rejected, even when placed after `--`.
- It does not install FreeToken or its large CUDA/PyTorch dependency stack.
- It does not claim general GGUF support. FreeToken loads HF safetensors and FTW
  checkpoints directly; its documented native GGUF path is Gemma-4-specific.
- It does not wrap ggrun's Claude reviewer/router flow. Once the server is ready,
  use FreeToken's native command in another terminal:

  ```bash
  ft launch claude --server http://127.0.0.1:1919
  ```

## A/B measurement

The reproducible real-agent protocol, live ggrun baseline, and hardware-window
matrix are specified in the
[runtime comparison plan](agentic-runtime-comparison-plan.md).

Query the served model ID and use ggrun's HTTP one-shot benchmark against the
FreeToken port:

```bash
curl -s http://127.0.0.1:1919/v1/models
ggrun benchmark --port 1919 --model <served-model-id>
```

Treat comparisons with a llama.cpp GGUF as exploratory unless the checkpoint,
quantization quality, context, sampling, prompt, and concurrency are equivalent.
The useful first experiment is operational: compare load time, steady decode,
long-turn prefix reuse, VRAM/RAM footprint, and stability on the same workload.
