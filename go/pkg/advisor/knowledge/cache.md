# Prompt-cache and recurrent-checkpoint knowledge

Primary runtime reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

- Prefix reuse is measured from protocol cache counters such as timings.cache_n or cached prompt tokens. Startup allocation does not prove cache correctness.
- Hybrid/recurrent state cannot be reconstructed from ordinary KV alone. One rolling checkpoint can be erased before an older branch returns; retain a bounded set and test both append and branch paths.
- A cache canary must cross at least two checkpoint boundaries, then run cold, strict extension, branch before the newest checkpoint, and identical replay.
- Logs explain misses, but protocol counters are primary evidence. A backend that exposes no counter remains unverified rather than being guessed healthy.
- Full SWA is a memory-shaping feature. If a backend/model rejects it and silently disables it, remove only a generated request, recompute the entire placement, and remeasure. Preserve explicit user requests and fail closed.
- The generic performance tuner does not own checkpoint retention.
