# Backend compatibility and update knowledge

Primary upstream: https://github.com/ggml-org/llama.cpp

- Backend family labels are not build identities. Allocation, flags, architecture support, and verified profiles are scoped to the exact binary/build identity.
- Architecture strings found in a binary are a discovery hint, not conformance. A candidate backend must load the target GGUF in a contained probe before activation.
- Build updates beside the active binary, run help/architecture/load/cache conformance, then atomically activate. Keep one known-good rollback.
- Unsupported generated flags may be removed only through typed feature events. Memory-shaping changes require a full replan.
- Nanbeige4.2 support was merged upstream in ggml-org/llama.cpp PR #25994 on 2026-07-27. Older binaries cannot serve architecture `nanbeige`.
- The support expert uses a separately verified backend. It must exit and release resources before main-model hardware detection and placement.
