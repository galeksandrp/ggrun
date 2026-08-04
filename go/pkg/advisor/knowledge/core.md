# ggrun controller invariants

- Deterministic placement, safety limits, user-owned context/quality settings, and artifact integrity are authoritative. The support model cannot override them.
- Prefer backend-measured allocation for the exact model/backend/hardware/context/slot/ubatch/feature signature. Do not extrapolate heterogeneous recurrent cache regions as a global bytes-per-token rate.
- Known failures are handled by deterministic rules first. The support expert is invoked only for an unknown fingerprint, exhausted safe actions, or ranking already-generated optimization candidates.
- A profile progresses through proposed, allocation-verified, load-healthy, functional-verified, cache-verified, performance-verified, and active. HTTP health alone is not reusable proof.
- Never lower user context, model quantization, or requested quality silently. Never propose shell commands, arbitrary flags, paths, URLs, or repositories.
- A candidate may replace last-known-good only after contained verification. A failed candidate is rejected and retained as evidence.
