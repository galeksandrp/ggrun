# ggrun backlog

Internal engineering backlog. Not a user-facing roadmap.

Audited against `main` on 2026-07-14. This replaces the stale Claude Code task
statuses with only the work that remains. Source references use
`<Claude task-list>/<task-number>`.

## P0 — finish generic MTP/DFlash performance validation

The generic foundation already exists: ggrun parses NextN metadata, supports the
llama.cpp/ik_llama MTP dialects, validates draft GGUFs, searches/downloads generic
draft and EAGLE models from Hugging Face, selects a draft GPU and emits draft flags.

- [x] Determine the actual DeepSeek-V4 arrangement. The official/current GGUF has
  no NextN layers and no compatible MTP-only head. Published DSpark/DFlash
  companions currently target separate DS4/Lucebox runtimes, not a llama-server
  backend that can load both the target and drafter.
- [x] Audit the apparent Hugging Face match by immutable revision, checksum,
  metadata and a real backend no-allocation load. It is intentionally blocked:
  mainline rejects its private GGML type 101, and the public DS4 branch that knows
  that type has no DeepSeek4 target/draft model loader.
- [x] Extend the resolver generically for embedded MTP, MTP-only companions and
  target-specific DFlash companions. Downloads are revision-pinned where known,
  resume partial files and retain offline/local behavior.
- [x] Include separate speculative models in the exact placement/preflight ledger instead
  of the older approximate draft-GPU calculation.
- [x] Fall back to non-speculative serving with one clear reason when no compatible
  MTP artifact exists.
- [x] Require the selected backend's own loader to accept local/downloaded MTP and
  DFlash companions; never borrow `llama-fit-params` from a different fork. A
  later full-context companion failure disables speculation and recomputes a
  clean target-only placement.
- [x] Account for an embedded MTP head's additional model context/KV allocation
  before enabling it near the VRAM limit. Mainline `llama-fit-params` does not
  accept `--spec-type`, while `llama-server` adds the MTP context to its own fit
  ledger; use a selected-backend estimate or a conservative metadata-derived
  bound and keep Auto off when that reservation cannot be proven. The selected
  backend's target ledger is augmented with a metadata-derived MTP KV bound and
  conservative per-GPU compute reserve; an unprovable CPU-KV/oracle case fails
  back to the already-proven target-only placement.
- [x] Add the repeatable core MTP harness: `ggrun spec-test` compares the same
  GGUF off/on after warmup, runs nine checked prompt types for repeated rounds,
  sweeps draft ceilings 1-4, includes a real 60k request when each slot can hold
  it, and records prompt/decode/wall/acceptance data. Profiles are scoped by all
  GGUF shard identities, backend build, hardware/driver, selected GPU set,
  context, sampling and parallelism. Auto requires correctness/stability,
  parallel and 60k proofs where applicable, at least 2% decode and wall-time
  gains, no more than 5% prompt regression, output-length parity and an exact
  post-tuning launch-argument identity.
- [ ] Extend `ggrun spec-test` with the remaining full matrix: deterministic plus
  model-recommended sampling, thinking on/off, explicit TTFT/mean accepted length,
  serial plus parallel-4 in one invocation, peak VRAM/RAM capture and a long-run
  soak. The deterministic reasoning-off serial matrix now ran live on the embedded
  Qwen3.5-4B MTP GGUF: baseline 183.93 t/s; ceilings 1/2/3/4 were
  200.55/203.25/179.59/161.53 t/s. Ceiling 2 improved mean wall time 12.4% with
  54.2% draft acceptance, but regressed prompt processing 15.6%, so the verifier
  correctly kept Auto off. The broader matrix in this item is still required.
- [ ] Re-test DeepSeek V4 DFlash only when one reproducible llama-server commit can
  load both the official target and a published drafter; until then Auto stays off.
- [ ] Repeat the live test on one other MTP-capable MoE to prove the path is generic.

Source: `5e91131f/24`, retargeted by the user on 2026-07-12.

## P0 — HY3 through a reusable fork recipe

- [x] Add reviewed, immutable fork recipes plus `ggrun backend install <recipe>`;
  safely refresh clean checkouts, record the built commit and auto-route by GGUF
  architecture without losing the backend's real IK/mainline flag dialect.
- [x] Add the verified HY3 recipe: `noonr48/ik_llama-hy3`, branch `hy3-support`,
  pinned commit `f46c95ee90d8c8200b0147c646b883405020b482`, route `hy_v3`.
- [x] Build the pinned recipe on the test host and verify its commit, shared
  libraries, IK dialect, `hy_v3` loader code and automatic architecture route.
- [ ] Parse and load a real HY3 GGUF, then complete correctness/load, serial MTP
  and parallel-4 non-speculative benchmarks before calling it stable. The pinned
  fork deliberately removes MTP above one slot; re-test combined MTP +
  parallel-4 only after its server lifts that guard.
- [x] Add recipe update/rollback UX so future model forks are one declarative
  entry rather than bespoke installer code. `ggrun backend update <tag>` rebuilds
  at the recipe's pinned commit and keeps the build it replaces (renamed to
  `build-<accel>-prev-<commit>`, so it is a rename rather than a copy);
  `ggrun backend rollback <tag>` points the registration back at it, and the swap
  is symmetric so the rollback itself can be undone. A failed build restores the
  previous tree automatically — the poolside Laguna fork failing at 95% on a
  missing include is what motivated this. Retention is one level.
- [ ] CI smoke builds for recipes, so a fork that stops compiling is caught before
  someone's backend is the thing that discovers it.
- [x] Suggest the right fork when a backend cannot serve a model's architecture,
  instead of letting the loader fail with a tensor count. `BackendSupportsArch`
  follows a backend's own library graph and matches NUL-bracketed architecture
  literals; it warns rather than blocks, because it cannot follow non-ELF
  (Windows) dependencies and a false negative must not break a working setup.

## P1 — finish Claude Code integration

The launcher, native `/v1/messages`, local aliases for every Claude tier, parallel-4
by default when the context can preserve useful slots,
model-native context capped at 1M total context, per-slot compaction, no practical
workflow deadline, anti-loop sampling, chunk-level prompt-cache reuse and
DuckDuckGo MCP wiring are implemented. Claude Auto's hidden classifier requests are
now routed to an isolated parallel-1 local Qwen3.5-2B reviewer while coding stays on
the selected model. A Workflow stall hook reports stalled requests without aborting
them. Default local launches use fail-closed Auto, never bypass mode.

- [ ] Run one complete acceptance workflow against a running ggrun model: file edits,
  commands/tests, four workflow agents, tool results, queueing, combined response and
  context compaction.
- [ ] In that workflow, verify MCP `search` plus `fetch_content`, including a failed
  lookup/retry from a subagent.
- [x] Implement and verify local Auto permissions. Claude 2.1.207 sends a distinctive
  two-stage security-monitor request (about 25k prompt tokens) to the same model ID.
  ggrun's loopback router sends only that structured system request to the local 2B
  reviewer. Safe Bash completed end to end with zero permission denials; invalid
  reviewer output was also verified to fail closed. The pinned reviewer cold-prefilled
  the captured prompt in 2.4–5.8 seconds depending on GPU and warm reviews took ~0.18s.
  Qwen blocked the simulated SSH-key upload in 2.4 seconds. Both the upstream
  MiniCPM5-1B Q4 and the Fable5 Thinking Q4 candidate false-allowed that upload;
  the latter also missed the 60-second CPU fallback deadline, so neither is eligible.
- [x] Enable backend-supported `--cache-reuse 256` for Claude mode while preserving
  explicit opt-out and older-backend compatibility on shiftable contexts. With cache
  RAM and context checkpoints disabled, the controlled transformer compaction case
  dropped from 4,506 processed tokens / 45.1 seconds to one processed token / 0.15
  seconds (4,514 reused tokens). Hybrid/recurrent contexts now skip this unsupported
  flag and use one bounded rolling checkpoint when host headroom permits.
- [ ] Validate the new DeepSeek-V4 recurrent policy live: `--ctx-checkpoints 1`, no
  unsupported cache-reuse flag, Claude logical batch 128 and the new parallel-4
  default. Repeat an
  append-only 60k turn and prove that only the new tail is evaluated; record checkpoint
  size, TTFT, foreground responsiveness, peak RAM/VRAM and a long-run OOM result.
  Partial 2026-07-14 evidence: parallel-2 completed 60,020 prompt tokens without
  OOM at 22.87 prompt / 5.34 decode t/s; an identical-prefix append evaluated only
  147 new tokens and finished in 11.4s. The reviewer-conditioned five-expert,
  logical-batch-128 run also stayed within 23.25/8.41/9.22 GiB and completed an
  8,212-token concurrent smoke at 23.91 prompt / 5.44 decode t/s. A complete real
  Claude workflow and long soak remain open.
- [ ] Turn the repeatable parts into a Claude acceptance harness for `/v1/messages`,
  tool-use/tool-result blocks, aliases, MCP, malformed tool recovery and timeouts.
- [ ] **Benchmark and auto-select Claude Workflow parallelism:** compare parallel
  1, 2 and 4, plus 8 only when context/VRAM capacity makes it viable, with the same
  model, quant, context and hardware. Include the 60k request and real workflow
  fan-out; record total wall time, aggregate and per-request tok/s, TTFT, queue time,
  prompt-cache reuse, peak RAM/VRAM, OOM/retries and foreground responsiveness.
  Repeat runs, cache the fastest stable choice by model/backend/hardware/context,
  and preserve an explicit user override.
- [x] **Add live local-request progress to Claude Code launches:** queued/prefill/
  generation/completed/failed state, prompt tokens and percentage, prompt/decode
  tok/s and elapsed time across all four slots. Prefer structured slots/metrics data;
  use versioned log parsing only as fallback. Present it through an opt-in status line,
  terminal title or companion `ggrun status` view that does not corrupt Claude's
  fullscreen TUI. Parser, state-machine, status-injection and monitor lifecycle tests
  are implemented. The existing 60k parallel-4 run supplies the long-request backend
  behavior; repeat it once with the new UI before the next release.

Sources: `db3f32cc/1`, `db3f32cc/2`, `db3f32cc/3`, user request 2026-07-12.

## P1 — detach/attach: decouple Claude sessions from the backend

Decouple the Claude Code frontend lifecycle from the backend process so N Claude
sessions can connect to one persistent ggrun backend, closing a frontend must not
kill the backend, and reopening connects back to the same backend instead of
paying a 2–3 minute cold reload plus a full re-prefill.

**Design summary.** Add a `--serve` mode to the existing `--claude-code` launch
path and a `ggrun attach` / implicit-detach pair. `serve` runs the full,
battle-tested `cmdLaunch` lifecycle (placement, reviewer companion, router,
calibration, OOM recovery) but skips the foreground client: after
`verifyAndActivateLaunch` it registers the backend in a small on-disk registry and
blocks on the signal loop. `attach` reads the registry, probes the live backend,
starts a fresh per-session `claudeauto` router pointed at it, resolves the session
(fresh or `--resume`), and runs the client; on close it refreshes the session
record and exits WITHOUT touching the backend.

**1. Architecture: a persistent shared backend.**

- Chosen form: `ggrun serve --model <model> [--port N]` — a foreground persistent
  process, not a new supervisor. It is literally the existing Claude launch path
  minus the client: placement computation through `verifyAndActivateLaunch`
  (`go/cmd/ggrun/main.go:4499–4802`) is reused verbatim, including the reviewer
  companion (`startClaudeAutoReviewer`), the Auto router (`claudeAuto.startRouter`),
  calibration and the CUDA-OOM recovery/relaunch machinery.
- `serve` hosts ONE model + placement, exactly like today's single-backend launch.
  A second serve on the same port is refused by the existing `guardPortFree`
  (`main.go:6631`). Multiple backends = multiple ports, one serve process each.
- Registry: `serve` writes `<cacheDir>/serving-backends.json` atomically (mirroring
  `claudesession.Save`): backend key → {model_path, port, server_args, reviewer_port,
  started_at, attached_sessions[]}. This is the attach-discovery handle and the
  ownership/refcount store.
- The existing `ggrun daemon` (`go/pkg/daemon/daemon.go`) is the alternative
  substrate but is deliberately stripped of the launch lifecycle (comment at
  `main.go:6777`); extending it means hand-re-adding placement/calibration/OOM
  recovery. `serve` gets all of that for free by reusing cmdLaunch. Leave the
  daemon as-is; it remains the lightweight control-only path.

**2. Connection model.**

- Each session connects to its OWN loopback `claudeauto` router on an ephemeral
  port (already per-launch, `main.go:4781`); N sessions run N routers, all pointing
  `mainBaseURL` at the shared backend. The router is the per-client adapter
  (message delimiters, image sanitization, classifier/utility lanes, per-session
  admission + metrics — `go/pkg/claudeauto/claudeauto.go:297`), so sharing the
  backend does not share session policy. `claudeCodeEnv` / `ANTHROPIC_BASE_URL`
  are unchanged.
- Slots: llama.cpp `--parallel N` divides `--ctx-size` into N slots of ctx/N.
  Multiple sessions interleave over the shared slots; each session is a distinct
  conversation (the router's scheduler already keys by `conversationKey`). More
  sessions = smaller slots = earlier per-session compaction; the total context
  budget is fixed. `--claude-max-active` per router caps each session's in-flight
  requests.
- Conversation state on frontend close: the DURABLE state is Claude Code's
  transcript JSONL + workflow journal (what `--resume` restores). The backend KV /
  prompt cache is an ephemeral performance cache only. A live backend retains the
  KV for a detached session, so reattach re-prefills from prompt cache
  (`--cache-reuse`) — fast but not guaranteed (other sessions or checkpoint
  compaction can evict it). Correctness never depends on the backend surviving:
  reattach degrades to full re-prefill, never to lost state.

**3. Detach.**

- What stops it today: `runClaudeCodeClient` returns, then `main.go:4935 p.Stop()`
  kills the backend, `claudeAuto.stop()` (4938) tears the router down, and
  `os.Exit(code)` (4939) leaves nothing behind. The backend is a child of the same
  process, so even skipping Stop() leaves it in the terminal's process group
  writing to the terminal — SIGHUP on terminal close kills it.
- Detach = `--detach` on the `--claude-code` path (and the default behavior of
  sessions started by `attach`): skip `p.Stop()`/`claudeAuto.stop()`, refresh the
  session record (`refreshClaudeSessionRecord`), decrement the backend's attached
  count, exit. Net-new: under `serve`, the backend is a child of `serve` (which
  owns the terminal), so the SIGHUP problem only bites the
  `--claude-code --detach` variant, which must setsid + redirect backend output to
  a log file so closing the frontend's terminal cannot signal the backend.

**4. Reattach.**

- `ggrun attach [session-id | latest | <model>]`: read the registry (or probe a
  recorded `Record.Port` via `isServerRunning`/`waitForHealth`, `main.go:6236`),
  start a fresh router against the live backend (reuse
  `claudeAutoRuntime.startRouter` with the serve-registered reviewer port), resolve
  the session through the existing `claudeResumeSpec` so the `ShapeMismatches`
  guard still applies against the LIVE backend's `server_args`, and run the client
  with `--resume <id>`.
- `Record.Port`/`ServerArgs` (`go/pkg/claudesession/claudesession.go:40`) finally
  get a consumer: attach resolves the backend by model+placement, then uses the
  recorded shape to guard the attach — exactly what `claude_resume.go:62` does for
  a relaunch.
- What the backend contributes to reattach: warm KV/prompt cache (speed only).
  What the Record contributes: session identity, resume handle, shape guard.

**5. Multi-session.**

- Supported: SEPARATE conversations (different sessions/transcripts) on one
  backend via shared slots. This is the primary goal — multiple ggrun sessions
  onto one backend, each with its own transcript.
- Not supported in phase 1: two frontends on the SAME session ID (parallel writers
  to one transcript JSONL). Enforced by a per-session-ID lockfile next to the
  record; launch/attach refuse while a client holds it. Same-conversation parallel
  slots would also thrash the shared KV (two tails on one prefix).
- The llama.cpp slot model is the boundary: sessions are not pinned to slots (slots
  are assigned dynamically by prefix match), and a detached session's KV is
  reclaimed by other traffic over time.

**6. Lifecycle / ownership.**

- `serve` owns the backend and stays up after the last session detaches (the
  requested default). Opt-in `--stop-when-idle` stops the backend when the
  attached count hits zero.
- Refcounting: serve derives the attached count from the registry's session list
  (attach POSTs a lease, detach releases it). serve's signal loop — already the
  non-claude branch, `main.go:4945+` — keeps the process alive; its backend
  crash/OOM relaunch loop and the health-monitor goroutine (`main.go:4856`) are
  reused as-is, so a serve backend that dies relaunches with OOM evidence recorded.
- UX: `ggrun serve ...` (start), `ggrun attach <session|model>` (reopen), closing
  claude (implicit detach), `ggrun serve status` (backends + attached sessions).

**7. Interaction with existing code — reuse vs net-new.**

Reuse (existing code, no new logic):
- Full launch lifecycle in `cmdLaunch` (`main.go:4475–4802`): placement, reviewer,
  router, calibration, OOM recovery — `serve` gates on a `req.Serve` flag.
- `claudeauto.StartRouter` + scheduler + `conversationKey`
  (`go/pkg/claudeauto/`): per-session routers already multi-conversation-safe.
- `claudesession.Record`, `claudeResumeSpec`, `ShapeMismatches`
  (`go/pkg/claudesession`, `go/cmd/ggrun/claude_resume.go`): the attach shape guard.
- `isServerRunning` / `waitForHealth` (`main.go:6236/6212`): attach probe.
- `runClaudeCodeClient`, `claudeCodeEnv`, `claudeSessionArgs`,
  `refreshClaudeSessionRecord`: client launch + session refresh.
- `guardPortFree` for serve (kept as refusal); bypassed on attach.

Net-new (the real work):
- `--serve` flag handling in cmdLaunch: skip the client block, write
  `<cacheDir>/serving-backends.json`, block on the signal loop.
- `ggrun attach` command + a `connectClaudeSessionToBackend(cfg, backend, spec)`
  helper that extracts the client block (`main.go:4843–4943`) into a reusable
  connect path with backend teardown swapped for detach.
- Detach plumbing: skip Stop, refresh record, decrement refcount; backend output
  to a log file; setsid/process-group separation for `--claude-code --detach`.
- Registry read/write + per-session-ID lockfile + `serve status` +
  `--stop-when-idle`.

**8. Risks.**

- Session isolation: two frontends sharing one backend share slots; one session's
  context growth evicts another's prompt cache and can starve it. Mitigation:
  per-router admission limits (existing), per-session compaction windows (existing
  env, derived from serverArgs), and surfacing slot/context per session in
  `serve status`.
- Same-conversation multi-writer: forbidden by the session-ID lockfile; must be
  enforced in both `launch` and `attach`, not just documented.
- Port contention: the registry can go stale (serve died, port reused by another
  process). Attach must probe (`isServerRunning`) and refuse on shape mismatch
  rather than trust the registry blindly; serve keeps `guardPortFree`.
- Backend crash with sessions attached: 502s to live frontends; durable transcripts
  survive so reattach recovers, but live sessions do not transparently re-home.
  serve's relaunch (existing OOM loop) recovers the backend; frontends re-run the
  failed turn from the transcript.
- Context growth: total context is fixed; each extra session shrinks the effective
  per-session budget. `--parallel` should be chosen for the expected session count
  and per-session compaction tuned to the real slot.
- RAM: a shared prompt cache grows with the number of distinct sessions' prefixes;
  bounded by the existing MemoryMax/MemoryHigh cgroup band.

Phases:
- [ ] Phase 1 — `serve` + single-session `attach`/`detach`: `--serve` flag +
  registry write + `connectClaudeSessionToBackend` + detach (skip stop, log-to-file)
  + `ggrun attach <session-id|latest>`. Verify: start serve, attach, work, close
  frontend, attach again against the SAME backend PID, second turn re-prefills fast
  (prompt cache) — no cold reload.
- [ ] Phase 2 — multi-session + ownership: `serve status`, refcount,
  `--stop-when-idle`, per-session-ID lockfile, multiple concurrent frontends on one
  backend. Verify: two sessions interleave on one backend, isolation via per-router
  admission, per-session compaction windows.
- [ ] Phase 3 — resilience: attach shape guard against the live backend (reuse
  ShapeMismatches), stale-registry recovery (port re-resolution by model), serve
  crash-relaunch with sessions notified, backend log rotation.

Sources: current-architecture audit 2026-08-11 (detach/attach design task).

## P0 — thinker/worker: live model switching between a big MoE thinker and a fast dense worker

Goal: ggrun runs TWO models live — a fast dense "worker" (e.g. Muse-Glimmer-30B,
Qwen3.6-27B) handles routine turns at full speed, and a big MoE "thinker"
(DeepSeek-V4) is consulted for hard problems — a system where the big MoE does
the real thinking and gets asked when a difficult question or problem arises.
Feature requested 2026-08-14.

Rationale: dense models (Muse etc.) are far faster and worth the load time but not
smart enough to compete with the very big MoE. So: worker does the bulk, escalate
to the thinker on hard problems. This is the reviewer/advisor tiered-model pattern
(already shipped for Auto permissions + calibration) applied to the MAIN model
loop, not just support roles.

- [ ] **Two live backends** — the worker (fast dense) and the thinker (big MoE)
  both resident, sharing the GPU split by VRAM budget. ggrun manages the placement
  of both (reviewer/advisor reservation pattern extended to a main-worker + main-
  thinker pair). The thinker may be cold (unloaded, load-on-escalate) to save VRAM
  when not needed, or warm if it fits.
- [ ] **Routing** — the Claude session's requests route to the worker by default.
  When a request is "hard" (the worker is uncertain / a problem occurs / a complex
  question), escalate to the thinker. Decision criteria: worker's own confidence
  signal, request complexity heuristics, explicit escalation on error/failure, or
  a per-request override.
- [ ] **Escalation contract** — the thinker receives the context + the hard
  problem, returns the reasoned answer, and the worker continues. Preserve the
  conversation state (prompt cache / checkpoints) across the worker→thinker→worker
  transition. The thinker is consulted (asked) for hard problems, not for every turn.
- [ ] **Live switching** — switch the active model mid-session without losing the
  conversation (the detach/attach backend work + the daemon /reload path are the
  substrate; extend them for a 2-model residency + routing).
- [ ] **Interaction with existing** — reuse: the reviewer/advisor tiered-model
  pattern (claudeReviewerReservation, startClaudeAutoReviewer), the detach/attach
  serve+attach (TODO above), the daemon /reload model-swap path (existing TODO),
  and the per-router admission (each model gets its own router/admission).
- [ ] **Loading cost** — a dense worker like Muse loads in ~1-2 min vs the big MoE
  thinker's ~5-10 min (and its VRAM), so the worker being always-on + the thinker
  escalating is worth the load time. Document the tradeoff.
- [ ] **Phases** — (1) two-model residency + worker-default routing + manual
  escalate (a `/think` or escalation trigger); (2) automatic escalation on
  uncertainty/failure; (3) hot-swap without conversation loss + shared prompt cache.

## P1 — audit + fix prompt-cache / cache-reuse correctness

Audit whether ggrun provides prompt caching and `--cache-reuse` correctly across
all launch shapes, and fix the known gaps. Observed 2026-08-14 on the Muse-Glimmer
dense run: `cached n_tokens` grows 71k-94k (cache-reuse working), `--cache-reuse 256`
+ `--cram 12288` active. But correctness gaps from this session:

- [ ] **swa-full key mismatch (known, live-confirmed)** — ggrun emits `--swa-full`,
  the backend silently disables it for models that don't support it (DeepSeek-V4,
  Qwen3.6), yet the probe cache records the key as `swa-full=true`. A future launch
  reads swa-full=true evidence for a config that actually ran swa-full=off, so
  prompt-cache sizing is keyed to a config that never ran. (The swa-full reconcile
  / CALPLAN-4 half.)
- [ ] **KV-rate multi-seq poisoning (F2)** — a `parallel>1` launch wrote the
  model-wide KV bytes/token rate from the Nx inflated total with no seq guard,
  which would poison prompt-cache sizing for later plans. (Fixed in f6d2611; keep
  a regression guard.)
- [ ] **cache-reuse x recurrent/SSM correctness** — `--cache-reuse` is gated off for
  recurrent (deepseek4) and laguna (multi-position RoPE) arches. Verify the gate is
  right and that no other arch gets a cache-reuse flag the backend silently ignores.
- [ ] **prompt-cache x no-mmap / host offload** — with `--no-mmap` + CUDA_Host
  offload, the prompt cache lives partly on host; verify reuse still hits correctly
  and the `-cram` budget accounts for it (the Muse run had 12 GB cram + 7 GB host).
- [ ] **per-model/per-shape cache correctness** — verify cache-reuse + prompt cache
  behave correctly per model/backend/hardware/ctx shape (the scope key), and that a
  stale cache entry cannot be applied to a different shape.

## P1 — TUI: model ordering, KV/ctx suggestions, reviewer selection

Three TUI features requested 2026-08-14 (test-driving ggrun with dense models):

- [x] **Model ordering by usage** — DONE (commit 9bd6b7c): per-model usage
  record (pkg/modelusage, launch count + last-used, persisted), TUI list sorts
  by usage desc then last-used desc then name.
- [ ] **KV-size / ctx-step suggestions** — when a dense model would offload to
  system RAM or its KV doesn't fit, suggest useful steps for the KV size /
  context: show "ctx N gives ~M GB KV" rungs (e.g. the auto-ctx-reduce
  rungs: 131k/64k/32k with their KV sizes), and for "max" vs "fit" show the
  tradeoff. Reuse the existing `swaEstimateContext`/`FitCtx` (pkg/tui) and
  the `lowerContextRungs`/`maybeReduceDenseAutoContext` (pkg/placement) to
  present the options. Surface on the launch screen: "at ctx 131072 this
  model fits all-GPU; at 250000 it needs 7 GB host offload."
- [ ] **Reviewer/worker selection for Claude Code — two named profiles.** The
  Claude Code option in the TUI should offer TWO profiles by tradeoff, not raw
  model names:
  - **Fast + big profile: a WORKER model** (nanbeige4.2) — the reviewer must be
    a worker-class model in this profile: it can review AND do work (the
    reviewer/worker companion). Better quality, heavier/resident.
  - **Small + light profile: the small reviewer** (Qwen3.5-2B) — review-only,
    lighter, faster to load, less capable.
  The `--claude-reviewer qwen|nanbeige|auto` + TUI toggle (already added in the
  TUI rework, commit 9bd6b7c) is the mechanical base. Enhance the TUI to present
  these as named profiles ("Reviewer/worker: big/fast (nanbeige)" vs
  "small/light (Qwen2B)") so the user picks by the quality-vs-resource tradeoff.
  Constraint: the big profile's model must be a WORKER (not just a bigger
  reviewer). Default stays auto (dense→Qwen, big MoE→nanbeige).
  (Base `--claude-reviewer qwen|nanbeige|auto` + TUI toggle DONE in 9bd6b7c;
  the two-named-profile presentation + big-must-be-worker is the open part.)
- [ ] **Help/usage completeness audit** — the `--help` text (main.go usage block)
  should reflect ALL current commands + flags generally, not just recent additions.
  Audit that: every command (launch/daemon/download/tune/recommend/support/models/
  config/backend/update/claude/gui/... + diagnostics) is listed; every launch flag
  (port/ctx/kv/kv-quality/gpus/backend/parallel/threads/cache-ram/claude-*/
  mmap/swa-full/ram-limit/calibrate/worker-benchmark + the newer --claude-reviewer/
  --chat-template/--cgroup-headroom/--claude-resume) is present + described. The
  two reviewer profiles ("big/fast worker (nanbeige)" vs "small/light (Qwen2B)")
  should be named in the --claude-reviewer help. Keep the help current as features
  land (a "keep --help in sync with new flags" rule).

## P1 — auto-relaunch on ANY unexpected server stop (not just CUDA OOM)

The runtime crash loop (cmd/ggrun/main.go:5349-5367) only relaunches a server
that crashed with a RECOGNIZED CUDA OOM (runtimeLogCUDAOOM). A non-OOM crash or
a clean-but-unexpected stop (no OOM message in the log — e.g. the Qwen3.8 q5_1
run that just "stopped" after 65 min with no error) hits `runtimeLogCUDAOOM` ->
`!ok` -> `os.Exit(1)` and gives up, leaving a dead server. Requested 2026-08-15:
"ggrun should relaunch it when it crashes like those without an error."

- [ ] **Auto-relaunch on ANY unexpected stop**: when waitForShutdownOrCrash detects
  a crash, if it is NOT a recognized CUDA OOM, still auto-relaunch the server
  (bounded retries, like maxRuntimeOOMRetries), re-using the verified config /
  placement cache. Only exit after the retry budget is exhausted.
- [ ] **Distinguish user-stop vs crash**: a clean shutdown (Ctrl+C / signal) should
  exit; an unexpected stop (process died on its own, no signal) should relaunch.
  waitForShutdownOrCrash already distinguishes these (returns false on signal,
  true on self-exit) — use that to gate the relaunch.
- [ ] **Bounded + safe**: a retry budget (like maxRuntimeOOMRetries) + fall back to
  normal placement if the verified config fails to relaunch. Never relaunch in a
  tight loop on a persistent crash.
- [ ] **Reuse the verified-config-saver**: on relaunch, start directly from the
  saved verified config (the new c4b76ac feature) so the re-start is fast + known
  good.

## P0 — generic data-driven rules: chat-template override + KV-quant selection

Hardcoded per-model fixes are not what ggrun is about (the nanbeige template
override at main.go:2707-2737, the deepseek4/inkling KV rules at
archconstraints.go:56-63, resolveKVQuality's arch branches) — they only work for
one model, not everyone's or future ones. Replace them with GENERIC, data-driven,
user-editable mechanisms that work for any model, exposed in the TUI. Requested
2026-08-14 (Qwen3.8 raised both: its Jinja raise_exception 500s every request,
and KV-quant selection is a ggrun-wide choice problem, not a Qwen3.8 one).

- [ ] **Generic chat-template override** — a user-extensible catalog of corrected
  chat templates keyed by model/arch (a data file, not Go code), so a model with
  a broken template (nanbeige raise_exception, Qwen3.8 raise_exception, future
  ones) is fixed by adding an entry — no code change. User can add their own.
  Format: `<arch>/<model> → corrected .jinja` (+ a flag `--chat-template <name>`
  and a TUI toggle). ggrun auto-applies when the model's GGUF template has the
  raise_exception pattern. Replaces the hardcoded nanbeigeTemplateArgs.
- [ ] **Generic KV-quant selection** — a data-driven KV-quant decision: per-arch
  correctness rules (the archKVRules table) + a user-chosen default (the TUI
  KV-quality field) + an auto recommendation (accuracy vs VRAM based on the
  model + free VRAM). No hardcoded per-model branches — a rule table + user
  override. Works for any model: "this model has no rule → use the user's choice
  + the VRAM-aware auto". Replaces the resolveKVQuality arch branches.
- [ ] **Rule catalog format + UI** — a JSON/TOML file (e.g. `<config>/rules.json`)
  with template-override + KV-quant entries, editable by the user + shown in the
  TUI (a "model rules" screen). ggrun ships a default catalog (nanbeige, deepseek4,
  inkling, qwen3.8) that users extend.
- [ ] **TUI integration** — the TUI shows the active rule for a model (template
  override? KV quant?) and lets the user change it per-model or set a default.

## P1 — finish existing product foundations

- [ ] **TUI extra parameters:** add a free-text field to the model configuration
  screen, parse without shell execution, show the resulting arguments, persist it and
  preserve the CLI's explicit last-wins behavior. CLI `--` extras already work.
  Source: `c69ee13f/3`.
- [ ] **Model swapping:** productionize the existing `ggrun daemon` `/reload` path.
  Add named models, lazy unload/load policy, bounded RAM/VRAM, tests, documentation,
  TUI controls and stable Claude/OpenAI aliases. Source: `c69ee13f/15`.
- [ ] **Architecture gotchas:** turn the DeepSeek/preflight knowledge currently spread
  through code comments and `docs/compute-preflight-plan.md` into one maintained,
  AI-facing reference plus machine-readable diagnostic rules. Source: `c69ee13f/13`.
- [ ] **Residual exact-memory audit:** the main large-MoE ledger is measured and
  preflighted; audit the remaining optional prompt-cache/CRAM constants and the old
  approximate draft-model GPU estimator. Do not disturb the validated MoE plan.
  Source: `ebffa9bc/9`.

### swa-full and expert placement (relocated from docs/fitting-the-hardware.md)

Ordered by expected value on agentic workloads:

- [ ] **Make `--swa-full` first-class** (config key + flag + emitted from the strategy).
  Today it is an ExtraArgs passthrough: unpersistable, absent from
  `planDerivedLaunchFlags`, and silently worth ~3x turn time when forgotten.
- [ ] **Skip context checkpoints when swa-full is on.** `server-context.cpp` gates
  checkpoint creation on `n_swa > 0`, which stays true under swa-full even though
  its own comment says checkpoints are for the non-swa-full case. They are
  redundant memory in that configuration.
- [ ] **Auto-enable swa-full** for iSWA models when KV resolves to GPU and the
  swa-full total fits. In that regime it measured 87% reuse at no decode cost —
  there is no trade to weigh. Leave the KV-on-CPU case alone, where it is a real
  tradeoff (117 s vs 174 s) supported by a single workload.
- [ ] **Close the plan-to-OOM gap.** A plan that needs OOM retries to launch is the
  planner under-reserving. This run was planned at `--n-cpu-moe 27` and reached 30
  by retry; those three layers are decode speed given away.
- [ ] **Search context size against turn time.** Context is currently taken as a user
  constraint and everything else bends around it, but it is the single largest
  lever on expert placement. The planner should be able to report "ctx 131072 buys
  you 5 expert layers" rather than requiring the operator to work it out.

## P2 — performance and installation

- [ ] **Generic first-run serving calibration:** after deterministic memory preflight,
  benchmark only safe candidates for dense single-GPU, dense multi-GPU, CPU/RAM
  offload and MoE. Compare GPU subsets, supported stable split modes, batch/ubatch,
  KV placement and mmap policy with short prefill/decode plus a context-boundary
  stability request. Cache the fastest stable result by model shards, backend build,
  driver, hardware topology, context, parallelism and sampling profile; explicit user
  flags always win. Until this exists, default placement is a safe measured heuristic,
  not a universal proof of maximum speed.
- [x] **Live serving-path matrix on the reference host:** Qwen3.5-4B Q4 measured
  180.06 t/s on one GPU and 10.90 t/s CPU-only. Qwen3.6-27B Q5 measured 40.15 t/s
  on GPU0, 18.64 t/s with the stable layer split on GPUs 1+2, and 2.91 t/s with
  forced RAM offload on physical GPU2. DeepSeek-V4 IQ4 completed the 60k parallel-2
  load and append proof above; with the reviewer resident, the fresh planner put
  three expert blocks on GPU2 and two on GPU1 and passed exact preflight. All paths
  loaded, generated, and shut down cleanly; restricted-GPU preflight now reports
  the renumbered CUDA device's real 12 GiB capacity rather than physical GPU0's 24.

- [ ] **Post-baseline MTP path matrix:** only after the V4, HY3, M3 and 27B
  context, mmap/no-mmap, prefill/decode and single-/multi-GPU baselines are
  complete, run `ggrun spec-test` for every model/backend pair that advertises a
  compatible MTP path. Compare target-only versus MTP with the same workload and
  record correctness, accepted length, TTFT, prefill/decode speed, wall time and
  peak RAM/VRAM. Record unsupported combinations as not applicable rather than
  forcing speculation.

- [ ] **Ship a small local AI-doc advisor with ggrun:** package a compact model
  plus a signed, versioned knowledge bundle covering llama.cpp/fork flags, GGUF
  architectures, artifact provenance, known backend failures and ggrun's test
  methods. Feed it live model metadata, backend help/version, hardware, placement,
  acceptance/timing logs and cached A/B evidence so it can explain cases such as
  "MTP exists but ceiling 16 is slower", propose the next bounded experiment and
  generate a test matrix. Keep all final launch decisions deterministic and
  verifier-gated; the advisor may recommend but cannot bless a model/fork/artifact
  or change serving flags without loader, correctness, memory and performance
  checks. It should work offline from the shipped bundle, optionally refresh only
  from allowlisted primary sources with citations, and run in leftover CPU/GPU
  capacity without reducing the served model's SLA.

- [ ] **Small-model decode ablation:** settle the historical ~184 versus current
  151.4 tok/s result by testing minimal versus generated flags, f16 versus q4_0 KV,
  runtime repack, `-mqkv`, `-khad` and defrag. Keep only reproducible quality-neutral
  wins. Source: `e8111d05/17`.
- [ ] **Compile-free Linux CUDA installation:** CPU/Vulkan/macOS Metal/Windows CPU
  bundles and Windows prebuilt CUDA exist. Ship a Linux CUDA backend bundle so normal
  installation no longer needs CUDA toolkit, CMake or a compiler; source build becomes
  an explicit fallback. Source: `5e91131f/17`.
- [ ] **Homebrew + AUR:** publish the formula/tap and AUR/optional `-git` package with
  backend wiring, checksums, automated version bumps and smoke tests.
  Sources: `5e91131f/25`, `a769b44a/2`.
- [ ] **Community tune pool:** the offline-safe client exists. Finish the format/design
  review, moderation, hardware deduplication and poisoning controls, then publish the
  seed repository after destination approval. Source: `a769b44a/5`.

## P3 — external and later work

- [ ] Decide whether reserving `ggrun` on PyPI/npm is still useful; obtain explicit
  confirmation immediately before publishing. Source: `a769b44a/18`.
- [ ] Draft the next release announcement after MTP and Claude workflow results are
  final. Source: `fb9a268c/1`.
- [ ] Scope the separate backend-agnostic personal agent based on OpenCode + ggrun;
  keep it outside this repository until its boundary is agreed. Source: `a769b44a/19`.

## Confirmed complete or obsolete Claude TODOs

- [x] DeepSeek-V4 baseline first-launch placement, full-layer expert storage, startup
  OOM recovery and the historical 60k parallel load test. A later parallel-4 runtime
  OOM reopened long-session validation; the safer parallel-2 recurrent-checkpoint
  acceptance run remains explicitly open above. Source: `fb9a268c/2`.
- [x] Mainline DeepSeek-V4 backend; old antirez/cchuter fork registration is obsolete.
  Sources: `5e91131f/19`, `ebffa9bc/10`.
- [x] DeepSeek-V4 recommender inclusion. Source: `5e91131f/20`.
- [x] DeepSeek-V4 1M KV decision: GPU KV + Flash Attention. Source: `ebffa9bc/8`.
- [x] RAM headroom through CLI/config/environment/TUI/recommender/placement.
  Source: `e8111d05/13`.
- [x] Loopback server bind by default. Source: `e8111d05/11`.
- [x] Windows Python/download dependency audit and CI smoke coverage.
  Source: `e8111d05/14`.
- [x] Base local web research through DuckDuckGo MCP. Source: `db3f32cc/1`.
- [x] Auto-mode model-tier routing to the local alias. Source: `db3f32cc/3`.
- [x] README and launch-performance benchmark numbers now agree on the dated retest;
  the later direction retains Ollama only as a reference column and no separate
  AI-tune result column. Sources: `e8111d05/15`, `5e91131f/22`.

## Definition of done

- Automatic behavior is capability/metadata-driven; explicit settings win.
- Offline and failure behavior is safe and clear.
- Unit/regression tests pass, plus real hardware validation where relevant.
- Performance claims record model, quant, backend, context, parallelism, hardware
  and raw results.
- External publication always gets destination-specific approval first.
