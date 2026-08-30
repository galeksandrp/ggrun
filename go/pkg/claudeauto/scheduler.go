package claudeauto

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Admission was a plain counting semaphore: first come, first served, with no
// idea which conversation a request belonged to. That wastes prefill on a
// server with more agents than slots. Each agent turn carries the whole
// conversation so far, and llama.cpp reuses a prefix only while the slot still
// holds it. When an unrelated request lands on that slot in between, the next
// turn re-evaluates from the start.
//
// The scheduler therefore orders waiting requests by, in priority order:
//
//  1. lane, so a safety review never waits behind bulk work;
//  2. fair share, preferring the conversation that has gone longest without a
//     slot, so no conversation's wait scales with the depth of the queue;
//  3. age, so nothing starves.
//
// This needs no backend support and no particular llama.cpp version: it only
// changes the order in which ggrun hands requests over.
//
// Step 2 used to be the opposite -- affinity, preferring the *most* recently
// served conversation on the theory that its prefix was still resident. Two
// measurements from one production run retired that:
//
//   - It was inert. conversationKey returned one key for every request, so
//     there was never more than one entry to compare (see its doc comment).
//     Reuse still came in at 64.9% across 37.1M prompt tokens, which says the
//     backend's own slot prefix matching carries reuse, not ggrun's ordering.
//   - The cost of getting it wrong is enormous. With ordering degenerated to
//     FIFO, a foreground turn arriving during a workflow fan-out waited a median
//     of 35.4 minutes and up to 125, against ~109 s of actual compute per turn.
//
// Affinity optimises the term that turned out not to need help, and fair share
// bounds the term that dominates: a conversation now waits behind at most one
// turn per *other active conversation*, not one per queued request.

// Lane ranks a request's scheduling class. Lower runs first.
type Lane int

const (
	// LaneSafety is reserved for permission reviews. They are short, they block
	// the user's tool call, and they must never queue behind a long extraction.
	LaneSafety Lane = iota
	// LaneInteractive is foreground and coordinator work.
	LaneInteractive
	// LaneBulk is everything else, which is most workflow fan-out.
	LaneBulk
)

const (
	// fairShareWindow is how long a conversation's turn is held against it.
	// Past it a conversation counts as idle again and regains full claim on the
	// next slot, which is what lets a user who stepped away for a coffee return
	// to a responsive session rather than to the back of the fan-out.
	fairShareWindow = 5 * time.Minute
	// agingAfter promotes a waiter that has been passed over for too long, so
	// no arrival pattern can starve an unlucky conversation.
	agingAfter = 90 * time.Second
)

// laneOf ranks one request.
//
// A safety review normally leaves through the reviewer route and never reaches
// admission at all, but it lands here whenever no separate reviewer model is
// configured. In that case it must still preempt bulk work: it is short and it
// blocks the user's tool call.
//
// Everything else is bulk on purpose. Separating foreground coordinator turns
// from workflow fan-out needs a signal the Anthropic request body does not
// carry: metadata.user_id is per-install, and a subagent's system prompt is not
// reliably distinguishable from the main loop's. A wrong guess would
// deprioritise the user's own turn, which is the failure this exists to prevent.
//
// Fair share below makes the classification unnecessary rather than merely
// deferred. A foreground turn is, by construction, the conversation that has
// gone longest without a slot while the fan-out cycles through its agents, so
// ordering by that promotes it without having to recognise it.
func laneOf(body []byte) Lane {
	if IsClassifierRequest(body) {
		return LaneSafety
	}
	return LaneBulk
}

type waiter struct {
	conversation string
	lane         Lane
	promptTokens int
	class        requestClass
	enqueued     time.Time
	ready        chan struct{}
	admission    *admission
}

type requestClass int

const (
	requestSmall requestClass = iota
	requestCold
)

// admission follows a request after it leaves the queue. The first response
// byte is the only backend-independent boundary we can observe reliably: before
// it the slot is prefilling, afterwards it is decoding. That is enough to avoid
// overlapping two expensive cold prefills while still using a second slot for
// a bounded append once the first request has begun decoding.
type admission struct {
	s            *scheduler
	conversation string
	promptTokens int
	prefilling   bool
	released     bool
}

// scheduler admits a bounded number of concurrent main-model requests.
type scheduler struct {
	mu      sync.Mutex
	limit   int
	active  int
	prefill int
	waiting []*waiter
	// recent records when a conversation last held a slot.
	recent map[string]time.Time
	// promptTokens is the last completed request size by conversation. A small
	// growth is a cache-hot append; a first or substantially changed long prompt
	// is cold and must not contend with another host-expert pass.
	promptTokens map[string]int
	phaseAware   bool
	coldTokens   int
	now          func() time.Time
}

func newScheduler(limit int) *scheduler {
	return &scheduler{
		limit:        limit,
		recent:       map[string]time.Time{},
		promptTokens: map[string]int{},
		coldTokens:   8192,
		now:          time.Now,
	}
}

func (s *scheduler) setPhaseAware(enabled bool, coldTokens int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.phaseAware = enabled && s.limit > 1
	if coldTokens > 0 {
		s.coldTokens = coldTokens
	}
}

// acquire blocks until the request may run. It returns false if ctx ended
// first, in which case the caller must not call release.
func (s *scheduler) acquire(ctx context.Context, conversation string, lane Lane) bool {
	return s.acquireRequest(ctx, conversation, lane, 0) != nil
}

func (s *scheduler) acquireRequest(ctx context.Context, conversation string, lane Lane, promptTokens int) *admission {
	if s == nil || s.limit <= 0 {
		return &admission{}
	}
	s.mu.Lock()
	class := s.classifyLocked(conversation, promptTokens)
	w := &waiter{conversation: conversation, lane: lane, promptTokens: promptTokens, class: class, enqueued: s.now(), ready: make(chan struct{})}
	if len(s.waiting) == 0 && s.canAdmitLocked(w) {
		a := s.admitLocked(w)
		s.mu.Unlock()
		return a
	}
	s.waiting = append(s.waiting, w)
	// Phase-aware eligibility is not FIFO: a cold waiter may be parked while a
	// later bounded append is safe to run beside decode. Re-evaluate immediately
	// so that append does not need an unrelated release event to wake it.
	s.promoteLocked()
	s.mu.Unlock()

	select {
	case <-w.ready:
		return w.admission
	case <-ctx.Done():
		s.mu.Lock()
		defer s.mu.Unlock()
		for i, other := range s.waiting {
			if other == w {
				s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
				return nil
			}
		}
		// Already promoted between the ctx firing and the lock: the slot is
		// ours, so hand it straight on rather than leaking it.
		s.releaseLocked(w.admission, false)
		s.promoteLocked()
		return nil
	}
}

func (s *scheduler) classifyLocked(conversation string, promptTokens int) requestClass {
	if !s.phaseAware || promptTokens < s.coldTokens {
		return requestSmall
	}
	previous, ok := s.promptTokens[conversation]
	if ok && promptTokens >= previous && promptTokens-previous < s.coldTokens {
		return requestSmall
	}
	return requestCold
}

func (s *scheduler) canAdmitLocked(w *waiter) bool {
	if s.active >= s.limit {
		return false
	}
	if !s.phaseAware {
		return true
	}
	if w.class == requestCold {
		return s.active == 0
	}
	return s.prefill == 0
}

func (s *scheduler) admitLocked(w *waiter) *admission {
	// A zero prompt size is the legacy acquire/release API used by callers which
	// cannot provide phase metadata. Keep its original counting-semaphore
	// semantics so release() cannot strand a synthetic prefill counter.
	a := &admission{s: s, conversation: w.conversation, promptTokens: w.promptTokens, prefilling: s.phaseAware && w.promptTokens > 0}
	s.active++
	if a.prefilling {
		s.prefill++
	}
	w.admission = a
	return a
}

func (a *admission) markDecode() {
	if a == nil || a.s == nil {
		return
	}
	s := a.s
	s.mu.Lock()
	defer s.mu.Unlock()
	if a.released || !a.prefilling {
		return
	}
	a.prefilling = false
	s.prefill--
	s.promoteLocked()
}

func (a *admission) release() {
	if a == nil || a.s == nil {
		return
	}
	s := a.s
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseLocked(a, true)
	s.promoteLocked()
}

func (s *scheduler) releaseLocked(a *admission, remember bool) {
	if a == nil || a.released {
		return
	}
	a.released = true
	if a.prefilling {
		s.prefill--
	}
	s.active--
	if remember && a.conversation != "" {
		s.recent[a.conversation] = s.now()
		if a.promptTokens > 0 {
			s.promptTokens[a.conversation] = a.promptTokens
		}
		s.pruneRecentLocked()
	}
}

// release returns a slot and admits the best waiting request.
func (s *scheduler) release(conversation string) {
	if s == nil || s.limit <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if conversation != "" {
		s.recent[conversation] = s.now()
		s.pruneRecentLocked()
	}
	s.active--
	s.promoteLocked()
}

// promoteLocked admits waiters while capacity allows.
func (s *scheduler) promoteLocked() {
	for s.active < s.limit && len(s.waiting) > 0 {
		idx := s.bestAdmissibleWaiterLocked()
		if idx < 0 {
			return
		}
		w := s.waiting[idx]
		s.waiting = append(s.waiting[:idx], s.waiting[idx+1:]...)
		s.admitLocked(w)
		close(w.ready)
	}
}

func (s *scheduler) bestAdmissibleWaiterLocked() int {
	now := s.now()
	best := -1
	var bestKey waiterKey
	for i, w := range s.waiting {
		if !s.canAdmitLocked(w) {
			continue
		}
		key := s.sortKeyLocked(w, now)
		if best < 0 || key.less(bestKey) {
			best, bestKey = i, key
		}
	}
	return best
}

type waiterKey struct {
	lane Lane
	// served is when this conversation last held a slot. The zero time means
	// "not recently", which sorts first: a conversation the scheduler has not
	// seen in a while has the strongest claim on the next one.
	served   time.Time
	enqueued time.Time
}

func (k waiterKey) less(other waiterKey) bool {
	if k.lane != other.lane {
		return k.lane < other.lane
	}
	if !k.served.Equal(other.served) {
		// Least recently served first. This is what bounds a foreground turn's
		// wait by the number of active conversations instead of the queue depth:
		// each agent in a fan-out becomes recently-served the moment it runs, so
		// it yields to everyone who has not.
		return k.served.Before(other.served)
	}
	return k.enqueued.Before(other.enqueued)
}

func (s *scheduler) sortKeyLocked(w *waiter, now time.Time) waiterKey {
	lane := w.lane
	// Aging: a waiter passed over for too long is promoted to the front of the
	// interactive lane, so a pathological arrival pattern cannot starve it.
	if lane > LaneInteractive && now.Sub(w.enqueued) >= agingAfter {
		lane = LaneInteractive
	}
	key := waiterKey{lane: lane, enqueued: w.enqueued}
	if w.conversation != "" {
		if seen, ok := s.recent[w.conversation]; ok && now.Sub(seen) <= fairShareWindow {
			key.served = seen
		}
	}
	return key
}

// pruneRecentLocked drops affinity entries that can no longer be warm, so a
// long run does not accumulate one entry per agent forever.
func (s *scheduler) pruneRecentLocked() {
	if len(s.recent) <= 256 {
		return
	}
	now := s.now()
	for conv, seen := range s.recent {
		if now.Sub(seen) > fairShareWindow {
			delete(s.recent, conv)
			delete(s.promptTokens, conv)
		}
	}
	if len(s.recent) <= 256 {
		return
	}
	// Still large: keep only the most recent entries.
	type entry struct {
		conv string
		seen time.Time
	}
	entries := make([]entry, 0, len(s.recent))
	for conv, seen := range s.recent {
		entries = append(entries, entry{conv, seen})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].seen.After(entries[j].seen) })
	for _, e := range entries[256:] {
		delete(s.recent, e.conv)
		delete(s.promptTokens, e.conv)
	}
}

// stats reports current occupancy for the status endpoint.
func (s *scheduler) stats() (active, queued, limit int) {
	if s == nil {
		return 0, 0, 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active, len(s.waiting), s.limit
}

func (s *scheduler) status() map[string]any {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return map[string]any{
		"phase_aware":        s.phaseAware,
		"active_prefill":     s.prefill,
		"active_decode":      max(0, s.active-s.prefill),
		"cold_prompt_tokens": s.coldTokens,
	}
}
