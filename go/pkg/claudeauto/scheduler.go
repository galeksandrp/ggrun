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
	enqueued     time.Time
	ready        chan struct{}
}

// scheduler admits a bounded number of concurrent main-model requests.
type scheduler struct {
	mu      sync.Mutex
	limit   int
	active  int
	waiting []*waiter
	// recent records when a conversation last held a slot.
	recent map[string]time.Time
	now    func() time.Time
}

func newScheduler(limit int) *scheduler {
	return &scheduler{
		limit:  limit,
		recent: map[string]time.Time{},
		now:    time.Now,
	}
}

// acquire blocks until the request may run. It returns false if ctx ended
// first, in which case the caller must not call release.
func (s *scheduler) acquire(ctx context.Context, conversation string, lane Lane) bool {
	if s == nil || s.limit <= 0 {
		return true
	}
	s.mu.Lock()
	if s.active < s.limit && len(s.waiting) == 0 {
		s.active++
		s.mu.Unlock()
		return true
	}
	w := &waiter{
		conversation: conversation,
		lane:         lane,
		enqueued:     s.now(),
		ready:        make(chan struct{}),
	}
	s.waiting = append(s.waiting, w)
	s.mu.Unlock()

	select {
	case <-w.ready:
		return true
	case <-ctx.Done():
		s.mu.Lock()
		defer s.mu.Unlock()
		for i, other := range s.waiting {
			if other == w {
				s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
				return false
			}
		}
		// Already promoted between the ctx firing and the lock: the slot is
		// ours, so hand it straight on rather than leaking it.
		s.active--
		s.promoteLocked()
		return false
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
		idx := s.bestWaiterLocked()
		w := s.waiting[idx]
		s.waiting = append(s.waiting[:idx], s.waiting[idx+1:]...)
		s.active++
		close(w.ready)
	}
}

// bestWaiterLocked picks the next request to admit.
func (s *scheduler) bestWaiterLocked() int {
	now := s.now()
	best := 0
	bestKey := s.sortKeyLocked(s.waiting[0], now)
	for i := 1; i < len(s.waiting); i++ {
		key := s.sortKeyLocked(s.waiting[i], now)
		if key.less(bestKey) {
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
