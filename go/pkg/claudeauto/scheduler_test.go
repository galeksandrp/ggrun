package claudeauto

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fakeClock lets the aging and affinity windows be tested without sleeping.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newTestScheduler(limit int) (*scheduler, *fakeClock) {
	clock := &fakeClock{t: time.Date(2026, 7, 25, 12, 0, 0, 0, time.UTC)}
	s := newScheduler(limit)
	s.now = clock.now
	return s, clock
}

// enqueue parks a waiter and returns a channel that closes once it is admitted.
// It waits for an exact queue depth: waiting for "at least one" would be
// satisfied by an earlier waiter, letting this one race past the caller's next
// release and making an ordering assertion meaningless.
func enqueue(t *testing.T, s *scheduler, conv string, lane Lane, wantQueued int) chan struct{} {
	t.Helper()
	admitted := make(chan struct{})
	go func() {
		if s.acquire(context.Background(), conv, lane) {
			close(admitted)
		}
	}()
	waitFor(t, func() bool {
		_, queued, _ := s.stats()
		return queued == wantQueued
	}, "waiter did not enqueue")
	return admitted
}

func waitFor(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal(msg)
}

func admittedWithin(ch chan struct{}, d time.Duration) bool {
	select {
	case <-ch:
		return true
	case <-time.After(d):
		return false
	}
}

func TestSchedulerAdmitsUpToTheLimitWithoutQueueing(t *testing.T) {
	s, _ := newTestScheduler(2)
	if !s.acquire(context.Background(), "a", LaneBulk) {
		t.Fatal("first acquire blocked")
	}
	if !s.acquire(context.Background(), "b", LaneBulk) {
		t.Fatal("second acquire blocked")
	}
	active, queued, limit := s.stats()
	if active != 2 || queued != 0 || limit != 2 {
		t.Errorf("stats = %d/%d/%d, want 2/0/2", active, queued, limit)
	}
}

// The whole point of the change: when a slot frees, prefer the conversation
// whose prefix the backend most likely still holds.
func TestSchedulerPrefersAWarmConversation(t *testing.T) {
	s, clock := newTestScheduler(1)
	// "warm" ran recently, so its prefix should still be resident.
	if !s.acquire(context.Background(), "warm", LaneBulk) {
		t.Fatal("acquire failed")
	}
	s.release("warm")

	if !s.acquire(context.Background(), "blocker", LaneBulk) {
		t.Fatal("acquire failed")
	}
	// Enqueue the cold one first, so FIFO alone would pick it.
	cold := enqueue(t, s, "cold", LaneBulk, 1)
	clock.advance(time.Second)
	warm := enqueue(t, s, "warm", LaneBulk, 2)

	s.release("blocker")
	if !admittedWithin(warm, time.Second) {
		t.Fatal("warm conversation was not preferred over a colder one queued earlier")
	}
	if admittedWithin(cold, 50*time.Millisecond) {
		t.Fatal("cold conversation was admitted while capacity was 1")
	}
	s.release("warm")
	if !admittedWithin(cold, time.Second) {
		t.Fatal("cold conversation never ran")
	}
}

func TestSchedulerForgetsAffinityAfterTheWindow(t *testing.T) {
	s, clock := newTestScheduler(1)
	if !s.acquire(context.Background(), "stale", LaneBulk) {
		t.Fatal("acquire failed")
	}
	s.release("stale")
	// Past the window the prefix is assumed evicted, so preferring it buys
	// nothing and plain fairness should win.
	clock.advance(affinityWindow + time.Minute)

	if !s.acquire(context.Background(), "blocker", LaneBulk) {
		t.Fatal("acquire failed")
	}
	first := enqueue(t, s, "fresh", LaneBulk, 1)
	clock.advance(time.Second)
	second := enqueue(t, s, "stale", LaneBulk, 2)

	s.release("blocker")
	if !admittedWithin(first, time.Second) {
		t.Fatal("stale affinity still won after the window expired")
	}
	s.release("fresh")
	<-second
}

// A safety review blocks the user's tool call, so it must jump bulk work even
// when the bulk request is warm and queued first.
func TestSchedulerRunsSafetyBeforeWarmBulk(t *testing.T) {
	s, _ := newTestScheduler(1)
	if !s.acquire(context.Background(), "bulk", LaneBulk) {
		t.Fatal("acquire failed")
	}
	s.release("bulk")
	if !s.acquire(context.Background(), "blocker", LaneBulk) {
		t.Fatal("acquire failed")
	}
	warmBulk := enqueue(t, s, "bulk", LaneBulk, 1)
	safety := enqueue(t, s, "review", LaneSafety, 2)

	s.release("blocker")
	if !admittedWithin(safety, time.Second) {
		t.Fatal("safety review queued behind warm bulk work")
	}
	if admittedWithin(warmBulk, 50*time.Millisecond) {
		t.Fatal("bulk ran while capacity was 1")
	}
	s.release("review")
	<-warmBulk
}

// Affinity must never starve an unlucky conversation.
func TestSchedulerAgesOutAStarvedWaiter(t *testing.T) {
	s, clock := newTestScheduler(1)
	if !s.acquire(context.Background(), "hot", LaneBulk) {
		t.Fatal("acquire failed")
	}
	s.release("hot")
	if !s.acquire(context.Background(), "blocker", LaneBulk) {
		t.Fatal("acquire failed")
	}
	starved := enqueue(t, s, "unlucky", LaneBulk, 1)
	clock.advance(agingAfter + time.Second)
	hot := enqueue(t, s, "hot", LaneBulk, 2)

	s.release("blocker")
	if !admittedWithin(starved, time.Second) {
		t.Fatal("a waiter past the aging threshold was still passed over for a warm one")
	}
	s.release("unlucky")
	<-hot
}

func TestSchedulerCancelledWaiterReleasesItsPlace(t *testing.T) {
	s, _ := newTestScheduler(1)
	if !s.acquire(context.Background(), "holder", LaneBulk) {
		t.Fatal("acquire failed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool, 1)
	go func() { done <- s.acquire(ctx, "gone", LaneBulk) }()
	waitFor(t, func() bool { _, q, _ := s.stats(); return q == 1 }, "waiter did not enqueue")
	cancel()
	if <-done {
		t.Fatal("cancelled acquire reported success")
	}
	waitFor(t, func() bool { _, q, _ := s.stats(); return q == 0 }, "cancelled waiter stayed queued")

	// The slot must still be usable by someone else.
	next := enqueue(t, s, "next", LaneBulk, 1)
	s.release("holder")
	if !admittedWithin(next, time.Second) {
		t.Fatal("slot was leaked by the cancelled waiter")
	}
}

func TestSchedulerWithoutALimitNeverBlocks(t *testing.T) {
	s := newScheduler(0)
	for i := 0; i < 5; i++ {
		if !s.acquire(context.Background(), "any", LaneBulk) {
			t.Fatal("unlimited scheduler blocked")
		}
	}
	s.release("any") // must not panic
}

func TestSchedulerConcurrentLoadNeverExceedsTheLimit(t *testing.T) {
	const limit = 3
	s, _ := newTestScheduler(limit)
	var (
		mu      sync.Mutex
		active  int
		peak    int
		wg      sync.WaitGroup
		convIDs = []string{"a", "b", "c", "d", "e"}
	)
	for i := 0; i < 60; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			conv := convIDs[i%len(convIDs)]
			if !s.acquire(context.Background(), conv, LaneBulk) {
				return
			}
			mu.Lock()
			active++
			if active > peak {
				peak = active
			}
			mu.Unlock()
			time.Sleep(time.Millisecond)
			mu.Lock()
			active--
			mu.Unlock()
			s.release(conv)
		}(i)
	}
	wg.Wait()
	if peak > limit {
		t.Errorf("peak concurrency %d exceeded limit %d", peak, limit)
	}
	if a, q, _ := s.stats(); a != 0 || q != 0 {
		t.Errorf("scheduler did not drain: active=%d queued=%d", a, q)
	}
}

func TestPruneRecentBoundsMemoryOnLongRuns(t *testing.T) {
	s, clock := newTestScheduler(1)
	for i := 0; i < 400; i++ {
		s.recent[string(rune('a'+i%26))+string(rune(i))] = clock.now()
	}
	clock.advance(affinityWindow + time.Minute)
	s.recent["fresh"] = clock.now()
	s.mu.Lock()
	s.pruneRecentLocked()
	got := len(s.recent)
	s.mu.Unlock()
	if got > 256 {
		t.Errorf("recent map kept %d entries, want <= 256", got)
	}
	if _, ok := s.recent["fresh"]; !ok {
		t.Error("pruning dropped the most recent entry")
	}
}

func TestLaneOfReservesSafety(t *testing.T) {
	classifier := []byte(`{"system":[{"type":"text","text":"` + ClassifierMarker + ` Review."}],"messages":[]}`)
	if got := laneOf(classifier); got != LaneSafety {
		t.Errorf("laneOf(classifier) = %v, want LaneSafety", got)
	}
	if got := laneOf([]byte(`{"messages":[{"role":"user","content":"hi"}]}`)); got != LaneBulk {
		t.Errorf("laneOf(normal) = %v, want LaneBulk", got)
	}
}

// The backend answers /metrics from the same task queue as inference, so a poll
// issued while a batch is running cannot return sooner than that batch. With
// -ub 512 at ~143 t/s that floor is 3.58 s, and the deadline used to be 3 s --
// shorter than the fastest possible reply under load, so every busy-time poll
// was cancelled by construction. The snapshot then only refreshed while the
// backend was idle, biasing the counters the router treats as authoritative on
// throughput.
func TestBackendPollDeadlineExceedsAMicrobatch(t *testing.T) {
	const observedMicrobatchSeconds = 3.58
	if backendPollTimeout.Seconds() <= observedMicrobatchSeconds {
		t.Errorf("poll deadline %s does not exceed one measured microbatch (%.2fs)",
			backendPollTimeout, observedMicrobatchSeconds)
	}
	// Fetch's nil-client default carries its own 5s timeout, and
	// http.Client.Timeout wins over a longer context. The explicit client must
	// therefore match the deadline, or raising it accomplishes nothing.
	if backendPollClient.Timeout != backendPollTimeout {
		t.Errorf("client timeout %s != context deadline %s; the shorter one silently wins",
			backendPollClient.Timeout, backendPollTimeout)
	}
}
