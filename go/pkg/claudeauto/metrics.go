package claudeauto

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// Per-request timing is the prerequisite for every scheduling change: chunked
// prefill, calibrated admission and priority lanes are all "change one setting,
// compare one number", and the router previously recorded neither. Records are
// written as JSON lines so a run can be replayed and compared offline.

// carryBytes is the longest usage key plus its value, so a token count split
// across two proxy writes is still matched exactly once.
const carryBytes = 64

// usageKeys are scanned incrementally in the proxied response. The leading
// quote keeps "input_tokens" from also matching "cache_read_input_tokens".
var usageKeys = []struct {
	key   []byte
	field func(*Usage) *int64
}{
	{[]byte(`"input_tokens":`), func(u *Usage) *int64 { return &u.InputTokens }},
	{[]byte(`"cache_read_input_tokens":`), func(u *Usage) *int64 { return &u.CacheReadTokens }},
	{[]byte(`"cache_creation_input_tokens":`), func(u *Usage) *int64 { return &u.CacheCreationTokens }},
	{[]byte(`"output_tokens":`), func(u *Usage) *int64 { return &u.OutputTokens }},
}

// Usage holds the token counts reported by the backend for one request.
type Usage struct {
	InputTokens         int64 `json:"input_tokens"`
	CacheReadTokens     int64 `json:"cache_read_tokens"`
	CacheCreationTokens int64 `json:"cache_creation_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
}

// PromptTokens is the work the backend actually had to prefill. Cache reads are
// excluded because they are not evaluated.
func (u Usage) PromptTokens() int64 {
	return u.InputTokens + u.CacheCreationTokens
}

// RequestRecord is one routed request. Queue, prefill and decode are kept
// separate: a large prefill starving active decodes is invisible in a single
// end-to-end duration.
type RequestRecord struct {
	Time           string `json:"time"`
	Route          string `json:"route"`
	Conversation   string `json:"conversation"`
	Status         int    `json:"status"`
	Stream         bool   `json:"stream"`
	Aborted        bool   `json:"aborted"`
	CancelPhase    string `json:"cancel_phase,omitempty"`
	DeadlineBucket string `json:"deadline_bucket,omitempty"`
	QueueMS        int64  `json:"queue_ms"`
	// TTFBMS is transport time to the first HTTP response byte. Anthropic SSE
	// sends envelopes before generation, so DecodeStartMS is the authoritative
	// model-phase boundary for streaming requests.
	TTFBMS        int64 `json:"ttfb_ms"`
	DecodeStartMS int64 `json:"decode_start_ms,omitempty"`
	TotalMS       int64 `json:"total_ms"`
	RequestBytes  int64 `json:"request_bytes"`
	ResponseBytes int64 `json:"response_bytes"`
	Usage         Usage `json:"usage"`
}

// PrefillMS is time to the first generated delta minus queue wait: the
// backend's own prompt-processing cost, not the wait ggrun imposed. Legacy and
// non-streaming records fall back to TTFB because no separate boundary exists.
func (r RequestRecord) PrefillMS() int64 {
	boundary := r.DecodeStartMS
	if boundary <= 0 && !r.Stream {
		boundary = r.TTFBMS
	}
	if boundary <= r.QueueMS {
		return 0
	}
	return boundary - r.QueueMS
}

// DecodeMS is the generation tail after the first generated delta.
func (r RequestRecord) DecodeMS() int64 {
	// A request cancelled while it is queued has no response writer and therefore
	// no first-byte timestamp. Treating total_ms - 0 as decode time made queue
	// cancellations look like slow generation even though they produced no token.
	boundary := r.DecodeStartMS
	if boundary <= 0 && !r.Stream {
		boundary = r.TTFBMS
	}
	if boundary <= 0 || r.TotalMS <= boundary {
		return 0
	}
	return r.TotalMS - boundary
}

type metricsSink struct {
	mu     sync.Mutex
	out    io.Writer
	closer io.Closer

	count            int64
	successful       int64
	failed           int64
	aborted          int64
	abortedMS        int64
	abortedQueueMS   int64
	abortedServiceMS int64
	abortedQueue     int64
	abortedService   int64
	deadline60       int64
	deadline600      int64
	queueMS          int64
	prefillMS        int64
	decodeMS         int64
	maxQueueMS       int64
	maxPrefillMS     int64
	usage            Usage
}

func (m *metricsSink) record(rec RequestRecord) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.count++
	m.queueMS += rec.QueueMS
	if rec.Aborted {
		m.aborted++
		m.abortedMS += rec.TotalMS
		m.abortedQueueMS += rec.QueueMS
		if rec.TotalMS > rec.QueueMS {
			m.abortedServiceMS += rec.TotalMS - rec.QueueMS
		}
		if rec.CancelPhase == "queue" {
			m.abortedQueue++
		} else if rec.CancelPhase == "service" {
			m.abortedService++
		}
		switch rec.DeadlineBucket {
		case "~60s":
			m.deadline60++
		case "~600s":
			m.deadline600++
		}
	} else if rec.Status >= 200 && rec.Status < 300 {
		m.successful++
		m.prefillMS += rec.PrefillMS()
		m.decodeMS += rec.DecodeMS()
		if p := rec.PrefillMS(); p > m.maxPrefillMS {
			m.maxPrefillMS = p
		}
		m.usage.InputTokens += rec.Usage.InputTokens
		m.usage.CacheReadTokens += rec.Usage.CacheReadTokens
		m.usage.CacheCreationTokens += rec.Usage.CacheCreationTokens
		m.usage.OutputTokens += rec.Usage.OutputTokens
	} else {
		m.failed++
	}
	if rec.QueueMS > m.maxQueueMS {
		m.maxQueueMS = rec.QueueMS
	}
	if m.out == nil {
		return
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	// A telemetry failure must never fail the user's request.
	_, _ = m.out.Write(append(line, '\n'))
}

func (m *metricsSink) summary() map[string]any {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	s := map[string]any{
		"requests":                 m.count,
		"successful":               m.successful,
		"failed":                   m.failed,
		"aborted":                  m.aborted,
		"aborted_ms_total":         m.abortedMS,
		"aborted_queue_ms_total":   m.abortedQueueMS,
		"aborted_service_ms_total": m.abortedServiceMS,
		"aborted_in_queue":         m.abortedQueue,
		"aborted_in_service":       m.abortedService,
		"deadline_60s_signature":   m.deadline60,
		"deadline_600s_signature":  m.deadline600,
		"queue_ms_total":           m.queueMS,
		"prefill_ms_total":         m.prefillMS,
		"decode_ms_total":          m.decodeMS,
		"queue_ms_max":             m.maxQueueMS,
		"prefill_ms_max":           m.maxPrefillMS,
		"prompt_tokens":            m.usage.PromptTokens(),
		"cache_read_tokens":        m.usage.CacheReadTokens,
		"output_tokens":            m.usage.OutputTokens,
		"prefill_tokens_per_s":     ratePerSecond(m.usage.PromptTokens(), m.prefillMS),
		"decode_tokens_per_s":      ratePerSecond(m.usage.OutputTokens, m.decodeMS),
	}
	if m.count > 0 {
		s["queue_ms_mean"] = m.queueMS / m.count
	}
	return s
}

func (m *metricsSink) close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closer == nil {
		return nil
	}
	err := m.closer.Close()
	m.closer, m.out = nil, nil
	return err
}

func ratePerSecond(tokens, ms int64) float64 {
	if ms <= 0 || tokens <= 0 {
		return 0
	}
	return float64(tokens) * 1000 / float64(ms)
}

// EnableMetrics writes one JSON line per routed request to path. It is optional:
// a metrics failure disables recording rather than failing a launch.
func (r *Router) EnableMetrics(path string) error {
	if r == nil {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	r.metrics = &metricsSink{out: f, closer: f}
	return nil
}

// MetricsSummary reports running totals, or nil when recording is disabled.
func (r *Router) MetricsSummary() map[string]any {
	if r == nil {
		return nil
	}
	return r.metrics.summary()
}

// conversationKey groups requests by agent without needing Claude Code to send
// an identifier: an agent's system prompt and opening message are stable across
// its turns, so their digest is a usable per-agent key. It is a local grouping
// key only and never leaves the machine.
//
// metadata.user_id used to short-circuit the digest, on the assumption that a
// client-supplied identifier beats a guess. Claude Code sends this instead:
//
//	{"device_id":"b5243393...","account_uuid":"","session_id":"072e63a1-..."}
//
// which identifies the *install*, not the conversation -- every subagent in a
// workflow fan-out carries the same blob as the foreground turn. Measured on a
// production run, all 4231 requests collapsed onto one key, which silently
// disabled every per-conversation feature built on top: slot affinity had a
// single entry to compare, and the scheduler degenerated to plain FIFO, so a
// foreground turn queued behind the whole fan-out (p50 wait 35.4 minutes).
//
// So user_id is folded into the digest rather than replacing it. It still
// separates two sessions or two devices sharing one server, and the system
// prompt and opening message go back to separating the agents within them.
func conversationKey(body []byte) string {
	var request struct {
		System   json.RawMessage   `json:"system"`
		Messages []json.RawMessage `json:"messages"`
		Metadata struct {
			UserID string `json:"user_id"`
		} `json:"metadata"`
	}
	if json.Unmarshal(body, &request) != nil {
		return ""
	}
	sum := sha256.New()
	sum.Write([]byte(request.Metadata.UserID))
	sum.Write(request.System)
	if len(request.Messages) > 0 {
		sum.Write(request.Messages[0])
	}
	return hex.EncodeToString(sum.Sum(nil))[:12]
}

func isStreamRequest(body []byte) bool {
	var request struct {
		Stream bool `json:"stream"`
	}
	return json.Unmarshal(body, &request) == nil && request.Stream
}

// meteredWriter records status, first-byte time and token usage while the proxy
// streams. It must forward Flush: Claude Code streams SSE, and swallowing
// flushes would buffer the whole response.
type meteredWriter struct {
	http.ResponseWriter
	start       time.Time
	status      int
	written     int64
	ttfb        time.Duration
	usage       Usage
	carry       []byte
	onDecode    func()
	decodeSeen  bool
	decodeStart time.Duration
}

func (w *meteredWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *meteredWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	if w.written == 0 && len(p) > 0 {
		w.ttfb = time.Since(w.start)
	}
	w.written += int64(len(p))
	w.scan(p)
	return w.ResponseWriter.Write(p)
}

func (w *meteredWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// scan reads token counts out of the response as it streams. Non-streaming
// bodies carry one usage object; SSE carries input counts in message_start and
// a growing output count in each message_delta, so the maximum is taken.
func (w *meteredWriter) scan(p []byte) {
	buf := p
	if len(w.carry) > 0 {
		buf = append(append(make([]byte, 0, len(w.carry)+len(p)), w.carry...), p...)
	}
	// Anthropic streams message_start/content_block_start envelopes before the
	// model has finished prompt processing. They are the first HTTP bytes, but
	// they are not a decode boundary. Only an actual generated delta proves the
	// backend reached generation; using the envelope admitted a second append
	// while the live slot was still thousands of tokens into a cold prefill.
	if !w.decodeSeen && containsGeneratedDelta(buf) {
		w.decodeSeen = true
		w.decodeStart = time.Since(w.start)
		if w.onDecode != nil {
			w.onDecode()
		}
	}
	for _, k := range usageKeys {
		from := 0
		for {
			i := bytes.Index(buf[from:], k.key)
			if i < 0 {
				break
			}
			at := from + i + len(k.key)
			if v, ok := parseLeadingInt(buf[at:]); ok {
				if field := k.field(&w.usage); v > *field {
					*field = v
				}
			}
			from = at
		}
	}
	if len(buf) > carryBytes {
		w.carry = append(w.carry[:0], buf[len(buf)-carryBytes:]...)
		return
	}
	w.carry = append(w.carry[:0], buf...)
}

func containsGeneratedDelta(buf []byte) bool {
	for _, marker := range [][]byte{
		[]byte(`"type":"text_delta"`), []byte(`"type": "text_delta"`),
		[]byte(`"type":"thinking_delta"`), []byte(`"type": "thinking_delta"`),
		[]byte(`"type":"input_json_delta"`), []byte(`"type": "input_json_delta"`),
	} {
		if bytes.Contains(buf, marker) {
			return true
		}
	}
	return false
}

func parseLeadingInt(b []byte) (int64, bool) {
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t') {
		i++
	}
	start := i
	for i < len(b) && b[i] >= '0' && b[i] <= '9' {
		i++
	}
	if i == start || i == len(b) {
		// Refuse a number that ends the buffer: it may still be truncated.
		return 0, false
	}
	v, err := strconv.ParseInt(string(b[start:i]), 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
