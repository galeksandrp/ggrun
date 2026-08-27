package claudeauto

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
	"strings"
	"time"
)

// errReviewerUnusable marks a reviewer response the router should not forward,
// so the review can be retried on the main model instead.
var errReviewerUnusable = errors.New("reviewer backend returned an unusable response")

type reviewerAttemptKey struct{}

// reviewerAttempt tracks one review sent to the separate reviewer. It is carried
// on the request context so the shared reviewer proxy's ModifyResponse and
// ErrorHandler can report a failure without writing anything to the client --
// only classifier traffic installs one, so utility traffic keeps its old
// behaviour of surfacing the error directly.
type reviewerAttempt struct{ failed bool }

func reviewerAttemptFrom(ctx context.Context) *reviewerAttempt {
	if ctx == nil {
		return nil
	}
	attempt, _ := ctx.Value(reviewerAttemptKey{}).(*reviewerAttempt)
	return attempt
}

// installReviewerFallbackHooks makes the reviewer proxy report transport errors
// and non-2xx responses as failed attempts instead of writing them to the client. A
// reviewer that crashed, never came up, or rejected the prompt (a context that
// will not hold it, a template it cannot apply) must not take the review down
// with it -- the main model can still answer.
func installReviewerFallbackHooks(proxy *httputil.ReverseProxy) {
	proxy.ModifyResponse = func(resp *http.Response) error {
		if resp == nil || (resp.StatusCode >= 200 && resp.StatusCode < 300) {
			return nil
		}
		if attempt := reviewerAttemptFrom(resp.Request.Context()); attempt != nil {
			attempt.failed = true
			return errReviewerUnusable
		}
		return nil
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		if attempt := reviewerAttemptFrom(r.Context()); attempt != nil {
			attempt.failed = true
			return // write nothing; the caller retries on the main model
		}
		proxyError(w, r, err)
	}
}

// validReviewerVerdict mirrors the response contract in Claude Code's
// security-monitor prompt. A thinking block, tool call, empty 200, or prose-only
// answer is not a permission decision and must be retried on the main model.
func validReviewerVerdict(text string) bool {
	text = strings.TrimSpace(text)
	no := strings.Count(text, "<block>no</block>")
	yes := strings.Count(text, "<block>yes</block>")
	return no+yes == 1
}

func reviewerJSONText(body []byte) string {
	var decoded struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Completion string `json:"completion"`
	}
	if json.Unmarshal(body, &decoded) != nil {
		return ""
	}
	texts := make([]string, 0, len(decoded.Content)+1)
	for _, block := range decoded.Content {
		if block.Type == "text" || block.Type == "" {
			texts = append(texts, block.Text)
		}
	}
	if decoded.Completion != "" {
		texts = append(texts, decoded.Completion)
	}
	return strings.Join(texts, "")
}

// reviewerSSEText joins only user-visible text deltas. Thinking and tool-use
// events are intentionally ignored: neither can satisfy Claude's verdict parser.
func reviewerSSEText(body []byte) string {
	var texts []string
	for _, rawLine := range bytes.Split(body, []byte("\n")) {
		line := bytes.TrimSpace(rawLine)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var event struct {
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
			ContentBlock struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content_block"`
			Completion string `json:"completion"`
		}
		if json.Unmarshal(payload, &event) != nil {
			continue
		}
		if event.Delta.Type == "text_delta" || (event.Delta.Type == "" && event.Delta.Text != "") {
			texts = append(texts, event.Delta.Text)
		}
		if event.ContentBlock.Type == "text" || (event.ContentBlock.Type == "" && event.ContentBlock.Text != "") {
			texts = append(texts, event.ContentBlock.Text)
		}
		if event.Completion != "" {
			texts = append(texts, event.Completion)
		}
	}
	return strings.Join(texts, "")
}

func reviewerResponseUsable(status int, header http.Header, body []byte) bool {
	if status < 200 || status >= 300 || len(bytes.TrimSpace(body)) == 0 {
		return false
	}
	contentType := strings.ToLower(header.Get("Content-Type"))
	if strings.Contains(contentType, "text/event-stream") || bytes.HasPrefix(bytes.TrimSpace(body), []byte("event:")) || bytes.HasPrefix(bytes.TrimSpace(body), []byte("data:")) {
		return validReviewerVerdict(reviewerSSEText(body))
	}
	return validReviewerVerdict(reviewerJSONText(body))
}

// deferredWriter withholds a response until the router decides whether to keep
// it. Nothing reaches the client until flush(), so a failed reviewer attempt can
// be abandoned and replayed against the main model with the client none the
// wiser. Reviews are short verdicts rather than long generations, so holding one
// in memory costs little and buys the retry.
type deferredWriter struct {
	http.ResponseWriter
	hdr    http.Header
	status int
	buf    bytes.Buffer
}

func newDeferredWriter(w http.ResponseWriter) *deferredWriter {
	return &deferredWriter{ResponseWriter: w, hdr: http.Header{}}
}

func (w *deferredWriter) Header() http.Header { return w.hdr }

func (w *deferredWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
}

func (w *deferredWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.buf.Write(p)
}

// Flush is deliberately a no-op: the point of this writer is that nothing leaves
// it until the outcome is known, so an upstream flush must not push bytes out.
func (w *deferredWriter) Flush() {}

// flush replays the withheld response onto the real writer.
func (w *deferredWriter) flush() {
	dst := w.ResponseWriter.Header()
	for key, values := range w.hdr {
		dst[key] = values
	}
	if w.status != 0 {
		w.ResponseWriter.WriteHeader(w.status)
	}
	if w.buf.Len() > 0 {
		_, _ = w.ResponseWriter.Write(w.buf.Bytes())
	}
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// tryReviewer sends one review to the separate reviewer, holding the response
// back until it is known to be usable. It reports whether the review was served;
// false means nothing was written to w and the caller should retry on the main
// model.
func (r *Router) tryReviewer(w http.ResponseWriter, req *http.Request, proxy *httputil.ReverseProxy, body []byte) bool {
	attempt := &reviewerAttempt{}
	req = req.WithContext(context.WithValue(req.Context(), reviewerAttemptKey{}, attempt))
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))

	start := time.Now()
	deferred := newDeferredWriter(w)
	metered := &meteredWriter{ResponseWriter: deferred, start: start}
	proxy.ServeHTTP(metered, req)
	if attempt.failed || !reviewerResponseUsable(deferred.status, deferred.hdr, deferred.buf.Bytes()) {
		return false
	}
	deferred.flush()
	r.record(routeReviewer, body, start, 0, metered)
	return true
}
