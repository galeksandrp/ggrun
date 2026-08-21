package claudeauto

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
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
// and 5xx responses as failed attempts instead of writing them to the client. A
// reviewer that crashed, never came up, or rejected the prompt (a context that
// will not hold it, a template it cannot apply) must not take the review down
// with it -- the main model can still answer.
func installReviewerFallbackHooks(proxy *httputil.ReverseProxy) {
	proxy.ModifyResponse = func(resp *http.Response) error {
		if resp == nil || resp.StatusCode < 500 {
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
	if attempt.failed {
		return false
	}
	deferred.flush()
	r.record(routeReviewer, body, start, 0, metered)
	return true
}
