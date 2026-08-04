package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// RunClaudeRouterCanary verifies the actual Anthropic-compatible workload path
// used by Claude Code. The OpenAI cache canary remains separate evidence for
// backend checkpoint reuse; this call covers /v1/messages routing, admission,
// request transforms, tool-schema transport, and Anthropic response decoding.
func (r *Runner) RunClaudeRouterCanary() error {
	body := map[string]interface{}{
		"model":       r.Model,
		"max_tokens":  96, // thinking models write a preamble before the token; 8 truncates before GGRUN_OK
		"temperature": 0,
		"stream":      false,
		"system":      "This is a deterministic ggrun Claude gateway verification. Reply with only GGRUN_OK and do not call tools.",
		"messages": []map[string]interface{}{{
			"role": "user", "content": []map[string]string{{"type": "text", "text": "Reply with only GGRUN_OK."}},
		}},
		"tools": []map[string]interface{}{{
			"name": "ggrun_canary_noop", "description": "Never call this tool during the canary.",
			"input_schema": map[string]interface{}{"type": "object", "properties": map[string]interface{}{}},
		}},
	}
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/messages", bytes.NewReader(data))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return fmt.Errorf("Claude router request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return fmt.Errorf("Claude router HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(detail)))
	}
	var decoded struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Completion string `json:"completion"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&decoded); err != nil {
		return fmt.Errorf("decode Claude router response: %w", err)
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
	if !validCanaryOutput(strings.Join(texts, " ")) {
		return fmt.Errorf("Claude router functional canary returned %q instead of a bounded non-empty answer", strings.TrimSpace(strings.Join(texts, " ")))
	}
	return nil
}
