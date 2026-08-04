package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// WorkerSuiteResult measures the small, typed decisions that ggrun may safely
// offload to a companion model. It is deliberately separate from Run: launch
// calibration must remain a pure throughput measurement.
type WorkerSuiteResult struct {
	Model            string             `json:"model"`
	Passed           int                `json:"passed"`
	Total            int                `json:"total"`
	Score            float64            `json:"score"`
	DurationS        float64            `json:"duration_s"`
	PromptTokens     int                `json:"prompt_tokens"`
	CompletionTokens int                `json:"completion_tokens"`
	PromptTPS        float64            `json:"prompt_tps,omitempty"`
	GenTPS           float64            `json:"gen_tps,omitempty"`
	Cases            []WorkerCaseResult `json:"cases"`
	Timestamp        int64              `json:"timestamp"`
}

type WorkerCaseResult struct {
	Name             string  `json:"name"`
	Passed           bool    `json:"passed"`
	LatencyS         float64 `json:"latency_s"`
	PromptTokens     int     `json:"prompt_tokens,omitempty"`
	CompletionTokens int     `json:"completion_tokens,omitempty"`
	PromptTPS        float64 `json:"prompt_tps,omitempty"`
	GenTPS           float64 `json:"gen_tps,omitempty"`
	Output           string  `json:"output"`
	Reason           string  `json:"reason,omitempty"`
}

type workerCase struct {
	name     string
	prompt   string
	expected map[string]interface{}
}

var workerCases = []workerCase{
	{
		name: "unsupported_flag_repair",
		prompt: "A generated llama-server launch failed with error: unrecognized argument: " +
			"--ctx-checkpoints-interval. Allowed action IDs are remove_generated_flag and lower_ubatch. " +
			"Return the safe repair as JSON with exactly the keys action and flag; action must be one allowed ID.",
		expected: map[string]interface{}{"action": "remove_generated_flag", "flag": "--ctx-checkpoints-interval"},
	},
	{
		name:     "resident_memory_math",
		prompt:   "A no-mmap placement has 1000 MiB available and requires 1408 MiB. Return JSON with exactly fits (boolean) and gap_mb (integer).",
		expected: map[string]interface{}{"fits": false, "gap_mb": float64(408)},
	},
	{
		name: "measured_candidate_selection",
		prompt: "Choose one candidate. Policy: only measured+healthy candidates may win; then maximize decode_tps within a 4000 MiB budget. " +
			"A={measured:true,healthy:true,decode_tps:32,vram_mb:3300}; " +
			"B={measured:false,healthy:true,decode_tps:50,vram_mb:2200}; " +
			"C={measured:true,healthy:true,decode_tps:37,vram_mb:3900}. Return JSON with exactly the key selected.",
		expected: map[string]interface{}{"selected": "C"},
	},
	{
		name: "runtime_oom_recovery_order",
		prompt: "A model loaded and passed health, then CUDA OOM occurred on GPU 0 during decode. " +
			"Policy: preserve batch/ubatch first and remove one model layer from the failed GPU before changing throughput knobs. " +
			"Allowed first_action IDs are move_layer_from_failed_gpu and lower_ubatch. " +
			"Return JSON with exactly the key first_action; its value must be one allowed ID.",
		expected: map[string]interface{}{"first_action": "move_layer_from_failed_gpu"},
	},
}

// RunWorkerSuite runs a bounded, deterministic set of ggrun support decisions.
// The expected answers are typed JSON so a high score reflects usable machine
// output, not a subjective prose grade.
func (r *Runner) RunWorkerSuite() (*WorkerSuiteResult, error) {
	result := &WorkerSuiteResult{Model: r.Model, Total: len(workerCases), Timestamp: time.Now().Unix()}
	started := time.Now()
	var weightedPrompt, weightedGen float64
	for _, test := range workerCases {
		caseStarted := time.Now()
		response, err := r.workerChat(test.prompt)
		caseResult := WorkerCaseResult{Name: test.name, LatencyS: time.Since(caseStarted).Seconds()}
		if err != nil {
			caseResult.Reason = err.Error()
			result.Cases = append(result.Cases, caseResult)
			continue
		}
		caseResult.PromptTokens = response.PromptTokens
		caseResult.CompletionTokens = response.CompletionTokens
		caseResult.PromptTPS = response.PromptTPS
		caseResult.GenTPS = response.GenTPS
		caseResult.Output = compactWorkerOutput(response.Content, 512)
		caseResult.Passed, caseResult.Reason = validateWorkerJSON(response.Content, test.expected)
		if caseResult.Passed {
			result.Passed++
		}
		result.PromptTokens += response.PromptTokens
		result.CompletionTokens += response.CompletionTokens
		weightedPrompt += response.PromptTPS * float64(response.PromptTokens)
		weightedGen += response.GenTPS * float64(response.CompletionTokens)
		result.Cases = append(result.Cases, caseResult)
	}
	result.DurationS = time.Since(started).Seconds()
	if result.Total > 0 {
		result.Score = float64(result.Passed) / float64(result.Total)
	}
	if result.PromptTokens > 0 {
		result.PromptTPS = weightedPrompt / float64(result.PromptTokens)
	}
	if result.CompletionTokens > 0 {
		result.GenTPS = weightedGen / float64(result.CompletionTokens)
	}
	return result, nil
}

func (r *Runner) workerChat(prompt string) (*chatResult, error) {
	body := map[string]interface{}{
		"model": r.Model,
		"messages": []map[string]string{
			{"role": "system", "content": "You are ggrun's deterministic support worker. Obey the requested JSON schema exactly. Return one JSON object and no explanation."},
			{"role": "user", "content": prompt},
		},
		"max_tokens":   64,
		"temperature":  0,
		"seed":         1,
		"cache_prompt": false,
		"stream":       false,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequest(http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return nil, fmt.Errorf("HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(detail)))
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Timings struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
		} `json:"timings"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("no choices")
	}
	return &chatResult{
		Content:          decoded.Choices[0].Message.Content,
		PromptTokens:     decoded.Usage.PromptTokens,
		CompletionTokens: decoded.Usage.CompletionTokens,
		PromptTPS:        decoded.Timings.PromptPerSecond,
		GenTPS:           decoded.Timings.PredictedPerSecond,
	}, nil
}

func validateWorkerJSON(output string, expected map[string]interface{}) (bool, string) {
	object, err := firstJSONObject(output)
	if err != nil {
		return false, "invalid JSON object: " + err.Error()
	}
	if len(object) != len(expected) {
		return false, fmt.Sprintf("schema mismatch: got %d keys, want %d", len(object), len(expected))
	}
	for key, want := range expected {
		got, ok := object[key]
		if !ok {
			return false, "missing key " + key
		}
		if fmt.Sprint(got) != fmt.Sprint(want) {
			return false, fmt.Sprintf("%s=%v, want %v", key, got, want)
		}
	}
	return true, ""
}

func firstJSONObject(output string) (map[string]interface{}, error) {
	start := strings.Index(output, "{")
	end := strings.LastIndex(output, "}")
	if start < 0 || end < start {
		return nil, fmt.Errorf("no object found")
	}
	var object map[string]interface{}
	if err := json.Unmarshal([]byte(output[start:end+1]), &object); err != nil {
		return nil, err
	}
	return object, nil
}

func compactWorkerOutput(output string, limit int) string {
	output = strings.TrimSpace(output)
	if limit <= 0 || len(output) <= limit {
		return output
	}
	return output[:limit]
}
