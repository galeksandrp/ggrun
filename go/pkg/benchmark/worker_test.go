package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestRunWorkerSuiteScoresTypedAnswers(t *testing.T) {
	answers := []string{
		`{"action":"remove_generated_flag","flag":"--ctx-checkpoints-interval"}`,
		`{"fits":false,"gap_mb":408}`,
		`{"selected":"C"}`,
		`{"first_action":"move_layer_from_failed_gpu"}`,
	}
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		index := int(calls.Add(1) - 1)
		if index >= len(answers) {
			t.Fatalf("unexpected request %d", index)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": answers[index]}}},
			"usage":   map[string]int{"prompt_tokens": 20, "completion_tokens": 8},
			"timings": map[string]float64{"prompt_per_second": 100, "predicted_per_second": 40},
		})
	}))
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "worker"}).RunWorkerSuite()
	if err != nil {
		t.Fatalf("RunWorkerSuite: %v", err)
	}
	if result.Passed != 4 || result.Total != 4 || result.Score != 1 {
		t.Fatalf("unexpected score: %#v", result)
	}
	if result.PromptTPS != 100 || result.GenTPS != 40 {
		t.Fatalf("unexpected weighted timings: %#v", result)
	}
}

func TestValidateWorkerJSONRejectsExtraKeysAndWrongValue(t *testing.T) {
	if ok, _ := validateWorkerJSON(`{"selected":"C","reason":"fast"}`, map[string]interface{}{"selected": "C"}); ok {
		t.Fatal("extra keys must fail the typed protocol")
	}
	if ok, _ := validateWorkerJSON(`{"selected":"B"}`, map[string]interface{}{"selected": "C"}); ok {
		t.Fatal("wrong decision must fail")
	}
}
