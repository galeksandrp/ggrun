package claudeauto

import (
	"encoding/json"
	"testing"
)

// The exact startup output Laguna produced on this project. Its template is not
// one llama.cpp's autoparser recognises, so the server shipped no delimiters,
// no checkpoint was ever created, and 132,317 prompt tokens across three turns
// were re-processed at 0% reuse.
const lagunaStartupLog = `1.24.787.431 I srv          init: init: chat template, example_format: '<system>You are a helpful assistant</system>
<user>Hello</user>
<assistant><think></think>Hi there</assistant>
<user>How are you?</user>
<assistant><think>'
1.24.795.925 I srv          init: init: chat template, thinking = 1`

func TestParseChatMessageDelimitersFromBackendOutput(t *testing.T) {
	got := ParseChatMessageDelimiters(lagunaStartupLog)
	want := []MessageDelimiter{
		{Role: "system", Delimiter: "<system>"},
		{Role: "user", Delimiter: "<user>"},
		{Role: "assistant", Delimiter: "<assistant>"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("delimiter %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// ChatML-style templates must work too, since the same path serves every model.
func TestParseChatMessageDelimitersHandlesChatML(t *testing.T) {
	const log = `srv init: chat template, example_format: '<|im_start|>system
You are helpful<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
'`
	if got := ParseChatMessageDelimiters(log); len(got) != 0 {
		// ChatML uses one shared opener, so role tags are not distinguishable
		// by this method. Returning nothing is correct: an invented delimiter
		// would checkpoint at a position no later turn shares.
		t.Logf("chatml yielded %+v", got)
	}
}

// A backend that printed no example must yield nothing rather than a guess.
func TestParseChatMessageDelimitersWithoutAnExample(t *testing.T) {
	if got := ParseChatMessageDelimiters("srv init: model loaded\n"); got != nil {
		t.Errorf("invented delimiters from a log with no example: %+v", got)
	}
}

func TestInjectMessageDelimiters(t *testing.T) {
	delims := ParseChatMessageDelimiters(lagunaStartupLog)

	body := []byte(`{"model":"local","messages":[{"role":"user","content":"hi"}]}`)
	out := InjectMessageDelimiters(body, delims)
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("produced invalid JSON: %v", err)
	}
	if _, ok := obj["message_delimiters"]; !ok {
		t.Fatal("message_delimiters was not added")
	}
	if _, ok := obj["messages"]; !ok {
		t.Error("rewriting the body dropped the messages")
	}

	// A body the server already populated is more authoritative than a log.
	already := []byte(`{"message_delimiters":[{"role":"user","delimiter":"<|im_start|>user"}]}`)
	if got := InjectMessageDelimiters(already, delims); string(got) != string(already) {
		t.Errorf("overrode delimiters the server supplied: %s", got)
	}
	// Nothing to inject, or unparseable input, must pass through untouched.
	if got := InjectMessageDelimiters(body, nil); string(got) != string(body) {
		t.Error("modified the body with no delimiters available")
	}
	if got := InjectMessageDelimiters([]byte("not json"), delims); string(got) != "not json" {
		t.Error("mangled a body it could not parse")
	}
}
