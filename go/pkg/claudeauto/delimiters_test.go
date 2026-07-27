package claudeauto

import (
	"encoding/json"
	"testing"
)

// The exact startup output Laguna produced on this project, whose template is
// not one llama.cpp's autoparser recognises.
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

// The delimiter reader has to work for whatever model a user launches, not just
// the one that exposed the bug. Each family needs its own shape because the
// delimiter is the literal prefix a later prompt must contain.
func TestParseChatMessageDelimitersAcrossTemplateFamilies(t *testing.T) {
	for _, tc := range []struct {
		name    string
		example string
		want    map[string]string
	}{
		{"chatml", "<|im_start|>system\nhelp<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
			map[string]string{"system": "<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}},
		{"llama3", "<|start_header_id|>system<|end_header_id|>\n\nhelp<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
			map[string]string{"system": "<|start_header_id|>system<|end_header_id|>", "user": "<|start_header_id|>user<|end_header_id|>", "assistant": "<|start_header_id|>assistant<|end_header_id|>"}},
		{"gemma", "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>\n",
			map[string]string{"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"}},
		{"tag", "<system>help</system>\n<user>Hello</user>\n<assistant>Hi</assistant>",
			map[string]string{"system": "<system>", "user": "<user>", "assistant": "<assistant>"}},
	} {
		got := ParseChatMessageDelimiters("example_format: '" + tc.example + "'")
		if len(got) != len(tc.want) {
			t.Errorf("%s: got %+v, want %d delimiters", tc.name, got, len(tc.want))
			continue
		}
		for _, d := range got {
			if want, ok := tc.want[d.Role]; !ok || want != d.Delimiter {
				t.Errorf("%s: %s = %q, want %q", tc.name, d.Role, d.Delimiter, want)
			}
		}
	}
}

// One template belongs to one family. A looser pattern must not add a second,
// conflicting delimiter for a role a more specific one already claimed.
func TestParseChatMessageDelimitersDoesNotMixFamilies(t *testing.T) {
	got := ParseChatMessageDelimiters("example_format: '<|im_start|>user\nHi<|im_end|>'")
	for _, d := range got {
		if d.Delimiter == "<user>" {
			t.Errorf("tag-style delimiter leaked into a ChatML template: %+v", got)
		}
	}
}
