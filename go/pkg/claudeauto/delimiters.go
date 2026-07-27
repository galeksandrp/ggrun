package claudeauto

import (
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

// Prefix reuse on a sliding-window model depends entirely on context
// checkpoints: partial KV removal is impossible there, so a checkpoint is the
// only thing a later turn can resume from. llama.cpp creates one at the start of
// a user message, found by matching role delimiters against the prompt tokens --
// and those delimiters arrive in the request body:
//
//	auto delimiters = common_chat_msg_delimiters_parse(
//	        json_value(data, "message_delimiters", json::array()));
//
// They default to empty. The server fills them in for templates its autoparser
// recognises, and ships nothing for the rest. Laguna's template is one of the
// rest, so on this project a 62k-token prefill with seven checkpoint
// opportunities created zero, and 132,317 prompt tokens across three turns were
// re-processed with 0% reuse while LCP similarity said 78% was available.
//
// ggrun does not have to guess what the delimiters are. The backend prints an
// example of its own rendered template at startup, so they can be read back the
// same way compute buffers, KV geometry and the reviewer's VRAM are.

// MessageDelimiter is one role marker in the backend's chat format.
type MessageDelimiter struct {
	Role      string `json:"role"`
	Delimiter string `json:"delimiter"`
}

// exampleFormatRe finds the backend's rendered template sample:
//
//	srv init: init: chat template, example_format: '<system>You are a helpful assistant</system>
//	<user>Hello</user>
//	<assistant><think></think>Hi there</assistant>
var exampleFormatRe = regexp.MustCompile(`example_format:\s*'`)

// roleOpenerRe matches an opening role tag in the rendered example. Only the
// four roles llama.cpp's parser understands are useful; anything else it maps to
// COMMON_CHAT_ROLE_UNKNOWN and ignores.
var roleOpenerRe = regexp.MustCompile(`(?i)(<\|?\s*(system|user|assistant|tool)\s*\|?>)`)

// ParseChatMessageDelimiters reads the role delimiters out of a backend launch
// log. Returns nil when the backend printed no example, in which case the caller
// must send nothing and let the server use whatever it derived itself -- an
// invented delimiter is worse than none, because a wrong split would checkpoint
// at a position no later turn shares.
func ParseChatMessageDelimiters(logText string) []MessageDelimiter {
	loc := exampleFormatRe.FindStringIndex(logText)
	if loc == nil {
		return nil
	}
	// The example runs to the closing quote the backend emitted. Bound the scan
	// so an unterminated sample cannot swallow the rest of a multi-hour log.
	rest := logText[loc[1]:]
	if end := strings.Index(rest, "'"); end >= 0 {
		rest = rest[:end]
	} else if len(rest) > 4096 {
		rest = rest[:4096]
	}

	firstAt := map[string]int{}
	seen := map[string]string{}
	for _, m := range roleOpenerRe.FindAllStringSubmatchIndex(rest, -1) {
		tag := rest[m[2]:m[3]]
		role := strings.ToLower(rest[m[4]:m[5]])
		if _, ok := seen[role]; ok {
			continue
		}
		seen[role] = tag
		firstAt[role] = m[0]
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]MessageDelimiter, 0, len(seen))
	for role, tag := range seen {
		out = append(out, MessageDelimiter{Role: role, Delimiter: tag})
	}
	// Stable order by first appearance, so an identical template always yields an
	// identical request body and cannot perturb anything keyed on the request.
	sort.Slice(out, func(i, j int) bool { return firstAt[out[i].Role] < firstAt[out[j].Role] })
	return out
}

// InjectMessageDelimiters adds message_delimiters to a request body that has
// none. A body that already carries them is returned untouched: the server
// derived them from a template it understood, and that is more authoritative
// than anything read out of a log.
func InjectMessageDelimiters(body []byte, delims []MessageDelimiter) []byte {
	if len(delims) == 0 || len(body) == 0 {
		return body
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	if existing, ok := obj["message_delimiters"]; ok && len(existing) > 0 && string(existing) != "null" && string(existing) != "[]" {
		return body
	}
	encoded, err := json.Marshal(delims)
	if err != nil {
		return body
	}
	obj["message_delimiters"] = encoded
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}
