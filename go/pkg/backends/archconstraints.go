package backends

import "strings"

// A backend answering "yes, I accept that" is not the same as that setting being
// correct for the model. BackendSupportsArch (archsupport.go) probes the first
// question -- can this build load this architecture -- and it is the right tool
// for that. It cannot answer the second.
//
// DeepSeek V4 is the case that proved the gap. ik_llama's loader accepts a q8_0
// K-cache for deepseek4 and says so in its own error text ("supports only f16,
// bf16, and q8_0"); mainline rejects it. Reading the ik acceptance as permission
// meant the launcher promoted ik launches to q8_0 -- and V4, whose attention
// weights are already FP8, then degenerates into repetition loops or '='-spam
// with no error reported anywhere. A silent wrong answer is the worst failure
// mode an agentic launcher has: every downstream tool call inherits it.
//
// So correctness constraints live here, keyed on the architecture read from the
// GGUF, and are applied after backend selection regardless of what the backend
// says it will take. Adding a newly discovered quirk is a table entry rather
// than another conditional threaded through the launch path.

// ArchKVRule constrains the KV cache types that produce correct output for one
// architecture. Empty Allowed means the architecture imposes no constraint.
type ArchKVRule struct {
	// Allowed lists the KV types known to compute correctly. The first entry is
	// the promotion target when a launch asks for something outside the list.
	Allowed []string
	// Reason is user-facing: it must say what goes wrong, not merely that the
	// value was changed, because the failure it prevents is invisible.
	Reason string
}

// Permits reports whether kvType is correct for this architecture.
func (rule ArchKVRule) Permits(kvType string) bool {
	kvType = strings.ToLower(strings.TrimSpace(kvType))
	for _, allowed := range rule.Allowed {
		if allowed == kvType {
			return true
		}
	}
	return false
}

// Target is the KV type a non-permitted launch should be promoted to.
func (rule ArchKVRule) Target() string {
	if len(rule.Allowed) == 0 {
		return ""
	}
	return rule.Allowed[0]
}

// archKVRules is the correctness table. An architecture absent from it is
// unconstrained: the launcher must not invent restrictions it has no evidence
// for, so silence here means "no known correctness constraint", never "unsafe".
var archKVRules = map[string]ArchKVRule{
	"deepseek4": {
		Allowed: []string{"f16", "bf16"},
		Reason: "DeepSeek V4 ships FP8 attention weights that are already quantized; " +
			"requantizing the K-cache compounds the loss and the model degenerates into " +
			"repetition loops or '='-spam with no error from the backend",
	},
}

// KVRuleForArch returns the KV correctness rule for a GGUF architecture. The
// second result is false when the architecture has no known constraint, which
// callers must treat as "leave the request alone".
func KVRuleForArch(arch string) (ArchKVRule, bool) {
	rule, ok := archKVRules[strings.ToLower(strings.TrimSpace(arch))]
	if !ok || len(rule.Allowed) == 0 {
		return ArchKVRule{}, false
	}
	return rule, true
}
