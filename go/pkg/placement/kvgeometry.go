package placement

import (
	"regexp"
	"strconv"
	"strings"
)

// KV size is not one number times the context length. Interleaved sliding-window
// models allocate two caches with different depths, and ggrun modelled them as a
// single bytes-per-token rate read at one context -- correct only at the context
// it was measured at.
//
// Laguna, measured from the backend's own load-time output:
//
//	non-SWA KV cache: 1048576 cells, 12 layers -> 13824.00 MiB
//	    SWA KV cache:    1024 cells, 36 layers ->    40.50 MiB
//
// Both work out to 1152 bytes per cell per layer, which agrees exactly with
// n_embd_k_gqa 1024 x 0.5625 B (q4_0) x 2 for K and V. So the geometry is
// exact once the layer split is known, and the layer split is printed. ggrun's
// own estimate for this model was 9238 MiB against an actual 13864 -- a 4.6 GB
// error feeding every placement decision.
//
// The distinction matters most for --swa-full, which gives the sliding-window
// layers the full context. A rate measured without it under-predicts by the
// SWA layer count: here 13.8 GB against 54.0 GB, a 4x miss that no linear model
// can express.

// kvCacheLineRe matches llama.cpp's per-cache summary:
//
//	llama_kv_cache: size = 13824.00 MiB (1048576 cells,  12 layers,  1/1 seqs), K (q4_0): ...
//
// The trailing seqs field is captured because "cells" is per sequence while
// "size" is the total across all of them. Measured on one model at one context,
// changing only --parallel:
//
//	1728.00 MiB (131072 cells, 12 layers, 1/1 seqs) -> 1152 B/cell/layer
//	3456.00 MiB (131072 cells, 12 layers, 2/2 seqs) -> 2304 B/cell/layer
//
// Dividing by cells x layers alone therefore scales the recorded width by
// n_seq_max, and the cache is keyed by model rather than by parallelism, so a
// single --parallel 2 launch poisons every later plan for that model.
var kvCacheLineRe = regexp.MustCompile(
	`size\s*=\s*([0-9.]+)\s*MiB\s*\(\s*(\d+)\s*cells,\s*(\d+)\s*layers(?:,\s*(\d+)\s*/\s*\d+\s*seqs)?`)

// KVGeometry is how the backend actually laid out the KV cache.
type KVGeometry struct {
	// FullLayers attend over the whole context; SWALayers attend over a window.
	FullLayers int
	SWALayers  int
	// SWACells is the depth the backend gave the windowed cache. It is not the
	// sliding window itself -- llama.cpp adds a microbatch of slack, measured
	// 1024 cells for n_swa 512 at -ub 512.
	SWACells int
	// BytesPerCellPerLayer is identical for both caches, so either one derives it.
	BytesPerCellPerLayer float64
}

// Measured reports whether the geometry carries a usable layout.
func (g KVGeometry) Measured() bool {
	return g.BytesPerCellPerLayer > 0 && (g.FullLayers > 0 || g.SWALayers > 0)
}

// TotalMB is the KV allocation for a context, with or without --swa-full.
//
// swaFull gives the windowed layers the full context, which is what makes
// prefix reuse possible on these models and is why the cost has to be
// predictable before the launch rather than discovered by an OOM.
func (g KVGeometry) TotalMB(ctxSize int, swaFull bool) int {
	if !g.Measured() || ctxSize <= 0 {
		return 0
	}
	cells := int64(g.FullLayers) * int64(ctxSize)
	if swaFull {
		cells += int64(g.SWALayers) * int64(ctxSize)
	} else {
		swaCells := g.SWACells
		if swaCells > ctxSize {
			swaCells = ctxSize
		}
		cells += int64(g.SWALayers) * int64(swaCells)
	}
	return int(float64(cells)*g.BytesPerCellPerLayer/(1024*1024) + 0.5)
}

// anyMeasuredKVGeometry returns one usable geometry for this model together
// with the KV type it was measured at. Which one does not matter -- the layout
// is identical across types and only the per-cell width differs -- but the map
// iterates in random order, so pick deterministically to keep a plan stable
// across runs.
func anyMeasuredKVGeometry(model *ModelProfile) (KVGeometry, string, bool) {
	if model == nil {
		return KVGeometry{}, "", false
	}
	best := ""
	for kvType, g := range model.MeasuredKVGeometry {
		if !g.Measured() {
			continue
		}
		if best == "" || kvType < best {
			best = kvType
		}
	}
	if best == "" {
		return KVGeometry{}, "", false
	}
	return model.MeasuredKVGeometry[best], best, true
}

// rescaleKVGeometry converts a geometry measured at one KV type to another by
// the ratio of their per-element widths. The cell counts are untouched: how
// many layers attend over what depth is set by the architecture, not by how
// the cells are quantised.
func rescaleKVGeometry(g KVGeometry, fromType, toType string) (KVGeometry, bool) {
	if !g.Measured() {
		return g, false
	}
	if strings.EqualFold(fromType, toType) {
		return g, true
	}
	fromBytes, ok1 := kvTypeBytesPerElement(fromType)
	toBytes, ok2 := kvTypeBytesPerElement(toType)
	if !ok1 || !ok2 || fromBytes <= 0 {
		return g, false
	}
	g.BytesPerCellPerLayer *= toBytes / fromBytes
	return g, true
}

// kvCacheLogIsSWAFlattened reports whether a log carries several KV caches that
// all have the same depth. That is the --swa-full signature, and it is the one
// case where a failed parse means "this launch cannot describe the model"
// rather than "this log has no KV lines in it".
func kvCacheLogIsSWAFlattened(logText string) bool {
	depths := map[int]bool{}
	count := 0
	for _, line := range strings.Split(logText, "\n") {
		if !strings.Contains(line, "llama_kv_cache") {
			continue
		}
		m := kvCacheLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		cells, err := strconv.Atoi(m[2])
		if err != nil || cells <= 0 {
			continue
		}
		depths[cells] = true
		count++
	}
	return count > 1 && len(depths) == 1
}

// ParseKVGeometry reads the layout out of a backend launch log.
//
// A model without sliding-window layers prints one cache line and lands in
// FullLayers with SWALayers zero, so the same accounting covers both cases.
func ParseKVGeometry(logText string) (KVGeometry, bool) {
	var g KVGeometry
	type entry struct {
		mib    float64
		cells  int
		layers int
		seqs   int
	}
	var entries []entry
	for _, line := range strings.Split(logText, "\n") {
		if !strings.Contains(line, "llama_kv_cache") {
			continue
		}
		m := kvCacheLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		mib, err1 := strconv.ParseFloat(m[1], 64)
		cells, err2 := strconv.Atoi(m[2])
		layers, err3 := strconv.Atoi(m[3])
		if err1 != nil || err2 != nil || err3 != nil || cells <= 0 || layers <= 0 || mib <= 0 {
			continue
		}
		// Absent on older builds that printed no seqs field; those were always
		// single-sequence, so one is the right default rather than a guess.
		seqs := 1
		if m[4] != "" {
			if parsed, err := strconv.Atoi(m[4]); err == nil && parsed > 0 {
				seqs = parsed
			}
		}
		entries = append(entries, entry{mib, cells, layers, seqs})
	}
	if len(entries) == 0 {
		return g, false
	}
	// The legacy geometry is deliberately limited to the uniform one- or
	// two-region layout it can represent. Recurrent hybrids such as DeepSeek-V4
	// print raw, CSA, HCA, and lightning-indexer regions with different depths
	// and effective widths. Collapsing those into "full + SWA" over-counts the
	// layer total and rescaling the result by KV quantization changes auxiliary
	// buffers that do not use that type. Keep such observations exact-shape-only
	// in the scoped allocation cache instead.
	if len(entries) > 2 {
		return KVGeometry{}, false
	}
	for _, e := range entries {
		// TotalMB has no sequence dimension and this model-wide cache is not
		// parallel-scoped. A multi-sequence observation must never be reused.
		if e.seqs != 1 {
			return KVGeometry{}, false
		}
	}
	// The deepest cache is the full-attention one; anything shallower is
	// windowed. Depth rather than layer count, because a model can have more
	// full layers than windowed ones.
	deepest := entries[0]
	shallowest := entries[0]
	for _, e := range entries {
		if e.cells > deepest.cells {
			deepest = e
		}
		if e.cells < shallowest.cells {
			shallowest = e
		}
	}
	// --swa-full sets size_swa = size_base, so every cache prints the same depth
	// and the windowed layers become indistinguishable from the full ones. The
	// split is what this geometry exists to record, and a log that flattened it
	// cannot supply it -- reading one anyway yields "all layers full attention",
	// which over-reserves 4x on a plain launch and makes --swa-full look free.
	// Report no measurement so an earlier, non-flattened one survives.
	if len(entries) > 1 && deepest.cells == shallowest.cells {
		return KVGeometry{}, false
	}
	deepestWidth := deepest.mib * 1024 * 1024 / (float64(deepest.cells) * float64(deepest.layers))
	for _, e := range entries {
		width := e.mib * 1024 * 1024 / (float64(e.cells) * float64(e.layers))
		// Allow only log-rounding noise. A material difference means the two
		// regions cannot share BytesPerCellPerLayer and must remain scoped.
		if width < deepestWidth*0.98 || width > deepestWidth*1.02 {
			return KVGeometry{}, false
		}
	}
	for _, e := range entries {
		if e.cells == deepest.cells {
			g.FullLayers += e.layers
		} else {
			g.SWALayers += e.layers
			if e.cells > g.SWACells {
				g.SWACells = e.cells
			}
		}
	}
	g.BytesPerCellPerLayer = deepestWidth
	return g, g.Measured()
}
