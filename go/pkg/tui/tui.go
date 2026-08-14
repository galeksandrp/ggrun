package tui

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/claudesession"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/modelusage"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/probe"
	"github.com/raketenkater/ggrun/pkg/recommend"
	"github.com/raketenkater/ggrun/pkg/tune"
)

var (
	titleStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#7D56F4"))
	subtitleStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#A0A0A0"))
	selectedStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#7D56F4")).Bold(true)
	highlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00FF00"))
	warningStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	errorStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF0000"))
	recommendStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00AAFF")).Bold(true)
	mutedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666666"))
)

// Screen represents the current TUI screen.
type Screen int

const (
	ScreenMain Screen = iota
	ScreenModelConfig
	ScreenPrelaunch
	ScreenTunedPicker
	ScreenSettings
	ScreenDownload
	ScreenBackend
	ScreenFirstRun
	ScreenRecommended
	ScreenChoice
)

// Model is the Bubble Tea model.
type Model struct {
	screen Screen
	width  int
	height int

	// Data
	caps   *detect.Capabilities
	models []ModelItem
	// modelUsage holds per-model launch usage (count + last-used), used to sort
	// the main list so the most-used and most-recent models surface first. It is
	// loaded once at startup; launch records are written by the CLI launcher.
	modelUsage             map[string]modelusage.Record
	backend                string
	modelDir               string
	settingsPath           string
	cacheDir               string
	port                   int
	recommendationGroups   recommend.Categories
	recommendations        []recommend.Recommendation
	selectedRecommendation int
	// pendingDownload holds the repo/quant already chosen while the destination
	// directory is being asked for, so a download is only emitted once both
	// halves are known.
	pendingDownload *LaunchRequest

	// Main menu list
	mainList list.Model

	// Quick launch / smart predictions
	selectedModel int

	// Advanced config
	ctxSize        string
	ctxMode        string
	kvPlacement    string
	kvQuality      string
	swaFull        bool
	vramHeadroomMB int
	ramHeadroomMB  int
	parallel       string
	parallelSet    bool
	aitune         bool
	aituneRounds   int
	benchmark      bool
	vision         bool
	claudeCode     bool
	// claudeReviewer picks the local worker/reviewer model for Claude Code:
	// "auto" keeps ggrun's automatic choice, "qwen" forces the Qwen profile,
	// "nanbeige" forces the NanoBeige4.2 worker. Empty means auto.
	claudeReviewer string
	supportExpert  string
	supportOnline  bool
	// noCachedConfig derives this launch fresh, ignoring cached placement/probe
	// measurements (--no-cached-config). Per-launch; never persisted.
	noCachedConfig bool
	// claudeProfile is deliberately per-launch: an empty value preserves the
	// CLI default instead of writing a scheduling policy into user config.
	claudeProfile string

	// Tuned config
	tunedConfigs []tune.ConfigEntry
	tunedIndex   int // -1 = auto, 0+ = selected
	tunePath     string

	// Inputs
	input     textinput.Model
	inputMode string

	// Settings screen (arrow-navigable list of all config options)
	settingsCfg      *config.Config
	swaFullTouched   bool
	kvQualityTouched bool
	settingsCursor   int
	ramLimitPercent  int

	// Advanced (per-launch) config screen cursor
	cfgCursor int

	// First-run / quick-launch action-menu cursor
	menuCursor int

	// Recommended-downloads memory reserve control focus: "", "vram", or "ram".
	recHeadroomFocus string

	// Generic arrow-select screen
	choiceTitle   string
	choiceOptions []string
	choiceCursor  int
	choiceApply   func(*Model, string)
	choiceReturn  Screen

	// Resumable Claude Code session for the working directory, discovered when
	// the pre-launch screen is opened.
	resumeSession string
	resumeRun     string
	resumeCached  int

	// Launch request (set when user chooses to launch)
	launchRequest *LaunchRequest
	// replayRequest is the exact last TUI request while its pre-launch review
	// screen is open. It is kept separate from the editable Model fields so
	// confirming a replay cannot silently regenerate different arguments.
	replayRequest *LaunchRequest
	replaySavedAt time.Time
	// backendRouteBypass is set only when the user explicitly chooses to try a
	// model once without its reviewed architecture backend. It is per-selection
	// runtime state and is never persisted.
	backendRouteBypass        bool
	backendRouteBypassBackend string

	// Messages
	message        string
	messageType    string // info, warning, error
	scanningModels bool
}

// ModelItem represents a discovered GGUF model.
type ModelItem struct {
	Name          string
	Path          string
	Tuned         int
	SizeGB        float64
	Arch          string
	Architecture  string // exact GGUF architecture, without display suffixes
	IsMoE         bool
	AutoBackend   string                  // installed route selected for this architecture
	BackendRecipe string                  // reviewed route recipe when no installed route exists
	MaxCtx        int                     // trained max context from GGUF
	FitCtx        int                     // empirically proven fit context from probes
	External      bool                    // found outside the configured primary model directory
	KVProfile     *placement.ModelProfile // metadata for the live Full-SWA KV estimate
}

type modelScanFinishedMsg struct {
	result   modelstore.ScanResult
	cacheErr error
}

func scanComputerModels(modelDir, cacheDir string) tea.Cmd {
	return func() tea.Msg {
		result := modelstore.ScanComputer(modelDir)
		return modelScanFinishedMsg{
			result:   result,
			cacheErr: modelstore.SaveDiscoveredScan(cacheDir, result),
		}
	}
}

func InitialModel() Model {
	cfg, err := config.Load()
	configErr := err
	if err != nil || cfg == nil {
		cfg = config.Defaults()
	}
	settingsPath := config.Path()
	backend := cfg.Backend
	if backend == "" {
		backend = "auto"
	}
	rounds := cfg.TuneRounds
	if rounds <= 0 {
		rounds = 8
	}
	ctxValue := cfg.CtxValue()
	ctxMode := cfg.CtxMode()
	parallel := ""
	if cfg.Parallel > 0 {
		parallel = strconv.Itoa(cfg.Parallel)
	}
	m := Model{
		screen:          ScreenMain,
		backend:         backend,
		modelDir:        cfg.ModelDir,
		settingsPath:    settingsPath,
		cacheDir:        cfg.CacheDir,
		port:            cfg.Port,
		ctxSize:         ctxValue,
		ctxMode:         ctxMode,
		kvPlacement:     cfg.KVPlacement,
		kvQuality:       cfg.KVQuality,
		swaFull:         cfg.SWAFull,
		parallel:        parallel,
		parallelSet:     cfg.Parallel > 0 && cfg.IsExplicit("PARALLEL"),
		vision:          cfg.Vision,
		supportExpert:   cfg.SupportExpert,
		supportOnline:   cfg.SupportOnline,
		aituneRounds:    rounds,
		ramLimitPercent: cfg.RAMLimitPercent,
	}
	m.modelUsage = modelusage.Load(cfg.CacheDir)
	if m.port <= 0 {
		m.port = 8081
	}
	if m.ctxSize == "" {
		m.ctxSize = "fit"
	}
	if m.kvPlacement == "" {
		m.kvPlacement = "auto"
	}
	if m.kvQuality == "" {
		m.kvQuality = "auto"
	}

	m.input = textinput.New()
	m.input.Placeholder = ""
	m.input.Focus()

	// Detect once, then reuse that result while enriching every discovered GGUF.
	caps, _ := detect.Detect()
	m.caps = caps
	m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)

	m.vramHeadroomMB = config.ParseBudgetMB(cfg.VRAMHeadroom)
	m.ramHeadroomMB = config.ParseBudgetMB(cfg.RAMHeadroom)
	m.refreshRecommendations()

	if len(m.models) == 0 {
		m.screen = ScreenFirstRun
	}

	m.mainList = newMainList(m.models)
	if configErr != nil {
		m.message = fmt.Sprintf("Warning: Configuration error: %v. Fix it with ggrun config edit or reset before launching.", configErr)
		m.messageType = "warning"
	}
	return m
}

func flattenRecommendationCategories(cats recommend.Categories) []recommend.Recommendation {
	total := len(cats.Balanced) + len(cats.Smartest) + len(cats.Fastest)
	rows := make([]recommend.Recommendation, 0, total)
	rows = append(rows, cats.Balanced...)
	rows = append(rows, cats.Smartest...)
	rows = append(rows, cats.Fastest...)
	return rows
}

func newMainList(models []ModelItem) list.Model {
	items := []list.Item{
		mainItem{title: "r. Recommended downloads", desc: "Best models and quants that fit this computer", isAction: true, action: "recommend"},
		mainItem{title: "l. Run latest configuration", desc: "Review and replay the exact previous TUI launch", isAction: true, action: "latest"},
		mainItem{title: "p. Scan computer for models", desc: "Find GGUFs on all local disks and remember their paths", isAction: true, action: "scan"},
	}
	for i, m := range models {
		desc := fmt.Sprintf("%.1fGB, %s", m.SizeGB, m.Arch)
		if m.AutoBackend != "" {
			desc += "  [backend: " + m.AutoBackend + "]"
		} else if m.BackendRecipe != "" {
			desc += "  [install backend: " + m.BackendRecipe + "]"
		}
		if m.Tuned > 0 {
			desc += fmt.Sprintf("  [tuned: %d]", m.Tuned)
		}
		if m.External {
			desc += "  [discovered: " + filepath.Dir(m.Path) + "]"
		}
		items = append(items, mainItem{
			title:   fmt.Sprintf("%d. %s", i+1, m.Name),
			desc:    desc,
			index:   i,
			isModel: true,
		})
	}
	// Minimal action items
	items = append(items, mainItem{title: "d. Download model", desc: "Get from Hugging Face", isAction: true, action: "download"})
	items = append(items, mainItem{title: "m. Model directory", desc: "Change search path", isAction: true, action: "modeldir"})
	items = append(items, mainItem{title: "b. Backend", desc: "Auto-select or choose an installed backend", isAction: true, action: "backend"})
	items = append(items, mainItem{title: "f. Backend forks", desc: "Install or manage llama.cpp-compatible forks", isAction: true, action: "backend-forks"})
	items = append(items, mainItem{title: "s. Settings", desc: "All options (arrow keys)", isAction: true, action: "settings"})
	items = append(items, mainItem{title: "u. Update", desc: "Update ggrun and installed backends", isAction: true, action: "update"})
	items = append(items, mainItem{title: "q. Quit", desc: "Exit", isAction: true, action: "quit"})

	l := list.New(items, mainItemDelegate{}, 40, 20)
	l.Title = ""
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(true)
	l.SetShowHelp(false)
	return l
}

// rebuildMainList replaces m.mainList (e.g. after the model set changes) and
// re-applies the real terminal size plus the outgoing list's cursor position
// and active search filter. newMainList's own list.New call always starts a
// brand-new list.Model at index 0, unfiltered, with a placeholder 40x20 size;
// without re-applying all three, any mid-session rebuild (a model delete, a
// model-directory change, a tuned-count refresh after switching backends)
// silently resets the user's scroll position and drops whatever they were
// searching for, on top of the terminal-size regression this helper was
// first written to fix. Before the first WindowSizeMsg arrives (m.width/
// m.height still zero, e.g. during initial Model construction), this
// intentionally leaves the placeholder size in place — the upcoming
// WindowSizeMsg sizes it correctly on its own; there's no prior selection or
// filter to restore at that point either.
func (m *Model) rebuildMainList() {
	prevIndex := m.mainList.Index()
	prevFilterState := m.mainList.FilterState()
	prevFilterValue := m.mainList.FilterValue()

	sortModels(m.models, m.modelUsage)
	m.mainList = newMainList(m.models)
	if m.width > 0 {
		m.mainList.SetWidth(m.width - 4)
	}
	if m.height > 0 {
		m.mainList.SetHeight(m.height - 12)
	}

	if prevFilterState != list.Unfiltered && prevFilterValue != "" {
		m.mainList.SetFilterText(prevFilterValue)
	}
	// Clamp against VisibleItems(), not Items(): once a filter is applied,
	// list.Select()/the paginator index into the FILTERED set, and clamping
	// against the raw (unfiltered) count can leave prevIndex pointing past
	// the last filtered page — the next View() then slices out of range and
	// panics, killing the whole TUI session.
	if itemCount := len(m.mainList.VisibleItems()); itemCount > 0 {
		if prevIndex >= itemCount {
			prevIndex = itemCount - 1
		}
		if prevIndex > 0 {
			m.mainList.Select(prevIndex)
		}
	}
}

// openTunedPicker rebuilds the tuned-config list for the current model and
// opens the picker with the cursor on whatever config is actually active
// (m.tunePath), instead of always defaulting to "Auto". Without this,
// reopening the picker just to look — then reflexively pressing Enter to
// close it — silently reverted an already-chosen tuned config back to Auto.
func (m *Model) openTunedPicker() {
	m.tunedConfigs = tune.ListTunedConfigs(m.cacheDir, m.models[m.selectedModel].Name, m.backendTag(), false)
	m.tunedIndex = -1
	if m.tunePath != "" {
		for i, c := range m.tunedConfigs {
			if c.Path == m.tunePath {
				m.tunedIndex = i
				break
			}
		}
	}
	m.screen = ScreenTunedPicker
}

type mainItem struct {
	title    string
	desc     string
	index    int
	isModel  bool
	isAction bool
	action   string
}

func (i mainItem) Title() string       { return i.title }
func (i mainItem) Description() string { return i.desc }
func (i mainItem) FilterValue() string { return i.title + " " + i.desc }

type mainItemDelegate struct{}

func (d mainItemDelegate) Height() int                             { return 2 }
func (d mainItemDelegate) Spacing() int                            { return 1 }
func (d mainItemDelegate) Update(_ tea.Msg, _ *list.Model) tea.Cmd { return nil }
func (d mainItemDelegate) Render(w io.Writer, m list.Model, index int, listItem list.Item) {
	i, ok := listItem.(mainItem)
	if !ok {
		return
	}
	if index == m.Index() {
		fmt.Fprint(w, selectedStyle.Render("▸ "+i.title)+"\n  "+subtitleStyle.Render(i.desc))
	} else {
		fmt.Fprint(w, "  "+i.title+"\n  "+subtitleStyle.Render(i.desc))
	}
}

func (m Model) Init() tea.Cmd {
	return nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case modelScanFinishedMsg:
		m.scanningModels = false
		// A scan can finish while the user is already configuring a model. Keep
		// both that active model and the highlighted Main-menu row attached to
		// their paths while the newly found models are merged and sorted.
		selectedModelID := ""
		if m.selectedModel >= 0 && m.selectedModel < len(m.models) {
			selectedModelID = modelIdentity(m.models[m.selectedModel].Path)
		}
		focusedModelID, focusedAction := "", ""
		if raw := m.mainList.SelectedItem(); raw != nil {
			if item, ok := raw.(mainItem); ok {
				focusedAction = item.action
				if item.isModel && item.index >= 0 && item.index < len(m.models) {
					focusedModelID = modelIdentity(m.models[item.index].Path)
				}
			}
		}
		// Use this scan's in-memory result even if persistence failed; a cache
		// write problem must not discard paths we just found for this session.
		items := mergeModelItems(discoverModels(m.modelDir), discoverModelsFromPaths(msg.result.Paths))
		m.models = enrichModelItems(items, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		if selectedModelID != "" {
			for i := range m.models {
				if modelIdentity(m.models[i].Path) == selectedModelID {
					m.selectedModel = i
					break
				}
			}
		}
		for i, raw := range m.mainList.VisibleItems() {
			item, ok := raw.(mainItem)
			if !ok {
				continue
			}
			if focusedModelID != "" && item.isModel && item.index >= 0 && item.index < len(m.models) && modelIdentity(m.models[item.index].Path) == focusedModelID {
				m.mainList.Select(i)
				break
			}
			if focusedModelID == "" && focusedAction != "" && item.action == focusedAction {
				m.mainList.Select(i)
				break
			}
		}
		external := 0
		for _, model := range m.models {
			if model.External {
				external++
			}
		}
		seconds := msg.result.Duration.Round(100 * time.Millisecond)
		m.message = fmt.Sprintf("Computer scan complete: %d runnable model(s), %d outside the primary directory (%s)", len(m.models), external, seconds)
		m.messageType = "info"
		if msg.result.Truncated {
			m.message += "; scan limit reached, showing partial results"
			m.messageType = "warning"
		}
		if msg.cacheErr != nil {
			m.message += fmt.Sprintf("; results could not be saved: %v", msg.cacheErr)
			m.messageType = "warning"
		}
		if m.screen == ScreenFirstRun && len(m.models) > 0 {
			m.screen = ScreenMain
		}
		return m, nil

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.mainList.SetWidth(msg.Width - 4)
		m.mainList.SetHeight(msg.Height - 12)
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q":
			if m.screen == ScreenMain {
				if m.mainList.FilterState() != list.Filtering {
					return m, tea.Quit
				}
			}
		case "esc":
			if m.screen == ScreenChoice {
				m.screen = m.choiceReturn
				return m, nil
			}
			// Cancel any in-progress free-text edit. On ScreenModelConfig/
			// ScreenSettings this stays put - there's a row menu on the same
			// screen to fall back into. ScreenDownload/ScreenBackend are pure
			// single-field prompts with no such fallback (updateInputScreen
			// has no "esc" case of its own), so clearing the input there also
			// has to leave the screen in the same keypress, or it dead-ends on
			// a blurred, unresponsive input box until a second Esc/Enter.
			if m.inputMode != "" {
				m.inputMode = ""
				m.input.Blur()
				if m.screen == ScreenDownload || m.screen == ScreenBackend {
					m.screen = ScreenMain
					m.message = ""
				}
				return m, nil
			}
			// These screens own their own "esc" case (back a level, or defocus a
			// control before leaving) - let the per-screen dispatch below run
			// instead of jumping straight to Main and skipping it.
			switch m.screen {
			case ScreenPrelaunch, ScreenTunedPicker, ScreenRecommended:
			case ScreenMain:
			default:
				m.screen = ScreenMain
				m.message = ""
				return m, nil
			}
		}
	}

	switch m.screen {
	case ScreenMain:
		return m.updateMain(msg)
	case ScreenModelConfig:
		return m.updateModelConfig(msg)
	case ScreenPrelaunch:
		return m.updatePrelaunch(msg)
	case ScreenTunedPicker:
		return m.updateTunedPicker(msg)
	case ScreenFirstRun:
		return m.updateFirstRun(msg)
	case ScreenRecommended:
		return m.updateRecommended(msg)
	case ScreenSettings:
		return m.updateSettings(msg)
	case ScreenChoice:
		return m.updateChoice(msg)
	case ScreenDownload, ScreenBackend:
		return m.updateInputScreen(msg)
	}

	return m, nil
}

func (m Model) updateMain(msg tea.Msg) (tea.Model, tea.Cmd) {
	// While the search box is active, every printable key belongs to the filter;
	// do not let shortcuts such as r/s/b steal letters from the query.
	if m.mainList.FilterState() == list.Filtering {
		var cmd tea.Cmd
		m.mainList, cmd = m.mainList.Update(msg)
		return m, cmd
	}
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "r", "R":
			m.selectedRecommendation = 0
			m.screen = ScreenRecommended
			return m, nil
		case "p", "P":
			return m.startComputerModelScan()
		case "l", "L":
			return m.openLatestLaunch()
		case "enter":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok {
				if item.isModel {
					// Open the config screen first so settings (context, KV,
					// Claude Code, …) are discoverable. The recommended defaults
					// are pre-filled, so launching is one more keypress: press L
					// (or Enter on the Launch row) to start.
					m.selectedModel = item.index
					m.backendRouteBypass = false
					m.replayRequest = nil
					m.replaySavedAt = time.Time{}
					m.cfgCursor = 0
					m.screen = ScreenModelConfig
					return m, nil
				}
				switch item.action {
				case "recommend":
					m.selectedRecommendation = 0
					m.screen = ScreenRecommended
					return m, nil
				case "scan":
					return m.startComputerModelScan()
				case "latest":
					return m.openLatestLaunch()
				case "download":
					m.screen = ScreenDownload
					m.inputMode = "download"
					m.input.SetValue("")
					m.input.Placeholder = "Hugging Face repo (e.g. unsloth/Llama-3.2-1B-Instruct)"
					m.input.Focus()
				case "modeldir":
					m.screen = ScreenBackend
					m.inputMode = "modeldir"
					m.input.SetValue(m.modelDir)
					m.input.Placeholder = "Path to model directory"
					m.input.Focus()
				case "backend":
					m.openBackendChoice(ScreenMain)
				case "backend-forks":
					m.openBackendManager(ScreenMain)
				case "settings":
					m.openSettings()
				case "update":
					m.launchRequest = &LaunchRequest{Update: true}
					return m, tea.Quit
				case "quit":
					return m, tea.Quit
				}
			}
		case "s", "S":
			m.openSettings()
		case "u", "U":
			m.launchRequest = &LaunchRequest{Update: true}
			return m, tea.Quit
		case "b", "B":
			m.openBackendChoice(ScreenMain)
		case "f", "F":
			m.openBackendManager(ScreenMain)
		case "c", "C":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok && item.isModel {
				m.selectedModel = item.index
				m.backendRouteBypass = false
				m.replayRequest = nil
				m.replaySavedAt = time.Time{}
				m.cfgCursor = 0
				m.screen = ScreenModelConfig
				return m, nil
			}
		case "x", "X":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok && item.isModel {
				m.openRemoveModelChoice(item.index)
				return m, nil
			}
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

func (m Model) startComputerModelScan() (tea.Model, tea.Cmd) {
	if m.scanningModels {
		m.message = "Computer model scan is already running"
		m.messageType = "info"
		return m, nil
	}
	m.scanningModels = true
	m.message = "Scanning local disks for GGUF models… You can keep using the TUI."
	m.messageType = "info"
	return m, scanComputerModels(m.modelDir, m.cacheDir)
}

// openLatestLaunch loads the last request emitted by the TUI, restores its
// presentation fields, and opens the normal pre-launch screen. Enter then
// returns the saved request byte-for-byte at the field level; Esc converts it
// back into an editable configuration instead of launching anything.
func (m Model) openLatestLaunch() (tea.Model, tea.Cmd) {
	req, savedAt, err := LoadLatestLaunch(m.cacheDir)
	if err != nil {
		if errors.Is(err, ErrNoLatestLaunch) {
			m.message = "No previous TUI launch configuration has been saved yet"
		} else {
			m.message = fmt.Sprintf("Could not load latest TUI configuration: %v", err)
		}
		m.messageType = "warning"
		return m, nil
	}
	if info, err := os.Stat(req.ModelPath); err != nil || info.IsDir() {
		m.message = fmt.Sprintf("Latest configuration is unavailable because its model is missing: %s", req.ModelPath)
		m.messageType = "warning"
		return m, nil
	}
	if req.Backend == "" {
		m.backend = "auto"
	} else {
		m.backend = req.Backend
	}

	if !m.selectModelPath(req.ModelPath) {
		m.message = fmt.Sprintf("Latest configuration model is not a runnable GGUF: %s", req.ModelPath)
		m.messageType = "warning"
		return m, nil
	}

	m.applyLaunchRequestFields(req)
	m.backendRouteBypass = false
	copyReq := *req
	copyReq.BackendArgs = append([]string(nil), req.BackendArgs...)
	m.replayRequest = &copyReq
	m.replaySavedAt = savedAt
	m.message = ""
	m.messageType = ""
	m.screen = ScreenPrelaunch
	return m, nil
}

// selectModelPath attaches a request to an already recognized model or adds
// that one existing GGUF without requiring another whole-computer scan.
func (m *Model) selectModelPath(path string) bool {
	want := modelIdentity(path)
	selected := -1
	for i := range m.models {
		if modelIdentity(m.models[i].Path) == want {
			selected = i
			break
		}
	}
	if selected < 0 {
		items := mergeModelItems(m.models, discoverModelsFromPaths([]string{path}))
		m.models = enrichModelItems(items, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		for i := range m.models {
			if modelIdentity(m.models[i].Path) == want {
				selected = i
				break
			}
		}
	}
	if selected < 0 {
		return false
	}
	m.selectedModel = selected
	return true
}

func (m *Model) applyLaunchRequestFields(req *LaunchRequest) {
	if req == nil {
		return
	}
	m.port = req.Port
	m.ctxMode, m.ctxSize = "fit", "fit"
	ctxFlag := strings.TrimSpace(req.CtxFlag)
	switch ctxFlag {
	case "", "fit", "auto":
	case "max", "native":
		m.ctxMode, m.ctxSize = "max", "max"
	default:
		m.ctxMode, m.ctxSize = "manual", ctxFlag
	}
	if ctxFlag == "" && req.CtxSize > 0 {
		m.ctxMode, m.ctxSize = "manual", strconv.Itoa(req.CtxSize)
	}
	if req.KVPlacement != "" {
		m.kvPlacement = req.KVPlacement
	}
	if req.KVQuality != "" {
		m.kvQuality = req.KVQuality
	}
	m.swaFull = req.SWAFull
	m.parallel, m.parallelSet = "", req.ParallelSet
	if req.ParallelSet && req.Parallel > 0 {
		m.parallel = strconv.Itoa(req.Parallel)
	}
	m.vision = req.Vision
	m.tunePath = req.TuneCache
	m.aitune = req.AITune
	m.aituneRounds = req.AITuneRounds
	m.benchmark = req.Benchmark
	m.claudeCode = req.ClaudeCode
	m.claudeProfile = req.ClaudeProfile
	m.claudeReviewer = req.ClaudeReviewerOverride
	if req.SupportSet || req.SupportExpert != "" {
		m.supportExpert = req.SupportExpert
		m.supportOnline = req.SupportOnline
	}
	m.noCachedConfig = req.NoCachedConfig
	m.resumeSession, m.resumeRun, m.resumeCached = req.ResumeSession, "", 0
	m.refreshTunedCounts()
}

// cfgRows returns the ordered focusable rows of the Advanced config screen.
func (m Model) cfgRows() []string {
	rows := []string{}
	if m.selectedBackendRecipe() != nil {
		rows = append(rows, "backend-install")
	}
	rows = append(rows, "context", "parallel", "kv", "kvq", "swa", "tuned", "aitune")
	if m.aitune {
		rows = append(rows, "rounds")
	}
	rows = append(rows, "vision", "claudecode")
	if m.claudeCode {
		rows = append(rows, "claudereviewer", "claudeprofile")
	}
	return append(rows, "benchmark", "launch", "dryrun")
}

func (m Model) selectedBackendRecipe() *backends.Recipe {
	if m.backendRouteBypass || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return nil
	}
	model := m.models[m.selectedModel]
	if model.AutoBackend != "" || model.BackendRecipe == "" {
		return nil
	}
	return backends.RecipeByName(model.BackendRecipe)
}

// effectiveBackend returns the model-specific installed route when one exists;
// otherwise it preserves the configured default backend. This keeps a Laguna,
// Hy3, or MiniMax fork scoped to the model that needs it instead of changing the
// user's global backend setting for every model.
func (m Model) effectiveBackend() string {
	if m.backendRouteBypass {
		if fallback := strings.TrimSpace(m.backendRouteBypassBackend); fallback != "" {
			return fallback
		}
	}
	if m.selectedModel >= 0 && m.selectedModel < len(m.models) {
		if routed := strings.TrimSpace(m.models[m.selectedModel].AutoBackend); routed != "" {
			return routed
		}
	}
	return m.backend
}

// openSelectedBackendInstall asks before the network clone/build. Confirming
// returns a normal backend CLI request carrying the model's current launch
// settings; cmdGUI can then reopen the same model at pre-launch with the newly
// registered route selected.
func (m *Model) openSelectedBackendInstall() bool {
	recipe := m.selectedBackendRecipe()
	if recipe == nil || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return false
	}
	model := m.models[m.selectedModel]
	arch := model.Architecture
	if arch == "" {
		arch = model.Arch
	}
	installLabel := "Install " + recipe.Name + " and select it"
	continueBackend := strings.TrimSpace(m.backend)
	if required := backends.RequiredBackendForArch(arch); required != "" {
		continueBackend = required
	}
	if continueBackend == "" {
		continueBackend = "auto"
	}
	continueLabel := "Continue once with " + continueBackend
	m.openChoice(
		"Backend for "+arch,
		[]string{"Cancel", installLabel, continueLabel},
		"Cancel",
		ScreenModelConfig,
		func(mm *Model, value string) {
			switch value {
			case installLabel:
				req := mm.buildLaunchRequest()
				if req == nil {
					return
				}
				req.Backend = recipe.Tag
				req.BackendArgs = []string{"install", recipe.Name}
				mm.launchRequest = req
			case continueLabel:
				mm.backendRouteBypass = true
				mm.backendRouteBypassBackend = continueBackend
				mm.message = "Continuing once with " + continueBackend + "; the model may fail if that backend lacks " + arch
				mm.messageType = "warning"
				mm.loadResumableSession()
				mm.screen = ScreenPrelaunch
			}
		},
	)
	return true
}

func (m *Model) openCfgInput(mode, val, placeholder string) {
	m.inputMode = mode
	m.input.SetValue(val)
	m.input.Placeholder = placeholder
	m.input.Focus()
}

func (m *Model) setCtx(val string) {
	val = strings.TrimSpace(val)
	switch val {
	case "", "fit":
		m.ctxMode = "fit"
		m.ctxSize = "fit"
	case "max":
		m.ctxMode = "max"
		m.ctxSize = "max"
	default:
		n, err := strconv.Atoi(val)
		if err != nil || n < 1 {
			m.message = "Warning: Context must be fit, max, or a positive token count"
			m.messageType = "warning"
			return
		}
		m.ctxMode = "manual"
		m.ctxSize = strconv.Itoa(n)
	}
}

// cycleCfgRow changes the focused Advanced-config row with ←/→ (dir -1/+1).
func (m *Model) cycleCfgRow(row string, dir int) {
	switch row {
	case "kv":
		order := []string{"auto", "gpu", "cpu"}
		if dir < 0 {
			m.kvPlacement = prevOption(order, m.kvPlacement)
		} else {
			m.kvPlacement = nextOption(order, m.kvPlacement)
		}
	case "kvq":
		// Same ordering as the Settings screen's "KV quality" enum, so arrow
		// cycling and the saved setting agree.
		order := []string{"auto", "high", "bf16", "mid", "q8_0", "q5_1", "q5_0", "q4_1", "low", "q4_0", "iq4_nl", "f32"}
		if dir < 0 {
			m.kvQuality = prevOption(order, m.kvQuality)
		} else {
			m.kvQuality = nextOption(order, m.kvQuality)
		}
		// Cycling this row is a deliberate per-launch choice, so it goes out as
		// an explicit flag. Merely inheriting the saved setting does not: that
		// stays in config, where ggrun still applies it but may withdraw it on a
		// memory failure.
		m.kvQualityTouched = true
	case "swa":
		m.swaFull = !m.swaFull
		// Cycling this row is a deliberate per-launch choice, so it goes out as
		// an explicit flag. Merely inheriting the saved setting does not: that
		// stays in config, where ggrun still applies it but may withdraw it on a
		// memory failure.
		m.swaFullTouched = true
	case "context":
		order := []string{"fit", "max"}
		cur := "fit"
		if m.ctxMode == "max" {
			cur = "max"
		}
		if dir < 0 {
			m.setCtx(prevOption(order, cur))
		} else {
			m.setCtx(nextOption(order, cur))
		}
	case "aitune":
		m.aitune = !m.aitune
		if m.aitune {
			m.benchmark = false
		}
	case "vision":
		m.vision = !m.vision
	case "claudecode":
		m.claudeCode = !m.claudeCode
	case "claudereviewer":
		order := []string{"auto", "qwen", "nanbeige"}
		if dir < 0 {
			m.claudeReviewer = prevOption(order, m.claudeReviewer)
		} else {
			m.claudeReviewer = nextOption(order, m.claudeReviewer)
		}
	case "claudeprofile":
		profiles := []string{"", "agent-interactive", "agent-parallel"}
		if dir < 0 {
			m.claudeProfile = prevOption(profiles, m.claudeProfile)
		} else {
			m.claudeProfile = nextOption(profiles, m.claudeProfile)
		}
	case "benchmark":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
	}
}

// activateCfgRow handles Enter on the focused Advanced-config row.
func (m Model) activateCfgRow(row string) (tea.Model, tea.Cmd) {
	switch row {
	case "backend-install":
		m.openSelectedBackendInstall()
	case "context":
		m.openCfgInput("ctx", m.ctxSize, "fit, max, or token count")
	case "parallel":
		m.openCfgInput("parallel", m.parallel, "Parallel slots (blank = let placement decide)")
	case "kv":
		m.cycleCfgRow("kv", 1)
	case "kvq":
		m.cycleCfgRow("kvq", 1)
	case "swa":
		m.cycleCfgRow("swa", 1)
	case "tuned":
		m.openTunedPicker()
	case "rounds":
		m.openCfgInput("aitune", strconv.Itoa(m.aituneRounds), "AI tune rounds (1-30, default 8)")
	case "aitune":
		m.aitune = !m.aitune
		if m.aitune {
			m.benchmark = false
		}
	case "vision":
		m.vision = !m.vision
	case "claudecode":
		m.claudeCode = !m.claudeCode
	case "claudereviewer":
		m.cycleCfgRow("claudereviewer", 1)
	case "claudeprofile":
		m.cycleCfgRow("claudeprofile", 1)
	case "benchmark":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
	case "launch":
		if m.openSelectedBackendInstall() {
			return m, nil
		}
		m.replayRequest = nil
		m.replaySavedAt = time.Time{}
		m.loadResumableSession()
		m.screen = ScreenPrelaunch
	case "dryrun":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	}
	return m, nil
}

func (m Model) updateModelConfig(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}

	// Free-text edit mode (context / parallel / AI-tune rounds).
	if m.inputMode != "" {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			val := m.input.Value()
			switch m.inputMode {
			case "ctx":
				m.setCtx(val)
			case "parallel":
				val = strings.TrimSpace(val)
				if val == "" {
					m.parallel = ""
					m.parallelSet = false
				} else if n, err := strconv.Atoi(val); err == nil && n > 0 {
					m.parallel = strconv.Itoa(n)
					m.parallelSet = true
				} else {
					m.message = "Warning: Parallel must be a positive integer"
					m.messageType = "warning"
				}
			case "aitune":
				if n, err := strconv.Atoi(strings.TrimSpace(val)); err == nil && n >= 1 && n <= 30 {
					m.aituneRounds = n
				}
			}
			m.inputMode = ""
		}
		return m, cmd
	}

	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	rows := m.cfgRows()
	if m.cfgCursor >= len(rows) {
		m.cfgCursor = len(rows) - 1
	}
	switch keyMsg.String() {
	// Arrow-key navigation (works alongside the letter hotkeys below).
	case "up":
		if m.cfgCursor > 0 {
			m.cfgCursor--
		}
	case "down":
		if m.cfgCursor < len(rows)-1 {
			m.cfgCursor++
		}
	case "left":
		m.cycleCfgRow(rows[m.cfgCursor], -1)
	case "right":
		m.cycleCfgRow(rows[m.cfgCursor], 1)
	case "enter":
		return m.activateCfgRow(rows[m.cfgCursor])
	// Letter hotkeys.
	case "c", "C":
		m.openCfgInput("ctx", m.ctxSize, "fit, max, or token count")
	case "p", "P":
		m.openCfgInput("parallel", m.parallel, "Parallel slots (blank = let placement decide)")
	case "K":
		m.cycleCfgRow("kv", 1)
	case "k":
		m.cycleCfgRow("kvq", 1)
	case "w", "W":
		m.cycleCfgRow("swa", 1)
	case "a", "A":
		m.aitune = !m.aitune
		if m.aitune {
			m.benchmark = false
		}
	case "r", "R":
		if m.aitune {
			m.openCfgInput("aitune", strconv.Itoa(m.aituneRounds), "AI tune rounds (1-30, default 8)")
		}
	case "b", "B":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
	case "v", "V":
		m.vision = !m.vision
	case "x", "X":
		m.claudeCode = !m.claudeCode
	case "l", "L":
		if m.openSelectedBackendInstall() {
			return m, nil
		}
		m.replayRequest = nil
		m.replaySavedAt = time.Time{}
		m.loadResumableSession()
		m.screen = ScreenPrelaunch
	case "d", "D":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	case "i", "I":
		m.openSelectedBackendInstall()
	case "t", "T":
		m.openTunedPicker()
	case "y", "Y":
		m.openClearCachesChoice(m.selectedModel)
	case "g", "G":
		m.noCachedConfig = !m.noCachedConfig
	case "q", "Q":
		return m, tea.Quit
	}
	return m, nil
}

// promptDownloadDir asks where a chosen download should land, pre-filled with
// the configured model directory so Enter keeps the existing behaviour.
func (m Model) promptDownloadDir(req *LaunchRequest) (tea.Model, tea.Cmd) {
	m.pendingDownload = req
	m.screen = ScreenDownload
	m.inputMode = "downloaddir"
	m.input.SetValue(m.modelDir)
	m.input.Placeholder = "Destination directory"
	m.input.Focus()
	m.message = fmt.Sprintf("Destination for %s (Enter keeps %s)", req.DownloadRepo, m.modelDir)
	m.messageType = "info"
	return m, nil
}

func firstRunActions() []string {
	return []string{"recommend", "latest", "scan", "download", "modeldir", "backends", "update", "quit"}
}

func (m Model) doFirstRunAction(action string) (tea.Model, tea.Cmd) {
	switch action {
	case "recommend":
		m.selectedRecommendation = 0
		m.screen = ScreenRecommended
	case "latest":
		return m.openLatestLaunch()
	case "scan":
		return m.startComputerModelScan()
	case "download":
		m.screen = ScreenDownload
		m.inputMode = "download"
		m.input.SetValue("")
		m.input.Placeholder = "Hugging Face repo"
		m.input.Focus()
	case "modeldir":
		m.screen = ScreenBackend
		m.inputMode = "modeldir"
		m.input.SetValue(m.modelDir)
		m.input.Placeholder = "Path to model directory"
		m.input.Focus()
	case "backends":
		m.openBackendManager(ScreenFirstRun)
	case "update":
		m.launchRequest = &LaunchRequest{Update: true}
		return m, tea.Quit
	case "quit":
		return m, tea.Quit
	}
	return m, nil
}

func (m Model) updateFirstRun(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	actions := firstRunActions()
	if m.menuCursor >= len(actions) {
		m.menuCursor = len(actions) - 1
	}
	switch keyMsg.String() {
	case "up":
		if m.menuCursor > 0 {
			m.menuCursor--
		}
	case "down":
		if m.menuCursor < len(actions)-1 {
			m.menuCursor++
		}
	case "enter":
		return m.doFirstRunAction(actions[m.menuCursor])
	case "r", "R":
		return m.doFirstRunAction("recommend")
	case "l", "L":
		return m.doFirstRunAction("latest")
	case "p", "P":
		return m.doFirstRunAction("scan")
	case "d", "D":
		return m.doFirstRunAction("download")
	case "m", "M":
		return m.doFirstRunAction("modeldir")
	case "f", "F":
		return m.doFirstRunAction("backends")
	case "u", "U":
		return m.doFirstRunAction("update")
	case "q", "Q":
		return m, tea.Quit
	}
	return m, nil
}

func (m Model) updateInputScreen(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
		val := m.input.Value()
		switch m.inputMode {
		case "download":
			val = strings.TrimSpace(val)
			if val != "" {
				return m.promptDownloadDir(&LaunchRequest{DownloadRepo: val})
			}
			m.message = "Warning: Enter a Hugging Face GGUF repository"
			m.messageType = "warning"
		case "downloaddir":
			req := m.pendingDownload
			if req == nil {
				m.inputMode = ""
				m.screen = ScreenFirstRun
				return m, nil
			}
			if dir := strings.TrimSpace(val); dir != "" && dir != m.modelDir {
				req.DownloadDir = dir
			}
			m.launchRequest = req
			return m, tea.Quit
		case "modeldir":
			val = strings.TrimSpace(val)
			if val != "" {
				m.modelDir = val
				m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)
				m.rebuildMainList()
				if err := persistConfig(func(c *config.Config) { c.ModelDir = val }); err != nil {
					m.message = fmt.Sprintf("Warning: Using %s for this session — could not save config: %v", val, err)
					m.messageType = "warning"
				} else {
					m.message = fmt.Sprintf("Model directory saved: %s (%d models)", val, len(m.models))
					m.messageType = "info"
				}
			}
		case "backend-add":
			fields := strings.Fields(val)
			if len(fields) == 0 {
				m.message = "Warning: Enter a Git repository URL"
				m.messageType = "warning"
				return m, cmd
			}
			args := []string{"add", fields[0]}
			if len(fields) > 1 {
				args = append(args, "--tag", fields[1])
			}
			if len(fields) > 2 {
				args = append(args, "--route-arch", fields[2])
			}
			m.launchRequest = &LaunchRequest{BackendArgs: args}
			return m, tea.Quit
		case "backend-register":
			fields := strings.Fields(val)
			if len(fields) < 2 {
				m.message = "Warning: Enter a tag and llama-server binary path"
				m.messageType = "warning"
				return m, cmd
			}
			args := []string{"register", "--tag", fields[0], "--path", fields[1]}
			if len(fields) > 2 {
				args = append(args, "--route-arch", fields[2])
			}
			m.launchRequest = &LaunchRequest{BackendArgs: args}
			return m, tea.Quit
		}
		m.inputMode = ""
		m.screen = ScreenMain
	}
	return m, cmd
}

func (m Model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	switch m.screen {
	case ScreenFirstRun:
		return m.viewFirstRun()
	case ScreenModelConfig:
		return m.viewModelConfig()
	case ScreenPrelaunch:
		return m.viewPrelaunch()
	case ScreenTunedPicker:
		return m.viewTunedPicker()
	case ScreenRecommended:
		return m.viewRecommended()
	case ScreenSettings:
		return m.viewSettings()
	case ScreenChoice:
		return m.viewChoice()
	case ScreenDownload, ScreenBackend:
		return m.viewInputScreen()
	default:
		return m.viewMain()
	}
}

func (m Model) viewMain() string {
	var b strings.Builder
	external := 0
	for _, model := range m.models {
		if model.External {
			external++
		}
	}

	b.WriteString(titleStyle.Render("═══ ggrun ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Backend:  %s\n", m.backend))
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString(fmt.Sprintf("  Models:   %d recognized (%d elsewhere)\n", len(m.models), external))
	b.WriteString(fmt.Sprintf("  Primary:  %s\n", m.modelDir))
	b.WriteString(fmt.Sprintf("  Settings: %s\n", m.settingsPath))
	b.WriteString("\n")

	if len(m.models) == 0 {
		b.WriteString("  (no GGUF models found)\n")
	}

	b.WriteString(m.mainList.View())

	b.WriteString("\n")
	b.WriteString(mutedStyle.Render("  Enter configure · l latest · / search · p scan disks · x delete · r downloads · s settings · u update · q quit"))

	if m.message != "" {
		b.WriteString("\n")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
	}

	return b.String()
}

func (m Model) viewFirstRun() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ ggrun First Run ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString(fmt.Sprintf("  No runnable GGUF models found in: %s\n", m.modelDir))
	b.WriteString("  Start with Recommended; ggrun will choose a model and quant that fit.\n")
	b.WriteString("\n")

	actions := firstRunActions()
	labels := map[string]string{
		"recommend": "[r] Recommended downloads for this machine",
		"latest":    "[l] Run latest saved configuration",
		"scan":      "[p] Scan all local disks for existing GGUF models",
		"download":  "[d] Manual Hugging Face repository",
		"modeldir":  "[m] Point at an existing model directory",
		"backends":  "[f] Install or manage backend forks",
		"update":    "[u] Update ggrun and backends",
		"quit":      "[q] Quit",
	}
	for i, a := range actions {
		if i == m.menuCursor {
			b.WriteString(selectedStyle.Render("▸ "+labels[a]) + "\n")
		} else {
			b.WriteString("  " + labels[a] + "\n")
		}
	}
	if m.message != "" {
		b.WriteString("\n  ")
		if m.messageType == "warning" {
			b.WriteString(warningStyle.Render(m.message))
		} else if m.messageType == "error" {
			b.WriteString(errorStyle.Render(m.message))
		} else {
			b.WriteString(highlightStyle.Render(m.message))
		}
		b.WriteString("\n")
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ move · Enter select · q quit"))
	return b.String()
}

func (m Model) viewModelConfig() string {
	if len(m.models) == 0 {
		return "No models"
	}
	model := m.models[m.selectedModel]
	var b strings.Builder

	b.WriteString(titleStyle.Render("⚙  Configure  ·  "+model.Name) + "\n")
	b.WriteString(mutedStyle.Render("  ↑/↓ move · ←/→ or Enter change · x Claude Code · Esc back") + "\n")

	rows := m.cfgRows()
	focused := ""
	if m.cfgCursor < len(rows) {
		focused = rows[m.cfgCursor]
	}
	line := func(key, label, value string) {
		if key == focused {
			b.WriteString(selectedStyle.Render(fmt.Sprintf("  ▸ %-26s %s", label, value)) + "\n")
		} else {
			b.WriteString(fmt.Sprintf("    %-26s ", label) + subtitleStyle.Render(value) + "\n")
		}
	}
	section := func(title string) {
		b.WriteString("\n" + recommendStyle.Render("  "+title) + "\n")
	}

	ctxLabel := m.ctxSize
	if m.ctxMode == "fit" {
		ctxLabel = "fit"
	}
	if model.FitCtx > 0 || model.MaxCtx > 0 {
		ctxHint := " ("
		if model.FitCtx > 0 {
			ctxHint += fmt.Sprintf("fits ~%d", model.FitCtx)
		}
		if model.FitCtx > 0 && model.MaxCtx > 0 {
			ctxHint += ", "
		}
		if model.MaxCtx > 0 {
			ctxHint += fmt.Sprintf("train max %d", model.MaxCtx)
		}
		ctxLabel += ctxHint + ")"
	}
	parallelLabel := m.parallel
	if !m.parallelSet && (parallelLabel == "" || parallelLabel == "1") {
		parallelLabel = "automatic (1 normally; up to 4 for Claude Code)"
	}
	kvLabel := "auto (GPU KV first)"
	if m.kvPlacement == "gpu" {
		kvLabel = "gpu (best long-context decode)"
	} else if m.kvPlacement == "cpu" {
		kvLabel = "cpu (more GPU experts for short chat)"
	}
	tuneLabel := "auto"
	if m.tunePath != "" {
		tuneLabel = filepath.Base(m.tunePath)
	}

	kvQualityLabel := map[string]string{
		"high": "high (f16)", "mid": "mid (q8_0)", "low": "low (q4_0)",
		"q4_1": "q4_1", "iq4_nl": "iq4_nl", "q5_0": "q5_0", "q5_1": "q5_1",
		"bf16": "bf16", "f16": "f16", "f32": "f32",
	}[m.kvQuality]
	if kvQualityLabel == "" {
		kvQualityLabel = m.kvQuality
	}

	section("Backend, context & memory")
	switch {
	case model.AutoBackend != "":
		line("backend", "Backend", model.AutoBackend+" (auto-selected for "+model.Architecture+")")
	case m.selectedBackendRecipe() != nil:
		line("backend-install", "[i] Backend", "install "+model.BackendRecipe+" for "+model.Architecture)
	case m.backendRouteBypass && model.BackendRecipe != "":
		line("backend", "Backend", m.effectiveBackend()+" (unsupported-route check bypassed once)")
	default:
		line("backend", "Backend", m.backend)
	}
	line("context", "[c] Context size", ctxLabel)
	line("parallel", "[p] Parallel slots", parallelLabel)
	line("kv", "[K] KV placement", kvLabel)
	line("kvq", "[k] KV quality", kvQualityLabel)
	swaLabel := m.swaLabel(model)
	line("swa", "[w] Full SWA cache", swaLabel)

	section("Tuning")
	line("tuned", "[t] Tuned config", tuneLabel)
	line("aitune", "[a] AI tune", boolLabel(m.aitune))
	if m.aitune {
		line("rounds", "[r] AI tune rounds", strconv.Itoa(m.aituneRounds))
	}

	section("Run mode")
	line("vision", "[v] Vision (mmproj)", boolLabel(m.vision))
	ccLabel := boolLabel(m.claudeCode)
	if m.claudeCode {
		ccLabel += " — serve + print Claude Code env (thinking on)"
	}
	line("claudecode", "[x] Claude Code", ccLabel)
	if m.claudeCode {
		line("claudereviewer", "Reviewer/worker", claudeReviewerLabel(m.claudeReviewer))
		line("claudeprofile", "Claude profile", claudeProfileLabel(m.claudeProfile))
	}
	line("benchmark", "[b] Benchmark mode", boolLabel(m.benchmark))
	nocacheLabel := "off (reuse cached placement/probes)"
	if m.noCachedConfig {
		nocacheLabel = "on (derive fresh, ignore cached config)"
	}
	line("nocached", "[g] Launch without cached config", nocacheLabel)

	section("Actions")
	line("launch", "[L] Launch", "▶ start the server")
	line("dryrun", "[D] Dry run", "print the command, don't run")
	line("clearcaches", "[y] Clear caches", "drop cached placement/calibration for this model (keep GGUF)")

	b.WriteString("\n" + mutedStyle.Render("  Enter on Launch to start · Esc to go back"))

	if m.inputMode != "" {
		b.WriteString("\n\n  " + m.input.View())
	}
	if m.message != "" {
		b.WriteString("\n  ")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
	}

	return b.String()
}

func (m Model) updatePrelaunch(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			if m.replayRequest != nil {
				req := *m.replayRequest
				req.BackendArgs = append([]string(nil), m.replayRequest.BackendArgs...)
				m.launchRequest = &req
			} else {
				m.launchRequest = m.buildLaunchRequest()
			}
			return m, tea.Quit
		case "r", "R":
			// Resume only makes sense in Claude Code mode with a recorded
			// session; otherwise fall through to no-op rather than launching
			// something the footer did not offer.
			if m.claudeCode && m.resumeSession != "" {
				req := m.buildLaunchRequest()
				if req != nil {
					req.ResumeSession = m.resumeSession
					m.launchRequest = req
					return m, tea.Quit
				}
			}
			return m, nil
		case "esc":
			m.replayRequest = nil
			m.replaySavedAt = time.Time{}
			m.screen = ScreenModelConfig
			return m, nil
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

// loadResumableSession looks for a Claude Code session recorded from this
// working directory, so the pre-launch screen can offer to continue it instead
// of starting a conversation and workflow from zero.
func (m *Model) loadResumableSession() {
	m.resumeSession, m.resumeRun, m.resumeCached = "", "", 0
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	rec, err := claudesession.Latest(m.cacheDir, workDir)
	if err != nil {
		return
	}
	m.resumeSession = rec.SessionID
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		m.resumeRun, m.resumeCached = wf.RunID, cached
	}
}

// claudeProjectsDir mirrors the CLI's lookup of Claude Code's project state.
func claudeProjectsDir() string {
	if dir := os.Getenv("CLAUDE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "projects")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", "projects")
}

// shortSessionID keeps the pre-launch line readable; the full ID is only needed
// by the resume itself.
func shortSessionID(id string) string {
	if len(id) <= 13 {
		return id
	}
	return id[:13] + "…"
}

func (m Model) viewPrelaunch() string {
	if len(m.models) == 0 {
		return "No model selected"
	}
	model := m.models[m.selectedModel]
	var b strings.Builder
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ Pre-launch: %s ═══", model.Name)) + "\n\n")
	if m.replayRequest != nil {
		when := m.replaySavedAt.Local().Format("2006-01-02 15:04:05 MST")
		b.WriteString(highlightStyle.Render("  Latest saved TUI configuration") + "\n")
		b.WriteString(fmt.Sprintf("  Saved:          %s\n", when))
		b.WriteString("  Enter replays the exact saved request; Esc makes it editable.\n\n")
	}

	ctx := m.ctxSize
	if m.ctxMode == "fit" {
		ctx = "fit"
	}
	b.WriteString(fmt.Sprintf("  Context:        %s\n", ctx))
	b.WriteString(fmt.Sprintf("  Model path:     %s\n", model.Path))
	prelaunchBackend := m.effectiveBackend()
	if m.replayRequest != nil && m.replayRequest.Backend != "" {
		prelaunchBackend = m.replayRequest.Backend
	}
	b.WriteString(fmt.Sprintf("  Backend:        %s\n", prelaunchBackend))
	if m.port > 0 {
		b.WriteString(fmt.Sprintf("  Port:           %d\n", m.port))
	}
	if model.FitCtx > 0 {
		b.WriteString(fmt.Sprintf("  Fit estimate:   ~%d tokens\n", model.FitCtx))
	}
	if model.MaxCtx > 0 {
		b.WriteString(fmt.Sprintf("  Train max:      %d tokens\n", model.MaxCtx))
	}
	b.WriteString(fmt.Sprintf("  Parallel:       %s\n", m.prelaunchParallelLabel()))
	b.WriteString(fmt.Sprintf("  KV placement:   %s\n", m.kvPlacement))
	b.WriteString(fmt.Sprintf("  KV quality:     %s\n", m.kvQuality))
	b.WriteString(fmt.Sprintf("  Full SWA cache: %s\n", m.swaLabel(model)))
	b.WriteString(fmt.Sprintf("  AI tune:        %s\n", boolLabel(m.aitune)))
	if m.aitune {
		b.WriteString(fmt.Sprintf("  AI tune rounds: %d\n", m.aituneRounds))
	}
	b.WriteString(fmt.Sprintf("  Vision:         %s\n", boolLabel(m.vision)))
	b.WriteString(fmt.Sprintf("  Benchmark:      %s\n", boolLabel(m.benchmark)))
	b.WriteString(fmt.Sprintf("  Support expert: %s\n", supportExpertLabel(m.supportExpert)))
	b.WriteString(fmt.Sprintf("  Online research: %s\n", boolLabel(m.supportOnline)))
	b.WriteString(fmt.Sprintf("  Claude Code:    %s\n", boolLabel(m.claudeCode)))
	if m.claudeCode {
		b.WriteString(fmt.Sprintf("  Reviewer/worker: %s\n", claudeReviewerLabel(m.claudeReviewer)))
		b.WriteString(fmt.Sprintf("  Claude profile: %s\n", claudeProfileLabel(m.claudeProfile)))
	}
	if m.tunePath != "" {
		b.WriteString(fmt.Sprintf("  Tuned config:   %s\n", filepath.Base(m.tunePath)))
	}
	if m.claudeCode && m.resumeSession != "" {
		b.WriteString(fmt.Sprintf("  Resumable:      session %s\n", shortSessionID(m.resumeSession)))
		if m.resumeRun != "" {
			b.WriteString(fmt.Sprintf("                  workflow %s, %d agents cached\n", m.resumeRun, m.resumeCached))
		}
	}
	b.WriteString("\n")
	b.WriteString(highlightStyle.Render("  [Enter] Confirm and launch"))
	b.WriteString("\n")
	if m.claudeCode && m.resumeSession != "" {
		b.WriteString(highlightStyle.Render("  [r] Resume that session and its workflow"))
		b.WriteString("\n")
	}
	b.WriteString("  [Esc] Back to config\n")
	if m.message != "" {
		b.WriteString("\n  ")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func (m Model) updateTunedPicker(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			m.screen = ScreenModelConfig
			return m, nil
		case "enter":
			if m.tunedIndex >= 0 && m.tunedIndex < len(m.tunedConfigs) {
				m.tunePath = m.tunedConfigs[m.tunedIndex].Path
			} else {
				m.tunePath = ""
			}
			m.screen = ScreenModelConfig
			return m, nil
		case "up", "k":
			m.tunedIndex--
			if m.tunedIndex < -1 {
				m.tunedIndex = len(m.tunedConfigs) - 1
			}
		case "down", "j":
			m.tunedIndex++
			if m.tunedIndex >= len(m.tunedConfigs) {
				m.tunedIndex = -1
			}
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m Model) viewTunedPicker() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Tuned Configs ═══") + "\n\n")
	if len(m.tunedConfigs) == 0 {
		b.WriteString("  No tuned configs match this model/backend.\n")
		b.WriteString("  Run AI tune to create one.\n")
	} else {
		if m.tunedIndex == -1 {
			b.WriteString(selectedStyle.Render("▸ [0] Auto / heuristic cache selection") + "\n")
		} else {
			b.WriteString("  [0] Auto / heuristic cache selection\n")
		}
		for i, entry := range m.tunedConfigs {
			if i == m.tunedIndex {
				b.WriteString(selectedStyle.Render(fmt.Sprintf("▸ [%d] %s", i+1, entry.Label)) + "\n")
				b.WriteString(subtitleStyle.Render(fmt.Sprintf("     %s", filepath.Base(entry.Path))) + "\n")
			} else {
				b.WriteString(fmt.Sprintf("  [%d] %s\n", i+1, entry.Label))
				b.WriteString(subtitleStyle.Render(fmt.Sprintf("     %s", filepath.Base(entry.Path))) + "\n")
			}
		}
	}
	b.WriteString("\n")
	b.WriteString("  [Enter] Select  [Esc] Cancel  [↑/↓] Navigate\n")
	return b.String()
}

func (m Model) updateRecommended(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			if m.recHeadroomFocus != "" {
				m.recHeadroomFocus = ""
				m.message = ""
				return m, nil
			}
			if len(m.models) == 0 {
				m.screen = ScreenFirstRun
			} else {
				m.screen = ScreenMain
			}
			return m, nil
		case "up", "k":
			if len(m.recommendations) == 0 {
				return m, nil
			}
			m.selectedRecommendation--
			if m.selectedRecommendation < 0 {
				m.selectedRecommendation = len(m.recommendations) - 1
			}
		case "down", "j":
			if len(m.recommendations) == 0 {
				return m, nil
			}
			m.selectedRecommendation++
			if m.selectedRecommendation >= len(m.recommendations) {
				m.selectedRecommendation = 0
			}
		case "left", "h":
			if m.recHeadroomFocus != "" {
				m.stepRecommendedHeadroom(-1)
				return m, nil
			}
		case "right", "l":
			if m.recHeadroomFocus != "" {
				m.stepRecommendedHeadroom(1)
				return m, nil
			}
		case "enter":
			if m.recHeadroomFocus != "" {
				m.recHeadroomFocus = ""
				return m, nil
			}
			if len(m.recommendations) > 0 && m.selectedRecommendation >= 0 && m.selectedRecommendation < len(m.recommendations) {
				rec := m.recommendations[m.selectedRecommendation]
				return m.promptDownloadDir(&LaunchRequest{DownloadRepo: rec.Repo, DownloadQuant: rec.QuantName})
			}
		case "d", "D":
			m.screen = ScreenDownload
			m.inputMode = "download"
			m.input.SetValue("")
			m.input.Placeholder = "Hugging Face repo"
			m.input.Focus()
			return m, nil
		case "v", "V":
			m.focusRecommendedHeadroom("vram")
			return m, nil
		case "m", "M":
			m.focusRecommendedHeadroom("ram")
			return m, nil
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m *Model) focusRecommendedHeadroom(kind string) {
	m.recHeadroomFocus = kind
	m.message = "Use ←/→ to reserve memory for desktop, browser, IDE, games, or other GPU/CPU work."
	m.messageType = "info"
}

func (m *Model) stepRecommendedHeadroom(dir int) {
	if m.recHeadroomFocus == "vram" {
		m.setRecommendedHeadroom("vram", stepHeadroomMB(m.vramHeadroomMB, recommendedVRAMHeadroomSteps(m.caps, m.vramHeadroomMB), dir))
	} else if m.recHeadroomFocus == "ram" {
		m.setRecommendedHeadroom("ram", stepHeadroomMB(m.ramHeadroomMB, recommendedRAMHeadroomSteps(m.caps, m.ramHeadroomMB), dir))
	}
}

func (m *Model) setRecommendedHeadroom(kind string, mb int) {
	val := ""
	if mb > 0 {
		val = formatHeadroomMB(mb)
	}
	label := "VRAM reserve"
	if kind == "ram" {
		label = "RAM reserve"
	}
	if err := persistConfig(func(c *config.Config) {
		if kind == "vram" {
			c.VRAMHeadroom = val
		} else {
			c.RAMHeadroom = val
		}
	}); err != nil {
		m.message = fmt.Sprintf("Warning: %s set for this session — save failed: %v", label, err)
		m.messageType = "warning"
	} else {
		m.message = fmt.Sprintf("Saved: %s = %s", label, formatHeadroomMB(mb))
		m.messageType = "info"
	}
	if kind == "vram" {
		m.vramHeadroomMB = mb
	} else {
		m.ramHeadroomMB = mb
	}
	m.refreshRecommendations()
}

func stepHeadroomMB(current int, steps []int, dir int) int {
	if len(steps) == 0 {
		return current
	}
	if dir > 0 {
		for _, step := range steps {
			if step > current {
				return step
			}
		}
		return steps[len(steps)-1]
	}
	for i := len(steps) - 1; i >= 0; i-- {
		if steps[i] < current {
			return steps[i]
		}
	}
	return steps[0]
}

func recommendedVRAMHeadroomSteps(caps *detect.Capabilities, current int) []int {
	steps := []int{0, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 36864, 40960}
	max := 0
	if caps != nil {
		max = caps.TotalVRAM()
	}
	return smartHeadroomSteps(steps, current, max)
}

func recommendedRAMHeadroomSteps(caps *detect.Capabilities, current int) []int {
	steps := []int{0, 4096, 8192, 16384, 32768, 49152, 65536, 98304}
	max := 0
	if caps != nil {
		max = caps.RAM.TotalMB
	}
	return smartHeadroomSteps(steps, current, max)
}

func smartHeadroomSteps(base []int, current, max int) []int {
	steps := append([]int(nil), base...)
	if current > 0 {
		steps = append(steps, current)
	}
	if max > 0 {
		// Leave at least a little memory visible to the recommender; reserving
		// everything is not useful as a preset.
		limit := max - min(2048, max/4)
		filtered := steps[:0]
		for _, step := range steps {
			if step <= limit {
				filtered = append(filtered, step)
			}
		}
		steps = filtered
	}
	sort.Ints(steps)
	uniq := steps[:0]
	last := -1
	for _, step := range steps {
		if step != last {
			uniq = append(uniq, step)
			last = step
		}
	}
	return uniq
}

func formatHeadroomMB(mb int) string {
	if mb <= 0 {
		return "0"
	}
	if mb%1024 == 0 {
		return fmt.Sprintf("%dG", mb/1024)
	}
	return fmt.Sprintf("%dM", mb)
}

func (m *Model) refreshRecommendations() {
	caps := detect.ApplyRAMLimitPercent(m.caps, m.ramLimitPercent)
	caps = detect.ApplyVRAMHeadroom(caps, m.vramHeadroomMB)
	caps = detect.ApplyRAMHeadroom(caps, m.ramHeadroomMB)
	m.recommendationGroups = recommend.TopCategories(caps, 4)
	m.recommendations = flattenRecommendationCategories(m.recommendationGroups)
	if len(m.recommendations) == 0 {
		m.selectedRecommendation = 0
		return
	}
	if m.selectedRecommendation < 0 {
		m.selectedRecommendation = 0
	}
	if m.selectedRecommendation >= len(m.recommendations) {
		m.selectedRecommendation = len(m.recommendations) - 1
	}
}

// wordWrap greedily packs words from s into lines no wider than width,
// breaking only at spaces. width <= 0 falls back to 78 columns.
func wordWrap(s string, width int) []string {
	if width <= 0 {
		width = 78
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	lines := []string{words[0]}
	for _, w := range words[1:] {
		last := lines[len(lines)-1]
		if len(last)+1+len(w) > width {
			lines = append(lines, w)
		} else {
			lines[len(lines)-1] = last + " " + w
		}
	}
	return lines
}

func (m Model) viewRecommended() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Recommended Downloads ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString("  " + m.recommendedHeadroomControls() + "\n")
	for _, line := range wordWrap(recommend.CatalogAttribution(), m.width) {
		b.WriteString("  " + line + "\n")
	}
	b.WriteString("\n")

	if len(m.recommendations) == 0 {
		b.WriteString(warningStyle.Render("  No safe recommendation fits the detected RAM/VRAM."))
		b.WriteString("\n  [v] VRAM reserve  [m] RAM reserve  [d] Manual Hugging Face repository  [Esc] Back\n")
		if m.recHeadroomFocus != "" {
			b.WriteString("  ←/→ smart reserve steps · Enter/Esc done\n")
		}
		if m.message != "" {
			b.WriteString("\n  ")
			switch m.messageType {
			case "error":
				b.WriteString(errorStyle.Render(m.message))
			case "warning":
				b.WriteString(warningStyle.Render(m.message))
			default:
				b.WriteString(highlightStyle.Render(m.message))
			}
			b.WriteString("\n")
		}
		return b.String()
	}

	idx := 0
	writeGroup := func(title string, rows []recommend.Recommendation) {
		if len(rows) == 0 {
			return
		}
		b.WriteString(recommendStyle.Render("  "+title) + "\n")
		for _, rec := range rows {
			prefix := "  "
			if idx == m.selectedRecommendation {
				prefix = selectedStyle.Render("▸ ")
			}
			quant := rec.QuantName
			if quant == "" {
				quant = "auto"
			}
			tps := "—"
			if rec.PredictedTPS > 0 {
				tps = fmt.Sprintf("~%.0f t/s", rec.PredictedTPS)
			}
			name := rec.Name
			if len(name) > 34 {
				name = name[:33] + "…"
			}
			line := fmt.Sprintf("%-34s %-9s %-11s %5.1fG %3.0f%% %7s",
				name, recommend.DisplayFit(rec.Fit), quant, rec.QuantSizeGB, rec.QualityRetained*100, tps)
			if idx == m.selectedRecommendation {
				b.WriteString(prefix + selectedStyle.Render(line) + "\n")
			} else {
				b.WriteString(prefix + line + "\n")
			}
			idx++
		}
		b.WriteString("\n")
	}
	writeGroup("Best overall — balanced quality, speed and fit", m.recommendationGroups.Balanced)
	writeGroup("Smartest — highest intelligence that fits", m.recommendationGroups.Smartest)
	writeGroup("Fastest — quickest while still capable", m.recommendationGroups.Fastest)
	b.WriteString(mutedStyle.Render("  Speeds are estimates; Benchmark measures this exact machine.") + "\n")
	b.WriteString(mutedStyle.Render("  Fit uses installed capacity; launch rechecks memory currently free.") + "\n\n")

	b.WriteString(highlightStyle.Render("  [Enter] Download selected"))
	b.WriteString("\n  [v] VRAM reserve  [m] RAM reserve  [d] Manual repo  [Esc] Back  [↑/↓] Navigate\n")
	if m.recHeadroomFocus != "" {
		b.WriteString("  ←/→ smart reserve steps · Enter/Esc done\n")
	}
	if m.message != "" {
		b.WriteString("\n  ")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func (m Model) recommendedHeadroomControls() string {
	vram := fmt.Sprintf("[v] VRAM %s", formatHeadroomMB(m.vramHeadroomMB))
	ram := fmt.Sprintf("[m] RAM %s", formatHeadroomMB(m.ramHeadroomMB))
	if m.recHeadroomFocus == "vram" {
		vram = selectedStyle.Render("▸ " + vram + " ◂")
	}
	if m.recHeadroomFocus == "ram" {
		ram = selectedStyle.Render("▸ " + ram + " ◂")
	}
	return fmt.Sprintf("Reserve for other apps: %s   %s", vram, ram)
}

func (m Model) viewInputScreen() string {
	var b strings.Builder
	var title string
	switch m.inputMode {
	case "download":
		title = "Download Model"
	case "downloaddir":
		title = "Download Destination"
	case "modeldir":
		title = "Model Directory"
	case "backend-add":
		title = "Add llama.cpp Fork"
	case "backend-register":
		title = "Register Built Backend"
	default:
		title = "Input"
	}
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", title)) + "\n\n")
	b.WriteString(m.input.View())
	b.WriteString("\n\n  Press Enter to confirm, Esc to cancel")
	if m.message != "" {
		b.WriteString("\n\n  ")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
	}
	return b.String()
}

func hwSummary(caps *detect.Capabilities) string {
	if caps == nil {
		return "detecting..."
	}
	ramGB := caps.RAM.TotalMB / 1024
	if len(caps.GPUs) == 0 {
		return fmt.Sprintf("%dGB RAM, %d cores (no GPU)", ramGB, caps.CPU.Cores)
	}
	parts := make([]string, 0, len(caps.GPUs))
	for _, g := range caps.GPUs {
		name := strings.TrimPrefix(g.Name, "NVIDIA GeForce ")
		parts = append(parts, fmt.Sprintf("%s %.0fG", name, float64(g.VRAMTotalMB)/1024))
	}
	return fmt.Sprintf("%s · %dGB RAM · %d cores", strings.Join(parts, " + "), ramGB, caps.CPU.Cores)
}

func boolLabel(v bool) string {
	if v {
		return "on"
	}
	return "off"
}

func isAuxiliaryModel(name, arch string) bool {
	if strings.EqualFold(strings.TrimSpace(arch), "dflash") {
		return true
	}
	for _, token := range strings.FieldsFunc(strings.ToLower(filepath.Base(name)), func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	}) {
		switch token {
		case "dflash", "draft", "mtp", "speculator":
			return true
		}
	}
	return false
}

func discoverModels(dir string) []ModelItem {
	var items []ModelItem
	seen := make(map[string]bool)

	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		item, key, ok := modelItemFromPath(path, info, false)
		if !ok || seen[key] {
			return nil
		}
		seen[key] = true
		items = append(items, item)
		return nil
	})

	return items
}

func modelItemFromPath(path string, info os.FileInfo, external bool) (ModelItem, string, bool) {
	if info == nil || info.IsDir() {
		return ModelItem{}, "", false
	}
	name := info.Name()
	lower := strings.ToLower(name)
	if !strings.HasSuffix(lower, ".gguf") || strings.Contains(lower, "mmproj") || isAuxiliaryModel(name, "") {
		return ModelItem{}, "", false
	}
	shardFiles, isMultiPart, shardErr := modelstore.ResolveGGUFShardFiles(path)
	if shardErr != nil {
		return ModelItem{}, "", false
	}

	baseName := name
	if shard := strings.Index(lower, "-00001-of-"); shard > 0 {
		baseName = name[:shard] + ".gguf"
	} else if strings.Contains(lower, "-of-") {
		return ModelItem{}, "", false
	}

	dirPath := filepath.Dir(path)
	modelKey := filepath.Join(dirPath, baseName)
	totalBytes := info.Size()
	if isMultiPart {
		totalBytes = 0
		for _, shardPath := range shardFiles {
			if st, err := os.Stat(shardPath); err == nil {
				totalBytes += st.Size()
			}
		}
	}

	arch := "dense"
	if strings.Contains(name, "A") && strings.Contains(name, "B") {
		// Check A[0-9]+B pattern for a cheap pre-header MoE hint.
		for i := 0; i < len(name)-1; i++ {
			if name[i] != 'A' && name[i] != 'a' {
				continue
			}
			j := i + 1
			for j < len(name) && name[j] >= '0' && name[j] <= '9' {
				j++
			}
			if j < len(name) && (name[j] == 'B' || name[j] == 'b') {
				arch = "MoE"
				break
			}
		}
	}
	return ModelItem{
		Name:     baseName,
		Path:     filepath.Clean(path),
		SizeGB:   float64(totalBytes) / (1024 * 1024 * 1024),
		Arch:     arch,
		External: external,
	}, modelKey, true
}

func discoverModelsFromPaths(paths []string) []ModelItem {
	items := make([]ModelItem, 0, len(paths))
	seen := make(map[string]bool)
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			continue
		}
		item, key, ok := modelItemFromPath(path, info, true)
		if !ok || seen[key] {
			continue
		}
		seen[key] = true
		items = append(items, item)
	}
	return items
}

func modelIdentity(path string) string {
	path = filepath.Clean(path)
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return path
}

// sortModels orders the list for display. When usage records are available the
// most-used and most-recently-used models float to the top; everything still
// falls back to a stable name sort so the list stays deterministic.
func sortModels(items []ModelItem, usage map[string]modelusage.Record) {
	sort.SliceStable(items, func(i, j int) bool {
		var left, right modelusage.Record
		if usage != nil {
			left = usage[modelIdentity(items[i].Path)]
			right = usage[modelIdentity(items[j].Path)]
		}
		if left.Launches != right.Launches {
			return left.Launches > right.Launches
		}
		if !left.LastUsedAt.Equal(right.LastUsedAt) {
			return left.LastUsedAt.After(right.LastUsedAt)
		}
		lname, rname := strings.ToLower(items[i].Name), strings.ToLower(items[j].Name)
		if lname != rname {
			return lname < rname
		}
		return strings.ToLower(items[i].Path) < strings.ToLower(items[j].Path)
	})
}

func mergeModelItems(groups ...[]ModelItem) []ModelItem {
	seen := make(map[string]bool)
	var merged []ModelItem
	for _, group := range groups {
		for _, item := range group {
			key := modelIdentity(item.Path)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			merged = append(merged, item)
		}
	}
	sortModels(merged, nil)
	return merged
}

func enrichModelItems(models []ModelItem, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	visible := models[:0]
	totalSysMemMB := 0
	if caps != nil {
		totalSysMemMB = caps.TotalVRAM() + caps.RAM.TotalMB
	}
	backendTag := strings.TrimSpace(backend)
	switch backendTag {
	case "", "auto", "llama":
		backendTag = "llama"
	case "ik_llama":
		backendTag = "ik"
	}
	for i := range models {
		modelBackendTag := backendTag
		if info, err := gguf.Parse(models[i].Path); err == nil {
			if isAuxiliaryModel(models[i].Name, info.Architecture) {
				continue
			}
			models[i].MaxCtx = info.ContextLength
			models[i].IsMoE = info.IsMoE
			models[i].Architecture = info.Architecture
			models[i].KVProfile = kvProfileFromGGUF(info)
			if info.Architecture != "" {
				models[i].Arch = info.Architecture
			}
			if info.IsMoE {
				models[i].Arch += " · MoE"
			}
			if info.Architecture != "" {
				// Architecture routes are per-model and therefore more specific than
				// the configured default backend. An installed reviewed/custom fork
				// is selected for this model even when the user's default is pinned to
				// mainline or generic ik_llama.
				if routed := backends.ForArch(info.Architecture); routed != nil {
					modelBackendTag = routed.Tag
					models[i].AutoBackend = routed.Tag
				} else if recipes := backends.RecipesForArch(info.Architecture); len(recipes) > 0 {
					// Catalog order is preference order. For architectures with an
					// alternative recipe (Laguna's upstream target-only fork), the first
					// entry is the reviewed full-featured default.
					models[i].BackendRecipe = recipes[0].Name
				}
			}
			models[i].FitCtx = probe.EstimateFitCtxForInfo(models[i].Path, cacheDir, info, totalSysMemMB)
		}
		models[i].Tuned = tune.CountTunedConfigs(cacheDir, models[i].Name, modelBackendTag)
		visible = append(visible, models[i])
	}
	return visible
}

func kvProfileFromGGUF(info *gguf.Info) *placement.ModelProfile {
	if info == nil {
		return nil
	}
	return &placement.ModelProfile{
		NumLayers:        info.BlockCount,
		HeadCountKV:      info.HeadCountKV,
		KeyLength:        info.KeyLength,
		ValueLength:      info.ValueLength,
		KVLoraRank:       info.KVLoraRank,
		RopeDim:          info.NRot,
		HasSSM:           info.SSM,
		FullAttnInterval: info.FullAttnInterval,
		SlidingWindow:    info.SlidingWindow,
		ModelArch:        info.Architecture,
		// The KV rate/geometry cache (kvCachePath) is keyed on the model's exact
		// byte size. Without it, the profile built here produces kv_<basename>_0
		// which never matches the kv_<basename>_<actualsize> written at launch,
		// so the TUI "Clear caches" action silently leaves the stale KV rate.
		SizeBytes: info.NonExpertBytes + info.ExpertBytes,
	}
}

func (m Model) swaEstimateContext(model ModelItem) int {
	value := strings.TrimSpace(m.ctxSize)
	if n, err := strconv.Atoi(value); err == nil && n > 0 {
		return n
	}
	if value == "max" || value == "native" {
		return model.MaxCtx
	}
	if m.claudeCode {
		ctx := model.MaxCtx
		if ctx > 1048576 {
			ctx = 1048576
		}
		if ctx <= 0 {
			ctx = 131072
		}
		return ctx
	}
	return model.FitCtx
}

func (m Model) swaExtraKVMB(model ModelItem) int {
	if model.KVProfile == nil {
		return -1
	}
	ctx := m.swaEstimateContext(model)
	if ctx <= 0 {
		return -1
	}
	kvType, err := placement.NormalizeKVType(m.kvQuality)
	if err != nil {
		return -1
	}
	plain := placement.EstimateKVCacheMB(model.KVProfile, ctx, kvType, false)
	full := placement.EstimateKVCacheMB(model.KVProfile, ctx, kvType, true)
	if full < plain {
		return 0
	}
	return full - plain
}

func (m Model) swaLabel(model ModelItem) string {
	extraMB := m.swaExtraKVMB(model)
	if extraMB < 0 {
		if m.swaFull {
			return "on (more cache hits; memory estimate after dry run)"
		}
		return "off (smaller KV cache)"
	}
	if extraMB == 0 {
		if m.swaFull {
			return "on (no extra KV for this model)"
		}
		return "off (this model has no priced SWA delta)"
	}
	delta := fmt.Sprintf("+%.1f GiB KV", float64(extraMB)/1024.0)
	if m.swaFull {
		return "on (more cache hits; " + delta + ")"
	}
	return "off (enable cache hits; " + delta + ")"
}

func loadModels(dir, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	return enrichModelItems(discoverModels(dir), cacheDir, backend, caps)
}

func loadRecognizedModels(dir, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	items := mergeModelItems(
		discoverModels(dir),
		discoverModelsFromPaths(modelstore.LoadDiscoveredPaths(cacheDir)),
	)
	items = enrichModelItems(items, cacheDir, backend, caps)
	// Re-apply the usage sort after enrichment: display order must reflect real
	// launch history, not discovery order.
	sortModels(items, modelusage.Load(cacheDir))
	return items
}

func (m Model) backendTag() string {
	backend := strings.TrimSpace(m.backend)
	switch backend {
	case "ik_llama":
		return "ik"
	case "":
		return "llama"
	case "auto":
		return "llama"
	default:
		return backend
	}
}

// persistConfig loads the current config, applies mutate, and writes it back to
// the canonical config file, preserving all other settings so GUI changes
// survive across sessions.
func persistConfig(mutate func(*config.Config)) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	mutate(cfg)
	return cfg.Save()
}

// refreshTunedCounts recomputes per-model tuned config counts for the current
// backend and rebuilds the main list so the counts reflect the active backend.
func (m *Model) refreshTunedCounts() {
	tag := m.backendTag()
	for i := range m.models {
		modelTag := tag
		if m.models[i].AutoBackend != "" {
			modelTag = m.models[i].AutoBackend
		}
		m.models[i].Tuned = tune.CountTunedConfigs(m.cacheDir, m.models[i].Name, modelTag)
	}
	m.rebuildMainList()
}

// settingRow describes one editable config setting on the Settings screen.
// kind is "enum" (pick from options), "bool" (toggle), or "text" (free input).
type settingRow struct {
	label   string
	kind    string
	options []string
	get     func(*config.Config) string
	set     func(*config.Config, string)
}

// settingRows returns every setting shown on the Settings screen, in order.
func settingRows() []settingRow {
	atoiSet := func(assign func(*config.Config, int)) func(*config.Config, string) {
		return func(c *config.Config, v string) {
			if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
				assign(c, n)
			}
		}
	}
	return []settingRow{
		{label: "Backend", kind: "enum", options: backendOptions(),
			get: func(c *config.Config) string { return c.Backend },
			set: func(c *config.Config, v string) { c.Backend = v }},
		{label: "Model directory", kind: "text",
			get: func(c *config.Config) string { return c.ModelDir },
			set: func(c *config.Config, v string) { c.ModelDir = v }},
		{label: "Context", kind: "enum", options: []string{"fit", "max"},
			get: func(c *config.Config) string { return c.CtxValue() },
			set: func(c *config.Config, v string) {
				_ = c.SetCtxValue(v)
			}},
		{label: "KV placement", kind: "enum", options: []string{"auto", "gpu", "cpu"},
			get: func(c *config.Config) string { return c.KVPlacement },
			set: func(c *config.Config, v string) { c.KVPlacement = v }},
		{label: "KV quality", kind: "enum", options: []string{"auto", "high", "bf16", "mid", "q8_0", "q5_1", "q5_0", "q4_1", "low", "q4_0", "iq4_nl", "f32"},
			get: func(c *config.Config) string { return c.KVQuality },
			set: func(c *config.Config, v string) { c.KVQuality = v }},
		{label: "Full SWA cache", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.SWAFull) },
			set: func(c *config.Config, v string) { c.SWAFull = v == "on" }},
		{label: "Support expert / optimizer", kind: "enum", options: []string{"auto", "on", "off"},
			get: func(c *config.Config) string { return c.SupportExpert },
			set: func(c *config.Config, v string) { c.SupportExpert = v }},
		{label: "Support online research", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.SupportOnline) },
			set: func(c *config.Config, v string) { c.SupportOnline = v == "on" }},
		{label: "VRAM headroom", kind: "text",
			get: func(c *config.Config) string {
				if strings.TrimSpace(c.VRAMHeadroom) == "" {
					return "0"
				}
				return c.VRAMHeadroom
			},
			set: func(c *config.Config, v string) { c.VRAMHeadroom = strings.TrimSpace(v) }},
		{label: "RAM headroom", kind: "text",
			get: func(c *config.Config) string {
				if strings.TrimSpace(c.RAMHeadroom) == "" {
					return "0"
				}
				return c.RAMHeadroom
			},
			set: func(c *config.Config, v string) { c.RAMHeadroom = strings.TrimSpace(v) }},
		{label: "RAM limit percent", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.RAMLimitPercent) },
			set: atoiSet(func(c *config.Config, n int) { c.RAMLimitPercent = n })},
		{label: "Speculative", kind: "enum",
			options: []string{"off", "auto", "draft", "eagle3", "ngram", "ngram-mod", "ngram-k4v", "mtp"},
			get:     func(c *config.Config) string { return c.Spec },
			set:     func(c *config.Config, v string) { c.Spec = v }},
		{label: "Vision", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.Vision) },
			set: func(c *config.Config, v string) { c.Vision = v == "on" }},
		{label: "Port", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.Port) },
			set: atoiSet(func(c *config.Config, n int) { c.Port = n })},
		{label: "Host", kind: "text",
			get: func(c *config.Config) string { return c.Host },
			set: func(c *config.Config, v string) { c.Host = strings.TrimSpace(v) }},
		{label: "Parallel", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.Parallel) },
			set: atoiSet(func(c *config.Config, n int) { c.Parallel = n })},
		{label: "AI-tune rounds", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.TuneRounds) },
			set: atoiSet(func(c *config.Config, n int) { c.TuneRounds = n })},
	}
}

// applySetting mutates the in-memory config, persists it to disk, and applies
// any side effects (re-scan models, refresh tuned counts) for the given row.
func (m *Model) applySetting(row settingRow, val string) {
	if err := validateSettingValue(row.label, val); err != nil {
		m.message = fmt.Sprintf("Warning: %s was not changed: %v", row.label, err)
		m.messageType = "warning"
		return
	}
	row.set(m.settingsCfg, val)
	if err := m.settingsCfg.Save(); err != nil {
		m.message = fmt.Sprintf("Warning: %s set to %s for this session — save failed: %v", row.label, val, err)
		m.messageType = "warning"
	} else {
		m.message = fmt.Sprintf("Saved: %s = %s", row.label, val)
		m.messageType = "info"
	}
	switch row.label {
	case "Context":
		m.setCtx(m.settingsCfg.CtxValue())
	case "KV placement":
		m.kvPlacement = val
	case "KV quality":
		// Sync the live session too — otherwise the saved value only applies
		// after a TUI restart while the current session keeps launching with
		// the startup-time quality.
		m.kvQuality = val
	case "Full SWA cache":
		m.swaFull = m.settingsCfg.SWAFull
	case "Support expert / optimizer":
		m.supportExpert = m.settingsCfg.SupportExpert
	case "Support online research":
		m.supportOnline = m.settingsCfg.SupportOnline
	case "VRAM headroom":
		m.vramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "RAM headroom":
		m.ramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "RAM limit percent":
		m.ramLimitPercent = m.settingsCfg.RAMLimitPercent
		m.refreshRecommendations()
	case "Backend":
		m.backend = val
		m.refreshTunedCounts()
	case "Model directory":
		m.modelDir = val
		m.models = loadRecognizedModels(val, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		if m.messageType != "warning" {
			m.message = fmt.Sprintf("Saved: Model directory = %s (%d models)", val, len(m.models))
		}
	case "Vision":
		m.vision = m.settingsCfg.Vision
	case "Port":
		m.port = m.settingsCfg.Port
	case "Parallel":
		m.parallel = ""
		m.parallelSet = m.settingsCfg.Parallel > 0
		if m.settingsCfg.Parallel > 0 {
			m.parallel = strconv.Itoa(m.settingsCfg.Parallel)
		}
	case "AI-tune rounds":
		m.aituneRounds = m.settingsCfg.TuneRounds
	}
}

func validateSettingValue(label, val string) error {
	switch label {
	case "Port":
		_, err := config.ParsePort(val)
		return err
	case "Parallel", "AI-tune rounds":
		n, err := strconv.Atoi(strings.TrimSpace(val))
		if err != nil || n < 0 {
			return fmt.Errorf("must be a non-negative integer")
		}
	case "VRAM headroom", "RAM headroom":
		_, err := config.ParseBudgetMBStrict(val)
		return err
	case "RAM limit percent":
		_, err := config.ParseRAMLimitPercent(val)
		return err
	}
	return nil
}

// openChoice configures and shows the generic arrow-select screen.
func (m *Model) openChoice(title string, options []string, current string, ret Screen, apply func(*Model, string)) {
	m.choiceTitle = title
	m.choiceOptions = options
	m.choiceCursor = indexOf(options, current)
	if m.choiceCursor < 0 {
		m.choiceCursor = 0
	}
	m.choiceApply = apply
	m.choiceReturn = ret
	m.screen = ScreenChoice
}

// openSettings loads the current config and shows the Settings screen.
func (m *Model) openSettings() {
	cfg, err := config.Load()
	if err != nil || cfg == nil {
		cfg = config.Defaults()
	}
	m.settingsCfg = cfg
	m.settingsCursor = 0
	m.inputMode = ""
	m.screen = ScreenSettings
}

// backendOptions lists the selectable backends: the built-ins plus any
// registered fork backends (so they show up in the TUI backend picker).
func backendOptions() []string {
	opts := []string{"auto", "llama", "ik_llama"}
	opts = append(opts, backends.Tags()...)
	return opts
}

// openBackendChoice shows the arrow-select backend chooser, persisting the
// choice and returning to ret afterwards.
func (m *Model) openBackendChoice(ret Screen) {
	m.openChoice("Backend", backendOptions(), m.backend, ret, func(mm *Model, v string) {
		mm.backend = v
		if err := persistConfig(func(c *config.Config) { c.Backend = v }); err != nil {
			mm.message = fmt.Sprintf("Warning: Backend set to %s for this session — save failed: %v", v, err)
			mm.messageType = "warning"
		} else {
			mm.message = "Backend saved: " + v
			mm.messageType = "info"
		}
		mm.refreshTunedCounts()
	})
}

func backendManagerOptions() []string {
	opts := make([]string, 0, len(backends.Recipes())+2+len(backends.Load()))
	for _, recipe := range backends.Recipes() {
		opts = append(opts, "Install reviewed: "+recipe.Name)
	}
	opts = append(opts, "Add custom fork from Git URL", "Register an existing llama-server binary")
	for _, backend := range backends.Load() {
		opts = append(opts, "Remove installed: "+backend.Tag)
	}
	return opts
}

func (m *Model) openBackendManager(ret Screen) {
	m.openChoice("Backend forks", backendManagerOptions(), "", ret, func(mm *Model, value string) {
		switch {
		case strings.HasPrefix(value, "Install reviewed: "):
			name := strings.TrimSpace(strings.TrimPrefix(value, "Install reviewed: "))
			mm.launchRequest = &LaunchRequest{BackendArgs: []string{"install", name}}
		case value == "Add custom fork from Git URL":
			mm.screen = ScreenBackend
			mm.inputMode = "backend-add"
			mm.input.SetValue("")
			mm.input.Placeholder = "Git URL [tag] [model architecture]"
			mm.input.Focus()
		case value == "Register an existing llama-server binary":
			mm.screen = ScreenBackend
			mm.inputMode = "backend-register"
			mm.input.SetValue("")
			mm.input.Placeholder = "tag /path/to/llama-server [model architecture]"
			mm.input.Focus()
		case strings.HasPrefix(value, "Remove installed: "):
			tag := strings.TrimSpace(strings.TrimPrefix(value, "Remove installed: "))
			mm.openChoice("Remove "+tag+"?", []string{"Cancel", "Confirm: remove " + tag}, "Cancel", ret, func(cm *Model, confirm string) {
				if confirm == "Cancel" {
					return
				}
				cm.launchRequest = &LaunchRequest{BackendArgs: []string{"remove", tag}}
			})
		}
	})
}

// openRemoveModelChoice confirms before deleting the selected model's GGUF
// file(s) from disk, reusing the same confirm-first choice screen as backend
// fork removal (default cursor on Cancel).
func (m *Model) openRemoveModelChoice(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	if m.models[idx].External {
		m.message = "Discovered models outside the primary directory are launch-only; remove the file at its source or make that directory primary first."
		m.messageType = "warning"
		return
	}
	name := m.models[idx].Name
	m.openChoice("Delete "+name+"?", []string{"Cancel", "Confirm: delete " + name}, "Cancel", ScreenMain, func(cm *Model, confirm string) {
		if confirm == "Cancel" {
			return
		}
		cm.removeModelAt(idx)
	})
}

// removeModelAt deletes a model's GGUF file(s) from disk — the same removal
// path as `ggrun models rm` — then refreshes the Main list. It matches by
// file path rather than display name: two models in different directories
// can share the same basename (see
// TestDiscoverModelsKeepsSameBasenameInDifferentDirectories), so the display
// name alone cannot be trusted to pick the right one.
func (m *Model) removeModelAt(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	item := m.models[idx]
	if item.External {
		m.message = "Discovered models outside the primary directory are launch-only; remove the file at its source or make that directory primary first."
		m.messageType = "warning"
		return
	}
	rel, err := filepath.Rel(m.modelDir, item.Path)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	rel = filepath.Clean(rel)
	all, err := modelstore.List(m.modelDir)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	var name string
matched:
	for _, candidate := range all {
		for _, f := range candidate.Files {
			if f == rel {
				name = candidate.Name
				break matched
			}
		}
	}
	if name == "" {
		m.message = fmt.Sprintf("Model not found on disk: %s", item.Name)
		m.messageType = "error"
		return
	}
	removed, err := modelstore.Remove(m.modelDir, name)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)
	m.rebuildMainList()
	m.message = fmt.Sprintf("Removed %s (%.1fGB freed).", removed.Name, float64(removed.Bytes)/(1024*1024*1024))
	m.messageType = "info"
}

// openClearCachesChoice confirms before removing the selected model's cached
// measurements (probe caches, KV rate/geometry, calibration decisions,
// placement caches) while keeping the GGUF itself. Default cursor on Cancel.
func (m *Model) openClearCachesChoice(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	name := m.models[idx].Name
	m.openChoice("Clear caches for "+name+"?", []string{"Cancel", "Confirm: clear caches for " + name}, "Cancel", ScreenModelConfig, func(cm *Model, confirm string) {
		if confirm == "Cancel" {
			return
		}
		cm.clearModelCachesAt(idx)
	})
}

// clearModelCachesAt removes the selected model's cached configs via
// placement.ClearModelCaches (probe/KV/calibration/placement caches), keeps the
// GGUF, and reports how many files were removed. Discovery can be stale (the
// model may no longer be on disk), so build the model profile from the path
// with whatever metadata the scanner already parsed; ClearModelCaches only
// needs the basename identity to match cache headers.
func (m *Model) clearModelCachesAt(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	item := m.models[idx]
	profile := &placement.ModelProfile{Path: item.Path, Basename: filepath.Base(item.Path)}
	if item.KVProfile != nil {
		copy := *item.KVProfile
		copy.Path = item.Path
		if copy.Basename == "" {
			copy.Basename = filepath.Base(item.Path)
		}
		profile = &copy
	}
	removed, err := placement.ClearModelCaches(m.cacheDir, profile)
	if err != nil {
		m.message = fmt.Sprintf("Error clearing caches for %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	m.message = fmt.Sprintf("Cleared %d cached config(s) for %s. Next launch re-measures placement.", removed, item.Name)
	m.messageType = "info"
}

func indexOf(opts []string, v string) int {
	for i, o := range opts {
		if o == v {
			return i
		}
	}
	return -1
}

func prevOption(opts []string, v string) string {
	i := indexOf(opts, v)
	if i <= 0 {
		return opts[len(opts)-1]
	}
	return opts[i-1]
}

func nextOption(opts []string, v string) string {
	i := indexOf(opts, v)
	if i < 0 || i >= len(opts)-1 {
		return opts[0]
	}
	return opts[i+1]
}

func (m Model) updateChoice(msg tea.Msg) (tea.Model, tea.Cmd) {
	if keyMsg, ok := msg.(tea.KeyMsg); ok {
		switch keyMsg.String() {
		case "up", "k":
			if m.choiceCursor > 0 {
				m.choiceCursor--
			}
		case "down", "j":
			if m.choiceCursor < len(m.choiceOptions)-1 {
				m.choiceCursor++
			}
		case "enter", " ":
			if m.choiceApply != nil && m.choiceCursor < len(m.choiceOptions) {
				m.screen = m.choiceReturn
				m.choiceApply(&m, m.choiceOptions[m.choiceCursor])
			}
			if m.launchRequest != nil && len(m.launchRequest.BackendArgs) > 0 {
				return m, tea.Quit
			}
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m Model) updateSettings(msg tea.Msg) (tea.Model, tea.Cmd) {
	rows := settingRows()

	// Free-text edit mode for "text" settings.
	if m.inputMode == "setting" {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			val := strings.TrimSpace(m.input.Value())
			if val != "" && m.settingsCursor < len(rows) {
				m.applySetting(rows[m.settingsCursor], val)
			}
			m.inputMode = ""
		}
		return m, cmd
	}

	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	row := rows[m.settingsCursor]
	switch keyMsg.String() {
	case "up", "k":
		if m.settingsCursor > 0 {
			m.settingsCursor--
		}
	case "down", "j":
		if m.settingsCursor < len(rows)-1 {
			m.settingsCursor++
		}
	case "enter":
		switch row.kind {
		case "enum":
			m.openChoice(row.label, row.options, row.get(m.settingsCfg), ScreenSettings,
				func(mm *Model, v string) { mm.applySetting(row, v) })
		case "bool":
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		case "text":
			m.inputMode = "setting"
			m.input.SetValue(row.get(m.settingsCfg))
			m.input.Placeholder = row.label
			m.input.Focus()
		}
	case "right", "l":
		if row.kind == "enum" {
			m.applySetting(row, nextOption(row.options, row.get(m.settingsCfg)))
		} else if row.kind == "bool" {
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		}
	case "left", "h":
		if row.kind == "enum" {
			m.applySetting(row, prevOption(row.options, row.get(m.settingsCfg)))
		} else if row.kind == "bool" {
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		}
	case "e", "E":
		editor := os.Getenv("EDITOR")
		if editor == "" {
			editor = "nano"
		}
		c := exec.Command(editor, m.settingsPath)
		c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
		c.Run()
		if cfg, err := config.Load(); err == nil {
			m.settingsCfg = cfg
		}
	case "q", "Q":
		return m, tea.Quit
	}
	return m, nil
}

func toggleBool(cur string) string {
	if cur == "on" {
		return "off"
	}
	return "on"
}

func (m Model) viewChoice() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", m.choiceTitle)) + "\n\n")
	for i, opt := range m.choiceOptions {
		if i == m.choiceCursor {
			b.WriteString("  " + selectedStyle.Render("> "+opt) + "\n")
		} else {
			b.WriteString("    " + opt + "\n")
		}
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ select · Enter confirm · Esc cancel"))
	return b.String()
}

func (m Model) viewSettings() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Settings ═══") + "\n\n")
	rows := settingRows()
	for i, row := range rows {
		line := fmt.Sprintf("%-17s %s", row.label+":", row.get(m.settingsCfg))
		if i == m.settingsCursor {
			b.WriteString("  " + selectedStyle.Render("> "+line) + "\n")
		} else {
			b.WriteString("    " + line + "\n")
		}
	}
	if m.inputMode == "setting" {
		b.WriteString("\n  " + m.input.View() + "\n")
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ navigate · Enter/→ change · ←/→ cycle enums · [e] edit file · Esc back"))
	b.WriteString("\n" + mutedStyle.Render("  config: "+m.settingsPath))
	if m.message != "" {
		b.WriteString("\n  ")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
	}
	return b.String()
}

func (m Model) buildLaunchRequest() *LaunchRequest {
	if len(m.models) == 0 || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return nil
	}
	model := m.models[m.selectedModel]
	// Default: 0 = auto-fit (Compute() finds max context that fits hardware)
	// Default to auto-fit (ctx=0): placement.Compute finds the max context that
	// actually fits — the same path the CLI uses. The old default fed the crude
	// computeRecommendation heuristic straight to the backend, which produced a
	// wrong context for big MoE models. Only an explicit max/manual choice
	// overrides auto-fit.
	ctx := 0
	ctxFlag := "fit"
	if m.ctxMode == "max" {
		ctxFlag = "max"
	} else if m.ctxMode == "manual" && m.ctxSize != "" {
		ctxFlag = m.ctxSize
		if n, err := strconv.Atoi(m.ctxSize); err == nil {
			ctx = n
		}
	}
	parallel := 1
	parallelSet := m.parallelSet
	if m.parallel != "" {
		if n, err := strconv.Atoi(m.parallel); err == nil {
			parallel = n
		}
	}
	return &LaunchRequest{
		ModelPath:   model.Path,
		Port:        m.port,
		CtxSize:     ctx,
		CtxFlag:     ctxFlag,
		KVPlacement: m.kvPlacement,
		// The configured KV quality, not a hardcoded default: passing a fixed
		// "mid" here overrode the user's saved setting with --kv-quality mid
		// on every TUI launch (settings appeared to save but never applied).
		KVQuality:    m.kvQuality,
		KVQualitySet: m.kvQualityTouched,
		SWAFull:      m.swaFull,
		// Only emit --swa-full/--no-swa-full when this launch actually deviates
		// from the saved setting. Emitting it unconditionally turned a stored
		// preference into a command-line flag, and ggrun treats a typed flag as
		// inviolable: userExplicitBackendFlag reads OriginalArgs, so the
		// recovery ladder, the advisory notice and the support expert's
		// remove_generated_feature action were all locked out of it. Measured
		// 2026-08-03: --swa-full cost 5.3 GiB of KV on CUDA0 (6196 MiB vs 871),
		// which is exactly what made the launch unfittable — and nothing was
		// permitted to drop it, because the TUI had "typed" it. A setting is a
		// preference; only a human at the command line is an instruction.
		SWAFullSet:    m.swaFullTouched,
		FlashAttn:     true,
		Parallel:      parallel,
		ParallelSet:   parallelSet,
		Vision:        m.vision,
		Backend:       m.effectiveBackend(),
		TuneCache:     m.tunePath,
		AITune:        m.aitune,
		AITuneRounds:  m.aituneRounds,
		Benchmark:     m.benchmark,
		ClaudeCode:    m.claudeCode,
		ClaudeProfile: m.claudeProfile,
		// Only emit the reviewer override when the user explicitly deviated from
		// the automatic choice; an empty value keeps the CLI default.
		ClaudeReviewerOverride: m.claudeReviewer,
		SupportExpert:          m.supportExpert,
		SupportOnline:          m.supportOnline,
		SupportSet:             true,
		NoCachedConfig:         m.noCachedConfig,
	}
}

func (m Model) buildArgs() []string {
	req := m.buildLaunchRequest()
	if req == nil {
		return nil
	}
	return append([]string{"ggrun", "dry-run"}, req.LaunchArgs()...)
}

// LaunchRequest is returned when the user chooses to launch a model.
type LaunchRequest struct {
	Update        bool
	BackendArgs   []string
	DownloadRepo  string
	DownloadQuant string
	// DownloadDir sends this one download somewhere other than the configured
	// model directory, without changing that setting. Large quants routinely
	// have to land on a different disk than the default one, and making the
	// user repoint ModelDir and then remember to put it back is both tedious
	// and easy to get wrong.
	DownloadDir   string
	ModelPath     string
	Port          int
	CtxSize       int
	CtxFlag       string
	KVPlacement   string
	KVQuality     string
	KVQualitySet  bool // TUI explicitly cycled KV quality; emit an explicit override
	SWAFull       bool
	SWAFullSet    bool // TUI explicitly selected on/off; emit an override either way
	FlashAttn     bool
	Parallel      int
	ParallelSet   bool // user typed a parallel value (claude-code mode must not override)
	Vision        bool
	Backend       string
	TuneCache     string
	AITune        bool
	AITuneRounds  int
	Benchmark     bool
	ClaudeCode    bool
	ClaudeProfile string
	// ClaudeReviewerOverride picks the local worker/reviewer model in Claude
	// Code mode: empty/"auto" keeps ggrun's automatic choice, "qwen" forces the
	// Qwen profile, "nanbeige" forces the NanoBeige4.2 worker.
	ClaudeReviewerOverride string
	ResumeSession          string // reopen this recorded Claude Code session
	SupportExpert          string // optional native support/optimizer policy: off, auto, on
	SupportOnline          bool   // allow typed official llama.cpp research
	SupportSet             bool   // TUI explicitly selected the support and research policy
	// NoCachedConfig derives this launch fresh, skipping cached placement/probe
	// measurements without deleting them (the "launch without cached config"
	// escape hatch for a stale placement or probe).
	NoCachedConfig bool
}

func (req *LaunchRequest) LaunchArgs() []string {
	if req == nil {
		return nil
	}
	args := []string{req.ModelPath}
	if req.Port > 0 {
		args = append(args, "--port", strconv.Itoa(req.Port))
	}
	if req.CtxFlag != "" {
		args = append(args, "--ctx-size", req.CtxFlag)
	} else if req.CtxSize > 0 {
		args = append(args, "--ctx-size", strconv.Itoa(req.CtxSize))
	} else {
		args = append(args, "--ctx-size", "fit")
	}
	if req.KVPlacement != "" {
		args = append(args, "--kv-placement", req.KVPlacement)
	}
	// Only emit --kv-quality when this launch actually deviates from the saved
	// setting, mirroring --swa-full above: a stored preference belongs in config,
	// where ggrun still applies it but may withdraw it on a memory failure. A
	// cycled row is an explicit per-launch choice and goes out as a typed flag.
	if req.KVQualitySet && req.KVQuality != "" {
		args = append(args, "--kv-quality", req.KVQuality)
	}
	if req.SWAFullSet {
		if req.SWAFull {
			args = append(args, "--swa-full")
		} else {
			args = append(args, "--no-swa-full")
		}
	}
	if req.Vision {
		args = append(args, "--vision")
	}
	if req.Backend != "" && req.Backend != "auto" {
		args = append(args, "--backend", req.Backend)
	}
	if req.TuneCache != "" {
		args = append(args, "--tune-cache", req.TuneCache)
	}
	if req.AITune && req.AITuneRounds > 0 {
		args = append(args, "--rounds", strconv.Itoa(req.AITuneRounds))
	}
	if req.ParallelSet && req.Parallel > 0 {
		args = append(args, "--parallel", strconv.Itoa(req.Parallel))
	}
	if req.Benchmark {
		args = append(args, "--benchmark")
	}
	if req.SupportExpert != "" {
		args = append(args, "--support-expert", req.SupportExpert)
	}
	if req.SupportSet {
		if req.SupportOnline {
			args = append(args, "--support-online")
		} else {
			args = append(args, "--no-support-online")
		}
	}
	if req.NoCachedConfig {
		args = append(args, "--no-cached-config")
	}
	if req.ClaudeCode {
		args = append(args, "--claude-code")
		if req.ClaudeProfile != "" {
			args = append(args, "--claude-profile", req.ClaudeProfile)
		}
		// Only emit the reviewer override when the user explicitly picked one;
		// an empty or "auto" value keeps the CLI default.
		if v := strings.TrimSpace(req.ClaudeReviewerOverride); v != "" && v != "auto" {
			args = append(args, "--claude-reviewer", v)
		}
		// Resume goes through the same launch argv as the CLI, so the TUI and
		// the command line share one implementation.
		if req.ResumeSession != "" {
			args = append(args, "--claude-resume", req.ResumeSession)
		}
	}
	return args
}

func supportExpertLabel(mode string) string {
	switch strings.TrimSpace(mode) {
	case "off":
		return "off"
	case "on":
		return "on (required, ephemeral)"
	default:
		return "auto (installed-only, ephemeral)"
	}
}

// claudeReviewerLabel describes the reviewer/worker selector value without
// converting its empty default into an explicit launch flag.
func claudeReviewerLabel(reviewer string) string {
	switch strings.TrimSpace(reviewer) {
	case "qwen":
		return "qwen (Qwen3.5-4B dense)"
	case "nanbeige":
		return "nanbeige (Nanbeige4.2-3B worker)"
	default:
		return "auto (automatic)"
	}
}

// claudeProfileLabel describes the selector value without converting its empty
// default into an explicit launch flag.
func claudeProfileLabel(profile string) string {
	switch profile {
	case "agent-interactive":
		return "agent-interactive (1 foreground agent)"
	case "agent-parallel":
		return "agent-parallel (4 workflow slots)"
	default:
		return "default (automatic)"
	}
}

// prelaunchParallelLabel mirrors the Claude scheduling policy closely enough
// for the confirmation screen. Context fitting may still reduce an automatic
// slot count; explicit --parallel remains authoritative.
func (m Model) prelaunchParallelLabel() string {
	if m.parallelSet && m.parallel != "" {
		return m.parallel + " (explicit)"
	}
	if m.claudeCode {
		switch m.claudeProfile {
		case "agent-interactive":
			return "agent-interactive (1 foreground slot)"
		case "agent-parallel":
			return "agent-parallel (4 workflow slots)"
		default:
			if m.parallel == "" || m.parallel == "1" {
				return "automatic (Claude Code policy; target 4 slots)"
			}
		}
	}
	if !m.parallelSet && (m.parallel == "" || m.parallel == "1") {
		return "automatic (1)"
	}
	return m.parallel
}

func runModel(initial Model) (*LaunchRequest, error) {
	p := tea.NewProgram(initial, tea.WithAltScreen())
	m, err := p.Run()
	if err != nil {
		return nil, err
	}
	if model, ok := m.(Model); ok && model.launchRequest != nil {
		return model.launchRequest, nil
	}
	return nil, nil
}

// Run starts the TUI and returns a launch request if the user chose to launch.
func Run() (*LaunchRequest, error) {
	return runModel(InitialModel())
}

// RunAfterBackendInstall reopens the same model and settings at pre-launch
// after a reviewed fork has built successfully. The recipe tag stays scoped to
// this request; it does not overwrite the user's default backend for unrelated
// models.
func RunAfterBackendInstall(req *LaunchRequest) (*LaunchRequest, error) {
	m := InitialModel()
	if req == nil {
		return runModel(m)
	}
	if req.Backend != "" {
		m.backend = req.Backend
	}
	if !m.selectModelPath(req.ModelPath) {
		m.message = "Backend installed, but the selected model is no longer available: " + req.ModelPath
		m.messageType = "warning"
		m.screen = ScreenMain
		return runModel(m)
	}
	m.applyLaunchRequestFields(req)
	m.backendRouteBypass = false
	m.replayRequest = nil
	m.replaySavedAt = time.Time{}
	m.message = "Backend installed and auto-selected for this model. Review the launch, then press Enter."
	m.messageType = "info"
	m.screen = ScreenPrelaunch
	return runModel(m)
}
