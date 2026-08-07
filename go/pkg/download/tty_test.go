package download

import (
	"os"
	"testing"
)

// /dev/null is a character device, so a plain ModeCharDevice test calls it a
// terminal. Scripted runs redirect stdin from /dev/null, so that mistake made
// ggrun withhold --yes and the downloader died on EOFError -- after it had
// already resolved the repo and the file list.
func TestStdinIsTerminalRejectsDevNull(t *testing.T) {
	devnull, err := os.Open(os.DevNull)
	if err != nil {
		t.Skipf("cannot open %s: %v", os.DevNull, err)
	}
	defer devnull.Close()

	saved := os.Stdin
	os.Stdin = devnull
	defer func() { os.Stdin = saved }()

	if stdinIsTerminal() {
		t.Fatal("/dev/null must not be reported as a terminal")
	}
}

// A pipe is the other way a download gets started without anyone watching.
func TestStdinIsTerminalRejectsPipe(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Skipf("cannot create pipe: %v", err)
	}
	defer r.Close()
	defer w.Close()

	saved := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = saved }()

	if stdinIsTerminal() {
		t.Fatal("a pipe must not be reported as a terminal")
	}
}

// A regular file redirected onto stdin must also not be mistaken for one.
func TestStdinIsTerminalRejectsRegularFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Skipf("cannot create temp file: %v", err)
	}
	defer f.Close()

	saved := os.Stdin
	os.Stdin = f
	defer func() { os.Stdin = saved }()

	if stdinIsTerminal() {
		t.Fatal("a regular file must not be reported as a terminal")
	}
}
