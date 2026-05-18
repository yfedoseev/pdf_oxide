//go:build !cgo

package pdfoxide

// v0.3.51 purego-backend parity test: the cgo-free build must now
// expose the #519 OCR model provisioning trio with the SAME
// signatures as the cgo backend. Runtime tests need the shared lib
// (skipped without it). Network-free — only the air-gapped manifest
// is asserted (no downloads; those belong to the model-gated Rust
// lane).

import (
	"strings"
	"testing"
)

func TestPurego_ModelManifest_V0351(t *testing.T) {
	requireLib(t)
	manifest := ModelManifest()
	if !strings.Contains(manifest, "det.onnx") {
		t.Errorf("ModelManifest() must list the shared detector det.onnx; got %q", manifest)
	}
	if !strings.Contains(manifest, "english") {
		t.Errorf("ModelManifest() must list the english recognition model; got %q", manifest)
	}

	// PrefetchAvailable() is a pure feature probe (no I/O); just
	// exercise the call path and signature.
	_ = PrefetchAvailable() // bool — matches the cgo signature
}
