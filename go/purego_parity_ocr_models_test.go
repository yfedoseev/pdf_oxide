//go:build !cgo

package pdfoxide

// purego-backend parity test: the cgo-free build must now expose the
// OCR model provisioning trio with the SAME signatures as the cgo
// backend. Runtime tests need the shared lib (skipped without it).
// Network-free — only the air-gapped manifest is asserted (no
// downloads; those belong to the model-gated Rust lane).

import (
	"strings"
	"testing"
)

func TestPurego_ModelManifest(t *testing.T) {
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
