// SPDX-License-Identifier: MIT OR Apache-2.0
//go:build pdf_oxide_dev

// #524 task 9: verify the OCR detection-unclip fix through the real Go
// cgo binding over the C-ABI (the SAME pdf_ocr_extract_text symbol the
// C# P/Invoke binding calls). Run:
//
//	ORT_DYLIB_PATH=/path/libonnxruntime.so \
//	  go test -tags pdf_oxide_dev -run TestOcrUnclipFixGoBinding -v
package pdfoxide

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOcrUnclipFixGoBinding(t *testing.T) {
	home, _ := os.UserHomeDir()
	md := filepath.Join(home, ".cache", "pdf_oxide", "models")
	for _, f := range []string{"det.onnx", "rec.onnx", "en_dict.txt"} {
		if _, err := os.Stat(filepath.Join(md, f)); err != nil {
			t.Skipf("model %s missing: %v", f, err)
		}
	}
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("ORT_DYLIB_PATH not set")
	}

	eng, err := NewOcrEngine(
		filepath.Join(md, "det.onnx"),
		filepath.Join(md, "rec.onnx"),
		filepath.Join(md, "en_dict.txt"),
	)
	if err != nil {
		t.Fatalf("NewOcrEngine: %v", err)
	}
	defer eng.Close()

	doc, err := Open("../tests/fixtures/ocr/auto_image_text_en.pdf")
	if err != nil {
		t.Fatalf("Open fixture: %v", err)
	}
	defer doc.Close()

	got, err := doc.ExtractTextWithOcr(0, eng)
	if err != nil {
		t.Fatalf("ExtractTextWithOcr: %v", err)
	}
	const want = "OCR fidelity test hello world 2024"
	if got != want {
		t.Fatalf("Go binding OCR mismatch (unclip regression?):\n got = %q\n want = %q", got, want)
	}
	t.Logf("Go binding OCR correct: %q", got)
}
