//go:build cgo

package pdfoxide

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// Core functional test-parity suite (Go) — mirrors the shared cross-language
// spec (docs/releases/plans/v0.3.61/core-test-parity-spec.md) with the
// idiomatic Go API. Every binding asserts the same behaviors.
//
// Go has no on-disk fixture dependency: each case builds its own input from
// text (the same approach the rest of the Go suite uses via createTestPDF).

func parityPDF(t *testing.T) []byte {
	t.Helper()
	creator, err := FromText("Core parity across all bindings.\nSecond line of text.")
	if err != nil {
		t.Skipf("FromText unavailable in this build: %v", err)
	}
	defer creator.Close()
	data, err := creator.SaveToBytes()
	if err != nil {
		t.Fatalf("SaveToBytes failed: %v", err)
	}
	return data
}

func parityOpen(t *testing.T) *PdfDocument {
	t.Helper()
	doc, err := OpenFromBytes(parityPDF(t))
	if err != nil {
		t.Fatalf("OpenFromBytes failed: %v", err)
	}
	return doc
}

func TestParity_OpenAndPageCount(t *testing.T) {
	doc := parityOpen(t)
	defer doc.Close()
	n, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	if n != 1 {
		t.Errorf("PageCount = %d, want 1", n)
	}
}

func TestParity_ExtractText(t *testing.T) {
	doc := parityOpen(t)
	defer doc.Close()
	if _, err := doc.ExtractText(0); err != nil {
		t.Errorf("ExtractText: %v", err)
	}
}

func TestParity_ConvertMarkdownHTMLPlain(t *testing.T) {
	doc := parityOpen(t)
	defer doc.Close()
	if _, err := doc.ToMarkdown(0); err != nil {
		t.Errorf("ToMarkdown: %v", err)
	}
	if _, err := doc.ToHtml(0); err != nil {
		t.Errorf("ToHtml: %v", err)
	}
	if _, err := doc.ToPlainText(0); err != nil {
		t.Errorf("ToPlainText: %v", err)
	}
}

func TestParity_Search(t *testing.T) {
	doc := parityOpen(t)
	defer doc.Close()
	if _, err := doc.SearchAll("parity", false); err != nil {
		t.Errorf("SearchAll: %v", err)
	}
}

func TestParity_Structured(t *testing.T) {
	doc := parityOpen(t)
	defer doc.Close()
	if _, err := doc.ExtractStructured(0); err != nil {
		t.Errorf("ExtractStructured: %v", err)
	}
}

func TestParity_CreatePDF(t *testing.T) {
	if data := parityPDF(t); !bytes.HasPrefix(data, []byte("%PDF")) {
		t.Errorf("created bytes do not start with %%PDF")
	}
}

func TestParity_OpenFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "parity.pdf")
	if err := os.WriteFile(path, parityPDF(t), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	doc, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	if n, _ := doc.PageCount(); n != 1 {
		t.Errorf("PageCount = %d, want 1", n)
	}
}

func TestParity_OpenError(t *testing.T) {
	if _, err := Open("/no/such/file/does/not/exist.pdf"); err == nil {
		t.Error("expected error opening a missing file, got nil")
	}
}
