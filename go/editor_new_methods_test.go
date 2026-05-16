//go:build cgo

package pdfoxide

// Tests for new DocumentEditor methods added in v0.3.39:
//   - OpenEditorFromBytes / SaveToBytes / SaveToBytesWithOptions
//   - Keywords / SetKeywords
//   - MergeFromBytes
//   - EmbedFile
//   - ApplyPageRedactions / ApplyAllRedactions
//   - RotateAllPages / RotatePageBy
//   - GetPageMediaBox / SetPageMediaBox
//   - GetPageCropBox / SetPageCropBox
//   - EraseRegions / ClearEraseRegions
//   - IsPageMarkedForFlatten / UnmarkPageForFlatten
//   - IsPageMarkedForRedaction / UnmarkPageForRedaction

import (
	"bytes"
	"os"
	"testing"
)

// openEditorForTest creates a minimal PDF via FromMarkdown, saves it to a temp
// file, opens it with OpenEditor, and returns the editor and a cleanup func.
func openEditorForTest(t *testing.T, content string) (*DocumentEditor, func()) {
	t.Helper()
	path := makeTempPDF(t, content)
	editor, err := OpenEditor(path)
	if err != nil {
		os.Remove(path)
		t.Fatalf("OpenEditor: %v", err)
	}
	return editor, func() {
		editor.Close()
		os.Remove(path)
	}
}

// ── SaveToBytes ────────────────────────────────────────────────────────────────

func TestDocumentEditor_SaveToBytes_ReturnsPDF(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# SaveToBytes test")
	defer cleanup()

	data, err := editor.SaveToBytes()
	if err != nil {
		t.Fatalf("SaveToBytes: %v", err)
	}
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		t.Errorf("SaveToBytes: result is not a PDF (first 5 bytes: %q)", data[:5])
	}
	if len(data) < 100 {
		t.Errorf("SaveToBytes: suspiciously small output (%d bytes)", len(data))
	}
}

func TestDocumentEditor_SaveToBytesWithOptions_ReturnsPDF(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# SaveToBytesWithOptions")
	defer cleanup()

	data, err := editor.SaveToBytesWithOptions(true, true, false)
	if err != nil {
		t.Fatalf("SaveToBytesWithOptions: %v", err)
	}
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		t.Errorf("output is not a PDF")
	}
}

// ── OpenEditorFromBytes ────────────────────────────────────────────────────────

func TestOpenEditorFromBytes_RoundTrip(t *testing.T) {
	// Create an editor, save to bytes, re-open from bytes
	editor, cleanup := openEditorForTest(t, "# OpenEditorFromBytes")
	defer cleanup()

	data, err := editor.SaveToBytes()
	if err != nil {
		t.Fatalf("SaveToBytes: %v", err)
	}

	editor2, err := OpenEditorFromBytes(data)
	if err != nil {
		t.Fatalf("OpenEditorFromBytes: %v", err)
	}
	defer editor2.Close()

	count, err := editor2.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	if count < 1 {
		t.Errorf("expected at least 1 page, got %d", count)
	}
}

func TestOpenEditorFromBytes_EmptyReturnsError(t *testing.T) {
	_, err := OpenEditorFromBytes(nil)
	if err == nil {
		t.Fatal("expected error for nil data")
	}
	_, err = OpenEditorFromBytes([]byte{})
	if err == nil {
		t.Fatal("expected error for empty data")
	}
}

// ── Keywords ──────────────────────────────────────────────────────────────────

func TestDocumentEditor_Keywords_SetGet(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# Keywords test")
	defer cleanup()

	if err := editor.SetKeywords("go, test, pdf"); err != nil {
		t.Fatalf("SetKeywords: %v", err)
	}
	kw, err := editor.Keywords()
	if err != nil {
		t.Fatalf("Keywords: %v", err)
	}
	if kw != "go, test, pdf" {
		t.Errorf("expected 'go, test, pdf', got %q", kw)
	}
}

// ── MergeFromBytes ────────────────────────────────────────────────────────────

func TestDocumentEditor_MergeFromBytes_IncreasesPageCount(t *testing.T) {
	pathA := makeTempPDF(t, "# A")
	pathB := makeTempPDF(t, "# B")
	defer os.Remove(pathA)
	defer os.Remove(pathB)

	editor, err := OpenEditor(pathA)
	if err != nil {
		t.Fatalf("OpenEditor: %v", err)
	}
	defer editor.Close()

	// Read pathB into bytes
	dataB, err := os.ReadFile(pathB)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	before, _ := editor.PageCount()
	if _, err := editor.MergeFromBytes(dataB); err != nil {
		t.Fatalf("MergeFromBytes: %v", err)
	}
	after, _ := editor.PageCount()
	if after <= before {
		t.Errorf("expected more pages after MergeFromBytes: before=%d after=%d", before, after)
	}
}

// ── EmbedFile ─────────────────────────────────────────────────────────────────

func TestDocumentEditor_EmbedFile_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# EmbedFile test")
	defer cleanup()

	payload := []byte("hello embedded world")
	if err := editor.EmbedFile("hello.txt", payload); err != nil {
		t.Fatalf("EmbedFile: %v", err)
	}
}

// ── ApplyPageRedactions / ApplyAllRedactions ──────────────────────────────────

func TestDocumentEditor_ApplyPageRedactions_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# Redact test")
	defer cleanup()

	// No redactions marked — should succeed (no-op)
	if err := editor.ApplyPageRedactions(0); err != nil {
		t.Fatalf("ApplyPageRedactions: %v", err)
	}
}

func TestDocumentEditor_ApplyAllRedactions_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# Redact all test")
	defer cleanup()

	if err := editor.ApplyAllRedactions(); err != nil {
		t.Fatalf("ApplyAllRedactions: %v", err)
	}
}

// ── Destructive redaction (#231): AddRedaction / RedactionCount / ApplyRedactions

func TestDocumentEditor_DestructiveRedaction_Parity(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# Destructive redact parity")
	defer cleanup()

	if n, err := editor.RedactionCount(0); err != nil || n != 0 {
		t.Fatalf("RedactionCount initial: n=%d err=%v", n, err)
	}
	if err := editor.AddRedaction(0, [4]float64{0, 0, 5000, 5000}, nil); err != nil {
		t.Fatalf("AddRedaction: %v", err)
	}
	if n, err := editor.RedactionCount(0); err != nil || n != 1 {
		t.Fatalf("RedactionCount after add: n=%d err=%v", n, err)
	}
	// Destructively apply — must not error on a simple-font document.
	if _, err := editor.ApplyRedactions(true); err != nil {
		t.Fatalf("ApplyRedactions: %v", err)
	}
}

// ── RotateAllPages ────────────────────────────────────────────────────────────

func TestDocumentEditor_RotateAllPages_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# RotateAll test")
	defer cleanup()

	if err := editor.RotateAllPages(90); err != nil {
		t.Fatalf("RotateAllPages: %v", err)
	}
}

// ── RotatePageBy ──────────────────────────────────────────────────────────────

func TestDocumentEditor_RotatePageBy_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# RotatePageBy test")
	defer cleanup()

	if err := editor.RotatePageBy(0, 180); err != nil {
		t.Fatalf("RotatePageBy: %v", err)
	}
}

// ── GetPageMediaBox / SetPageMediaBox ─────────────────────────────────────────

func TestDocumentEditor_MediaBox_GetAndSet(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# MediaBox test")
	defer cleanup()

	// Get existing media box
	x, y, w, h, err := editor.GetPageMediaBox(0)
	if err != nil {
		t.Fatalf("GetPageMediaBox: %v", err)
	}
	if w <= 0 || h <= 0 {
		t.Errorf("expected positive dimensions, got w=%f h=%f", w, h)
	}

	// Set and read back
	if err := editor.SetPageMediaBox(0, x, y, w, h); err != nil {
		t.Fatalf("SetPageMediaBox: %v", err)
	}
}

// ── GetPageCropBox / SetPageCropBox ───────────────────────────────────────────

func TestDocumentEditor_CropBox_SetAndGet(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# CropBox test")
	defer cleanup()

	// Get (may return 0,0,0,0 if no CropBox)
	_, _, _, _, err := editor.GetPageCropBox(0)
	if err != nil {
		t.Fatalf("GetPageCropBox: %v", err)
	}

	// Set a crop box
	if err := editor.SetPageCropBox(0, 10, 10, 500, 700); err != nil {
		t.Fatalf("SetPageCropBox: %v", err)
	}
}

// ── EraseRegions / ClearEraseRegions ─────────────────────────────────────────

func TestDocumentEditor_EraseRegions_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# EraseRegions test")
	defer cleanup()

	rects := [][4]float64{
		{10, 10, 100, 50},
		{200, 200, 80, 40},
	}
	if err := editor.EraseRegions(0, rects); err != nil {
		t.Fatalf("EraseRegions: %v", err)
	}
}

func TestDocumentEditor_ClearEraseRegions_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# ClearEraseRegions test")
	defer cleanup()

	if err := editor.ClearEraseRegions(0); err != nil {
		t.Fatalf("ClearEraseRegions: %v", err)
	}
}

// ── IsPageMarkedForFlatten / UnmarkPageForFlatten ─────────────────────────────

func TestDocumentEditor_FlattenMark_DefaultFalse(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# FlattenMark test")
	defer cleanup()

	if editor.IsPageMarkedForFlatten(0) {
		t.Error("expected page 0 not marked for flatten by default")
	}
}

func TestDocumentEditor_UnmarkPageForFlatten_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# UnmarkFlatten test")
	defer cleanup()

	if err := editor.UnmarkPageForFlatten(0); err != nil {
		t.Fatalf("UnmarkPageForFlatten: %v", err)
	}
}

// ── IsPageMarkedForRedaction / UnmarkPageForRedaction ────────────────────────

func TestDocumentEditor_RedactionMark_DefaultFalse(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# RedactionMark test")
	defer cleanup()

	if editor.IsPageMarkedForRedaction(0) {
		t.Error("expected page 0 not marked for redaction by default")
	}
}

func TestDocumentEditor_UnmarkPageForRedaction_DoesNotError(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# UnmarkRedaction test")
	defer cleanup()

	if err := editor.UnmarkPageForRedaction(0); err != nil {
		t.Fatalf("UnmarkPageForRedaction: %v", err)
	}
}
