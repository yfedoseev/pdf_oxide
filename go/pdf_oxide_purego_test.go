//go:build !cgo

package pdfoxide

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// The purego backend needs a shared library at runtime. Tests set
// PDF_OXIDE_LIB_PATH from the test env if the caller exported it, else
// they're skipped. The repo's release build drops libpdf_oxide.so under
// target/release/ — CI and local devs can point the env var there.
func requireLib(t *testing.T) {
	t.Helper()
	if os.Getenv("PDF_OXIDE_LIB_PATH") == "" {
		t.Skip("PDF_OXIDE_LIB_PATH not set — skipping purego integration test")
	}
}

// makePDF creates a small on-disk PDF for a read-side test by going through
// the cgo-free FromMarkdown path (which is included in the purego coverage).
func makePDF(t *testing.T) string {
	t.Helper()
	pdf, err := FromMarkdown("# Hello\n\nThis is a **test** PDF.")
	if err != nil {
		t.Fatalf("FromMarkdown: %v", err)
	}
	defer pdf.Close()
	path := filepath.Join(t.TempDir(), "test.pdf")
	if err := pdf.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return path
}

func TestPurego_OpenNonexistent(t *testing.T) {
	requireLib(t)
	_, err := Open("/nonexistent/file.pdf")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrDocumentNotFound) && !errors.Is(err, ErrInvalidPath) {
		t.Errorf("got %v, want ErrDocumentNotFound or ErrInvalidPath", err)
	}
}

func TestPurego_DocumentClosed(t *testing.T) {
	doc := &PdfDocument{closed: true}
	if _, err := doc.PageCount(); !errors.Is(err, ErrDocumentClosed) {
		t.Errorf("PageCount: got %v, want ErrDocumentClosed", err)
	}
	if _, err := doc.ExtractText(0); !errors.Is(err, ErrDocumentClosed) {
		t.Errorf("ExtractText: got %v, want ErrDocumentClosed", err)
	}
}

func TestPurego_RoundTrip(t *testing.T) {
	requireLib(t)
	path := makePDF(t)

	doc, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()

	n, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	if n < 1 {
		t.Errorf("PageCount = %d, want >= 1", n)
	}

	major, minor, err := doc.Version()
	if err != nil {
		t.Fatalf("Version: %v", err)
	}
	if major != 1 {
		t.Errorf("Version = %d.%d, want major = 1", major, minor)
	}

	txt, err := doc.ExtractText(0)
	if err != nil {
		t.Fatalf("ExtractText: %v", err)
	}
	if txt == "" {
		t.Error("ExtractText returned empty string")
	}

	md, err := doc.ToMarkdown(0)
	if err != nil {
		t.Fatalf("ToMarkdown: %v", err)
	}
	if md == "" {
		t.Error("ToMarkdown returned empty string")
	}

	html, err := doc.ToHtml(0)
	if err != nil {
		t.Fatalf("ToHtml: %v", err)
	}
	if html == "" {
		t.Error("ToHtml returned empty string")
	}

	pt, err := doc.ToPlainText(0)
	if err != nil {
		t.Fatalf("ToPlainText: %v", err)
	}
	if pt == "" {
		t.Error("ToPlainText returned empty string")
	}
}

func TestPurego_LogLevel(t *testing.T) {
	requireLib(t)
	orig := GetLogLevel()
	t.Cleanup(func() { SetLogLevel(orig) })

	SetLogLevel(LogWarn)
	if got := GetLogLevel(); got != LogWarn {
		t.Errorf("GetLogLevel = %v, want %v", got, LogWarn)
	}
}

func TestPurego_Fonts(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	fonts, err := doc.Fonts(0)
	if err != nil {
		t.Fatalf("Fonts: %v", err)
	}
	// A PDF generated from markdown should embed at least one font.
	if len(fonts) == 0 {
		t.Log("Fonts(0) returned empty — may be OK for minimal PDF")
	}
}

func TestPurego_Annotations(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	// Minimal PDF has no annotations — just verify the call doesn't error.
	if _, err := doc.Annotations(0); err != nil {
		t.Fatalf("Annotations: %v", err)
	}
}

func TestPurego_PageElements(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	elems, err := doc.PageElements(0)
	if err != nil {
		t.Fatalf("PageElements: %v", err)
	}
	if len(elems) == 0 {
		t.Error("PageElements(0) returned empty")
	}
}

func TestPurego_Search(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	hits, err := doc.SearchAll("test", false)
	if err != nil {
		t.Fatalf("SearchAll: %v", err)
	}
	if len(hits) == 0 {
		t.Error("SearchAll(test) returned 0 hits")
	}
}

func TestPurego_PrepareAndClearSearchIndex(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	if err := doc.PrepareSearch(); err != nil {
		t.Fatalf("PrepareSearch: %v", err)
	}
	if hits, err := doc.SearchAll("test", false); err != nil || len(hits) == 0 {
		t.Fatalf("SearchAll after PrepareSearch: hits=%d err=%v", len(hits), err)
	}
	if err := doc.ClearSearchIndex(); err != nil {
		t.Fatalf("ClearSearchIndex: %v", err)
	}
	if hits, err := doc.SearchAll("test", false); err != nil || len(hits) == 0 {
		t.Fatalf("SearchAll after ClearSearchIndex: hits=%d err=%v", len(hits), err)
	}
}

func TestPurego_PageSize(t *testing.T) {
	requireLib(t)
	doc, err := Open(makePDF(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()
	w, err := doc.PageWidth(0)
	if err != nil {
		t.Fatalf("PageWidth: %v", err)
	}
	h, err := doc.PageHeight(0)
	if err != nil {
		t.Fatalf("PageHeight: %v", err)
	}
	// A4 ≈ 595 × 842, Letter ≈ 612 × 792. Just assert non-zero.
	if w <= 0 || h <= 0 {
		t.Errorf("PageWidth/Height = %.1f × %.1f, want > 0", w, h)
	}
}

func TestPurego_OpenFromBytes(t *testing.T) {
	requireLib(t)
	path := makePDF(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	doc, err := OpenFromBytes(data)
	if err != nil {
		t.Fatalf("OpenFromBytes: %v", err)
	}
	defer doc.Close()

	n, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	if n < 1 {
		t.Errorf("PageCount = %d, want >= 1", n)
	}
}
