//go:build cgo

package pdfoxide

// Broad API coverage tests — one test per public method not already covered
// in pdf_oxide_test.go / document_builder_test.go.
// Each test is self-contained: creates its own PDF via FromMarkdown and
// exercises exactly one method, then cleans up.

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ── test helpers ──────────────────────────────────────────────────────────────

func makeTempPDF(t *testing.T, markdown string) string {
	t.Helper()
	pdf, err := FromMarkdown(markdown)
	if err != nil {
		t.Fatalf("FromMarkdown: %v", err)
	}
	defer pdf.Close()
	f, err := os.CreateTemp("", "pdfoxide-cov-*.pdf")
	if err != nil {
		t.Fatalf("TempFile: %v", err)
	}
	path := f.Name()
	f.Close()
	if err := pdf.Save(path); err != nil {
		os.Remove(path)
		t.Fatalf("Save: %v", err)
	}
	return path
}

func openCovDoc(t *testing.T, markdown string) (*PdfDocument, func()) {
	t.Helper()
	path := makeTempPDF(t, markdown)
	doc, err := Open(path)
	if err != nil {
		os.Remove(path)
		t.Fatalf("Open: %v", err)
	}
	return doc, func() { doc.Close(); os.Remove(path) }
}

func tempPath(prefix string) string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return filepath.Join(os.TempDir(),
		fmt.Sprintf("%s-%016x.pdf", prefix, r.Int63()))
}

// 1×1 white PNG (69 bytes, valid with correct zlib + CRC checksums)
var pngBytes = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
	0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
	0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
	0x54, 0x78, 0xda, 0x63, 0xf8, 0xff, 0xff, 0x3f,
	0x00, 0x05, 0xfe, 0x02, 0xfe, 0x33, 0x12, 0x95,
	0x14, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
	0x44, 0xae, 0x42, 0x60, 0x82,
}

// ── Text extraction ────────────────────────────────────────────────────────────

func TestExtractWords(t *testing.T) {
	doc, cleanup := openCovDoc(t, "WORDTOKEN hello world")
	defer cleanup()
	words, err := doc.ExtractWords(0)
	if err != nil {
		t.Fatalf("ExtractWords: %v", err)
	}
	if len(words) == 0 {
		t.Fatal("expected non-empty word list")
	}
	found := false
	for _, w := range words {
		if strings.Contains(w.Text, "WORDTOKEN") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("WORDTOKEN not found in extracted words: %v", words)
	}
	for _, w := range words {
		if w.Sequence < 0 {
			t.Errorf("expected non-negative word Sequence, got %d for %q", w.Sequence, w.Text)
		}
	}
}

func TestExtractChars(t *testing.T) {
	doc, cleanup := openCovDoc(t, "ABC")
	defer cleanup()
	chars, err := doc.ExtractChars(0)
	if err != nil {
		t.Fatalf("ExtractChars: %v", err)
	}
	if len(chars) == 0 {
		t.Fatal("expected non-empty char list")
	}
	if chars[0].FontSize <= 0 {
		t.Errorf("expected positive FontSize, got %v", chars[0].FontSize)
	}
}

func TestExtractTextLines(t *testing.T) {
	doc, cleanup := openCovDoc(t, "LINETOKEN")
	defer cleanup()
	lines, err := doc.ExtractTextLines(0)
	if err != nil {
		t.Fatalf("ExtractTextLines: %v", err)
	}
	if len(lines) == 0 {
		t.Fatal("expected non-empty line list")
	}
	found := false
	for _, l := range lines {
		if strings.Contains(l.Text, "LINETOKEN") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("LINETOKEN not found in lines: %v", lines)
	}
}

func TestExtractAllText(t *testing.T) {
	doc, cleanup := openCovDoc(t, "ALLTEXTMARKER")
	defer cleanup()
	text, err := doc.ExtractAllText()
	if err != nil {
		t.Fatalf("ExtractAllText: %v", err)
	}
	if !strings.Contains(text, "ALLTEXTMARKER") {
		t.Errorf("marker not found in all-text: %q", text)
	}
}

// ── Conversion ────────────────────────────────────────────────────────────────

func TestToMarkdown(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Heading\n\nBody text.")
	defer cleanup()
	md, err := doc.ToMarkdown(0)
	if err != nil {
		t.Fatalf("ToMarkdown: %v", err)
	}
	if md == "" {
		t.Fatal("expected non-empty markdown")
	}
}

func TestToMarkdownAll(t *testing.T) {
	doc, cleanup := openCovDoc(t, "MD_ALL_MARKER")
	defer cleanup()
	md, err := doc.ToMarkdownAll()
	if err != nil {
		t.Fatalf("ToMarkdownAll: %v", err)
	}
	if md == "" {
		t.Fatal("expected non-empty markdown")
	}
}

func TestToHtml(t *testing.T) {
	doc, cleanup := openCovDoc(t, "HTML_MARKER")
	defer cleanup()
	html, err := doc.ToHtml(0)
	if err != nil {
		t.Fatalf("ToHtml: %v", err)
	}
	if !strings.Contains(html, "<") {
		t.Errorf("expected HTML tags in output, got: %q", html)
	}
}

func TestToHtmlAll(t *testing.T) {
	doc, cleanup := openCovDoc(t, "HTMLALL_MARKER")
	defer cleanup()
	html, err := doc.ToHtmlAll()
	if err != nil {
		t.Fatalf("ToHtmlAll: %v", err)
	}
	if html == "" {
		t.Fatal("expected non-empty HTML")
	}
}

func TestToPlainText(t *testing.T) {
	doc, cleanup := openCovDoc(t, "PLAINMARKER")
	defer cleanup()
	text, err := doc.ToPlainText(0)
	if err != nil {
		t.Fatalf("ToPlainText: %v", err)
	}
	if !strings.Contains(text, "PLAINMARKER") {
		t.Errorf("marker not found: %q", text)
	}
}

func TestToPlainTextAll(t *testing.T) {
	doc, cleanup := openCovDoc(t, "PLAINALLMARKER")
	defer cleanup()
	text, err := doc.ToPlainTextAll()
	if err != nil {
		t.Fatalf("ToPlainTextAll: %v", err)
	}
	if !strings.Contains(text, "PLAINALLMARKER") {
		t.Errorf("marker not found: %q", text)
	}
}

// ── Search ────────────────────────────────────────────────────────────────────

func TestSearchPage_FindsKnownTerm(t *testing.T) {
	doc, cleanup := openCovDoc(t, "SEARCHTOKENPAGE")
	defer cleanup()
	results, err := doc.SearchPage(0, "SEARCHTOKENPAGE", false)
	if err != nil {
		t.Fatalf("SearchPage: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one search result")
	}
}

func TestSearchAll_FindsKnownTerm(t *testing.T) {
	doc, cleanup := openCovDoc(t, "SEARCHTOKENALL")
	defer cleanup()
	results, err := doc.SearchAll("SEARCHTOKENALL", false)
	if err != nil {
		t.Fatalf("SearchAll: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one search result")
	}
}

func TestSearchAll_MissingTerm_ReturnsEmpty(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Hello")
	defer cleanup()
	results, err := doc.SearchAll("ZZZNOTPRESENTZZZ", false)
	if err != nil {
		t.Fatalf("SearchAll: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %d", len(results))
	}
}

// ── Merge ──────────────────────────────────────────────────────────────────────

func TestMerge_ProducesCombinedPDF(t *testing.T) {
	pathA := makeTempPDF(t, "# Page A")
	pathB := makeTempPDF(t, "# Page B")
	defer os.Remove(pathA)
	defer os.Remove(pathB)

	merged, err := Merge([]string{pathA, pathB})
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if !bytes.HasPrefix(merged, []byte("%PDF-")) {
		t.Fatal("merged output is not a PDF")
	}
	doc, err := OpenFromBytes(merged)
	if err != nil {
		t.Fatalf("OpenFromBytes: %v", err)
	}
	defer doc.Close()
	count, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	if count < 2 {
		t.Errorf("expected >=2 pages in merged PDF, got %d", count)
	}
}

func TestMerge_EmptySlice_ReturnsError(t *testing.T) {
	_, err := Merge([]string{})
	if err == nil {
		t.Fatal("expected error for empty path list")
	}
}

// ── DocumentBuilder extras ─────────────────────────────────────────────────────

func TestDocumentBuilderSave_NonEncrypted(t *testing.T) {
	b, err := NewDocumentBuilder()
	if err != nil {
		t.Skipf("NewDocumentBuilder: %v", err)
	}
	defer b.Close()
	p, _ := b.A4Page()
	if _, err := p.Paragraph("plain save").Done(); err != nil {
		t.Fatalf("page: %v", err)
	}
	path := tempPath("pdfoxide-save")
	defer os.Remove(path)
	if err := b.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	info, _ := os.Stat(path)
	if info.Size() < 100 {
		t.Fatalf("saved PDF too small: %d bytes", info.Size())
	}
}

func TestDocumentBuilderLetterPage(t *testing.T) {
	b, err := NewDocumentBuilder()
	if err != nil {
		t.Skipf("NewDocumentBuilder: %v", err)
	}
	defer b.Close()
	p, err := b.LetterPage()
	if err != nil {
		t.Fatalf("LetterPage: %v", err)
	}
	if _, err := p.Paragraph("US Letter").Done(); err != nil {
		t.Fatalf("page: %v", err)
	}
	data, err := b.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		t.Fatal("output is not a PDF")
	}
}

func TestDocumentBuilderCustomPageSize(t *testing.T) {
	b, err := NewDocumentBuilder()
	if err != nil {
		t.Skipf("NewDocumentBuilder: %v", err)
	}
	defer b.Close()
	p, err := b.Page(300, 400)
	if err != nil {
		t.Fatalf("Page: %v", err)
	}
	if _, err := p.Paragraph("custom size").Done(); err != nil {
		t.Fatalf("page: %v", err)
	}
	data, err := b.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(data) < 100 {
		t.Fatalf("PDF too small: %d bytes", len(data))
	}
}

func TestDocumentBuilderMetadata(t *testing.T) {
	b, err := NewDocumentBuilder()
	if err != nil {
		t.Skipf("NewDocumentBuilder: %v", err)
	}
	defer b.Close()
	if err := b.Title("My Title"); err != nil {
		t.Fatalf("Title: %v", err)
	}
	if err := b.Author("Alice"); err != nil {
		t.Fatalf("Author: %v", err)
	}
	if err := b.Subject("Testing"); err != nil {
		t.Fatalf("Subject: %v", err)
	}
	if err := b.Keywords("pdf, test"); err != nil {
		t.Fatalf("Keywords: %v", err)
	}
	if err := b.Creator("go-test"); err != nil {
		t.Fatalf("Creator: %v", err)
	}
	p, _ := b.A4Page()
	if _, err := p.Paragraph("metadata").Done(); err != nil {
		t.Fatalf("page: %v", err)
	}
	data, err := b.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(data) < 100 {
		t.Fatalf("PDF too small")
	}
}

// ── DocumentEditor mutations ───────────────────────────────────────────────────

func TestDocumentEditor_DeletePage(t *testing.T) {
	pathA := makeTempPDF(t, "# Page A")
	pathB := makeTempPDF(t, "# Page B")
	defer os.Remove(pathA)
	defer os.Remove(pathB)

	editor, err := OpenEditor(pathA)
	if err != nil {
		t.Fatalf("OpenEditor: %v", err)
	}
	defer editor.Close()

	if _, err := editor.MergeFrom(pathB); err != nil {
		t.Fatalf("MergeFrom: %v", err)
	}
	before, _ := editor.PageCount()
	if before < 2 {
		t.Skip("need multi-page PDF")
	}
	if err := editor.DeletePage(0); err != nil {
		t.Fatalf("DeletePage: %v", err)
	}
	after, _ := editor.PageCount()
	if after != before-1 {
		t.Errorf("expected %d pages after delete, got %d", before-1, after)
	}
}

func TestDocumentEditor_MovePage(t *testing.T) {
	// Build a 3-page document so MovePage has unambiguous before/after state.
	pathA := makeTempPDF(t, "FIRSTPAGE")
	pathB := makeTempPDF(t, "SECONDPAGE")
	pathC := makeTempPDF(t, "THIRDPAGE")
	mergedPath := tempPath("pdfoxide-merged")
	outPath := tempPath("pdfoxide-move")
	defer os.Remove(pathA)
	defer os.Remove(pathB)
	defer os.Remove(pathC)
	defer os.Remove(mergedPath)
	defer os.Remove(outPath)

	// Merge into one file first, then reopen — merged_pages must be flushed to
	// disk before MovePage or Rust's page_order vec panics (pre-existing bug).
	editor, err := OpenEditor(pathA)
	if err != nil {
		t.Fatalf("OpenEditor: %v", err)
	}
	if _, err := editor.MergeFrom(pathB); err != nil {
		t.Fatalf("MergeFrom B: %v", err)
	}
	if _, err := editor.MergeFrom(pathC); err != nil {
		t.Fatalf("MergeFrom C: %v", err)
	}
	if err := editor.Save(mergedPath); err != nil {
		t.Fatalf("Save merged: %v", err)
	}
	editor.Close()

	editor2, err := OpenEditor(mergedPath)
	if err != nil {
		t.Fatalf("OpenEditor2: %v", err)
	}
	defer editor2.Close()

	// Move last page (THIRDPAGE) to front → [THIRDPAGE, FIRSTPAGE, SECONDPAGE]
	if err := editor2.MovePage(2, 0); err != nil {
		t.Fatalf("MovePage: %v", err)
	}
	if err := editor2.Save(outPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	doc, err := Open(outPath)
	if err != nil {
		t.Fatalf("Open result: %v", err)
	}
	defer doc.Close()
	words, _ := doc.ExtractWords(0)
	var text string
	for _, w := range words {
		text += w.Text + " "
	}
	if !strings.Contains(text, "THIRDPAGE") {
		t.Errorf("expected THIRDPAGE on page 0 after move, got: %q", text)
	}
}

func TestDocumentEditor_SetTitle(t *testing.T) {
	path := makeTempPDF(t, "# Title Test")
	defer os.Remove(path)

	editor, err := OpenEditor(path)
	if err != nil {
		t.Fatalf("OpenEditor: %v", err)
	}
	defer editor.Close()

	if err := editor.SetTitle("New Title"); err != nil {
		t.Fatalf("SetTitle: %v", err)
	}
	title, err := editor.Title()
	if err != nil {
		t.Fatalf("Title: %v", err)
	}
	if title != "New Title" {
		t.Errorf("expected 'New Title', got %q", title)
	}
}

func TestDocumentEditor_MergeFrom(t *testing.T) {
	pathA := makeTempPDF(t, "# A")
	pathB := makeTempPDF(t, "# B")
	defer os.Remove(pathA)
	defer os.Remove(pathB)

	editor, err := OpenEditor(pathA)
	if err != nil {
		t.Fatalf("OpenEditor: %v", err)
	}
	defer editor.Close()

	before, _ := editor.PageCount()
	if _, err := editor.MergeFrom(pathB); err != nil {
		t.Fatalf("MergeFrom: %v", err)
	}
	after, _ := editor.PageCount()
	if after <= before {
		t.Errorf("expected more pages after merge, before=%d after=%d", before, after)
	}
}

// ── FromImageBytes ─────────────────────────────────────────────────────────────

func TestFromImageBytes_ProducesPDF(t *testing.T) {
	pdf, err := FromImageBytes(pngBytes)
	if err != nil {
		if isUnsupportedError(err) {
			t.Skipf("FromImageBytes unavailable: %v", err)
		}
		t.Fatalf("FromImageBytes: %v", err)
	}
	defer pdf.Close()
	data, err := pdf.SaveToBytes()
	if err != nil {
		t.Fatalf("SaveToBytes: %v", err)
	}
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		t.Fatal("output is not a PDF")
	}
}

// ── Signatures (unsigned) ──────────────────────────────────────────────────────

func TestSignatureCount_UnsignedIsZero(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Unsigned")
	defer cleanup()
	count, err := doc.SignatureCount()
	if err != nil {
		if isUnsupportedError(err) {
			t.Skipf("SignatureCount unavailable: %v", err)
		}
		t.Fatalf("SignatureCount: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0, got %d", count)
	}
}

func TestSignatures_UnsignedReturnsEmpty(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Unsigned")
	defer cleanup()
	sigs, err := doc.Signatures()
	if err != nil {
		if isUnsupportedError(err) {
			t.Skipf("Signatures unavailable: %v", err)
		}
		t.Fatalf("Signatures: %v", err)
	}
	if len(sigs) != 0 {
		t.Errorf("expected 0 signatures, got %d", len(sigs))
	}
}
