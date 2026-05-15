// Office format conversion: PDF → DOCX / PPTX / XLSX — v0.3.41
//
// Demonstrates office format export:
//   1. Build a PDF in memory
//   2. Export to DOCX bytes (PDF → DOCX)
//   3. Export to PPTX bytes (PDF → PPTX)
//   4. Export to XLSX bytes (PDF → XLSX)
//
// Run: go run -tags pdf_oxide_dev main.go

package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	pdfoxide "github.com/yfedoseev/pdf_oxide/go"
)

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	outDir := "output"
	must(os.MkdirAll(outDir, 0o755))

	// Build a simple PDF in memory.
	b, err := pdfoxide.NewDocumentBuilder()
	must(err)
	defer b.Close()
	must(b.Title("Office Conversion Demo"))

	page, err := b.LetterPage()
	must(err)
	_, err = page.
		Font("Helvetica", 14).
		At(72, 720).
		Heading(1, "Office Conversion Demo").
		Font("Helvetica", 11).
		At(72, 690).
		Paragraph("This PDF will be exported to DOCX, PPTX, and XLSX formats.").
		Done()
	must(err)

	pdfBytes, err := b.Build()
	must(err)
	fmt.Printf("Built sample PDF: %d bytes\n", len(pdfBytes))

	doc, err := pdfoxide.OpenFromBytes(pdfBytes)
	must(err)
	defer doc.Close()

	// 1. PDF → DOCX
	docxBytes, err := doc.ToDocxBytes()
	must(err)
	if len(docxBytes) < 2 || docxBytes[0] != 'P' || docxBytes[1] != 'K' {
		log.Fatal("DOCX output is not a valid ZIP/DOCX")
	}
	fmt.Printf("PDF → DOCX: %d bytes — PASS\n", len(docxBytes))
	must(os.WriteFile(filepath.Join(outDir, "output.docx"), docxBytes, 0o644))

	// 2. PDF → PPTX
	pptxBytes, err := doc.ToPptxBytes()
	must(err)
	if len(pptxBytes) < 2 || pptxBytes[0] != 'P' || pptxBytes[1] != 'K' {
		log.Fatal("PPTX output is not a valid ZIP/PPTX")
	}
	fmt.Printf("PDF → PPTX: %d bytes — PASS\n", len(pptxBytes))
	must(os.WriteFile(filepath.Join(outDir, "output.pptx"), pptxBytes, 0o644))

	// 3. PDF → XLSX
	xlsxBytes, err := doc.ToXlsxBytes()
	must(err)
	if len(xlsxBytes) < 2 || xlsxBytes[0] != 'P' || xlsxBytes[1] != 'K' {
		log.Fatal("XLSX output is not a valid ZIP/XLSX")
	}
	fmt.Printf("PDF → XLSX: %d bytes — PASS\n", len(xlsxBytes))
	must(os.WriteFile(filepath.Join(outDir, "output.xlsx"), xlsxBytes, 0o644))

	// Round-trips: office → PDF → office
	docxDoc, err := pdfoxide.OpenFromDocxBytes(docxBytes)
	must(err)
	defer docxDoc.Close()
	docxBytes2, err := docxDoc.ToDocxBytes()
	must(err)
	if len(docxBytes2) < 2 || docxBytes2[0] != 'P' || docxBytes2[1] != 'K' {
		log.Fatal("DOCX round-trip output invalid")
	}
	fmt.Printf("DOCX → PDF → DOCX: %d bytes — PASS\n", len(docxBytes2))

	pptxDoc, err := pdfoxide.OpenFromPptxBytes(pptxBytes)
	must(err)
	defer pptxDoc.Close()
	pptxBytes2, err := pptxDoc.ToPptxBytes()
	must(err)
	if len(pptxBytes2) < 2 || pptxBytes2[0] != 'P' || pptxBytes2[1] != 'K' {
		log.Fatal("PPTX round-trip output invalid")
	}
	fmt.Printf("PPTX → PDF → PPTX: %d bytes — PASS\n", len(pptxBytes2))

	xlsxDoc, err := pdfoxide.OpenFromXlsxBytes(xlsxBytes)
	must(err)
	defer xlsxDoc.Close()
	xlsxBytes2, err := xlsxDoc.ToXlsxBytes()
	must(err)
	if len(xlsxBytes2) < 2 || xlsxBytes2[0] != 'P' || xlsxBytes2[1] != 'K' {
		log.Fatal("XLSX round-trip output invalid")
	}
	fmt.Printf("XLSX → PDF → XLSX: %d bytes — PASS\n", len(xlsxBytes2))

	fmt.Println("\nAll office conversion checks passed.")
}
