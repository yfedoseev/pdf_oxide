// Office format conversion: PDF → DOCX / PPTX / XLSX — v0.3.41
// Run: node index.js

import { DocumentBuilder, PdfDocument } from "pdf-oxide";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const outDir = join(import.meta.dirname ?? ".", "output");
mkdirSync(outDir, { recursive: true });

// Build a simple PDF in memory
const builder = DocumentBuilder.create().title("Office Conversion Demo");
builder.letterPage()
  .font("Helvetica", 14)
  .at(72, 720).heading(1, "Office Conversion Demo")
  .at(72, 690).paragraph("This PDF will be exported to DOCX, PPTX, and XLSX formats.")
  .done();

const pdfBytes = builder.build();
console.log(`Built sample PDF: ${pdfBytes.length} bytes`);

const doc = PdfDocument.openFromBuffer(pdfBytes);

// 1. PDF → DOCX
const docxBytes = doc.toDocxBytes();
if (docxBytes[0] !== 0x50 || docxBytes[1] !== 0x4B) throw new Error("DOCX output is not a valid ZIP/DOCX");
console.log(`PDF → DOCX: ${docxBytes.length} bytes — PASS`);
writeFileSync(join(outDir, "output.docx"), docxBytes);

// 2. PDF → PPTX
const pptxBytes = doc.toPptxBytes();
if (pptxBytes[0] !== 0x50 || pptxBytes[1] !== 0x4B) throw new Error("PPTX output is not a valid ZIP/PPTX");
console.log(`PDF → PPTX: ${pptxBytes.length} bytes — PASS`);
writeFileSync(join(outDir, "output.pptx"), pptxBytes);

// 3. PDF → XLSX
const xlsxBytes = doc.toXlsxBytes();
if (xlsxBytes[0] !== 0x50 || xlsxBytes[1] !== 0x4B) throw new Error("XLSX output is not a valid ZIP/XLSX");
console.log(`PDF → XLSX: ${xlsxBytes.length} bytes — PASS`);
writeFileSync(join(outDir, "output.xlsx"), xlsxBytes);

// Round-trips: office → PDF → office
const docxDoc = PdfDocument.openFromDocxBytes(docxBytes);
const docxBytes2 = docxDoc.toDocxBytes();
if (docxBytes2[0] !== 0x50 || docxBytes2[1] !== 0x4B) throw new Error("DOCX round-trip output invalid");
console.log(`DOCX → PDF → DOCX: ${docxBytes2.length} bytes — PASS`);
docxDoc.close();

const pptxDoc = PdfDocument.openFromPptxBytes(pptxBytes);
const pptxBytes2 = pptxDoc.toPptxBytes();
if (pptxBytes2[0] !== 0x50 || pptxBytes2[1] !== 0x4B) throw new Error("PPTX round-trip output invalid");
console.log(`PPTX → PDF → PPTX: ${pptxBytes2.length} bytes — PASS`);
pptxDoc.close();

const xlsxDoc = PdfDocument.openFromXlsxBytes(xlsxBytes);
const xlsxBytes2 = xlsxDoc.toXlsxBytes();
if (xlsxBytes2[0] !== 0x50 || xlsxBytes2[1] !== 0x4B) throw new Error("XLSX round-trip output invalid");
console.log(`XLSX → PDF → XLSX: ${xlsxBytes2.length} bytes — PASS`);
xlsxDoc.close();

doc.close();
console.log("\nAll office conversion checks passed.");
process.exit(0);
