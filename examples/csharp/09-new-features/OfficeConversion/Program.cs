// Office format conversion: PDF → DOCX / PPTX / XLSX — v0.3.41
// Run: dotnet run
using System;
using System.IO;
using PdfOxide.Core;

const string OutDir = "output";
Directory.CreateDirectory(OutDir);

Console.WriteLine("Demonstrating office format conversion...");

using var builder = DocumentBuilder.Create();
builder.Title("Office Conversion Demo");
builder.LetterPage()
    .Font("Helvetica", 14)
    .At(72, 720).Heading(1, "Office Conversion Demo")
    .Font("Helvetica", 11)
    .At(72, 690).Paragraph("This PDF will be exported to DOCX, PPTX, and XLSX formats.")
    .Done();
byte[] pdfBytes = builder.Build();
Console.WriteLine($"Built sample PDF: {pdfBytes.Length} bytes");

using var doc = PdfDocument.Open(pdfBytes);

// 1. PDF → DOCX
byte[] docxBytes = doc.ToDocxBytes();
if (docxBytes.Length < 2 || docxBytes[0] != 0x50 || docxBytes[1] != 0x4B)
    throw new Exception("DOCX output is not a valid ZIP/DOCX");
Console.WriteLine($"PDF → DOCX: {docxBytes.Length} bytes — PASS");
File.WriteAllBytes(Path.Combine(OutDir, "output.docx"), docxBytes);

// 2. PDF → PPTX
byte[] pptxBytes = doc.ToPptxBytes();
if (pptxBytes.Length < 2 || pptxBytes[0] != 0x50 || pptxBytes[1] != 0x4B)
    throw new Exception("PPTX output is not a valid ZIP/PPTX");
Console.WriteLine($"PDF → PPTX: {pptxBytes.Length} bytes — PASS");
File.WriteAllBytes(Path.Combine(OutDir, "output.pptx"), pptxBytes);

// 3. PDF → XLSX
byte[] xlsxBytes = doc.ToXlsxBytes();
if (xlsxBytes.Length < 2 || xlsxBytes[0] != 0x50 || xlsxBytes[1] != 0x4B)
    throw new Exception("XLSX output is not a valid ZIP/XLSX");
Console.WriteLine($"PDF → XLSX: {xlsxBytes.Length} bytes — PASS");
File.WriteAllBytes(Path.Combine(OutDir, "output.xlsx"), xlsxBytes);

// Round-trips: office → PDF → office
using var docxDoc = PdfDocument.OpenFromDocxBytes(docxBytes);
byte[] docxBytes2 = docxDoc.ToDocxBytes();
if (docxBytes2.Length < 2 || docxBytes2[0] != 0x50 || docxBytes2[1] != 0x4B)
    throw new Exception("DOCX round-trip output invalid");
Console.WriteLine($"DOCX → PDF → DOCX: {docxBytes2.Length} bytes — PASS");

using var pptxDoc = PdfDocument.OpenFromPptxBytes(pptxBytes);
byte[] pptxBytes2 = pptxDoc.ToPptxBytes();
if (pptxBytes2.Length < 2 || pptxBytes2[0] != 0x50 || pptxBytes2[1] != 0x4B)
    throw new Exception("PPTX round-trip output invalid");
Console.WriteLine($"PPTX → PDF → PPTX: {pptxBytes2.Length} bytes — PASS");

using var xlsxDoc = PdfDocument.OpenFromXlsxBytes(xlsxBytes);
byte[] xlsxBytes2 = xlsxDoc.ToXlsxBytes();
if (xlsxBytes2.Length < 2 || xlsxBytes2[0] != 0x50 || xlsxBytes2[1] != 0x4B)
    throw new Exception("XLSX round-trip output invalid");
Console.WriteLine($"XLSX → PDF → XLSX: {xlsxBytes2.Length} bytes — PASS");

Console.WriteLine("\nAll office conversion checks passed.");
