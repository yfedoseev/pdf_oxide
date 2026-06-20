// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
import PdfOxide

let pdf = try Pdf.fromMarkdown("# Hello pdf_oxide\n\nThis is a **Swift** binding smoke example.\n")
let doc = try Document.openFromBytes(try pdf.toBytes())

print("pages:   \(try doc.pageCount())")
print("version: \(try doc.version())")
print("--- text (page 0) ---")
print(try doc.extractText(0))
print("--- markdown (all) ---")
print(try doc.toMarkdownAll())
