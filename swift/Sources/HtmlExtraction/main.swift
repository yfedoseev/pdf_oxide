// html_extraction — build a PDF from Markdown, then render it back to HTML.
// Shared-scenario regression example. Exits non-zero on assertion failure.
import Foundation
import PdfOxide

let md = "# Hello pdf_oxide\n\nThis is a **Swift** regression example.\n"
let pdf = try Pdf.fromMarkdown(md)
let doc = try Document.openFromBytes(try pdf.toBytes())

let html = try doc.toHtmlAll()
print(html)

guard html.contains("<") else {
    FileHandle.standardError.write(Data("assertion failed: html missing '<'\n".utf8))
    exit(1)
}
guard html.contains("pdf_oxide") else {
    FileHandle.standardError.write(Data("assertion failed: html missing 'pdf_oxide'\n".utf8))
    exit(1)
}
print("HTML OK")
