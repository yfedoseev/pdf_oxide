// words_geometry — build a PDF from Markdown, then extract positioned words.
// Shared-scenario regression example. Exits non-zero on assertion failure.
import Foundation
import PdfOxide

let md = "# Hello pdf_oxide\n\nThis is a **Swift** regression example.\n"
let pdf = try Pdf.fromMarkdown(md)
let doc = try Document.openFromBytes(try pdf.toBytes())

let words = try doc.extractWords(0)
guard !words.isEmpty else {
    FileHandle.standardError.write(Data("assertion failed: no words extracted\n".utf8))
    exit(1)
}
let first = words[0]
print("words: \(words.count)")
print("first word: \"\(first.text)\"")
print("bbox: x=\(first.bbox.x) y=\(first.bbox.y) w=\(first.bbox.width) h=\(first.bbox.height)")

guard first.text == "Hello" else {
    FileHandle.standardError.write(Data("assertion failed: first word != \"Hello\"\n".utf8))
    exit(1)
}
guard first.bbox.width > 0, first.bbox.height > 0 else {
    FileHandle.standardError.write(Data("assertion failed: first word has no bbox\n".utf8))
    exit(1)
}
print("WORDS OK")
