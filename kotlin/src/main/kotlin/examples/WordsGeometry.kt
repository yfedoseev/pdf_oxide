// words_geometry — build a PDF from Markdown, then extract word geometry.
// Shared-scenario regression example run in CI (no external fixture). Exits
// non-zero on assertion failure; prints "WORDS OK" on success. Uses the Java-
// backed API through the Kotlin facade (`use { }` on AutoCloseable handles).
package examples

import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument

private const val MD = "# Hello pdf_oxide\n\nThis is a **Kotlin** regression example.\n"

fun main() {
    Pdf.fromMarkdown(MD).use { pdf ->
        PdfDocument.open(pdf.save()).use { doc ->
            val words = doc.page(0).words()
            check(words.isNotEmpty()) { "words_geometry: expected at least one word" }
            val first = words.first()
            println("words: ${words.size}; first: '${first.text()}' @ ${first.bbox()}")
            check(first.text() == "Hello") { "words_geometry: expected first word 'Hello', got '${first.text()}'" }
            check(first.bbox().width() >= 0.0) { "words_geometry: expected first word to have a bbox" }
            println("WORDS OK")
        }
    }
}
