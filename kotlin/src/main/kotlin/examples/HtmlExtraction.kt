// html_extraction — build a PDF from Markdown, then extract it back to HTML.
// Shared-scenario regression example run in CI (no external fixture). Exits
// non-zero on assertion failure; prints "HTML OK" on success. Uses the Java-
// backed API through the Kotlin facade (`use { }` on AutoCloseable handles).
package examples

import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument

private const val MD = "# Hello pdf_oxide\n\nThis is a **Kotlin** regression example.\n"

fun main() {
    Pdf.fromMarkdown(MD).use { pdf ->
        PdfDocument.open(pdf.save()).use { doc ->
            val html = doc.toHtml()
            println(html)
            check(html.contains("<")) { "html_extraction: expected HTML to contain '<'" }
            check(html.contains("pdf_oxide")) { "html_extraction: expected HTML to contain 'pdf_oxide'" }
            println("HTML OK")
        }
    }
}
