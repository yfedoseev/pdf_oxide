// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture). Uses the Java-backed API
// through the Kotlin facade (note `use { }` on the AutoCloseable handles).
package examples

import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument
import fyi.oxide.pdf.producerOrNull

fun main() {
    Pdf
        .fromMarkdown("# Hello pdf_oxide\n\nThis is a **Kotlin** binding smoke example.\n")
        .use { pdf ->
            PdfDocument.open(pdf.save()).use { doc ->
                println("pages:    ${doc.pageCount()}")
                println("producer: ${doc.producerOrNull() ?: "(none)"}")
                println("--- text (page 0) ---")
                println(doc.extractText(0))
                println("--- markdown (all) ---")
                println(doc.toMarkdown())
                println("--- words (page 0) ---")
                doc
                    .page(0)
                    .words()
                    .take(8)
                    .forEach { println("  ${it.text()} @ ${it.bbox()}") }
            }
        }
}
