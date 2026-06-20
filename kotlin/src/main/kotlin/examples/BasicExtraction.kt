// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
package examples

import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument

fun main() {
    Pdf
        .fromMarkdown("# Hello pdf_oxide\n\nThis is a **Kotlin** binding smoke example.\n")
        .use { pdf ->
            PdfDocument.openFromBytes(pdf.saveToBytes()).use { doc ->
                println("pages:   ${doc.pageCount()}")
                println("version: ${doc.version()}")
                println("--- text (page 0) ---")
                println(doc.extractText(0))
                println("--- markdown (all) ---")
                println(doc.toMarkdownAll())
            }
        }
}
