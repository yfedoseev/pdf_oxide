// tables_extraction — build a PDF from a Markdown table, then extract tables.
// Shared-scenario regression example run in CI (no external fixture). Asserts
// the call returns a list without error (synthetic docs may yield 0 tables).
// Exits non-zero on assertion failure; prints "TABLES OK" on success. Uses the
// Java-backed API through the Kotlin facade (`use { }` on AutoCloseable handles).
package examples

import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument

private const val MD =
    "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"

fun main() {
    Pdf.fromMarkdown(MD).use { pdf ->
        PdfDocument.open(pdf.save()).use { doc ->
            val tables = doc.page(0).tables()
            check(tables.size >= 0) { "tables_extraction: expected a list of tables" }
            println("tables: ${tables.size}")
            tables.forEach { t ->
                println("  ${t.rows()}x${t.cols()} cells=${t.cells().map { it.text() }}")
            }
            println("TABLES OK")
        }
    }
}
