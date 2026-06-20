// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
package examples

import fyi.oxide.pdf.{Pdf, PdfDocument}
import scala.util.Using

@main def basicExtraction(): Unit =
  Using.resource(
    Pdf.fromMarkdown("# Hello pdf_oxide\n\nThis is a **Scala** binding smoke example.\n")
  ): pdf =>
    Using.resource(PdfDocument.openFromBytes(pdf.saveToBytes())): doc =>
      println(s"pages:   ${doc.pageCount()}")
      println(s"version: ${doc.version()}")
      println("--- text (page 0) ---")
      println(doc.extractText(0))
      println("--- markdown (all) ---")
      println(doc.toMarkdownAll())
