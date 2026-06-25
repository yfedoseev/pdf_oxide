// html_extraction — build a PDF from Markdown, then extract it back to HTML.
// Shared-scenario regression example run in CI (no external fixture). Throws
// (non-zero exit) on assertion failure; prints "HTML OK" on success. Uses the
// Java-backed API through the Scala facade (`Using.resource` on AutoCloseable).
package examples

import fyi.oxide.pdf.{Pdf, PdfDocument}
import scala.util.Using

@main def htmlExtraction(): Unit =
  val md = "# Hello pdf_oxide\n\nThis is a **Scala** regression example.\n"
  Using.resource(Pdf.fromMarkdown(md)): pdf =>
    Using.resource(PdfDocument.open(pdf.save())): doc =>
      val html = doc.toHtml()
      println(html)
      assert(html.contains("<"), "html_extraction: expected HTML to contain '<'")
      assert(html.contains("pdf_oxide"), "html_extraction: expected HTML to contain 'pdf_oxide'")
      println("HTML OK")
