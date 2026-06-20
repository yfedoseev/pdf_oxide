// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture). Uses the Java-backed API
// through the Scala facade (`Using.resource` on the AutoCloseable handles,
// `producerOption` for Optional -> Option, `wordsSeq` for List -> Seq).
package examples

import fyi.oxide.pdf.{Pdf, PdfDocument, producerOption, wordsSeq}
import scala.util.Using

@main def basicExtraction(): Unit =
  Using.resource(
    Pdf.fromMarkdown("# Hello pdf_oxide\n\nThis is a **Scala** binding smoke example.\n")
  ): pdf =>
    Using.resource(PdfDocument.open(pdf.save())): doc =>
      println(s"pages:    ${doc.pageCount()}")
      println(s"producer: ${doc.producerOption.getOrElse("(none)")}")
      println("--- text (page 0) ---")
      println(doc.extractText(0))
      println("--- markdown (all) ---")
      println(doc.toMarkdown())
      println("--- words (page 0) ---")
      doc.page(0).wordsSeq.take(8).foreach(w => println(s"  ${w.text} @ ${w.bbox}"))
