// words_geometry — build a PDF from Markdown, then extract word geometry.
// Shared-scenario regression example run in CI (no external fixture). Throws
// (non-zero exit) on assertion failure; prints "WORDS OK" on success. Uses the
// Java-backed API through the Scala facade (`wordsSeq` for List -> Seq).
package examples

import fyi.oxide.pdf.{Pdf, PdfDocument, wordsSeq}
import scala.util.Using

@main def wordsGeometry(): Unit =
  val md = "# Hello pdf_oxide\n\nThis is a **Scala** regression example.\n"
  Using.resource(Pdf.fromMarkdown(md)): pdf =>
    Using.resource(PdfDocument.open(pdf.save())): doc =>
      val words = doc.page(0).wordsSeq
      assert(words.nonEmpty, "words_geometry: expected at least one word")
      val first = words.head
      println(s"words: ${words.size}; first: '${first.text}' @ ${first.bbox}")
      assert(
        first.text == "Hello",
        s"words_geometry: expected first word 'Hello', got '${first.text}'"
      )
      assert(first.bbox.width >= 0.0, "words_geometry: expected first word to have a bbox")
      println("WORDS OK")
