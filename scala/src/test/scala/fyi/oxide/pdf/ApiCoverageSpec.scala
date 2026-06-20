// Coverage for the Scala facade over the Java binding. Self-contained: builds
// its own PDF from Markdown, then exercises the main Java entry points plus the
// Scala idioms (Optional -> Option, List -> Seq, Using on AutoCloseable).
package fyi.oxide.pdf

import org.scalatest.funsuite.AnyFunSuite
import scala.util.Using

class ApiCoverageSpec extends AnyFunSuite:
  private val md = "# Alpha Heading\n\nHello world from the Scala facade. Beta gamma delta.\n"

  private def samplePdf(): Array[Byte] =
    Using.resource(Pdf.fromMarkdown(md))(_.save())

  test("Pdf.fromMarkdown + save"):
    val bytes = samplePdf()
    assert(bytes.length > 100)
    assert(bytes(0) == '%'.toByte) // %PDF header

  test("PdfDocument open + core extraction"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      assert(doc.isOpen)
      assert(doc.pageCount() >= 1)
      val text = doc.extractText(0)
      assert(text.contains("Hello") || text.contains("Alpha"))
      assert(doc.toMarkdown().nonEmpty)
      assert(doc.toHtml().contains("<"))

  test("PdfPage element extraction as Seq"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val page = doc.page(0)
      assert(page.width() > 0 && page.height() > 0)
      val words = page.wordsSeq
      assert(words.nonEmpty)
      assert(words.head.text.nonEmpty)
      assert(words.head.bbox.width >= 0)
      assert(page.linesSeq != null)
      assert(page.charsSeq != null)
      assert(page.tablesSeq != null)
      assert(page.imagesSeq != null)
      assert(page.annotationsSeq != null)

  test("search + forms"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val matches = doc.searchSeq("Hello")
      assert(matches.nonEmpty)
      assert(matches.head.text.contains("Hello"))
      assert(doc.formFieldsSeq != null)

  test("render page"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val png = doc.render(0)
      assert(png.length > 100)

  test("metadata Optional -> Option"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val producer: Option[String] = doc.producerOption
      val creator: Option[String] = doc.creatorOption
      assert(producer.forall(_.nonEmpty))
      assert(creator.forall(_.nonEmpty))

  test("DocumentEditor round-trip"):
    Using.resource(DocumentEditor.open(samplePdf())): ed =>
      assert(ed.isOpen)
      ed.scrubMetadata()
      val out = ed.save()
      assert(out.length > 100)

  test("AutoExtractor"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val auto = AutoExtractor.of(doc)
      val text = auto.extractText()
      assert(text.contains("Hello") || text.contains("Alpha"))
