// Coverage for the Scala facade over the Java binding. Self-contained: builds
// its own PDF from Markdown, exercises the main Java entry points, AND invokes
// every Scala facade extension (Optional -> Option, List -> Seq) at least once —
// matching the "one test per public member" standard of the other bindings.
package fyi.oxide.pdf

import fyi.oxide.pdf.annotation.{Annotation, AnnotationType}
import fyi.oxide.pdf.compliance.ValidationViolation
import fyi.oxide.pdf.form.{FormField, FormFieldType}
import fyi.oxide.pdf.geometry.BBox
import org.scalatest.funsuite.AnyFunSuite

import java.util.Optional
import scala.util.Using

class ApiCoverageSpec extends AnyFunSuite:
  private val md = "# Alpha Heading\n\nHello world from the Scala facade. Beta gamma delta.\n"

  private def samplePdf(): Array[Byte] =
    Using.resource(Pdf.fromMarkdown(md))(_.save())

  // ── Java entry points ─────────────────────────────────────────────────────
  test("Pdf.fromMarkdown + save"):
    val bytes = samplePdf()
    assert(bytes.length > 100)
    assert(bytes(0) == '%'.toByte)

  test("PdfDocument open + core extraction"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      assert(doc.isOpen)
      assert(doc.pageCount() >= 1)
      val text = doc.extractText(0)
      assert(text.contains("Hello") || text.contains("Alpha"))
      assert(doc.toMarkdown().nonEmpty)
      assert(doc.toHtml().contains("<"))

  test("PdfPage element extraction as Seq (every *Seq extension)"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val page = doc.page(0)
      assert(page.width() > 0 && page.height() > 0)
      assert(page.wordsSeq.nonEmpty)
      assert(page.wordsSeq.head.text.nonEmpty)
      assert(page.wordsSeq.head.bbox.width >= 0)
      assert(page.linesSeq != null)
      assert(page.charsSeq != null)
      assert(page.tablesSeq != null)
      assert(page.imagesSeq != null)
      assert(page.annotationsSeq != null)

  test("PdfDocument Seq extensions (formFieldsSeq / pagesSeq / searchSeq)"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      assert(doc.formFieldsSeq != null)
      assert(doc.pagesSeq.size == doc.pageCount())
      val matches = doc.searchSeq("Hello")
      assert(matches.nonEmpty)
      assert(matches.head.text.contains("Hello"))

  test("render page"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      assert(doc.render(0).length > 100)

  test("DocumentEditor round-trip"):
    Using.resource(DocumentEditor.open(samplePdf())): ed =>
      assert(ed.isOpen)
      ed.scrubMetadata()
      assert(ed.save().length > 100)

  test("AutoExtractor + AutoResult Option/Seq extensions"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      val r = AutoExtractor.of(doc).extractDocument()
      assert(r.text.contains("Hello") || r.text.contains("Alpha"))
      assert(r.markdownOption.forall(_.nonEmpty)) // Optional -> Option
      assert(r.htmlOption.forall(_.nonEmpty))
      assert(r.pagesNeedingOcrSeq != null) // List -> Seq

  // ── Facade extensions: every Optional -> Option converter ──────────────────
  test("generic toOption"):
    assert(Optional.of("x").toOption.contains("x"))
    assert(Optional.empty[String]().toOption.isEmpty)

  test("PdfDocument metadata Option"):
    Using.resource(PdfDocument.open(samplePdf())): doc =>
      assert(doc.producerOption.forall(_.nonEmpty))
      assert(doc.creatorOption.forall(_.nonEmpty))

  test("FormField Option extensions"):
    val withValue = FormField("f", FormFieldType.TEXT, "v", BBox(0.0, 0.0, 1.0, 1.0), 0)
    assert(withValue.valueOption.contains("v"))
    assert(withValue.bboxOption.isDefined)
    val empty = FormField("g", FormFieldType.CHECKBOX, null, null, 0)
    assert(empty.valueOption.isEmpty)
    assert(empty.bboxOption.isEmpty)

  test("Annotation Option extensions"):
    val a =
      Annotation(AnnotationType.LINK, 0, BBox(0.0, 0.0, 1.0, 1.0), "note", "https://oxide.fyi")
    assert(a.contentsOption.contains("note"))
    assert(a.uriOption.contains("https://oxide.fyi"))
    val bare = Annotation(AnnotationType.TEXT, 0, BBox(0.0, 0.0, 1.0, 1.0), null, null)
    assert(bare.contentsOption.isEmpty)
    assert(bare.uriOption.isEmpty)

  test("ValidationViolation Option extension"):
    assert(ValidationViolation("RULE-1", "desc", 3).pageIndexOption.contains(3))
    assert(ValidationViolation("RULE-2", "desc", null).pageIndexOption.isEmpty)
