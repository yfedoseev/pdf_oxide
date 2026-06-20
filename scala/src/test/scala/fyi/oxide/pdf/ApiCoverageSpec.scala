// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
package fyi.oxide.pdf

import org.scalatest.funsuite.AnyFunSuite
import java.io.File
import scala.util.Using

class ApiCoverageSpec extends AnyFunSuite:
  private def samplePdf(): Array[Byte] =
    Using.resource(Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n"))(_.saveToBytes())

  // ── Pdf builder ────────────────────────────────────────────────────────────
  test("fromMarkdown + saveToBytes"):
    Using.resource(Pdf.fromMarkdown("# md\n\nbody\n"))(p => assert(p.saveToBytes().length > 100))
  test("fromHtml"):
    Using.resource(Pdf.fromHtml("<h1>h</h1><p>b</p>"))(p => assert(p.saveToBytes().length > 100))
  test("fromText"):
    Using.resource(Pdf.fromText("plain text body"))(p => assert(p.saveToBytes().length > 100))
  test("save"):
    val f = File.createTempFile("pdfoxide-scala", ".pdf")
    Using.resource(Pdf.fromMarkdown("# f\n\nx\n"))(_.save(f.getAbsolutePath))
    assert(f.length() > 100); f.delete()

  // ── Document open paths ──────────────────────────────────────────────────────
  test("openFromBytes + pageCount"):
    Using.resource(PdfDocument.openFromBytes(samplePdf()))(d => assert(d.pageCount() >= 1))
  test("open (path)"):
    val f = File.createTempFile("pdfoxide-scala-open", ".pdf")
    Using.resource(Pdf.fromMarkdown("# f\n\nx\n"))(_.save(f.getAbsolutePath))
    Using.resource(PdfDocument.open(f.getAbsolutePath))(d => assert(d.pageCount() >= 1)); f.delete()

  // ── Document inspection + extraction ─────────────────────────────────────────
  test("inspection + extraction"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      assert(doc.version().major >= 1)                 // version
      assert(!doc.isEncrypted())                        // isEncrypted
      doc.hasStructureTree()                            // hasStructureTree (smoke)
      assert(doc.extractText(0).contains("Alpha"))      // extractText
      assert(doc.toPlainText(0).nonEmpty)               // toPlainText
      assert(doc.toMarkdown(0).nonEmpty)                // toMarkdown
      assert(doc.toHtml(0).contains("<"))               // toHtml
      assert(doc.toMarkdownAll().nonEmpty)              // toMarkdownAll
      assert(doc.extractStructuredJson(0).nonEmpty)     // extractStructuredJson

  // ── Error path ───────────────────────────────────────────────────────────────
  test("open nonexistent throws PdfOxideException"):
    assertThrows[PdfOxideException](PdfDocument.open("/nonexistent/nope.pdf"))
