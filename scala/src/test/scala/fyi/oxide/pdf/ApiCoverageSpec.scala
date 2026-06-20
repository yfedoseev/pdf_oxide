// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
package fyi.oxide.pdf

import org.scalatest.funsuite.AnyFunSuite
import java.io.File
import scala.util.Using

class ApiCoverageSpec extends AnyFunSuite:
  private def samplePdf(): Array[Byte] =
    Using.resource(
      Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")
    )(_.toBytes())

  // ── Pdf builder ────────────────────────────────────────────────────────────
  test("fromMarkdown + toBytes"):
    Using.resource(Pdf.fromMarkdown("# md\n\nbody\n"))(p => assert(p.toBytes().length > 100))
  test("fromHtml"):
    Using.resource(Pdf.fromHtml("<h1>h</h1><p>b</p>"))(p => assert(p.toBytes().length > 100))
  test("fromText"):
    Using.resource(Pdf.fromText("plain text body"))(p => assert(p.toBytes().length > 100))
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
      assert(doc.version().major >= 1) // version
      assert(!doc.isEncrypted()) // isEncrypted
      doc.hasStructureTree() // hasStructureTree (smoke)
      assert(doc.extractText(0).contains("Alpha")) // extractText
      assert(doc.toPlainText(0).nonEmpty) // toPlainText
      assert(doc.toMarkdown(0).nonEmpty) // toMarkdown
      assert(doc.toHtml(0).contains("<")) // toHtml
      assert(doc.toMarkdownAll().nonEmpty) // toMarkdownAll
      assert(doc.toHtmlAll().contains("<")) // toHtmlAll
      assert(doc.toPlainTextAll().nonEmpty) // toPlainTextAll
      assert(doc.extractStructuredJson(0).nonEmpty) // extractStructuredJson

  // ── Phase-1 element extraction ───────────────────────────────────────────────
  test("extractWords / extractChars / extractTextLines / extractTables"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      val words = doc.extractWords(0) // extractWords
      assert(words.nonEmpty)
      assert(words.head.text.nonEmpty) // word[0].text non-empty
      val b = words.head.bbox // word[0] has a bbox
      assert(b.width >= 0 && b.height >= 0)
      assert(doc.extractChars(0).nonEmpty) // extractChars
      assert(doc.extractTextLines(0).nonEmpty) // extractTextLines
      val tables = doc.extractTables(0) // extractTables (may be empty)
      assert(tables ne null)

  // ── Phase-2 element extraction ───────────────────────────────────────────────
  test("embeddedFonts / embeddedImages / pageAnnotations / extractPaths"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      val fonts = doc.embeddedFonts(0) // embeddedFonts (may be empty)
      assert(fonts ne null)
      val images = doc.embeddedImages(0) // embeddedImages (may be empty)
      assert(images ne null)
      val annots = doc.pageAnnotations(0) // pageAnnotations (may be empty)
      assert(annots ne null)
      val paths = doc.extractPaths(0) // extractPaths (may be empty)
      assert(paths ne null)

  test("search / searchAll"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      val hits = doc.search(0, "Alpha", false) // search
      assert(hits.nonEmpty)
      assert(hits.head.text.contains("Alpha")) // first hit text contains Alpha
      assert(hits.head.page >= 0) // page >= 0
      val allHits = doc.searchAll("Alpha", false) // searchAll
      assert(allHits.nonEmpty)
      assert(allHits.head.text.contains("Alpha"))
      assert(allHits.head.page >= 0)

  // ── authenticate ─────────────────────────────────────────────────────────────
  test("authenticate"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      val ok: Boolean = doc.authenticate("anything") // returns a bool without error
      assert(ok == true || ok == false)

  // ── Phase-3 page rendering ───────────────────────────────────────────────────
  test("renderPage / renderPageZoom / renderPageThumbnail"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      Using.resource(doc.renderPage(0)): img => // renderPage (PNG, 0-based)
        assert(img.width > 0) // width > 0
        assert(img.height > 0) // height > 0
        assert(img.data.nonEmpty) // non-empty encoded image data
      Using.resource(doc.renderPageZoom(0, 2.0f)): img => // renderPageZoom
        assert(img.width > 0 && img.height > 0)
      Using.resource(doc.renderPageThumbnail(0, 64)): img => // renderPageThumbnail
        assert(img.width > 0 && img.height > 0)

  test("RenderedImage.save"):
    val f = File.createTempFile("pdfoxide-scala-render", ".png")
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      Using.resource(doc.renderPage(0))(_.save(f.getAbsolutePath))
    assert(f.length() > 0); f.delete()

  // ── Page model ───────────────────────────────────────────────────────────────
  test("page model"):
    Using.resource(PdfDocument.openFromBytes(samplePdf())): doc =>
      val page = doc.page(0) // page(index)
      assert(page.text().contains("Alpha")) // text
      assert(page.markdown().nonEmpty) // markdown
      assert(page.html().nonEmpty) // html
      assert(page.plainText().nonEmpty) // plainText

  // ── Error path ───────────────────────────────────────────────────────────────
  test("open nonexistent throws PdfOxideException"):
    assertThrows[PdfOxideException](PdfDocument.open("/nonexistent/nope.pdf"))
