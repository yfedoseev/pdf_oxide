// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
package fyi.oxide.pdf

import kotlinx.coroutines.test.runTest
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ApiCoverageTest {
    private fun samplePdf(): ByteArray =
        Pdf
            .fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")
            .use { it.toBytes() }

    private lateinit var doc: PdfDocument

    @BeforeTest fun setUp() {
        doc = PdfDocument.openFromBytes(samplePdf())
    }

    @AfterTest fun tearDown() {
        doc.close()
    }

    // ── Pdf builder ──────────────────────────────────────────────────────────
    @Test fun fromMarkdownAndSaveToBytes() = Pdf.fromMarkdown("# md\n\nbody\n").use { assertTrue(it.toBytes().size > 100) }

    @Test fun fromHtml() = Pdf.fromHtml("<h1>html</h1><p>body</p>").use { assertTrue(it.toBytes().size > 100) }

    @Test fun fromText() = Pdf.fromText("plain text body").use { assertTrue(it.toBytes().size > 100) }

    @Test fun save() {
        val f = File.createTempFile("pdfoxide-kt", ".pdf")
        Pdf.fromMarkdown("# f\n\nx\n").use { it.save(f.absolutePath) }
        assertTrue(f.length() > 100)
        f.delete()
    }

    // ── Document open paths ──────────────────────────────────────────────────
    @Test fun openFromBytesAndPageCount() = assertTrue(doc.pageCount() >= 1)

    @Test fun openPath() {
        val f = File.createTempFile("pdfoxide-kt-open", ".pdf")
        Pdf.fromMarkdown("# f\n\nx\n").use { it.save(f.absolutePath) }
        PdfDocument.open(f.absolutePath).use { assertTrue(it.pageCount() >= 1) }
        f.delete()
    }

    // ── Document inspection + extraction ─────────────────────────────────────
    @Test fun version() = assertTrue(doc.version().major >= 1)

    @Test fun isEncrypted() = assertEquals(false, doc.isEncrypted())

    @Test fun hasStructureTree() {
        doc.hasStructureTree()
    } // smoke

    @Test fun extractText() = assertContains(doc.extractText(0), "Alpha")

    @Test fun toPlainText() = assertTrue(doc.toPlainText(0).isNotEmpty())

    @Test fun toMarkdown() = assertTrue(doc.toMarkdown(0).isNotEmpty())

    @Test fun toHtml() = assertContains(doc.toHtml(0), "<")

    @Test fun toMarkdownAll() = assertTrue(doc.toMarkdownAll().isNotEmpty())

    @Test fun toHtmlAll() = assertContains(doc.toHtmlAll(), "<")

    @Test fun toPlainTextAll() = assertTrue(doc.toPlainTextAll().isNotEmpty())

    @Test fun authenticate() {
        val ok: Boolean = doc.authenticate("")
        assertTrue(ok || !ok) // returns a bool without raising
    }

    // ── Page model ───────────────────────────────────────────────────────────
    @Test fun pageText() = assertContains(doc.page(0).text(), "Alpha")

    @Test fun pageMarkdown() = assertTrue(doc.page(0).markdown().isNotEmpty())

    @Test fun pageHtml() = assertContains(doc.page(0).html(), "<")

    @Test fun pagePlainText() = assertTrue(doc.page(0).plainText().isNotEmpty())

    @Test fun extractStructuredJson() = assertTrue(doc.extractStructuredJson(0).isNotEmpty())

    // ── Coroutine helpers ────────────────────────────────────────────────────
    @Test fun coroutineExtraction() =
        runTest {
            assertContains(doc.extractTextAsync(0), "Alpha")
            assertTrue(doc.toMarkdownAsync(0).isNotEmpty())
            assertTrue(doc.toMarkdownAllAsync().isNotEmpty())
        }

    // ── Error path ───────────────────────────────────────────────────────────
    @Test fun errorOnMissingFile() {
        assertFailsWith<PdfOxideException> { PdfDocument.open("/nonexistent/nope.pdf") }
    }
}
