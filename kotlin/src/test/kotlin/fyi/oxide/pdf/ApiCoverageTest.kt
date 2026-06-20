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

    // ── Phase-1 element extraction ───────────────────────────────────────────
    @Test fun extractWords() {
        val words = doc.extractWords(0)
        assertTrue(words.isNotEmpty())
        val w = words[0]
        assertTrue(w.text.isNotEmpty())
        assertTrue(w.bbox.width >= 0f && w.bbox.height >= 0f)
    }

    @Test fun extractChars() = assertTrue(doc.extractChars(0).isNotEmpty())

    @Test fun extractTextLines() = assertTrue(doc.extractTextLines(0).isNotEmpty())

    @Test fun extractTables() {
        val tables = doc.extractTables(0) // may be empty; must not throw
        assertTrue(tables.size >= 0)
    }

    // ── Phase-2 element extraction ───────────────────────────────────────────
    @Test fun embeddedFonts() {
        val fonts = doc.embeddedFonts(0) // may be empty; must not throw
        assertTrue(fonts.size >= 0)
    }

    @Test fun embeddedImages() {
        val images = doc.embeddedImages(0) // may be empty; must not throw
        assertTrue(images.size >= 0)
    }

    @Test fun pageAnnotations() {
        val annotations = doc.pageAnnotations(0) // may be empty; must not throw
        assertTrue(annotations.size >= 0)
    }

    @Test fun extractPaths() {
        val paths = doc.extractPaths(0) // may be empty; must not throw
        assertTrue(paths.size >= 0)
    }

    @Test fun search() {
        val hits = doc.search(0, "Alpha", false)
        assertTrue(hits.isNotEmpty())
        assertContains(hits[0].text, "Alpha")
        assertTrue(hits[0].page >= 0)
    }

    @Test fun searchAll() {
        val hits = doc.searchAll("Alpha", false)
        assertTrue(hits.isNotEmpty())
        assertContains(hits[0].text, "Alpha")
        assertTrue(hits[0].page >= 0)
    }

    // ── Phase-3 page rendering ───────────────────────────────────────────────
    @Test fun renderPage() =
        doc.renderPage(0).use { img ->
            assertTrue(img.width > 0)
            assertTrue(img.height > 0)
            assertTrue(img.data.isNotEmpty())
        }

    @Test fun renderPageZoom() =
        doc.renderPageZoom(0, 2.0f).use { img ->
            assertTrue(img.width > 0)
            assertTrue(img.height > 0)
            assertTrue(img.data.isNotEmpty())
        }

    @Test fun renderPageThumbnail() =
        doc.renderPageThumbnail(0, 128).use { img ->
            assertTrue(img.width > 0)
            assertTrue(img.height > 0)
            assertTrue(img.data.isNotEmpty())
        }

    @Test fun renderedImageSave() {
        val f = File.createTempFile("pdfoxide-kt-render", ".png")
        doc.renderPage(0).use { it.save(f.absolutePath) }
        assertTrue(f.length() > 0)
        f.delete()
    }

    // ── Coroutine helpers ────────────────────────────────────────────────────
    @Test fun coroutineExtraction() =
        runTest {
            assertContains(doc.extractTextAsync(0), "Alpha")
            assertTrue(doc.toMarkdownAsync(0).isNotEmpty())
            assertTrue(doc.toMarkdownAllAsync().isNotEmpty())
        }

    // ── DocumentEditor ───────────────────────────────────────────────────────
    @Test fun documentEditor() =
        DocumentEditor.openFromBytes(samplePdf()).use { editor ->
            assertTrue(editor.pageCount() >= 1)
            val modified: Boolean = editor.isModified()
            assertTrue(modified || !modified) // returns a bool without raising
            editor.rotateAllPages(90)
            val rotation = editor.getPageRotation(0)
            assertEquals(90, rotation)
            editor.setProducer("x")
            assertEquals("x", editor.getProducer())
            assertTrue(editor.saveToBytes().isNotEmpty())
        }

    // ── Error path ───────────────────────────────────────────────────────────
    @Test fun errorOnMissingFile() {
        assertFailsWith<PdfOxideException> { PdfDocument.open("/nonexistent/nope.pdf") }
    }
}
