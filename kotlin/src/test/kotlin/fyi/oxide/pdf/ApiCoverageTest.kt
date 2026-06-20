// Coverage for the Kotlin facade over the Java binding. Self-contained: builds
// its own PDF from Markdown, then exercises the main Java entry points plus the
// Kotlin idioms (Optional -> nullable, AutoCloseable `use`).
package fyi.oxide.pdf

import fyi.oxide.pdf.geometry.BBox
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private const val MD = "# Alpha Heading\n\nHello world from the Kotlin facade. Beta gamma delta.\n"

private fun sampleBytes(): ByteArray = Pdf.fromMarkdown(MD).use { it.save() }

class ApiCoverageTest {
    @Test fun pdfFromMarkdownAndSave() {
        val bytes = sampleBytes()
        assertTrue(bytes.size > 100, "save() should produce a non-trivial PDF")
        assertEquals('%'.code.toByte(), bytes[0]) // %PDF header
    }

    @Test fun documentOpenAndCoreExtraction() {
        PdfDocument.open(sampleBytes()).use { doc ->
            assertTrue(doc.isOpen)
            assertTrue(doc.pageCount() >= 1)
            val text = doc.extractText(0)
            assertTrue(text.contains("Hello") || text.contains("Alpha"))
            assertTrue(doc.toMarkdown().isNotEmpty())
            assertTrue(doc.toHtml().contains("<"))
        }
    }

    @Test fun pageElementExtraction() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val page = doc.page(0)
            assertTrue(page.width() > 0 && page.height() > 0)
            val words = page.words()
            assertTrue(words.isNotEmpty())
            assertTrue(words.first().text().isNotEmpty())
            val box: BBox = words.first().bbox()
            assertTrue(box.width() >= 0)
            assertNotNull(page.lines())
            assertNotNull(page.chars())
            assertNotNull(page.tables())
            assertNotNull(page.images())
            assertNotNull(page.annotations())
        }
    }

    @Test fun searchAndForms() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val matches = doc.search("Hello")
            assertTrue(matches.isNotEmpty())
            assertTrue(matches.first().text().contains("Hello"))
            assertNotNull(doc.formFields())
        }
    }

    @Test fun renderPage() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val png = doc.render(0)
            assertTrue(png.size > 100, "render() should produce PNG bytes")
        }
    }

    @Test fun metadataNullableExtensions() {
        PdfDocument.open(sampleBytes()).use { doc ->
            // Kotlin facade idiom: Optional<String> -> String? (may be null here)
            val producer: String? = doc.producerOrNull()
            val creator: String? = doc.creatorOrNull()
            // Just assert the nullable accessors are callable without throwing.
            assertTrue(producer == null || producer.isNotEmpty())
            assertTrue(creator == null || creator.isNotEmpty())
        }
    }

    @Test fun documentEditorRoundTrip() {
        DocumentEditor.open(sampleBytes()).use { ed ->
            assertTrue(ed.isOpen)
            ed.scrubMetadata()
            val out = ed.save()
            assertTrue(out.size > 100)
        }
    }

    @Test fun autoExtractor() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val auto = AutoExtractor.of(doc)
            val text = auto.extractText()
            assertTrue(text.contains("Hello") || text.contains("Alpha"))
        }
    }
}
