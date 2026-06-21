// Coverage for the Kotlin facade over the Java binding. Self-contained: builds
// its own PDF from Markdown, exercises the main Java entry points, AND invokes
// every Kotlin facade extension (Optional -> nullable) at least once — matching
// the "one test per public member" standard of the other pdf_oxide bindings.
package fyi.oxide.pdf

import fyi.oxide.pdf.annotation.Annotation
import fyi.oxide.pdf.annotation.AnnotationType
import fyi.oxide.pdf.form.FormField
import fyi.oxide.pdf.form.FormFieldType
import fyi.oxide.pdf.geometry.BBox
import java.util.Optional
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

private const val MD = "# Alpha Heading\n\nHello world from the Kotlin facade. Beta gamma delta.\n"

private fun sampleBytes(): ByteArray = Pdf.fromMarkdown(MD).use { it.save() }

class ApiCoverageTest {
    // ── Java entry points ───────────────────────────────────────────────────
    @Test fun pdfFromMarkdownAndSave() {
        val bytes = sampleBytes()
        assertTrue(bytes.size > 100)
        assertEquals('%'.code.toByte(), bytes[0])
    }

    @Test fun documentOpenAndCoreExtraction() {
        PdfDocument.open(sampleBytes()).use { doc ->
            assertTrue(doc.isOpen)
            assertTrue(doc.pageCount() >= 1)
            assertTrue(doc.extractText(0).let { it.contains("Hello") || it.contains("Alpha") })
            assertTrue(doc.toMarkdown().isNotEmpty())
            assertTrue(doc.toHtml().contains("<"))
        }
    }

    @Test fun pageElementExtraction() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val page = doc.page(0)
            assertTrue(page.width() > 0 && page.height() > 0)
            assertTrue(page.words().isNotEmpty())
            assertTrue(
                page
                    .words()
                    .first()
                    .text()
                    .isNotEmpty(),
            )
            assertTrue(
                page
                    .words()
                    .first()
                    .bbox()
                    .width() >= 0,
            )
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
            assertTrue(doc.render(0).size > 100)
        }
    }

    @Test fun documentEditorRoundTrip() {
        DocumentEditor.open(sampleBytes()).use { ed ->
            assertTrue(ed.isOpen)
            ed.scrubMetadata()
            assertTrue(ed.save().size > 100)
        }
    }

    @Test fun autoExtractorAndAutoResultExtensions() {
        PdfDocument.open(sampleBytes()).use { doc ->
            val result = AutoExtractor.of(doc).extractDocument()
            assertTrue(result.text().let { it.contains("Hello") || it.contains("Alpha") })
            // AutoResult.markdownOrNull / htmlOrNull (Optional -> nullable)
            val md: String? = result.markdownOrNull()
            val html: String? = result.htmlOrNull()
            assertTrue(md == null || md.isNotEmpty())
            assertTrue(html == null || html.isNotEmpty())
        }
    }

    // ── Facade extensions: every Optional -> nullable converter ──────────────
    @Test fun genericOrNull() {
        assertEquals("x", Optional.of("x").orNull())
        assertNull(Optional.empty<String>().orNull())
    }

    @Test fun documentMetadataOrNull() {
        PdfDocument.open(sampleBytes()).use { doc ->
            assertTrue(doc.producerOrNull().let { it == null || it.isNotEmpty() })
            assertTrue(doc.creatorOrNull().let { it == null || it.isNotEmpty() })
        }
    }

    @Test fun formFieldOrNull() {
        // No form fields in the sample, so construct the reachable value type.
        val withValue = FormField("f", FormFieldType.TEXT, "v", BBox(0.0, 0.0, 1.0, 1.0), 0)
        assertEquals("v", withValue.valueOrNull())
        assertNotNull(withValue.bboxOrNull())
        val empty = FormField("g", FormFieldType.CHECKBOX, null, null, 0)
        assertNull(empty.valueOrNull())
        assertNull(empty.bboxOrNull())
    }

    @Test fun annotationOrNull() {
        val a = Annotation(AnnotationType.LINK, 0, BBox(0.0, 0.0, 1.0, 1.0), "note", "https://oxide.fyi")
        assertEquals("note", a.contentsOrNull())
        assertEquals("https://oxide.fyi", a.uriOrNull())
        val bare = Annotation(AnnotationType.TEXT, 0, BBox(0.0, 0.0, 1.0, 1.0), null, null)
        assertNull(bare.contentsOrNull())
        assertNull(bare.uriOrNull())
    }

    @Test fun validationViolationOrNull() {
        val v =
            fyi.oxide.pdf.compliance
                .ValidationViolation("RULE-1", "desc", 3)
        assertEquals(3, v.pageIndexOrNull())
        val docLevel =
            fyi.oxide.pdf.compliance
                .ValidationViolation("RULE-2", "desc", null)
        assertNull(docLevel.pageIndexOrNull())
    }
}
