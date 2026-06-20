// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
import XCTest
@testable import PdfOxide

final class ApiCoverageTests: XCTestCase {
    private func samplePdf() throws -> [UInt8] {
        try Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n").toBytes()
    }

    // ── Pdf builder ──────────────────────────────────────────────────────────
    func testFromMarkdownAndSaveToBytes() throws {
        XCTAssertGreaterThan(try Pdf.fromMarkdown("# md\n\nbody\n").toBytes().count, 100)
    }
    func testFromHtml() throws {
        XCTAssertGreaterThan(try Pdf.fromHtml("<h1>h</h1><p>b</p>").toBytes().count, 100)
    }
    func testFromText() throws {
        XCTAssertGreaterThan(try Pdf.fromText("plain text body").toBytes().count, 100)
    }
    func testSave() throws {
        let path = NSTemporaryDirectory() + "pdfoxide_swift.pdf"
        try Pdf.fromMarkdown("# f\n\nx\n").save(path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        try? FileManager.default.removeItem(atPath: path)
    }

    // ── Document open paths ──────────────────────────────────────────────────
    func testOpenFromBytesAndPageCount() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        XCTAssertGreaterThanOrEqual(try doc.pageCount(), 1)
    }
    func testOpenPath() throws {
        let path = NSTemporaryDirectory() + "pdfoxide_swift_open.pdf"
        try Pdf.fromMarkdown("# f\n\nx\n").save(path)
        let doc = try Document.open(path)
        XCTAssertGreaterThanOrEqual(try doc.pageCount(), 1)
        try? FileManager.default.removeItem(atPath: path)
    }

    // ── Document inspection + extraction ─────────────────────────────────────
    func testInspectionAndExtraction() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        XCTAssertGreaterThanOrEqual(try doc.version().major, 1) // version
        XCTAssertFalse(try doc.isEncrypted())                  // isEncrypted
        _ = try doc.hasStructureTree()                        // hasStructureTree (smoke)
        XCTAssertTrue(try doc.extractText(0).contains("Alpha"))// extractText
        XCTAssertFalse(try doc.toPlainText(0).isEmpty)         // toPlainText
        XCTAssertFalse(try doc.toMarkdown(0).isEmpty)          // toMarkdown
        XCTAssertTrue(try doc.toHtml(0).contains("<"))         // toHtml
        XCTAssertFalse(try doc.toMarkdownAll().isEmpty)        // toMarkdownAll
        XCTAssertTrue(try doc.toHtmlAll().contains("<"))       // toHtmlAll
        XCTAssertFalse(try doc.toPlainTextAll().isEmpty)       // toPlainTextAll
        XCTAssertFalse(try doc.extractStructuredJson(0).isEmpty) // extractStructuredJson
        _ = try doc.authenticate("")                          // authenticate (returns a Bool; unencrypted)
    }

    // ── Phase-1 element extraction ───────────────────────────────────────────
    func testElementExtraction() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let words = try doc.extractWords(0)            // extractWords
        XCTAssertFalse(words.isEmpty)
        XCTAssertFalse(words[0].text.isEmpty)
        XCTAssertGreaterThan(words[0].bbox.width, 0)
        _ = words[0].fontName
        _ = words[0].fontSize
        _ = words[0].bold

        let chars = try doc.extractChars(0)            // extractChars
        XCTAssertFalse(chars.isEmpty)
        XCTAssertGreaterThan(chars[0].character, 0)
        _ = chars[0].bbox
        _ = chars[0].fontName
        _ = chars[0].fontSize

        let lines = try doc.extractTextLines(0)        // extractTextLines
        XCTAssertFalse(lines.isEmpty)
        XCTAssertFalse(lines[0].text.isEmpty)
        _ = lines[0].bbox
        _ = lines[0].wordCount

        let tables = try doc.extractTables(0)          // extractTables (may be empty)
        for table in tables {
            if table.rowCount > 0 && table.colCount > 0 {
                _ = table.cell(0, 0)
            }
            _ = table.hasHeader
        }
        XCTAssertGreaterThanOrEqual(tables.count, 0)
    }

    // ── Phase-2 element extraction ───────────────────────────────────────────
    func testPhase2Extraction() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let fonts = try doc.embeddedFonts(0)           // embeddedFonts (may be empty)
        for font in fonts {
            _ = font.name; _ = font.type; _ = font.encoding; _ = font.embedded; _ = font.subset
        }
        XCTAssertGreaterThanOrEqual(fonts.count, 0)

        let images = try doc.embeddedImages(0)         // embeddedImages (may be empty)
        for image in images {
            _ = image.width; _ = image.height; _ = image.bitsPerComponent
            _ = image.format; _ = image.colorspace; _ = image.data
        }
        XCTAssertGreaterThanOrEqual(images.count, 0)

        let annotations = try doc.pageAnnotations(0)   // pageAnnotations (may be empty)
        for ann in annotations {
            _ = ann.type; _ = ann.subtype; _ = ann.content; _ = ann.author
            _ = ann.rect; _ = ann.borderWidth
        }
        XCTAssertGreaterThanOrEqual(annotations.count, 0)

        let paths = try doc.extractPaths(0)            // extractPaths (may be empty)
        for path in paths {
            _ = path.bbox; _ = path.strokeWidth; _ = path.hasStroke
            _ = path.hasFill; _ = path.operationCount
        }
        XCTAssertGreaterThanOrEqual(paths.count, 0)
    }

    // ── Full-text search ─────────────────────────────────────────────────────
    func testSearch() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let hits = try doc.search(0, "Alpha", false)   // search
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].text.contains("Alpha"))
        XCTAssertGreaterThanOrEqual(hits[0].page, 0)
        _ = hits[0].bbox

        let allHits = try doc.searchAll("Alpha", false) // searchAll
        XCTAssertFalse(allHits.isEmpty)
        XCTAssertTrue(allHits[0].text.contains("Alpha"))
        XCTAssertGreaterThanOrEqual(allHits[0].page, 0)
        _ = allHits[0].bbox
    }

    // ── Phase-3 page rendering ───────────────────────────────────────────────
    func testRenderPage() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let img = try doc.renderPage(0)                // renderPage (PNG)
        XCTAssertGreaterThan(img.width, 0)
        XCTAssertGreaterThan(img.height, 0)
        XCTAssertFalse(img.data.isEmpty)

        // save(_:) uses the live native handle.
        let path = NSTemporaryDirectory() + "pdfoxide_swift_render.png"
        try img.save(path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        try? FileManager.default.removeItem(atPath: path)

        let zoomed = try doc.renderPageZoom(0, zoom: 2.0)  // renderPageZoom
        XCTAssertGreaterThan(zoomed.width, 0)
        XCTAssertGreaterThan(zoomed.height, 0)
        XCTAssertFalse(zoomed.data.isEmpty)

        let thumb = try doc.renderPageThumbnail(0, size: 64) // renderPageThumbnail
        XCTAssertGreaterThan(thumb.width, 0)
        XCTAssertGreaterThan(thumb.height, 0)
        XCTAssertFalse(thumb.data.isEmpty)
    }

    // ── Page model ───────────────────────────────────────────────────────────
    func testPage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let page = doc.page(0)
        XCTAssertTrue(try page.text().contains("Alpha")) // text
        XCTAssertFalse(try page.markdown().isEmpty)      // markdown
        XCTAssertTrue(try page.html().contains("<"))     // html
        XCTAssertFalse(try page.plainText().isEmpty)     // plainText
    }

    // ── close() is idempotent; use-after-close throws ───────────────────────
    func testClose() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        doc.close()
        doc.close() // idempotent
        XCTAssertThrowsError(try doc.pageCount()) { error in
            XCTAssertTrue(error is PdfOxideError)
        }
    }

    // ── Error path ───────────────────────────────────────────────────────────
    func testErrorOnMissingFile() {
        XCTAssertThrowsError(try Document.open("/nonexistent/nope.pdf")) { error in
            XCTAssertTrue(error is PdfOxideError)
        }
    }
}
