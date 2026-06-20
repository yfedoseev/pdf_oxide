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
        XCTAssertFalse(try doc.extractStructuredJson(0).isEmpty) // extractStructuredJson
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
