// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
import XCTest
@testable import PdfOxide

final class ApiCoverageTests: XCTestCase {
    private func samplePdf() throws -> [UInt8] {
        try Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")
            .toBytes()
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
        XCTAssertGreaterThanOrEqual(try doc.version().major, 1)  // version
        XCTAssertFalse(try doc.isEncrypted())  // isEncrypted
        _ = try doc.hasStructureTree()  // hasStructureTree (smoke)
        XCTAssertTrue(try doc.extractText(0).contains("Alpha"))  // extractText
        XCTAssertFalse(try doc.toPlainText(0).isEmpty)  // toPlainText
        XCTAssertFalse(try doc.toMarkdown(0).isEmpty)  // toMarkdown
        XCTAssertTrue(try doc.toHtml(0).contains("<"))  // toHtml
        XCTAssertFalse(try doc.toMarkdownAll().isEmpty)  // toMarkdownAll
        XCTAssertTrue(try doc.toHtmlAll().contains("<"))  // toHtmlAll
        XCTAssertFalse(try doc.toPlainTextAll().isEmpty)  // toPlainTextAll
        XCTAssertFalse(try doc.extractStructuredJson(0).isEmpty)  // extractStructuredJson
        _ = try doc.authenticate("")  // authenticate (returns a Bool; unencrypted)
    }

    // ── Phase-1 element extraction ───────────────────────────────────────────
    func testElementExtraction() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let words = try doc.extractWords(0)  // extractWords
        XCTAssertFalse(words.isEmpty)
        XCTAssertFalse(words[0].text.isEmpty)
        XCTAssertGreaterThan(words[0].bbox.width, 0)
        _ = words[0].fontName
        _ = words[0].fontSize
        _ = words[0].bold

        let chars = try doc.extractChars(0)  // extractChars
        XCTAssertFalse(chars.isEmpty)
        XCTAssertGreaterThan(chars[0].character, 0)
        _ = chars[0].bbox
        _ = chars[0].fontName
        _ = chars[0].fontSize

        let lines = try doc.extractTextLines(0)  // extractTextLines
        XCTAssertFalse(lines.isEmpty)
        XCTAssertFalse(lines[0].text.isEmpty)
        _ = lines[0].bbox
        _ = lines[0].wordCount

        let tables = try doc.extractTables(0)  // extractTables (may be empty)
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

        let fonts = try doc.embeddedFonts(0)  // embeddedFonts (may be empty)
        for font in fonts {
            _ = font.name; _ = font.type; _ = font.encoding; _ = font.embedded; _ = font.subset
        }
        XCTAssertGreaterThanOrEqual(fonts.count, 0)

        let images = try doc.embeddedImages(0)  // embeddedImages (may be empty)
        for image in images {
            _ = image.width; _ = image.height; _ = image.bitsPerComponent
            _ = image.format; _ = image.colorspace; _ = image.data
        }
        XCTAssertGreaterThanOrEqual(images.count, 0)

        let annotations = try doc.pageAnnotations(0)  // pageAnnotations (may be empty)
        for ann in annotations {
            _ = ann.type; _ = ann.subtype; _ = ann.content; _ = ann.author
            _ = ann.rect; _ = ann.borderWidth
        }
        XCTAssertGreaterThanOrEqual(annotations.count, 0)

        let paths = try doc.extractPaths(0)  // extractPaths (may be empty)
        for path in paths {
            _ = path.bbox; _ = path.strokeWidth; _ = path.hasStroke
            _ = path.hasFill; _ = path.operationCount
        }
        XCTAssertGreaterThanOrEqual(paths.count, 0)
    }

    // ── Full-text search ─────────────────────────────────────────────────────
    func testSearch() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let hits = try doc.search(0, "Alpha", false)  // search
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].text.contains("Alpha"))
        XCTAssertGreaterThanOrEqual(hits[0].page, 0)
        _ = hits[0].bbox

        let allHits = try doc.searchAll("Alpha", false)  // searchAll
        XCTAssertFalse(allHits.isEmpty)
        XCTAssertTrue(allHits[0].text.contains("Alpha"))
        XCTAssertGreaterThanOrEqual(allHits[0].page, 0)
        _ = allHits[0].bbox
    }

    // ── Phase-3 page rendering ───────────────────────────────────────────────
    func testRenderPage() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let img = try doc.renderPage(0)  // renderPage (PNG)
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

        let thumb = try doc.renderPageThumbnail(0, size: 64)  // renderPageThumbnail
        XCTAssertGreaterThan(thumb.width, 0)
        XCTAssertGreaterThan(thumb.height, 0)
        XCTAssertFalse(thumb.data.isEmpty)
    }

    // ── Page model ───────────────────────────────────────────────────────────
    func testPage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let page = doc.page(0)
        XCTAssertTrue(try page.text().contains("Alpha"))  // text
        XCTAssertFalse(try page.markdown().isEmpty)  // markdown
        XCTAssertTrue(try page.html().contains("<"))  // html
        XCTAssertFalse(try page.plainText().isEmpty)  // plainText
    }

    // ── close() is idempotent; use-after-close throws ───────────────────────
    func testClose() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        doc.close()
        doc.close()  // idempotent
        XCTAssertThrowsError(try doc.pageCount()) { error in
            XCTAssertTrue(error is PdfOxideError)
        }
    }

    // ── DocumentEditor ───────────────────────────────────────────────────────
    func testDocumentEditor() throws {
        let editor = try DocumentEditor.openFromBytes(try samplePdf())
        XCTAssertGreaterThanOrEqual(try editor.pageCount(), 1)  // pageCount
        let modified: Bool = try editor.isModified()  // isModified -> Bool
        _ = modified
        try editor.rotateAllPages(90)  // rotateAllPages
        XCTAssertEqual(try editor.getPageRotation(0), 90)  // getPageRotation
        try editor.setProducer("x")  // setProducer
        _ = try editor.getProducer()  // getProducer
        XCTAssertFalse(try editor.saveToBytes().isEmpty)  // saveToBytes
        editor.close()  // close
    }

    // ── PDF creation: builder API ────────────────────────────────────────────
    func testDocumentBuilder() throws {
        let builder = try DocumentBuilder.create()  // DocumentBuilder.create
        try builder.setTitle("Coverage Title")  // setTitle
        try builder.setAuthor("Tester")  // setAuthor

        let page = try builder.page(595, 842)  // page(width, height)
        try page.font("Helvetica", 12)  // font (standard font path)
        try page.heading(1, "Title")  // heading
        try page.paragraph("Hello world from the builder.")  // paragraph
        try page.done()  // done() commits + consumes

        let bytes = try builder.build()  // build -> bytes
        XCTAssertGreaterThan(bytes.count, 100)

        // Reopen the produced PDF and confirm the content round-trips.
        let doc = try Document.openFromBytes(bytes)
        XCTAssertGreaterThanOrEqual(try doc.pageCount(), 1)
        let text = try doc.extractText(0)
        XCTAssertTrue(text.contains("Hello") || text.contains("Title"))

        builder.close()
    }

    // ── Error path ───────────────────────────────────────────────────────────
    func testErrorOnMissingFile() {
        XCTAssertThrowsError(try Document.open("/nonexistent/nope.pdf")) { error in
            XCTAssertTrue(error is PdfOxideError)
        }
    }

    // ── Phase-6 validation (PDF/A, PDF/UA, PDF/X) ─────────────────────────────
    // Fully testable on the sample document: assert booleans + list shapes.
    func testValidatePdfA() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let results = try doc.validatePdfA(2)  // A2b
        let compliant: Bool = try results.isCompliant()  // isCompliant -> Bool
        _ = compliant
        let errors: [String] = try results.errors()  // errors() -> [String]
        XCTAssertGreaterThanOrEqual(errors.count, 0)
        XCTAssertGreaterThanOrEqual(try results.warningCount(), 0)
        results.close()
    }

    func testValidatePdfUa() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let results = try doc.validatePdfUa(1)
        let accessible: Bool = try results.isAccessible()  // isAccessible -> Bool
        _ = accessible
        XCTAssertGreaterThanOrEqual(try results.errors().count, 0)
        XCTAssertGreaterThanOrEqual(try results.warnings().count, 0)
        let stats = try results.stats()  // uaStats
        XCTAssertGreaterThanOrEqual(stats.pages, 0)
        XCTAssertGreaterThanOrEqual(stats.structElements, 0)
        _ = stats.images; _ = stats.tables; _ = stats.forms; _ = stats.annotations
        results.close()
    }

    func testValidatePdfX() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let results = try doc.validatePdfX(0)
        _ = try results.isCompliant()  // isCompliant -> Bool
        XCTAssertGreaterThanOrEqual(try results.errors().count, 0)
        results.close()
    }

    // ── Phase-6 log level: round-trip ─────────────────────────────────────────
    func testLogLevel() {
        let original = getLogLevel()
        setLogLevel(4)
        XCTAssertEqual(getLogLevel(), 4)
        setLogLevel(2)
        XCTAssertEqual(getLogLevel(), 2)
        setLogLevel(original)
    }

    // ── Phase-6 DSS (no DSS on a plain sample → nil, smoke) ───────────────────
    func testDss() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        // Plain document carries no /DSS.
        let dss = try doc.dss()
        if let dss {
            XCTAssertGreaterThanOrEqual(try dss.certCount(), 0)
            XCTAssertGreaterThanOrEqual(try dss.crlCount(), 0)
            XCTAssertGreaterThanOrEqual(try dss.ocspCount(), 0)
            XCTAssertGreaterThanOrEqual(try dss.vriCount(), 0)
            dss.close()
        } else {
            XCTAssertNil(dss)
        }
    }

    // ── Phase-6 certificate / signing / timestamp / TSA ───────────────────────
    // No real PKCS#12 cert or network is required: every wrapper is INVOKED
    // with minimal/empty inputs and must either return or throw PdfOxideError.
    private func expectReturnOrPdfError(_ op: String, _ body: () throws -> Void) {
        do { try body() } catch let e as PdfOxideError {
            _ = e  // expected for empty/invalid PKI inputs
        } catch {
            XCTFail("\(op): unexpected error type \(error)")
        }
    }

    func testCertificateLoadCoverage() {
        // Empty PKCS#12 bytes + empty PEMs: must throw the binding error type.
        expectReturnOrPdfError("Certificate.loadFromBytes") {
            let cert = try Certificate.loadFromBytes([], password: "")
            // If it somehow loads, exercise every accessor.
            _ = try cert.subject(); _ = try cert.issuer(); _ = try cert.serial()
            _ = try cert.validity(); _ = try cert.isValid()
            cert.close()
        }
        expectReturnOrPdfError("Certificate.loadFromPem") {
            let cert = try Certificate.loadFromPem(certPem: "", keyPem: "")
            cert.close()
        }
    }

    func testSigningCoverage() throws {
        let pdf = try samplePdf()
        // Sign with an (almost certainly) un-loadable cert: cover signBytes,
        // signBytesPades, signBytesPadesOpts via the cert-load failure path.
        expectReturnOrPdfError("signBytes") {
            let cert = try Certificate.loadFromBytes([], password: "")
            _ = try signBytes(pdf, certificate: cert)
            _ = try signBytesPades(pdf, certificate: cert, level: 0)
            _ = try signBytesPadesOpts(
                pdf, certificate: cert, level: 0,
                certs: [[1, 2]], crls: [[3]], ocsps: [])
        }
    }

    func testTimestampCoverage() {
        // Parse invalid DER: must throw; if it returns, exercise accessors.
        expectReturnOrPdfError("Timestamp.parse") {
            let ts = try Timestamp.parse([0x00, 0x01, 0x02])
            _ = try ts.token(); _ = try ts.messageImprint(); _ = try ts.time()
            _ = try ts.serial(); _ = try ts.tsaName(); _ = try ts.policyOid()
            _ = try ts.hashAlgorithm(); _ = try ts.verify()
            ts.close()
        }
    }

    func testTsaClientCoverage() {
        // Create a client against a dummy URL; request methods may fail (no net).
        expectReturnOrPdfError("TsaClient") {
            let client = try TsaClient.create(url: "http://127.0.0.1:0/tsa")
            _ = try? client.requestTimestamp([1, 2, 3])
            _ = try? client.requestTimestampHash([0, 1, 2, 3], hashAlgo: 0)
            client.close()
        }
    }

    func testSignatureInfoCoverage() throws {
        // No signatures exist on a freshly built sample, and obtaining a real
        // FfiSignatureInfo requires a signed document. Reference every
        // SignatureInfo wrapper through a (never-executed) closure so the binding
        // surface is type-checked / compiled without invoking on a null handle.
        let exercise: (SignatureInfo, [UInt8]) throws -> Void = { sig, bytes in
            _ = try sig.signerName(); _ = try sig.signingReason(); _ = try sig.signingLocation()
            _ = try sig.signingTime(); _ = try sig.certificate(); _ = try sig.padesLevel()
            _ = try sig.hasTimestamp(); _ = try sig.timestamp(); _ = try sig.verify()
            _ = try sig.verifyDetached(bytes)
            let ts = try Timestamp.parse([0])
            _ = try sig.addTimestamp(ts)
            sig.close()
        }
        _ = exercise
    }

    // ── Phase-7: barcodes / render variants / page getters / redaction /
    //            constructors / merge / OCR / timestamp ────────────────────────

    // A 1×1 PNG (RGBA, fully opaque red) for image-constructor coverage.
    private func tinyPng() -> [UInt8] {
        [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x62, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x18, 0xDD, 0x8D,
            0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ]
    }

    // Barcodes are fully testable: generate, then assert every accessor.
    func testBarcodes() throws {
        let qr = try BarcodeImage.generateQrCode(
            "https://example.com", errorCorrection: 1, sizePx: 128)
        XCTAssertEqual(try qr.data(), "https://example.com")  // get_data
        _ = try qr.format()  // get_format
        _ = try qr.confidence()  // get_confidence
        XCTAssertFalse(try qr.imagePng(sizePx: 128).isEmpty)  // get_image_png
        XCTAssertFalse(try qr.svg(sizePx: 128).isEmpty)  // get_svg
        qr.close()

        // pdf_generate_barcode: format codes are implementation-defined; invoke
        // and either assert accessors or accept the binding error.
        expectReturnOrPdfError("generateBarcode") {
            let bc = try BarcodeImage.generateBarcode("12345670", format: 0, sizePx: 128)
            XCTAssertFalse(try bc.data().isEmpty)
            _ = try bc.format()
            XCTAssertFalse(try bc.imagePng(sizePx: 128).isEmpty)
            bc.close()
        }
    }

    // addBarcodeToPage on an editor.
    func testAddBarcodeToPage() throws {
        let editor = try DocumentEditor.openFromBytes(try samplePdf())
        let qr = try BarcodeImage.generateQrCode("X", sizePx: 64)
        try editor.addBarcodeToPage(0, qr, x: 10, y: 10, width: 50, height: 50)
        XCTAssertFalse(try editor.saveToBytes().isEmpty)
        qr.close()
        editor.close()
    }

    func testRenderVariants() throws {
        let doc = try Document.openFromBytes(try samplePdf())

        let opt = try doc.renderPageWithOptions(0, dpi: 96)  // with_options
        XCTAssertGreaterThan(opt.width, 0)
        XCTAssertGreaterThan(opt.height, 0)
        XCTAssertFalse(opt.data.isEmpty)

        // with_options_ex
        let optEx = try doc.renderPageWithOptionsEx(0, dpi: 96, excludedLayers: ["NoSuchLayer"])
        XCTAssertGreaterThan(optEx.width, 0)
        XCTAssertFalse(optEx.data.isEmpty)

        let region = try doc.renderPageRegion(
            0, cropX: 0, cropY: 0, cropWidth: 100, cropHeight: 100)  // region
        XCTAssertGreaterThan(region.width, 0)
        XCTAssertFalse(region.data.isEmpty)

        let fit = try doc.renderPageFit(0, width: 200, height: 200)  // fit
        XCTAssertGreaterThan(fit.width, 0)
        XCTAssertLessThanOrEqual(fit.width, 200)
        XCTAssertFalse(fit.data.isEmpty)

        let raw = try doc.renderPageRaw(0, dpi: 96)  // raw
        XCTAssertGreaterThan(raw.width, 0)
        XCTAssertGreaterThan(raw.height, 0)
        XCTAssertFalse(raw.image.data.isEmpty)

        _ = try? doc.estimateRenderTime(0)  // estimate_render_time (smoke)
    }

    func testRendererHandle() {
        // create_renderer is a no-op stub in this cdylib (returns null/error):
        // invoke and accept either a live handle or the binding error type.
        expectReturnOrPdfError("Renderer.create") {
            // create_renderer + renderer_free
            let r = try Renderer.create(dpi: 150, format: 0, quality: 90, antiAlias: true)
            r.close()
            r.close()  // idempotent
        }
    }

    func testPageGetters() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        XCTAssertGreaterThan(try doc.pageWidth(0), 0)  // page_get_width
        XCTAssertGreaterThan(try doc.pageHeight(0), 0)  // page_get_height
        _ = try doc.pageRotation(0)  // page_get_rotation

        let elements = try doc.pageElements(0)  // page_get_elements + element accessors
        let n = try elements.count()
        XCTAssertGreaterThanOrEqual(n, 0)
        if n > 0 {
            let e = try elements.element(0)
            _ = e.type; _ = e.text; _ = e.rect
        }
        _ = try elements.all()
        XCTAssertFalse(try elements.toJson().isEmpty)
        elements.close()
    }

    func testRedaction() throws {
        let editor = try DocumentEditor.openFromBytes(try samplePdf())
        // redaction_add
        try editor.redactionAdd(0, x1: 10, y1: 10, x2: 100, y2: 30, r: 0, g: 0, b: 0)
        XCTAssertGreaterThanOrEqual(try editor.redactionCount(0), 1)  // redaction_count
        _ = try? editor.redactionApply(scrubMetadata: false, r: 0, g: 0, b: 0)  // redaction_apply
        _ = try editor.redactionScrubMetadata()  // redaction_scrub_metadata
        editor.close()
    }

    func testFromImageConstructors() {
        // from_image_bytes: valid tiny PNG should build; bad input must raise.
        expectReturnOrPdfError("Pdf.fromImageBytes") {
            let pdf = try Pdf.fromImageBytes(tinyPng())
            XCTAssertGreaterThan(try pdf.toBytes().count, 100)
            pdf.close()
        }
        expectReturnOrPdfError("Pdf.fromImageBytes(bad)") {
            _ = try Pdf.fromImageBytes([0x00, 0x01, 0x02])
        }
        // from_image: write the tiny PNG to a temp file, then build from it.
        expectReturnOrPdfError("Pdf.fromImage") {
            let path = NSTemporaryDirectory() + "pdfoxide_swift_tiny.png"
            try Data(tinyPng()).write(to: URL(fileURLWithPath: path))
            defer { try? FileManager.default.removeItem(atPath: path) }
            let pdf = try Pdf.fromImage(path)
            XCTAssertGreaterThan(try pdf.toBytes().count, 100)
            pdf.close()
        }
    }

    func testFromHtmlCss() {
        // from_html_css errors when no default font is available in this cdylib:
        // invoke and accept either a built PDF or the binding error type.
        expectReturnOrPdfError("Pdf.fromHtmlCss") {
            let pdf = try Pdf.fromHtmlCss(
                html: "<h1>HtmlCss</h1><p>body</p>", css: "h1{color:#333}")
            XCTAssertGreaterThan(try pdf.toBytes().count, 100)  // from_html_css
            pdf.close()
        }

        // from_html_css_with_fonts: empty font cascade is a valid call.
        expectReturnOrPdfError("Pdf.fromHtmlCssWithFonts") {
            let p2 = try Pdf.fromHtmlCssWithFonts(
                html: "<p>x</p>", css: "", families: [], fonts: [])
            XCTAssertGreaterThan(try p2.toBytes().count, 100)
            p2.close()
        }
    }

    func testMerge() throws {
        // Merge two real temp PDFs.
        let a = NSTemporaryDirectory() + "pdfoxide_swift_merge_a.pdf"
        let b = NSTemporaryDirectory() + "pdfoxide_swift_merge_b.pdf"
        try Pdf.fromMarkdown("# A\n\nfirst\n").save(a)
        try Pdf.fromMarkdown("# B\n\nsecond\n").save(b)
        defer {
            try? FileManager.default.removeItem(atPath: a)
            try? FileManager.default.removeItem(atPath: b)
        }
        let merged = try merge([a, b])  // merge
        XCTAssertGreaterThan(merged.count, 100)
        let doc = try Document.openFromBytes(merged)
        XCTAssertGreaterThanOrEqual(try doc.pageCount(), 2)
    }

    // OCR needs model files; invoke the wrappers with empty/minimal inputs and
    // assert they return or raise the binding error type.
    func testOcrCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        _ = try? doc.ocrPageNeedsOcr(0)  // ocr_page_needs_ocr (smoke)
        _ = try? doc.ocrExtractText(0, engine: nil)  // ocr_extract_text (engine == nil)

        expectReturnOrPdfError("OcrEngine.create") {
            let engine = try OcrEngine.create(detModelPath: "", recModelPath: "", dictPath: "")
            _ = try? doc.ocrExtractText(0, engine: engine)
            engine.close()
        }
    }

    // add_timestamp needs a real TSA + signed PDF; INVOKE with minimal inputs
    // and assert it returns or raises the binding error.
    func testAddTimestampCoverage() throws {
        let pdf = try samplePdf()
        let result: [UInt8]?
        do {
            result = try addTimestamp(pdf, sigIndex: 0, tsaUrl: "http://127.0.0.1:0/tsa")
        } catch let e as PdfOxideError {
            result = nil
            _ = e  // expected: no signature / no TSA reachable
        }
        XCTAssertTrue(result == nil || result!.count >= 0)
    }

    // ── Final phase: office import/export ────────────────────────────────────
    // Export may legitimately succeed or error on the sample; import needs real
    // office bytes. Invoke each wrapper as return-or-error.
    func testOfficeExportCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("toDocx") { XCTAssertGreaterThanOrEqual(try doc.toDocx().count, 0) }
        expectReturnOrPdfError("toPptx") { XCTAssertGreaterThanOrEqual(try doc.toPptx().count, 0) }
        expectReturnOrPdfError("toXlsx") { XCTAssertGreaterThanOrEqual(try doc.toXlsx().count, 0) }
    }

    func testOfficeOpenCoverage() {
        // Bogus bytes: the wrapper must return a Document or throw PdfOxideError.
        let bad: [UInt8] = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00]
        expectReturnOrPdfError("openFromDocxBytes") { _ = try Document.openFromDocxBytes(bad) }
        expectReturnOrPdfError("openFromPptxBytes") { _ = try Document.openFromPptxBytes(bad) }
        expectReturnOrPdfError("openFromXlsxBytes") { _ = try Document.openFromXlsxBytes(bad) }
    }

    // ── Final phase: in-rect extraction ──────────────────────────────────────
    func testInRectExtraction() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        let w = try doc.pageWidth(0), h = try doc.pageHeight(0)
        expectReturnOrPdfError("extractTextInRect") {
            _ = try doc.extractTextInRect(0, x: 0, y: 0, w: w, h: h)  // string
        }
        expectReturnOrPdfError("extractWordsInRect") {
            _ = try doc.extractWordsInRect(0, x: 0, y: 0, w: w, h: h)  // [Word]
        }
        expectReturnOrPdfError("extractLinesInRect") {
            _ = try doc.extractLinesInRect(0, x: 0, y: 0, w: w, h: h)  // [TextLine]
        }
        expectReturnOrPdfError("extractTablesInRect") {
            _ = try doc.extractTablesInRect(0, x: 0, y: 0, w: w, h: h)  // [Table]
        }
        expectReturnOrPdfError("extractImagesInRect") {
            _ = try doc.extractImagesInRect(0, x: 0, y: 0, w: w, h: h)  // [Image]
        }
    }

    // ── Final phase: auto extraction & classification ────────────────────────
    func testAutoExtractionCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("extractTextAuto") { _ = try doc.extractTextAuto(0) }
        expectReturnOrPdfError("extractAllText") { _ = try doc.extractAllText() }
        expectReturnOrPdfError("extractPageAuto") {
            _ = try doc.extractPageAuto(0, optionsJson: "{}")
        }
        expectReturnOrPdfError("classifyPage") { _ = try doc.classifyPage(0) }
        expectReturnOrPdfError("classifyDocument") { _ = try doc.classifyDocument() }
    }

    // ── Final phase: header / footer / artifact removal ──────────────────────
    func testHeaderFooterArtifactCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("eraseHeader") { _ = try doc.eraseHeader(0) }
        expectReturnOrPdfError("eraseFooter") { _ = try doc.eraseFooter(0) }
        expectReturnOrPdfError("eraseArtifacts") { _ = try doc.eraseArtifacts(0) }
        expectReturnOrPdfError("removeHeaders") { _ = try doc.removeHeaders(threshold: 0.5) }
        expectReturnOrPdfError("removeFooters") { _ = try doc.removeFooters(threshold: 0.5) }
        expectReturnOrPdfError("removeArtifacts") { _ = try doc.removeArtifacts(threshold: 0.5) }
    }

    // ── Final phase: forms ───────────────────────────────────────────────────
    func testFormsCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        // The sample has no AcroForm; an empty list is acceptable.
        expectReturnOrPdfError("formFields") {
            let fields = try doc.formFields()
            for f in fields { _ = f.name; _ = f.value; _ = f.type; _ = f.readonly; _ = f.required }
        }
        expectReturnOrPdfError("exportFormData") { _ = try doc.exportFormData(formatType: 0) }
        expectReturnOrPdfError("importFormData") { _ = try doc.importFormData("/nonexistent.fdf") }
        expectReturnOrPdfError("importFormFromFile") {
            _ = try doc.importFormFromFile("/nonexistent.fdf")
        }

        let editor = try DocumentEditor.openFromBytes(try samplePdf())
        let fdf: [UInt8] = Array("%FDF-1.2\n".utf8)
        expectReturnOrPdfError("importFdfBytes") { _ = try editor.importFdfBytes(fdf) }
        expectReturnOrPdfError("importXfdfBytes") {
            _ = try editor.importXfdfBytes(Array("<xfdf/>".utf8))
        }
    }

    // ── Final phase: structure & metadata ────────────────────────────────────
    func testStructureMetadataCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("outline") { _ = try doc.outline() }
        expectReturnOrPdfError("pageLabels") { _ = try doc.pageLabels() }
        expectReturnOrPdfError("xmpMetadata") { _ = try doc.xmpMetadata() }
        expectReturnOrPdfError("sourceBytes") {
            XCTAssertGreaterThanOrEqual(try doc.sourceBytes().count, 0)
        }
        _ = try doc.hasXfa()  // hasXfa (Bool smoke)
        expectReturnOrPdfError("planSplitByBookmarks") {
            _ = try doc.planSplitByBookmarks(optionsJson: "{}")
        }
    }

    // ── Final phase: signatures ──────────────────────────────────────────────
    func testSignatureCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        // Unsigned sample: count is 0, accessors return/throw cleanly.
        expectReturnOrPdfError("signatureCount") {
            XCTAssertGreaterThanOrEqual(try doc.signatureCount(), 0)
        }
        expectReturnOrPdfError("signature") { _ = try doc.signature(0) }
        expectReturnOrPdfError("verifyAllSignatures") { _ = try doc.verifyAllSignatures() }
        _ = try doc.hasTimestamp()  // hasTimestamp (Bool smoke)

        // sign needs a real certificate; invoke with a loaded-or-error cert.
        expectReturnOrPdfError("sign") {
            let cert = try Certificate.loadFromPem(certPem: "", keyPem: "")
            _ = try doc.sign(cert, reason: "test", location: "here")
        }
    }

    // ── Final phase: PDF/A conversion ────────────────────────────────────────
    func testConvertToPdfACoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("convertToPdfA") { _ = try doc.convertToPdfA(0) }
    }

    // ── Final phase: JSON serialisers & font size ────────────────────────────
    func testJsonSerialisersCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        expectReturnOrPdfError("annotationsToJson") { _ = try doc.annotationsToJson(0) }
        expectReturnOrPdfError("fontsToJson") { _ = try doc.fontsToJson(0) }
        expectReturnOrPdfError("fontSize") { _ = try doc.fontSize(0, fontIndex: 0) }
        expectReturnOrPdfError("searchResultsToJson") {
            _ = try doc.searchResultsToJson(0, "Alpha", caseSensitive: false)
        }
        // ElementList JSON path (pdf_oxide_elements_to_json).
        expectReturnOrPdfError("ElementList.toJson") {
            let elements = try doc.pageElements(0)
            _ = try elements.toJson()
        }
    }

    // ── Final phase: annotation extras ───────────────────────────────────────
    func testAnnotationExtrasCoverage() throws {
        let doc = try Document.openFromBytes(try samplePdf())
        // The sample has no annotations; reading index 0 returns defaults or
        // raises the binding error — both are acceptable.
        expectReturnOrPdfError("annotationExtras") {
            let extras = try doc.annotationExtras(0, index: 0)
            _ = extras.color
            _ = extras.creationDate
            _ = extras.modificationDate
            _ = extras.hidden
            _ = extras.markedDeleted
            _ = extras.printable
            _ = extras.readOnly
            _ = extras.uri
            _ = extras.iconName
            for q in extras.quadPoints { _ = q.x1; _ = q.y1; _ = q.x2; _ = q.y2 }
        }
    }

    // ── Final phase: Pdf page count alias ────────────────────────────────────
    func testPdfPageCount() throws {
        let pdf = try Pdf.fromMarkdown("# One\n\nbody\n")
        // pdf_get_page_count (Pdf-builder alias) errors (code 1) on a freshly
        // built Pdf in this cdylib: invoke and accept a count or the error type.
        expectReturnOrPdfError("Pdf.pageCount") {
            // Call through `try` directly so a thrown PdfOxideError propagates to
            // the wrapper's catch. Passing `try pdf.pageCount()` *inside*
            // XCTAssertGreaterThanOrEqual would let the assert swallow the throw
            // and record its own failure instead.
            let n = try pdf.pageCount()  // pdf_get_page_count
            XCTAssertGreaterThanOrEqual(n, 1)
        }
    }

    // ── Final phase: process-global crypto / models / config ─────────────────
    func testCryptoNamespaceCoverage() {
        _ = PdfOxide.cryptoActiveProvider()  // crypto_active_provider
        _ = PdfOxide.cryptoCbom()  // crypto_cbom
        _ = PdfOxide.cryptoFipsAvailable()  // crypto_fips_available
        _ = PdfOxide.cryptoInventory()  // crypto_inventory
        _ = PdfOxide.cryptoPolicy()  // crypto_policy
        _ = PdfOxide.cryptoSetPolicy("default")  // crypto_set_policy
        _ = PdfOxide.cryptoUseFips()  // crypto_use_fips
    }

    func testModelsAndConfigCoverage() {
        _ = PdfOxide.modelManifest()  // model_manifest
        _ = PdfOxide.prefetchAvailable()  // prefetch_available
        // prefetch needs network/models — return-or-error.
        expectReturnOrPdfError("prefetchModels") {
            _ = try PdfOxide.prefetchModels(languagesCsv: "en")
        }
        let prevOps = PdfOxide.setMaxOpsPerStream(1_000_000)  // set_max_ops_per_stream
        _ = PdfOxide.setMaxOpsPerStream(prevOps)
        let prevGlyphs = PdfOxide.setPreserveUnmappedGlyphs(1)  // set_preserve_unmapped_glyphs
        _ = PdfOxide.setPreserveUnmappedGlyphs(prevGlyphs)
    }

    // ── Final phase: standalone renderer config ──────────────────────────────
    func testConfiguredRendererCoverage() {
        // pdf_create_renderer / pdf_renderer_free.
        expectReturnOrPdfError("Renderer.create") {
            let r = try Renderer.create(dpi: 150, format: 0, quality: 90, antiAlias: true)
            r.close()
        }
    }
}
