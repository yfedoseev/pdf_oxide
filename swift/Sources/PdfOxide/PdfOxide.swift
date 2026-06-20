// pdf_oxide — idiomatic Swift bindings over the C ABI.
//
// Handles are owned by classes (freed in deinit); returned C strings/buffers
// are copied into Swift String/[UInt8] and freed via free_string; non-success
// C-ABI error codes are thrown as PdfOxideError.
//
// API surface mirrors the other language bindings; coverage is asserted by
// PdfOxideTests (one test per public method).
import CPdfOxide
import Foundation

/// Thrown on any non-success C-ABI error code.
public struct PdfOxideError: Error, CustomStringConvertible {
    public let code: Int32
    public let op: String
    public var description: String { "PdfOxideError: \(op) failed (error code \(code))" }
}

/// PDF version (e.g. 1.7).
public struct PdfVersion: CustomStringConvertible {
    public let major: Int
    public let minor: Int
    public var description: String { "\(major).\(minor)" }
}

/// An axis-aligned bounding box in PDF user-space units.
public struct Bbox {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double
}

/// A single extracted character.
public struct Char {
    /// The Unicode scalar value (codepoint) of the character.
    public let character: UInt32
    public let bbox: Bbox
    public let fontName: String
    public let fontSize: Double
}

/// A single extracted word.
public struct Word {
    public let text: String
    public let bbox: Bbox
    public let fontName: String
    public let fontSize: Double
    public let bold: Bool
}

/// A single extracted line of text.
public struct TextLine {
    public let text: String
    public let bbox: Bbox
    public let wordCount: Int
}

/// A single extracted table. Cells are read on demand via `cell(_:_:)`.
public struct Table {
    public let rowCount: Int
    public let colCount: Int
    public let hasHeader: Bool
    private let cells: [[String]]

    fileprivate init(rowCount: Int, colCount: Int, hasHeader: Bool, cells: [[String]]) {
        self.rowCount = rowCount
        self.colCount = colCount
        self.hasHeader = hasHeader
        self.cells = cells
    }

    /// The text of the cell at (row, col); empty string if out of bounds.
    public func cell(_ row: Int, _ col: Int) -> String {
        guard row >= 0, row < cells.count, col >= 0, col < cells[row].count else { return "" }
        return cells[row][col]
    }
}

/// A single embedded font.
public struct Font {
    public let name: String
    public let type: String
    public let encoding: String
    public let embedded: Bool
    public let subset: Bool
}

/// A single embedded image.
public struct Image {
    public let width: Int
    public let height: Int
    public let bitsPerComponent: Int
    public let format: String
    public let colorspace: String
    public let data: [UInt8]
}

/// A single page annotation.
public struct Annotation {
    public let type: String
    public let subtype: String
    public let content: String
    public let author: String
    public let rect: Bbox
    public let borderWidth: Double
}

/// A single vector path.
public struct Path {
    public let bbox: Bbox
    public let strokeWidth: Double
    public let hasStroke: Bool
    public let hasFill: Bool
    public let operationCount: Int
}

/// A single full-text search hit.
public struct SearchResult {
    public let text: String
    public let page: Int
    public let bbox: Bbox
}

/// A rendered page image. Owns the native FfiRenderedImage handle so that
/// `save(_:)` can delegate to the renderer's own encoder; `width`/`height`/`data`
/// are read eagerly at construction. The handle is freed in `deinit`/`close()`.
public final class RenderedImage {
    private var handle: OpaquePointer?

    /// Pixel width of the rendered image.
    public let width: Int
    /// Pixel height of the rendered image.
    public let height: Int
    /// The encoded image bytes (e.g. PNG), copied from the native buffer.
    public let data: [UInt8]

    // Takes ownership of `handle`; reads width/height/data eagerly.
    fileprivate init(_ handle: OpaquePointer, _ op: String) throws {
        self.handle = handle
        var code: Int32 = 0
        self.width = Int(pdf_get_rendered_image_width(handle, &code))
        self.height = Int(pdf_get_rendered_image_height(handle, &code))
        var dataLen: Int32 = 0
        if let p = pdf_get_rendered_image_data(handle, &dataLen, &code) {
            // Encoded image buffers free via free_bytes, not free_string.
            defer { free_bytes(p) }
            let len = dataLen < 0 ? 0 : Int(dataLen)
            self.data = Array(UnsafeBufferPointer(start: p, count: len))
        } else {
            self.data = []
        }
    }

    deinit { if let h = handle { pdf_rendered_image_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "RenderedImage is closed") }
        return h
    }

    /// Save the rendered image to `path` using the renderer's own encoder.
    public func save(_ path: String) throws {
        var code: Int32 = 0
        if pdf_save_rendered_image(try ptr(), path, &code) != 0 {
            throw PdfOxideError(code: code, op: "RenderedImage.save")
        }
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_rendered_image_free(h); handle = nil }
    }
}

// Copy a C string return into a Swift String and free it via free_string.
private func takeString(_ ptr: UnsafeMutablePointer<CChar>?, _ code: Int32, _ op: String) throws -> String {
    guard let ptr else { throw PdfOxideError(code: code, op: op) }
    defer { free_string(ptr) }
    return String(cString: ptr)
}

/// An opened PDF for extraction/inspection.
public final class Document {
    private var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_document_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "Document is closed") }
        return h
    }

    /// Open a PDF from a filesystem path.
    public static func open(_ path: String) throws -> Document {
        var code: Int32 = 0
        guard let h = pdf_document_open(path, &code) else { throw PdfOxideError(code: code, op: "open") }
        return Document(h)
    }

    /// Open a PDF from in-memory bytes.
    public static func openFromBytes(_ bytes: [UInt8]) throws -> Document {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_document_open_from_bytes(buf.baseAddress, buf.count, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "openFromBytes") }
        return Document(h)
    }

    /// Open a password-protected PDF.
    public static func openWithPassword(_ path: String, password: String) throws -> Document {
        var code: Int32 = 0
        guard let h = pdf_document_open_with_password(path, password, &code) else {
            throw PdfOxideError(code: code, op: "openWithPassword")
        }
        return Document(h)
    }

    public func pageCount() throws -> Int {
        var code: Int32 = 0
        let n = pdf_document_get_page_count(try ptr(), &code)
        if n < 0 { throw PdfOxideError(code: code, op: "pageCount") }
        return Int(n)
    }

    public func version() throws -> PdfVersion {
        var major: UInt8 = 0, minor: UInt8 = 0
        pdf_document_get_version(try ptr(), &major, &minor)
        return PdfVersion(major: Int(major), minor: Int(minor))
    }

    public func isEncrypted() throws -> Bool { pdf_document_is_encrypted(try ptr()) }
    public func hasStructureTree() throws -> Bool { pdf_document_has_structure_tree(try ptr()) }

    public func extractText(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_extract_text(try ptr(), Int32(page), &code), code, "extractText")
    }
    public func toPlainText(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_plain_text(try ptr(), Int32(page), &code), code, "toPlainText")
    }
    public func toMarkdown(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_markdown(try ptr(), Int32(page), &code), code, "toMarkdown")
    }
    public func toHtml(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_html(try ptr(), Int32(page), &code), code, "toHtml")
    }
    public func toMarkdownAll() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_markdown_all(try ptr(), &code), code, "toMarkdownAll")
    }
    public func toHtmlAll() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_html_all(try ptr(), &code), code, "toHtmlAll")
    }
    public func toPlainTextAll() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_plain_text_all(try ptr(), &code), code, "toPlainTextAll")
    }

    /// Authenticate against an encrypted document. Returns true on success;
    /// returns false for a wrong password without throwing.
    public func authenticate(_ password: String) throws -> Bool {
        var code: Int32 = 0
        return pdf_document_authenticate(try ptr(), password, &code)
    }
    public func extractStructuredJson(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_extract_structured_to_json(try ptr(), Int32(page), &code), code, "extractStructuredJson")
    }

    // ── Phase-1 element extraction ───────────────────────────────────────────

    /// Extract individual characters from a (0-based) page.
    public func extractChars(_ pageIndex: Int) throws -> [Char] {
        var code: Int32 = 0
        guard let list = pdf_document_extract_chars(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "extractChars")
        }
        defer { pdf_oxide_char_list_free(list) }
        let n = Int(pdf_oxide_char_count(list))
        var result: [Char] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let character = pdf_oxide_char_get_char(list, idx, &code)
            let fontName = try takeString(pdf_oxide_char_get_font_name(list, idx, &code), code, "extractChars.fontName")
            let fontSize = pdf_oxide_char_get_font_size(list, idx, &code)
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_char_get_bbox(list, idx, &x, &y, &w, &h, &code)
            result.append(Char(
                character: character,
                bbox: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h)),
                fontName: fontName,
                fontSize: Double(fontSize)
            ))
        }
        return result
    }

    /// Extract words from a (0-based) page.
    public func extractWords(_ pageIndex: Int) throws -> [Word] {
        var code: Int32 = 0
        guard let list = pdf_document_extract_words(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "extractWords")
        }
        defer { pdf_oxide_word_list_free(list) }
        let n = Int(pdf_oxide_word_count(list))
        var result: [Word] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let text = try takeString(pdf_oxide_word_get_text(list, idx, &code), code, "extractWords.text")
            let fontName = try takeString(pdf_oxide_word_get_font_name(list, idx, &code), code, "extractWords.fontName")
            let fontSize = pdf_oxide_word_get_font_size(list, idx, &code)
            let bold = pdf_oxide_word_is_bold(list, idx, &code)
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_word_get_bbox(list, idx, &x, &y, &w, &h, &code)
            result.append(Word(
                text: text,
                bbox: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h)),
                fontName: fontName,
                fontSize: Double(fontSize),
                bold: bold
            ))
        }
        return result
    }

    /// Extract text lines from a (0-based) page.
    public func extractTextLines(_ pageIndex: Int) throws -> [TextLine] {
        var code: Int32 = 0
        guard let list = pdf_document_extract_text_lines(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "extractTextLines")
        }
        defer { pdf_oxide_line_list_free(list) }
        let n = Int(pdf_oxide_line_count(list))
        var result: [TextLine] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let text = try takeString(pdf_oxide_line_get_text(list, idx, &code), code, "extractTextLines.text")
            let wordCount = Int(pdf_oxide_line_get_word_count(list, idx, &code))
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_line_get_bbox(list, idx, &x, &y, &w, &h, &code)
            result.append(TextLine(
                text: text,
                bbox: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h)),
                wordCount: wordCount
            ))
        }
        return result
    }

    /// Extract tables from a (0-based) page.
    public func extractTables(_ pageIndex: Int) throws -> [Table] {
        var code: Int32 = 0
        guard let list = pdf_document_extract_tables(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "extractTables")
        }
        defer { pdf_oxide_table_list_free(list) }
        let n = Int(pdf_oxide_table_count(list))
        var result: [Table] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let rowCount = Int(pdf_oxide_table_get_row_count(list, idx, &code))
            let colCount = Int(pdf_oxide_table_get_col_count(list, idx, &code))
            let hasHeader = pdf_oxide_table_has_header(list, idx, &code)
            var cells: [[String]] = []
            cells.reserveCapacity(rowCount)
            for r in 0..<max(0, rowCount) {
                var row: [String] = []
                row.reserveCapacity(colCount)
                for c in 0..<max(0, colCount) {
                    let cell = try takeString(
                        pdf_oxide_table_get_cell_text(list, idx, Int32(r), Int32(c), &code),
                        code, "extractTables.cell"
                    )
                    row.append(cell)
                }
                cells.append(row)
            }
            result.append(Table(rowCount: rowCount, colCount: colCount, hasHeader: hasHeader, cells: cells))
        }
        return result
    }

    // ── Phase-2 element extraction ───────────────────────────────────────────

    /// Extract embedded fonts from a (0-based) page.
    public func embeddedFonts(_ pageIndex: Int) throws -> [Font] {
        var code: Int32 = 0
        guard let list = pdf_document_get_embedded_fonts(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "embeddedFonts")
        }
        defer { pdf_oxide_font_list_free(list) }
        let n = Int(pdf_oxide_font_count(list))
        var result: [Font] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let name = try takeString(pdf_oxide_font_get_name(list, idx, &code), code, "embeddedFonts.name")
            let type = try takeString(pdf_oxide_font_get_type(list, idx, &code), code, "embeddedFonts.type")
            let encoding = try takeString(pdf_oxide_font_get_encoding(list, idx, &code), code, "embeddedFonts.encoding")
            let embedded = pdf_oxide_font_is_embedded(list, idx, &code) != 0
            let subset = pdf_oxide_font_is_subset(list, idx, &code) != 0
            result.append(Font(name: name, type: type, encoding: encoding, embedded: embedded, subset: subset))
        }
        return result
    }

    /// Extract embedded images from a (0-based) page.
    public func embeddedImages(_ pageIndex: Int) throws -> [Image] {
        var code: Int32 = 0
        guard let list = pdf_document_get_embedded_images(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "embeddedImages")
        }
        defer { pdf_oxide_image_list_free(list) }
        let n = Int(pdf_oxide_image_count(list))
        var result: [Image] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let width = Int(pdf_oxide_image_get_width(list, idx, &code))
            let height = Int(pdf_oxide_image_get_height(list, idx, &code))
            let bpc = Int(pdf_oxide_image_get_bits_per_component(list, idx, &code))
            let format = try takeString(pdf_oxide_image_get_format(list, idx, &code), code, "embeddedImages.format")
            let colorspace = try takeString(pdf_oxide_image_get_colorspace(list, idx, &code), code, "embeddedImages.colorspace")
            var dataLen: Int32 = 0
            let data: [UInt8]
            if let p = pdf_oxide_image_get_data(list, idx, &dataLen, &code) {
                // Raw image buffers free via free_bytes, not free_string.
                defer { free_bytes(p) }
                let len = dataLen < 0 ? 0 : Int(dataLen)
                data = Array(UnsafeBufferPointer(start: p, count: len))
            } else {
                data = []
            }
            result.append(Image(
                width: width, height: height, bitsPerComponent: bpc,
                format: format, colorspace: colorspace, data: data
            ))
        }
        return result
    }

    /// Extract annotations from a (0-based) page.
    public func pageAnnotations(_ pageIndex: Int) throws -> [Annotation] {
        var code: Int32 = 0
        guard let list = pdf_document_get_page_annotations(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "pageAnnotations")
        }
        defer { pdf_oxide_annotation_list_free(list) }
        let n = Int(pdf_oxide_annotation_count(list))
        var result: [Annotation] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let type = try takeString(pdf_oxide_annotation_get_type(list, idx, &code), code, "pageAnnotations.type")
            let subtype = try takeString(pdf_oxide_annotation_get_subtype(list, idx, &code), code, "pageAnnotations.subtype")
            let content = try takeString(pdf_oxide_annotation_get_content(list, idx, &code), code, "pageAnnotations.content")
            let author = try takeString(pdf_oxide_annotation_get_author(list, idx, &code), code, "pageAnnotations.author")
            let borderWidth = pdf_oxide_annotation_get_border_width(list, idx, &code)
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_annotation_get_rect(list, idx, &x, &y, &w, &h, &code)
            result.append(Annotation(
                type: type, subtype: subtype, content: content, author: author,
                rect: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h)),
                borderWidth: Double(borderWidth)
            ))
        }
        return result
    }

    /// Extract vector paths from a (0-based) page.
    public func extractPaths(_ pageIndex: Int) throws -> [Path] {
        var code: Int32 = 0
        guard let list = pdf_document_extract_paths(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "extractPaths")
        }
        defer { pdf_oxide_path_list_free(list) }
        let n = Int(pdf_oxide_path_count(list))
        var result: [Path] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let strokeWidth = pdf_oxide_path_get_stroke_width(list, idx, &code)
            let hasStroke = pdf_oxide_path_has_stroke(list, idx, &code)
            let hasFill = pdf_oxide_path_has_fill(list, idx, &code)
            let operationCount = Int(pdf_oxide_path_get_operation_count(list, idx, &code))
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_path_get_bbox(list, idx, &x, &y, &w, &h, &code)
            result.append(Path(
                bbox: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h)),
                strokeWidth: Double(strokeWidth),
                hasStroke: hasStroke, hasFill: hasFill, operationCount: operationCount
            ))
        }
        return result
    }

    // Marshal an FfiSearchResults handle into [SearchResult]; frees the handle.
    private func collectSearchResults(_ list: OpaquePointer, _ op: String) throws -> [SearchResult] {
        defer { pdf_oxide_search_result_free(list) }
        var code: Int32 = 0
        let n = Int(pdf_oxide_search_result_count(list))
        var result: [SearchResult] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            let text = try takeString(pdf_oxide_search_result_get_text(list, idx, &code), code, "\(op).text")
            let page = Int(pdf_oxide_search_result_get_page(list, idx, &code))
            var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0
            pdf_oxide_search_result_get_bbox(list, idx, &x, &y, &w, &h, &code)
            result.append(SearchResult(
                text: text, page: page,
                bbox: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(h))
            ))
        }
        return result
    }

    /// Search a single (0-based) page for `term`.
    public func search(_ pageIndex: Int, _ term: String, _ caseSensitive: Bool) throws -> [SearchResult] {
        var code: Int32 = 0
        guard let list = pdf_document_search_page(try ptr(), Int32(pageIndex), term, caseSensitive, &code) else {
            throw PdfOxideError(code: code, op: "search")
        }
        return try collectSearchResults(list, "search")
    }

    /// Search the entire document for `term`.
    public func searchAll(_ term: String, _ caseSensitive: Bool) throws -> [SearchResult] {
        var code: Int32 = 0
        guard let list = pdf_document_search_all(try ptr(), term, caseSensitive, &code) else {
            throw PdfOxideError(code: code, op: "searchAll")
        }
        return try collectSearchResults(list, "searchAll")
    }

    // ── Phase-3 page rendering ───────────────────────────────────────────────

    /// Render a (0-based) page to an image. `format` is 0=PNG (default), 1=JPEG.
    public func renderPage(_ pageIndex: Int, format: Int32 = 0) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page(try ptr(), Int32(pageIndex), format, &code) else {
            throw PdfOxideError(code: code, op: "renderPage")
        }
        return try RenderedImage(img, "renderPage")
    }

    /// Render a (0-based) page at the given `zoom` factor. `format` is 0=PNG, 1=JPEG.
    public func renderPageZoom(_ pageIndex: Int, zoom: Float, format: Int32 = 0) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page_zoom(try ptr(), Int32(pageIndex), zoom, format, &code) else {
            throw PdfOxideError(code: code, op: "renderPageZoom")
        }
        return try RenderedImage(img, "renderPageZoom")
    }

    /// Render a thumbnail of a (0-based) page fitting `size` pixels. `format` is 0=PNG, 1=JPEG.
    public func renderPageThumbnail(_ pageIndex: Int, size: Int, format: Int32 = 0) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page_thumbnail(try ptr(), Int32(pageIndex), Int32(size), format, &code) else {
            throw PdfOxideError(code: code, op: "renderPageThumbnail")
        }
        return try RenderedImage(img, "renderPageThumbnail")
    }

    /// A lightweight view of a single (0-based) page. Holds a strong reference to
    /// its Document so the native handle outlives the Page.
    public func page(_ index: Int) -> Page {
        Page(document: self, index: index)
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_document_free(h); handle = nil }
    }
}

/// A single page of a Document. Keeps the owning Document alive via a strong
/// reference; each accessor delegates to the corresponding per-page Document method.
public final class Page {
    private let document: Document
    public let index: Int

    fileprivate init(document: Document, index: Int) {
        self.document = document
        self.index = index
    }

    public func text() throws -> String { try document.extractText(index) }
    public func markdown() throws -> String { try document.toMarkdown(index) }
    public func html() throws -> String { try document.toHtml(index) }
    public func plainText() throws -> String { try document.toPlainText(index) }
}

/// A PDF produced by a builder.
public final class Pdf {
    private var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "Pdf is closed") }
        return h
    }

    public static func fromMarkdown(_ md: String) throws -> Pdf {
        var code: Int32 = 0
        guard let h = pdf_from_markdown(md, &code) else { throw PdfOxideError(code: code, op: "fromMarkdown") }
        return Pdf(h)
    }
    public static func fromHtml(_ html: String) throws -> Pdf {
        var code: Int32 = 0
        guard let h = pdf_from_html(html, &code) else { throw PdfOxideError(code: code, op: "fromHtml") }
        return Pdf(h)
    }
    public static func fromText(_ text: String) throws -> Pdf {
        var code: Int32 = 0
        guard let h = pdf_from_text(text, &code) else { throw PdfOxideError(code: code, op: "fromText") }
        return Pdf(h)
    }

    public func save(_ path: String) throws {
        var code: Int32 = 0
        if pdf_save(try ptr(), path, &code) != 0 { throw PdfOxideError(code: code, op: "save") }
    }

    public func toBytes() throws -> [UInt8] {
        var len: Int32 = 0, code: Int32 = 0
        guard let p = pdf_save_to_bytes(try ptr(), &len, &code) else { throw PdfOxideError(code: code, op: "toBytes") }
        // Raw byte buffers free via free_bytes, not free_string.
        defer { free_bytes(p) }
        let n = len < 0 ? 0 : Int(len)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_free(h); handle = nil }
    }
}
