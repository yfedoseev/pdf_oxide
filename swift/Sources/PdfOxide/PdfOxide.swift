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

    // ── Phase-7 render variants ──────────────────────────────────────────────

    /// Render with the full RenderOptions surface. Background channels are 0.0..1.0;
    /// set `transparentBackground` to true to drop the fill. `format`: 0=PNG 1=JPEG.
    public func renderPageWithOptions(
        _ pageIndex: Int, dpi: Int32 = 150, format: Int32 = 0,
        bgR: Float = 1, bgG: Float = 1, bgB: Float = 1, bgA: Float = 1,
        transparentBackground: Bool = false, renderAnnotations: Bool = true, jpegQuality: Int32 = 90
    ) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page_with_options(
            try ptr(), Int32(pageIndex), dpi, format, bgR, bgG, bgB, bgA,
            transparentBackground ? 1 : 0, renderAnnotations ? 1 : 0, jpegQuality, &code
        ) else {
            throw PdfOxideError(code: code, op: "renderPageWithOptions")
        }
        return try RenderedImage(img, "renderPageWithOptions")
    }

    /// Render with full RenderOptions plus a list of OCG `/Name`s to suppress.
    public func renderPageWithOptionsEx(
        _ pageIndex: Int, dpi: Int32 = 150, format: Int32 = 0,
        bgR: Float = 1, bgG: Float = 1, bgB: Float = 1, bgA: Float = 1,
        transparentBackground: Bool = false, renderAnnotations: Bool = true, jpegQuality: Int32 = 90,
        excludedLayers: [String] = []
    ) throws -> RenderedImage {
        let h = try ptr()
        var code: Int32 = 0
        let img = withCStringArray(excludedLayers) { layersPtr in
            pdf_render_page_with_options_ex(
                h, Int32(pageIndex), dpi, format, bgR, bgG, bgB, bgA,
                transparentBackground ? 1 : 0, renderAnnotations ? 1 : 0, jpegQuality,
                excludedLayers.isEmpty ? nil : layersPtr, excludedLayers.count, &code
            )
        }
        guard let img else { throw PdfOxideError(code: code, op: "renderPageWithOptionsEx") }
        return try RenderedImage(img, "renderPageWithOptionsEx")
    }

    /// Render a rectangular region (PDF user-space points, origin bottom-left).
    public func renderPageRegion(
        _ pageIndex: Int, cropX: Float, cropY: Float, cropWidth: Float, cropHeight: Float, format: Int32 = 0
    ) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page_region(
            try ptr(), Int32(pageIndex), cropX, cropY, cropWidth, cropHeight, format, &code
        ) else {
            throw PdfOxideError(code: code, op: "renderPageRegion")
        }
        return try RenderedImage(img, "renderPageRegion")
    }

    /// Render the page to fit inside `width`×`height` pixels, preserving aspect ratio.
    public func renderPageFit(_ pageIndex: Int, width: Int32, height: Int32, format: Int32 = 0) throws -> RenderedImage {
        var code: Int32 = 0
        guard let img = pdf_render_page_fit(try ptr(), Int32(pageIndex), width, height, format, &code) else {
            throw PdfOxideError(code: code, op: "renderPageFit")
        }
        return try RenderedImage(img, "renderPageFit")
    }

    /// Render to a raw premultiplied RGBA8888 buffer; also returns the pixel dimensions.
    public func renderPageRaw(_ pageIndex: Int, dpi: Int32 = 150) throws -> (image: RenderedImage, width: Int, height: Int) {
        var code: Int32 = 0
        var outW: Int32 = 0, outH: Int32 = 0
        guard let img = pdf_render_page_raw(try ptr(), Int32(pageIndex), dpi, &outW, &outH, &code) else {
            throw PdfOxideError(code: code, op: "renderPageRaw")
        }
        return (try RenderedImage(img, "renderPageRaw"), Int(outW), Int(outH))
    }

    /// Estimate the render time (implementation-defined units) for a page.
    public func estimateRenderTime(_ pageIndex: Int) throws -> Int32 {
        var code: Int32 = 0
        let r = pdf_estimate_render_time(UnsafeRawPointer(try ptr()), Int32(pageIndex), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "estimateRenderTime") }
        return r
    }

    // ── Phase-7 page getters (0-based) ───────────────────────────────────────

    public func pageWidth(_ pageIndex: Int) throws -> Float {
        var code: Int32 = 0
        let w = pdf_page_get_width(try ptr(), Int32(pageIndex), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "pageWidth") }
        return w
    }
    public func pageHeight(_ pageIndex: Int) throws -> Float {
        var code: Int32 = 0
        let h = pdf_page_get_height(try ptr(), Int32(pageIndex), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "pageHeight") }
        return h
    }
    public func pageRotation(_ pageIndex: Int) throws -> Int {
        var code: Int32 = 0
        let r = pdf_page_get_rotation(try ptr(), Int32(pageIndex), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "pageRotation") }
        return Int(r)
    }

    /// Extract the page's elements as an `ElementList` (freed on `close()`/`deinit`).
    public func pageElements(_ pageIndex: Int) throws -> ElementList {
        var code: Int32 = 0
        guard let h = pdf_page_get_elements(try ptr(), Int32(pageIndex), &code) else {
            throw PdfOxideError(code: code, op: "pageElements")
        }
        return ElementList(h)
    }

    // ── Phase-7 OCR ──────────────────────────────────────────────────────────

    /// Whether a (0-based) page needs OCR (i.e. is scanned/hybrid).
    public func ocrPageNeedsOcr(_ pageIndex: Int) throws -> Bool {
        var code: Int32 = 0
        let needs = pdf_ocr_page_needs_ocr(try ptr(), Int32(pageIndex), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "ocrPageNeedsOcr") }
        return needs
    }

    /// Extract text from a page via OCR. `engine` may be nil for native-only extraction.
    public func ocrExtractText(_ pageIndex: Int, engine: OcrEngine? = nil) throws -> String {
        let h = try ptr()
        var code: Int32 = 0
        let enginePtr = engine.flatMap { $0.handle }
        return try takeString(
            pdf_ocr_extract_text(h, Int32(pageIndex), enginePtr.map { UnsafeRawPointer($0) }, &code),
            code, "ocrExtractText"
        )
    }

    /// A lightweight view of a single (0-based) page. Holds a strong reference to
    /// its Document so the native handle outlives the Page.
    public func page(_ index: Int) -> Page {
        Page(document: self, index: index)
    }

    // ── Phase-6 validation (PDF/A, PDF/UA, PDF/X) ────────────────────────────

    /// Validate the document against a PDF/A conformance level.
    /// `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
    public func validatePdfA(_ level: Int32) throws -> PdfAResults {
        var code: Int32 = 0
        guard let h = pdf_validate_pdf_a_level(try ptr(), level, &code) else {
            throw PdfOxideError(code: code, op: "validatePdfA")
        }
        return PdfAResults(h)
    }

    /// Validate the document against a PDF/UA accessibility level.
    public func validatePdfUa(_ level: Int32) throws -> UaResults {
        var code: Int32 = 0
        guard let h = pdf_validate_pdf_ua(try ptr(), level, &code) else {
            throw PdfOxideError(code: code, op: "validatePdfUa")
        }
        return UaResults(h)
    }

    /// Validate the document against a PDF/X conformance level.
    public func validatePdfX(_ level: Int32) throws -> PdfXResults {
        var code: Int32 = 0
        guard let h = pdf_validate_pdf_x_level(try ptr(), level, &code) else {
            throw PdfOxideError(code: code, op: "validatePdfX")
        }
        return PdfXResults(h)
    }

    /// Read the document's `/DSS` (Document Security Store), if present.
    /// Returns `nil` when the document carries no DSS.
    public func dss() throws -> Dss? {
        var code: Int32 = 0
        guard let h = pdf_document_get_dss(UnsafeRawPointer(try ptr()), &code) else {
            if code != 0 { throw PdfOxideError(code: code, op: "dss") }
            return nil  // no DSS is not an error
        }
        return Dss(h)
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

    // ── Phase-7 image / HTML+CSS constructors ────────────────────────────────

    /// Build a single-page PDF wrapping the image at `path`.
    public static func fromImage(_ path: String) throws -> Pdf {
        var code: Int32 = 0
        guard let h = pdf_from_image(path, &code) else { throw PdfOxideError(code: code, op: "fromImage") }
        return Pdf(h)
    }

    /// Build a single-page PDF wrapping the in-memory image `bytes`.
    public static func fromImageBytes(_ bytes: [UInt8]) throws -> Pdf {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_from_image_bytes(buf.baseAddress, Int32(buf.count), &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "fromImageBytes") }
        return Pdf(h)
    }

    /// Build a PDF from HTML + CSS with a single optional embedded font.
    public static func fromHtmlCss(html: String, css: String, fontBytes: [UInt8] = []) throws -> Pdf {
        var code: Int32 = 0
        let h = fontBytes.withUnsafeBufferPointer { buf in
            pdf_from_html_css(html, css, fontBytes.isEmpty ? nil : buf.baseAddress, buf.count, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "fromHtmlCss") }
        return Pdf(h)
    }

    /// Build a PDF from HTML + CSS with a multi-font cascade. `families` and
    /// `fonts` are parallel arrays.
    public static func fromHtmlCssWithFonts(html: String, css: String, families: [String], fonts: [[UInt8]]) throws -> Pdf {
        var code: Int32 = 0
        let h = withCStringArray(families) { famPtr -> OpaquePointer? in
            withByteArrayArray(fonts) { fontPtrs, fontLens in
                pdf_from_html_css_with_fonts(html, css, famPtr, fontPtrs, fontLens, UInt(families.count), &code)
            }
        }
        guard let h else { throw PdfOxideError(code: code, op: "fromHtmlCssWithFonts") }
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

/// An opened PDF for in-place editing (rotate / crop / redact / flatten / merge / save).
///
/// Wraps every `document_editor_*` C function. Status-returning functions throw
/// `PdfOxideError` on a non-zero status or a set error code; the `is_*` query
/// functions are surfaced as `Bool` (1 == true).
public final class DocumentEditor {
    private var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { document_editor_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "DocumentEditor is closed") }
        return h
    }

    // Copy a C byte buffer return into [UInt8] and free it via free_bytes.
    private func takeBytes(_ p: UnsafeMutablePointer<UInt8>?, _ len: Int, _ code: Int32, _ op: String) throws -> [UInt8] {
        guard let p else { throw PdfOxideError(code: code, op: op) }
        defer { free_bytes(p) }
        let n = len < 0 ? 0 : len
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    // ── open / lifecycle ─────────────────────────────────────────────────────

    /// Open a PDF for editing from a filesystem path.
    public static func openEditor(_ path: String) throws -> DocumentEditor {
        var code: Int32 = 0
        guard let h = document_editor_open(path, &code) else {
            throw PdfOxideError(code: code, op: "openEditor")
        }
        return DocumentEditor(h)
    }

    /// Alias for `openEditor(_:)`.
    public static func open(_ path: String) throws -> DocumentEditor { try openEditor(path) }

    /// Open a PDF for editing from in-memory bytes.
    public static func openFromBytes(_ bytes: [UInt8]) throws -> DocumentEditor {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            document_editor_open_from_bytes(buf.baseAddress, buf.count, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "openFromBytes") }
        return DocumentEditor(h)
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { document_editor_free(h); handle = nil }
    }

    /// Alias for `close()`.
    public func free() { close() }

    // ── document-level queries ───────────────────────────────────────────────

    public func pageCount() throws -> Int {
        var code: Int32 = 0
        let n = document_editor_get_page_count(try ptr(), &code)
        if n < 0 { throw PdfOxideError(code: code, op: "pageCount") }
        return Int(n)
    }

    public func version() throws -> PdfVersion {
        var major: UInt8 = 0, minor: UInt8 = 0
        document_editor_get_version(try ptr(), &major, &minor)
        return PdfVersion(major: Int(major), minor: Int(minor))
    }

    public func isModified() throws -> Bool { document_editor_is_modified(try ptr()) }

    public func getSourcePath() throws -> String {
        var code: Int32 = 0
        return try takeString(document_editor_get_source_path(try ptr(), &code), code, "getSourcePath")
    }

    public func getProducer() throws -> String {
        var code: Int32 = 0
        return try takeString(document_editor_get_producer(try ptr(), &code), code, "getProducer")
    }
    public func setProducer(_ value: String) throws {
        var code: Int32 = 0
        if document_editor_set_producer(try ptr(), value, &code) != 0 {
            throw PdfOxideError(code: code, op: "setProducer")
        }
    }

    public func getCreationDate() throws -> String {
        var code: Int32 = 0
        return try takeString(document_editor_get_creation_date(try ptr(), &code), code, "getCreationDate")
    }
    public func setCreationDate(_ date: String) throws {
        var code: Int32 = 0
        if document_editor_set_creation_date(try ptr(), date, &code) != 0 {
            throw PdfOxideError(code: code, op: "setCreationDate")
        }
    }

    // ── page operations ──────────────────────────────────────────────────────

    public func deletePage(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_delete_page(try ptr(), Int32(page), &code) != 0 {
            throw PdfOxideError(code: code, op: "deletePage")
        }
    }

    public func movePage(_ from: Int, _ to: Int) throws {
        var code: Int32 = 0
        if document_editor_move_page(try ptr(), Int32(from), Int32(to), &code) != 0 {
            throw PdfOxideError(code: code, op: "movePage")
        }
    }

    public func rotatePageBy(_ page: Int, _ degrees: Int) throws {
        var code: Int32 = 0
        if document_editor_rotate_page_by(try ptr(), page, Int32(degrees), &code) != 0 {
            throw PdfOxideError(code: code, op: "rotatePageBy")
        }
    }

    public func rotateAllPages(_ degrees: Int) throws {
        var code: Int32 = 0
        if document_editor_rotate_all_pages(try ptr(), Int32(degrees), &code) != 0 {
            throw PdfOxideError(code: code, op: "rotateAllPages")
        }
    }

    public func setPageRotation(_ page: Int, _ degrees: Int) throws {
        var code: Int32 = 0
        if document_editor_set_page_rotation(try ptr(), Int32(page), Int32(degrees), &code) != 0 {
            throw PdfOxideError(code: code, op: "setPageRotation")
        }
    }

    public func getPageRotation(_ page: Int) throws -> Int {
        var code: Int32 = 0
        let r = document_editor_get_page_rotation(try ptr(), Int32(page), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "getPageRotation") }
        return Int(r)
    }

    public func cropMargins(left: Float, right: Float, top: Float, bottom: Float) throws {
        var code: Int32 = 0
        if document_editor_crop_margins(try ptr(), left, right, top, bottom, &code) != 0 {
            throw PdfOxideError(code: code, op: "cropMargins")
        }
    }

    // ── page boxes ───────────────────────────────────────────────────────────

    public func getPageCropBox(_ page: Int) throws -> Bbox {
        var code: Int32 = 0
        var x = 0.0, y = 0.0, w = 0.0, h = 0.0
        if document_editor_get_page_crop_box(try ptr(), page, &x, &y, &w, &h, &code) != 0 {
            throw PdfOxideError(code: code, op: "getPageCropBox")
        }
        return Bbox(x: x, y: y, width: w, height: h)
    }
    public func setPageCropBox(_ page: Int, x: Double, y: Double, width: Double, height: Double) throws {
        var code: Int32 = 0
        if document_editor_set_page_crop_box(try ptr(), page, x, y, width, height, &code) != 0 {
            throw PdfOxideError(code: code, op: "setPageCropBox")
        }
    }

    public func getPageMediaBox(_ page: Int) throws -> Bbox {
        var code: Int32 = 0
        var x = 0.0, y = 0.0, w = 0.0, h = 0.0
        if document_editor_get_page_media_box(try ptr(), page, &x, &y, &w, &h, &code) != 0 {
            throw PdfOxideError(code: code, op: "getPageMediaBox")
        }
        return Bbox(x: x, y: y, width: w, height: h)
    }
    public func setPageMediaBox(_ page: Int, x: Double, y: Double, width: Double, height: Double) throws {
        var code: Int32 = 0
        if document_editor_set_page_media_box(try ptr(), page, x, y, width, height, &code) != 0 {
            throw PdfOxideError(code: code, op: "setPageMediaBox")
        }
    }

    // ── redaction / erase ────────────────────────────────────────────────────

    public func applyAllRedactions() throws {
        var code: Int32 = 0
        if document_editor_apply_all_redactions(try ptr(), &code) != 0 {
            throw PdfOxideError(code: code, op: "applyAllRedactions")
        }
    }
    public func applyPageRedactions(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_apply_page_redactions(try ptr(), page, &code) != 0 {
            throw PdfOxideError(code: code, op: "applyPageRedactions")
        }
    }

    public func eraseRegion(_ page: Int, x: Float, y: Float, width: Float, height: Float) throws {
        var code: Int32 = 0
        if document_editor_erase_region(try ptr(), Int32(page), x, y, width, height, &code) != 0 {
            throw PdfOxideError(code: code, op: "eraseRegion")
        }
    }

    /// Erase multiple regions on a page. Each rectangle is `(x, y, width, height)`.
    public func eraseRegions(_ page: Int, _ rects: [(Double, Double, Double, Double)]) throws {
        let h = try ptr()
        var code: Int32 = 0
        var flat: [Double] = []
        flat.reserveCapacity(rects.count * 4)
        for r in rects { flat.append(r.0); flat.append(r.1); flat.append(r.2); flat.append(r.3) }
        let status = flat.withUnsafeBufferPointer { buf in
            document_editor_erase_regions(h, page, buf.baseAddress, rects.count, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "eraseRegions") }
    }

    public func clearEraseRegions(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_clear_erase_regions(try ptr(), page, &code) != 0 {
            throw PdfOxideError(code: code, op: "clearEraseRegions")
        }
    }

    /// 1 == marked, 0 == not. Throws on a -1 error status.
    public func isPageMarkedForRedaction(_ page: Int) throws -> Bool {
        let r = document_editor_is_page_marked_for_redaction(try ptr(), page)
        if r < 0 { throw PdfOxideError(code: r, op: "isPageMarkedForRedaction") }
        return r == 1
    }
    public func unmarkPageForRedaction(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_unmark_page_for_redaction(try ptr(), page, &code) != 0 {
            throw PdfOxideError(code: code, op: "unmarkPageForRedaction")
        }
    }

    // ── Phase-7 redaction (geometric add / apply / scrub) ────────────────────

    /// Queue a redaction rectangle (`x1,y1`–`x2,y2`) with an overlay colour
    /// (`r,g,b`, DeviceRGB 0..1) on a (0-based) page.
    public func redactionAdd(_ page: Int, x1: Double, y1: Double, x2: Double, y2: Double, r: Double, g: Double, b: Double) throws {
        var code: Int32 = 0
        if pdf_redaction_add(try ptr(), UInt(page), x1, y1, x2, y2, r, g, b, &code) != 0 {
            throw PdfOxideError(code: code, op: "redactionAdd")
        }
    }

    /// Number of queued redaction regions for a (0-based) page.
    public func redactionCount(_ page: Int) throws -> Int {
        var code: Int32 = 0
        let n = pdf_redaction_count(try ptr(), UInt(page), &code)
        if n < 0 { throw PdfOxideError(code: code, op: "redactionCount") }
        return Int(n)
    }

    /// Destructively apply all queued redactions. Returns the number of glyphs
    /// physically removed. `r,g,b` is the overlay colour (DeviceRGB 0..1).
    @discardableResult
    public func redactionApply(scrubMetadata: Bool, r: Double, g: Double, b: Double) throws -> Int {
        var code: Int32 = 0
        let n = pdf_redaction_apply(try ptr(), scrubMetadata, r, g, b, &code)
        if n < 0 { throw PdfOxideError(code: code, op: "redactionApply") }
        return Int(n)
    }

    /// Strip document metadata / JavaScript / embedded files without geometric
    /// redaction. Returns the number of top-level constructs removed.
    @discardableResult
    public func redactionScrubMetadata() throws -> Int {
        var code: Int32 = 0
        let n = pdf_redaction_scrub_metadata(try ptr(), &code)
        if n < 0 { throw PdfOxideError(code: code, op: "redactionScrubMetadata") }
        return Int(n)
    }

    // ── Phase-7 barcode placement ────────────────────────────────────────────

    /// Draw `barcode` onto a (0-based) page at `(x, y)` sized `width`×`height`
    /// (PDF user-space points).
    public func addBarcodeToPage(_ page: Int, _ barcode: BarcodeImage, x: Float, y: Float, width: Float, height: Float) throws {
        let h = try ptr()
        guard let bc = barcode.handle else {
            throw PdfOxideError(code: 0, op: "addBarcodeToPage: barcode is closed")
        }
        var code: Int32 = 0
        if pdf_add_barcode_to_page(h, Int32(page), bc, x, y, width, height, &code) != 0 {
            throw PdfOxideError(code: code, op: "addBarcodeToPage")
        }
    }

    // ── flattening (forms + annotations) ─────────────────────────────────────

    public func flattenForms() throws {
        var code: Int32 = 0
        if document_editor_flatten_forms(try ptr(), &code) != 0 {
            throw PdfOxideError(code: code, op: "flattenForms")
        }
    }
    public func flattenFormsOnPage(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_flatten_forms_on_page(try ptr(), Int32(page), &code) != 0 {
            throw PdfOxideError(code: code, op: "flattenFormsOnPage")
        }
    }

    public func flattenAnnotations(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_flatten_annotations(try ptr(), Int32(page), &code) != 0 {
            throw PdfOxideError(code: code, op: "flattenAnnotations")
        }
    }
    public func flattenAllAnnotations() throws {
        var code: Int32 = 0
        if document_editor_flatten_all_annotations(try ptr(), &code) != 0 {
            throw PdfOxideError(code: code, op: "flattenAllAnnotations")
        }
    }

    /// Number of warnings from the last form-flattening save (-1 if handle null).
    public func flattenWarningsCount() throws -> Int {
        Int(document_editor_flatten_warnings_count(try ptr()))
    }
    public func flattenWarning(_ index: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(document_editor_flatten_warning(try ptr(), Int32(index), &code), code, "flattenWarning")
    }

    /// 1 == marked for flatten, 0 == not. Throws on a -1 error status.
    public func isPageMarkedForFlatten(_ page: Int) throws -> Bool {
        let r = document_editor_is_page_marked_for_flatten(try ptr(), page)
        if r < 0 { throw PdfOxideError(code: r, op: "isPageMarkedForFlatten") }
        return r == 1
    }
    public func unmarkPageForFlatten(_ page: Int) throws {
        var code: Int32 = 0
        if document_editor_unmark_page_for_flatten(try ptr(), page, &code) != 0 {
            throw PdfOxideError(code: code, op: "unmarkPageForFlatten")
        }
    }

    // ── forms / merge / embed / convert ──────────────────────────────────────

    public func setFormFieldValue(_ name: String, _ value: String) throws {
        var code: Int32 = 0
        if document_editor_set_form_field_value(try ptr(), name, value, &code) != 0 {
            throw PdfOxideError(code: code, op: "setFormFieldValue")
        }
    }

    public func mergeFrom(_ sourcePath: String) throws {
        var code: Int32 = 0
        if document_editor_merge_from(try ptr(), sourcePath, &code) != 0 {
            throw PdfOxideError(code: code, op: "mergeFrom")
        }
    }
    public func mergeFromBytes(_ bytes: [UInt8]) throws {
        let h = try ptr()
        var code: Int32 = 0
        let status = bytes.withUnsafeBufferPointer { buf in
            document_editor_merge_from_bytes(h, buf.baseAddress, buf.count, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "mergeFromBytes") }
    }

    /// Convert to PDF/A in place. level: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
    public func convertToPdfA(_ level: Int) throws {
        var code: Int32 = 0
        if document_editor_convert_to_pdf_a(try ptr(), Int32(level), &code) != 0 {
            throw PdfOxideError(code: code, op: "convertToPdfA")
        }
    }

    public func embedFile(_ name: String, _ data: [UInt8]) throws {
        let h = try ptr()
        var code: Int32 = 0
        let status = data.withUnsafeBufferPointer { buf in
            document_editor_embed_file(h, name, buf.baseAddress, buf.count, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "embedFile") }
    }

    /// Extract a subset of (0-based) pages to a new in-memory PDF.
    public func extractPagesToBytes(_ pages: [Int]) throws -> [UInt8] {
        let h = try ptr()
        var code: Int32 = 0
        var len = 0
        let idx = pages.map { Int32($0) }
        let p = idx.withUnsafeBufferPointer { buf in
            document_editor_extract_pages_to_bytes(h, buf.baseAddress, pages.count, &len, &code)
        }
        return try takeBytes(p, len, code, "extractPagesToBytes")
    }

    // ── save ─────────────────────────────────────────────────────────────────

    public func save(_ path: String) throws {
        var code: Int32 = 0
        if document_editor_save(try ptr(), path, &code) != 0 {
            throw PdfOxideError(code: code, op: "save")
        }
    }

    public func saveToBytes() throws -> [UInt8] {
        var code: Int32 = 0
        var len = 0
        let p = document_editor_save_to_bytes(try ptr(), &len, &code)
        return try takeBytes(p, len, code, "saveToBytes")
    }

    public func saveToBytesWithOptions(compress: Bool, garbageCollect: Bool, linearize: Bool) throws -> [UInt8] {
        var code: Int32 = 0
        var len = 0
        let p = document_editor_save_to_bytes_with_options(try ptr(), compress, garbageCollect, linearize, &len, &code)
        return try takeBytes(p, len, code, "saveToBytesWithOptions")
    }

    public func saveEncrypted(_ path: String, userPassword: String, ownerPassword: String) throws {
        var code: Int32 = 0
        if document_editor_save_encrypted(try ptr(), path, userPassword, ownerPassword, &code) != 0 {
            throw PdfOxideError(code: code, op: "saveEncrypted")
        }
    }

    public func saveEncryptedToBytes(userPassword: String, ownerPassword: String) throws -> [UInt8] {
        var code: Int32 = 0
        var len = 0
        let p = document_editor_save_encrypted_to_bytes(try ptr(), userPassword, ownerPassword, &len, &code)
        return try takeBytes(p, len, code, "saveEncryptedToBytes")
    }
}

// ── PDF creation: builder API ────────────────────────────────────────────────

/// A loadable TTF/OTF font, ready to be registered with a `DocumentBuilder`.
///
/// Wraps the `pdf_embedded_font_*` C functions. The native handle is freed in
/// `deinit`/`close()` — BUT a successful `DocumentBuilder.registerEmbeddedFont`
/// **consumes** the handle (the builder takes ownership); this wrapper nulls its
/// handle out at that point so it is not double-freed.
public final class EmbeddedFont {
    fileprivate var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_embedded_font_free(h) } }

    /// Load a TTF/OTF font from a filesystem path.
    public static func fromFile(_ path: String) throws -> EmbeddedFont {
        var code: Int32 = 0
        guard let h = pdf_embedded_font_from_file(path, &code) else {
            throw PdfOxideError(code: code, op: "EmbeddedFont.fromFile")
        }
        return EmbeddedFont(h)
    }

    /// Load a font from a byte buffer. `name` may be nil to use the font's own
    /// PostScript name.
    public static func fromBytes(_ bytes: [UInt8], name: String? = nil) throws -> EmbeddedFont {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_embedded_font_from_bytes(buf.baseAddress, buf.count, name, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "EmbeddedFont.fromBytes") }
        return EmbeddedFont(h)
    }

    // The builder takes ownership on a successful register; this releases the
    // wrapper's claim so deinit/close won't double-free.
    fileprivate func release() -> OpaquePointer? {
        let h = handle
        handle = nil
        return h
    }

    /// Free the native handle now (idempotent). No-op once consumed by a builder.
    public func close() {
        if let h = handle { pdf_embedded_font_free(h); handle = nil }
    }
}

/// A page being built inside a `DocumentBuilder`.
///
/// Wraps every `pdf_page_builder_*` C function as a fluent op (each returns
/// `self`, throwing on a non-success status). Obtain one from
/// `DocumentBuilder.page(_:_:)` / `.a4Page()` / `.letterPage()`. Commit it with
/// `done()` (consumes the native handle) or discard it with `close()`. The
/// native handle is also freed in `deinit` if neither was called.
public final class PageBuilder {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_page_builder_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "PageBuilder is closed") }
        return h
    }

    // Run a status-returning op and return self for chaining.
    @discardableResult
    private func op(_ name: String, _ body: (OpaquePointer, inout Int32) -> Int32) throws -> PageBuilder {
        let h = try ptr()
        var code: Int32 = 0
        if body(h, &code) != 0 { throw PdfOxideError(code: code, op: name) }
        return self
    }

    // ── text / layout ─────────────────────────────────────────────────────────

    @discardableResult public func font(_ name: String, _ size: Float) throws -> PageBuilder {
        try op("font") { pdf_page_builder_font($0, name, size, &$1) }
    }
    @discardableResult public func at(_ x: Float, _ y: Float) throws -> PageBuilder {
        try op("at") { pdf_page_builder_at($0, x, y, &$1) }
    }
    @discardableResult public func text(_ text: String) throws -> PageBuilder {
        try op("text") { pdf_page_builder_text($0, text, &$1) }
    }
    @discardableResult public func heading(_ level: Int, _ text: String) throws -> PageBuilder {
        try op("heading") { pdf_page_builder_heading($0, UInt8(level), text, &$1) }
    }
    @discardableResult public func paragraph(_ text: String) throws -> PageBuilder {
        try op("paragraph") { pdf_page_builder_paragraph($0, text, &$1) }
    }
    @discardableResult public func space(_ points: Float) throws -> PageBuilder {
        try op("space") { pdf_page_builder_space($0, points, &$1) }
    }
    @discardableResult public func horizontalRule() throws -> PageBuilder {
        try op("horizontalRule") { pdf_page_builder_horizontal_rule($0, &$1) }
    }
    @discardableResult public func columns(_ columnCount: UInt32, _ gapPt: Float, _ text: String) throws -> PageBuilder {
        try op("columns") { pdf_page_builder_columns($0, columnCount, gapPt, text, &$1) }
    }
    @discardableResult public func footnote(_ refMark: String, _ noteText: String) throws -> PageBuilder {
        try op("footnote") { pdf_page_builder_footnote($0, refMark, noteText, &$1) }
    }

    // ── inline runs ─────────────────────────────────────────────────────────────

    @discardableResult public func inline(_ text: String) throws -> PageBuilder {
        try op("inline") { pdf_page_builder_inline($0, text, &$1) }
    }
    @discardableResult public func inlineBold(_ text: String) throws -> PageBuilder {
        try op("inlineBold") { pdf_page_builder_inline_bold($0, text, &$1) }
    }
    @discardableResult public func inlineItalic(_ text: String) throws -> PageBuilder {
        try op("inlineItalic") { pdf_page_builder_inline_italic($0, text, &$1) }
    }
    @discardableResult public func inlineColor(_ r: Float, _ g: Float, _ b: Float, _ text: String) throws -> PageBuilder {
        try op("inlineColor") { pdf_page_builder_inline_color($0, r, g, b, text, &$1) }
    }
    @discardableResult public func newline() throws -> PageBuilder {
        try op("newline") { pdf_page_builder_newline($0, &$1) }
    }

    // ── links ─────────────────────────────────────────────────────────────────

    @discardableResult public func linkUrl(_ url: String) throws -> PageBuilder {
        try op("linkUrl") { pdf_page_builder_link_url($0, url, &$1) }
    }
    @discardableResult public func linkPage(_ page: Int) throws -> PageBuilder {
        try op("linkPage") { pdf_page_builder_link_page($0, page, &$1) }
    }
    @discardableResult public func linkNamed(_ destination: String) throws -> PageBuilder {
        try op("linkNamed") { pdf_page_builder_link_named($0, destination, &$1) }
    }
    @discardableResult public func linkJavascript(_ script: String) throws -> PageBuilder {
        try op("linkJavascript") { pdf_page_builder_link_javascript($0, script, &$1) }
    }

    // ── page / field actions ─────────────────────────────────────────────────────

    @discardableResult public func onOpen(_ script: String) throws -> PageBuilder {
        try op("onOpen") { pdf_page_builder_on_open($0, script, &$1) }
    }
    @discardableResult public func onClose(_ script: String) throws -> PageBuilder {
        try op("onClose") { pdf_page_builder_on_close($0, script, &$1) }
    }
    @discardableResult public func fieldKeystroke(_ script: String) throws -> PageBuilder {
        try op("fieldKeystroke") { pdf_page_builder_field_keystroke($0, script, &$1) }
    }
    @discardableResult public func fieldFormat(_ script: String) throws -> PageBuilder {
        try op("fieldFormat") { pdf_page_builder_field_format($0, script, &$1) }
    }
    @discardableResult public func fieldValidate(_ script: String) throws -> PageBuilder {
        try op("fieldValidate") { pdf_page_builder_field_validate($0, script, &$1) }
    }
    @discardableResult public func fieldCalculate(_ script: String) throws -> PageBuilder {
        try op("fieldCalculate") { pdf_page_builder_field_calculate($0, script, &$1) }
    }

    // ── text-markup annotations ──────────────────────────────────────────────────

    @discardableResult public func highlight(_ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("highlight") { pdf_page_builder_highlight($0, r, g, b, &$1) }
    }
    @discardableResult public func underline(_ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("underline") { pdf_page_builder_underline($0, r, g, b, &$1) }
    }
    @discardableResult public func strikeout(_ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("strikeout") { pdf_page_builder_strikeout($0, r, g, b, &$1) }
    }
    @discardableResult public func squiggly(_ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("squiggly") { pdf_page_builder_squiggly($0, r, g, b, &$1) }
    }

    // ── notes / stamps / watermarks ──────────────────────────────────────────────

    @discardableResult public func stickyNote(_ text: String) throws -> PageBuilder {
        try op("stickyNote") { pdf_page_builder_sticky_note($0, text, &$1) }
    }
    @discardableResult public func stickyNoteAt(_ x: Float, _ y: Float, _ text: String) throws -> PageBuilder {
        try op("stickyNoteAt") { pdf_page_builder_sticky_note_at($0, x, y, text, &$1) }
    }
    @discardableResult public func watermark(_ text: String) throws -> PageBuilder {
        try op("watermark") { pdf_page_builder_watermark($0, text, &$1) }
    }
    @discardableResult public func watermarkConfidential() throws -> PageBuilder {
        try op("watermarkConfidential") { pdf_page_builder_watermark_confidential($0, &$1) }
    }
    @discardableResult public func watermarkDraft() throws -> PageBuilder {
        try op("watermarkDraft") { pdf_page_builder_watermark_draft($0, &$1) }
    }
    @discardableResult public func stamp(_ typeName: String) throws -> PageBuilder {
        try op("stamp") { pdf_page_builder_stamp($0, typeName, &$1) }
    }
    @discardableResult public func freetext(_ x: Float, _ y: Float, _ w: Float, _ h: Float, _ text: String) throws -> PageBuilder {
        try op("freetext") { pdf_page_builder_freetext($0, x, y, w, h, text, &$1) }
    }

    // ── form fields ──────────────────────────────────────────────────────────────

    @discardableResult public func textField(_ name: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float, defaultValue: String? = nil) throws -> PageBuilder {
        try op("textField") { pdf_page_builder_text_field($0, name, x, y, w, h, defaultValue, &$1) }
    }
    @discardableResult public func checkbox(_ name: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float, checked: Bool) throws -> PageBuilder {
        try op("checkbox") { pdf_page_builder_checkbox($0, name, x, y, w, h, checked ? 1 : 0, &$1) }
    }
    @discardableResult public func pushButton(_ name: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float, _ caption: String) throws -> PageBuilder {
        try op("pushButton") { pdf_page_builder_push_button($0, name, x, y, w, h, caption, &$1) }
    }
    @discardableResult public func signatureField(_ name: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float) throws -> PageBuilder {
        try op("signatureField") { pdf_page_builder_signature_field($0, name, x, y, w, h, &$1) }
    }

    /// Add a dropdown combo-box. `selected` may be nil for no initial selection.
    @discardableResult public func comboBox(_ name: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float, options: [String], selected: String? = nil) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = withCStringArray(options) { optsPtr in
            pdf_page_builder_combo_box(h0, name, x, y, w, h, optsPtr, options.count, selected, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "comboBox") }
        return self
    }

    /// Add a radio-button group. `values`/`xs`/`ys`/`ws`/`hs` are parallel arrays.
    @discardableResult public func radioGroup(_ name: String, values: [String], xs: [Float], ys: [Float], ws: [Float], hs: [Float], selected: String? = nil) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let count = values.count
        let status = withCStringArray(values) { valsPtr in
            xs.withUnsafeBufferPointer { xsP in
                ys.withUnsafeBufferPointer { ysP in
                    ws.withUnsafeBufferPointer { wsP in
                        hs.withUnsafeBufferPointer { hsP in
                            pdf_page_builder_radio_group(
                                h0, name, valsPtr,
                                xsP.baseAddress, ysP.baseAddress, wsP.baseAddress, hsP.baseAddress,
                                count, selected, &code
                            )
                        }
                    }
                }
            }
        }
        if status != 0 { throw PdfOxideError(code: code, op: "radioGroup") }
        return self
    }

    // ── barcodes ─────────────────────────────────────────────────────────────────

    @discardableResult public func barcode1d(_ barcodeType: Int32, _ data: String, _ x: Float, _ y: Float, _ w: Float, _ h: Float) throws -> PageBuilder {
        try op("barcode1d") { pdf_page_builder_barcode_1d($0, barcodeType, data, x, y, w, h, &$1) }
    }
    @discardableResult public func barcodeQr(_ data: String, _ x: Float, _ y: Float, _ size: Float) throws -> PageBuilder {
        try op("barcodeQr") { pdf_page_builder_barcode_qr($0, data, x, y, size, &$1) }
    }

    // ── images ───────────────────────────────────────────────────────────────────

    @discardableResult public func image(_ bytes: [UInt8], _ x: Float, _ y: Float, _ w: Float, _ h: Float) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = bytes.withUnsafeBufferPointer { buf in
            pdf_page_builder_image(h0, buf.baseAddress, buf.count, x, y, w, h, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "image") }
        return self
    }
    @discardableResult public func imageWithAlt(_ bytes: [UInt8], _ x: Float, _ y: Float, _ w: Float, _ h: Float, altText: String) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = bytes.withUnsafeBufferPointer { buf in
            pdf_page_builder_image_with_alt(h0, buf.baseAddress, buf.count, x, y, w, h, altText, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "imageWithAlt") }
        return self
    }
    @discardableResult public func imageArtifact(_ bytes: [UInt8], _ x: Float, _ y: Float, _ w: Float, _ h: Float) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = bytes.withUnsafeBufferPointer { buf in
            pdf_page_builder_image_artifact(h0, buf.baseAddress, buf.count, x, y, w, h, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "imageArtifact") }
        return self
    }

    // ── vector graphics ──────────────────────────────────────────────────────────

    @discardableResult public func rect(_ x: Float, _ y: Float, _ w: Float, _ h: Float) throws -> PageBuilder {
        try op("rect") { pdf_page_builder_rect($0, x, y, w, h, &$1) }
    }
    @discardableResult public func filledRect(_ x: Float, _ y: Float, _ w: Float, _ h: Float, _ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("filledRect") { pdf_page_builder_filled_rect($0, x, y, w, h, r, g, b, &$1) }
    }
    @discardableResult public func line(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float) throws -> PageBuilder {
        try op("line") { pdf_page_builder_line($0, x1, y1, x2, y2, &$1) }
    }
    @discardableResult public func strokeRect(_ x: Float, _ y: Float, _ w: Float, _ h: Float, width: Float, _ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("strokeRect") { pdf_page_builder_stroke_rect($0, x, y, w, h, width, r, g, b, &$1) }
    }
    @discardableResult public func strokeLine(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float, width: Float, _ r: Float, _ g: Float, _ b: Float) throws -> PageBuilder {
        try op("strokeLine") { pdf_page_builder_stroke_line($0, x1, y1, x2, y2, width, r, g, b, &$1) }
    }

    @discardableResult public func strokeRectDashed(_ x: Float, _ y: Float, _ w: Float, _ h: Float, width: Float, _ r: Float, _ g: Float, _ b: Float, dashArray: [Float], phase: Float) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = dashArray.withUnsafeBufferPointer { buf in
            pdf_page_builder_stroke_rect_dashed(h0, x, y, w, h, width, r, g, b, buf.baseAddress, dashArray.count, phase, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "strokeRectDashed") }
        return self
    }
    @discardableResult public func strokeLineDashed(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float, width: Float, _ r: Float, _ g: Float, _ b: Float, dashArray: [Float], phase: Float) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = dashArray.withUnsafeBufferPointer { buf in
            pdf_page_builder_stroke_line_dashed(h0, x1, y1, x2, y2, width, r, g, b, buf.baseAddress, dashArray.count, phase, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "strokeLineDashed") }
        return self
    }

    @discardableResult public func textInRect(_ x: Float, _ y: Float, _ w: Float, _ h: Float, _ text: String, align: Int32) throws -> PageBuilder {
        try op("textInRect") { pdf_page_builder_text_in_rect($0, x, y, w, h, text, align, &$1) }
    }
    @discardableResult public func newPageSameSize() throws -> PageBuilder {
        try op("newPageSameSize") { pdf_page_builder_new_page_same_size($0, &$1) }
    }

    // ── tables ───────────────────────────────────────────────────────────────────

    /// Buffer a static table. `cellStrings` is row-major (`row * nColumns + col`).
    @discardableResult public func table(nColumns: Int, widths: [Float], aligns: [Int32], nRows: Int, cellStrings: [String], hasHeader: Bool) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = widths.withUnsafeBufferPointer { wP in
            aligns.withUnsafeBufferPointer { aP in
                withCStringArray(cellStrings) { cellsPtr in
                    pdf_page_builder_table(h0, nColumns, wP.baseAddress, aP.baseAddress, nRows, cellsPtr, hasHeader ? 1 : 0, &code)
                }
            }
        }
        if status != 0 { throw PdfOxideError(code: code, op: "table") }
        return self
    }

    @discardableResult public func streamingTableBegin(nColumns: Int, headers: [String], widths: [Float], aligns: [Int32], repeatHeader: Bool) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = widths.withUnsafeBufferPointer { wP in
            aligns.withUnsafeBufferPointer { aP in
                withCStringArray(headers) { hdrPtr in
                    pdf_page_builder_streaming_table_begin(h0, nColumns, hdrPtr, wP.baseAddress, aP.baseAddress, repeatHeader ? 1 : 0, &code)
                }
            }
        }
        if status != 0 { throw PdfOxideError(code: code, op: "streamingTableBegin") }
        return self
    }

    @discardableResult public func streamingTableBeginV2(nColumns: Int, headers: [String], widths: [Float], aligns: [Int32], repeatHeader: Bool, mode: Int32, sampleRows: Int, minColWidthPt: Float, maxColWidthPt: Float, maxRowspan: Int) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = widths.withUnsafeBufferPointer { wP in
            aligns.withUnsafeBufferPointer { aP in
                withCStringArray(headers) { hdrPtr in
                    pdf_page_builder_streaming_table_begin_v2(h0, nColumns, hdrPtr, wP.baseAddress, aP.baseAddress, repeatHeader ? 1 : 0, mode, sampleRows, minColWidthPt, maxColWidthPt, maxRowspan, &code)
                }
            }
        }
        if status != 0 { throw PdfOxideError(code: code, op: "streamingTableBeginV2") }
        return self
    }

    @discardableResult public func streamingTableSetBatchSize(_ batchSize: Int) throws -> PageBuilder {
        try op("streamingTableSetBatchSize") { pdf_page_builder_streaming_table_set_batch_size($0, batchSize, &$1) }
    }
    public func streamingTablePendingRowCount() throws -> Int {
        Int(pdf_page_builder_streaming_table_pending_row_count(try ptr()))
    }
    public func streamingTableBatchCount() throws -> Int {
        Int(pdf_page_builder_streaming_table_batch_count(try ptr()))
    }
    @discardableResult public func streamingTableFlush() throws -> PageBuilder {
        try op("streamingTableFlush") { pdf_page_builder_streaming_table_flush($0, &$1) }
    }

    @discardableResult public func streamingTablePushRow(_ cells: [String]) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let status = withCStringArray(cells) { cellsPtr in
            pdf_page_builder_streaming_table_push_row(h0, cells.count, cellsPtr, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "streamingTablePushRow") }
        return self
    }

    @discardableResult public func streamingTablePushRowV2(_ cells: [String], rowspans: [Int]?) throws -> PageBuilder {
        let h0 = try ptr()
        var code: Int32 = 0
        let spans = rowspans
        let status = withCStringArray(cells) { cellsPtr -> Int32 in
            if let spans {
                return spans.withUnsafeBufferPointer { sP in
                    pdf_page_builder_streaming_table_push_row_v2(h0, cells.count, cellsPtr, sP.baseAddress, &code)
                }
            }
            return pdf_page_builder_streaming_table_push_row_v2(h0, cells.count, cellsPtr, nil, &code)
        }
        if status != 0 { throw PdfOxideError(code: code, op: "streamingTablePushRowV2") }
        return self
    }

    @discardableResult public func streamingTableFinish() throws -> PageBuilder {
        try op("streamingTableFinish") { pdf_page_builder_streaming_table_finish($0, &$1) }
    }

    // ── commit / discard ─────────────────────────────────────────────────────────

    /// Commit this page's buffered ops to its parent builder. **Consumes** the
    /// native handle — the wrapper is invalid afterwards (idempotent guard).
    public func done() throws {
        let h = try ptr()
        var code: Int32 = 0
        if pdf_page_builder_done(h, &code) != 0 {
            throw PdfOxideError(code: code, op: "done")
        }
        // done() consumed the native handle; do not free it again.
        handle = nil
    }

    /// Drop this page's uncommitted handle (idempotent). Does NOT apply ops.
    public func close() {
        if let h = handle { pdf_page_builder_free(h); handle = nil }
    }
}

/// Builds a brand-new PDF from scratch.
///
/// Wraps every `pdf_document_builder_*` C function. Set metadata, start pages
/// via `page(_:_:)` / `a4Page()` / `letterPage()`, then `build()` / `save(_:)`.
/// The native handle is freed in `deinit`/`close()`.
public final class DocumentBuilder {
    private var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_document_builder_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "DocumentBuilder is closed") }
        return h
    }

    // Copy a C byte buffer return into [UInt8] and free it via free_bytes.
    private func takeBytes(_ p: UnsafeMutablePointer<UInt8>?, _ len: Int, _ code: Int32, _ op: String) throws -> [UInt8] {
        guard let p else { throw PdfOxideError(code: code, op: op) }
        defer { free_bytes(p) }
        let n = len < 0 ? 0 : len
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    @discardableResult
    private func op(_ name: String, _ body: (OpaquePointer, inout Int32) -> Int32) throws -> DocumentBuilder {
        let h = try ptr()
        var code: Int32 = 0
        if body(h, &code) != 0 { throw PdfOxideError(code: code, op: name) }
        return self
    }

    /// Create a new, empty document builder.
    public static func create() throws -> DocumentBuilder {
        var code: Int32 = 0
        guard let h = pdf_document_builder_create(&code) else {
            throw PdfOxideError(code: code, op: "DocumentBuilder.create")
        }
        return DocumentBuilder(h)
    }

    // ── metadata ─────────────────────────────────────────────────────────────────

    @discardableResult public func setTitle(_ title: String) throws -> DocumentBuilder {
        try op("setTitle") { pdf_document_builder_set_title($0, title, &$1) }
    }
    @discardableResult public func setAuthor(_ author: String) throws -> DocumentBuilder {
        try op("setAuthor") { pdf_document_builder_set_author($0, author, &$1) }
    }
    @discardableResult public func setSubject(_ subject: String) throws -> DocumentBuilder {
        try op("setSubject") { pdf_document_builder_set_subject($0, subject, &$1) }
    }
    @discardableResult public func setKeywords(_ keywords: String) throws -> DocumentBuilder {
        try op("setKeywords") { pdf_document_builder_set_keywords($0, keywords, &$1) }
    }
    @discardableResult public func setCreator(_ creator: String) throws -> DocumentBuilder {
        try op("setCreator") { pdf_document_builder_set_creator($0, creator, &$1) }
    }
    @discardableResult public func onOpen(_ script: String) throws -> DocumentBuilder {
        try op("onOpen") { pdf_document_builder_on_open($0, script, &$1) }
    }
    @discardableResult public func taggedPdfUa1() throws -> DocumentBuilder {
        try op("taggedPdfUa1") { pdf_document_builder_tagged_pdf_ua1($0, &$1) }
    }
    @discardableResult public func language(_ lang: String) throws -> DocumentBuilder {
        try op("language") { pdf_document_builder_language($0, lang, &$1) }
    }
    @discardableResult public func roleMap(custom: String, standard: String) throws -> DocumentBuilder {
        try op("roleMap") { pdf_document_builder_role_map($0, custom, standard, &$1) }
    }

    /// Register a TTF/OTF font under `name`. On success the builder **consumes**
    /// `font` (its handle is released so it won't be double-freed).
    @discardableResult public func registerEmbeddedFont(_ name: String, _ font: EmbeddedFont) throws -> DocumentBuilder {
        let h = try ptr()
        guard let fontHandle = font.handle else {
            throw PdfOxideError(code: 0, op: "registerEmbeddedFont: font already consumed")
        }
        var code: Int32 = 0
        if pdf_document_builder_register_embedded_font(h, name, fontHandle, &code) != 0 {
            // On error the font handle is NOT consumed; leave the wrapper owning it.
            throw PdfOxideError(code: code, op: "registerEmbeddedFont")
        }
        // Success: builder took ownership. Detach so deinit won't double-free.
        _ = font.release()
        return self
    }

    // ── pages ────────────────────────────────────────────────────────────────────

    /// Start an A4 page.
    public func a4Page() throws -> PageBuilder {
        var code: Int32 = 0
        guard let p = pdf_document_builder_a4_page(try ptr(), &code) else {
            throw PdfOxideError(code: code, op: "a4Page")
        }
        return PageBuilder(p)
    }
    /// Start a US Letter page.
    public func letterPage() throws -> PageBuilder {
        var code: Int32 = 0
        guard let p = pdf_document_builder_letter_page(try ptr(), &code) else {
            throw PdfOxideError(code: code, op: "letterPage")
        }
        return PageBuilder(p)
    }
    /// Start a custom-size page (dimensions in PDF points, 72pt = 1in).
    public func page(_ width: Float, _ height: Float) throws -> PageBuilder {
        var code: Int32 = 0
        guard let p = pdf_document_builder_page(try ptr(), width, height, &code) else {
            throw PdfOxideError(code: code, op: "page")
        }
        return PageBuilder(p)
    }

    // ── build / save ───────────────────────────────────────────────────────────

    /// Build the PDF and return its bytes. Consumes the builder *state*; the
    /// wrapper handle remains valid and is freed on `close()`/`deinit`.
    public func build() throws -> [UInt8] {
        var len = 0, code: Int32 = 0
        let p = pdf_document_builder_build(try ptr(), &len, &code)
        return try takeBytes(p, len, code, "build")
    }

    /// Build and save the PDF to `path`.
    public func save(_ path: String) throws {
        var code: Int32 = 0
        if pdf_document_builder_save(try ptr(), path, &code) != 0 {
            throw PdfOxideError(code: code, op: "save")
        }
    }

    /// Build and save with AES-256 encryption.
    public func saveEncrypted(_ path: String, userPassword: String, ownerPassword: String) throws {
        var code: Int32 = 0
        if pdf_document_builder_save_encrypted(try ptr(), path, userPassword, ownerPassword, &code) != 0 {
            throw PdfOxideError(code: code, op: "saveEncrypted")
        }
    }

    /// Build encrypted bytes (AES-256).
    public func toBytesEncrypted(userPassword: String, ownerPassword: String) throws -> [UInt8] {
        var len = 0, code: Int32 = 0
        let p = pdf_document_builder_to_bytes_encrypted(try ptr(), userPassword, ownerPassword, &len, &code)
        return try takeBytes(p, len, code, "toBytesEncrypted")
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_document_builder_free(h); handle = nil }
    }
}

// ── Phase-6: digital signatures / PKI / timestamps / TSA / validation ─────────

// Copy an owned C byte buffer return into [UInt8] and free it via free_bytes.
private func takeBytes(_ p: UnsafeMutablePointer<UInt8>?, _ len: Int, _ code: Int32, _ op: String) throws -> [UInt8] {
    guard let p else { throw PdfOxideError(code: code, op: op) }
    defer { free_bytes(p) }
    let n = len < 0 ? 0 : len
    return Array(UnsafeBufferPointer(start: p, count: n))
}

/// Validity window of a certificate, as Unix epoch seconds.
public struct CertificateValidity {
    public let notBefore: Int64
    public let notAfter: Int64
}

/// Per-PDF/UA structural statistics returned by `UaResults.stats()`.
public struct UaStats {
    public let structElements: Int
    public let images: Int
    public let tables: Int
    public let forms: Int
    public let annotations: Int
    public let pages: Int
}

/// A signing certificate / credentials handle (opaque, freed in `deinit`/`close()`).
///
/// Obtain one via `Certificate.loadFromBytes` (PKCS#12) or `Certificate.loadFromPem`,
/// or from `SignatureInfo.certificate()`. Wraps the `pdf_certificate_*` C functions.
public final class Certificate {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_certificate_free(UnsafeMutableRawPointer(h)) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "Certificate is closed") }
        return h
    }

    // Expose the raw handle (for signing) — throws if closed.
    fileprivate func rawPtr() throws -> UnsafeRawPointer {
        UnsafeRawPointer(try ptr())
    }

    /// Load a PKCS#12 (PFX) certificate + key from bytes, decrypting with `password`.
    public static func loadFromBytes(_ bytes: [UInt8], password: String) throws -> Certificate {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_certificate_load_from_bytes(buf.baseAddress, Int32(buf.count), password, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "Certificate.loadFromBytes") }
        return Certificate(OpaquePointer(h))
    }

    /// Load signing credentials from PEM-encoded certificate + private-key strings.
    public static func loadFromPem(certPem: String, keyPem: String) throws -> Certificate {
        var code: Int32 = 0
        guard let h = pdf_certificate_load_from_pem(certPem, keyPem, &code) else {
            throw PdfOxideError(code: code, op: "Certificate.loadFromPem")
        }
        return Certificate(OpaquePointer(h))
    }

    public func subject() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_certificate_get_subject(UnsafeRawPointer(try ptr()), &code), code, "Certificate.subject")
    }
    public func issuer() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_certificate_get_issuer(UnsafeRawPointer(try ptr()), &code), code, "Certificate.issuer")
    }
    public func serial() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_certificate_get_serial(UnsafeRawPointer(try ptr()), &code), code, "Certificate.serial")
    }

    /// The certificate's validity window (epoch seconds).
    public func validity() throws -> CertificateValidity {
        var code: Int32 = 0
        var notBefore: Int64 = 0, notAfter: Int64 = 0
        pdf_certificate_get_validity(UnsafeRawPointer(try ptr()), &notBefore, &notAfter, &code)
        if code != 0 { throw PdfOxideError(code: code, op: "Certificate.validity") }
        return CertificateValidity(notBefore: notBefore, notAfter: notAfter)
    }

    /// Whether the certificate is currently valid (1 == valid).
    public func isValid() throws -> Bool {
        var code: Int32 = 0
        return pdf_certificate_is_valid(UnsafeRawPointer(try ptr()), &code) == 1
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_certificate_free(UnsafeMutableRawPointer(h)); handle = nil }
    }
}

/// Information about a single PDF signature (opaque `FfiSignatureInfo`, freed in
/// `deinit`/`close()`). Wraps the `pdf_signature_*` C functions.
public final class SignatureInfo {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_signature_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "SignatureInfo is closed") }
        return h
    }

    public func signerName() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_signature_get_signer_name(try ptr(), &code), code, "SignatureInfo.signerName")
    }
    public func signingReason() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_signature_get_signing_reason(try ptr(), &code), code, "SignatureInfo.signingReason")
    }
    public func signingLocation() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_signature_get_signing_location(try ptr(), &code), code, "SignatureInfo.signingLocation")
    }

    /// Signing time as Unix epoch seconds.
    public func signingTime() throws -> Int64 {
        var code: Int32 = 0
        let t = pdf_signature_get_signing_time(try ptr(), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "SignatureInfo.signingTime") }
        return t
    }

    /// The signer's certificate, if available.
    public func certificate() throws -> Certificate? {
        var code: Int32 = 0
        guard let c = pdf_signature_get_certificate(UnsafeRawPointer(try ptr()), &code) else { return nil }
        return Certificate(OpaquePointer(c))
    }

    /// The PAdES level code classified from the signature's CMS attributes (-1 on error).
    public func padesLevel() throws -> Int32 {
        var code: Int32 = 0
        return pdf_signature_get_pades_level(UnsafeRawPointer(try ptr()), &code)
    }

    /// Whether this signature carries an embedded RFC 3161 timestamp.
    public func hasTimestamp() throws -> Bool {
        var code: Int32 = 0
        return pdf_signature_has_timestamp(UnsafeRawPointer(try ptr()), &code)
    }

    /// The signature's embedded timestamp, if any.
    public func timestamp() throws -> Timestamp? {
        var code: Int32 = 0
        guard let t = pdf_signature_get_timestamp(UnsafeRawPointer(try ptr()), &code) else { return nil }
        return Timestamp(OpaquePointer(t))
    }

    /// Attach `ts` to this signature. Returns true on success.
    @discardableResult
    public func addTimestamp(_ ts: Timestamp) throws -> Bool {
        var code: Int32 = 0
        let ok = pdf_signature_add_timestamp(UnsafeRawPointer(try ptr()), try ts.rawPtr(), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "SignatureInfo.addTimestamp") }
        return ok
    }

    /// Run the signer-attributes crypto check. Returns 1=valid, 0=invalid, -1=unknown/unsupported.
    public func verify() throws -> Int32 {
        var code: Int32 = 0
        return pdf_signature_verify(UnsafeRawPointer(try ptr()), &code)
    }

    /// End-to-end verify against the full PDF bytes. Returns 1=valid, 0=invalid, -1=unknown.
    public func verifyDetached(_ pdf: [UInt8]) throws -> Int32 {
        let h = UnsafeRawPointer(try ptr())
        var code: Int32 = 0
        return pdf.withUnsafeBufferPointer { buf in
            pdf_signature_verify_detached(h, buf.baseAddress, UInt(buf.count), &code)
        }
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_signature_free(h); handle = nil }
    }
}

/// A parsed RFC 3161 timestamp token (opaque, freed in `deinit`/`close()`).
/// Wraps the `pdf_timestamp_*` C functions.
public final class Timestamp {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_timestamp_free(UnsafeMutableRawPointer(h)) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "Timestamp is closed") }
        return h
    }
    fileprivate func rawPtr() throws -> UnsafeRawPointer { UnsafeRawPointer(try ptr()) }

    /// Parse a DER-encoded RFC 3161 TimeStampToken (or bare TSTInfo).
    public static func parse(_ bytes: [UInt8]) throws -> Timestamp {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_timestamp_parse(buf.baseAddress, UInt(buf.count), &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "Timestamp.parse") }
        return Timestamp(OpaquePointer(h))
    }

    // The token / message-imprint accessors return a BORROWED const pointer that
    // must be copied (NOT freed) — the bytes are owned by the handle.
    private func copyConstBytes(_ p: UnsafePointer<UInt8>?, _ len: Int, _ code: Int32, _ op: String) throws -> [UInt8] {
        guard let p else { throw PdfOxideError(code: code, op: op) }
        let n = len < 0 ? 0 : len
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// The raw timestamp token bytes (copied from a borrowed buffer).
    public func token() throws -> [UInt8] {
        let h = UnsafeRawPointer(try ptr())
        var outLen: UInt = 0, code: Int32 = 0
        let p = pdf_timestamp_get_token(h, &outLen, &code)
        return try copyConstBytes(p, Int(outLen), code, "Timestamp.token")
    }

    /// The message imprint bytes (copied from a borrowed buffer).
    public func messageImprint() throws -> [UInt8] {
        let h = UnsafeRawPointer(try ptr())
        var outLen: UInt = 0, code: Int32 = 0
        let p = pdf_timestamp_get_message_imprint(h, &outLen, &code)
        return try copyConstBytes(p, Int(outLen), code, "Timestamp.messageImprint")
    }

    /// Timestamp time as Unix epoch seconds.
    public func time() throws -> Int64 {
        var code: Int32 = 0
        let t = pdf_timestamp_get_time(UnsafeRawPointer(try ptr()), &code)
        if code != 0 { throw PdfOxideError(code: code, op: "Timestamp.time") }
        return t
    }
    public func serial() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_timestamp_get_serial(UnsafeRawPointer(try ptr()), &code), code, "Timestamp.serial")
    }
    public func tsaName() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_timestamp_get_tsa_name(UnsafeRawPointer(try ptr()), &code), code, "Timestamp.tsaName")
    }
    public func policyOid() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_timestamp_get_policy_oid(UnsafeRawPointer(try ptr()), &code), code, "Timestamp.policyOid")
    }
    public func hashAlgorithm() throws -> Int32 {
        var code: Int32 = 0
        return pdf_timestamp_get_hash_algorithm(UnsafeRawPointer(try ptr()), &code)
    }

    /// Verify the timestamp token's internal consistency.
    public func verify() throws -> Bool {
        var code: Int32 = 0
        return pdf_timestamp_verify(UnsafeRawPointer(try ptr()), &code)
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_timestamp_free(UnsafeMutableRawPointer(h)); handle = nil }
    }
}

/// An RFC 3161 Time-Stamp Authority client (opaque, freed in `deinit`/`close()`).
/// Wraps the `pdf_tsa_client_*` and `pdf_tsa_request_*` C functions.
public final class TsaClient {
    private var handle: OpaquePointer?

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_tsa_client_free(UnsafeMutableRawPointer(h)) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "TsaClient is closed") }
        return h
    }

    /// Create a TSA client. `username`/`password` may be nil for unauthenticated TSAs.
    public static func create(
        url: String, username: String? = nil, password: String? = nil,
        timeout: Int32 = 30, hashAlgo: Int32 = 0, useNonce: Bool = true, certReq: Bool = true
    ) throws -> TsaClient {
        var code: Int32 = 0
        guard let h = pdf_tsa_client_create(url, username, password, timeout, hashAlgo, useNonce, certReq, &code) else {
            throw PdfOxideError(code: code, op: "TsaClient.create")
        }
        return TsaClient(OpaquePointer(h))
    }

    /// Request a timestamp over `data` (the TSA hashes it). Returns the token.
    public func requestTimestamp(_ data: [UInt8]) throws -> Timestamp {
        let h = UnsafeRawPointer(try ptr())
        var code: Int32 = 0
        let t = data.withUnsafeBufferPointer { buf in
            pdf_tsa_request_timestamp(h, buf.baseAddress, UInt(buf.count), &code)
        }
        guard let t else { throw PdfOxideError(code: code, op: "TsaClient.requestTimestamp") }
        return Timestamp(OpaquePointer(t))
    }

    /// Request a timestamp over a pre-computed `hash` (algorithm = `hashAlgo`).
    public func requestTimestampHash(_ hash: [UInt8], hashAlgo: Int32) throws -> Timestamp {
        let h = UnsafeRawPointer(try ptr())
        var code: Int32 = 0
        let t = hash.withUnsafeBufferPointer { buf in
            pdf_tsa_request_timestamp_hash(h, buf.baseAddress, UInt(buf.count), hashAlgo, &code)
        }
        guard let t else { throw PdfOxideError(code: code, op: "TsaClient.requestTimestampHash") }
        return Timestamp(OpaquePointer(t))
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_tsa_client_free(UnsafeMutableRawPointer(h)); handle = nil }
    }
}

/// A document Security Store (`/DSS`) handle (opaque, freed in `deinit`/`close()`).
/// Wraps the `pdf_dss_*` C functions.
public final class Dss {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_dss_free(UnsafeMutableRawPointer(h)) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "Dss is closed") }
        return h
    }

    public func certCount() throws -> Int { Int(pdf_dss_cert_count(UnsafeRawPointer(try ptr()))) }
    public func crlCount() throws -> Int { Int(pdf_dss_crl_count(UnsafeRawPointer(try ptr()))) }
    public func ocspCount() throws -> Int { Int(pdf_dss_ocsp_count(UnsafeRawPointer(try ptr()))) }
    public func vriCount() throws -> Int { Int(pdf_dss_vri_count(UnsafeRawPointer(try ptr()))) }

    /// The DER bytes of the certificate at `index`.
    public func cert(_ index: Int) throws -> [UInt8] {
        let h = UnsafeRawPointer(try ptr())
        var outLen: UInt = 0, code: Int32 = 0
        let p = pdf_dss_get_cert(h, Int32(index), &outLen, &code)
        return try takeBytes(p, Int(outLen), code, "Dss.cert")
    }
    /// The DER bytes of the CRL at `index`.
    public func crl(_ index: Int) throws -> [UInt8] {
        let h = UnsafeRawPointer(try ptr())
        var outLen: UInt = 0, code: Int32 = 0
        let p = pdf_dss_get_crl(h, Int32(index), &outLen, &code)
        return try takeBytes(p, Int(outLen), code, "Dss.crl")
    }
    /// The DER bytes of the OCSP response at `index`.
    public func ocsp(_ index: Int) throws -> [UInt8] {
        let h = UnsafeRawPointer(try ptr())
        var outLen: UInt = 0, code: Int32 = 0
        let p = pdf_dss_get_ocsp(h, Int32(index), &outLen, &code)
        return try takeBytes(p, Int(outLen), code, "Dss.ocsp")
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_dss_free(UnsafeMutableRawPointer(h)); handle = nil }
    }
}

/// Result of a PDF/A validation (opaque `FfiPdfAResults`, freed in `deinit`/`close()`).
public final class PdfAResults {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_pdf_a_results_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "PdfAResults is closed") }
        return h
    }

    /// Whether the document is PDF/A compliant.
    public func isCompliant() throws -> Bool {
        var code: Int32 = 0
        return pdf_pdf_a_is_compliant(try ptr(), &code)
    }

    /// The list of compliance errors.
    public func errors() throws -> [String] {
        let h = try ptr()
        var code: Int32 = 0
        let n = Int(pdf_pdf_a_error_count(h))
        var out: [String] = []
        out.reserveCapacity(max(0, n))
        for i in 0..<max(0, n) {
            out.append(try takeString(pdf_pdf_a_get_error(h, Int32(i), &code), code, "PdfAResults.error"))
        }
        return out
    }

    /// Number of warnings.
    public func warningCount() throws -> Int { Int(pdf_pdf_a_warning_count(try ptr())) }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_pdf_a_results_free(h); handle = nil }
    }
}

/// Result of a PDF/UA validation (opaque `FfiUaResults`, freed in `deinit`/`close()`).
public final class UaResults {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_pdf_ua_results_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "UaResults is closed") }
        return h
    }

    /// Whether the document is PDF/UA accessible.
    public func isAccessible() throws -> Bool {
        var code: Int32 = 0
        return pdf_pdf_ua_is_accessible(try ptr(), &code)
    }

    /// The list of accessibility errors.
    public func errors() throws -> [String] {
        let h = try ptr()
        var code: Int32 = 0
        let n = Int(pdf_pdf_ua_error_count(h))
        var out: [String] = []
        out.reserveCapacity(max(0, n))
        for i in 0..<max(0, n) {
            out.append(try takeString(pdf_pdf_ua_get_error(h, Int32(i), &code), code, "UaResults.error"))
        }
        return out
    }

    /// The list of accessibility warnings.
    public func warnings() throws -> [String] {
        let h = try ptr()
        var code: Int32 = 0
        let n = Int(pdf_pdf_ua_warning_count(h))
        var out: [String] = []
        out.reserveCapacity(max(0, n))
        for i in 0..<max(0, n) {
            out.append(try takeString(pdf_pdf_ua_get_warning(h, Int32(i), &code), code, "UaResults.warning"))
        }
        return out
    }

    /// Structural statistics for the document.
    public func stats() throws -> UaStats {
        let h = try ptr()
        var s: Int32 = 0, im: Int32 = 0, t: Int32 = 0, f: Int32 = 0, a: Int32 = 0, p: Int32 = 0, code: Int32 = 0
        if !pdf_pdf_ua_get_stats(h, &s, &im, &t, &f, &a, &p, &code) {
            throw PdfOxideError(code: code, op: "UaResults.stats")
        }
        return UaStats(
            structElements: Int(s), images: Int(im), tables: Int(t),
            forms: Int(f), annotations: Int(a), pages: Int(p)
        )
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_pdf_ua_results_free(h); handle = nil }
    }
}

/// Result of a PDF/X validation (opaque `FfiPdfXResults`, freed in `deinit`/`close()`).
public final class PdfXResults {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_pdf_x_results_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "PdfXResults is closed") }
        return h
    }

    /// Whether the document is PDF/X compliant.
    public func isCompliant() throws -> Bool {
        var code: Int32 = 0
        return pdf_pdf_x_is_compliant(try ptr(), &code)
    }

    /// The list of compliance errors.
    public func errors() throws -> [String] {
        let h = try ptr()
        var code: Int32 = 0
        let n = Int(pdf_pdf_x_error_count(h))
        var out: [String] = []
        out.reserveCapacity(max(0, n))
        for i in 0..<max(0, n) {
            out.append(try takeString(pdf_pdf_x_get_error(h, Int32(i), &code), code, "PdfXResults.error"))
        }
        return out
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_pdf_x_results_free(h); handle = nil }
    }
}

// ── Phase-6 top-level: signing + log level ───────────────────────────────────

/// Sign raw PDF `pdf` bytes with `certificate`, returning the signed PDF bytes.
public func signBytes(_ pdf: [UInt8], certificate: Certificate, reason: String? = nil, location: String? = nil) throws -> [UInt8] {
    let cert = try certificate.rawPtr()
    var outLen: UInt = 0, code: Int32 = 0
    let p = pdf.withUnsafeBufferPointer { buf in
        pdf_sign_bytes(buf.baseAddress, UInt(buf.count), cert, reason, location, &outLen, &code)
    }
    return try takeBytes(p, Int(outLen), code, "signBytes")
}

/// Sign raw PDF `pdf` bytes at a PAdES baseline `level` (0=B-B 1=B-T 2=B-LT).
/// `tsaUrl` is required for level >= 1. The three revocation-material arrays
/// (DER certs / CRLs / OCSPs) carry the B-LT validation data.
public func signBytesPades(
    _ pdf: [UInt8], certificate: Certificate, level: Int32,
    tsaUrl: String? = nil, reason: String? = nil, location: String? = nil,
    certs: [[UInt8]] = [], crls: [[UInt8]] = [], ocsps: [[UInt8]] = []
) throws -> [UInt8] {
    let cert = try certificate.rawPtr()
    var outLen: UInt = 0, code: Int32 = 0
    let p = try withByteArrayArray(certs) { certsPtr, certLens in
        try withByteArrayArray(crls) { crlsPtr, crlLens in
            try withByteArrayArray(ocsps) { ocspsPtr, ocspLens in
                pdf.withUnsafeBufferPointer { buf -> UnsafeMutablePointer<UInt8>? in
                    pdf_sign_bytes_pades(
                        buf.baseAddress, UInt(buf.count), cert, level, tsaUrl, reason, location,
                        certsPtr, certLens, UInt(certs.count),
                        crlsPtr, crlLens, UInt(crls.count),
                        ocspsPtr, ocspLens, UInt(ocsps.count),
                        &outLen, &code
                    )
                }
            }
        }
    }
    return try takeBytes(p, Int(outLen), code, "signBytesPades")
}

/// Struct-options variant of `signBytesPades` — builds a `PadesSignOptionsC`
/// and delegates to the native struct-pointer entry point.
public func signBytesPadesOpts(
    _ pdf: [UInt8], certificate: Certificate, level: Int32,
    tsaUrl: String? = nil, reason: String? = nil, location: String? = nil,
    certs: [[UInt8]] = [], crls: [[UInt8]] = [], ocsps: [[UInt8]] = []
) throws -> [UInt8] {
    let cert = try certificate.rawPtr()
    var outLen: UInt = 0, code: Int32 = 0

    // C strings must outlive the call; strdup them and free afterwards.
    let tsaC = tsaUrl.map { strdup($0) } ?? nil
    let reasonC = reason.map { strdup($0) } ?? nil
    let locationC = location.map { strdup($0) } ?? nil
    defer {
        if let p = tsaC { free(p) }
        if let p = reasonC { free(p) }
        if let p = locationC { free(p) }
    }

    let p = try withByteArrayArray(certs) { certsPtr, certLens in
        try withByteArrayArray(crls) { crlsPtr, crlLens in
            try withByteArrayArray(ocsps) { ocspsPtr, ocspLens in
                pdf.withUnsafeBufferPointer { buf -> UnsafeMutablePointer<UInt8>? in
                    var opts = PadesSignOptionsC(
                        certificate_handle: cert,
                        certs: certsPtr, cert_lens: certLens, n_certs: UInt(certs.count),
                        crls: crlsPtr, crl_lens: crlLens, n_crls: UInt(crls.count),
                        ocsps: ocspsPtr, ocsp_lens: ocspLens, n_ocsps: UInt(ocsps.count),
                        tsa_url: tsaC.map { UnsafePointer($0) }, reason: reasonC.map { UnsafePointer($0) },
                        location: locationC.map { UnsafePointer($0) }, level: level
                    )
                    return pdf_sign_bytes_pades_opts(buf.baseAddress, UInt(buf.count), &opts, &outLen, &code)
                }
            }
        }
    }
    return try takeBytes(p, Int(outLen), code, "signBytesPadesOpts")
}

/// Set the global library log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace).
public func setLogLevel(_ level: Int32) {
    pdf_oxide_set_log_level(level)
}

/// Get the current global library log level (0-5).
public func getLogLevel() -> Int32 {
    pdf_oxide_get_log_level()
}

// ── Phase-7 top-level: merge + timestamp ─────────────────────────────────────

/// Merge the PDFs at `paths` (in order) into a single in-memory PDF.
public func merge(_ paths: [String]) throws -> [UInt8] {
    var dataLen: Int32 = 0, code: Int32 = 0
    let p = withCStringArray(paths) { pathsPtr in
        pdf_merge(pathsPtr, Int32(paths.count), &dataLen, &code)
    }
    guard let p else { throw PdfOxideError(code: code, op: "merge") }
    defer { free_bytes(p) }
    let n = dataLen < 0 ? 0 : Int(dataLen)
    return Array(UnsafeBufferPointer(start: p, count: n))
}

/// Add an RFC 3161 timestamp to the signature at `sigIndex` in `pdfData`,
/// fetched from `tsaUrl`. Returns the re-saved PDF bytes.
public func addTimestamp(_ pdfData: [UInt8], sigIndex: Int32, tsaUrl: String) throws -> [UInt8] {
    var outData: UnsafeMutablePointer<UInt8>? = nil
    var outLen: UInt = 0
    var code: Int32 = 0
    let ok = pdfData.withUnsafeBufferPointer { buf in
        pdf_add_timestamp(buf.baseAddress, UInt(buf.count), sigIndex, tsaUrl, &outData, &outLen, &code)
    }
    if !ok { throw PdfOxideError(code: code, op: "addTimestamp") }
    return try takeBytes(outData, Int(outLen), code, "addTimestamp")
}

// ── Phase-7: barcodes / OCR / render variants / redaction / constructors /
//            page getters / element lists / timestamp ────────────────────────

/// A single extracted page element (text run / image / path) read from an
/// `ElementList`. `text` is empty for non-text elements.
public struct Element {
    public let type: String
    public let text: String
    public let rect: Bbox
}

/// An opaque list of page elements (`FfiElementList`), freed in `deinit`/`close()`.
/// Wraps `pdf_page_get_elements` and the `pdf_oxide_element_*` accessor family.
public final class ElementList {
    private var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_oxide_elements_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "ElementList is closed") }
        return h
    }

    /// Number of elements in the list.
    public func count() throws -> Int { Int(pdf_oxide_element_count(try ptr())) }

    /// The element at `index`.
    public func element(_ index: Int) throws -> Element {
        let h = try ptr()
        var code: Int32 = 0
        let idx = Int32(index)
        let type = try takeString(pdf_oxide_element_get_type(h, idx, &code), code, "ElementList.type")
        let text = try takeString(pdf_oxide_element_get_text(h, idx, &code), code, "ElementList.text")
        var x: Float = 0, y: Float = 0, w: Float = 0, hgt: Float = 0
        pdf_oxide_element_get_rect(h, idx, &x, &y, &w, &hgt, &code)
        return Element(
            type: type, text: text,
            rect: Bbox(x: Double(x), y: Double(y), width: Double(w), height: Double(hgt))
        )
    }

    /// Materialise every element into an array.
    public func all() throws -> [Element] {
        let n = try count()
        var out: [Element] = []
        out.reserveCapacity(max(0, n))
        for i in 0..<max(0, n) { out.append(try element(i)) }
        return out
    }

    /// Serialise the whole list to JSON.
    public func toJson() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_oxide_elements_to_json(try ptr(), &code), code, "ElementList.toJson")
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_oxide_elements_free(h); handle = nil }
    }
}

/// A generated/decoded barcode or QR-code image (opaque `FfiBarcodeImage`,
/// freed in `deinit`/`close()`). Wraps the `pdf_barcode_*` / `pdf_generate_*`
/// C functions. Add it to an editor page via `DocumentEditor.addBarcodeToPage`.
public final class BarcodeImage {
    fileprivate var handle: OpaquePointer?

    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { if let h = handle { pdf_barcode_free(h) } }

    private func ptr() throws -> OpaquePointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "BarcodeImage is closed") }
        return h
    }

    /// Generate a QR code. `errorCorrection`: 0=L 1=M 2=Q 3=H. `sizePx` is the
    /// requested module-grid pixel size.
    public static func generateQrCode(_ data: String, errorCorrection: Int32 = 1, sizePx: Int32 = 256) throws -> BarcodeImage {
        var code: Int32 = 0
        guard let h = pdf_generate_qr_code(data, errorCorrection, sizePx, &code) else {
            throw PdfOxideError(code: code, op: "BarcodeImage.generateQrCode")
        }
        return BarcodeImage(h)
    }

    /// Generate a 1-D / 2-D barcode of the given `format` code.
    public static func generateBarcode(_ data: String, format: Int32, sizePx: Int32 = 256) throws -> BarcodeImage {
        var code: Int32 = 0
        guard let h = pdf_generate_barcode(data, format, sizePx, &code) else {
            throw PdfOxideError(code: code, op: "BarcodeImage.generateBarcode")
        }
        return BarcodeImage(h)
    }

    /// The barcode's decoded payload string.
    public func data() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_barcode_get_data(try ptr(), &code), code, "BarcodeImage.data")
    }

    /// The barcode format code.
    public func format() throws -> Int32 {
        var code: Int32 = 0
        return pdf_barcode_get_format(try ptr(), &code)
    }

    /// The decode confidence (1.0 for generated barcodes).
    public func confidence() throws -> Float {
        var code: Int32 = 0
        return pdf_barcode_get_confidence(try ptr(), &code)
    }

    /// Render the barcode to PNG bytes at `sizePx` pixels.
    public func imagePng(sizePx: Int32 = 256) throws -> [UInt8] {
        let h = try ptr()
        var outLen: Int32 = 0, code: Int32 = 0
        guard let p = pdf_barcode_get_image_png(h, sizePx, &outLen, &code) else {
            throw PdfOxideError(code: code, op: "BarcodeImage.imagePng")
        }
        defer { free_bytes(p) }
        let n = outLen < 0 ? 0 : Int(outLen)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// Render the barcode to an SVG string at `sizePx` pixels.
    public func svg(sizePx: Int32 = 256) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_barcode_get_svg(try ptr(), sizePx, &code), code, "BarcodeImage.svg")
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_barcode_free(h); handle = nil }
    }
}

/// An OCR engine (opaque `void*` `Box<OcrEngine>`, freed in `deinit`/`close()`).
/// Wraps `pdf_ocr_engine_create` / `pdf_ocr_engine_free`. Pass to
/// `Document.ocrExtractText` to run recognition over a page's images.
public final class OcrEngine {
    fileprivate var handle: UnsafeMutableRawPointer?

    fileprivate init(_ handle: UnsafeMutableRawPointer) { self.handle = handle }
    deinit { if let h = handle { pdf_ocr_engine_free(h) } }

    fileprivate func rawPtr() throws -> UnsafeMutableRawPointer {
        guard let h = handle else { throw PdfOxideError(code: 0, op: "OcrEngine is closed") }
        return h
    }

    /// Create an OCR engine from detection / recognition model + dictionary paths.
    public static func create(detModelPath: String, recModelPath: String, dictPath: String) throws -> OcrEngine {
        var code: Int32 = 0
        guard let h = pdf_ocr_engine_create(detModelPath, recModelPath, dictPath, &code) else {
            throw PdfOxideError(code: code, op: "OcrEngine.create")
        }
        return OcrEngine(h)
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_ocr_engine_free(h); handle = nil }
    }
}

/// A standalone renderer config handle (opaque `void*`, freed in `deinit`/`close()`).
/// Wraps `pdf_create_renderer` / `pdf_renderer_free`.
public final class Renderer {
    private var handle: UnsafeMutableRawPointer?

    private init(_ handle: UnsafeMutableRawPointer) { self.handle = handle }
    deinit { if let h = handle { pdf_renderer_free(h) } }

    /// Create a renderer. `format`: 0=PNG 1=JPEG.
    public static func create(dpi: Int32 = 150, format: Int32 = 0, quality: Int32 = 90, antiAlias: Bool = true) throws -> Renderer {
        var code: Int32 = 0
        guard let h = pdf_create_renderer(dpi, format, quality, antiAlias, &code) else {
            throw PdfOxideError(code: code, op: "Renderer.create")
        }
        return Renderer(h)
    }

    /// Free the native handle now (idempotent).
    public func close() {
        if let h = handle { pdf_renderer_free(h); handle = nil }
    }
}

// Marshal `[[UInt8]]` into a C `const uint8_t* const*` + parallel `const uintptr_t*`
// lengths array for the duration of `body`. Pointers are valid only inside `body`.
private func withByteArrayArray<R>(
    _ arrays: [[UInt8]],
    _ body: (UnsafePointer<UnsafePointer<UInt8>?>?, UnsafePointer<UInt>?) throws -> R
) rethrows -> R {
    // Recursively pin each inner buffer so its baseAddress stays valid for `body`.
    var ptrs: [UnsafePointer<UInt8>?] = []
    var lens: [UInt] = []
    ptrs.reserveCapacity(arrays.count)
    lens.reserveCapacity(arrays.count)

    func pin(_ i: Int) throws -> R {
        if i == arrays.count {
            return try ptrs.withUnsafeBufferPointer { pBuf in
                try lens.withUnsafeBufferPointer { lBuf in
                    try body(pBuf.baseAddress, lBuf.baseAddress)
                }
            }
        }
        return try arrays[i].withUnsafeBufferPointer { buf in
            ptrs.append(buf.baseAddress)
            lens.append(UInt(buf.count))
            return try pin(i + 1)
        }
    }
    return try pin(0)
}

// Marshal a [String] into a C `const char* const*` for the duration of `body`.
// The C strings (and their backing buffer) are valid only inside `body`.
private func withCStringArray<R>(_ strings: [String], _ body: (UnsafePointer<UnsafePointer<CChar>?>?) -> R) -> R {
    var cstrs: [UnsafeMutablePointer<CChar>?] = strings.map { strdup($0) }
    defer { for p in cstrs where p != nil { free(p) } }
    return cstrs.withUnsafeMutableBufferPointer { buf in
        // Reinterpret [char*] as `const char* const*`.
        buf.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: buf.count) { rebased in
            body(rebased)
        }
    }
}
