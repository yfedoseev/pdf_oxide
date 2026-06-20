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
