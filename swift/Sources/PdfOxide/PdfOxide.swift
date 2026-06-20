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
    private let handle: OpaquePointer

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { pdf_document_free(handle) }

    /// Open a PDF from a filesystem path.
    public static func open(_ path: String) throws -> Document {
        var code: Int32 = 0
        guard let h = pdf_document_open(path, &code) else { throw PdfOxideError(code: code, op: "open") }
        return Document(h)
    }

    /// Open a PDF from in-memory bytes.
    public static func open(bytes: [UInt8]) throws -> Document {
        var code: Int32 = 0
        let h = bytes.withUnsafeBufferPointer { buf in
            pdf_document_open_from_bytes(buf.baseAddress, buf.count, &code)
        }
        guard let h else { throw PdfOxideError(code: code, op: "openFromBytes") }
        return Document(h)
    }

    /// Open a password-protected PDF.
    public static func open(_ path: String, password: String) throws -> Document {
        var code: Int32 = 0
        guard let h = pdf_document_open_with_password(path, password, &code) else {
            throw PdfOxideError(code: code, op: "openWithPassword")
        }
        return Document(h)
    }

    public func pageCount() throws -> Int {
        var code: Int32 = 0
        let n = pdf_document_get_page_count(handle, &code)
        if n < 0 { throw PdfOxideError(code: code, op: "pageCount") }
        return Int(n)
    }

    public func version() -> PdfVersion {
        var major: UInt8 = 0, minor: UInt8 = 0
        pdf_document_get_version(handle, &major, &minor)
        return PdfVersion(major: Int(major), minor: Int(minor))
    }

    public var isEncrypted: Bool { pdf_document_is_encrypted(handle) }
    public var hasStructureTree: Bool { pdf_document_has_structure_tree(handle) }

    public func extractText(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_extract_text(handle, Int32(page), &code), code, "extractText")
    }
    public func toPlainText(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_plain_text(handle, Int32(page), &code), code, "toPlainText")
    }
    public func toMarkdown(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_markdown(handle, Int32(page), &code), code, "toMarkdown")
    }
    public func toHtml(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_html(handle, Int32(page), &code), code, "toHtml")
    }
    public func toMarkdownAll() throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_to_markdown_all(handle, &code), code, "toMarkdownAll")
    }
    public func extractStructuredJson(_ page: Int) throws -> String {
        var code: Int32 = 0
        return try takeString(pdf_document_extract_structured_to_json(handle, Int32(page), &code), code, "extractStructuredJson")
    }
}

/// A PDF produced by a builder.
public final class Pdf {
    private let handle: OpaquePointer

    private init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { pdf_free(handle) }

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
        if pdf_save(handle, path, &code) != 0 { throw PdfOxideError(code: code, op: "save") }
    }

    public func saveToBytes() throws -> [UInt8] {
        var len: Int32 = 0, code: Int32 = 0
        guard let p = pdf_save_to_bytes(handle, &len, &code) else { throw PdfOxideError(code: code, op: "saveToBytes") }
        defer { free_string(UnsafeMutableRawPointer(p).assumingMemoryBound(to: CChar.self)) }
        let n = len < 0 ? 0 : Int(len)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}
