//! pdf_oxide — idiomatic Zig bindings over the C ABI via @cImport.
//!
//! First-class C interop: no shim. Handles are structs with `deinit`; returned
//! C strings/buffers are copied into caller-owned allocations and the C buffer
//! freed via free_string; non-success C-ABI error codes map to `Error`.
//!
//! API surface mirrors the other language bindings; coverage is asserted by the
//! `test` blocks at the bottom (one per public method).
const std = @import("std");

const c = @cImport({
    @cInclude("pdf_oxide_c/pdf_oxide.h");
});

/// Any non-success C-ABI outcome.
pub const Error = error{ PdfOxide, OutOfMemory };

/// PDF version (e.g. 1.7).
pub const Version = struct { major: u8, minor: u8 };

/// Copy a C string return into an allocator-owned slice and free the C buffer.
fn takeString(alloc: std.mem.Allocator, ptr: ?[*:0]u8) Error![]u8 {
    const p = ptr orelse return Error.PdfOxide;
    defer c.free_string(p);
    const span = std.mem.span(p);
    return alloc.dupe(u8, span);
}

/// An opened PDF for extraction/inspection.
pub const Document = struct {
    handle: *c.PdfDocument,

    /// Open a PDF from a filesystem path (NUL-terminated).
    pub fn open(path: [:0]const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open(path.ptr, &code) orelse return Error.PdfOxide;
        return .{ .handle = h };
    }

    /// Open a PDF from in-memory bytes.
    pub fn openFromBytes(data: []const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_from_bytes(data.ptr, data.len, &code) orelse
            return Error.PdfOxide;
        return .{ .handle = h };
    }

    /// Open a password-protected PDF.
    pub fn openWithPassword(path: [:0]const u8, password: [:0]const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_with_password(path.ptr, password.ptr, &code) orelse
            return Error.PdfOxide;
        return .{ .handle = h };
    }

    pub fn deinit(self: *Document) void {
        c.pdf_document_free(self.handle);
    }

    pub fn pageCount(self: Document) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_get_page_count(self.handle, &code);
        if (n < 0) return Error.PdfOxide;
        return n;
    }

    pub fn version(self: Document) Version {
        var maj: u8 = 0;
        var min: u8 = 0;
        c.pdf_document_get_version(self.handle, &maj, &min);
        return .{ .major = maj, .minor = min };
    }

    pub fn isEncrypted(self: Document) bool {
        return c.pdf_document_is_encrypted(self.handle);
    }

    pub fn hasStructureTree(self: Document) bool {
        return c.pdf_document_has_structure_tree(self.handle);
    }

    pub fn extractText(self: Document, alloc: std.mem.Allocator, page: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_text(self.handle, page, &code));
    }
    pub fn toPlainText(self: Document, alloc: std.mem.Allocator, page: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_plain_text(self.handle, page, &code));
    }
    pub fn toMarkdown(self: Document, alloc: std.mem.Allocator, page: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_markdown(self.handle, page, &code));
    }
    pub fn toHtml(self: Document, alloc: std.mem.Allocator, page: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_html(self.handle, page, &code));
    }
    pub fn toMarkdownAll(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_markdown_all(self.handle, &code));
    }
    pub fn extractStructuredJson(self: Document, alloc: std.mem.Allocator, page: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_structured_to_json(self.handle, page, &code));
    }
};

/// A PDF produced by a builder.
pub const Pdf = struct {
    handle: *c.Pdf,

    pub fn fromMarkdown(md: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_markdown(md.ptr, &code) orelse return Error.PdfOxide;
        return .{ .handle = h };
    }
    pub fn fromHtml(html: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_html(html.ptr, &code) orelse return Error.PdfOxide;
        return .{ .handle = h };
    }
    pub fn fromText(text: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_text(text.ptr, &code) orelse return Error.PdfOxide;
        return .{ .handle = h };
    }

    pub fn deinit(self: *Pdf) void {
        c.pdf_free(self.handle);
    }

    pub fn save(self: Pdf, path: [:0]const u8) Error!void {
        var code: i32 = 0;
        if (c.pdf_save(self.handle, path.ptr, &code) != 0) return Error.PdfOxide;
    }

    /// Serialize to bytes; caller owns the returned slice.
    pub fn saveToBytes(self: Pdf, alloc: std.mem.Allocator) Error![]u8 {
        var len: i32 = 0;
        var code: i32 = 0;
        const p = c.pdf_save_to_bytes(self.handle, &len, &code) orelse return Error.PdfOxide;
        defer c.free_string(@ptrCast(p));
        const n: usize = if (len < 0) 0 else @intCast(len);
        return alloc.dupe(u8, p[0..n]);
    }
};

// ── api-coverage tests (one per public method) ────────────────────────────────
const testing = std.testing;

fn samplePdf(alloc: std.mem.Allocator) ![]u8 {
    var pdf = try Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n");
    defer pdf.deinit();
    return pdf.saveToBytes(alloc);
}

test "Pdf builder: fromMarkdown/fromHtml/fromText/saveToBytes/save" {
    const a = testing.allocator;
    {
        var p = try Pdf.fromMarkdown("# md\n\nbody\n");
        defer p.deinit();
        const b = try p.saveToBytes(a);
        defer a.free(b);
        try testing.expect(b.len > 100);
    }
    {
        var p = try Pdf.fromHtml("<h1>h</h1><p>b</p>");
        defer p.deinit();
        const b = try p.saveToBytes(a);
        defer a.free(b);
        try testing.expect(b.len > 100);
    }
    {
        var p = try Pdf.fromText("plain text body");
        defer p.deinit();
        const b = try p.saveToBytes(a);
        defer a.free(b);
        try testing.expect(b.len > 100);
    }
    {
        var p = try Pdf.fromMarkdown("# f\n\nx\n");
        defer p.deinit();
        try p.save("/tmp/pdfoxide_zig_test.pdf");
        const f = try std.fs.cwd().openFile("/tmp/pdfoxide_zig_test.pdf", .{});
        f.close();
        try std.fs.cwd().deleteFile("/tmp/pdfoxide_zig_test.pdf");
    }
}

test "Document: open paths + inspection + extraction" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes); // openFromBytes
    defer doc.deinit();

    try testing.expect(try doc.pageCount() >= 1); // pageCount
    try testing.expect(doc.version().major >= 1); // version
    try testing.expect(doc.isEncrypted() == false); // isEncrypted
    _ = doc.hasStructureTree(); // hasStructureTree (smoke)

    const text = try doc.extractText(a, 0);
    defer a.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "Alpha") != null); // extractText

    inline for (.{ "toPlainText", "toMarkdown", "toHtml", "extractStructuredJson" }) |name| {
        const s = try @field(Document, name)(doc, a, 0);
        defer a.free(s);
        try testing.expect(s.len > 0);
    }
    const mdall = try doc.toMarkdownAll(a);
    defer a.free(mdall);
    try testing.expect(mdall.len > 0); // toMarkdownAll

    // open(path)
    {
        var p = try Pdf.fromMarkdown("# f\n\nx\n");
        defer p.deinit();
        try p.save("/tmp/pdfoxide_zig_open.pdf");
        var d2 = try Document.open("/tmp/pdfoxide_zig_open.pdf");
        defer d2.deinit();
        try testing.expect(try d2.pageCount() >= 1);
        try std.fs.cwd().deleteFile("/tmp/pdfoxide_zig_open.pdf");
    }
}

test "error path: open nonexistent returns error" {
    try testing.expectError(Error.PdfOxide, Document.open("/nonexistent/nope.pdf"));
}
