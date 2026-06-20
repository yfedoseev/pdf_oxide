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

/// The C-ABI error code from the most recent failure on this thread. Zig error
/// values cannot carry a payload, so the code is surfaced here (read it right
/// after catching `Error.PdfOxide`). Mirrors the `{code, op}` payload the other
/// bindings carry.
pub threadlocal var last_error_code: i32 = 0;

/// Code of the most recent failure on this thread.
pub fn lastErrorCode() i32 {
    return last_error_code;
}

/// Record `code` and return the binding's error (keeps call sites terse).
fn fail(code: i32) Error {
    last_error_code = code;
    return Error.PdfOxide;
}

/// PDF version (e.g. 1.7).
pub const Version = struct { major: u8, minor: u8 };

/// An axis-aligned bounding box in page coordinates.
pub const Bbox = struct { x: f32, y: f32, width: f32, height: f32 };

/// A single extracted glyph. `fontName` is allocator-owned (free it).
pub const Char = struct {
    character: u32,
    bbox: Bbox,
    fontName: []u8,
    fontSize: f32,
};

/// A single extracted word. `text`/`fontName` are allocator-owned (free them).
pub const Word = struct {
    text: []u8,
    bbox: Bbox,
    fontName: []u8,
    fontSize: f32,
    bold: bool,
};

/// A single extracted text line. `text` is allocator-owned (free it).
pub const TextLine = struct {
    text: []u8,
    bbox: Bbox,
    wordCount: i32,
};

/// A single extracted table. `cells` is a row-major grid of allocator-owned
/// strings (free each cell, then the slice).
pub const Table = struct {
    rowCount: i32,
    colCount: i32,
    hasHeader: bool,
    cells: [][]u8,

    /// Cell text at (row, col), 0-based. Out of range yields an empty string.
    pub fn cell(self: Table, row: i32, col: i32) []const u8 {
        if (row < 0 or col < 0 or row >= self.rowCount or col >= self.colCount) return "";
        const r: usize = @intCast(row);
        const cl: usize = @intCast(col);
        const cols: usize = @intCast(self.colCount);
        return self.cells[r * cols + cl];
    }

    /// Free every cell string and the backing slice.
    pub fn deinit(self: *Table, alloc: std.mem.Allocator) void {
        for (self.cells) |cl| alloc.free(cl);
        alloc.free(self.cells);
    }
};

/// Copy a C string return into an allocator-owned slice and free the C buffer.
fn takeString(alloc: std.mem.Allocator, ptr: ?[*:0]u8, code: i32) Error![]u8 {
    const p = ptr orelse return fail(code);
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
        const h = c.pdf_document_open(path.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Open a PDF from in-memory bytes.
    pub fn openFromBytes(data: []const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_from_bytes(data.ptr, data.len, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    /// Open a password-protected PDF.
    pub fn openWithPassword(path: [:0]const u8, password: [:0]const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_with_password(path.ptr, password.ptr, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    pub fn deinit(self: *Document) void {
        c.pdf_document_free(self.handle);
    }

    pub fn pageCount(self: Document) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_get_page_count(self.handle, &code);
        if (n < 0) return fail(code);
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

    pub fn extractText(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_text(self.handle, page_index, &code), code);
    }
    pub fn toPlainText(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_plain_text(self.handle, page_index, &code), code);
    }
    pub fn toMarkdown(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_markdown(self.handle, page_index, &code), code);
    }
    pub fn toHtml(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_html(self.handle, page_index, &code), code);
    }
    pub fn toMarkdownAll(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_markdown_all(self.handle, &code), code);
    }
    pub fn toHtmlAll(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_html_all(self.handle, &code), code);
    }
    pub fn toPlainTextAll(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_to_plain_text_all(self.handle, &code), code);
    }
    /// Authenticate with a password. Returns true on success, false for a wrong
    /// password (a wrong password is not a C-ABI error). Mirrors the bool C-ABI
    /// convention: a non-zero error_code maps to `Error.PdfOxide`.
    pub fn authenticate(self: Document, password: [:0]const u8) Error!bool {
        var code: i32 = 0;
        const ok = c.pdf_document_authenticate(self.handle, password.ptr, &code);
        if (code != 0) return fail(code);
        return ok;
    }
    pub fn extractStructuredJson(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_structured_to_json(self.handle, page_index, &code), code);
    }

    /// Glyph-level extraction for a (0-based) page. Caller owns the returned slice
    /// and each element's `fontName`; free with `freeChars`.
    pub fn extractChars(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Char {
        var code: i32 = 0;
        const list = c.pdf_document_extract_chars(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_char_list_free(list);
        const n = c.pdf_oxide_char_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Char, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |ch| alloc.free(ch.fontName);
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const character = c.pdf_oxide_char_get_char(list, idx, &code);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_char_get_bbox(list, idx, &x, &y, &w, &h, &code);
            const font_name = try takeString(alloc, c.pdf_oxide_char_get_font_name(list, idx, &code), code);
            const font_size = c.pdf_oxide_char_get_font_size(list, idx, &code);
            out[i] = .{
                .character = character,
                .bbox = .{ .x = x, .y = y, .width = w, .height = h },
                .fontName = font_name,
                .fontSize = font_size,
            };
        }
        return out;
    }

    /// Free a slice returned by `extractChars`.
    pub fn freeChars(alloc: std.mem.Allocator, chars: []Char) void {
        for (chars) |ch| alloc.free(ch.fontName);
        alloc.free(chars);
    }

    /// Word-level extraction for a (0-based) page. Caller owns the returned slice
    /// and each element's `text`/`fontName`; free with `freeWords`.
    pub fn extractWords(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Word {
        var code: i32 = 0;
        const list = c.pdf_document_extract_words(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_word_list_free(list);
        const n = c.pdf_oxide_word_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Word, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |wd| {
            alloc.free(wd.text);
            alloc.free(wd.fontName);
        };
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const word_text = try takeString(alloc, c.pdf_oxide_word_get_text(list, idx, &code), code);
            errdefer alloc.free(word_text);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_word_get_bbox(list, idx, &x, &y, &w, &h, &code);
            const font_name = try takeString(alloc, c.pdf_oxide_word_get_font_name(list, idx, &code), code);
            const font_size = c.pdf_oxide_word_get_font_size(list, idx, &code);
            const bold = c.pdf_oxide_word_is_bold(list, idx, &code);
            out[i] = .{
                .text = word_text,
                .bbox = .{ .x = x, .y = y, .width = w, .height = h },
                .fontName = font_name,
                .fontSize = font_size,
                .bold = bold,
            };
        }
        return out;
    }

    /// Free a slice returned by `extractWords`.
    pub fn freeWords(alloc: std.mem.Allocator, words: []Word) void {
        for (words) |wd| {
            alloc.free(wd.text);
            alloc.free(wd.fontName);
        }
        alloc.free(words);
    }

    /// Line-level extraction for a (0-based) page. Caller owns the returned slice
    /// and each element's `text`; free with `freeTextLines`.
    pub fn extractTextLines(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]TextLine {
        var code: i32 = 0;
        const list = c.pdf_document_extract_text_lines(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_line_list_free(list);
        const n = c.pdf_oxide_line_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(TextLine, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |ln| alloc.free(ln.text);
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const line_text = try takeString(alloc, c.pdf_oxide_line_get_text(list, idx, &code), code);
            errdefer alloc.free(line_text);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_line_get_bbox(list, idx, &x, &y, &w, &h, &code);
            const word_count = c.pdf_oxide_line_get_word_count(list, idx, &code);
            out[i] = .{
                .text = line_text,
                .bbox = .{ .x = x, .y = y, .width = w, .height = h },
                .wordCount = word_count,
            };
        }
        return out;
    }

    /// Free a slice returned by `extractTextLines`.
    pub fn freeTextLines(alloc: std.mem.Allocator, lines: []TextLine) void {
        for (lines) |ln| alloc.free(ln.text);
        alloc.free(lines);
    }

    /// Table extraction for a (0-based) page. Caller owns the returned slice and
    /// each table's cells; free with `freeTables`.
    pub fn extractTables(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Table {
        var code: i32 = 0;
        const list = c.pdf_document_extract_tables(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_table_list_free(list);
        const n = c.pdf_oxide_table_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Table, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |*tbl| tbl.deinit(alloc);
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const rows = c.pdf_oxide_table_get_row_count(list, idx, &code);
            if (rows < 0) return fail(code);
            const cols = c.pdf_oxide_table_get_col_count(list, idx, &code);
            if (cols < 0) return fail(code);
            const has_header = c.pdf_oxide_table_has_header(list, idx, &code);
            const cell_total: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(cols));
            const cells = try alloc.alloc([]u8, cell_total);
            errdefer alloc.free(cells);
            var j: usize = 0;
            errdefer for (cells[0..j]) |cl| alloc.free(cl);
            var r: i32 = 0;
            while (r < rows) : (r += 1) {
                var cc: i32 = 0;
                while (cc < cols) : (cc += 1) {
                    cells[j] = try takeString(alloc, c.pdf_oxide_table_get_cell_text(list, idx, r, cc, &code), code);
                    j += 1;
                }
            }
            out[i] = .{
                .rowCount = rows,
                .colCount = cols,
                .hasHeader = has_header,
                .cells = cells,
            };
        }
        return out;
    }

    /// Free a slice returned by `extractTables`.
    pub fn freeTables(alloc: std.mem.Allocator, tables: []Table) void {
        for (tables) |*tbl| tbl.deinit(alloc);
        alloc.free(tables);
    }

    /// A lightweight view of a single (0-based) page. The returned `Page` borrows
    /// this `Document`'s handle, so the `Document` MUST outlive the `Page`.
    pub fn page(self: Document, index: i32) Page {
        return .{ .doc = self, .index = index };
    }
};

/// A single page of a `Document`. Holds a copy of the owning `Document` (which is
/// just a borrowed handle pointer); the `Document` must not be freed while the
/// `Page` is in use. Each method delegates to the corresponding per-page
/// `Document` method with the stored index.
pub const Page = struct {
    doc: Document,
    index: i32,

    pub fn text(self: Page, alloc: std.mem.Allocator) Error![]u8 {
        return self.doc.extractText(alloc, self.index);
    }
    pub fn plainText(self: Page, alloc: std.mem.Allocator) Error![]u8 {
        return self.doc.toPlainText(alloc, self.index);
    }
    pub fn markdown(self: Page, alloc: std.mem.Allocator) Error![]u8 {
        return self.doc.toMarkdown(alloc, self.index);
    }
    pub fn html(self: Page, alloc: std.mem.Allocator) Error![]u8 {
        return self.doc.toHtml(alloc, self.index);
    }
};

/// A PDF produced by a builder.
pub const Pdf = struct {
    handle: *c.Pdf,

    pub fn fromMarkdown(md: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_markdown(md.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }
    pub fn fromHtml(html: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_html(html.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }
    pub fn fromText(text: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_text(text.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    pub fn deinit(self: *Pdf) void {
        c.pdf_free(self.handle);
    }

    pub fn save(self: Pdf, path: [:0]const u8) Error!void {
        var code: i32 = 0;
        if (c.pdf_save(self.handle, path.ptr, &code) != 0) return fail(code);
    }

    /// Serialize to bytes; caller owns the returned slice.
    pub fn toBytes(self: Pdf, alloc: std.mem.Allocator) Error![]u8 {
        var len: i32 = 0;
        var code: i32 = 0;
        const p = c.pdf_save_to_bytes(self.handle, &len, &code) orelse return fail(code);
        defer c.free_bytes(p);
        const n: usize = if (len < 0) 0 else @intCast(len);
        return alloc.dupe(u8, p[0..n]);
    }
};

// ── api-coverage tests (one per public method) ────────────────────────────────
const testing = std.testing;

fn samplePdf(alloc: std.mem.Allocator) ![]u8 {
    var pdf = try Pdf.fromMarkdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n");
    defer pdf.deinit();
    return pdf.toBytes(alloc);
}

test "Pdf builder: fromMarkdown/fromHtml/fromText/toBytes/save" {
    const a = testing.allocator;
    {
        var p = try Pdf.fromMarkdown("# md\n\nbody\n");
        defer p.deinit();
        const b = try p.toBytes(a);
        defer a.free(b);
        try testing.expect(b.len > 100);
    }
    {
        var p = try Pdf.fromHtml("<h1>h</h1><p>b</p>");
        defer p.deinit();
        const b = try p.toBytes(a);
        defer a.free(b);
        try testing.expect(b.len > 100);
    }
    {
        var p = try Pdf.fromText("plain text body");
        defer p.deinit();
        const b = try p.toBytes(a);
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

    const htmlall = try doc.toHtmlAll(a);
    defer a.free(htmlall);
    try testing.expect(htmlall.len > 0); // toHtmlAll
    try testing.expect(std.mem.indexOf(u8, htmlall, "<") != null or
        std.mem.indexOf(u8, htmlall, "Alpha") != null);

    const ptall = try doc.toPlainTextAll(a);
    defer a.free(ptall);
    try testing.expect(ptall.len > 0); // toPlainTextAll

    // authenticate: returns a bool without error on an unencrypted sample
    _ = try doc.authenticate(""); // authenticate

    // page(index) model
    {
        const pg = doc.page(0); // page
        const t = try pg.text(a);
        defer a.free(t);
        try testing.expect(std.mem.indexOf(u8, t, "Alpha") != null); // Page.text

        inline for (.{ "plainText", "markdown", "html" }) |name| {
            const s = try @field(Page, name)(pg, a);
            defer a.free(s);
            try testing.expect(s.len > 0); // Page.plainText/markdown/html
        }
    }

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

test "Document: element extraction (chars/words/lines/tables)" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // extractWords: non-empty, word[0].text non-empty, has a bbox
    const words = try doc.extractWords(a, 0);
    defer Document.freeWords(a, words);
    try testing.expect(words.len > 0);
    try testing.expect(words[0].text.len > 0);
    try testing.expect(words[0].bbox.width >= 0);

    // extractChars: non-empty
    const chars = try doc.extractChars(a, 0);
    defer Document.freeChars(a, chars);
    try testing.expect(chars.len > 0);

    // extractTextLines: non-empty
    const lines = try doc.extractTextLines(a, 0);
    defer Document.freeTextLines(a, lines);
    try testing.expect(lines.len > 0);

    // extractTables: returns a list (may be empty) without error
    const tables = try doc.extractTables(a, 0);
    defer Document.freeTables(a, tables);
    if (tables.len > 0) {
        const t = tables[0];
        _ = t.cell(0, 0); // cell accessor (smoke)
    }
}

test "error path: open nonexistent returns error" {
    try testing.expectError(Error.PdfOxide, Document.open("/nonexistent/nope.pdf"));
}
