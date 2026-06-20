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

/// An embedded font on a page. `name`/`type`/`encoding` are allocator-owned.
pub const Font = struct {
    name: []u8,
    type: []u8,
    encoding: []u8,
    embedded: bool,
    subset: bool,
};

/// An embedded image on a page. `format`/`colorspace`/`data` are allocator-owned.
pub const Image = struct {
    width: i32,
    height: i32,
    bitsPerComponent: i32,
    format: []u8,
    colorspace: []u8,
    data: []u8,
};

/// A page annotation. `type`/`subtype`/`content`/`author` are allocator-owned.
pub const Annotation = struct {
    type: []u8,
    subtype: []u8,
    content: []u8,
    author: []u8,
    rect: Bbox,
    borderWidth: f32,
};

/// A vector path on a page.
pub const Path = struct {
    bbox: Bbox,
    strokeWidth: f32,
    hasStroke: bool,
    hasFill: bool,
    operationCount: i32,
};

/// A single search hit. `text` is allocator-owned (free it).
pub const SearchResult = struct {
    text: []u8,
    page: i32,
    bbox: Bbox,
};

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

    /// Embedded fonts on a (0-based) page. Caller owns the returned slice and each
    /// element's `name`/`type`/`encoding`; free with `freeFonts`.
    pub fn embeddedFonts(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Font {
        var code: i32 = 0;
        const list = c.pdf_document_get_embedded_fonts(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_font_list_free(list);
        const n = c.pdf_oxide_font_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Font, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |ft| {
            alloc.free(ft.name);
            alloc.free(ft.type);
            alloc.free(ft.encoding);
        };
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const name = try takeString(alloc, c.pdf_oxide_font_get_name(list, idx, &code), code);
            errdefer alloc.free(name);
            const ftype = try takeString(alloc, c.pdf_oxide_font_get_type(list, idx, &code), code);
            errdefer alloc.free(ftype);
            const encoding = try takeString(alloc, c.pdf_oxide_font_get_encoding(list, idx, &code), code);
            const embedded = c.pdf_oxide_font_is_embedded(list, idx, &code) != 0;
            const subset = c.pdf_oxide_font_is_subset(list, idx, &code) != 0;
            out[i] = .{
                .name = name,
                .type = ftype,
                .encoding = encoding,
                .embedded = embedded,
                .subset = subset,
            };
        }
        return out;
    }

    /// Free a slice returned by `embeddedFonts`.
    pub fn freeFonts(alloc: std.mem.Allocator, fonts: []Font) void {
        for (fonts) |ft| {
            alloc.free(ft.name);
            alloc.free(ft.type);
            alloc.free(ft.encoding);
        }
        alloc.free(fonts);
    }

    /// Embedded images on a (0-based) page. Caller owns the returned slice and each
    /// element's `format`/`colorspace`/`data`; free with `freeImages`.
    pub fn embeddedImages(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Image {
        var code: i32 = 0;
        const list = c.pdf_document_get_embedded_images(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_image_list_free(list);
        const n = c.pdf_oxide_image_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Image, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |im| {
            alloc.free(im.format);
            alloc.free(im.colorspace);
            alloc.free(im.data);
        };
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const width = c.pdf_oxide_image_get_width(list, idx, &code);
            const height = c.pdf_oxide_image_get_height(list, idx, &code);
            const bpc = c.pdf_oxide_image_get_bits_per_component(list, idx, &code);
            const format = try takeString(alloc, c.pdf_oxide_image_get_format(list, idx, &code), code);
            errdefer alloc.free(format);
            const colorspace = try takeString(alloc, c.pdf_oxide_image_get_colorspace(list, idx, &code), code);
            errdefer alloc.free(colorspace);
            var data_len: i32 = 0;
            const data_ptr = c.pdf_oxide_image_get_data(list, idx, &data_len, &code) orelse return fail(code);
            defer c.free_bytes(data_ptr);
            const dn: usize = if (data_len < 0) 0 else @intCast(data_len);
            const data = try alloc.dupe(u8, data_ptr[0..dn]);
            out[i] = .{
                .width = width,
                .height = height,
                .bitsPerComponent = bpc,
                .format = format,
                .colorspace = colorspace,
                .data = data,
            };
        }
        return out;
    }

    /// Free a slice returned by `embeddedImages`.
    pub fn freeImages(alloc: std.mem.Allocator, images: []Image) void {
        for (images) |im| {
            alloc.free(im.format);
            alloc.free(im.colorspace);
            alloc.free(im.data);
        }
        alloc.free(images);
    }

    /// Annotations on a (0-based) page. Caller owns the returned slice and each
    /// element's `type`/`subtype`/`content`/`author`; free with `freeAnnotations`.
    pub fn pageAnnotations(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Annotation {
        var code: i32 = 0;
        const list = c.pdf_document_get_page_annotations(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_annotation_list_free(list);
        const n = c.pdf_oxide_annotation_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Annotation, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |an| {
            alloc.free(an.type);
            alloc.free(an.subtype);
            alloc.free(an.content);
            alloc.free(an.author);
        };
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const atype = try takeString(alloc, c.pdf_oxide_annotation_get_type(list, idx, &code), code);
            errdefer alloc.free(atype);
            const subtype = try takeString(alloc, c.pdf_oxide_annotation_get_subtype(list, idx, &code), code);
            errdefer alloc.free(subtype);
            const content = try takeString(alloc, c.pdf_oxide_annotation_get_content(list, idx, &code), code);
            errdefer alloc.free(content);
            const author = try takeString(alloc, c.pdf_oxide_annotation_get_author(list, idx, &code), code);
            errdefer alloc.free(author);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_annotation_get_rect(list, idx, &x, &y, &w, &h, &code);
            const border_width = c.pdf_oxide_annotation_get_border_width(list, idx, &code);
            out[i] = .{
                .type = atype,
                .subtype = subtype,
                .content = content,
                .author = author,
                .rect = .{ .x = x, .y = y, .width = w, .height = h },
                .borderWidth = border_width,
            };
        }
        return out;
    }

    /// Free a slice returned by `pageAnnotations`.
    pub fn freeAnnotations(alloc: std.mem.Allocator, annotations: []Annotation) void {
        for (annotations) |an| {
            alloc.free(an.type);
            alloc.free(an.subtype);
            alloc.free(an.content);
            alloc.free(an.author);
        }
        alloc.free(annotations);
    }

    /// Vector paths on a (0-based) page. Caller owns the returned slice; free with
    /// `freePaths`.
    pub fn extractPaths(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]Path {
        var code: i32 = 0;
        const list = c.pdf_document_extract_paths(self.handle, page_index, &code) orelse return fail(code);
        defer c.pdf_oxide_path_list_free(list);
        const n = c.pdf_oxide_path_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(Path, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_path_get_bbox(list, idx, &x, &y, &w, &h, &code);
            const stroke_width = c.pdf_oxide_path_get_stroke_width(list, idx, &code);
            const has_stroke = c.pdf_oxide_path_has_stroke(list, idx, &code);
            const has_fill = c.pdf_oxide_path_has_fill(list, idx, &code);
            const op_count = c.pdf_oxide_path_get_operation_count(list, idx, &code);
            out[i] = .{
                .bbox = .{ .x = x, .y = y, .width = w, .height = h },
                .strokeWidth = stroke_width,
                .hasStroke = has_stroke,
                .hasFill = has_fill,
                .operationCount = op_count,
            };
        }
        return out;
    }

    /// Free a slice returned by `extractPaths`.
    pub fn freePaths(alloc: std.mem.Allocator, paths: []Path) void {
        alloc.free(paths);
    }

    /// Marshal an `FfiSearchResults` handle into an owned slice. Frees `list` on
    /// every path (including error). Caller owns each element's `text`.
    fn collectSearchResults(alloc: std.mem.Allocator, list: *c.FfiSearchResults) Error![]SearchResult {
        defer c.pdf_oxide_search_result_free(list);
        var code: i32 = 0;
        const n = c.pdf_oxide_search_result_count(list);
        if (n < 0) return fail(code);
        const count: usize = @intCast(n);
        const out = try alloc.alloc(SearchResult, count);
        errdefer alloc.free(out);
        var i: usize = 0;
        errdefer for (out[0..i]) |sr| alloc.free(sr.text);
        while (i < count) : (i += 1) {
            const idx: i32 = @intCast(i);
            const result_text = try takeString(alloc, c.pdf_oxide_search_result_get_text(list, idx, &code), code);
            errdefer alloc.free(result_text);
            const page_no = c.pdf_oxide_search_result_get_page(list, idx, &code);
            var x: f32 = 0;
            var y: f32 = 0;
            var w: f32 = 0;
            var h: f32 = 0;
            c.pdf_oxide_search_result_get_bbox(list, idx, &x, &y, &w, &h, &code);
            out[i] = .{
                .text = result_text,
                .page = page_no,
                .bbox = .{ .x = x, .y = y, .width = w, .height = h },
            };
        }
        return out;
    }

    /// Search a single (0-based) page for `term`. Caller owns the returned slice
    /// and each element's `text`; free with `freeSearchResults`.
    pub fn search(self: Document, alloc: std.mem.Allocator, page_index: i32, term: [:0]const u8, case_sensitive: bool) Error![]SearchResult {
        var code: i32 = 0;
        const list = c.pdf_document_search_page(self.handle, page_index, term.ptr, case_sensitive, &code) orelse return fail(code);
        return collectSearchResults(alloc, list);
    }

    /// Search every page for `term`. Caller owns the returned slice and each
    /// element's `text`; free with `freeSearchResults`.
    pub fn searchAll(self: Document, alloc: std.mem.Allocator, term: [:0]const u8, case_sensitive: bool) Error![]SearchResult {
        var code: i32 = 0;
        const list = c.pdf_document_search_all(self.handle, term.ptr, case_sensitive, &code) orelse return fail(code);
        return collectSearchResults(alloc, list);
    }

    /// Free a slice returned by `search`/`searchAll`.
    pub fn freeSearchResults(alloc: std.mem.Allocator, results: []SearchResult) void {
        for (results) |sr| alloc.free(sr.text);
        alloc.free(results);
    }

    /// Render a (0-based) page to an encoded image (`format`: 0 = PNG). Caller
    /// owns the returned `RenderedImage`; free it with `deinit`.
    pub fn renderPage(self: Document, alloc: std.mem.Allocator, page_index: i32, format: i32) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page(self.handle, page_index, format, &code) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Render a (0-based) page at the given `zoom` factor (`format`: 0 = PNG).
    /// Caller owns the returned `RenderedImage`; free it with `deinit`.
    pub fn renderPageZoom(self: Document, alloc: std.mem.Allocator, page_index: i32, zoom: f32, format: i32) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page_zoom(self.handle, page_index, zoom, format, &code) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Render a thumbnail of a (0-based) page fitting inside `size`×`size`
    /// pixels (`format`: 0 = PNG). Caller owns the returned `RenderedImage`;
    /// free it with `deinit`.
    pub fn renderPageThumbnail(self: Document, alloc: std.mem.Allocator, page_index: i32, size: i32, format: i32) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page_thumbnail(self.handle, page_index, size, format, &code) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// A lightweight view of a single (0-based) page. The returned `Page` borrows
    /// this `Document`'s handle, so the `Document` MUST outlive the `Page`.
    pub fn page(self: Document, index: i32) Page {
        return .{ .doc = self, .index = index };
    }
};

/// A rendered page image. Owns the native `FfiRenderedImage` handle so that
/// `save` can defer to the Rust encoder; `width`/`height`/`data` are read
/// eagerly and `data` is copied into an allocator-owned slice. Free with
/// `deinit` (releases both the copied bytes and the native handle).
pub const RenderedImage = struct {
    handle: *c.FfiRenderedImage,
    alloc: std.mem.Allocator,
    width: i32,
    height: i32,
    data: []u8,

    /// Adopt an `FfiRenderedImage` handle: read width/height, copy the encoded
    /// bytes (freeing the C buffer), and keep the handle alive for `save`. The
    /// handle is freed by `deinit`. On error the handle is freed here.
    fn take(alloc: std.mem.Allocator, img: *c.FfiRenderedImage) Error!RenderedImage {
        errdefer c.pdf_rendered_image_free(img);
        var code: i32 = 0;
        const width = c.pdf_get_rendered_image_width(img, &code);
        if (width < 0) return fail(code);
        const height = c.pdf_get_rendered_image_height(img, &code);
        if (height < 0) return fail(code);
        var data_len: i32 = 0;
        const data_ptr = c.pdf_get_rendered_image_data(img, &data_len, &code) orelse return fail(code);
        defer c.free_bytes(data_ptr);
        const dn: usize = if (data_len < 0) 0 else @intCast(data_len);
        const data = try alloc.dupe(u8, data_ptr[0..dn]);
        return .{
            .handle = img,
            .alloc = alloc,
            .width = width,
            .height = height,
            .data = data,
        };
    }

    /// Write the rendered image to `file_path` (NUL-terminated) using the Rust
    /// encoder. Uses the live native handle.
    pub fn save(self: RenderedImage, file_path: [:0]const u8) Error!void {
        var code: i32 = 0;
        if (c.pdf_save_rendered_image(self.handle, file_path.ptr, &code) != 0) return fail(code);
    }

    /// Free the copied bytes and the native handle.
    pub fn deinit(self: *RenderedImage) void {
        self.alloc.free(self.data);
        c.pdf_rendered_image_free(self.handle);
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

test "Document: phase-2 extraction (fonts/images/annotations/paths/search)" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // embeddedFonts: returns a list (may be empty) without error
    const fonts = try doc.embeddedFonts(a, 0);
    defer Document.freeFonts(a, fonts);

    // embeddedImages: returns a list (may be empty) without error
    const images = try doc.embeddedImages(a, 0);
    defer Document.freeImages(a, images);

    // pageAnnotations: returns a list (may be empty) without error
    const annotations = try doc.pageAnnotations(a, 0);
    defer Document.freeAnnotations(a, annotations);

    // extractPaths: returns a list (may be empty) without error
    const paths = try doc.extractPaths(a, 0);
    defer Document.freePaths(a, paths);

    // search: non-empty, first result text contains "Alpha", page >= 0
    const hits = try doc.search(a, 0, "Alpha", false);
    defer Document.freeSearchResults(a, hits);
    try testing.expect(hits.len > 0);
    try testing.expect(std.mem.indexOf(u8, hits[0].text, "Alpha") != null);
    try testing.expect(hits[0].page >= 0);

    // searchAll: non-empty, first result text contains "Alpha", page >= 0
    const all_hits = try doc.searchAll(a, "Alpha", false);
    defer Document.freeSearchResults(a, all_hits);
    try testing.expect(all_hits.len > 0);
    try testing.expect(std.mem.indexOf(u8, all_hits[0].text, "Alpha") != null);
    try testing.expect(all_hits[0].page >= 0);
}

test "Document: phase-3 page rendering (renderPage/renderPageZoom/renderPageThumbnail)" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // renderPage(0) as PNG: width > 0, height > 0, non-empty data
    {
        var img = try doc.renderPage(a, 0, 0); // renderPage
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.height > 0);
        try testing.expect(img.data.len > 0);

        // RenderedImage.save: writes the image without error
        try img.save("/tmp/pdfoxide_zig_render.png");
        const f = try std.fs.cwd().openFile("/tmp/pdfoxide_zig_render.png", .{});
        f.close();
        try std.fs.cwd().deleteFile("/tmp/pdfoxide_zig_render.png");
    }

    // renderPageZoom: returns a RenderedImage without error
    {
        var img = try doc.renderPageZoom(a, 0, 2.0, 0); // renderPageZoom
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.height > 0);
        try testing.expect(img.data.len > 0);
    }

    // renderPageThumbnail: returns a RenderedImage without error
    {
        var img = try doc.renderPageThumbnail(a, 0, 128, 0); // renderPageThumbnail
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.height > 0);
        try testing.expect(img.data.len > 0);
    }
}

test "error path: open nonexistent returns error" {
    try testing.expectError(Error.PdfOxide, Document.open("/nonexistent/nope.pdf"));
}
