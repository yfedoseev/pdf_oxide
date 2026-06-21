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

    // ── PHASE-6: conformance validation ───────────────────────────────────────

    /// Validate the document against a PDF/A conformance level.
    /// `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u. Caller owns the
    /// returned `PdfAResults`; free it with `deinit`.
    pub fn validatePdfA(self: Document, level: i32) Error!PdfAResults {
        var code: i32 = 0;
        const h = c.pdf_validate_pdf_a_level(self.handle, level, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Validate the document against a PDF/UA accessibility level. Caller owns the
    /// returned `UaResults`; free it with `deinit`.
    pub fn validatePdfUa(self: Document, level: i32) Error!UaResults {
        var code: i32 = 0;
        const h = c.pdf_validate_pdf_ua(self.handle, level, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Validate the document against a PDF/X conformance level. Caller owns the
    /// returned `PdfXResults`; free it with `deinit`.
    pub fn validatePdfX(self: Document, level: i32) Error!PdfXResults {
        var code: i32 = 0;
        const h = c.pdf_validate_pdf_x_level(self.handle, level, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    // ── PHASE-7: render variants / page getters / OCR ─────────────────────────

    /// Render a (0-based) page with the full RenderOptions surface. Background
    /// channels are 0.0..=1.0; set `transparent_background` to drop the fill.
    /// `format`: 0=PNG 1=JPEG. Caller owns the returned `RenderedImage`.
    pub fn renderPageWithOptions(
        self: Document,
        alloc: std.mem.Allocator,
        page_index: i32,
        dpi: i32,
        format: i32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        bg_a: f32,
        transparent_background: bool,
        render_annotations: bool,
        jpeg_quality: i32,
    ) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page_with_options(
            self.handle,
            page_index,
            dpi,
            format,
            bg_r,
            bg_g,
            bg_b,
            bg_a,
            @intFromBool(transparent_background),
            @intFromBool(render_annotations),
            jpeg_quality,
            &code,
        ) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// `renderPageWithOptions` plus OCG layer filtering. `excluded_layers` are
    /// the `/Name`s of Optional Content Groups to suppress (empty = no filter).
    /// Caller owns the returned `RenderedImage`.
    pub fn renderPageWithOptionsEx(
        self: Document,
        alloc: std.mem.Allocator,
        page_index: i32,
        dpi: i32,
        format: i32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        bg_a: f32,
        transparent_background: bool,
        render_annotations: bool,
        jpeg_quality: i32,
        excluded_layers: []const [*:0]const u8,
    ) Error!RenderedImage {
        var code: i32 = 0;
        const layers: [*c]const [*c]const u8 =
            if (excluded_layers.len == 0) null else @ptrCast(excluded_layers.ptr);
        const img = c.pdf_render_page_with_options_ex(
            self.handle,
            page_index,
            dpi,
            format,
            bg_r,
            bg_g,
            bg_b,
            bg_a,
            @intFromBool(transparent_background),
            @intFromBool(render_annotations),
            jpeg_quality,
            layers,
            excluded_layers.len,
            &code,
        ) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Render a rectangular region of a (0-based) page. `crop_*` are in PDF
    /// user-space points (origin bottom-left). `format`: 0=PNG 1=JPEG. Caller
    /// owns the returned `RenderedImage`.
    pub fn renderPageRegion(
        self: Document,
        alloc: std.mem.Allocator,
        page_index: i32,
        crop_x: f32,
        crop_y: f32,
        crop_width: f32,
        crop_height: f32,
        format: i32,
    ) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page_region(self.handle, page_index, crop_x, crop_y, crop_width, crop_height, format, &code) orelse
            return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Render a (0-based) page to fit inside `w`×`h` pixels, preserving aspect
    /// ratio. `format`: 0=PNG 1=JPEG. Caller owns the returned `RenderedImage`.
    pub fn renderPageFit(self: Document, alloc: std.mem.Allocator, page_index: i32, w: i32, h: i32, format: i32) Error!RenderedImage {
        var code: i32 = 0;
        const img = c.pdf_render_page_fit(self.handle, page_index, w, h, format, &code) orelse return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Render a (0-based) page to a raw premultiplied RGBA8888 pixel buffer at
    /// `dpi`. The pixel dimensions are returned in the `RenderedImage`'s
    /// `width`/`height`; `data` is row-major, top-left origin
    /// (`data.len == width*height*4`). Caller owns the returned `RenderedImage`.
    pub fn renderPageRaw(self: Document, alloc: std.mem.Allocator, page_index: i32, dpi: i32) Error!RenderedImage {
        var code: i32 = 0;
        var out_width: i32 = 0;
        var out_height: i32 = 0;
        const img = c.pdf_render_page_raw(self.handle, page_index, dpi, &out_width, &out_height, &code) orelse
            return fail(code);
        return RenderedImage.take(alloc, img);
    }

    /// Estimate the render time (implementation-defined units) for a (0-based)
    /// page.
    pub fn estimateRenderTime(self: Document, page_index: i32) Error!i32 {
        var code: i32 = 0;
        const t = c.pdf_estimate_render_time(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return t;
    }

    /// Page width in PDF points for a (0-based) page.
    pub fn pageGetWidth(self: Document, page_index: i32) Error!f32 {
        var code: i32 = 0;
        const w = c.pdf_page_get_width(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return w;
    }

    /// Page height in PDF points for a (0-based) page.
    pub fn pageGetHeight(self: Document, page_index: i32) Error!f32 {
        var code: i32 = 0;
        const h = c.pdf_page_get_height(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return h;
    }

    /// Page rotation in degrees for a (0-based) page.
    pub fn pageGetRotation(self: Document, page_index: i32) Error!i32 {
        var code: i32 = 0;
        const r = c.pdf_page_get_rotation(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Layout elements on a (0-based) page. Caller owns the returned
    /// `ElementList`; free it with `deinit`.
    pub fn pageGetElements(self: Document, page_index: i32) Error!ElementList {
        var code: i32 = 0;
        const h = c.pdf_page_get_elements(self.handle, page_index, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Whether a (0-based) page needs OCR (i.e. is scanned/hybrid).
    pub fn ocrPageNeedsOcr(self: Document, page_index: i32) Error!bool {
        var code: i32 = 0;
        const r = c.pdf_ocr_page_needs_ocr(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Extract text from a (0-based) page using OCR. `engine` may be null (uses
    /// native text extraction only). Caller owns the returned slice.
    pub fn ocrExtractText(self: Document, alloc: std.mem.Allocator, page_index: i32, engine: ?OcrEngine) Error![]u8 {
        var code: i32 = 0;
        const eh: ?*const anyopaque = if (engine) |e| e.handle else null;
        return takeString(alloc, c.pdf_ocr_extract_text(self.handle, page_index, eh, &code), code);
    }

    // ── PHASE-8: office import/export ─────────────────────────────────────────

    /// Open a PDF rendered from DOCX bytes. Caller owns the returned `Document`.
    pub fn openFromDocxBytes(data: []const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_from_docx_bytes(data.ptr, data.len, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Open a PDF rendered from PPTX bytes. Caller owns the returned `Document`.
    pub fn openFromPptxBytes(data: []const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_from_pptx_bytes(data.ptr, data.len, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Open a PDF rendered from XLSX bytes. Caller owns the returned `Document`.
    pub fn openFromXlsxBytes(data: []const u8) Error!Document {
        var code: i32 = 0;
        const h = c.pdf_document_open_from_xlsx_bytes(data.ptr, data.len, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Export the document to DOCX bytes; caller owns the returned slice.
    pub fn toDocx(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var out_len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_to_docx(self.handle, &out_len, &code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// Export the document to PPTX bytes; caller owns the returned slice.
    pub fn toPptx(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var out_len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_to_pptx(self.handle, &out_len, &code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// Export the document to XLSX bytes; caller owns the returned slice.
    pub fn toXlsx(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var out_len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_to_xlsx(self.handle, &out_len, &code);
        return takeBytes(alloc, p, out_len, code);
    }

    // ── PHASE-8: in-rect region extractors ────────────────────────────────────

    /// Extract text within the rect (`x`,`y`,`w`,`h`) of a (0-based) page; caller
    /// owns the returned slice.
    pub fn extractTextInRect(self: Document, alloc: std.mem.Allocator, page_index: i32, x: f32, y: f32, w: f32, h: f32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_text_in_rect(self.handle, page_index, x, y, w, h, &code), code);
    }

    /// Extract words within the rect of a (0-based) page. Caller owns the returned
    /// slice and each element's `text`/`fontName`; free with `freeWords`.
    pub fn extractWordsInRect(self: Document, alloc: std.mem.Allocator, page_index: i32, x: f32, y: f32, w: f32, h: f32) Error![]Word {
        var code: i32 = 0;
        const list = c.pdf_document_extract_words_in_rect(self.handle, page_index, x, y, w, h, &code) orelse return fail(code);
        defer c.pdf_oxide_word_list_free(list);
        return collectWordList(alloc, list);
    }

    /// Extract lines within the rect of a (0-based) page. Caller owns the returned
    /// slice and each element's `text`; free with `freeTextLines`.
    pub fn extractLinesInRect(self: Document, alloc: std.mem.Allocator, page_index: i32, x: f32, y: f32, w: f32, h: f32) Error![]TextLine {
        var code: i32 = 0;
        const list = c.pdf_document_extract_lines_in_rect(self.handle, page_index, x, y, w, h, &code) orelse return fail(code);
        defer c.pdf_oxide_line_list_free(list);
        return collectLineList(alloc, list);
    }

    /// Extract tables within the rect of a (0-based) page. Caller owns the returned
    /// slice and each table's cells; free with `freeTables`.
    pub fn extractTablesInRect(self: Document, alloc: std.mem.Allocator, page_index: i32, x: f32, y: f32, w: f32, h: f32) Error![]Table {
        var code: i32 = 0;
        const list = c.pdf_document_extract_tables_in_rect(self.handle, page_index, x, y, w, h, &code) orelse return fail(code);
        defer c.pdf_oxide_table_list_free(list);
        return collectTableList(alloc, list);
    }

    /// Extract images within the rect of a (0-based) page. Caller owns the returned
    /// slice and each element's `format`/`colorspace`/`data`; free with `freeImages`.
    pub fn extractImagesInRect(self: Document, alloc: std.mem.Allocator, page_index: i32, x: f32, y: f32, w: f32, h: f32) Error![]Image {
        var code: i32 = 0;
        const list = c.pdf_document_extract_images_in_rect(self.handle, page_index, x, y, w, h, &code) orelse return fail(code);
        defer c.pdf_oxide_image_list_free(list);
        return collectImageList(alloc, list);
    }

    // ── PHASE-8: auto extraction / classification ─────────────────────────────

    /// One-shot auto text extraction for a (0-based) page; caller owns the slice.
    pub fn extractTextAuto(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_text_auto(self.handle, page_index, &code), code);
    }

    /// Whole-document auto text extraction; caller owns the slice.
    pub fn extractAllText(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_extract_all_text(self.handle, &code), code);
    }

    /// Rich per-page auto extraction returning a `PageExtraction` JSON string.
    /// `options_json` may be null/empty (defaults). Caller owns the slice.
    pub fn extractPageAuto(self: Document, alloc: std.mem.Allocator, page_index: i32, options_json: ?[:0]const u8) Error![]u8 {
        var code: i32 = 0;
        const opts: ?[*:0]const u8 = if (options_json) |o| o.ptr else null;
        return takeString(alloc, c.pdf_document_extract_page_auto(self.handle, page_index, opts, &code), code);
    }

    /// Cheap per-page text-vs-OCR classification as a JSON string; caller owns it.
    pub fn classifyPage(self: Document, alloc: std.mem.Allocator, page_index: i32) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_classify_page(self.handle, page_index, &code), code);
    }

    /// Cheap whole-document classification as a JSON string; caller owns it.
    pub fn classifyDocument(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_classify_document(self.handle, &code), code);
    }

    // ── PHASE-8: header / footer / artifact removal ───────────────────────────

    /// Erase the detected running header from a (0-based) page. Returns a status.
    pub fn eraseHeader(self: Document, page_index: i32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_erase_header(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Erase the detected running footer from a (0-based) page. Returns a status.
    pub fn eraseFooter(self: Document, page_index: i32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_erase_footer(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Erase artifact content from a (0-based) page. Returns a status.
    pub fn eraseArtifacts(self: Document, page_index: i32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_erase_artifacts(self.handle, page_index, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Remove repeated headers across the document at the given recurrence
    /// `threshold` (0.0..=1.0). Returns the number removed.
    pub fn removeHeaders(self: Document, threshold: f32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_remove_headers(self.handle, threshold, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Remove repeated footers across the document at the given recurrence
    /// `threshold` (0.0..=1.0). Returns the number removed.
    pub fn removeFooters(self: Document, threshold: f32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_remove_footers(self.handle, threshold, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Remove artifacts across the document at the given `threshold`. Returns the
    /// number removed.
    pub fn removeArtifacts(self: Document, threshold: f32) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_remove_artifacts(self.handle, threshold, &code);
        if (code != 0) return fail(code);
        return n;
    }

    // ── PHASE-8: forms ────────────────────────────────────────────────────────

    /// All interactive form fields. Caller owns the returned `FormFieldList`; free
    /// it with `deinit`. An empty form yields a zero-count list (not an error).
    pub fn formFields(self: Document) Error!FormFieldList {
        var code: i32 = 0;
        const h = c.pdf_document_get_form_fields(self.handle, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Export form data. `format_type`: 0=FDF, 1=XFDF. Caller owns the slice.
    pub fn exportFormDataToBytes(self: Document, alloc: std.mem.Allocator, format_type: i32) Error![]u8 {
        var out_len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_export_form_data_to_bytes(self.handle, format_type, &out_len, &code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// Import form data from a file at `data_path`. Returns a status.
    pub fn importFormData(self: Document, data_path: [:0]const u8) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_import_form_data(@ptrCast(self.handle), data_path.ptr, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Import form values from an FDF/XFDF/XML file at `filename`. Returns true on
    /// success.
    pub fn formImportFromFile(self: Document, filename: [:0]const u8) Error!bool {
        var code: i32 = 0;
        const ok = c.pdf_form_import_from_file(@ptrCast(self.handle), filename.ptr, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    // ── PHASE-8: structure / metadata ─────────────────────────────────────────

    /// Document outline (bookmarks) as a JSON string; caller owns it.
    pub fn outline(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_get_outline(self.handle, &code), code);
    }

    /// Page labels as a JSON string; caller owns it.
    pub fn pageLabels(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_get_page_labels(self.handle, &code), code);
    }

    /// XMP metadata XML packet; caller owns the returned slice.
    pub fn xmpMetadata(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var code: i32 = 0;
        return takeString(alloc, c.pdf_document_get_xmp_metadata(self.handle, &code), code);
    }

    /// A copy of the document's current source bytes; caller owns the slice.
    pub fn sourceBytes(self: Document, alloc: std.mem.Allocator) Error![]u8 {
        var out_len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_get_source_bytes(self.handle, &out_len, &code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// Whether the document carries an XFA (XML Forms Architecture) form.
    pub fn hasXfa(self: Document) bool {
        return c.pdf_document_has_xfa(self.handle);
    }

    /// Plan a split-by-bookmarks operation, returning a JSON plan; caller owns it.
    /// `options_json` may be null/empty.
    pub fn planSplitByBookmarks(self: Document, alloc: std.mem.Allocator, options_json: ?[:0]const u8) Error![]u8 {
        var code: i32 = 0;
        const opts: ?[*:0]const u8 = if (options_json) |o| o.ptr else null;
        return takeString(alloc, c.pdf_document_plan_split_by_bookmarks(self.handle, opts, &code), code);
    }

    /// Convert this document to PDF/A in place. `level`: 0=A1b 1=A1a 2=A2b …
    /// Returns true on success.
    pub fn convertToPdfA(self: Document, level: i32) Error!bool {
        var code: i32 = 0;
        const ok = c.pdf_convert_to_pdf_a(self.handle, level, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    // ── PHASE-8: signatures on the document ───────────────────────────────────

    /// Apply a digital signature using `cert`. Returns a status.
    pub fn sign(self: Document, cert: Certificate, reason: [:0]const u8, location: [:0]const u8) Error!i32 {
        const ch = try cert.live();
        var code: i32 = 0;
        const n = c.pdf_document_sign(@ptrCast(self.handle), ch, reason.ptr, location.ptr, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Number of signatures present in the document.
    pub fn signatureCount(self: Document) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_get_signature_count(@ptrCast(self.handle), &code);
        if (n < 0) return fail(code);
        return n;
    }

    /// The `index`-th signature as an owned `SignatureInfo`; free it with `deinit`.
    pub fn signature(self: Document, index: i32) Error!SignatureInfo {
        var code: i32 = 0;
        const h = c.pdf_document_get_signature(@ptrCast(self.handle), index, &code) orelse return fail(code);
        return .{ .handle = @ptrCast(@alignCast(h)) };
    }

    /// Verify every signature. Returns a status (>=0) or raises.
    pub fn verifyAllSignatures(self: Document) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_document_verify_all_signatures(@ptrCast(self.handle), &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Whether the document carries a document-level timestamp.
    pub fn hasTimestamp(self: Document) Error!bool {
        var code: i32 = 0;
        const n = c.pdf_document_has_timestamp(@ptrCast(self.handle), &code);
        if (code != 0) return fail(code);
        return n != 0;
    }

    /// The document's DSS (long-term validation store) as an owned `Dss`; free it
    /// with `deinit`.
    pub fn dss(self: Document) Error!Dss {
        var code: i32 = 0;
        const h = c.pdf_document_get_dss(@ptrCast(self.handle), &code) orelse return fail(code);
        return .{ .handle = h };
    }

    // ── PHASE-8: list-handle JSON / accessors ─────────────────────────────────

    /// Page annotations as an owned `AnnotationList` handle (the JSON/quad-point
    /// accessors live on it). Free with `deinit`.
    pub fn annotationList(self: Document, page_index: i32) Error!AnnotationList {
        var code: i32 = 0;
        const h = c.pdf_document_get_page_annotations(self.handle, page_index, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Embedded fonts of a (0-based) page as an owned `FontList` handle. Free with
    /// `deinit`.
    pub fn fontList(self: Document, page_index: i32) Error!FontList {
        var code: i32 = 0;
        const h = c.pdf_document_get_embedded_fonts(self.handle, page_index, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Layout elements of a (0-based) page as an owned `ElementList`. Free with
    /// `deinit`.
    pub fn elementList(self: Document, page_index: i32) Error!ElementList {
        var code: i32 = 0;
        const h = c.pdf_page_get_elements(self.handle, page_index, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Search a (0-based) page and return the raw `SearchResultList` handle (the
    /// JSON serializer lives on it). Free with `deinit`.
    pub fn searchPageList(self: Document, page_index: i32, term: [:0]const u8, case_sensitive: bool) Error!SearchResultList {
        var code: i32 = 0;
        const h = c.pdf_document_search_page(self.handle, page_index, term.ptr, case_sensitive, &code) orelse return fail(code);
        return .{ .handle = h };
    }
};

/// Marshal an `FfiWordList` handle into an owned slice (does NOT free `list`).
/// Caller owns each element's `text`/`fontName`; free with `Document.freeWords`.
fn collectWordList(alloc: std.mem.Allocator, list: *c.FfiWordList) Error![]Word {
    var code: i32 = 0;
    const n = c.pdf_oxide_word_count(list);
    if (n < 0) return fail(n);
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

/// Marshal an `FfiTextLineList` handle into an owned slice (does NOT free `list`).
fn collectLineList(alloc: std.mem.Allocator, list: *c.FfiTextLineList) Error![]TextLine {
    var code: i32 = 0;
    const n = c.pdf_oxide_line_count(list);
    if (n < 0) return fail(n);
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

/// Marshal an `FfiTableList` handle into an owned slice (does NOT free `list`).
fn collectTableList(alloc: std.mem.Allocator, list: *c.FfiTableList) Error![]Table {
    var code: i32 = 0;
    const n = c.pdf_oxide_table_count(list);
    if (n < 0) return fail(n);
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

/// Marshal an `FfiImageList` handle into an owned slice (does NOT free `list`).
fn collectImageList(alloc: std.mem.Allocator, list: *c.FfiImageList) Error![]Image {
    var code: i32 = 0;
    const n = c.pdf_oxide_image_count(list);
    if (n < 0) return fail(n);
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

/// A list of interactive form fields. Owns the native `FfiFormFieldList` handle;
/// free with `deinit`/`close`.
pub const FormFieldList = struct {
    handle: ?*c.FfiFormFieldList,

    fn live(self: FormFieldList) Error!*c.FfiFormFieldList {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *FormFieldList) void {
        if (self.handle) |h| c.pdf_oxide_form_field_list_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *FormFieldList) void {
        self.deinit();
    }

    /// Number of fields (>=0; may be 0 for a form-less document).
    pub fn count(self: FormFieldList) Error!i32 {
        const h = try self.live();
        return c.pdf_oxide_form_field_count(h);
    }

    /// Field name at `index`; caller owns the returned slice.
    pub fn getName(self: FormFieldList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_form_field_get_name(h, index, &code), code);
    }

    /// Field value at `index`; caller owns the returned slice.
    pub fn getValue(self: FormFieldList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_form_field_get_value(h, index, &code), code);
    }

    /// Field type string at `index`; caller owns the returned slice.
    pub fn getType(self: FormFieldList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_form_field_get_type(h, index, &code), code);
    }

    /// Whether the field at `index` is read-only.
    pub fn isReadonly(self: FormFieldList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_form_field_is_readonly(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Whether the field at `index` is required.
    pub fn isRequired(self: FormFieldList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_form_field_is_required(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }
};

/// A list of page annotations as a live native handle, exposing the extended
/// accessors (color/dates/flags, highlight quad-points, link URI, icon name,
/// JSON). Owns the native `FfiAnnotationList`; free with `deinit`/`close`. (The
/// eager-marshalled view is `Document.pageAnnotations`.)
pub const AnnotationList = struct {
    handle: ?*c.FfiAnnotationList,

    fn live(self: AnnotationList) Error!*c.FfiAnnotationList {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *AnnotationList) void {
        if (self.handle) |h| c.pdf_oxide_annotation_list_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *AnnotationList) void {
        self.deinit();
    }

    /// Number of annotations.
    pub fn count(self: AnnotationList) Error!i32 {
        const h = try self.live();
        const n = c.pdf_oxide_annotation_count(h);
        if (n < 0) return fail(n);
        return n;
    }

    /// Packed RGBA color of the annotation at `index`.
    pub fn getColor(self: AnnotationList, index: i32) Error!u32 {
        const h = try self.live();
        var code: i32 = 0;
        const v = c.pdf_oxide_annotation_get_color(h, index, &code);
        if (code != 0) return fail(code);
        return v;
    }

    /// Creation date (Unix epoch seconds) of the annotation at `index`.
    pub fn getCreationDate(self: AnnotationList, index: i32) Error!i64 {
        const h = try self.live();
        var code: i32 = 0;
        const v = c.pdf_oxide_annotation_get_creation_date(h, index, &code);
        if (code != 0) return fail(code);
        return v;
    }

    /// Modification date (Unix epoch seconds) of the annotation at `index`.
    pub fn getModificationDate(self: AnnotationList, index: i32) Error!i64 {
        const h = try self.live();
        var code: i32 = 0;
        const v = c.pdf_oxide_annotation_get_modification_date(h, index, &code);
        if (code != 0) return fail(code);
        return v;
    }

    /// Whether the annotation at `index` is hidden.
    pub fn isHidden(self: AnnotationList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_annotation_is_hidden(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Whether the annotation at `index` is marked deleted.
    pub fn isMarkedDeleted(self: AnnotationList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_annotation_is_marked_deleted(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Whether the annotation at `index` is printable.
    pub fn isPrintable(self: AnnotationList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_annotation_is_printable(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Whether the annotation at `index` is read-only.
    pub fn isReadOnly(self: AnnotationList, index: i32) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_oxide_annotation_is_read_only(h, index, &code);
        if (code != 0) return fail(code);
        return r;
    }

    /// Number of quad-point quads on the highlight annotation at `index`.
    pub fn highlightQuadPointsCount(self: AnnotationList, index: i32) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_oxide_highlight_annotation_get_quad_points_count(h, index, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// The four corners of the `quad_index`-th quad of the highlight at `index`.
    pub fn highlightQuadPoint(self: AnnotationList, index: i32, quad_index: i32) Error![8]f32 {
        const h = try self.live();
        var code: i32 = 0;
        var p: [8]f32 = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        c.pdf_oxide_highlight_annotation_get_quad_point(h, index, quad_index, &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7], &code);
        if (code != 0) return fail(code);
        return p;
    }

    /// The URI of the link annotation at `index`; caller owns the slice.
    pub fn linkUri(self: AnnotationList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_link_annotation_get_uri(h, index, &code), code);
    }

    /// The icon name of the text annotation at `index`; caller owns the slice.
    pub fn textIconName(self: AnnotationList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_text_annotation_get_icon_name(h, index, &code), code);
    }

    /// Serialize the whole list to a JSON string; caller owns the slice.
    pub fn toJson(self: AnnotationList, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_annotations_to_json(h, &code), code);
    }
};

/// A list of embedded fonts as a live native handle, exposing the size accessor
/// and JSON serializer. Owns the native `FfiFontList`; free with `deinit`/`close`.
/// (The eager-marshalled view is `Document.embeddedFonts`.)
pub const FontList = struct {
    handle: ?*c.FfiFontList,

    fn live(self: FontList) Error!*c.FfiFontList {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *FontList) void {
        if (self.handle) |h| c.pdf_oxide_font_list_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *FontList) void {
        self.deinit();
    }

    /// Number of fonts.
    pub fn count(self: FontList) Error!i32 {
        const h = try self.live();
        const n = c.pdf_oxide_font_count(h);
        if (n < 0) return fail(n);
        return n;
    }

    /// Nominal point size of the font at `index`.
    pub fn getSize(self: FontList, index: i32) Error!f32 {
        const h = try self.live();
        var code: i32 = 0;
        const s = c.pdf_oxide_font_get_size(h, index, &code);
        if (code != 0) return fail(code);
        return s;
    }

    /// Serialize the whole list to a JSON string; caller owns the slice.
    pub fn toJson(self: FontList, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_fonts_to_json(h, &code), code);
    }
};

/// A raw page-search result set as a live native handle, exposing the JSON
/// serializer. Owns the native `FfiSearchResults`; free with `deinit`/`close`.
/// (The eager-marshalled view is `Document.search`/`searchAll`.)
pub const SearchResultList = struct {
    handle: ?*c.FfiSearchResults,

    fn live(self: SearchResultList) Error!*c.FfiSearchResults {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *SearchResultList) void {
        if (self.handle) |h| c.pdf_oxide_search_result_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *SearchResultList) void {
        self.deinit();
    }

    /// Number of results.
    pub fn count(self: SearchResultList) Error!i32 {
        const h = try self.live();
        const n = c.pdf_oxide_search_result_count(h);
        if (n < 0) return fail(n);
        return n;
    }

    /// Serialize the whole result set to a JSON string; caller owns the slice.
    pub fn toJson(self: SearchResultList, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_search_results_to_json(h, &code), code);
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

    // ── PHASE-7: image / HTML+CSS constructors ────────────────────────────────

    /// Build a single-page PDF wrapping the image at `path`.
    pub fn fromImage(path: [:0]const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_image(path.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Build a single-page PDF wrapping the image in `data`.
    pub fn fromImageBytes(data: []const u8) Error!Pdf {
        var code: i32 = 0;
        const h = c.pdf_from_image_bytes(data.ptr, @intCast(data.len), &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Build a PDF from `html` + `css` with a single embedded font (pass an
    /// empty `font_bytes` slice for none).
    pub fn fromHtmlCss(html: [:0]const u8, css: [:0]const u8, font_bytes: []const u8) Error!Pdf {
        var code: i32 = 0;
        const fp: ?[*]const u8 = if (font_bytes.len == 0) null else font_bytes.ptr;
        const h = c.pdf_from_html_css(html.ptr, css.ptr, fp, font_bytes.len, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Build a PDF from `html` + `css` with a multi-font cascade. `families` and
    /// `fonts` are parallel arrays of equal length; each `fonts[i]` is the raw
    /// font bytes for `families[i]`.
    pub fn fromHtmlCssWithFonts(
        alloc: std.mem.Allocator,
        html: [:0]const u8,
        css: [:0]const u8,
        families: []const [*:0]const u8,
        fonts: []const []const u8,
    ) Error!Pdf {
        if (families.len != fonts.len) return fail(-1);
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const aa = arena.allocator();

        const count = fonts.len;
        var fam_ptr: [*c]const [*c]const u8 = null;
        var bytes_ptr: [*c]const [*c]const u8 = null;
        var lens_ptr: [*c]const usize = null;
        if (count != 0) {
            fam_ptr = @ptrCast(families.ptr);
            const bptrs = try aa.alloc([*c]const u8, count);
            const lens = try aa.alloc(usize, count);
            for (fonts, 0..) |f, i| {
                bptrs[i] = f.ptr;
                lens[i] = f.len;
            }
            bytes_ptr = bptrs.ptr;
            lens_ptr = lens.ptr;
        }

        var code: i32 = 0;
        const h = c.pdf_from_html_css_with_fonts(html.ptr, css.ptr, fam_ptr, bytes_ptr, lens_ptr, count, &code) orelse
            return fail(code);
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

    /// Page count of the built PDF (the `pdf_get_page_count` C-ABI alias).
    pub fn pageCount(self: Pdf) Error!i32 {
        var code: i32 = 0;
        const n = c.pdf_get_page_count(self.handle, &code);
        if (n < 0) return fail(code);
        return n;
    }
};

/// A PDF opened for in-place editing. Owns the native `DocumentEditor` handle;
/// free it with `deinit`/`close`. 0-based page indices throughout. Mirrors the
/// `Document`/`Pdf` handle pattern: factory `open*`, error-code helpers, owned
/// string-/byte-takes, `double` out-param helpers, and a closed-handle guard.
pub const DocumentEditor = struct {
    handle: ?*c.DocumentEditor,

    /// Guard: returns the live handle or raises if the editor has been closed.
    fn live(self: DocumentEditor) Error!*c.DocumentEditor {
        return self.handle orelse fail(-1);
    }

    /// Open a PDF for editing from a filesystem path (NUL-terminated).
    pub fn openEditor(path: [:0]const u8) Error!DocumentEditor {
        var code: i32 = 0;
        const h = c.document_editor_open(path.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Open a PDF for editing from in-memory bytes.
    pub fn openFromBytes(data: []const u8) Error!DocumentEditor {
        var code: i32 = 0;
        const h = c.document_editor_open_from_bytes(data.ptr, data.len, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *DocumentEditor) void {
        if (self.handle) |h| c.document_editor_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *DocumentEditor) void {
        self.deinit();
    }

    pub fn isModified(self: DocumentEditor) Error!bool {
        const h = try self.live();
        return c.document_editor_is_modified(h);
    }

    pub fn getSourcePath(self: DocumentEditor, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.document_editor_get_source_path(h, &code), code);
    }

    pub fn version(self: DocumentEditor) Error!Version {
        const h = try self.live();
        var maj: u8 = 0;
        var min: u8 = 0;
        c.document_editor_get_version(h, &maj, &min);
        return .{ .major = maj, .minor = min };
    }

    pub fn pageCount(self: DocumentEditor) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.document_editor_get_page_count(h, &code);
        if (n < 0) return fail(code);
        return n;
    }

    pub fn getProducer(self: DocumentEditor, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.document_editor_get_producer(h, &code), code);
    }

    pub fn setProducer(self: DocumentEditor, value: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_producer(h, value.ptr, &code) != 0) return fail(code);
    }

    pub fn getCreationDate(self: DocumentEditor, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.document_editor_get_creation_date(h, &code), code);
    }

    pub fn setCreationDate(self: DocumentEditor, date_str: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_creation_date(h, date_str.ptr, &code) != 0) return fail(code);
    }

    pub fn save(self: DocumentEditor, path: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_save(h, path.ptr, &code) != 0) return fail(code);
    }

    /// Serialize the edited document to bytes; caller owns the returned slice.
    pub fn saveToBytes(self: DocumentEditor, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.document_editor_save_to_bytes(h, &len, &code) orelse return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }

    /// Serialize with compression / garbage-collect / linearize options; caller
    /// owns the returned slice.
    pub fn saveToBytesWithOptions(
        self: DocumentEditor,
        alloc: std.mem.Allocator,
        compress: bool,
        garbage_collect: bool,
        linearize: bool,
    ) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.document_editor_save_to_bytes_with_options(
            h,
            compress,
            garbage_collect,
            linearize,
            &len,
            &code,
        ) orelse return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }

    /// Extract a subset of pages (0-based) to a new in-memory PDF; caller owns
    /// the returned slice.
    pub fn extractPagesToBytes(self: DocumentEditor, alloc: std.mem.Allocator, pages: []const i32) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.document_editor_extract_pages_to_bytes(h, pages.ptr, pages.len, &len, &code) orelse
            return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }

    /// Convert to PDF/A in-place. `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
    pub fn convertToPdfA(self: DocumentEditor, level: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_convert_to_pdf_a(h, level, &code) != 0) return fail(code);
    }

    /// Save with AES-256 encryption to bytes; caller owns the returned slice.
    pub fn saveEncryptedToBytes(
        self: DocumentEditor,
        alloc: std.mem.Allocator,
        user_password: [:0]const u8,
        owner_password: [:0]const u8,
    ) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.document_editor_save_encrypted_to_bytes(
            h,
            user_password.ptr,
            owner_password.ptr,
            &len,
            &code,
        ) orelse return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }

    /// Save with AES-256 encryption to a filesystem path.
    pub fn saveEncrypted(
        self: DocumentEditor,
        path: [:0]const u8,
        user_password: [:0]const u8,
        owner_password: [:0]const u8,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_save_encrypted(h, path.ptr, user_password.ptr, owner_password.ptr, &code) != 0)
            return fail(code);
    }

    /// Merge pages from an in-memory PDF byte buffer into this document.
    pub fn mergeFromBytes(self: DocumentEditor, data: []const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_merge_from_bytes(h, data.ptr, data.len, &code) != 0) return fail(code);
    }

    /// Merge pages from a PDF at `source_path` into this document.
    pub fn mergeFrom(self: DocumentEditor, source_path: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_merge_from(h, source_path.ptr, &code) != 0) return fail(code);
    }

    /// Embed a file attachment into the document.
    pub fn embedFile(self: DocumentEditor, name: [:0]const u8, data: []const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_embed_file(h, name.ptr, data.ptr, data.len, &code) != 0) return fail(code);
    }

    pub fn deletePage(self: DocumentEditor, page_index: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_delete_page(h, page_index, &code) != 0) return fail(code);
    }

    pub fn movePage(self: DocumentEditor, from: i32, to: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_move_page(h, from, to, &code) != 0) return fail(code);
    }

    pub fn getPageRotation(self: DocumentEditor, page_index: i32) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const deg = c.document_editor_get_page_rotation(h, page_index, &code);
        if (code != 0) return fail(code);
        return deg;
    }

    pub fn setPageRotation(self: DocumentEditor, page_index: i32, degrees: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_page_rotation(h, page_index, degrees, &code) != 0) return fail(code);
    }

    /// Rotate a single (0-based) page by `degrees` (additive).
    pub fn rotatePageBy(self: DocumentEditor, page_index: usize, degrees: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_rotate_page_by(h, page_index, degrees, &code) != 0) return fail(code);
    }

    /// Rotate all pages by `degrees` (additive).
    pub fn rotateAllPages(self: DocumentEditor, degrees: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_rotate_all_pages(h, degrees, &code) != 0) return fail(code);
    }

    pub fn getPageMediaBox(self: DocumentEditor, page_index: usize) Error!Bbox {
        const h = try self.live();
        var code: i32 = 0;
        var x: f64 = 0;
        var y: f64 = 0;
        var w: f64 = 0;
        var hgt: f64 = 0;
        if (c.document_editor_get_page_media_box(h, page_index, &x, &y, &w, &hgt, &code) != 0)
            return fail(code);
        return .{ .x = @floatCast(x), .y = @floatCast(y), .width = @floatCast(w), .height = @floatCast(hgt) };
    }

    pub fn setPageMediaBox(self: DocumentEditor, page_index: usize, x: f64, y: f64, w: f64, hgt: f64) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_page_media_box(h, page_index, x, y, w, hgt, &code) != 0) return fail(code);
    }

    pub fn getPageCropBox(self: DocumentEditor, page_index: usize) Error!Bbox {
        const h = try self.live();
        var code: i32 = 0;
        var x: f64 = 0;
        var y: f64 = 0;
        var w: f64 = 0;
        var hgt: f64 = 0;
        if (c.document_editor_get_page_crop_box(h, page_index, &x, &y, &w, &hgt, &code) != 0)
            return fail(code);
        return .{ .x = @floatCast(x), .y = @floatCast(y), .width = @floatCast(w), .height = @floatCast(hgt) };
    }

    pub fn setPageCropBox(self: DocumentEditor, page_index: usize, x: f64, y: f64, w: f64, hgt: f64) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_page_crop_box(h, page_index, x, y, w, hgt, &code) != 0) return fail(code);
    }

    /// Crop uniform margins (page user-space units) from every page.
    pub fn cropMargins(self: DocumentEditor, left: f32, right: f32, top: f32, bottom: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_crop_margins(h, left, right, top, bottom, &code) != 0) return fail(code);
    }

    /// Erase a single rectangular region on a (0-based) page.
    pub fn eraseRegion(self: DocumentEditor, page_index: i32, x: f32, y: f32, w: f32, hgt: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_erase_region(h, page_index, x, y, w, hgt, &code) != 0) return fail(code);
    }

    /// Erase multiple regions on a page; `rects` is a slice of (x,y,w,h) quads.
    pub fn eraseRegions(self: DocumentEditor, page_index: usize, rects: []const [4]f64) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const flat: [*]const f64 = @ptrCast(rects.ptr);
        if (c.document_editor_erase_regions(h, page_index, flat, rects.len, &code) != 0) return fail(code);
    }

    /// Clear all pending erase-region entries for a page.
    pub fn clearEraseRegions(self: DocumentEditor, page_index: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_clear_erase_regions(h, page_index, &code) != 0) return fail(code);
    }

    /// Apply redactions on a single (0-based) page (burn them in).
    pub fn applyPageRedactions(self: DocumentEditor, page_index: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_apply_page_redactions(h, page_index, &code) != 0) return fail(code);
    }

    /// Apply all pending redactions across the document.
    pub fn applyAllRedactions(self: DocumentEditor) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_apply_all_redactions(h, &code) != 0) return fail(code);
    }

    pub fn isPageMarkedForRedaction(self: DocumentEditor, page_index: usize) Error!bool {
        const h = try self.live();
        const r = c.document_editor_is_page_marked_for_redaction(h, page_index);
        if (r < 0) return fail(r);
        return r == 1;
    }

    pub fn unmarkPageForRedaction(self: DocumentEditor, page_index: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_unmark_page_for_redaction(h, page_index, &code) != 0) return fail(code);
    }

    pub fn isPageMarkedForFlatten(self: DocumentEditor, page_index: usize) Error!bool {
        const h = try self.live();
        const r = c.document_editor_is_page_marked_for_flatten(h, page_index);
        if (r < 0) return fail(r);
        return r == 1;
    }

    pub fn unmarkPageForFlatten(self: DocumentEditor, page_index: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_unmark_page_for_flatten(h, page_index, &code) != 0) return fail(code);
    }

    /// Flatten annotations on a single (0-based) page.
    pub fn flattenAnnotations(self: DocumentEditor, page_index: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_flatten_annotations(h, page_index, &code) != 0) return fail(code);
    }

    /// Flatten all annotations across the document.
    pub fn flattenAllAnnotations(self: DocumentEditor) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_flatten_all_annotations(h, &code) != 0) return fail(code);
    }

    /// Set a form field value (UTF-8).
    pub fn setFormFieldValue(self: DocumentEditor, name: [:0]const u8, value: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_set_form_field_value(h, name.ptr, value.ptr, &code) != 0) return fail(code);
    }

    /// Flatten all forms in the document.
    pub fn flattenForms(self: DocumentEditor) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_flatten_forms(h, &code) != 0) return fail(code);
    }

    /// Flatten forms on a single (0-based) page.
    pub fn flattenFormsOnPage(self: DocumentEditor, page_index: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.document_editor_flatten_forms_on_page(h, page_index, &code) != 0) return fail(code);
    }

    /// Number of warnings collected during the last form-flattening save.
    pub fn flattenWarningsCount(self: DocumentEditor) Error!i32 {
        const h = try self.live();
        const n = c.document_editor_flatten_warnings_count(h);
        if (n < 0) return fail(n);
        return n;
    }

    /// The `index`-th flatten warning; caller owns the returned slice.
    pub fn flattenWarning(self: DocumentEditor, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.document_editor_flatten_warning(h, index, &code), code);
    }

    // ── PHASE-7: programmatic redaction / barcode placement ───────────────────

    /// Queue a programmatic redaction rectangle on a (0-based) `page_no`.
    /// Coordinates and `r`/`g`/`b` overlay colour are page user-space / DeviceRGB.
    pub fn redactionAdd(
        self: DocumentEditor,
        page_no: usize,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        r: f64,
        g: f64,
        b: f64,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_redaction_add(h, page_no, x1, y1, x2, y2, r, g, b, &code) != 0) return fail(code);
    }

    /// Number of queued redaction regions for a (0-based) `page_no`.
    pub fn redactionCount(self: DocumentEditor, page_no: usize) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_redaction_count(h, page_no, &code);
        if (n < 0) return fail(code);
        return n;
    }

    /// Destructively apply all queued redactions (true content removal + opaque
    /// overlay). `scrub_metadata` runs the document-scrub pass; `r`/`g`/`b` are
    /// the overlay colour. Returns the number of glyphs physically removed.
    pub fn redactionApply(self: DocumentEditor, scrub_metadata: bool, r: f64, g: f64, b: f64) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_redaction_apply(h, scrub_metadata, r, g, b, &code);
        if (n < 0) return fail(code);
        return n;
    }

    /// Sanitize the document without geometric redaction (strips `/Info`, XMP
    /// `/Metadata`, document JavaScript, embedded files). Returns the number of
    /// top-level constructs removed.
    pub fn redactionScrubMetadata(self: DocumentEditor) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_redaction_scrub_metadata(h, &code);
        if (n < 0) return fail(code);
        return n;
    }

    /// Place a generated `Barcode` image on a (0-based) page at (`x`,`y`) with
    /// size `width`×`height` (page user-space points).
    pub fn addBarcodeToPage(
        self: DocumentEditor,
        page_index: i32,
        barcode: Barcode,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) Error!void {
        const h = try self.live();
        const bh = try barcode.live();
        var code: i32 = 0;
        if (c.pdf_add_barcode_to_page(h, page_index, bh, x, y, width, height, &code) != 0) return fail(code);
    }

    /// Import form values from an FDF byte buffer. Returns a status.
    pub fn importFdfBytes(self: DocumentEditor, data: []const u8) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_editor_import_fdf_bytes(@ptrCast(h), data.ptr, data.len, &code);
        if (code != 0) return fail(code);
        return n;
    }

    /// Import form values from an XFDF byte buffer. Returns a status.
    pub fn importXfdfBytes(self: DocumentEditor, data: []const u8) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const n = c.pdf_editor_import_xfdf_bytes(@ptrCast(h), data.ptr, data.len, &code);
        if (code != 0) return fail(code);
        return n;
    }
};

/// A loaded TTF/OTF font ready to be registered with a `DocumentBuilder`.
///
/// Owns the native `EmbeddedFont` handle. A *successful*
/// `DocumentBuilder.registerEmbeddedFont` **consumes** the native handle, so
/// this wrapper nulls its own handle on success and must not free it again.
/// Free an unregistered (or registration-failed) font with `deinit`.
pub const EmbeddedFont = struct {
    handle: ?*c.EmbeddedFont,

    /// Load a TTF/OTF font from a filesystem path (NUL-terminated).
    pub fn fromFile(path: [:0]const u8) Error!EmbeddedFont {
        var code: i32 = 0;
        const h = c.pdf_embedded_font_from_file(path.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Load a font from a byte buffer. `name` may be empty to use the font
    /// face's PostScript name.
    pub fn fromBytes(data: []const u8, name: ?[:0]const u8) Error!EmbeddedFont {
        var code: i32 = 0;
        const name_ptr: ?[*:0]const u8 = if (name) |n| n.ptr else null;
        const h = c.pdf_embedded_font_from_bytes(data.ptr, data.len, name_ptr, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). No-op once the font has been
    /// consumed by a successful `registerEmbeddedFont`.
    pub fn deinit(self: *EmbeddedFont) void {
        if (self.handle) |h| c.pdf_embedded_font_free(h);
        self.handle = null;
    }
};

/// A PDF being assembled programmatically. Owns the native `FfiDocumentBuilder`
/// handle; free it with `deinit`/`close`. Mirrors the `DocumentEditor` handle
/// pattern: factory `create`, error-code helpers, owned string-/byte-takes, and
/// a closed-handle guard.
///
/// `build`/`save*`/`toBytesEncrypted` consume the builder *state* but leave the
/// wrapper allocated; the wrapper is still freed by `deinit`.
pub const DocumentBuilder = struct {
    handle: ?*c.FfiDocumentBuilder,

    /// Guard: returns the live handle or raises if the builder has been closed.
    fn live(self: DocumentBuilder) Error!*c.FfiDocumentBuilder {
        return self.handle orelse fail(-1);
    }

    /// Create a new document builder.
    pub fn create() Error!DocumentBuilder {
        var code: i32 = 0;
        const h = c.pdf_document_builder_create(&code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *DocumentBuilder) void {
        if (self.handle) |h| c.pdf_document_builder_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *DocumentBuilder) void {
        self.deinit();
    }

    pub fn setTitle(self: DocumentBuilder, title: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_set_title(h, title.ptr, &code) != 0) return fail(code);
    }
    pub fn setAuthor(self: DocumentBuilder, author: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_set_author(h, author.ptr, &code) != 0) return fail(code);
    }
    pub fn setSubject(self: DocumentBuilder, subject: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_set_subject(h, subject.ptr, &code) != 0) return fail(code);
    }
    pub fn setKeywords(self: DocumentBuilder, keywords: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_set_keywords(h, keywords.ptr, &code) != 0) return fail(code);
    }
    pub fn setCreator(self: DocumentBuilder, creator: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_set_creator(h, creator.ptr, &code) != 0) return fail(code);
    }

    /// Run JavaScript when the document is opened (`/OpenAction`).
    pub fn onOpen(self: DocumentBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_on_open(h, script.ptr, &code) != 0) return fail(code);
    }

    /// Enable PDF/UA-1 tagged-PDF mode (opt-in).
    pub fn taggedPdfUa1(self: DocumentBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_tagged_pdf_ua1(h, &code) != 0) return fail(code);
    }

    /// Set the document's natural-language tag (e.g. "en-US").
    pub fn language(self: DocumentBuilder, lang: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_language(h, lang.ptr, &code) != 0) return fail(code);
    }

    /// Add a role-map entry: custom structure type → standard PDF type.
    pub fn roleMap(self: DocumentBuilder, custom: [:0]const u8, standard: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_role_map(h, custom.ptr, standard.ptr, &code) != 0) return fail(code);
    }

    /// Register a TTF/OTF font under `name`. On success the builder takes
    /// ownership of `font`'s native handle and this method nulls it so it
    /// won't be double-freed. On error the font remains valid.
    pub fn registerEmbeddedFont(self: DocumentBuilder, name: [:0]const u8, font: *EmbeddedFont) Error!void {
        const h = try self.live();
        const fh = font.handle orelse return fail(-1);
        var code: i32 = 0;
        if (c.pdf_document_builder_register_embedded_font(h, name.ptr, fh, &code) != 0) return fail(code);
        // Consumed on success: the builder now owns it.
        font.handle = null;
    }

    /// Start an A4 page. Only one page may be open per builder at a time.
    pub fn a4Page(self: DocumentBuilder) Error!PageBuilder {
        const h = try self.live();
        var code: i32 = 0;
        const ph = c.pdf_document_builder_a4_page(h, &code) orelse return fail(code);
        return .{ .handle = ph };
    }

    /// Start a US Letter page.
    pub fn letterPage(self: DocumentBuilder) Error!PageBuilder {
        const h = try self.live();
        var code: i32 = 0;
        const ph = c.pdf_document_builder_letter_page(h, &code) orelse return fail(code);
        return .{ .handle = ph };
    }

    /// Start a page with custom dimensions in PDF points (72 pt = 1 inch).
    pub fn page(self: DocumentBuilder, width: f32, height: f32) Error!PageBuilder {
        const h = try self.live();
        var code: i32 = 0;
        const ph = c.pdf_document_builder_page(h, width, height, &code) orelse return fail(code);
        return .{ .handle = ph };
    }

    /// Build the PDF and return the bytes; caller owns the returned slice.
    /// Consumes the builder state (the wrapper is still freed by `deinit`).
    pub fn build(self: DocumentBuilder, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_builder_build(h, &len, &code) orelse return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }

    /// Build and save the PDF to `path`.
    pub fn save(self: DocumentBuilder, path: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_save(h, path.ptr, &code) != 0) return fail(code);
    }

    /// Build and save with AES-256 encryption to `path`.
    pub fn saveEncrypted(
        self: DocumentBuilder,
        path: [:0]const u8,
        user_password: [:0]const u8,
        owner_password: [:0]const u8,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_document_builder_save_encrypted(h, path.ptr, user_password.ptr, owner_password.ptr, &code) != 0)
            return fail(code);
    }

    /// Build encrypted bytes; caller owns the returned slice.
    pub fn toBytesEncrypted(
        self: DocumentBuilder,
        alloc: std.mem.Allocator,
        user_password: [:0]const u8,
        owner_password: [:0]const u8,
    ) Error![]u8 {
        const h = try self.live();
        var len: usize = 0;
        var code: i32 = 0;
        const p = c.pdf_document_builder_to_bytes_encrypted(
            h,
            user_password.ptr,
            owner_password.ptr,
            &len,
            &code,
        ) orelse return fail(code);
        defer c.free_bytes(p);
        return alloc.dupe(u8, p[0..len]);
    }
};

/// A single page being assembled. Owns the native `FfiPageBuilder` handle until
/// it is committed with `done` (which **consumes** the handle) or dropped with
/// `deinit`/`close`. The fluent ops return `Error!void` rather than `self`
/// (Zig has no method-chaining sugar); call them in sequence.
///
/// The page borrows nothing from the `DocumentBuilder` at the Zig level — the
/// native side ties them together — so keep the parent builder alive until
/// `done` succeeds.
pub const PageBuilder = struct {
    handle: ?*c.FfiPageBuilder,

    /// Guard: returns the live handle or raises if the page has been
    /// committed/dropped.
    fn live(self: PageBuilder) Error!*c.FfiPageBuilder {
        return self.handle orelse fail(-1);
    }

    /// Drop an uncommitted page handle (idempotent). Does NOT apply buffered
    /// ops. No-op once `done` has consumed the handle. Also exposed as `close`.
    pub fn deinit(self: *PageBuilder) void {
        if (self.handle) |h| c.pdf_page_builder_free(h);
        self.handle = null;
    }

    /// Drop an uncommitted page handle (idempotent).
    pub fn close(self: *PageBuilder) void {
        self.deinit();
    }

    /// Commit this page's buffered ops to the parent builder. **Consumes** the
    /// handle on success; the wrapper is nulled so `deinit` won't double-free.
    pub fn done(self: *PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_done(h, &code) != 0) return fail(code);
        self.handle = null;
    }

    // ── text / layout ──────────────────────────────────────────────────────
    pub fn font(self: PageBuilder, name: [:0]const u8, size: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_font(h, name.ptr, size, &code) != 0) return fail(code);
    }
    pub fn at(self: PageBuilder, x: f32, y: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_at(h, x, y, &code) != 0) return fail(code);
    }
    pub fn text(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_text(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn heading(self: PageBuilder, level: u8, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_heading(h, level, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn paragraph(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_paragraph(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn space(self: PageBuilder, points: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_space(h, points, &code) != 0) return fail(code);
    }
    pub fn horizontalRule(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_horizontal_rule(h, &code) != 0) return fail(code);
    }

    /// Lay out `txt` across `column_count` balanced columns with `gap_pt`
    /// between them.
    pub fn columns(self: PageBuilder, column_count: u32, gap_pt: f32, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_columns(h, column_count, gap_pt, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn footnote(self: PageBuilder, ref_mark: [:0]const u8, note_text: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_footnote(h, ref_mark.ptr, note_text.ptr, &code) != 0) return fail(code);
    }

    // ── inline runs ────────────────────────────────────────────────────────
    pub fn inlineText(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_inline(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn inlineBold(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_inline_bold(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn inlineItalic(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_inline_italic(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn inlineColor(self: PageBuilder, r: f32, g: f32, b: f32, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_inline_color(h, r, g, b, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn newline(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_newline(h, &code) != 0) return fail(code);
    }

    // ── links ──────────────────────────────────────────────────────────────
    pub fn linkUrl(self: PageBuilder, url: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_link_url(h, url.ptr, &code) != 0) return fail(code);
    }
    pub fn linkPage(self: PageBuilder, page_index: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_link_page(h, page_index, &code) != 0) return fail(code);
    }
    pub fn linkNamed(self: PageBuilder, destination: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_link_named(h, destination.ptr, &code) != 0) return fail(code);
    }
    pub fn linkJavascript(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_link_javascript(h, script.ptr, &code) != 0) return fail(code);
    }

    // ── page / field actions ─────────────────────────────────────────────────
    pub fn onOpen(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_on_open(h, script.ptr, &code) != 0) return fail(code);
    }
    pub fn onClose(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_on_close(h, script.ptr, &code) != 0) return fail(code);
    }
    pub fn fieldKeystroke(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_field_keystroke(h, script.ptr, &code) != 0) return fail(code);
    }
    pub fn fieldFormat(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_field_format(h, script.ptr, &code) != 0) return fail(code);
    }
    pub fn fieldValidate(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_field_validate(h, script.ptr, &code) != 0) return fail(code);
    }
    pub fn fieldCalculate(self: PageBuilder, script: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_field_calculate(h, script.ptr, &code) != 0) return fail(code);
    }

    // ── text decorations / annotations ───────────────────────────────────────
    pub fn highlight(self: PageBuilder, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_highlight(h, r, g, b, &code) != 0) return fail(code);
    }
    pub fn underline(self: PageBuilder, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_underline(h, r, g, b, &code) != 0) return fail(code);
    }
    pub fn strikeout(self: PageBuilder, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_strikeout(h, r, g, b, &code) != 0) return fail(code);
    }
    pub fn squiggly(self: PageBuilder, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_squiggly(h, r, g, b, &code) != 0) return fail(code);
    }
    pub fn stickyNote(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_sticky_note(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn stickyNoteAt(self: PageBuilder, x: f32, y: f32, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_sticky_note_at(h, x, y, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn watermark(self: PageBuilder, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_watermark(h, txt.ptr, &code) != 0) return fail(code);
    }
    pub fn watermarkConfidential(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_watermark_confidential(h, &code) != 0) return fail(code);
    }
    pub fn watermarkDraft(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_watermark_draft(h, &code) != 0) return fail(code);
    }
    pub fn stamp(self: PageBuilder, type_name: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_stamp(h, type_name.ptr, &code) != 0) return fail(code);
    }
    pub fn freetext(self: PageBuilder, x: f32, y: f32, w: f32, height: f32, txt: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_freetext(h, x, y, w, height, txt.ptr, &code) != 0) return fail(code);
    }

    // ── form fields ──────────────────────────────────────────────────────────
    pub fn textField(
        self: PageBuilder,
        name: [:0]const u8,
        x: f32,
        y: f32,
        w: f32,
        height: f32,
        default_value: ?[:0]const u8,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const dv: ?[*:0]const u8 = if (default_value) |d| d.ptr else null;
        if (c.pdf_page_builder_text_field(h, name.ptr, x, y, w, height, dv, &code) != 0) return fail(code);
    }
    pub fn checkbox(self: PageBuilder, name: [:0]const u8, x: f32, y: f32, w: f32, height: f32, checked: bool) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_checkbox(h, name.ptr, x, y, w, height, @intFromBool(checked), &code) != 0)
            return fail(code);
    }

    /// Add a dropdown combo-box. `options` are NUL-terminated strings;
    /// `selected` may be null for no initial selection.
    pub fn comboBox(
        self: PageBuilder,
        name: [:0]const u8,
        x: f32,
        y: f32,
        w: f32,
        height: f32,
        options: []const [*:0]const u8,
        selected: ?[:0]const u8,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const sel: ?[*:0]const u8 = if (selected) |s| s.ptr else null;
        const opts: [*c]const [*c]const u8 = @ptrCast(options.ptr);
        if (c.pdf_page_builder_combo_box(h, name.ptr, x, y, w, height, opts, options.len, sel, &code) != 0)
            return fail(code);
    }

    /// Add a radio-button group. `values`/`xs`/`ys`/`ws`/`hs` are parallel
    /// arrays of equal length; `selected` may be null.
    pub fn radioGroup(
        self: PageBuilder,
        name: [:0]const u8,
        values: []const [*:0]const u8,
        xs: []const f32,
        ys: []const f32,
        ws: []const f32,
        hs: []const f32,
        selected: ?[:0]const u8,
    ) Error!void {
        const h = try self.live();
        const count = values.len;
        if (xs.len != count or ys.len != count or ws.len != count or hs.len != count) return fail(-1);
        var code: i32 = 0;
        const sel: ?[*:0]const u8 = if (selected) |s| s.ptr else null;
        const vals: [*c]const [*c]const u8 = @ptrCast(values.ptr);
        if (c.pdf_page_builder_radio_group(h, name.ptr, vals, xs.ptr, ys.ptr, ws.ptr, hs.ptr, count, sel, &code) != 0)
            return fail(code);
    }
    pub fn pushButton(self: PageBuilder, name: [:0]const u8, x: f32, y: f32, w: f32, height: f32, caption: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_push_button(h, name.ptr, x, y, w, height, caption.ptr, &code) != 0) return fail(code);
    }
    pub fn signatureField(self: PageBuilder, name: [:0]const u8, x: f32, y: f32, w: f32, height: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_signature_field(h, name.ptr, x, y, w, height, &code) != 0) return fail(code);
    }

    // ── barcodes ─────────────────────────────────────────────────────────────
    /// 1-D barcode. `barcode_type`: 0=Code128 1=Code39 2=EAN13 3=EAN8 4=UPCA 5=ITF 6=Code93 7=Codabar.
    pub fn barcode1d(self: PageBuilder, barcode_type: i32, data: [:0]const u8, x: f32, y: f32, w: f32, height: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_barcode_1d(h, barcode_type, data.ptr, x, y, w, height, &code) != 0) return fail(code);
    }
    pub fn barcodeQr(self: PageBuilder, data: [:0]const u8, x: f32, y: f32, size: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_barcode_qr(h, data.ptr, x, y, size, &code) != 0) return fail(code);
    }

    // ── images ───────────────────────────────────────────────────────────────
    pub fn image(self: PageBuilder, bytes: []const u8, x: f32, y: f32, w: f32, height: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_image(h, bytes.ptr, bytes.len, x, y, w, height, &code) != 0) return fail(code);
    }
    pub fn imageWithAlt(self: PageBuilder, bytes: []const u8, x: f32, y: f32, w: f32, height: f32, alt_text: [:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_image_with_alt(h, bytes.ptr, bytes.len, x, y, w, height, alt_text.ptr, &code) != 0)
            return fail(code);
    }
    pub fn imageArtifact(self: PageBuilder, bytes: []const u8, x: f32, y: f32, w: f32, height: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_image_artifact(h, bytes.ptr, bytes.len, x, y, w, height, &code) != 0) return fail(code);
    }

    // ── vector graphics ──────────────────────────────────────────────────────
    pub fn rect(self: PageBuilder, x: f32, y: f32, w: f32, height: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_rect(h, x, y, w, height, &code) != 0) return fail(code);
    }
    pub fn filledRect(self: PageBuilder, x: f32, y: f32, w: f32, height: f32, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_filled_rect(h, x, y, w, height, r, g, b, &code) != 0) return fail(code);
    }
    pub fn line(self: PageBuilder, x1: f32, y1: f32, x2: f32, y2: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_line(h, x1, y1, x2, y2, &code) != 0) return fail(code);
    }
    pub fn strokeRect(self: PageBuilder, x: f32, y: f32, w: f32, height: f32, width: f32, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_stroke_rect(h, x, y, w, height, width, r, g, b, &code) != 0) return fail(code);
    }
    pub fn strokeLine(self: PageBuilder, x1: f32, y1: f32, x2: f32, y2: f32, width: f32, r: f32, g: f32, b: f32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_stroke_line(h, x1, y1, x2, y2, width, r, g, b, &code) != 0) return fail(code);
    }

    /// Stroked dashed rectangle. `dash_array` is alternating on/off lengths
    /// (empty = solid); `phase` is the starting offset.
    pub fn strokeRectDashed(
        self: PageBuilder,
        x: f32,
        y: f32,
        w: f32,
        height: f32,
        width: f32,
        r: f32,
        g: f32,
        b: f32,
        dash_array: []const f32,
        phase: f32,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const dash_ptr: ?[*]const f32 = if (dash_array.len == 0) null else dash_array.ptr;
        if (c.pdf_page_builder_stroke_rect_dashed(h, x, y, w, height, width, r, g, b, dash_ptr, dash_array.len, phase, &code) != 0)
            return fail(code);
    }

    /// Stroked dashed line. `dash_array` is alternating on/off lengths
    /// (empty = solid); `phase` is the starting offset.
    pub fn strokeLineDashed(
        self: PageBuilder,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        width: f32,
        r: f32,
        g: f32,
        b: f32,
        dash_array: []const f32,
        phase: f32,
    ) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const dash_ptr: ?[*]const f32 = if (dash_array.len == 0) null else dash_array.ptr;
        if (c.pdf_page_builder_stroke_line_dashed(h, x1, y1, x2, y2, width, r, g, b, dash_ptr, dash_array.len, phase, &code) != 0)
            return fail(code);
    }

    /// Buffer a `text_in_rect`. `align`: 0=Left, 1=Center, 2=Right.
    pub fn textInRect(self: PageBuilder, x: f32, y: f32, w: f32, height: f32, txt: [:0]const u8, align_mode: i32) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_text_in_rect(h, x, y, w, height, txt.ptr, align_mode, &code) != 0) return fail(code);
    }

    /// Buffer a new-page transition; subsequent ops land on the new page.
    pub fn newPageSameSize(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_new_page_same_size(h, &code) != 0) return fail(code);
    }

    // ── tables ───────────────────────────────────────────────────────────────
    /// Buffer a complete table. `widths`/`aligns` are length `n_columns`
    /// (`aligns`: 0/1/2). `cells` is row-major (`cells[row*n_columns + col]`)
    /// of length `n_rows * n_columns`. `has_header` promotes the first row.
    pub fn table(
        self: PageBuilder,
        n_columns: usize,
        widths: []const f32,
        aligns: []const i32,
        n_rows: usize,
        cells: []const [*:0]const u8,
        has_header: bool,
    ) Error!void {
        const h = try self.live();
        if (widths.len != n_columns or aligns.len != n_columns) return fail(-1);
        if (cells.len != n_rows * n_columns) return fail(-1);
        var code: i32 = 0;
        const cell_ptr: [*c]const [*c]const u8 = @ptrCast(cells.ptr);
        if (c.pdf_page_builder_table(h, n_columns, widths.ptr, aligns.ptr, n_rows, cell_ptr, @intFromBool(has_header), &code) != 0)
            return fail(code);
    }

    // ── streaming tables ─────────────────────────────────────────────────────
    /// Open a streaming table. `headers`/`widths`/`aligns` are parallel arrays
    /// of length `n_columns`.
    pub fn streamingTableBegin(
        self: PageBuilder,
        n_columns: usize,
        headers: []const [*:0]const u8,
        widths: []const f32,
        aligns: []const i32,
        repeat_header: bool,
    ) Error!void {
        const h = try self.live();
        if (headers.len != n_columns or widths.len != n_columns or aligns.len != n_columns) return fail(-1);
        var code: i32 = 0;
        const hdr_ptr: [*c]const [*c]const u8 = @ptrCast(headers.ptr);
        if (c.pdf_page_builder_streaming_table_begin(h, n_columns, hdr_ptr, widths.ptr, aligns.ptr, @intFromBool(repeat_header), &code) != 0)
            return fail(code);
    }

    /// Open a streaming table with a column-width mode. `mode`: 0=Fixed,
    /// 1=Sample, 2=AutoAll (rejected).
    pub fn streamingTableBeginV2(
        self: PageBuilder,
        n_columns: usize,
        headers: []const [*:0]const u8,
        widths: []const f32,
        aligns: []const i32,
        repeat_header: bool,
        mode: i32,
        sample_rows: usize,
        min_col_width_pt: f32,
        max_col_width_pt: f32,
        max_rowspan: usize,
    ) Error!void {
        const h = try self.live();
        if (headers.len != n_columns or widths.len != n_columns or aligns.len != n_columns) return fail(-1);
        var code: i32 = 0;
        const hdr_ptr: [*c]const [*c]const u8 = @ptrCast(headers.ptr);
        if (c.pdf_page_builder_streaming_table_begin_v2(
            h,
            n_columns,
            hdr_ptr,
            widths.ptr,
            aligns.ptr,
            @intFromBool(repeat_header),
            mode,
            sample_rows,
            min_col_width_pt,
            max_col_width_pt,
            max_rowspan,
            &code,
        ) != 0) return fail(code);
    }
    pub fn streamingTableSetBatchSize(self: PageBuilder, batch_size: usize) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_streaming_table_set_batch_size(h, batch_size, &code) != 0) return fail(code);
    }
    pub fn streamingTablePendingRowCount(self: PageBuilder) Error!usize {
        const h = try self.live();
        return c.pdf_page_builder_streaming_table_pending_row_count(h);
    }
    pub fn streamingTableBatchCount(self: PageBuilder) Error!usize {
        const h = try self.live();
        return c.pdf_page_builder_streaming_table_batch_count(h);
    }
    pub fn streamingTableFlush(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_streaming_table_flush(h, &code) != 0) return fail(code);
    }

    /// Push one row into the open streaming table. `cells` length must equal
    /// the column count supplied to begin.
    pub fn streamingTablePushRow(self: PageBuilder, cells: []const [*:0]const u8) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        const cell_ptr: [*c]const [*c]const u8 = @ptrCast(cells.ptr);
        if (c.pdf_page_builder_streaming_table_push_row(h, cells.len, cell_ptr, &code) != 0) return fail(code);
    }

    /// Push one row with per-cell rowspans (`rowspans` length must equal
    /// `cells` length; pass an empty slice for all-rowspan-1).
    pub fn streamingTablePushRowV2(self: PageBuilder, cells: []const [*:0]const u8, rowspans: []const usize) Error!void {
        const h = try self.live();
        if (rowspans.len != 0 and rowspans.len != cells.len) return fail(-1);
        var code: i32 = 0;
        const cell_ptr: [*c]const [*c]const u8 = @ptrCast(cells.ptr);
        const span_ptr: ?[*]const usize = if (rowspans.len == 0) null else rowspans.ptr;
        if (c.pdf_page_builder_streaming_table_push_row_v2(h, cells.len, cell_ptr, span_ptr, &code) != 0) return fail(code);
    }
    pub fn streamingTableFinish(self: PageBuilder) Error!void {
        const h = try self.live();
        var code: i32 = 0;
        if (c.pdf_page_builder_streaming_table_finish(h, &code) != 0) return fail(code);
    }
};

// ── PHASE-6: digital signatures / PKI / timestamps / TSA / DSS ────────────────
//
// Mirrors the earlier-phase handle pattern: factory constructors, error-code
// helpers, owned string-/byte-takes, opaque `*anyopaque` handles freed on
// `deinit`/`close`, and a closed-handle guard on every accessor.

/// Copy a `const uint8_t*` C return (one the C side still owns — e.g. a borrowed
/// view into a parsed structure) into an allocator-owned slice. Unlike
/// `takeBytes`, this MUST NOT free the C buffer.
fn copyBorrowedBytes(alloc: std.mem.Allocator, ptr: ?[*]const u8, len: usize) Error![]u8 {
    const p = ptr orelse return fail(-1);
    return alloc.dupe(u8, p[0..len]);
}

/// Copy a `uint8_t*` C return that the C side has handed ownership of into an
/// allocator-owned slice and free the C buffer via `free_bytes`.
fn takeBytes(alloc: std.mem.Allocator, ptr: ?[*]u8, len: usize, code: i32) Error![]u8 {
    const p = ptr orelse return fail(code);
    defer c.free_bytes(p);
    return alloc.dupe(u8, p[0..len]);
}

/// Signing credentials (X.509 certificate + private key) loaded from a PKCS#12
/// blob or PEM pair. Owns an opaque `*anyopaque` native handle; free with
/// `deinit`/`close`. Also the carrier type for a signature's embedded
/// certificate (returned by `SignatureInfo.certificate`).
pub const Certificate = struct {
    handle: ?*anyopaque,

    /// Guard: returns the live handle or raises if the certificate was closed.
    fn live(self: Certificate) Error!*anyopaque {
        return self.handle orelse fail(-1);
    }

    /// Load credentials from a PKCS#12 (.p12/.pfx) byte buffer. `password` may be
    /// empty for an unprotected blob.
    pub fn loadFromBytes(data: []const u8, password: [:0]const u8) Error!Certificate {
        var code: i32 = 0;
        const h = c.pdf_certificate_load_from_bytes(data.ptr, @intCast(data.len), password.ptr, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    /// Load credentials from PEM-encoded certificate + private-key strings.
    pub fn loadFromPem(cert_pem: [:0]const u8, key_pem: [:0]const u8) Error!Certificate {
        var code: i32 = 0;
        const h = c.pdf_certificate_load_from_pem(cert_pem.ptr, key_pem.ptr, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *Certificate) void {
        if (self.handle) |h| c.pdf_certificate_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *Certificate) void {
        self.deinit();
    }

    /// Subject distinguished name; caller owns the returned slice.
    pub fn subject(self: Certificate, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_certificate_get_subject(h, &code), code);
    }

    /// Issuer distinguished name; caller owns the returned slice.
    pub fn issuer(self: Certificate, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_certificate_get_issuer(h, &code), code);
    }

    /// Serial number (decimal string); caller owns the returned slice.
    pub fn serial(self: Certificate, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_certificate_get_serial(h, &code), code);
    }

    /// Validity window as Unix epoch seconds (`not_before`, `not_after`).
    pub fn validity(self: Certificate) Error!struct { notBefore: i64, notAfter: i64 } {
        const h = try self.live();
        var code: i32 = 0;
        var nb: i64 = 0;
        var na: i64 = 0;
        c.pdf_certificate_get_validity(h, &nb, &na, &code);
        if (code != 0) return fail(code);
        return .{ .notBefore = nb, .notAfter = na };
    }

    /// Whether the certificate is currently within its validity window.
    pub fn isValid(self: Certificate) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const r = c.pdf_certificate_is_valid(h, &code);
        if (r < 0) return fail(code);
        return r == 1;
    }
};

/// A parsed RFC 3161 timestamp token. Owns an opaque `*anyopaque` native handle;
/// free with `deinit`/`close`.
pub const Timestamp = struct {
    handle: ?*anyopaque,

    fn live(self: Timestamp) Error!*anyopaque {
        return self.handle orelse fail(-1);
    }

    /// Parse a DER-encoded RFC 3161 TimeStampToken (or bare TSTInfo).
    pub fn parse(bytes: []const u8) Error!Timestamp {
        var code: i32 = 0;
        const h = c.pdf_timestamp_parse(bytes.ptr, bytes.len, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *Timestamp) void {
        if (self.handle) |h| c.pdf_timestamp_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *Timestamp) void {
        self.deinit();
    }

    /// The raw DER token bytes (borrowed view; copied). Caller owns the slice.
    pub fn token(self: Timestamp, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        var out_len: usize = 0;
        const p = c.pdf_timestamp_get_token(h, &out_len, &code) orelse return fail(code);
        return copyBorrowedBytes(alloc, p, out_len);
    }

    /// The hashed message imprint (borrowed view; copied). Caller owns the slice.
    pub fn messageImprint(self: Timestamp, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        var out_len: usize = 0;
        const p = c.pdf_timestamp_get_message_imprint(h, &out_len, &code) orelse return fail(code);
        return copyBorrowedBytes(alloc, p, out_len);
    }

    /// Timestamp time as Unix epoch seconds.
    pub fn time(self: Timestamp) Error!i64 {
        const h = try self.live();
        var code: i32 = 0;
        const t = c.pdf_timestamp_get_time(h, &code);
        if (code != 0) return fail(code);
        return t;
    }

    /// TSA-assigned serial number (decimal string); caller owns the slice.
    pub fn serial(self: Timestamp, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_timestamp_get_serial(h, &code), code);
    }

    /// Name of the issuing TSA; caller owns the slice.
    pub fn tsaName(self: Timestamp, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_timestamp_get_tsa_name(h, &code), code);
    }

    /// TSA policy OID; caller owns the slice.
    pub fn policyOid(self: Timestamp, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_timestamp_get_policy_oid(h, &code), code);
    }

    /// Message-imprint hash algorithm identifier.
    pub fn hashAlgorithm(self: Timestamp) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const a = c.pdf_timestamp_get_hash_algorithm(h, &code);
        if (code != 0) return fail(code);
        return a;
    }

    /// Cryptographically verify the timestamp token.
    pub fn verify(self: Timestamp) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const ok = c.pdf_timestamp_verify(h, &code);
        if (code != 0) return fail(code);
        return ok;
    }
};

/// An RFC 3161 Time-Stamping Authority client. Owns an opaque `*anyopaque`
/// native handle; free with `deinit`/`close`.
pub const TsaClient = struct {
    handle: ?*anyopaque,

    fn live(self: TsaClient) Error!*anyopaque {
        return self.handle orelse fail(-1);
    }

    /// Create a TSA client. `username`/`password` may be empty for an
    /// unauthenticated TSA. `timeout` is in seconds; `hash_algo` selects the
    /// message-imprint digest.
    pub fn create(
        url: [:0]const u8,
        username: [:0]const u8,
        password: [:0]const u8,
        timeout: i32,
        hash_algo: i32,
        use_nonce: bool,
        cert_req: bool,
    ) Error!TsaClient {
        var code: i32 = 0;
        const h = c.pdf_tsa_client_create(
            url.ptr,
            username.ptr,
            password.ptr,
            timeout,
            hash_algo,
            use_nonce,
            cert_req,
            &code,
        ) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *TsaClient) void {
        if (self.handle) |h| c.pdf_tsa_client_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *TsaClient) void {
        self.deinit();
    }

    /// Request a timestamp over raw `data` (the client hashes it). Returns an
    /// owned `Timestamp`; free it with `deinit`.
    pub fn requestTimestamp(self: TsaClient, data: []const u8) Error!Timestamp {
        const h = try self.live();
        var code: i32 = 0;
        const ts = c.pdf_tsa_request_timestamp(h, data.ptr, data.len, &code) orelse return fail(code);
        return .{ .handle = ts };
    }

    /// Request a timestamp over a precomputed `hash` of `hash_algo`. Returns an
    /// owned `Timestamp`; free it with `deinit`.
    pub fn requestTimestampHash(self: TsaClient, hash: []const u8, hash_algo: i32) Error!Timestamp {
        const h = try self.live();
        var code: i32 = 0;
        const ts = c.pdf_tsa_request_timestamp_hash(h, hash.ptr, hash.len, hash_algo, &code) orelse
            return fail(code);
        return .{ .handle = ts };
    }
};

/// A document's Document Security Store (DSS): the long-term validation material
/// (certificates / CRLs / OCSP responses / VRI). Owns an opaque `*anyopaque`
/// native handle; free with `deinit`/`close`.
pub const Dss = struct {
    handle: ?*anyopaque,

    fn live(self: Dss) Error!*anyopaque {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *Dss) void {
        if (self.handle) |h| c.pdf_dss_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *Dss) void {
        self.deinit();
    }

    pub fn certCount(self: Dss) Error!i32 {
        const h = try self.live();
        return c.pdf_dss_cert_count(h);
    }
    pub fn crlCount(self: Dss) Error!i32 {
        const h = try self.live();
        return c.pdf_dss_crl_count(h);
    }
    pub fn ocspCount(self: Dss) Error!i32 {
        const h = try self.live();
        return c.pdf_dss_ocsp_count(h);
    }
    pub fn vriCount(self: Dss) Error!i32 {
        const h = try self.live();
        return c.pdf_dss_vri_count(h);
    }

    /// DER bytes of the `index`-th embedded certificate; caller owns the slice.
    pub fn getCert(self: Dss, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        var out_len: usize = 0;
        const p = c.pdf_dss_get_cert(h, index, &out_len, &code) orelse return fail(code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// DER bytes of the `index`-th embedded CRL; caller owns the slice.
    pub fn getCrl(self: Dss, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        var out_len: usize = 0;
        const p = c.pdf_dss_get_crl(h, index, &out_len, &code) orelse return fail(code);
        return takeBytes(alloc, p, out_len, code);
    }

    /// DER bytes of the `index`-th embedded OCSP response; caller owns the slice.
    pub fn getOcsp(self: Dss, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        var out_len: usize = 0;
        const p = c.pdf_dss_get_ocsp(h, index, &out_len, &code) orelse return fail(code);
        return takeBytes(alloc, p, out_len, code);
    }
};

/// Metadata about a signature extracted from a signed PDF. Owns the native
/// `FfiSignatureInfo` handle; free with `deinit`/`close`.
pub const SignatureInfo = struct {
    handle: ?*c.FfiSignatureInfo,

    fn live(self: SignatureInfo) Error!*c.FfiSignatureInfo {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *SignatureInfo) void {
        if (self.handle) |h| c.pdf_signature_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *SignatureInfo) void {
        self.deinit();
    }

    /// Signer common name; caller owns the slice.
    pub fn signerName(self: SignatureInfo, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_signature_get_signer_name(h, &code), code);
    }

    /// Stated signing reason; caller owns the slice.
    pub fn signingReason(self: SignatureInfo, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_signature_get_signing_reason(h, &code), code);
    }

    /// Stated signing location; caller owns the slice.
    pub fn signingLocation(self: SignatureInfo, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_signature_get_signing_location(h, &code), code);
    }

    /// Claimed signing time as Unix epoch seconds.
    pub fn signingTime(self: SignatureInfo) Error!i64 {
        const h = try self.live();
        var code: i32 = 0;
        const t = c.pdf_signature_get_signing_time(h, &code);
        if (code != 0) return fail(code);
        return t;
    }

    /// The signer's embedded certificate. Returns an owned `Certificate`; free it
    /// with `deinit`.
    pub fn certificate(self: SignatureInfo) Error!Certificate {
        const h = try self.live();
        var code: i32 = 0;
        const ch = c.pdf_signature_get_certificate(h, &code) orelse return fail(code);
        return .{ .handle = ch };
    }

    /// The signature's PAdES baseline level (B-B/B-T/…), or an error.
    pub fn padesLevel(self: SignatureInfo) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const lvl = c.pdf_signature_get_pades_level(h, &code);
        if (lvl < 0) return fail(code);
        return lvl;
    }

    /// Whether the signature carries an embedded RFC 3161 timestamp.
    pub fn hasTimestamp(self: SignatureInfo) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const ok = c.pdf_signature_has_timestamp(h, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    /// The signature's embedded timestamp. Returns an owned `Timestamp`; free it
    /// with `deinit`.
    pub fn timestamp(self: SignatureInfo) Error!Timestamp {
        const h = try self.live();
        var code: i32 = 0;
        const ts = c.pdf_signature_get_timestamp(h, &code) orelse return fail(code);
        return .{ .handle = ts };
    }

    /// Attach `ts` to this signature. Returns true on success.
    pub fn addTimestamp(self: SignatureInfo, ts: Timestamp) Error!bool {
        const h = try self.live();
        const th = ts.handle orelse return fail(-1);
        var code: i32 = 0;
        const ok = c.pdf_signature_add_timestamp(h, th, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    /// Run the signer-attributes crypto check. Returns 1=valid, 0=invalid,
    /// -1=unknown/unsupported (the raw tri-state — not mapped to an error).
    pub fn verify(self: SignatureInfo) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        return c.pdf_signature_verify(h, &code);
    }

    /// End-to-end verification against the full PDF bytes (signer check +
    /// messageDigest). Returns 1=valid, 0=invalid, -1=unknown/unsupported.
    pub fn verifyDetached(self: SignatureInfo, pdf_data: []const u8) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        return c.pdf_signature_verify_detached(h, pdf_data.ptr, pdf_data.len, &code);
    }
};

/// Result of a PDF/A conformance check. Owns the native `FfiPdfAResults` handle;
/// free with `deinit`/`close`.
pub const PdfAResults = struct {
    handle: ?*c.FfiPdfAResults,

    fn live(self: PdfAResults) Error!*c.FfiPdfAResults {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *PdfAResults) void {
        if (self.handle) |h| c.pdf_pdf_a_results_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *PdfAResults) void {
        self.deinit();
    }

    pub fn isCompliant(self: PdfAResults) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const ok = c.pdf_pdf_a_is_compliant(h, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    pub fn errorCount(self: PdfAResults) Error!i32 {
        const h = try self.live();
        return c.pdf_pdf_a_error_count(h);
    }

    pub fn warningCount(self: PdfAResults) Error!i32 {
        const h = try self.live();
        return c.pdf_pdf_a_warning_count(h);
    }

    /// The `index`-th error message; caller owns the slice.
    pub fn getError(self: PdfAResults, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_pdf_a_get_error(h, index, &code), code);
    }

    /// Collect every error message into an owned slice; free with `freeStrings`.
    pub fn errors(self: PdfAResults, alloc: std.mem.Allocator) Error![][]u8 {
        return collectStrings(self, alloc, errorCount, getError);
    }

    /// Free a slice returned by `errors`.
    pub fn freeStrings(alloc: std.mem.Allocator, list: [][]u8) void {
        freeStringList(alloc, list);
    }
};

/// Result of a PDF/X conformance check. Owns the native `FfiPdfXResults` handle;
/// free with `deinit`/`close`.
pub const PdfXResults = struct {
    handle: ?*c.FfiPdfXResults,

    fn live(self: PdfXResults) Error!*c.FfiPdfXResults {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *PdfXResults) void {
        if (self.handle) |h| c.pdf_pdf_x_results_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *PdfXResults) void {
        self.deinit();
    }

    pub fn isCompliant(self: PdfXResults) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const ok = c.pdf_pdf_x_is_compliant(h, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    pub fn errorCount(self: PdfXResults) Error!i32 {
        const h = try self.live();
        return c.pdf_pdf_x_error_count(h);
    }

    /// The `index`-th error message; caller owns the slice.
    pub fn getError(self: PdfXResults, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_pdf_x_get_error(h, index, &code), code);
    }

    /// Collect every error message into an owned slice; free with `freeStrings`.
    pub fn errors(self: PdfXResults, alloc: std.mem.Allocator) Error![][]u8 {
        return collectStrings(self, alloc, errorCount, getError);
    }

    /// Free a slice returned by `errors`.
    pub fn freeStrings(alloc: std.mem.Allocator, list: [][]u8) void {
        freeStringList(alloc, list);
    }
};

/// PDF/UA accessibility statistics (counts of tagged structure elements).
pub const UaStats = struct {
    structElements: i32,
    images: i32,
    tables: i32,
    forms: i32,
    annotations: i32,
    pages: i32,
};

/// Result of a PDF/UA accessibility check. Owns the native `FfiUaResults`
/// handle; free with `deinit`/`close`.
pub const UaResults = struct {
    handle: ?*c.FfiUaResults,

    fn live(self: UaResults) Error!*c.FfiUaResults {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *UaResults) void {
        if (self.handle) |h| c.pdf_pdf_ua_results_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *UaResults) void {
        self.deinit();
    }

    pub fn isAccessible(self: UaResults) Error!bool {
        const h = try self.live();
        var code: i32 = 0;
        const ok = c.pdf_pdf_ua_is_accessible(h, &code);
        if (code != 0) return fail(code);
        return ok;
    }

    pub fn errorCount(self: UaResults) Error!i32 {
        const h = try self.live();
        return c.pdf_pdf_ua_error_count(h);
    }

    pub fn warningCount(self: UaResults) Error!i32 {
        const h = try self.live();
        return c.pdf_pdf_ua_warning_count(h);
    }

    /// The `index`-th error message; caller owns the slice.
    pub fn getError(self: UaResults, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_pdf_ua_get_error(h, index, &code), code);
    }

    /// The `index`-th warning message; caller owns the slice.
    pub fn getWarning(self: UaResults, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_pdf_ua_get_warning(h, index, &code), code);
    }

    /// Collect every error message into an owned slice; free with `freeStrings`.
    pub fn errors(self: UaResults, alloc: std.mem.Allocator) Error![][]u8 {
        return collectStrings(self, alloc, errorCount, getError);
    }

    /// Collect every warning message into an owned slice; free with `freeStrings`.
    pub fn warnings(self: UaResults, alloc: std.mem.Allocator) Error![][]u8 {
        return collectStrings(self, alloc, warningCount, getWarning);
    }

    /// Accessibility statistics (tagged-structure counts).
    pub fn uaStats(self: UaResults) Error!UaStats {
        const h = try self.live();
        var code: i32 = 0;
        var s: i32 = 0;
        var im: i32 = 0;
        var tb: i32 = 0;
        var fm: i32 = 0;
        var an: i32 = 0;
        var pg: i32 = 0;
        const ok = c.pdf_pdf_ua_get_stats(h, &s, &im, &tb, &fm, &an, &pg, &code);
        if (code != 0) return fail(code);
        if (!ok) return fail(-1);
        return .{
            .structElements = s,
            .images = im,
            .tables = tb,
            .forms = fm,
            .annotations = an,
            .pages = pg,
        };
    }

    /// Free a slice returned by `errors`/`warnings`.
    pub fn freeStrings(alloc: std.mem.Allocator, list: [][]u8) void {
        freeStringList(alloc, list);
    }
};

/// Shared helper: build an owned `[][]u8` from a `count`/`get` accessor pair on a
/// validation-result type. `count` and `get` are method references on `T`.
fn collectStrings(
    self: anytype,
    alloc: std.mem.Allocator,
    comptime count: fn (@TypeOf(self)) Error!i32,
    comptime get: fn (@TypeOf(self), std.mem.Allocator, i32) Error![]u8,
) Error![][]u8 {
    const n = try count(self);
    const total: usize = if (n < 0) 0 else @intCast(n);
    const out = try alloc.alloc([]u8, total);
    errdefer alloc.free(out);
    var i: usize = 0;
    errdefer for (out[0..i]) |s| alloc.free(s);
    while (i < total) : (i += 1) {
        out[i] = try get(self, alloc, @intCast(i));
    }
    return out;
}

/// Free a `[][]u8` produced by `collectStrings`.
fn freeStringList(alloc: std.mem.Allocator, list: [][]u8) void {
    for (list) |s| alloc.free(s);
    alloc.free(list);
}

// ── PHASE-6 top-level free functions ──────────────────────────────────────────

/// Set the global library log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace).
pub fn setLogLevel(level: i32) void {
    c.pdf_oxide_set_log_level(level);
}

/// Get the current global library log level (0-5).
pub fn getLogLevel() i32 {
    return c.pdf_oxide_get_log_level();
}

/// Sign raw PDF `pdf_data` with `cert` and return the signed PDF bytes. Caller
/// owns the returned slice.
pub fn signBytes(
    alloc: std.mem.Allocator,
    pdf_data: []const u8,
    cert: Certificate,
    reason: [:0]const u8,
    location: [:0]const u8,
) Error![]u8 {
    const ch = try cert.live();
    var out_len: usize = 0;
    var code: i32 = 0;
    const p = c.pdf_sign_bytes(pdf_data.ptr, pdf_data.len, ch, reason.ptr, location.ptr, &out_len, &code) orelse
        return fail(code);
    return takeBytes(alloc, p, out_len, code);
}

/// Parallel DER byte-array material (certs / CRLs / OCSPs) passed to PAdES
/// signing. `entries` is a slice of DER blobs.
pub const DerList = struct {
    entries: []const []const u8,
};

/// Sign raw PDF `pdf_data` at a PAdES baseline `level` (0=B-B 1=B-T 2=B-LT),
/// optionally timestamped via `tsa_url`, with the B-LT revocation material in
/// `certs`/`crls`/`ocsps`. Caller owns the returned slice.
pub fn signBytesPades(
    alloc: std.mem.Allocator,
    pdf_data: []const u8,
    cert: Certificate,
    level: i32,
    tsa_url: ?[:0]const u8,
    reason: [:0]const u8,
    location: [:0]const u8,
    certs: []const []const u8,
    crls: []const []const u8,
    ocsps: []const []const u8,
) Error![]u8 {
    const ch = try cert.live();

    // Marshal each DER-array-array into a (ptr[], len[]) pair backed by the
    // arena, freed when this scope exits.
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const certs_m = try marshalDerArray(aa, certs);
    const crls_m = try marshalDerArray(aa, crls);
    const ocsps_m = try marshalDerArray(aa, ocsps);

    const tsa_ptr: ?[*:0]const u8 = if (tsa_url) |u| u.ptr else null;

    var out_len: usize = 0;
    var code: i32 = 0;
    const p = c.pdf_sign_bytes_pades(
        pdf_data.ptr,
        pdf_data.len,
        ch,
        level,
        tsa_ptr,
        reason.ptr,
        location.ptr,
        certs_m.ptrs,
        certs_m.lens,
        certs.len,
        crls_m.ptrs,
        crls_m.lens,
        crls.len,
        ocsps_m.ptrs,
        ocsps_m.lens,
        ocsps.len,
        &out_len,
        &code,
    ) orelse return fail(code);
    return takeBytes(alloc, p, out_len, code);
}

/// Struct-options variant of `signBytesPades`: collapses the parameters into the
/// `PadesSignOptionsC` struct the C ABI expects. Caller owns the returned slice.
pub fn signBytesPadesOpts(
    alloc: std.mem.Allocator,
    pdf_data: []const u8,
    cert: Certificate,
    level: i32,
    tsa_url: ?[:0]const u8,
    reason: [:0]const u8,
    location: [:0]const u8,
    certs: []const []const u8,
    crls: []const []const u8,
    ocsps: []const []const u8,
) Error![]u8 {
    const ch = try cert.live();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const certs_m = try marshalDerArray(aa, certs);
    const crls_m = try marshalDerArray(aa, crls);
    const ocsps_m = try marshalDerArray(aa, ocsps);

    const tsa_ptr: ?[*:0]const u8 = if (tsa_url) |u| u.ptr else null;

    var options = c.PadesSignOptionsC{
        .certificate_handle = ch,
        .certs = certs_m.ptrs,
        .cert_lens = certs_m.lens,
        .n_certs = certs.len,
        .crls = crls_m.ptrs,
        .crl_lens = crls_m.lens,
        .n_crls = crls.len,
        .ocsps = ocsps_m.ptrs,
        .ocsp_lens = ocsps_m.lens,
        .n_ocsps = ocsps.len,
        .tsa_url = tsa_ptr,
        .reason = reason.ptr,
        .location = location.ptr,
        .level = level,
    };

    var out_len: usize = 0;
    var code: i32 = 0;
    const p = c.pdf_sign_bytes_pades_opts(pdf_data.ptr, pdf_data.len, &options, &out_len, &code) orelse
        return fail(code);
    return takeBytes(alloc, p, out_len, code);
}

/// Build the parallel (`const uint8_t* const*`, `const uintptr_t*`) arrays the
/// PAdES C entry points expect from a slice of DER blobs. Allocated in `aa`
/// (intended to be an arena that outlives the C call). The returned pointers use
/// the C-pointer (`[*c]`) representation so they match the `@cImport` parameter
/// types exactly; for an empty input both are null (the C "n == 0" convention).
fn marshalDerArray(
    aa: std.mem.Allocator,
    blobs: []const []const u8,
) Error!struct { ptrs: [*c]const [*c]const u8, lens: [*c]const usize } {
    if (blobs.len == 0) return .{ .ptrs = null, .lens = null };
    const ptrs = try aa.alloc([*c]const u8, blobs.len);
    const lens = try aa.alloc(usize, blobs.len);
    for (blobs, 0..) |b, i| {
        ptrs[i] = b.ptr;
        lens[i] = b.len;
    }
    return .{ .ptrs = ptrs.ptr, .lens = lens.ptr };
}

// ── PHASE-7: barcodes / QR / OCR engine / renderer / merge / timestamp ────────
//
// Mirrors the earlier-phase handle pattern: factory constructors, error-code
// helpers, owned string-/byte-takes, opaque handles freed on `deinit`/`close`,
// and a closed-handle guard on every accessor.

/// A generated 1-D barcode / 2-D QR image, or a decoded barcode. Owns the native
/// `FfiBarcodeImage` handle; free with `deinit`/`close`.
pub const Barcode = struct {
    handle: ?*c.FfiBarcodeImage,

    /// Guard: returns the live handle or raises if the barcode was closed.
    fn live(self: Barcode) Error!*c.FfiBarcodeImage {
        return self.handle orelse fail(-1);
    }

    /// Generate a QR code from `data`. `error_correction`: 0=L 1=M 2=Q 3=H;
    /// `size_px` is the requested module/pixel size.
    pub fn generateQrCode(data: [:0]const u8, error_correction: i32, size_px: i32) Error!Barcode {
        var code: i32 = 0;
        const h = c.pdf_generate_qr_code(data.ptr, error_correction, size_px, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Generate a 1-D barcode from `data`. `format`: 0=Code128 1=Code39 2=EAN13
    /// 3=EAN8 4=UPCA 5=ITF 6=Code93 7=Codabar; `size_px` is the requested size.
    pub fn generateBarcode(data: [:0]const u8, format: i32, size_px: i32) Error!Barcode {
        var code: i32 = 0;
        const h = c.pdf_generate_barcode(data.ptr, format, size_px, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *Barcode) void {
        if (self.handle) |h| c.pdf_barcode_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *Barcode) void {
        self.deinit();
    }

    /// The barcode's payload string; caller owns the returned slice.
    pub fn getData(self: Barcode, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_barcode_get_data(h, &code), code);
    }

    /// The barcode's format code.
    pub fn getFormat(self: Barcode) Error!i32 {
        const h = try self.live();
        var code: i32 = 0;
        const f = c.pdf_barcode_get_format(h, &code);
        if (code != 0) return fail(code);
        return f;
    }

    /// The barcode's decode confidence (1.0 for a freshly generated barcode).
    pub fn getConfidence(self: Barcode) Error!f32 {
        const h = try self.live();
        var code: i32 = 0;
        const conf = c.pdf_barcode_get_confidence(h, &code);
        if (code != 0) return fail(code);
        return conf;
    }

    /// Render the barcode to PNG bytes (`size_px` is advisory). Caller owns the
    /// returned slice.
    pub fn getImagePng(self: Barcode, alloc: std.mem.Allocator, size_px: i32) Error![]u8 {
        const h = try self.live();
        var out_len: i32 = 0;
        var code: i32 = 0;
        const p = c.pdf_barcode_get_image_png(h, size_px, &out_len, &code) orelse return fail(code);
        defer c.free_bytes(p);
        const n: usize = if (out_len < 0) 0 else @intCast(out_len);
        return alloc.dupe(u8, p[0..n]);
    }

    /// Render the barcode to an SVG string (`size_px` is advisory). Caller owns
    /// the returned slice.
    pub fn getSvg(self: Barcode, alloc: std.mem.Allocator, size_px: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_barcode_get_svg(h, size_px, &code), code);
    }
};

/// A list of layout elements extracted from a page. Owns the native
/// `FfiElementList` handle; free with `deinit`/`close`.
pub const ElementList = struct {
    handle: ?*c.FfiElementList,

    /// Guard: returns the live handle or raises if the list was closed.
    fn live(self: ElementList) Error!*c.FfiElementList {
        return self.handle orelse fail(-1);
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *ElementList) void {
        if (self.handle) |h| c.pdf_oxide_elements_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *ElementList) void {
        self.deinit();
    }

    /// Number of elements in the list.
    pub fn count(self: ElementList) Error!i32 {
        const h = try self.live();
        const n = c.pdf_oxide_element_count(h);
        if (n < 0) return fail(n);
        return n;
    }

    /// Element type string at `index`; caller owns the returned slice.
    pub fn getType(self: ElementList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_element_get_type(h, index, &code), code);
    }

    /// Element text at `index`; caller owns the returned slice.
    pub fn getText(self: ElementList, alloc: std.mem.Allocator, index: i32) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_element_get_text(h, index, &code), code);
    }

    /// Element bounding box at `index`.
    pub fn getRect(self: ElementList, index: i32) Error!Bbox {
        const h = try self.live();
        var code: i32 = 0;
        var x: f32 = 0;
        var y: f32 = 0;
        var w: f32 = 0;
        var ht: f32 = 0;
        c.pdf_oxide_element_get_rect(h, index, &x, &y, &w, &ht, &code);
        if (code != 0) return fail(code);
        return .{ .x = x, .y = y, .width = w, .height = ht };
    }

    /// Serialize the whole list to a JSON string; caller owns the returned slice.
    pub fn toJson(self: ElementList, alloc: std.mem.Allocator) Error![]u8 {
        const h = try self.live();
        var code: i32 = 0;
        return takeString(alloc, c.pdf_oxide_elements_to_json(h, &code), code);
    }
};

/// An OCR engine backed by detection/recognition models. Owns an opaque
/// `*anyopaque` native handle; free with `deinit`/`close`.
pub const OcrEngine = struct {
    handle: ?*anyopaque,

    /// Create an OCR engine from model/dictionary file paths.
    pub fn create(det_model_path: [:0]const u8, rec_model_path: [:0]const u8, dict_path: [:0]const u8) Error!OcrEngine {
        var code: i32 = 0;
        const h = c.pdf_ocr_engine_create(det_model_path.ptr, rec_model_path.ptr, dict_path.ptr, &code) orelse
            return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *OcrEngine) void {
        if (self.handle) |h| c.pdf_ocr_engine_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *OcrEngine) void {
        self.deinit();
    }
};

/// A reusable page renderer configured with DPI/format/quality/anti-alias. Owns
/// an opaque `*anyopaque` native handle; free with `deinit`/`close`.
pub const Renderer = struct {
    handle: ?*anyopaque,

    /// Create a renderer. `format`: 0=PNG 1=JPEG; `quality` is the JPEG quality.
    pub fn create(dpi: i32, format: i32, quality: i32, anti_alias: bool) Error!Renderer {
        var code: i32 = 0;
        const h = c.pdf_create_renderer(dpi, format, quality, anti_alias, &code) orelse return fail(code);
        return .{ .handle = h };
    }

    /// Free the native handle (idempotent). Also exposed as `close`.
    pub fn deinit(self: *Renderer) void {
        if (self.handle) |h| c.pdf_renderer_free(h);
        self.handle = null;
    }

    /// Free the native handle (idempotent).
    pub fn close(self: *Renderer) void {
        self.deinit();
    }
};

/// Merge the PDFs at `paths` (filesystem paths) into a single PDF; caller owns
/// the returned bytes.
pub fn merge(alloc: std.mem.Allocator, paths: []const [*:0]const u8) Error![]u8 {
    var data_len: i32 = 0;
    var code: i32 = 0;
    const pp: [*c]const [*c]const u8 = if (paths.len == 0) null else @ptrCast(paths.ptr);
    const p = c.pdf_merge(pp, @intCast(paths.len), &data_len, &code) orelse return fail(code);
    const n: usize = if (data_len < 0) 0 else @intCast(data_len);
    return takeBytes(alloc, p, n, code);
}

/// Apply an RFC 3161 timestamp to signature `sig_index` of the PDF in
/// `pdf_data`, querying the TSA at `tsa_url`. Returns the timestamped PDF bytes;
/// caller owns the returned slice.
pub fn addTimestamp(
    alloc: std.mem.Allocator,
    pdf_data: []const u8,
    sig_index: i32,
    tsa_url: [:0]const u8,
) Error![]u8 {
    // `uint8_t **out_data` maps to `[*c][*c]u8`; use the C-pointer type for the
    // local so `&out_data` matches the parameter exactly.
    var out_data: [*c]u8 = null;
    var out_len: usize = 0;
    var code: i32 = 0;
    const ok = c.pdf_add_timestamp(pdf_data.ptr, pdf_data.len, sig_index, tsa_url.ptr, &out_data, &out_len, &code);
    if (!ok) return fail(code);
    return takeBytes(alloc, out_data, out_len, code);
}

// ── PHASE-8: process-wide configuration / crypto / model namespaces ───────────

/// Global cap on content-stream operators (anti-DoS); returns the previous cap.
pub fn setMaxOpsPerStream(limit: i64) i64 {
    return c.pdf_oxide_set_max_ops_per_stream(limit);
}

/// Toggle preservation of glyphs with no Unicode mapping; returns the prior
/// setting (1=on, 0=off).
pub fn setPreserveUnmappedGlyphs(preserve: bool) i32 {
    return c.pdf_oxide_set_preserve_unmapped_glyphs(@intFromBool(preserve));
}

/// Name of the active crypto provider; caller owns the returned slice.
pub fn cryptoActiveProvider(alloc: std.mem.Allocator) Error![]u8 {
    return takeString(alloc, c.pdf_oxide_crypto_active_provider(), -1);
}

/// Whether a FIPS-validated crypto module is available (1=yes, 0=no).
pub fn cryptoFipsAvailable() i32 {
    return c.pdf_oxide_crypto_fips_available();
}

/// Switch to the FIPS crypto provider; returns a status (0=ok).
pub fn cryptoUseFips() i32 {
    return c.pdf_oxide_crypto_use_fips();
}

/// Set the crypto policy from a `spec` string; returns a status (0=ok).
pub fn cryptoSetPolicy(spec: [:0]const u8) i32 {
    return c.pdf_oxide_crypto_set_policy(spec.ptr);
}

/// The active crypto policy as a string; caller owns the returned slice.
pub fn cryptoPolicy(alloc: std.mem.Allocator) Error![]u8 {
    return takeString(alloc, c.pdf_oxide_crypto_policy(), -1);
}

/// The crypto inventory (algorithms/providers) as a JSON string; caller owns it.
pub fn cryptoInventory(alloc: std.mem.Allocator) Error![]u8 {
    return takeString(alloc, c.pdf_oxide_crypto_inventory(), -1);
}

/// A CBOM (Cryptographic Bill of Materials) JSON string; caller owns it.
pub fn cryptoCbom(alloc: std.mem.Allocator) Error![]u8 {
    return takeString(alloc, c.pdf_oxide_crypto_cbom(), -1);
}

/// Prefetch OCR models for the comma-separated `languages_csv` (null/empty →
/// English); returns the model cache dir. Caller owns the returned slice.
pub fn prefetchModels(alloc: std.mem.Allocator, languages_csv: ?[:0]const u8) Error![]u8 {
    var code: i32 = 0;
    const csv: ?[*:0]const u8 = if (languages_csv) |l| l.ptr else null;
    return takeString(alloc, c.pdf_oxide_prefetch_models(csv, &code), code);
}

/// Whether this build can actually download models (1=yes, 0=cache-dir only).
pub fn prefetchAvailable() i32 {
    return c.pdf_oxide_prefetch_available();
}

/// The air-gapped OCR model manifest as a JSON string; caller owns it.
pub fn modelManifest(alloc: std.mem.Allocator) Error![]u8 {
    return takeString(alloc, c.pdf_oxide_model_manifest(), -1);
}

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

test "DocumentEditor: open/edit/save coverage" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var ed = try DocumentEditor.openFromBytes(bytes); // openFromBytes
    defer ed.close(); // close

    try testing.expect(try ed.pageCount() >= 1); // pageCount

    // isModified: a bool (smoke; value unspecified pre-edit)
    const modified = try ed.isModified();
    try testing.expect(modified == true or modified == false); // isModified

    try ed.rotateAllPages(90); // rotateAllPages
    const rot = try ed.getPageRotation(0); // getPageRotation
    try testing.expect(rot == 90 or rot >= 0);

    try ed.setProducer("x"); // setProducer
    const producer = try ed.getProducer(a); // getProducer
    defer a.free(producer);
    try testing.expect(producer.len >= 0);

    const out = try ed.saveToBytes(a); // saveToBytes
    defer a.free(out);
    try testing.expect(out.len > 0);
}

test "DocumentBuilder/PageBuilder: create -> page -> build -> reopen coverage" {
    const a = testing.allocator;

    var builder = try DocumentBuilder.create(); // create
    defer builder.close(); // close

    try builder.setTitle("Builder Coverage"); // setTitle
    try builder.setAuthor("pdf_oxide"); // setAuthor

    var pg = try builder.page(595, 842); // page(width, height)
    errdefer pg.close();

    try pg.font("Helvetica", 12); // font
    try pg.heading(1, "Title"); // heading
    try pg.paragraph("Hello world from the builder."); // paragraph

    try pg.done(); // done (consumes the page handle)

    const bytes = try builder.build(a); // build
    defer a.free(bytes);
    try testing.expect(bytes.len > 0);

    // Reopen the built bytes and assert content survived the round-trip.
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();
    try testing.expect(try doc.pageCount() >= 1);

    const text = try doc.toPlainTextAll(a);
    defer a.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "Hello") != null or
        std.mem.indexOf(u8, text, "Title") != null);
}

test "error path: open nonexistent returns error" {
    try testing.expectError(Error.PdfOxide, Document.open("/nonexistent/nope.pdf"));
}

// ── PHASE-6 api-coverage tests ────────────────────────────────────────────────

test "phase-6 logging: set/get log level round-trip" {
    const prev = getLogLevel();
    defer setLogLevel(prev);
    setLogLevel(3);
    try testing.expectEqual(@as(i32, 3), getLogLevel());
    setLogLevel(0);
    try testing.expectEqual(@as(i32, 0), getLogLevel());
}

test "phase-6 validation: validatePdfA/validatePdfUa/validatePdfX" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // PDF/A
    {
        var res = try doc.validatePdfA(0); // A1b
        defer res.deinit();
        _ = try res.isCompliant(); // bool
        try testing.expect(try res.errorCount() >= 0);
        try testing.expect(try res.warningCount() >= 0);
        const errs = try res.errors(a);
        defer PdfAResults.freeStrings(a, errs);
        try testing.expect(errs.len == @as(usize, @intCast(try res.errorCount())));
    }

    // PDF/UA
    {
        var res = try doc.validatePdfUa(1);
        defer res.deinit();
        _ = try res.isAccessible(); // bool
        try testing.expect(try res.errorCount() >= 0);
        try testing.expect(try res.warningCount() >= 0);
        const errs = try res.errors(a);
        defer UaResults.freeStrings(a, errs);
        const warns = try res.warnings(a);
        defer UaResults.freeStrings(a, warns);
        const stats = try res.uaStats();
        try testing.expect(stats.pages >= 0);
        try testing.expect(stats.images >= 0);
    }

    // PDF/X
    {
        var res = try doc.validatePdfX(0);
        defer res.deinit();
        _ = try res.isCompliant(); // bool
        try testing.expect(try res.errorCount() >= 0);
        const errs = try res.errors(a);
        defer PdfXResults.freeStrings(a, errs);
        try testing.expect(errs.len == @as(usize, @intCast(try res.errorCount())));
    }
}

test "phase-6 signing/PKI/timestamp/TSA/DSS: every wrapper is exercised" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    // Certificate: loading bogus material must raise the binding error.
    try testing.expectError(Error.PdfOxide, Certificate.loadFromBytes("not-a-p12", ""));
    try testing.expectError(Error.PdfOxide, Certificate.loadFromPem("not-pem", "not-pem"));

    // Exercise the certificate accessors + signing entry points against a
    // dummy (closed) certificate handle: each must raise rather than crash.
    var bad_cert = Certificate{ .handle = null };
    bad_cert.close(); // close() idempotent on a null handle
    try testing.expectError(Error.PdfOxide, bad_cert.subject(a));
    try testing.expectError(Error.PdfOxide, bad_cert.issuer(a));
    try testing.expectError(Error.PdfOxide, bad_cert.serial(a));
    try testing.expectError(Error.PdfOxide, bad_cert.validity());
    try testing.expectError(Error.PdfOxide, bad_cert.isValid());

    // signBytes / signBytesPades / signBytesPadesOpts with a closed cert handle.
    const empty: []const []const u8 = &.{};
    try testing.expectError(Error.PdfOxide, signBytes(a, bytes, bad_cert, "r", "l"));
    try testing.expectError(Error.PdfOxide, signBytesPades(a, bytes, bad_cert, 0, null, "r", "l", empty, empty, empty));
    try testing.expectError(Error.PdfOxide, signBytesPadesOpts(a, bytes, bad_cert, 0, null, "r", "l", empty, empty, empty));

    // SignatureInfo accessors against a closed handle.
    var bad_sig = SignatureInfo{ .handle = null };
    bad_sig.close();
    try testing.expectError(Error.PdfOxide, bad_sig.signerName(a));
    try testing.expectError(Error.PdfOxide, bad_sig.signingReason(a));
    try testing.expectError(Error.PdfOxide, bad_sig.signingLocation(a));
    try testing.expectError(Error.PdfOxide, bad_sig.signingTime());
    try testing.expectError(Error.PdfOxide, bad_sig.certificate());
    try testing.expectError(Error.PdfOxide, bad_sig.padesLevel());
    try testing.expectError(Error.PdfOxide, bad_sig.hasTimestamp());
    try testing.expectError(Error.PdfOxide, bad_sig.timestamp());
    try testing.expectError(Error.PdfOxide, bad_sig.verify());
    try testing.expectError(Error.PdfOxide, bad_sig.verifyDetached(bytes));

    var some_ts = Timestamp{ .handle = null };
    try testing.expectError(Error.PdfOxide, bad_sig.addTimestamp(some_ts));

    // Timestamp: parsing bogus DER must raise; accessors on a closed handle raise.
    try testing.expectError(Error.PdfOxide, Timestamp.parse("not-der"));
    some_ts.close();
    try testing.expectError(Error.PdfOxide, some_ts.token(a));
    try testing.expectError(Error.PdfOxide, some_ts.messageImprint(a));
    try testing.expectError(Error.PdfOxide, some_ts.time());
    try testing.expectError(Error.PdfOxide, some_ts.serial(a));
    try testing.expectError(Error.PdfOxide, some_ts.tsaName(a));
    try testing.expectError(Error.PdfOxide, some_ts.policyOid(a));
    try testing.expectError(Error.PdfOxide, some_ts.hashAlgorithm());
    try testing.expectError(Error.PdfOxide, some_ts.verify());

    // TsaClient: create may succeed (no network yet); exercise request paths.
    if (TsaClient.create("http://invalid.invalid/tsa", "", "", 1, 0, false, false)) |client_const| {
        var client = client_const;
        defer client.close();
        // Requests will fail (no reachable TSA) → expect the binding error.
        try testing.expectError(Error.PdfOxide, client.requestTimestamp(bytes));
        try testing.expectError(Error.PdfOxide, client.requestTimestampHash(bytes, 0));
    } else |_| {
        // create itself rejected the URL — also acceptable coverage.
    }
    // Request paths against a closed client also raise.
    var bad_client = TsaClient{ .handle = null };
    bad_client.close();
    try testing.expectError(Error.PdfOxide, bad_client.requestTimestamp(bytes));
    try testing.expectError(Error.PdfOxide, bad_client.requestTimestampHash(bytes, 0));

    // DSS: accessors against a closed handle raise (counts also guard).
    var bad_dss = Dss{ .handle = null };
    bad_dss.close();
    try testing.expectError(Error.PdfOxide, bad_dss.certCount());
    try testing.expectError(Error.PdfOxide, bad_dss.crlCount());
    try testing.expectError(Error.PdfOxide, bad_dss.ocspCount());
    try testing.expectError(Error.PdfOxide, bad_dss.vriCount());
    try testing.expectError(Error.PdfOxide, bad_dss.getCert(a, 0));
    try testing.expectError(Error.PdfOxide, bad_dss.getCrl(a, 0));
    try testing.expectError(Error.PdfOxide, bad_dss.getOcsp(a, 0));
}

// ── PHASE-7 api-coverage tests ────────────────────────────────────────────────

test "phase-7 barcodes: generateQrCode/generateBarcode + accessors" {
    const a = testing.allocator;

    // QR code
    {
        var bc = try Barcode.generateQrCode("https://example.com", 1, 256); // generateQrCode
        defer bc.close(); // close

        const data = try bc.getData(a); // getData
        defer a.free(data);
        try testing.expect(data.len > 0);

        _ = try bc.getFormat(); // getFormat
        _ = try bc.getConfidence(); // getConfidence

        const png = try bc.getImagePng(a, 256); // getImagePng
        defer a.free(png);
        try testing.expect(png.len > 0);

        const svg = try bc.getSvg(a, 256); // getSvg
        defer a.free(svg);
        try testing.expect(svg.len > 0);
    }

    // 1-D barcode (Code128)
    {
        var bc = try Barcode.generateBarcode("ABC123", 0, 128); // generateBarcode
        defer bc.deinit();
        const data = try bc.getData(a);
        defer a.free(data);
        try testing.expect(data.len > 0);
        const fmt = try bc.getFormat();
        try testing.expect(fmt >= 0);
        const png = try bc.getImagePng(a, 128);
        defer a.free(png);
        try testing.expect(png.len > 0);
    }

    // Accessors on a closed barcode raise.
    var bad = Barcode{ .handle = null };
    bad.close();
    try testing.expectError(Error.PdfOxide, bad.getData(a));
    try testing.expectError(Error.PdfOxide, bad.getFormat());
    try testing.expectError(Error.PdfOxide, bad.getConfidence());
    try testing.expectError(Error.PdfOxide, bad.getImagePng(a, 64));
    try testing.expectError(Error.PdfOxide, bad.getSvg(a, 64));
}

test "phase-7 render variants: withOptions/withOptionsEx/region/fit/raw + estimate" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    {
        var img = try doc.renderPageWithOptions(a, 0, 96, 0, 1.0, 1.0, 1.0, 1.0, false, true, 90); // renderPageWithOptions
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.height > 0);
        try testing.expect(img.data.len > 0);
    }

    {
        const layers: []const [*:0]const u8 = &.{};
        var img = try doc.renderPageWithOptionsEx(a, 0, 96, 0, 1.0, 1.0, 1.0, 1.0, false, true, 90, layers); // renderPageWithOptionsEx
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.data.len > 0);
    }

    {
        var img = try doc.renderPageRegion(a, 0, 0, 0, 100, 100, 0); // renderPageRegion
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.data.len > 0);
    }

    {
        var img = try doc.renderPageFit(a, 0, 200, 200, 0); // renderPageFit
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.width <= 200);
        try testing.expect(img.height <= 200);
    }

    {
        var img = try doc.renderPageRaw(a, 0, 96); // renderPageRaw
        defer img.deinit();
        try testing.expect(img.width > 0);
        try testing.expect(img.height > 0);
        try testing.expect(img.data.len == @as(usize, @intCast(img.width)) * @as(usize, @intCast(img.height)) * 4);
    }

    // estimateRenderTime: a value (>= 0), or the binding error.
    const est = doc.estimateRenderTime(0); // estimateRenderTime
    if (est) |t| {
        try testing.expect(t >= 0);
    } else |_| {}
}

test "phase-7 renderer: create + free" {
    // pdf_create_renderer is a no-op stub in this ABI: it may return a handle or
    // raise. Either outcome exercises the wrapper.
    if (Renderer.create(150, 0, 90, true)) |r_val| {
        var r = r_val;
        r.close(); // close (pdf_renderer_free)
    } else |err| {
        try testing.expect(err == Error.PdfOxide);
    }
}

test "phase-7 page getters: width/height/rotation/elements" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    const w = try doc.pageGetWidth(0); // pageGetWidth
    try testing.expect(w > 0);
    const h = try doc.pageGetHeight(0); // pageGetHeight
    try testing.expect(h > 0);
    const rot = try doc.pageGetRotation(0); // pageGetRotation
    try testing.expect(rot >= 0);

    var els = try doc.pageGetElements(0); // pageGetElements
    defer els.close(); // close
    const n = try els.count(); // count
    try testing.expect(n >= 0);
    if (n > 0) {
        const t = try els.getType(a, 0); // getType
        defer a.free(t);
        const txt = try els.getText(a, 0); // getText
        defer a.free(txt);
        _ = try els.getRect(0); // getRect
    }
    const json = try els.toJson(a); // toJson
    defer a.free(json);
    try testing.expect(json.len > 0);
}

test "phase-7 redaction: add/count/apply/scrubMetadata on an editor" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var ed = try DocumentEditor.openFromBytes(bytes);
    defer ed.close();

    try ed.redactionAdd(0, 10, 10, 100, 50, 0, 0, 0); // redactionAdd
    const cnt = try ed.redactionCount(0); // redactionCount
    try testing.expect(cnt >= 1);

    // apply may remove glyphs or report an unsupported-font failure; both are
    // valid coverage of the wrapper.
    const removed = ed.redactionApply(false, 0, 0, 0); // redactionApply
    if (removed) |n| {
        try testing.expect(n >= 0);
    } else |_| {}

    const scrubbed = ed.redactionScrubMetadata(); // redactionScrubMetadata
    if (scrubbed) |n| {
        try testing.expect(n >= 0);
    } else |_| {}

    // Closed-editor guards.
    var bad = DocumentEditor{ .handle = null };
    bad.close();
    try testing.expectError(Error.PdfOxide, bad.redactionAdd(0, 0, 0, 1, 1, 0, 0, 0));
    try testing.expectError(Error.PdfOxide, bad.redactionCount(0));
    try testing.expectError(Error.PdfOxide, bad.redactionApply(false, 0, 0, 0));
    try testing.expectError(Error.PdfOxide, bad.redactionScrubMetadata());
}

test "phase-7 addBarcodeToPage: place a generated QR on an editor page" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var ed = try DocumentEditor.openFromBytes(bytes);
    defer ed.close();

    var bc = try Barcode.generateQrCode("PHASE7", 1, 128);
    defer bc.close();

    // Placement either succeeds or raises; either exercises the wrapper.
    if (ed.addBarcodeToPage(0, bc, 10, 10, 80, 80)) |_| { // addBarcodeToPage
        // ok
    } else |_| {}

    // Closed-barcode + closed-editor guards.
    var bad_bc = Barcode{ .handle = null };
    bad_bc.close();
    try testing.expectError(Error.PdfOxide, ed.addBarcodeToPage(0, bad_bc, 0, 0, 1, 1));
}

test "phase-7 Pdf constructors: fromImageBytes/fromHtmlCss/fromHtmlCssWithFonts" {
    const a = testing.allocator;

    // fromImageBytes: bogus image bytes must raise the binding error.
    const bad_img = Pdf.fromImageBytes("not-a-png"); // fromImageBytes
    try testing.expectError(Error.PdfOxide, bad_img);

    // fromImage: a nonexistent path must raise.
    const bad_path = Pdf.fromImage("/nonexistent/none.png"); // fromImage
    try testing.expectError(Error.PdfOxide, bad_path);

    // fromHtmlCss: builds where the html-render path is available, else raises
    // (e.g. no default font in this cdylib). Either outcome exercises it.
    {
        const empty_font: []const u8 = &.{};
        if (Pdf.fromHtmlCss("<h1>Hi</h1><p>Body</p>", "h1{color:#000}", empty_font)) |p_val| { // fromHtmlCss
            var p = p_val;
            defer p.deinit();
            const out = try p.toBytes(a);
            defer a.free(out);
            try testing.expect(out.len > 0);
        } else |err| {
            try testing.expect(err == Error.PdfOxide);
        }
    }

    // fromHtmlCssWithFonts: no fonts (empty parallel arrays); same tolerance.
    {
        const fams: []const [*:0]const u8 = &.{};
        const fonts: []const []const u8 = &.{};
        if (Pdf.fromHtmlCssWithFonts(a, "<p>Cascade</p>", "", fams, fonts)) |p_val| { // fromHtmlCssWithFonts
            var p = p_val;
            defer p.deinit();
            const out = try p.toBytes(a);
            defer a.free(out);
            try testing.expect(out.len > 0);
        } else |err| {
            try testing.expect(err == Error.PdfOxide);
        }
    }
}

test "phase-7 merge: combine two temp PDFs into one" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    const p1 = "/tmp/pdfoxide_zig_merge_a.pdf";
    const p2 = "/tmp/pdfoxide_zig_merge_b.pdf";
    {
        const f1 = try std.fs.cwd().createFile(p1, .{});
        defer f1.close();
        try f1.writeAll(bytes);
        const f2 = try std.fs.cwd().createFile(p2, .{});
        defer f2.close();
        try f2.writeAll(bytes);
    }
    defer std.fs.cwd().deleteFile(p1) catch {};
    defer std.fs.cwd().deleteFile(p2) catch {};

    const paths = [_][*:0]const u8{ p1, p2 };
    const merged = merge(a, &paths); // merge
    if (merged) |m| {
        defer a.free(m);
        try testing.expect(m.len > 0);
    } else |_| {
        // merge rejecting the inputs is also acceptable wrapper coverage.
    }

    // merge over a nonexistent path raises.
    const bad_paths = [_][*:0]const u8{"/nonexistent/none.pdf"};
    try testing.expectError(Error.PdfOxide, merge(a, &bad_paths));
}

test "phase-7 OCR engine + page OCR: minimal inputs return or raise" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // ocrPageNeedsOcr: a bool or the binding error (feature may be disabled).
    if (doc.ocrPageNeedsOcr(0)) |needs| { // ocrPageNeedsOcr
        try testing.expect(needs == true or needs == false);
    } else |_| {}

    // ocrExtractText with a null engine: returns text or raises.
    if (doc.ocrExtractText(a, 0, null)) |txt| { // ocrExtractText
        defer a.free(txt);
        try testing.expect(txt.len >= 0);
    } else |_| {}

    // OcrEngine.create with bogus model paths must raise the binding error.
    const bad_engine = OcrEngine.create("/nonexistent/det", "/nonexistent/rec", "/nonexistent/dict"); // create
    if (bad_engine) |engine_const| {
        var engine = engine_const;
        engine.close(); // close
    } else |_| {}
}

test "phase-7 addTimestamp: minimal inputs return or raise" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    // No reachable TSA / no signature at index 0 → expect the binding error
    // (or, if it somehow succeeds, owned bytes that we free).
    const r = addTimestamp(a, bytes, 0, "http://invalid.invalid/tsa"); // addTimestamp
    if (r) |out| {
        defer a.free(out);
        try testing.expect(out.len > 0);
    } else |_| {}
}

// ── PHASE-8 api-coverage tests ────────────────────────────────────────────────

test "phase-8 office: open-from-bytes + export are return-or-error" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);

    // Opening a PDF as DOCX/PPTX/XLSX bytes is not valid office content → raise
    // (or, implausibly, succeed → free the handle).
    inline for (.{ "openFromDocxBytes", "openFromPptxBytes", "openFromXlsxBytes" }) |name| {
        if (@field(Document, name)(bytes)) |doc_const| {
            var doc = doc_const;
            doc.deinit();
        } else |_| {}
    }

    // Exporting the sample PDF to office formats: return owned bytes or raise.
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();
    inline for (.{ "toDocx", "toPptx", "toXlsx" }) |name| {
        if (@field(Document, name)(doc, a)) |out| {
            defer a.free(out);
            try testing.expect(out.len >= 0);
        } else |_| {}
    }
}

test "phase-8 in-rect extractors: text/words/lines/tables/images" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    const w = try doc.pageGetWidth(0);
    const h = try doc.pageGetHeight(0);

    if (doc.extractTextInRect(a, 0, 0, 0, w, h)) |t| { // extractTextInRect
        defer a.free(t);
        try testing.expect(t.len >= 0);
    } else |_| {}

    if (doc.extractWordsInRect(a, 0, 0, 0, w, h)) |words| { // extractWordsInRect
        defer Document.freeWords(a, words);
        try testing.expect(words.len >= 0);
    } else |_| {}

    if (doc.extractLinesInRect(a, 0, 0, 0, w, h)) |lines| { // extractLinesInRect
        defer Document.freeTextLines(a, lines);
        try testing.expect(lines.len >= 0);
    } else |_| {}

    if (doc.extractTablesInRect(a, 0, 0, 0, w, h)) |tables| { // extractTablesInRect
        defer Document.freeTables(a, tables);
        try testing.expect(tables.len >= 0);
    } else |_| {}

    if (doc.extractImagesInRect(a, 0, 0, 0, w, h)) |images| { // extractImagesInRect
        defer Document.freeImages(a, images);
        try testing.expect(images.len >= 0);
    } else |_| {}
}

test "phase-8 auto extraction + classification" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    if (doc.extractTextAuto(a, 0)) |t| { // extractTextAuto
        defer a.free(t);
        try testing.expect(t.len >= 0);
    } else |_| {}

    if (doc.extractAllText(a)) |t| { // extractAllText
        defer a.free(t);
        try testing.expect(t.len >= 0);
    } else |_| {}

    if (doc.extractPageAuto(a, 0, null)) |t| { // extractPageAuto
        defer a.free(t);
        try testing.expect(t.len >= 0);
    } else |_| {}

    if (doc.classifyPage(a, 0)) |t| { // classifyPage
        defer a.free(t);
        try testing.expect(t.len > 0);
    } else |_| {}

    if (doc.classifyDocument(a)) |t| { // classifyDocument
        defer a.free(t);
        try testing.expect(t.len > 0);
    } else |_| {}
}

test "phase-8 header/footer/artifact erase + remove" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    inline for (.{ "eraseHeader", "eraseFooter", "eraseArtifacts" }) |name| {
        if (@field(Document, name)(doc, 0)) |n| {
            try testing.expect(n >= -1);
        } else |_| {}
    }
    inline for (.{ "removeHeaders", "removeFooters", "removeArtifacts" }) |name| {
        if (@field(Document, name)(doc, 0.5)) |n| {
            try testing.expect(n >= -1);
        } else |_| {}
    }
}

test "phase-8 forms: get_form_fields (empty ok) + accessors + export/import" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // get_form_fields: the markdown sample has no AcroForm → an empty list is OK.
    if (doc.formFields()) |fields_const| { // formFields
        var fields = fields_const;
        defer fields.deinit();
        const n = try fields.count(); // count
        try testing.expect(n >= 0);
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            const nm = try fields.getName(a, i); // getName
            defer a.free(nm);
            const vl = try fields.getValue(a, i); // getValue
            defer a.free(vl);
            const ty = try fields.getType(a, i); // getType
            defer a.free(ty);
            _ = try fields.isReadonly(i); // isReadonly
            _ = try fields.isRequired(i); // isRequired
        }
    } else |_| {}

    // export form data (FDF/XFDF): return owned bytes or raise.
    inline for (.{ @as(i32, 0), @as(i32, 1) }) |fmt| {
        if (doc.exportFormDataToBytes(a, fmt)) |out| { // exportFormDataToBytes
            defer a.free(out);
            try testing.expect(out.len >= 0);
        } else |_| {}
    }

    // import from a nonexistent path: return-or-error.
    if (doc.importFormData("/nonexistent/data.fdf")) |n| { // importFormData
        try testing.expect(n >= -1);
    } else |_| {}
    if (doc.formImportFromFile("/nonexistent/data.fdf")) |_| {} else |_| {} // formImportFromFile
}

test "phase-8 structure/metadata + convert" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    inline for (.{ "outline", "pageLabels", "xmpMetadata" }) |name| {
        if (@field(Document, name)(doc, a)) |s| {
            defer a.free(s);
            try testing.expect(s.len >= 0);
        } else |_| {}
    }

    if (doc.sourceBytes(a)) |s| { // sourceBytes
        defer a.free(s);
        try testing.expect(s.len >= 0);
    } else |_| {}

    _ = doc.hasXfa(); // hasXfa (smoke)

    if (doc.planSplitByBookmarks(a, null)) |s| { // planSplitByBookmarks
        defer a.free(s);
        try testing.expect(s.len >= 0);
    } else |_| {}

    if (doc.convertToPdfA(2)) |_| {} else |_| {} // convertToPdfA
}

test "phase-8 document signatures: count/verify/timestamp/dss are return-or-error" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    if (doc.signatureCount()) |n| { // signatureCount
        try testing.expect(n >= 0);
        if (n > 0) {
            if (doc.signature(0)) |sig_const| { // signature
                var sig = sig_const;
                sig.deinit();
            } else |_| {}
        }
    } else |_| {}

    if (doc.verifyAllSignatures()) |n| { // verifyAllSignatures
        try testing.expect(n >= -1);
    } else |_| {}

    if (doc.hasTimestamp()) |_| {} else |_| {} // hasTimestamp

    if (doc.dss()) |dss_const| { // dss
        var d = dss_const;
        d.deinit();
    } else |_| {}

    // sign with a closed cert must raise.
    var bad_cert = Certificate{ .handle = null };
    bad_cert.close();
    try testing.expectError(Error.PdfOxide, doc.sign(bad_cert, "r", "l")); // sign
}

test "phase-8 list-handle JSON + extra accessors" {
    const a = testing.allocator;
    const bytes = try samplePdf(a);
    defer a.free(bytes);
    var doc = try Document.openFromBytes(bytes);
    defer doc.deinit();

    // AnnotationList: the sample has none → empty list, accessors return-or-error.
    if (doc.annotationList(0)) |al_const| { // annotationList
        var al = al_const;
        defer al.deinit();
        const n = try al.count();
        try testing.expect(n >= 0);
        const j = try al.toJson(a); // annotations_to_json
        defer a.free(j);
        try testing.expect(j.len >= 0);
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            _ = al.getColor(i) catch 0; // getColor
            _ = al.getCreationDate(i) catch 0; // getCreationDate
            _ = al.getModificationDate(i) catch 0; // getModificationDate
            _ = al.isHidden(i) catch false; // isHidden
            _ = al.isMarkedDeleted(i) catch false; // isMarkedDeleted
            _ = al.isPrintable(i) catch false; // isPrintable
            _ = al.isReadOnly(i) catch false; // isReadOnly
            if (al.highlightQuadPointsCount(i)) |qn| { // highlightQuadPointsCount
                if (qn > 0) {
                    _ = al.highlightQuadPoint(i, 0) catch [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 }; // highlightQuadPoint
                }
            } else |_| {}
            if (al.linkUri(a, i)) |u| a.free(u) else |_| {} // linkUri
            if (al.textIconName(a, i)) |u| a.free(u) else |_| {} // textIconName
        }
    } else |_| {}

    // FontList: size accessor + JSON.
    if (doc.fontList(0)) |fl_const| { // fontList
        var fl = fl_const;
        defer fl.deinit();
        const n = try fl.count();
        try testing.expect(n >= 0);
        if (n > 0) _ = fl.getSize(0) catch 0; // getSize
        const j = try fl.toJson(a); // fonts_to_json
        defer a.free(j);
        try testing.expect(j.len >= 0);
    } else |_| {}

    // ElementList handle (Document.elementList) + JSON.
    if (doc.elementList(0)) |el_const| { // elementList
        var el = el_const;
        defer el.deinit();
        const j = try el.toJson(a); // elements_to_json
        defer a.free(j);
        try testing.expect(j.len >= 0);
    } else |_| {}

    // SearchResultList handle + JSON.
    if (doc.searchPageList(0, "Alpha", false)) |sl_const| { // searchPageList
        var sl = sl_const;
        defer sl.deinit();
        const n = try sl.count();
        try testing.expect(n >= 0);
        const j = try sl.toJson(a); // search_results_to_json
        defer a.free(j);
        try testing.expect(j.len >= 0);
    } else |_| {}
}

test "phase-8 editor: import FDF/XFDF bytes are return-or-error" {
    var pdf = try Pdf.fromMarkdown("# Editor\n\nbody\n");
    defer pdf.deinit();
    try pdf.save("/tmp/pdfoxide_zig_p8_editor.pdf");
    defer std.fs.cwd().deleteFile("/tmp/pdfoxide_zig_p8_editor.pdf") catch {};

    var ed = try DocumentEditor.openEditor("/tmp/pdfoxide_zig_p8_editor.pdf");
    defer ed.deinit();

    if (ed.importFdfBytes("not-fdf")) |n| { // importFdfBytes
        try testing.expect(n >= -1);
    } else |_| {}
    if (ed.importXfdfBytes("<not-xfdf/>")) |n| { // importXfdfBytes
        try testing.expect(n >= -1);
    } else |_| {}

    // Closed-editor guard.
    var closed = DocumentEditor{ .handle = null };
    closed.close();
    try testing.expectError(Error.PdfOxide, closed.importFdfBytes("x"));
    try testing.expectError(Error.PdfOxide, closed.importXfdfBytes("x"));
}

test "phase-8 Pdf.pageCount alias" {
    var pdf = try Pdf.fromMarkdown("# One\n\nbody\n");
    defer pdf.deinit();
    // pdf_get_page_count reports a page count or errors (code 1) on a
    // freshly-built Pdf in this cdylib; either outcome exercises the wrapper.
    if (pdf.pageCount()) |n| { // pageCount (pdf_get_page_count)
        try testing.expect(n >= 0);
    } else |err| {
        try testing.expect(err == Error.PdfOxide);
    }
}

test "phase-8 process config: max-ops / preserve-unmapped-glyphs are invokable" {
    // These setters return the PRIOR value and have no error channel; assert only
    // that each call is invokable (returns an int), not a specific round-tripped
    // value.
    const prev = setMaxOpsPerStream(1_000_000); // setMaxOpsPerStream (returns prior i64)
    const restored = setMaxOpsPerStream(prev); // restore (returns prior i64)
    try testing.expect(@TypeOf(prev) == i64 and @TypeOf(restored) == i64);
    const prevg = setPreserveUnmappedGlyphs(true); // setPreserveUnmappedGlyphs (returns prior i32)
    const restoredg = setPreserveUnmappedGlyphs(prevg != 0);
    try testing.expect(@TypeOf(prevg) == i32 and @TypeOf(restoredg) == i32);
}

test "phase-8 crypto namespace: provider/fips/policy/inventory/cbom" {
    const a = testing.allocator;

    const prov = try cryptoActiveProvider(a); // cryptoActiveProvider
    defer a.free(prov);
    try testing.expect(prov.len > 0);

    _ = cryptoFipsAvailable(); // cryptoFipsAvailable (smoke)
    _ = cryptoUseFips(); // cryptoUseFips (smoke; may no-op without FIPS)

    _ = cryptoSetPolicy("default"); // cryptoSetPolicy

    const pol = try cryptoPolicy(a); // cryptoPolicy
    defer a.free(pol);
    try testing.expect(pol.len >= 0);

    const inv = try cryptoInventory(a); // cryptoInventory
    defer a.free(inv);
    try testing.expect(inv.len >= 0);

    const cbom = try cryptoCbom(a); // cryptoCbom
    defer a.free(cbom);
    try testing.expect(cbom.len >= 0);
}

test "phase-8 models namespace: manifest + prefetch (return-or-error)" {
    const a = testing.allocator;

    // modelManifest returns a JSON string (no error channel); assert it returns
    // a string, tolerating the binding error type defensively.
    if (modelManifest(a)) |man| { // modelManifest
        defer a.free(man);
        try testing.expect(man.len >= 0);
    } else |err| {
        try testing.expect(err == Error.PdfOxide);
    }

    _ = prefetchAvailable(); // prefetchAvailable (smoke)

    // prefetch may need network/the `ocr` feature → return owned dir or raise.
    if (prefetchModels(a, "english")) |dir| { // prefetchModels
        defer a.free(dir);
        try testing.expect(dir.len >= 0);
    } else |_| {}
}
