// words_geometry — build a PDF from Markdown, open it, extract word geometry.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "WORDS OK" on success.
const std = @import("std");
const pdf_oxide = @import("pdf_oxide");

pub fn main() !void {
    const a = std.heap.page_allocator;

    var pdf = try pdf_oxide.Pdf.fromMarkdown(
        "# Hello pdf_oxide\n\nThis is a **Zig** regression example.\n",
    );
    defer pdf.deinit();

    const bytes = try pdf.toBytes(a);
    defer a.free(bytes);

    var doc = try pdf_oxide.Document.openFromBytes(bytes);
    defer doc.deinit();

    const words = try doc.extractWords(a, 0);
    defer pdf_oxide.Document.freeWords(a, words);

    var buf: [4096]u8 = undefined;
    var fw = std.fs.File.stdout().writer(&buf);
    const stdout = &fw.interface;

    try stdout.print("word count: {d}\n", .{words.len});

    if (words.len == 0) {
        try stdout.print("assertion failed: no words extracted\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    const first = words[0];
    try stdout.print("first word: \"{s}\"  bbox=({d}, {d}, {d}, {d})\n", .{
        first.text, first.bbox.x, first.bbox.y, first.bbox.width, first.bbox.height,
    });

    if (!std.mem.eql(u8, first.text, "Hello")) {
        try stdout.print("assertion failed: first word is not \"Hello\"\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }
    if (!(first.bbox.width > 0.0 and first.bbox.height > 0.0)) {
        try stdout.print("assertion failed: first word has no bbox\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    try stdout.print("WORDS OK\n", .{});
    try stdout.flush();
}
