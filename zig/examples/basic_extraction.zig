// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
const std = @import("std");
const pdf_oxide = @import("pdf_oxide");

pub fn main() !void {
    const a = std.heap.page_allocator;

    var pdf = try pdf_oxide.Pdf.fromMarkdown(
        "# Hello pdf_oxide\n\nThis is a **Zig** binding smoke example.\n",
    );
    defer pdf.deinit();

    const bytes = try pdf.toBytes(a);
    defer a.free(bytes);

    var doc = try pdf_oxide.Document.openFromBytes(bytes);
    defer doc.deinit();

    // Zig 0.15 "Writergate": stdout via a buffered File.Writer interface.
    var buf: [4096]u8 = undefined;
    var fw = std.fs.File.stdout().writer(&buf);
    const stdout = &fw.interface;

    try stdout.print("pages:   {d}\n", .{try doc.pageCount()});
    const v = doc.version();
    try stdout.print("version: {d}.{d}\n", .{ v.major, v.minor });

    const text = try doc.extractText(a, 0);
    defer a.free(text);
    try stdout.print("--- text (page 0) ---\n{s}\n", .{text});

    const md = try doc.toMarkdownAll(a);
    defer a.free(md);
    try stdout.print("--- markdown (all) ---\n{s}\n", .{md});
    try stdout.flush();
}
