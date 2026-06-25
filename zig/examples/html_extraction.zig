// html_extraction — build a PDF from Markdown, open it, render HTML.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "HTML OK" on success.
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

    const html = try doc.toHtmlAll(a);
    defer a.free(html);

    var buf: [4096]u8 = undefined;
    var fw = std.fs.File.stdout().writer(&buf);
    const stdout = &fw.interface;

    try stdout.print("--- html (all) ---\n{s}\n", .{html});

    if (std.mem.indexOfScalar(u8, html, '<') == null) {
        try stdout.print("assertion failed: html does not contain '<'\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }
    if (std.mem.indexOf(u8, html, "pdf_oxide") == null) {
        try stdout.print("assertion failed: html does not contain 'pdf_oxide'\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    try stdout.print("HTML OK\n", .{});
    try stdout.flush();
}
