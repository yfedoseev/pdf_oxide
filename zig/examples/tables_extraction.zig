// tables_extraction — build a PDF from Markdown with a table, extract tables.
//
// Shared regression scenario (mirrored across language bindings). The synthetic
// doc may yield zero tables; the contract is only that the call succeeds and
// returns a list (count >= 0). Exits non-zero on error; prints "TABLES OK".
const std = @import("std");
const pdf_oxide = @import("pdf_oxide");

pub fn main() !void {
    const a = std.heap.page_allocator;

    var pdf = try pdf_oxide.Pdf.fromMarkdown(
        "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n",
    );
    defer pdf.deinit();

    const bytes = try pdf.toBytes(a);
    defer a.free(bytes);

    var doc = try pdf_oxide.Document.openFromBytes(bytes);
    defer doc.deinit();

    const tables = try doc.extractTables(a, 0);
    defer pdf_oxide.Document.freeTables(a, tables);

    var buf: [4096]u8 = undefined;
    var fw = std.fs.File.stdout().writer(&buf);
    const stdout = &fw.interface;

    try stdout.print("table count: {d}\n", .{tables.len});

    for (tables, 0..) |tbl, t| {
        try stdout.print("table {d}: {d}x{d}\n", .{ t, tbl.rowCount, tbl.colCount });
        var r: i32 = 0;
        while (r < tbl.rowCount) : (r += 1) {
            var col: i32 = 0;
            while (col < tbl.colCount) : (col += 1) {
                try stdout.print("  cell({d},{d})=\"{s}\"\n", .{ r, col, tbl.cell(r, col) });
            }
        }
    }

    // extractTables returned without error -> the call yielded a valid list.
    try stdout.print("TABLES OK\n", .{});
    try stdout.flush();
}
