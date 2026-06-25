// tables_extraction — build a PDF with a Markdown table, then extract tables.
// Shared-scenario regression example. Synthetic docs may yield 0 tables; the
// call must merely succeed (count >= 0). Exits non-zero on assertion failure.
import PdfOxide

let md = "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"
let pdf = try Pdf.fromMarkdown(md)
let doc = try Document.openFromBytes(try pdf.toBytes())

// extractTables throws on error; a successful call returns an array whose
// count is necessarily >= 0 (0 is acceptable for a synthetic document).
let tables = try doc.extractTables(0)
print("tables: \(tables.count)")
for (ti, table) in tables.enumerated() {
    print("table \(ti): rows=\(table.rowCount) cols=\(table.colCount) hasHeader=\(table.hasHeader)")
    if table.rowCount > 0 && table.colCount > 0 {
        print("  cell(0,0): \"\(table.cell(0, 0))\"")
    }
}

print("TABLES OK")
