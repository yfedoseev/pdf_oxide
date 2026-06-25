# tables_extraction — build a PDF containing a Markdown table, open it, and
# call extract_tables on page 0. A synthetic document may yield zero detected
# tables; the contract is only that the call succeeds and returns a vector.
# Exits non-zero on any assertion failure (CI smoke).
using PdfOxide

const MD = "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"

pdf = from_markdown(MD)
doc = open_from_bytes(to_bytes(pdf))

tables = extract_tables(doc, 0)

if !(tables isa Vector{Table})
    println("ASSERT FAILED: extract_tables did not return a Vector{Table}")
    exit(1)
end

println("table count: ", length(tables))
for (i, t) in enumerate(tables)
    println("table ", i - 1, ": ", t.row_count, " rows x ", t.col_count, " cols")
end

close!(doc)
println("TABLES OK")
