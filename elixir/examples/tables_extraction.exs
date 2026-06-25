# tables_extraction — build a PDF containing a Markdown table, open it, and
# call extract_tables on page 0. A synthetic document may yield zero detected
# tables; the contract is only that the call succeeds and returns a list.
# Exits non-zero on any assertion failure (CI smoke).
md = "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"

{:ok, pdf} = PdfOxide.from_markdown(md)
{:ok, bytes} = PdfOxide.to_bytes(pdf)
{:ok, doc} = PdfOxide.open_from_bytes(bytes)

{:ok, tables} = PdfOxide.extract_tables(doc, 0)

unless is_list(tables) do
  IO.puts("ASSERT FAILED: extract_tables did not return a list")
  System.halt(1)
end

IO.puts("table count: #{length(tables)}")

for {t, i} <- Enum.with_index(tables) do
  IO.puts("table #{i}: #{t.row_count} rows x #{t.col_count} cols")
end

PdfOxide.close(doc)
IO.puts("TABLES OK")
