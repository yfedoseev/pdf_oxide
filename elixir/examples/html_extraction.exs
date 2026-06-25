# html_extraction — build a PDF from Markdown, open it, render the whole
# document to HTML. Exits non-zero on any assertion failure (CI smoke).
md = "# Hello pdf_oxide\n\nThis is a **Elixir** regression example.\n"

{:ok, pdf} = PdfOxide.from_markdown(md)
{:ok, bytes} = PdfOxide.to_bytes(pdf)
{:ok, doc} = PdfOxide.open_from_bytes(bytes)

{:ok, html} = PdfOxide.to_html_all(doc)
IO.puts("--- html (all) ---")
IO.puts(html)

unless String.contains?(html, "<") do
  IO.puts("ASSERT FAILED: html does not contain '<'")
  System.halt(1)
end

unless String.contains?(html, "pdf_oxide") do
  IO.puts("ASSERT FAILED: html does not contain 'pdf_oxide'")
  System.halt(1)
end

PdfOxide.close(doc)
IO.puts("HTML OK")
