# basic_extraction — build a PDF from Markdown, then extract it back.
# Run in CI as a smoke example: `mix run examples/basic_extraction.exs`
{:ok, pdf} =
  PdfOxide.from_markdown("# Hello pdf_oxide\n\nThis is an **Elixir** binding smoke example.\n")

{:ok, bytes} = PdfOxide.to_bytes(pdf)
{:ok, doc} = PdfOxide.open_from_bytes(bytes)

{:ok, pages} = PdfOxide.page_count(doc)
IO.puts("pages:   #{pages}")
%{major: maj, minor: min} = PdfOxide.version(doc)
IO.puts("version: #{maj}.#{min}")
{:ok, text} = PdfOxide.extract_text(doc, 0)
IO.puts("--- text (page 0) ---")
IO.puts(text)
{:ok, md} = PdfOxide.to_markdown_all(doc)
IO.puts("--- markdown (all) ---")
IO.puts(md)
