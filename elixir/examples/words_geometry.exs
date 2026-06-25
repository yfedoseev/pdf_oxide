# words_geometry — build a PDF from Markdown, open it, extract positioned
# words from page 0 and assert the first word + its bbox. Exits non-zero on
# any assertion failure (CI smoke).
md = "# Hello pdf_oxide\n\nThis is a **Elixir** regression example.\n"

{:ok, pdf} = PdfOxide.from_markdown(md)
{:ok, bytes} = PdfOxide.to_bytes(pdf)
{:ok, doc} = PdfOxide.open_from_bytes(bytes)

{:ok, words} = PdfOxide.extract_words(doc, 0)
IO.puts("word count: #{length(words)}")

if words == [] do
  IO.puts("ASSERT FAILED: no words extracted")
  System.halt(1)
end

first = hd(words)
bbox = first.bbox
IO.puts("first word: #{inspect(first.text)}")
IO.puts("first bbox: x=#{bbox.x} y=#{bbox.y} w=#{bbox.width} h=#{bbox.height}")

unless first.text == "Hello" do
  IO.puts("ASSERT FAILED: first word is #{inspect(first.text)}, expected \"Hello\"")
  System.halt(1)
end

unless is_number(bbox.x) and is_number(bbox.width) do
  IO.puts("ASSERT FAILED: first word has no numeric bbox")
  System.halt(1)
end

PdfOxide.close(doc)
IO.puts("WORDS OK")
