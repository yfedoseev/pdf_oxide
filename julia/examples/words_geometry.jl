# words_geometry — build a PDF from Markdown, open it, extract positioned
# words from page 0 and assert the first word + its bbox. Exits non-zero on
# any assertion failure (CI smoke).
using PdfOxide

const MD = "# Hello pdf_oxide\n\nThis is a **Julia** regression example.\n"

pdf = from_markdown(MD)
doc = open_from_bytes(to_bytes(pdf))

words = extract_words(doc, 0)
println("word count: ", length(words))

if isempty(words)
    println("ASSERT FAILED: no words extracted")
    exit(1)
end

first = words[1]
bb = first.bbox
println("first word: ", repr(first.text))
println("first bbox: x=", bb.x, " y=", bb.y, " w=", bb.width, " h=", bb.height)

if first.text != "Hello"
    println("ASSERT FAILED: first word is ", repr(first.text), ", expected \"Hello\"")
    exit(1)
end
if !(bb.width >= 0)
    println("ASSERT FAILED: first word has no valid bbox")
    exit(1)
end

close!(doc)
println("WORDS OK")
