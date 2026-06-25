# html_extraction — build a PDF from Markdown, open it, render the whole
# document to HTML. Exits non-zero on any assertion failure (CI smoke).
using PdfOxide

const MD = "# Hello pdf_oxide\n\nThis is a **Julia** regression example.\n"

pdf = from_markdown(MD)
doc = open_from_bytes(to_bytes(pdf))

html = to_html_all(doc)
println("--- html (all) ---")
println(html)

if !occursin("<", html)
    println("ASSERT FAILED: html does not contain '<'")
    exit(1)
end
if !occursin("pdf_oxide", html)
    println("ASSERT FAILED: html does not contain 'pdf_oxide'")
    exit(1)
end

close!(doc)
println("HTML OK")
