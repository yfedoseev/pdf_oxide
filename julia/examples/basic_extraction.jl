# basic_extraction — build a PDF from Markdown, then extract it back.
# Run in CI as a smoke example (no external fixture).
using PdfOxide

pdf = from_markdown("# Hello pdf_oxide\n\nThis is a **Julia** binding smoke example.\n")
doc = open_from_bytes(to_bytes(pdf))

println("pages:   ", page_count(doc))
v = version(doc)
println("version: ", v.major, ".", v.minor)
println("--- text (page 0) ---")
println(extract_text(doc, 0))
println("--- markdown (all) ---")
println(to_markdown_all(doc))
