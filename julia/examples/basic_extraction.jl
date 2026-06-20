# basic_extraction — build a PDF from Markdown, then extract it back.
# Run in CI as a smoke example (no external fixture).
using PdfOxide

pdf = from_markdown("# Hello pdf_oxide\n\nThis is a **Julia** binding smoke example.\n")
doc = open_from_bytes(save_to_bytes(pdf))

println("pages:   ", page_count(doc))
v = version(doc)
println("version: ", v[1], ".", v[2])
println("--- text (page 0) ---")
println(extract_text(doc, 0))
println("--- markdown (all) ---")
println(to_markdown_all(doc))
