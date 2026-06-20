# basic_extraction — build a PDF from Markdown, then extract it back.
# Run in CI as a smoke example (no external fixture).
library(pdfoxide)

pdf <- pdf_from_markdown(
  "# Hello pdf_oxide\n\nThis is an **R** binding smoke example.\n")
doc <- pdf_open_from_bytes(pdf_to_bytes(pdf))

cat("pages:  ", pdf_page_count(doc), "\n")
v <- pdf_version(doc)
cat("version:", paste(v$major, v$minor, sep = "."), "\n")
cat("--- text (page 0) ---\n", pdf_extract_text(doc, 0), "\n", sep = "")
cat("--- markdown (all) ---\n", pdf_to_markdown_all(doc), "\n", sep = "")
