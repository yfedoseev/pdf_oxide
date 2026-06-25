# html_extraction — build a PDF from Markdown, open it, render the whole
# document to HTML. stop() (non-zero exit) on any assertion failure (CI smoke).
library(pdfoxide)

md <- "# Hello pdf_oxide\n\nThis is a **R** regression example.\n"

pdf <- pdf_from_markdown(md)
doc <- pdf_open_from_bytes(pdf_to_bytes(pdf))

html <- pdf_to_html_all(doc)
cat("--- html (all) ---\n", html, "\n", sep = "")

if (!grepl("<", html, fixed = TRUE)) stop("ASSERT FAILED: html does not contain '<'")
if (!grepl("pdf_oxide", html, fixed = TRUE)) {
  stop("ASSERT FAILED: html does not contain 'pdf_oxide'")
}

pdf_close(doc)
cat("HTML OK\n")
