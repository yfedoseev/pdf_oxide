# tables_extraction — build a PDF containing a Markdown table, open it, and call
# pdf_extract_tables on page 0. A synthetic document may yield zero detected
# tables; the contract is only that the call succeeds and returns a list.
# stop() (non-zero exit) on any assertion failure (CI smoke).
library(pdfoxide)

md <- "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"

pdf <- pdf_from_markdown(md)
doc <- pdf_open_from_bytes(pdf_to_bytes(pdf))

tables <- pdf_extract_tables(doc, 0)

if (!is.list(tables)) stop("ASSERT FAILED: pdf_extract_tables did not return a list")

cat("table count:", length(tables), "\n")
for (i in seq_along(tables)) {
  t <- tables[[i]]
  cat("table ", i - 1, ": ", t$row_count, " rows x ", t$col_count, " cols\n", sep = "")
}

pdf_close(doc)
cat("TABLES OK\n")
