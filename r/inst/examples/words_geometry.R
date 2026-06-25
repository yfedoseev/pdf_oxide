# words_geometry — build a PDF from Markdown, open it, extract positioned words
# from page 0 and assert the first word + its bbox. stop() (non-zero exit) on
# any assertion failure (CI smoke).
library(pdfoxide)

md <- "# Hello pdf_oxide\n\nThis is a **R** regression example.\n"

pdf <- pdf_from_markdown(md)
doc <- pdf_open_from_bytes(pdf_to_bytes(pdf))

words <- pdf_extract_words(doc, 0)
cat("word count:", length(words), "\n")

if (length(words) == 0) stop("ASSERT FAILED: no words extracted")

first <- words[[1]]
bb <- first$bbox
cat("first word:", first$text, "\n")
cat("first bbox: x=", bb$x, " y=", bb$y, " w=", bb$width, " h=", bb$height, "\n", sep = "")

if (!identical(first$text, "Hello")) {
  stop(sprintf("ASSERT FAILED: first word is '%s', expected 'Hello'", first$text))
}
if (!all(c("x", "y", "width", "height") %in% names(bb)) || !is.numeric(bb$width)) {
  stop("ASSERT FAILED: first word has no numeric bbox")
}

pdf_close(doc)
cat("WORDS OK\n")
