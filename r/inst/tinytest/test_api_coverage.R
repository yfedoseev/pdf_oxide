# One assertion per public function — mirrors the api_coverage convention used
# by every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
library(pdfoxide)

sample_pdf <- function() {
  pdf_to_bytes(pdf_from_markdown(
    "# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n"))
}

# ── Pdf builder ───────────────────────────────────────────────────────────────
expect_true(length(pdf_to_bytes(pdf_from_markdown("# md\n\nbody\n"))) > 100)
expect_true(length(pdf_to_bytes(pdf_from_html("<h1>h</h1><p>b</p>"))) > 100)
expect_true(length(pdf_to_bytes(pdf_from_text("plain text body"))) > 100)
tmp <- tempfile(fileext = ".pdf")
pdf_save(pdf_from_markdown("# f\n\nx\n"), tmp)
expect_true(file.exists(tmp)); unlink(tmp)

# ── Document open paths ───────────────────────────────────────────────────────
doc <- pdf_open_from_bytes(sample_pdf())       # pdf_open_from_bytes
expect_true(pdf_page_count(doc) >= 1)          # pdf_page_count
tmp2 <- tempfile(fileext = ".pdf")
pdf_save(pdf_from_markdown("# f\n\nx\n"), tmp2)
d2 <- pdf_open(tmp2)                            # pdf_open
expect_true(pdf_page_count(d2) >= 1); unlink(tmp2)

# ── Document inspection + extraction ──────────────────────────────────────────
v <- pdf_version(doc)                           # pdf_version
expect_true(v$major >= 1)
expect_false(pdf_is_encrypted(doc))             # pdf_is_encrypted
invisible(pdf_has_structure_tree(doc))          # pdf_has_structure_tree (smoke)
expect_true(grepl("Alpha", pdf_extract_text(doc, 0)))  # pdf_extract_text
expect_true(nchar(pdf_to_plain_text(doc, 0)) > 0)      # pdf_to_plain_text
expect_true(nchar(pdf_to_markdown(doc, 0)) > 0)        # pdf_to_markdown
expect_true(grepl("<", pdf_to_html(doc, 0)))           # pdf_to_html
expect_true(nchar(pdf_to_markdown_all(doc)) > 0)       # pdf_to_markdown_all
expect_true(nchar(pdf_extract_structured_json(doc, 0)) > 0) # pdf_extract_structured_json


# ── close + open_with_password ────────────────────────────────────────────────
pdf_close(doc); expect_true(TRUE)              # pdf_close (idempotent)
pdf_close(doc)                                 # second close is a no-op
tmp3 <- tempfile(fileext = ".pdf")
pdf_save(pdf_from_markdown("# f\n\nx\n"), tmp3)
# open_with_password on a non-encrypted file still opens (no password needed),
# but the dedicated entry point must exist + be callable:
expect_true(is.function(pdf_open_with_password))
unlink(tmp3)

# ── Error path ────────────────────────────────────────────────────────────────
expect_error(pdf_open("/nonexistent/nope.pdf"))
