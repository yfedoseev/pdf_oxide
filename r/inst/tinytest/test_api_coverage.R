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
expect_true(grepl("<", pdf_to_html_all(doc)))          # pdf_to_html_all
expect_true(nchar(pdf_to_plain_text_all(doc)) > 0)     # pdf_to_plain_text_all
expect_true(is.logical(pdf_authenticate(doc, "")))     # pdf_authenticate (bool, no error)
expect_true(nchar(pdf_extract_structured_json(doc, 0)) > 0) # pdf_extract_structured_json

# ── Page model ────────────────────────────────────────────────────────────────
pg <- pdf_page(doc, 0)                                 # pdf_page (0-based)
expect_true(grepl("Alpha", pdf_page_text(pg)))         # pdf_page_text
expect_true(nchar(pdf_page_markdown(pg)) > 0)          # pdf_page_markdown
expect_true(grepl("<", pdf_page_html(pg)))             # pdf_page_html
expect_true(nchar(pdf_page_plain_text(pg)) > 0)        # pdf_page_plain_text


# ── Phase-1 element extraction ────────────────────────────────────────────────
words <- pdf_extract_words(doc, 0)                      # pdf_extract_words
expect_true(length(words) > 0)
expect_true(nchar(words[[1]]$text) > 0)
bb <- words[[1]]$bbox
expect_true(all(c("x", "y", "width", "height") %in% names(bb)))
expect_true(is.numeric(bb$width))
chars <- pdf_extract_chars(doc, 0)                      # pdf_extract_chars
expect_true(length(chars) > 0)
expect_true(is.integer(chars[[1]]$character) && chars[[1]]$character > 0)
lines <- pdf_extract_text_lines(doc, 0)                 # pdf_extract_text_lines
expect_true(length(lines) > 0)
expect_true(nchar(lines[[1]]$text) > 0)
expect_true(lines[[1]]$word_count >= 1)
tbls <- pdf_extract_tables(doc, 0)                      # pdf_extract_tables (may be empty)
expect_true(is.list(tbls))

# ── Phase-2 element extraction ────────────────────────────────────────────────
fonts <- pdf_embedded_fonts(doc, 0)                    # pdf_embedded_fonts (may be empty)
expect_true(is.list(fonts))
images <- pdf_embedded_images(doc, 0)                  # pdf_embedded_images (may be empty)
expect_true(is.list(images))
annots <- pdf_page_annotations(doc, 0)                 # pdf_page_annotations (may be empty)
expect_true(is.list(annots))
paths <- pdf_extract_paths(doc, 0)                     # pdf_extract_paths (may be empty)
expect_true(is.list(paths))

hits <- pdf_search(doc, 0, "Alpha", FALSE)             # pdf_search (non-empty)
expect_true(length(hits) > 0)
expect_true(grepl("Alpha", hits[[1]]$text))
expect_true(hits[[1]]$page >= 0)
hits_all <- pdf_search_all(doc, "Alpha", FALSE)        # pdf_search_all (non-empty)
expect_true(length(hits_all) > 0)
expect_true(grepl("Alpha", hits_all[[1]]$text))
expect_true(hits_all[[1]]$page >= 0)

# ── Phase-3 page rendering ────────────────────────────────────────────────────
img <- pdf_render_page(doc, 0)                          # pdf_render_page (PNG)
expect_inherits(img, "pdfoxide_rendered_image")
expect_true(img$width > 0)
expect_true(img$height > 0)
expect_true(length(img$data) > 0)
imgf <- tempfile(fileext = ".png")
pdf_rendered_image_save(img, imgf)                      # pdf_rendered_image_save
expect_true(file.exists(imgf)); unlink(imgf)
pdf_rendered_image_close(img)                           # pdf_rendered_image_close (idempotent)
imgz <- pdf_render_page_zoom(doc, 0, 1.5)               # pdf_render_page_zoom
expect_true(imgz$width > 0 && imgz$height > 0)
imgt <- pdf_render_page_thumbnail(doc, 0, 64L)          # pdf_render_page_thumbnail
expect_true(imgt$width > 0 && imgt$height > 0)

# ── close + open_with_password ────────────────────────────────────────────────
pdf_close(doc); expect_true(TRUE)              # pdf_close (idempotent)
pdf_close(doc)                                 # second close is a no-op
tmp3 <- tempfile(fileext = ".pdf")
pdf_save(pdf_from_markdown("# f\n\nx\n"), tmp3)
# open_with_password on a non-encrypted file still opens (no password needed),
# but the dedicated entry point must exist + be callable:
expect_true(is.function(pdf_open_with_password))
unlink(tmp3)

# ── DocumentEditor ────────────────────────────────────────────────────────────
ed <- pdf_editor_open_from_bytes(sample_pdf())          # pdf_editor_open_from_bytes
expect_true(pdf_editor_page_count(ed) >= 1)             # pdf_editor_page_count
expect_true(is.logical(pdf_editor_is_modified(ed)))     # pdf_editor_is_modified (bool)
ev <- pdf_editor_version(ed)                            # pdf_editor_version
expect_true(ev$major >= 1)
pdf_editor_rotate_all_pages(ed, 90L)                    # pdf_editor_rotate_all_pages
expect_true(pdf_editor_get_page_rotation(ed, 0) == 90)  # pdf_editor_get_page_rotation
pdf_editor_set_producer(ed, "x")                        # pdf_editor_set_producer
expect_true(is.character(pdf_editor_get_producer(ed)))  # pdf_editor_get_producer
edb <- pdf_editor_save_to_bytes(ed)                     # pdf_editor_save_to_bytes
expect_true(length(edb) > 0)
pdf_editor_close(ed); expect_true(TRUE)                 # pdf_editor_close (idempotent)
pdf_editor_close(ed)                                    # second close is a no-op

# ── Error path ────────────────────────────────────────────────────────────────
expect_error(pdf_open("/nonexistent/nope.pdf"))
