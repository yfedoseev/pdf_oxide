#' pdf_oxide — idiomatic R bindings for fast PDF text/Markdown/HTML extraction.
#'
#' Wraps the pdf_oxide C ABI. Handles are external pointers freed by the GC.
#' Page indices are 0-based to match the underlying engine.
#'
#' @useDynLib pdfoxide, .registration = TRUE, .fixes = "C_"
#' @keywords internal
"_PACKAGE"

# ── Pdf builder ───────────────────────────────────────────────────────────────

#' Build a PDF from Markdown / HTML / plain text.
#' @param markdown,html,text Source string.
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_markdown <- function(markdown) {
  structure(.Call(C_r_pdf_from_markdown, markdown), class = "pdfoxide_pdf")
}
#' @rdname pdf_from_markdown
#' @export
pdf_from_html <- function(html) {
  structure(.Call(C_r_pdf_from_html, html), class = "pdfoxide_pdf")
}
#' @rdname pdf_from_markdown
#' @export
pdf_from_text <- function(text) {
  structure(.Call(C_r_pdf_from_text, text), class = "pdfoxide_pdf")
}

#' Save a built PDF to a path.
#' @param pdf A `pdfoxide_pdf`.
#' @param path Output path.
#' @export
pdf_save <- function(pdf, path) {
  invisible(.Call(C_r_pdf_save, pdf, path))
}

#' Serialize a built PDF to a raw vector.
#' @param pdf A `pdfoxide_pdf`.
#' @return A `raw` vector.
#' @export
pdf_to_bytes <- function(pdf) {
  .Call(C_r_pdf_save_to_bytes, pdf)
}

# ── Document ──────────────────────────────────────────────────────────────────

#' Open a PDF document for extraction.
#' @param path Path to a PDF.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open <- function(path) {
  structure(.Call(C_r_doc_open, path), class = "pdfoxide_document")
}

#' Open a password-protected PDF document.
#' @param path Path to a PDF.
#' @param password The document password.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open_with_password <- function(path, password) {
  structure(.Call(C_r_doc_open_with_password, path, password),
            class = "pdfoxide_document")
}

#' Open a PDF document from a raw vector.
#' @param bytes A `raw` vector.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open_from_bytes <- function(bytes) {
  structure(.Call(C_r_doc_open_from_bytes, bytes), class = "pdfoxide_document")
}

#' Number of pages.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_page_count <- function(doc) .Call(C_r_doc_page_count, doc)

#' PDF version as a named list `list(major=, minor=)`.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_version <- function(doc) {
  v <- .Call(C_r_doc_version, doc)
  list(major = v[1], minor = v[2])
}

#' Whether the document is encrypted.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_is_encrypted <- function(doc) .Call(C_r_doc_is_encrypted, doc)

#' Whether the document has a logical structure tree (tagged PDF).
#' @param doc A `pdfoxide_document`.
#' @export
pdf_has_structure_tree <- function(doc) .Call(C_r_doc_has_structure_tree, doc)

#' Extract reading-order text for one (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_extract_text <- function(doc, page) {
  .Call(C_r_doc_extract_text, doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_plain_text <- function(doc, page) {
  .Call(C_r_doc_to_plain_text, doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_markdown <- function(doc, page) {
  .Call(C_r_doc_to_markdown, doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_html <- function(doc, page) {
  .Call(C_r_doc_to_html, doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_extract_structured_json <- function(doc, page) {
  .Call(C_r_doc_extract_structured_json, doc, as.integer(page))
}

#' Markdown for the whole document.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_to_markdown_all <- function(doc) .Call(C_r_doc_to_markdown_all, doc)

#' HTML for the whole document.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_to_html_all <- function(doc) .Call(C_r_doc_to_html_all, doc)

#' Plain text for the whole document.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_to_plain_text_all <- function(doc) .Call(C_r_doc_to_plain_text_all, doc)

#' Authenticate an encrypted document with a password.
#'
#' Returns `TRUE` if the password unlocks the document and `FALSE` for a wrong
#' password; raises only on a real C-ABI failure.
#' @param doc A `pdfoxide_document`.
#' @param password The document password.
#' @return A logical scalar.
#' @export
pdf_authenticate <- function(doc, password) {
  .Call(C_r_doc_authenticate, doc, password)
}

# ── Phase-1 element extraction ────────────────────────────────────────────────

#' Extract positioned characters for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Char` records, each `list(character=, bbox=, font_name=,
#'   font_size=)` where `bbox` is `list(x=, y=, width=, height=)` and
#'   `character` is the Unicode codepoint as an integer.
#' @export
pdf_extract_chars <- function(doc, page) {
  .Call(C_r_doc_extract_chars, doc, as.integer(page))
}

#' Extract positioned words for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Word` records, each `list(text=, bbox=, font_name=,
#'   font_size=, bold=)`.
#' @export
pdf_extract_words <- function(doc, page) {
  .Call(C_r_doc_extract_words, doc, as.integer(page))
}

#' Extract reading-order text lines for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `TextLine` records, each `list(text=, bbox=, word_count=)`.
#' @export
pdf_extract_text_lines <- function(doc, page) {
  .Call(C_r_doc_extract_text_lines, doc, as.integer(page))
}

#' Extract detected tables for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Table` records, each `list(row_count=, col_count=,
#'   has_header=, cells=)` where `cells` is a `row_count` x `col_count` character
#'   matrix; index a cell with `tbl$cells[row, col]` (1-based).
#' @export
pdf_extract_tables <- function(doc, page) {
  .Call(C_r_doc_extract_tables, doc, as.integer(page))
}

# ── Phase-2 element extraction ────────────────────────────────────────────────

#' Extract embedded fonts for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Font` records, each `list(name=, type=, encoding=,
#'   embedded=, subset=)`.
#' @export
pdf_embedded_fonts <- function(doc, page) {
  .Call(C_r_doc_embedded_fonts, doc, as.integer(page))
}

#' Extract embedded images for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Image` records, each `list(width=, height=,
#'   bits_per_component=, format=, colorspace=, data=)` where `data` is a `raw`
#'   vector of the image bytes.
#' @export
pdf_embedded_images <- function(doc, page) {
  .Call(C_r_doc_embedded_images, doc, as.integer(page))
}

#' Extract annotations for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Annotation` records, each `list(type=, subtype=, content=,
#'   author=, rect=, border_width=)` where `rect` is `list(x=, y=, width=,
#'   height=)`.
#' @export
pdf_page_annotations <- function(doc, page) {
  .Call(C_r_doc_page_annotations, doc, as.integer(page))
}

#' Extract vector paths for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @return A list of `Path` records, each `list(bbox=, stroke_width=, has_stroke=,
#'   has_fill=, operation_count=)`.
#' @export
pdf_extract_paths <- function(doc, page) {
  .Call(C_r_doc_extract_paths, doc, as.integer(page))
}

#' Search a single (0-based) page for a term.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @param term The search term.
#' @param case_sensitive Whether to match case.
#' @return A list of `SearchResult` records, each `list(text=, page=, bbox=)`.
#' @export
pdf_search <- function(doc, page, term, case_sensitive = FALSE) {
  .Call(C_r_doc_search, doc, as.integer(page), term,
        isTRUE(case_sensitive))
}

#' Search the whole document for a term.
#'
#' @param doc A `pdfoxide_document`.
#' @param term The search term.
#' @param case_sensitive Whether to match case.
#' @return A list of `SearchResult` records, each `list(text=, page=, bbox=)`.
#' @export
pdf_search_all <- function(doc, term, case_sensitive = FALSE) {
  .Call(C_r_doc_search_all, doc, term, isTRUE(case_sensitive))
}

# ── Phase-3 page rendering ────────────────────────────────────────────────────

#' Render a (0-based) page to a raster image.
#'
#' `format` is an integer image format (`0` = PNG, the default).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index (required).
#' @param zoom Scale factor (`render_page_zoom`).
#' @param size Largest side in
#'   pixels (`render_page_thumbnail`).
#' @param format Image format (`0` = PNG).
#' @return A `pdfoxide_rendered_image` with elements `width`, `height` and `data`
#'   (a `raw` vector of the encoded image bytes), plus a `save(path)` method.
#' @export
pdf_render_page <- function(doc, page, format = 0L) {
  img <- .Call(C_r_doc_render_page, doc, as.integer(page), as.integer(format))
  new_rendered_image(img)
}
#' @rdname pdf_render_page
#' @export
pdf_render_page_zoom <- function(doc, page, zoom, format = 0L) {
  img <- .Call(C_r_doc_render_page_zoom, doc, as.integer(page),
               as.double(zoom), as.integer(format))
  new_rendered_image(img)
}
#' @rdname pdf_render_page
#' @export
pdf_render_page_thumbnail <- function(doc, page, size, format = 0L) {
  img <- .Call(C_r_doc_render_page_thumbnail, doc, as.integer(page),
               as.integer(size), as.integer(format))
  new_rendered_image(img)
}

# Build the RenderedImage model from a live FfiRenderedImage external pointer:
# read width/height/data eagerly, keep the handle so `save(path)` can use it.
new_rendered_image <- function(handle) {
  structure(
    list(
      handle = handle,
      width  = .Call(C_r_rendered_image_width, handle),
      height = .Call(C_r_rendered_image_height, handle),
      data   = .Call(C_r_rendered_image_data, handle)
    ),
    class = "pdfoxide_rendered_image")
}

#' Save a rendered image to a file path.
#'
#' Writes the encoded image (format chosen at render time) using the live native
#' handle.
#' @param image A `pdfoxide_rendered_image`.
#' @param path Output file path.
#' @export
pdf_rendered_image_save <- function(image, path) {
  if (!inherits(image, "pdfoxide_rendered_image"))
    stop("pdf_rendered_image_save: expected a pdfoxide_rendered_image")
  invisible(.Call(C_r_rendered_image_save, image$handle, path))
}

#' Free a rendered image's native handle now (idempotent).
#' @param image A `pdfoxide_rendered_image`.
#' @export
pdf_rendered_image_close <- function(image) {
  if (!inherits(image, "pdfoxide_rendered_image"))
    stop("pdf_rendered_image_close: expected a pdfoxide_rendered_image")
  invisible(.Call(C_r_rendered_image_close, image$handle))
}

# ── Page ────────────────────────────────────────────────────────────────────

#' A single (0-based) page of a document.
#'
#' Holds a reference to its parent `pdfoxide_document` so the document is kept
#' alive for as long as the page is reachable; the page must not outlive it.
#' @param doc A `pdfoxide_document`.
#' @param index 0-based page index (required).
#' @return A `pdfoxide_page`.
#' @export
pdf_page <- function(doc, index) {
  if (!inherits(doc, "pdfoxide_document"))
    stop("pdf_page: expected a pdfoxide_document")
  structure(list(doc = doc, index = as.integer(index)),
            class = "pdfoxide_page")
}

#' Extract reading-order text for a page.
#' @param page A `pdfoxide_page`.
#' @export
pdf_page_text <- function(page) {
  .Call(C_r_doc_extract_text, page$doc, page$index)
}
#' @rdname pdf_page_text
#' @export
pdf_page_markdown <- function(page) {
  .Call(C_r_doc_to_markdown, page$doc, page$index)
}
#' @rdname pdf_page_text
#' @export
pdf_page_html <- function(page) {
  .Call(C_r_doc_to_html, page$doc, page$index)
}
#' @rdname pdf_page_text
#' @export
pdf_page_plain_text <- function(page) {
  .Call(C_r_doc_to_plain_text, page$doc, page$index)
}

#' Close a document or built PDF, freeing the native handle now (idempotent).
#' @param x A `pdfoxide_document` or `pdfoxide_pdf` handle.
#' @export
pdf_close <- function(x) {
  if (inherits(x, "pdfoxide_document")) {
    invisible(.Call(C_r_doc_close, x))
  } else if (inherits(x, "pdfoxide_pdf")) {
    invisible(.Call(C_r_pdf_close, x))
  } else {
    stop("pdf_close: expected a pdfoxide_document or pdfoxide_pdf")
  }
}

# ── DocumentEditor ────────────────────────────────────────────────────────────
# Editing handle mirroring the pdfoxide_document/pdfoxide_pdf pattern: the owned
# native DocumentEditor* is an external pointer freed by the GC finalizer (or now
# via pdf_editor_close). Page indices are 0-based. Free `pdf_editor_*` functions.

#' Open a PDF for editing.
#' @param path Path to a PDF.
#' @return A `pdfoxide_editor` handle.
#' @export
pdf_editor_open <- function(path) {
  structure(.Call(C_r_editor_open, path), class = "pdfoxide_editor")
}

#' Open a PDF for editing from a raw vector.
#' @param bytes A `raw` vector.
#' @return A `pdfoxide_editor` handle.
#' @export
pdf_editor_open_from_bytes <- function(bytes) {
  structure(.Call(C_r_editor_open_from_bytes, bytes), class = "pdfoxide_editor")
}

#' Number of pages in the editor.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_page_count <- function(editor) .Call(C_r_editor_page_count, editor)

#' PDF version as a named list `list(major=, minor=)`.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_version <- function(editor) {
  v <- .Call(C_r_editor_version, editor)
  list(major = v[1], minor = v[2])
}

#' Whether the editor has unsaved modifications.
#' @param editor A `pdfoxide_editor`.
#' @return A logical scalar.
#' @export
pdf_editor_is_modified <- function(editor) {
  .Call(C_r_editor_is_modified, editor)
}

#' Source path of the editor.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_source_path <- function(editor) {
  .Call(C_r_editor_source_path, editor)
}

#' Get / set the document producer (`/Info.Producer`).
#' @param editor A `pdfoxide_editor`.
#' @param value New producer string.
#' @export
pdf_editor_get_producer <- function(editor) {
  .Call(C_r_editor_get_producer, editor)
}
#' @rdname pdf_editor_get_producer
#' @export
pdf_editor_set_producer <- function(editor, value) {
  invisible(.Call(C_r_editor_set_producer, editor, value))
}

#' Get / set the document creation date (`/Info.CreationDate`, raw PDF date).
#' @param editor A `pdfoxide_editor`.
#' @param value Raw PDF date string.
#' @export
pdf_editor_get_creation_date <- function(editor) {
  .Call(C_r_editor_get_creation_date, editor)
}
#' @rdname pdf_editor_get_creation_date
#' @export
pdf_editor_set_creation_date <- function(editor, value) {
  invisible(.Call(C_r_editor_set_creation_date, editor, value))
}

#' Delete a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_delete_page <- function(editor, page) {
  invisible(.Call(C_r_editor_delete_page, editor, as.integer(page)))
}

#' Move a page from one (0-based) index to another.
#' @param editor A `pdfoxide_editor`.
#' @param from,to 0-based page indices.
#' @export
pdf_editor_move_page <- function(editor, from, to) {
  invisible(.Call(C_r_editor_move_page, editor, as.integer(from), as.integer(to)))
}

#' Rotate a single (0-based) page by `degrees` (additive).
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param degrees Degrees to rotate.
#' @export
pdf_editor_rotate_page_by <- function(editor, page, degrees) {
  invisible(.Call(C_r_editor_rotate_page_by, editor, as.integer(page),
                  as.integer(degrees)))
}

#' Rotate all pages by `degrees` (additive).
#' @param editor A `pdfoxide_editor`.
#' @param degrees Degrees to rotate.
#' @export
pdf_editor_rotate_all_pages <- function(editor, degrees) {
  invisible(.Call(C_r_editor_rotate_all_pages, editor, as.integer(degrees)))
}

#' Get / set the absolute rotation of a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param degrees Absolute rotation in degrees.
#' @export
pdf_editor_get_page_rotation <- function(editor, page) {
  .Call(C_r_editor_get_page_rotation, editor, as.integer(page))
}
#' @rdname pdf_editor_get_page_rotation
#' @export
pdf_editor_set_page_rotation <- function(editor, page, degrees) {
  invisible(.Call(C_r_editor_set_page_rotation, editor, as.integer(page),
                  as.integer(degrees)))
}

#' Crop all pages by the given margins (left, right, top, bottom).
#' @param editor A `pdfoxide_editor`.
#' @param left,right,top,bottom Margins.
#' @export
pdf_editor_crop_margins <- function(editor, left, right, top, bottom) {
  invisible(.Call(C_r_editor_crop_margins, editor, as.double(left),
                  as.double(right), as.double(top), as.double(bottom)))
}

#' Get / set the CropBox for a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param x,y,w,h Box coordinates and size.
#' @return For the getter, a Bbox `list(x=, y=, width=, height=)`.
#' @export
pdf_editor_get_page_crop_box <- function(editor, page) {
  .Call(C_r_editor_get_page_crop_box, editor, as.integer(page))
}
#' @rdname pdf_editor_get_page_crop_box
#' @export
pdf_editor_set_page_crop_box <- function(editor, page, x, y, w, h) {
  invisible(.Call(C_r_editor_set_page_crop_box, editor, as.integer(page),
                  as.double(x), as.double(y), as.double(w), as.double(h)))
}

#' Get / set the MediaBox for a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param x,y,w,h Box coordinates and size.
#' @return For the getter, a Bbox `list(x=, y=, width=, height=)`.
#' @export
pdf_editor_get_page_media_box <- function(editor, page) {
  .Call(C_r_editor_get_page_media_box, editor, as.integer(page))
}
#' @rdname pdf_editor_get_page_media_box
#' @export
pdf_editor_set_page_media_box <- function(editor, page, x, y, w, h) {
  invisible(.Call(C_r_editor_set_page_media_box, editor, as.integer(page),
                  as.double(x), as.double(y), as.double(w), as.double(h)))
}

#' Apply all pending redactions across the whole document (burn them in).
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_apply_all_redactions <- function(editor) {
  invisible(.Call(C_r_editor_apply_all_redactions, editor))
}

#' Apply pending redactions on a single (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_apply_page_redactions <- function(editor, page) {
  invisible(.Call(C_r_editor_apply_page_redactions, editor, as.integer(page)))
}

#' Whether a (0-based) page is marked for redaction.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @return A logical scalar.
#' @export
pdf_editor_is_page_marked_for_redaction <- function(editor, page) {
  .Call(C_r_editor_is_page_marked_for_redaction, editor, as.integer(page))
}

#' Remove the redaction mark from a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_unmark_page_for_redaction <- function(editor, page) {
  invisible(.Call(C_r_editor_unmark_page_for_redaction, editor, as.integer(page)))
}

#' Erase a single rectangular region on a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param x,y,w,h Rectangle in page user-space.
#' @export
pdf_editor_erase_region <- function(editor, page, x, y, w, h) {
  invisible(.Call(C_r_editor_erase_region, editor, as.integer(page),
                  as.double(x), as.double(y), as.double(w), as.double(h)))
}

#' Erase multiple rectangular regions on a (0-based) page.
#'
#' `rects` is a flat numeric vector of `[x, y, w, h]` quads (length 4*N) or a
#' 4-column matrix/data.frame of rectangles.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param rects Flat numeric vector or 4-column matrix of rectangles.
#' @export
pdf_editor_erase_regions <- function(editor, page, rects) {
  if (is.matrix(rects) || is.data.frame(rects)) {
    rects <- as.double(t(as.matrix(rects)))
  } else {
    rects <- as.double(rects)
  }
  invisible(.Call(C_r_editor_erase_regions, editor, as.integer(page), rects))
}

#' Clear all pending erase-region entries for a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_clear_erase_regions <- function(editor, page) {
  invisible(.Call(C_r_editor_clear_erase_regions, editor, as.integer(page)))
}

#' Flatten all forms in the document (bake form values into page content).
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_flatten_forms <- function(editor) {
  invisible(.Call(C_r_editor_flatten_forms, editor))
}

#' Flatten forms on a single (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_flatten_forms_on_page <- function(editor, page) {
  invisible(.Call(C_r_editor_flatten_forms_on_page, editor, as.integer(page)))
}

#' Set a form field value.
#' @param editor A `pdfoxide_editor`.
#' @param name Field name.
#' @param value Value.
#' @export
pdf_editor_set_form_field_value <- function(editor, name, value) {
  invisible(.Call(C_r_editor_set_form_field_value, editor, name, value))
}

#' Flatten annotations on a single (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_flatten_annotations <- function(editor, page) {
  invisible(.Call(C_r_editor_flatten_annotations, editor, as.integer(page)))
}

#' Flatten all annotations across the document.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_flatten_all_annotations <- function(editor) {
  invisible(.Call(C_r_editor_flatten_all_annotations, editor))
}

#' Number of warnings from the last form-flattening save.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_flatten_warnings_count <- function(editor) {
  .Call(C_r_editor_flatten_warnings_count, editor)
}

#' Get the `index`-th flatten warning.
#' @param editor A `pdfoxide_editor`.
#' @param index 0-based warning index.
#' @export
pdf_editor_flatten_warning <- function(editor, index) {
  .Call(C_r_editor_flatten_warning, editor, as.integer(index))
}

#' Whether a (0-based) page is marked for annotation-flatten.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @return A logical scalar.
#' @export
pdf_editor_is_page_marked_for_flatten <- function(editor, page) {
  .Call(C_r_editor_is_page_marked_for_flatten, editor, as.integer(page))
}

#' Remove the flatten mark from a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_editor_unmark_page_for_flatten <- function(editor, page) {
  invisible(.Call(C_r_editor_unmark_page_for_flatten, editor, as.integer(page)))
}

#' Merge pages from another PDF file into this document.
#' @param editor A `pdfoxide_editor`.
#' @param source_path Path to the source PDF.
#' @export
pdf_editor_merge_from <- function(editor, source_path) {
  invisible(.Call(C_r_editor_merge_from, editor, source_path))
}

#' Merge pages from an in-memory PDF (raw vector) into this document.
#' @param editor A `pdfoxide_editor`.
#' @param bytes A `raw` vector.
#' @export
pdf_editor_merge_from_bytes <- function(editor, bytes) {
  invisible(.Call(C_r_editor_merge_from_bytes, editor, bytes))
}

#' Convert the document to PDF/A in place.
#'
#' `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
#' @param editor A `pdfoxide_editor`.
#' @param level PDF/A conformance level.
#' @export
pdf_editor_convert_to_pdf_a <- function(editor, level) {
  invisible(.Call(C_r_editor_convert_to_pdf_a, editor, as.integer(level)))
}

#' Embed a file attachment into the document.
#' @param editor A `pdfoxide_editor`.
#' @param name Attachment name.
#' @param bytes A `raw` vector of the file contents.
#' @export
pdf_editor_embed_file <- function(editor, name, bytes) {
  invisible(.Call(C_r_editor_embed_file, editor, name, bytes))
}

#' Extract a subset of (0-based) pages to a new in-memory PDF.
#' @param editor A `pdfoxide_editor`.
#' @param pages Integer vector of 0-based pages.
#' @return A `raw` vector.
#' @export
pdf_editor_extract_pages_to_bytes <- function(editor, pages) {
  .Call(C_r_editor_extract_pages_to_bytes, editor, as.integer(pages))
}

#' Save the edited document to a path.
#' @param editor A `pdfoxide_editor`.
#' @param path Output path.
#' @export
pdf_editor_save <- function(editor, path) {
  invisible(.Call(C_r_editor_save, editor, path))
}

#' Serialize the edited document to a raw vector.
#' @param editor A `pdfoxide_editor`.
#' @return A `raw` vector.
#' @export
pdf_editor_save_to_bytes <- function(editor) {
  .Call(C_r_editor_save_to_bytes, editor)
}

#' Serialize the edited document with compress / GC / linearize options.
#' @param editor A `pdfoxide_editor`.
#' @param compress,garbage_collect,linearize Logical save options.
#' @return A `raw` vector.
#' @export
pdf_editor_save_to_bytes_with_options <- function(editor, compress = TRUE,
                                                  garbage_collect = TRUE,
                                                  linearize = FALSE) {
  .Call(C_r_editor_save_to_bytes_with_options, editor, isTRUE(compress),
        isTRUE(garbage_collect), isTRUE(linearize))
}

#' Save the edited document with AES-256 encryption to a path.
#' @param editor A `pdfoxide_editor`.
#' @param path Output path.
#' @param user_password,owner_password Encryption passwords.
#' @export
pdf_editor_save_encrypted <- function(editor, path, user_password,
                                      owner_password) {
  invisible(.Call(C_r_editor_save_encrypted, editor, path, user_password,
                  owner_password))
}

#' Save the edited document with AES-256 encryption to a raw vector.
#' @param editor A `pdfoxide_editor`.
#' @param user_password,owner_password Encryption passwords.
#' @return A `raw` vector.
#' @export
pdf_editor_save_encrypted_to_bytes <- function(editor, user_password,
                                               owner_password) {
  .Call(C_r_editor_save_encrypted_to_bytes, editor, user_password,
        owner_password)
}

#' Close an editor, freeing the native handle now (idempotent).
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_editor_close <- function(editor) {
  if (!inherits(editor, "pdfoxide_editor"))
    stop("pdf_editor_close: expected a pdfoxide_editor")
  invisible(.Call(C_r_editor_close, editor))
}

# ── PDF creation builder API ──────────────────────────────────────────────────
# Three owned native handle types, each an external pointer freed by the GC
# finalizer (or now via the explicit `*_close` helpers), mirroring the
# pdfoxide_document/pdfoxide_pdf/pdfoxide_editor pattern. Page indices and link
# targets are 0-based. PageBuilder ops are fluent (each returns the page invisibly
# so they chain via the pipe). `pdf_builder_*` = DocumentBuilder, `pdf_page_*` =
# PageBuilder, `pdf_embedded_font_*` = EmbeddedFont.

# ── EmbeddedFont ──

#' Load a TTF / OTF font from a file path for embedding.
#' @param path Path to a TTF / OTF font file.
#' @return A `pdfoxide_embedded_font` handle.
#' @export
pdf_embedded_font_from_file <- function(path) {
  structure(.Call(C_r_embedded_font_from_file, path),
            class = "pdfoxide_embedded_font")
}

#' Load a font for embedding from a raw vector of TTF / OTF bytes.
#' @param bytes A `raw` vector of font bytes.
#' @param name Optional PostScript name; `NULL` uses the name from the font face.
#' @return A `pdfoxide_embedded_font` handle.
#' @export
pdf_embedded_font_from_bytes <- function(bytes, name = NULL) {
  structure(.Call(C_r_embedded_font_from_bytes, bytes, name),
            class = "pdfoxide_embedded_font")
}

#' Free an embedded-font handle now (idempotent).
#'
#' After a successful [pdf_builder_register_embedded_font()] the builder owns the
#' font and this becomes a no-op.
#' @param font A `pdfoxide_embedded_font`.
#' @export
pdf_embedded_font_close <- function(font) {
  if (!inherits(font, "pdfoxide_embedded_font"))
    stop("pdf_embedded_font_close: expected a pdfoxide_embedded_font")
  invisible(.Call(C_r_embedded_font_close, font))
}

# ── DocumentBuilder ──

#' Create a new PDF document builder.
#' @return A `pdfoxide_builder` handle.
#' @export
pdf_builder_create <- function() {
  structure(.Call(C_r_builder_create), class = "pdfoxide_builder")
}

#' Close a document builder, freeing the native handle now (idempotent).
#' @param builder A `pdfoxide_builder`.
#' @export
pdf_builder_close <- function(builder) {
  if (!inherits(builder, "pdfoxide_builder"))
    stop("pdf_builder_close: expected a pdfoxide_builder")
  invisible(.Call(C_r_builder_close, builder))
}

#' Set document metadata on the builder.
#'
#' Each returns the `builder` invisibly so calls chain.
#' @param builder A `pdfoxide_builder`.
#' @param value The metadata string.
#' @export
pdf_builder_set_title <- function(builder, value) {
  .Call(C_r_builder_set_title, builder, value)
  invisible(builder)
}
#' @rdname pdf_builder_set_title
#' @export
pdf_builder_set_author <- function(builder, value) {
  .Call(C_r_builder_set_author, builder, value)
  invisible(builder)
}
#' @rdname pdf_builder_set_title
#' @export
pdf_builder_set_subject <- function(builder, value) {
  .Call(C_r_builder_set_subject, builder, value)
  invisible(builder)
}
#' @rdname pdf_builder_set_title
#' @export
pdf_builder_set_keywords <- function(builder, value) {
  .Call(C_r_builder_set_keywords, builder, value)
  invisible(builder)
}
#' @rdname pdf_builder_set_title
#' @export
pdf_builder_set_creator <- function(builder, value) {
  .Call(C_r_builder_set_creator, builder, value)
  invisible(builder)
}

#' Run JavaScript when the document is opened (`/OpenAction`).
#' @param builder A `pdfoxide_builder`.
#' @param script JavaScript source.
#' @export
pdf_builder_on_open <- function(builder, script) {
  .Call(C_r_builder_on_open, builder, script)
  invisible(builder)
}

#' Set the document's natural language tag (e.g. "en-US"), emitted as `/Lang`.
#' @param builder A `pdfoxide_builder`.
#' @param lang BCP-47 language tag.
#' @export
pdf_builder_language <- function(builder, lang) {
  .Call(C_r_builder_language, builder, lang)
  invisible(builder)
}

#' Enable PDF/UA-1 tagged-PDF mode.
#' @param builder A `pdfoxide_builder`.
#' @export
pdf_builder_tagged_pdf_ua1 <- function(builder) {
  .Call(C_r_builder_tagged_pdf_ua1, builder)
  invisible(builder)
}

#' Add a role-map entry: custom structure type to standard PDF structure type.
#' @param builder A `pdfoxide_builder`.
#' @param custom Custom structure type.
#' @param standard Standard PDF structure type.
#' @export
pdf_builder_role_map <- function(builder, custom, standard) {
  .Call(C_r_builder_role_map, builder, custom, standard)
  invisible(builder)
}

#' Register a TTF / OTF font for embedding under `name`.
#'
#' On success the builder takes ownership of `font`; do not use or close it after.
#' @param builder A `pdfoxide_builder`.
#' @param name Font name to register under.
#' @param font A `pdfoxide_embedded_font`.
#' @export
pdf_builder_register_embedded_font <- function(builder, name, font) {
  if (!inherits(font, "pdfoxide_embedded_font"))
    stop("pdf_builder_register_embedded_font: expected a pdfoxide_embedded_font")
  .Call(C_r_builder_register_embedded_font, builder, name, font)
  invisible(builder)
}

#' Start a page on the builder.
#'
#' `pdf_builder_page` takes custom dimensions in PDF points (72 pt = 1 inch);
#' `pdf_builder_a4_page` and `pdf_builder_letter_page` use standard sizes. Only
#' one page may be open at a time; call [pdf_page_done()] to commit it.
#' @param builder A `pdfoxide_builder`.
#' @param width,height Page size in points.
#' @return A `pdfoxide_page_builder` handle.
#' @export
pdf_builder_page <- function(builder, width, height) {
  structure(.Call(C_r_builder_page, builder, as.double(width), as.double(height)),
            class = "pdfoxide_page_builder")
}
#' @rdname pdf_builder_page
#' @export
pdf_builder_a4_page <- function(builder) {
  structure(.Call(C_r_builder_a4_page, builder),
            class = "pdfoxide_page_builder")
}
#' @rdname pdf_builder_page
#' @export
pdf_builder_letter_page <- function(builder) {
  structure(.Call(C_r_builder_letter_page, builder),
            class = "pdfoxide_page_builder")
}

#' Build the document to a raw vector of PDF bytes.
#' @param builder A `pdfoxide_builder`.
#' @return A `raw` vector.
#' @export
pdf_builder_build <- function(builder) {
  .Call(C_r_builder_build, builder)
}

#' Build and save the document to a path.
#' @param builder A `pdfoxide_builder`.
#' @param path Output path.
#' @export
pdf_builder_save <- function(builder, path) {
  invisible(.Call(C_r_builder_save, builder, path))
}

#' Build and save the document with AES-256 encryption to a path.
#' @param builder A `pdfoxide_builder`.
#' @param path Output path.
#' @param user_password,owner_password Encryption passwords.
#' @export
pdf_builder_save_encrypted <- function(builder, path, user_password,
                                       owner_password) {
  invisible(.Call(C_r_builder_save_encrypted, builder, path, user_password,
                  owner_password))
}

#' Build the document with AES-256 encryption to a raw vector.
#' @param builder A `pdfoxide_builder`.
#' @param user_password,owner_password Encryption passwords.
#' @return A `raw` vector.
#' @export
pdf_builder_to_bytes_encrypted <- function(builder, user_password,
                                           owner_password) {
  .Call(C_r_builder_to_bytes_encrypted, builder, user_password, owner_password)
}

# ── PageBuilder ──
# All ops return the page invisibly so they chain via the pipe.

#' Set the font + size for subsequent text on this page.
#' @param page A `pdfoxide_page_builder`.
#' @param name Font name.
#' @param size pt.
#' @export
pdf_page_font <- function(page, name, size) {
  .Call(C_r_page_font, page, name, as.double(size))
  invisible(page)
}

#' Move the cursor to absolute coordinates (PDF points, from lower-left).
#' @param page A `pdfoxide_page_builder`.
#' @param x,y Coordinates in points.
#' @export
pdf_page_at <- function(page, x, y) {
  .Call(C_r_page_at, page, as.double(x), as.double(y))
  invisible(page)
}

#' Emit a line of text at the cursor, then advance one line-height.
#' @param page A `pdfoxide_page_builder`.
#' @param text The text.
#' @export
pdf_page_builder_text <- function(page, text) {
  .Call(C_r_page_text, page, text)
  invisible(page)
}

#' Emit a heading with the given level (1-6) and text.
#' @param page A `pdfoxide_page_builder`.
#' @param level Heading level 1-6.
#' @param text Heading text.
#' @export
pdf_page_heading <- function(page, level, text) {
  .Call(C_r_page_heading, page, as.integer(level), text)
  invisible(page)
}

#' Emit a paragraph with automatic line wrapping.
#' @param page A `pdfoxide_page_builder`.
#' @param text The paragraph text.
#' @export
pdf_page_paragraph <- function(page, text) {
  .Call(C_r_page_paragraph, page, text)
  invisible(page)
}

#' Advance the cursor down by `points`.
#' @param page A `pdfoxide_page_builder`.
#' @param points Vertical advance in pts.
#' @export
pdf_page_space <- function(page, points) {
  .Call(C_r_page_space, page, as.double(points))
  invisible(page)
}

#' Draw a horizontal rule across the page.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_horizontal_rule <- function(page) {
  .Call(C_r_page_horizontal_rule, page)
  invisible(page)
}

#' Attach a link to the previously-emitted text element.
#'
#' `pdf_page_link_page` targets a 0-based internal page index.
#' @param page A `pdfoxide_page_builder`.
#' @param url URL.
#' @param index 0-based
#'   page index.
#' @param destination Named destination.
#' @param script JavaScript.
#' @export
pdf_page_link_url <- function(page, url) {
  .Call(C_r_page_link_url, page, url)
  invisible(page)
}
#' @rdname pdf_page_link_url
#' @export
pdf_page_link_page <- function(page, index) {
  .Call(C_r_page_link_page, page, as.integer(index))
  invisible(page)
}
#' @rdname pdf_page_link_url
#' @export
pdf_page_link_named <- function(page, destination) {
  .Call(C_r_page_link_named, page, destination)
  invisible(page)
}
#' @rdname pdf_page_link_url
#' @export
pdf_page_link_javascript <- function(page, script) {
  .Call(C_r_page_link_javascript, page, script)
  invisible(page)
}

#' Run JavaScript on page open / close (`/AA /O`, `/AA /C`).
#' @param page A `pdfoxide_page_builder`.
#' @param script JavaScript source.
#' @export
pdf_page_on_open <- function(page, script) {
  .Call(C_r_page_on_open, page, script)
  invisible(page)
}
#' @rdname pdf_page_on_open
#' @export
pdf_page_on_close <- function(page, script) {
  .Call(C_r_page_on_close, page, script)
  invisible(page)
}

#' Set a JS action on the most-recently-added form field.
#' @param page A `pdfoxide_page_builder`.
#' @param script JavaScript source.
#' @export
pdf_page_field_keystroke <- function(page, script) {
  .Call(C_r_page_field_keystroke, page, script)
  invisible(page)
}
#' @rdname pdf_page_field_keystroke
#' @export
pdf_page_field_format <- function(page, script) {
  .Call(C_r_page_field_format, page, script)
  invisible(page)
}
#' @rdname pdf_page_field_keystroke
#' @export
pdf_page_field_validate <- function(page, script) {
  .Call(C_r_page_field_validate, page, script)
  invisible(page)
}
#' @rdname pdf_page_field_keystroke
#' @export
pdf_page_field_calculate <- function(page, script) {
  .Call(C_r_page_field_calculate, page, script)
  invisible(page)
}

#' Decorate the previous text with an RGB colour (channels 0.0-1.0).
#' @param page A `pdfoxide_page_builder`.
#' @param r,g,b Colour channels (0-1).
#' @export
pdf_page_highlight <- function(page, r, g, b) {
  .Call(C_r_page_highlight, page, as.double(r), as.double(g), as.double(b))
  invisible(page)
}
#' @rdname pdf_page_highlight
#' @export
pdf_page_underline <- function(page, r, g, b) {
  .Call(C_r_page_underline, page, as.double(r), as.double(g), as.double(b))
  invisible(page)
}
#' @rdname pdf_page_highlight
#' @export
pdf_page_strikeout <- function(page, r, g, b) {
  .Call(C_r_page_strikeout, page, as.double(r), as.double(g), as.double(b))
  invisible(page)
}
#' @rdname pdf_page_highlight
#' @export
pdf_page_squiggly <- function(page, r, g, b) {
  .Call(C_r_page_squiggly, page, as.double(r), as.double(g), as.double(b))
  invisible(page)
}

#' Attach a sticky-note annotation to the previous text.
#' @param page A `pdfoxide_page_builder`.
#' @param text Note text.
#' @export
pdf_page_sticky_note <- function(page, text) {
  .Call(C_r_page_sticky_note, page, text)
  invisible(page)
}

#' Place a free-standing sticky note at an absolute page position.
#' @param page A `pdfoxide_page_builder`.
#' @param x,y Position in points.
#' @param text Note text.
#' @export
pdf_page_sticky_note_at <- function(page, x, y, text) {
  .Call(C_r_page_sticky_note_at, page, as.double(x), as.double(y), text)
  invisible(page)
}

#' Apply a text watermark to the page.
#' @param page A `pdfoxide_page_builder`.
#' @param text Watermark text.
#' @export
pdf_page_watermark <- function(page, text) {
  .Call(C_r_page_watermark, page, text)
  invisible(page)
}

#' Apply a standard diagonal watermark ("CONFIDENTIAL" / "DRAFT").
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_watermark_confidential <- function(page) {
  .Call(C_r_page_watermark_confidential, page)
  invisible(page)
}
#' @rdname pdf_page_watermark_confidential
#' @export
pdf_page_watermark_draft <- function(page) {
  .Call(C_r_page_watermark_draft, page)
  invisible(page)
}

#' Attach a standard stamp annotation at the current cursor position.
#' @param page A `pdfoxide_page_builder`.
#' @param type_name Stamp name.
#' @export
pdf_page_stamp <- function(page, type_name) {
  .Call(C_r_page_stamp, page, type_name)
  invisible(page)
}

#' Place a free-flowing text annotation inside a rectangle.
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @param text The annotation text.
#' @export
pdf_page_freetext <- function(page, x, y, w, h, text) {
  .Call(C_r_page_freetext, page, as.double(x), as.double(y), as.double(w),
        as.double(h), text)
  invisible(page)
}

#' Add a footnote reference mark inline and record its body for page-end placement.
#' @param page A `pdfoxide_page_builder`.
#' @param ref_mark Superscript label.
#' @param note_text Footnote body text.
#' @export
pdf_page_footnote <- function(page, ref_mark, note_text) {
  .Call(C_r_page_footnote, page, ref_mark, note_text)
  invisible(page)
}

#' Lay out text across `column_count` balanced columns at the cursor.
#' @param page A `pdfoxide_page_builder`.
#' @param column_count Number of columns.
#' @param gap_pt Inter-column gap in points.
#' @param text Text to flow.
#' @export
pdf_page_columns <- function(page, column_count, gap_pt, text) {
  .Call(C_r_page_columns, page, as.integer(column_count), as.double(gap_pt), text)
  invisible(page)
}

#' Emit an inline run (advances the cursor horizontally, not vertically).
#'
#' Call [pdf_page_newline()] to advance to the next line.
#' @param page A `pdfoxide_page_builder`.
#' @param text The text.
#' @export
pdf_page_inline <- function(page, text) {
  .Call(C_r_page_inline, page, text)
  invisible(page)
}
#' @rdname pdf_page_inline
#' @export
pdf_page_inline_bold <- function(page, text) {
  .Call(C_r_page_inline_bold, page, text)
  invisible(page)
}
#' @rdname pdf_page_inline
#' @export
pdf_page_inline_italic <- function(page, text) {
  .Call(C_r_page_inline_italic, page, text)
  invisible(page)
}

#' Emit an inline coloured run (RGB 0.0-1.0).
#' @param page A `pdfoxide_page_builder`.
#' @param r,g,b Colour channels (0-1).
#' @param text The text.
#' @export
pdf_page_inline_color <- function(page, r, g, b, text) {
  .Call(C_r_page_inline_color, page, as.double(r), as.double(g), as.double(b),
        text)
  invisible(page)
}

#' Advance the cursor by one line-height and reset to the left margin.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_newline <- function(page) {
  .Call(C_r_page_newline, page)
  invisible(page)
}

#' Add a single-line text form field.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param x,y,w,h Field rectangle in points.
#' @param default_value Initial value, or `NULL` for blank.
#' @export
pdf_page_text_field <- function(page, name, x, y, w, h, default_value = NULL) {
  .Call(C_r_page_text_field, page, name, as.double(x), as.double(y),
        as.double(w), as.double(h), default_value)
  invisible(page)
}

#' Add a checkbox form field.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param x,y,w,h Field rectangle in points.
#' @param checked Initially ticked?
#' @export
pdf_page_checkbox <- function(page, name, x, y, w, h, checked = FALSE) {
  .Call(C_r_page_checkbox, page, name, as.double(x), as.double(y), as.double(w),
        as.double(h), isTRUE(checked))
  invisible(page)
}

#' Add a dropdown combo-box form field.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param x,y,w,h Field rectangle in points.
#' @param options Character vector of options.
#' @param selected Initially-selected option, or `NULL`.
#' @export
pdf_page_combo_box <- function(page, name, x, y, w, h, options, selected = NULL) {
  .Call(C_r_page_combo_box, page, name, as.double(x), as.double(y), as.double(w),
        as.double(h), as.character(options), selected)
  invisible(page)
}

#' Add a radio-button group.
#'
#' `values`, `xs`, `ys`, `ws`, `hs` are parallel vectors of equal length.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param values Character vector of export values.
#' @param xs,ys,ws,hs Numeric vectors of per-button rectangles.
#' @param selected Initially-selected export value, or `NULL`.
#' @export
pdf_page_radio_group <- function(page, name, values, xs, ys, ws, hs,
                                 selected = NULL) {
  .Call(C_r_page_radio_group, page, name, as.character(values), as.double(xs),
        as.double(ys), as.double(ws), as.double(hs), selected)
  invisible(page)
}

#' Add a clickable push button with a visible caption.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param x,y,w,h Field rectangle in points.
#' @param caption Button caption.
#' @export
pdf_page_push_button <- function(page, name, x, y, w, h, caption) {
  .Call(C_r_page_push_button, page, name, as.double(x), as.double(y),
        as.double(w), as.double(h), caption)
  invisible(page)
}

#' Add an unsigned signature placeholder field.
#' @param page A `pdfoxide_page_builder`.
#' @param name Field name.
#' @param x,y,w,h Field rectangle in points.
#' @export
pdf_page_signature_field <- function(page, name, x, y, w, h) {
  .Call(C_r_page_signature_field, page, name, as.double(x), as.double(y),
        as.double(w), as.double(h))
  invisible(page)
}

#' Place a 1-D barcode image on the page.
#'
#' `barcode_type`: 0=Code128 1=Code39 2=EAN13 3=EAN8 4=UPCA 5=ITF 6=Code93
#' 7=Codabar.
#' @param page A `pdfoxide_page_builder`.
#' @param barcode_type Symbology code.
#' @param data Barcode data.
#' @param x,y,w,h Rectangle in points.
#' @export
pdf_page_barcode_1d <- function(page, barcode_type, data, x, y, w, h) {
  .Call(C_r_page_barcode_1d, page, as.integer(barcode_type), data, as.double(x),
        as.double(y), as.double(w), as.double(h))
  invisible(page)
}

#' Place a QR-code image on the page (square: `size` x `size` points).
#' @param page A `pdfoxide_page_builder`.
#' @param data QR data.
#' @param x,y Position in points.
#' @param size Side length in points.
#' @export
pdf_page_barcode_qr <- function(page, data, x, y, size) {
  .Call(C_r_page_barcode_qr, page, data, as.double(x), as.double(y),
        as.double(size))
  invisible(page)
}

#' Embed an image on the page.
#'
#' `bytes` is a `raw` vector of raw JPEG / PNG / WebP image data.
#' `pdf_page_image_with_alt` adds accessibility alt text; `pdf_page_image_artifact`
#' marks the image as a decorative `/Artifact`.
#' @param page A `pdfoxide_page_builder`.
#' @param bytes A `raw` vector.
#' @param x,y,w,h Image rectangle in points.
#' @param alt_text Accessibility text.
#' @export
pdf_page_image <- function(page, bytes, x, y, w, h) {
  .Call(C_r_page_image, page, bytes, as.double(x), as.double(y), as.double(w),
        as.double(h))
  invisible(page)
}
#' @rdname pdf_page_image
#' @export
pdf_page_image_with_alt <- function(page, bytes, x, y, w, h, alt_text) {
  .Call(C_r_page_image_with_alt, page, bytes, as.double(x), as.double(y),
        as.double(w), as.double(h), alt_text)
  invisible(page)
}
#' @rdname pdf_page_image
#' @export
pdf_page_image_artifact <- function(page, bytes, x, y, w, h) {
  .Call(C_r_page_image_artifact, page, bytes, as.double(x), as.double(y),
        as.double(w), as.double(h))
  invisible(page)
}

#' Draw a stroked rectangle outline (1pt black).
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @export
pdf_page_rect <- function(page, x, y, w, h) {
  .Call(C_r_page_rect, page, as.double(x), as.double(y), as.double(w),
        as.double(h))
  invisible(page)
}

#' Draw a filled rectangle in RGB colour (channels 0-1).
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @param r,g,b Fill colour channels (0-1).
#' @export
pdf_page_filled_rect <- function(page, x, y, w, h, r, g, b) {
  .Call(C_r_page_filled_rect, page, as.double(x), as.double(y), as.double(w),
        as.double(h), as.double(r), as.double(g), as.double(b))
  invisible(page)
}

#' Draw a line with a 1pt black stroke.
#' @param page A `pdfoxide_page_builder`.
#' @param x1,y1,x2,y2 Endpoints in points.
#' @export
pdf_page_line <- function(page, x1, y1, x2, y2) {
  .Call(C_r_page_line, page, as.double(x1), as.double(y1), as.double(x2),
        as.double(y2))
  invisible(page)
}

#' Draw a stroked rectangle with explicit width and RGB colour.
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @param width Stroke width in points.
#' @param r,g,b Stroke colour (0-1).
#' @export
pdf_page_stroke_rect <- function(page, x, y, w, h, width, r, g, b) {
  .Call(C_r_page_stroke_rect, page, as.double(x), as.double(y), as.double(w),
        as.double(h), as.double(width), as.double(r), as.double(g),
        as.double(b))
  invisible(page)
}

#' Draw a stroked line with explicit width and RGB colour.
#' @param page A `pdfoxide_page_builder`.
#' @param x1,y1,x2,y2 Endpoints in points.
#' @param width Stroke width in points.
#' @param r,g,b Stroke colour (0-1).
#' @export
pdf_page_stroke_line <- function(page, x1, y1, x2, y2, width, r, g, b) {
  .Call(C_r_page_stroke_line, page, as.double(x1), as.double(y1), as.double(x2),
        as.double(y2), as.double(width), as.double(r), as.double(g),
        as.double(b))
  invisible(page)
}

#' Draw a dashed stroked rectangle.
#'
#' `dash` is a numeric vector of alternating on/off lengths in points (empty for
#' solid); `phase` is the starting offset into the pattern.
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @param width Stroke width in points.
#' @param r,g,b Stroke colour (0-1).
#' @param dash Numeric dash pattern.
#' @param phase Dash phase offset.
#' @export
pdf_page_stroke_rect_dashed <- function(page, x, y, w, h, width, r, g, b,
                                        dash = numeric(0), phase = 0) {
  .Call(C_r_page_stroke_rect_dashed, page, as.double(x), as.double(y),
        as.double(w), as.double(h), as.double(width), as.double(r), as.double(g),
        as.double(b), as.double(dash), as.double(phase))
  invisible(page)
}

#' Draw a dashed stroked line.
#' @param page A `pdfoxide_page_builder`.
#' @param x1,y1,x2,y2 Endpoints in points.
#' @param width Stroke width in points.
#' @param r,g,b Stroke colour (0-1).
#' @param dash Numeric dash pattern.
#' @param phase Dash phase offset.
#' @export
pdf_page_stroke_line_dashed <- function(page, x1, y1, x2, y2, width, r, g, b,
                                        dash = numeric(0), phase = 0) {
  .Call(C_r_page_stroke_line_dashed, page, as.double(x1), as.double(y1),
        as.double(x2), as.double(y2), as.double(width), as.double(r),
        as.double(g), as.double(b), as.double(dash), as.double(phase))
  invisible(page)
}

#' Lay out text inside a rectangle with an alignment.
#'
#' `align`: 0=Left, 1=Center, 2=Right.
#' @param page A `pdfoxide_page_builder`.
#' @param x,y,w,h Rectangle in points.
#' @param text The text.
#' @param align Alignment code (0/1/2).
#' @export
pdf_page_text_in_rect <- function(page, x, y, w, h, text, align = 0L) {
  .Call(C_r_page_text_in_rect, page, as.double(x), as.double(y), as.double(w),
        as.double(h), text, as.integer(align))
  invisible(page)
}

#' Start a new page of the same size; subsequent ops land on the new page.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_new_page_same_size <- function(page) {
  .Call(C_r_page_new_page_same_size, page)
  invisible(page)
}

#' Emit a table.
#'
#' `cells` is a character vector or matrix. A matrix is read row-major; a flat
#' vector must already be row-major of length `n_rows * n_columns`. `widths` and
#' `aligns` are length-`n_columns` vectors (aligns: 0=Left, 1=Center, 2=Right).
#' @param page A `pdfoxide_page_builder`.
#' @param widths Column widths (numeric).
#' @param aligns Per-column alignment codes.
#' @param cells Cell text (row-major).
#' @param has_header Promote the first row to a header?
#' @param n_columns,n_rows Dimensions; defaulted from a matrix `cells`.
#' @export
pdf_page_table <- function(page, widths, aligns, cells, has_header = FALSE,
                           n_columns = length(widths), n_rows = NULL) {
  if (is.matrix(cells)) {
    if (missing(n_columns)) n_columns <- ncol(cells)
    if (is.null(n_rows)) n_rows <- nrow(cells)
    cells <- as.character(t(cells))  # row-major
  } else {
    cells <- as.character(cells)
    if (is.null(n_rows)) n_rows <- length(cells) %/% max(1L, n_columns)
  }
  .Call(C_r_page_table, page, as.integer(n_columns), as.double(widths),
        as.integer(aligns), as.integer(n_rows), cells, isTRUE(has_header))
  invisible(page)
}

#' Open a streaming table on the page.
#'
#' `headers`, `widths`, `aligns` are length-`n_columns` parallel vectors
#' (aligns: 0=Left, 1=Center, 2=Right). Feed rows with
#' [pdf_page_streaming_table_push_row()]; close with
#' [pdf_page_streaming_table_finish()].
#' @param page A `pdfoxide_page_builder`.
#' @param headers Column headers.
#' @param widths Column widths.
#' @param aligns Per-column alignment codes.
#' @param repeat_header Repeat the header on each page break?
#' @param n_columns Column count; defaulted from `headers`.
#' @export
pdf_page_streaming_table_begin <- function(page, headers, widths, aligns,
                                           repeat_header = FALSE,
                                           n_columns = length(headers)) {
  .Call(C_r_page_streaming_table_begin, page, as.integer(n_columns),
        as.character(headers), as.double(widths), as.integer(aligns),
        isTRUE(repeat_header))
  invisible(page)
}

#' Open a streaming table with a column-width mode and optional rowspan.
#'
#' `mode`: 0=Fixed, 1=Sample(sample_rows, min/max width). `max_rowspan` of 0/1
#' disables rowspan.
#' @inheritParams pdf_page_streaming_table_begin
#' @param mode Column-width mode.
#' @param sample_rows Rows to sample (mode 1).
#' @param min_col_width_pt,max_col_width_pt Width bounds (mode 1).
#' @param max_rowspan Maximum rowspan (>=2 enables).
#' @export
pdf_page_streaming_table_begin_v2 <- function(page, headers, widths, aligns,
                                              repeat_header = FALSE, mode = 0L,
                                              sample_rows = 0L,
                                              min_col_width_pt = 0,
                                              max_col_width_pt = 0,
                                              max_rowspan = 0L,
                                              n_columns = length(headers)) {
  .Call(C_r_page_streaming_table_begin_v2, page, as.integer(n_columns),
        as.character(headers), as.double(widths), as.integer(aligns),
        isTRUE(repeat_header), as.integer(mode), as.integer(sample_rows),
        as.double(min_col_width_pt), as.double(max_col_width_pt),
        as.integer(max_rowspan))
  invisible(page)
}

#' Set the batch size for the currently-open streaming table (0 defaults to 256).
#' @param page A `pdfoxide_page_builder`.
#' @param batch_size Rows per batch.
#' @export
pdf_page_streaming_table_set_batch_size <- function(page, batch_size) {
  .Call(C_r_page_streaming_table_set_batch_size, page, as.integer(batch_size))
  invisible(page)
}

#' Number of rows pushed since the last batch boundary.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_streaming_table_pending_row_count <- function(page) {
  .Call(C_r_page_streaming_table_pending_row_count, page)
}

#' Number of complete batches recorded so far.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_streaming_table_batch_count <- function(page) {
  .Call(C_r_page_streaming_table_batch_count, page)
}

#' Mark a batch boundary in the currently-open streaming table.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_streaming_table_flush <- function(page) {
  .Call(C_r_page_streaming_table_flush, page)
  invisible(page)
}

#' Push one row into the currently-open streaming table.
#'
#' `cells` length must equal the column count given to begin.
#' @param page A `pdfoxide_page_builder`.
#' @param cells Character vector of cells.
#' @export
pdf_page_streaming_table_push_row <- function(page, cells) {
  .Call(C_r_page_streaming_table_push_row, page, as.character(cells))
  invisible(page)
}

#' Push one row with per-cell rowspan values.
#'
#' `rowspans` is an integer vector parallel to `cells` (1=normal, >=2=span), or
#' `NULL` to treat all cells as rowspan=1.
#' @param page A `pdfoxide_page_builder`.
#' @param cells Character vector of cells.
#' @param rowspans Integer rowspan vector, or `NULL`.
#' @export
pdf_page_streaming_table_push_row_v2 <- function(page, cells, rowspans = NULL) {
  if (!is.null(rowspans)) rowspans <- as.integer(rowspans)
  .Call(C_r_page_streaming_table_push_row_v2, page, as.character(cells),
        rowspans)
  invisible(page)
}

#' Close the currently-open streaming table.
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_streaming_table_finish <- function(page) {
  .Call(C_r_page_streaming_table_finish, page)
  invisible(page)
}

#' Commit the page's buffered operations to its parent builder.
#'
#' Consumes the page handle; the handle must not be used afterward (later use
#' raises "handle is closed").
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_done <- function(page) {
  if (!inherits(page, "pdfoxide_page_builder"))
    stop("pdf_page_done: expected a pdfoxide_page_builder")
  invisible(.Call(C_r_page_done, page))
}

#' Drop an uncommitted page handle without applying its operations (idempotent).
#' @param page A `pdfoxide_page_builder`.
#' @export
pdf_page_close <- function(page) {
  if (!inherits(page, "pdfoxide_page_builder"))
    stop("pdf_page_close: expected a pdfoxide_page_builder")
  invisible(.Call(C_r_page_close, page))
}

# ── PHASE-6: digital signatures / PKI / timestamps / TSA / DSS / validation ────
# Five owned native handle families, each an external pointer freed by the GC
# finalizer (or now via the explicit `*_close` helper): pdfoxide_certificate,
# pdfoxide_signature, pdfoxide_timestamp, pdfoxide_tsa_client, pdfoxide_dss, plus
# the validation result types pdfoxide_pdf_a_results / pdfoxide_ua_results /
# pdfoxide_pdf_x_results. Epoch times are returned as numeric seconds; const
# byte returns (timestamp token / message imprint) are copied into `raw`.

# ── Logging ──

#' Set the global library log level.
#'
#' Levels: `0` = Off, `1` = Error, `2` = Warn, `3` = Info, `4` = Debug,
#' `5` = Trace.
#' @param level Integer log level (0-5).
#' @export
pdf_set_log_level <- function(level) {
  invisible(.Call(C_r_oxide_set_log_level, as.integer(level)))
}

#' Get the current library log level (0-5).
#' @return An integer log level.
#' @export
pdf_get_log_level <- function() .Call(C_r_oxide_get_log_level)

# ── Certificate ──

#' Load signing credentials (certificate + private key) from PKCS#12 bytes.
#' @param bytes A `raw` vector of PKCS#12 (.p12 / .pfx) data.
#' @param password Passphrase protecting the PKCS#12, or `NULL`.
#' @return A `pdfoxide_certificate` handle.
#' @export
pdf_certificate_load_from_bytes <- function(bytes, password = NULL) {
  structure(.Call(C_r_certificate_load_from_bytes, bytes, password),
            class = "pdfoxide_certificate")
}

#' Load signing credentials from PEM-encoded certificate + private-key strings.
#' @param cert_pem PEM certificate string.
#' @param key_pem PEM private-key string.
#' @return A `pdfoxide_certificate` handle.
#' @export
pdf_certificate_load_from_pem <- function(cert_pem, key_pem) {
  structure(.Call(C_r_certificate_load_from_pem, cert_pem, key_pem),
            class = "pdfoxide_certificate")
}

#' Certificate subject / issuer distinguished name and serial number.
#' @param cert A `pdfoxide_certificate`.
#' @export
pdf_certificate_subject <- function(cert) .Call(C_r_certificate_get_subject, cert)
#' @rdname pdf_certificate_subject
#' @export
pdf_certificate_issuer <- function(cert) .Call(C_r_certificate_get_issuer, cert)
#' @rdname pdf_certificate_subject
#' @export
pdf_certificate_serial <- function(cert) .Call(C_r_certificate_get_serial, cert)

#' Certificate validity window as `list(not_before=, not_after=)` epoch seconds.
#' @param cert A `pdfoxide_certificate`.
#' @export
pdf_certificate_validity <- function(cert) {
  v <- .Call(C_r_certificate_get_validity, cert)
  list(not_before = v[1], not_after = v[2])
}

#' Whether the certificate is currently within its validity window.
#' @param cert A `pdfoxide_certificate`.
#' @return A logical scalar.
#' @export
pdf_certificate_is_valid <- function(cert) .Call(C_r_certificate_is_valid, cert)

#' Free a certificate handle now (idempotent).
#' @param cert A `pdfoxide_certificate`.
#' @export
pdf_certificate_close <- function(cert) {
  if (!inherits(cert, "pdfoxide_certificate"))
    stop("pdf_certificate_close: expected a pdfoxide_certificate")
  invisible(.Call(C_r_certificate_close, cert))
}

# ── Signing ──

#' Sign raw PDF bytes with a certificate, returning the signed PDF.
#' @param pdf A `raw` vector of PDF bytes.
#' @param cert A `pdfoxide_certificate`.
#' @param reason Signing reason, or `NULL`.
#' @param location Location, or `NULL`.
#' @return A `raw` vector of the signed PDF.
#' @export
pdf_sign_bytes <- function(pdf, cert, reason = NULL, location = NULL) {
  if (!inherits(cert, "pdfoxide_certificate"))
    stop("pdf_sign_bytes: expected a pdfoxide_certificate")
  .Call(C_r_sign_bytes, pdf, cert, reason, location)
}

#' Sign raw PDF bytes at a PAdES baseline level.
#'
#' `level`: `0` = B-B, `1` = B-T, `2` = B-LT (`3` = B-LTA is unsupported).
#' `tsa_url` is required for `level >= 1`. `certs`, `crls`, `ocsps` are lists of
#' `raw` vectors (DER) carrying the B-LT revocation material.
#' @param pdf A `raw` vector of PDF bytes.
#' @param cert A `pdfoxide_certificate`.
#' @param level PAdES baseline level (0-2).
#' @param tsa_url RFC 3161 TSA URL.
#' @param reason Signing reason.
#' @param location Location.
#' @param certs,crls,ocsps Lists of `raw` DER vectors (may be empty).
#' @return A `raw` vector of the signed PDF.
#' @export
pdf_sign_bytes_pades <- function(pdf, cert, level = 0L, tsa_url = NULL,
                                 reason = NULL, location = NULL,
                                 certs = list(), crls = list(), ocsps = list()) {
  if (!inherits(cert, "pdfoxide_certificate"))
    stop("pdf_sign_bytes_pades: expected a pdfoxide_certificate")
  .Call(C_r_sign_bytes_pades, pdf, cert, as.integer(level), tsa_url, reason,
        location, as.list(certs), as.list(crls), as.list(ocsps))
}

#' Sign raw PDF bytes at a PAdES level via the struct-options entry point.
#'
#' Functionally identical to [pdf_sign_bytes_pades()]; uses the
#' `PadesSignOptionsC` struct variant of the C ABI.
#' @inheritParams pdf_sign_bytes_pades
#' @return A `raw` vector of the signed PDF.
#' @export
pdf_sign_bytes_pades_opts <- function(pdf, cert, level = 0L, tsa_url = NULL,
                                      reason = NULL, location = NULL,
                                      certs = list(), crls = list(),
                                      ocsps = list()) {
  if (!inherits(cert, "pdfoxide_certificate"))
    stop("pdf_sign_bytes_pades_opts: expected a pdfoxide_certificate")
  .Call(C_r_sign_bytes_pades_opts, pdf, cert, as.integer(level), tsa_url, reason,
        location, as.list(certs), as.list(crls), as.list(ocsps))
}

# ── SignatureInfo ──

#' Number of signatures present in a document.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_signature_count <- function(doc) .Call(C_r_doc_signature_count, doc)

#' Get the `index`-th signature (0-based) of a document.
#' @param doc A `pdfoxide_document`.
#' @param index 0-based signature index.
#' @return A `pdfoxide_signature` handle.
#' @export
pdf_get_signature <- function(doc, index) {
  structure(.Call(C_r_doc_get_signature, doc, as.integer(index)),
            class = "pdfoxide_signature")
}

#' Signature signer name / signing reason / signing location.
#' @param sig A `pdfoxide_signature`.
#' @export
pdf_signature_signer_name <- function(sig) {
  .Call(C_r_signature_get_signer_name, sig)
}
#' @rdname pdf_signature_signer_name
#' @export
pdf_signature_signing_reason <- function(sig) {
  .Call(C_r_signature_get_signing_reason, sig)
}
#' @rdname pdf_signature_signer_name
#' @export
pdf_signature_signing_location <- function(sig) {
  .Call(C_r_signature_get_signing_location, sig)
}

#' Signing time as epoch seconds.
#' @param sig A `pdfoxide_signature`.
#' @export
pdf_signature_signing_time <- function(sig) {
  .Call(C_r_signature_get_signing_time, sig)
}

#' Get the signer certificate of a signature.
#' @param sig A `pdfoxide_signature`.
#' @return A `pdfoxide_certificate` handle.
#' @export
pdf_signature_certificate <- function(sig) {
  structure(.Call(C_r_signature_get_certificate, sig),
            class = "pdfoxide_certificate")
}

#' PAdES level of a signature (`0` = B-B, `1` = B-T, `2` = B-LT).
#' @param sig A `pdfoxide_signature`.
#' @export
pdf_signature_pades_level <- function(sig) {
  .Call(C_r_signature_get_pades_level, sig)
}

#' Whether a signature carries an embedded timestamp.
#' @param sig A `pdfoxide_signature`.
#' @return A logical scalar.
#' @export
pdf_signature_has_timestamp <- function(sig) {
  .Call(C_r_signature_has_timestamp, sig)
}

#' Get the embedded timestamp of a signature.
#' @param sig A `pdfoxide_signature`.
#' @return A `pdfoxide_timestamp` handle.
#' @export
pdf_signature_timestamp <- function(sig) {
  structure(.Call(C_r_signature_get_timestamp, sig),
            class = "pdfoxide_timestamp")
}

#' Attach a timestamp to a signature.
#' @param sig A `pdfoxide_signature`.
#' @param timestamp A `pdfoxide_timestamp`.
#' @return A logical scalar (TRUE on success).
#' @export
pdf_signature_add_timestamp <- function(sig, timestamp) {
  if (!inherits(timestamp, "pdfoxide_timestamp"))
    stop("pdf_signature_add_timestamp: expected a pdfoxide_timestamp")
  .Call(C_r_signature_add_timestamp, sig, timestamp)
}

#' Verify a signature's signer-attributes crypto check.
#'
#' Returns `1` valid, `0` invalid, `-1` unknown / unsupported.
#' @param sig A `pdfoxide_signature`.
#' @export
pdf_signature_verify <- function(sig) .Call(C_r_signature_verify, sig)

#' Verify a signature end-to-end against the full PDF bytes.
#'
#' Returns `1` valid, `0` invalid, `-1` unknown / unsupported.
#' @param sig A `pdfoxide_signature`.
#' @param pdf A `raw` vector of the full PDF.
#' @export
pdf_signature_verify_detached <- function(sig, pdf) {
  .Call(C_r_signature_verify_detached, sig, pdf)
}

#' Free a signature handle now (idempotent).
#' @param sig A `pdfoxide_signature`.
#' @export
pdf_signature_close <- function(sig) {
  if (!inherits(sig, "pdfoxide_signature"))
    stop("pdf_signature_close: expected a pdfoxide_signature")
  invisible(.Call(C_r_signature_close, sig))
}

# ── Timestamp ──

#' Parse a DER-encoded RFC 3161 TimeStampToken into a timestamp handle.
#' @param bytes A `raw` vector of DER TimeStampToken / TSTInfo.
#' @return A `pdfoxide_timestamp` handle.
#' @export
pdf_timestamp_parse <- function(bytes) {
  structure(.Call(C_r_timestamp_parse, bytes), class = "pdfoxide_timestamp")
}

#' Raw DER timestamp token / message imprint as a `raw` vector.
#' @param timestamp A `pdfoxide_timestamp`.
#' @export
pdf_timestamp_token <- function(timestamp) {
  .Call(C_r_timestamp_get_token, timestamp)
}
#' @rdname pdf_timestamp_token
#' @export
pdf_timestamp_message_imprint <- function(timestamp) {
  .Call(C_r_timestamp_get_message_imprint, timestamp)
}

#' Timestamp time as epoch seconds.
#' @param timestamp A `pdfoxide_timestamp`.
#' @export
pdf_timestamp_time <- function(timestamp) {
  .Call(C_r_timestamp_get_time, timestamp)
}

#' Timestamp serial number / TSA name / policy OID strings.
#' @param timestamp A `pdfoxide_timestamp`.
#' @export
pdf_timestamp_serial <- function(timestamp) {
  .Call(C_r_timestamp_get_serial, timestamp)
}
#' @rdname pdf_timestamp_serial
#' @export
pdf_timestamp_tsa_name <- function(timestamp) {
  .Call(C_r_timestamp_get_tsa_name, timestamp)
}
#' @rdname pdf_timestamp_serial
#' @export
pdf_timestamp_policy_oid <- function(timestamp) {
  .Call(C_r_timestamp_get_policy_oid, timestamp)
}

#' Timestamp message-imprint hash algorithm code.
#' @param timestamp A `pdfoxide_timestamp`.
#' @export
pdf_timestamp_hash_algorithm <- function(timestamp) {
  .Call(C_r_timestamp_get_hash_algorithm, timestamp)
}

#' Verify a timestamp token.
#' @param timestamp A `pdfoxide_timestamp`.
#' @return A logical scalar.
#' @export
pdf_timestamp_verify <- function(timestamp) {
  .Call(C_r_timestamp_verify, timestamp)
}

#' Free a timestamp handle now (idempotent).
#' @param timestamp A `pdfoxide_timestamp`.
#' @export
pdf_timestamp_close <- function(timestamp) {
  if (!inherits(timestamp, "pdfoxide_timestamp"))
    stop("pdf_timestamp_close: expected a pdfoxide_timestamp")
  invisible(.Call(C_r_timestamp_close, timestamp))
}

# ── TSA client ──

#' Create an RFC 3161 Time-Stamping-Authority client.
#' @param url TSA endpoint URL.
#' @param username,password HTTP auth, or `NULL`.
#' @param timeout Request timeout in seconds.
#' @param hash_algo Hash algorithm code.
#' @param use_nonce Include a nonce?
#' @param cert_req Request the TSA certificate?
#' @return A `pdfoxide_tsa_client` handle.
#' @export
pdf_tsa_client_create <- function(url, username = NULL, password = NULL,
                                  timeout = 30L, hash_algo = 0L,
                                  use_nonce = TRUE, cert_req = TRUE) {
  structure(.Call(C_r_tsa_client_create, url, username, password,
                  as.integer(timeout), as.integer(hash_algo),
                  isTRUE(use_nonce), isTRUE(cert_req)),
            class = "pdfoxide_tsa_client")
}

#' Request a timestamp over raw data bytes.
#' @param client A `pdfoxide_tsa_client`.
#' @param data A `raw` vector to stamp.
#' @return A `pdfoxide_timestamp` handle.
#' @export
pdf_tsa_request_timestamp <- function(client, data) {
  structure(.Call(C_r_tsa_request_timestamp, client, data),
            class = "pdfoxide_timestamp")
}

#' Request a timestamp over a precomputed message-imprint hash.
#' @param client A `pdfoxide_tsa_client`.
#' @param hash A `raw` vector hash.
#' @param hash_algo Hash algorithm code matching `hash`.
#' @return A `pdfoxide_timestamp` handle.
#' @export
pdf_tsa_request_timestamp_hash <- function(client, hash, hash_algo = 0L) {
  structure(.Call(C_r_tsa_request_timestamp_hash, client, hash,
                  as.integer(hash_algo)),
            class = "pdfoxide_timestamp")
}

#' Free a TSA-client handle now (idempotent).
#' @param client A `pdfoxide_tsa_client`.
#' @export
pdf_tsa_client_close <- function(client) {
  if (!inherits(client, "pdfoxide_tsa_client"))
    stop("pdf_tsa_client_close: expected a pdfoxide_tsa_client")
  invisible(.Call(C_r_tsa_client_close, client))
}

# ── DSS (Document Security Store) ──

#' Read the document's `/DSS` into a handle.
#'
#' Returns `NULL` when the document has no DSS (not an error).
#' @param doc A `pdfoxide_document`.
#' @return A `pdfoxide_dss` handle, or `NULL`.
#' @export
pdf_get_dss <- function(doc) {
  h <- .Call(C_r_doc_get_dss, doc)
  if (is.null(h)) return(NULL)
  structure(h, class = "pdfoxide_dss")
}

#' DSS certificate / CRL / OCSP / VRI entry counts.
#' @param dss A `pdfoxide_dss`.
#' @export
pdf_dss_cert_count <- function(dss) .Call(C_r_dss_cert_count, dss)
#' @rdname pdf_dss_cert_count
#' @export
pdf_dss_crl_count <- function(dss) .Call(C_r_dss_crl_count, dss)
#' @rdname pdf_dss_cert_count
#' @export
pdf_dss_ocsp_count <- function(dss) .Call(C_r_dss_ocsp_count, dss)
#' @rdname pdf_dss_cert_count
#' @export
pdf_dss_vri_count <- function(dss) .Call(C_r_dss_vri_count, dss)

#' Get the `index`-th DSS certificate / CRL / OCSP entry (0-based) as `raw` DER.
#' @param dss A `pdfoxide_dss`.
#' @param index 0-based entry index.
#' @export
pdf_dss_get_cert <- function(dss, index) {
  .Call(C_r_dss_get_cert, dss, as.integer(index))
}
#' @rdname pdf_dss_get_cert
#' @export
pdf_dss_get_crl <- function(dss, index) {
  .Call(C_r_dss_get_crl, dss, as.integer(index))
}
#' @rdname pdf_dss_get_cert
#' @export
pdf_dss_get_ocsp <- function(dss, index) {
  .Call(C_r_dss_get_ocsp, dss, as.integer(index))
}

#' Free a DSS handle now (idempotent).
#' @param dss A `pdfoxide_dss`.
#' @export
pdf_dss_close <- function(dss) {
  if (!inherits(dss, "pdfoxide_dss"))
    stop("pdf_dss_close: expected a pdfoxide_dss")
  invisible(.Call(C_r_dss_close, dss))
}

# ── Validation (PDF/A, PDF/UA, PDF/X) ──

#' Validate a document against a PDF/A conformance level.
#'
#' `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
#' @param doc A `pdfoxide_document`.
#' @param level PDF/A level.
#' @return A `pdfoxide_pdf_a_results` handle.
#' @export
pdf_validate_pdf_a <- function(doc, level = 0L) {
  structure(.Call(C_r_validate_pdf_a, doc, as.integer(level)),
            class = "pdfoxide_pdf_a_results")
}

#' Whether the document is PDF/A compliant at the validated level.
#' @param results A `pdfoxide_pdf_a_results`.
#' @return A logical scalar.
#' @export
pdf_a_is_compliant <- function(results) .Call(C_r_pdf_a_is_compliant, results)

#' PDF/A validation errors / warnings as a character vector.
#' @param results A `pdfoxide_pdf_a_results`.
#' @export
pdf_a_errors <- function(results) {
  n <- .Call(C_r_pdf_a_error_count, results)
  if (n <= 0) return(character(0))
  vapply(seq_len(n) - 1L, function(i) .Call(C_r_pdf_a_get_error, results, i),
         character(1))
}
#' @rdname pdf_a_errors
#' @export
pdf_a_warning_count <- function(results) .Call(C_r_pdf_a_warning_count, results)

#' Free a PDF/A results handle now (idempotent).
#' @param results A `pdfoxide_pdf_a_results`.
#' @export
pdf_a_results_close <- function(results) {
  if (!inherits(results, "pdfoxide_pdf_a_results"))
    stop("pdf_a_results_close: expected a pdfoxide_pdf_a_results")
  invisible(.Call(C_r_pdf_a_results_close, results))
}

#' Validate a document against a PDF/UA accessibility level.
#' @param doc A `pdfoxide_document`.
#' @param level PDF/UA level.
#' @return A `pdfoxide_ua_results` handle.
#' @export
pdf_validate_pdf_ua <- function(doc, level = 0L) {
  structure(.Call(C_r_validate_pdf_ua, doc, as.integer(level)),
            class = "pdfoxide_ua_results")
}

#' Whether the document is PDF/UA accessible at the validated level.
#' @param results A `pdfoxide_ua_results`.
#' @return A logical scalar.
#' @export
pdf_ua_is_accessible <- function(results) {
  .Call(C_r_pdf_ua_is_accessible, results)
}

#' PDF/UA validation errors / warnings as a character vector.
#' @param results A `pdfoxide_ua_results`.
#' @export
pdf_ua_errors <- function(results) {
  n <- .Call(C_r_pdf_ua_error_count, results)
  if (n <= 0) return(character(0))
  vapply(seq_len(n) - 1L, function(i) .Call(C_r_pdf_ua_get_error, results, i),
         character(1))
}
#' @rdname pdf_ua_errors
#' @export
pdf_ua_warnings <- function(results) {
  n <- .Call(C_r_pdf_ua_warning_count, results)
  if (n <= 0) return(character(0))
  vapply(seq_len(n) - 1L, function(i) .Call(C_r_pdf_ua_get_warning, results, i),
         character(1))
}

#' PDF/UA structural statistics as a named list.
#' @param results A `pdfoxide_ua_results`.
#' @return `list(struct=, images=, tables=, forms=, annotations=, pages=)`.
#' @export
pdf_ua_stats <- function(results) {
  s <- .Call(C_r_pdf_ua_get_stats, results)
  list(struct = s[1], images = s[2], tables = s[3], forms = s[4],
       annotations = s[5], pages = s[6])
}

#' Free a PDF/UA results handle now (idempotent).
#' @param results A `pdfoxide_ua_results`.
#' @export
pdf_ua_results_close <- function(results) {
  if (!inherits(results, "pdfoxide_ua_results"))
    stop("pdf_ua_results_close: expected a pdfoxide_ua_results")
  invisible(.Call(C_r_pdf_ua_results_close, results))
}

#' Validate a document against a PDF/X conformance level.
#' @param doc A `pdfoxide_document`.
#' @param level PDF/X level.
#' @return A `pdfoxide_pdf_x_results` handle.
#' @export
pdf_validate_pdf_x <- function(doc, level = 0L) {
  structure(.Call(C_r_validate_pdf_x, doc, as.integer(level)),
            class = "pdfoxide_pdf_x_results")
}

#' Whether the document is PDF/X compliant at the validated level.
#' @param results A `pdfoxide_pdf_x_results`.
#' @return A logical scalar.
#' @export
pdf_x_is_compliant <- function(results) .Call(C_r_pdf_x_is_compliant, results)

#' PDF/X validation errors as a character vector.
#' @param results A `pdfoxide_pdf_x_results`.
#' @export
pdf_x_errors <- function(results) {
  n <- .Call(C_r_pdf_x_error_count, results)
  if (n <= 0) return(character(0))
  vapply(seq_len(n) - 1L, function(i) .Call(C_r_pdf_x_get_error, results, i),
         character(1))
}

#' Free a PDF/X results handle now (idempotent).
#' @param results A `pdfoxide_pdf_x_results`.
#' @export
pdf_x_results_close <- function(results) {
  if (!inherits(results, "pdfoxide_pdf_x_results"))
    stop("pdf_x_results_close: expected a pdfoxide_pdf_x_results")
  invisible(.Call(C_r_pdf_x_results_close, results))
}

# ── PHASE-7: barcodes / QR ────────────────────────────────────────────────────
# A `pdfoxide_barcode` is an external pointer to a native FfiBarcodeImage freed by
# the GC finalizer (or now via pdf_barcode_close). `error_correction`: 0=L 1=M
# 2=Q 3=H. `format`: 0=Code128 1=Code39 2=EAN13 ... (see pdf_page_barcode_1d).

#' Generate a QR code.
#' @param data The data to encode.
#' @param error_correction 0=L 1=M 2=Q 3=H.
#' @param size_px Side length in pixels.
#' @return A `pdfoxide_barcode` handle.
#' @export
pdf_generate_qr_code <- function(data, error_correction = 1L, size_px = 256L) {
  structure(.Call(C_r_generate_qr_code, data, as.integer(error_correction),
                  as.integer(size_px)), class = "pdfoxide_barcode")
}

#' Generate a 1-D / 2-D barcode of the given symbology.
#' @param data The data to encode.
#' @param format Symbology code.
#' @param size_px Target size in pixels.
#' @return A `pdfoxide_barcode` handle.
#' @export
pdf_generate_barcode <- function(data, format = 0L, size_px = 256L) {
  structure(.Call(C_r_generate_barcode, data, as.integer(format),
                  as.integer(size_px)), class = "pdfoxide_barcode")
}

#' Decoded barcode data string.
#' @param barcode A `pdfoxide_barcode`.
#' @export
pdf_barcode_get_data <- function(barcode) .Call(C_r_barcode_get_data, barcode)

#' Barcode symbology format code.
#' @param barcode A `pdfoxide_barcode`.
#' @export
pdf_barcode_get_format <- function(barcode) .Call(C_r_barcode_get_format, barcode)

#' Barcode decode confidence (0.0-1.0).
#' @param barcode A `pdfoxide_barcode`.
#' @export
pdf_barcode_get_confidence <- function(barcode) {
  .Call(C_r_barcode_get_confidence, barcode)
}

#' Render the barcode to PNG bytes.
#' @param barcode A `pdfoxide_barcode`.
#' @param size_px Render size in pixels.
#' @return A `raw` vector of PNG bytes.
#' @export
pdf_barcode_get_image_png <- function(barcode, size_px = 256L) {
  .Call(C_r_barcode_get_image_png, barcode, as.integer(size_px))
}

#' Render the barcode to an SVG string.
#' @param barcode A `pdfoxide_barcode`.
#' @param size_px Render size in pixels.
#' @export
pdf_barcode_get_svg <- function(barcode, size_px = 256L) {
  .Call(C_r_barcode_get_svg, barcode, as.integer(size_px))
}

#' Free a barcode handle now (idempotent).
#' @param barcode A `pdfoxide_barcode`.
#' @export
pdf_barcode_close <- function(barcode) {
  if (!inherits(barcode, "pdfoxide_barcode"))
    stop("pdf_barcode_close: expected a pdfoxide_barcode")
  invisible(.Call(C_r_barcode_close, barcode))
}

#' Place a generated barcode onto an editor page at the given rectangle.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param barcode A `pdfoxide_barcode`.
#' @param x,y,width,height Rectangle (pts).
#' @export
pdf_editor_add_barcode_to_page <- function(editor, page, barcode, x, y, width,
                                           height) {
  if (!inherits(barcode, "pdfoxide_barcode"))
    stop("pdf_editor_add_barcode_to_page: expected a pdfoxide_barcode")
  invisible(.Call(C_r_editor_add_barcode_to_page, editor, as.integer(page),
                  barcode, as.double(x), as.double(y), as.double(width),
                  as.double(height)))
}

# ── PHASE-7: OCR ──────────────────────────────────────────────────────────────
# A `pdfoxide_ocr_engine` is an external pointer to a native OCR engine freed by
# the GC finalizer (or now via pdf_ocr_engine_close).

#' Create an OCR engine from detection / recognition / dictionary model files.
#' @param det_model_path,rec_model_path,dict_path Paths to the model files.
#' @return A `pdfoxide_ocr_engine` handle.
#' @export
pdf_ocr_engine_create <- function(det_model_path, rec_model_path, dict_path) {
  structure(.Call(C_r_ocr_engine_create, det_model_path, rec_model_path,
                  dict_path), class = "pdfoxide_ocr_engine")
}

#' Free an OCR engine handle now (idempotent).
#' @param engine A `pdfoxide_ocr_engine`.
#' @export
pdf_ocr_engine_close <- function(engine) {
  if (!inherits(engine, "pdfoxide_ocr_engine"))
    stop("pdf_ocr_engine_close: expected a pdfoxide_ocr_engine")
  invisible(.Call(C_r_ocr_engine_close, engine))
}

#' Whether a (0-based) page needs OCR (i.e. is scanned / hybrid).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @return A logical scalar.
#' @export
pdf_ocr_page_needs_ocr <- function(doc, page) {
  .Call(C_r_ocr_page_needs_ocr, doc, as.integer(page))
}

#' Extract text from a (0-based) page using OCR.
#'
#' `engine` may be `NULL` to use native text extraction only.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param engine A `pdfoxide_ocr_engine` or `NULL`.
#' @export
pdf_ocr_extract_text <- function(doc, page, engine = NULL) {
  .Call(C_r_ocr_extract_text, doc, as.integer(page), engine)
}

# ── PHASE-7: render variants ──────────────────────────────────────────────────
# All return a `pdfoxide_rendered_image` built via new_rendered_image().

#' Render a (0-based) page with the full RenderOptions surface.
#'
#' `format`: 0=PNG 1=JPEG. Background channels are 0.0-1.0; set
#' `transparent_background = TRUE` to drop the fill.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param dpi Resolution.
#' @param format Image format.
#' @param bg_r,bg_g,bg_b,bg_a Background RGBA (0-1).
#' @param transparent_background Drop the background fill?
#' @param render_annotations Render annotations?
#' @param jpeg_quality JPEG quality (1-100), used only for JPEG.
#' @return A `pdfoxide_rendered_image`.
#' @export
pdf_render_page_with_options <- function(doc, page, dpi = 150L, format = 0L,
                                         bg_r = 1, bg_g = 1, bg_b = 1, bg_a = 1,
                                         transparent_background = FALSE,
                                         render_annotations = TRUE,
                                         jpeg_quality = 85L) {
  img <- .Call(C_r_render_page_with_options, doc, as.integer(page),
               as.integer(dpi), as.integer(format), as.double(bg_r),
               as.double(bg_g), as.double(bg_b), as.double(bg_a),
               as.integer(isTRUE(transparent_background)),
               as.integer(isTRUE(render_annotations)), as.integer(jpeg_quality))
  new_rendered_image(img)
}

#' Render a (0-based) page with RenderOptions plus OCG layer filtering.
#'
#' `excluded_layers` is a character vector of OCG `/Name`s to suppress.
#' @inheritParams pdf_render_page_with_options
#' @param excluded_layers Character vector of OCG names to exclude, or `NULL`.
#' @return A `pdfoxide_rendered_image`.
#' @export
pdf_render_page_with_options_ex <- function(doc, page, dpi = 150L, format = 0L,
                                            bg_r = 1, bg_g = 1, bg_b = 1,
                                            bg_a = 1,
                                            transparent_background = FALSE,
                                            render_annotations = TRUE,
                                            jpeg_quality = 85L,
                                            excluded_layers = NULL) {
  layers <- if (is.null(excluded_layers)) NULL else as.character(excluded_layers)
  img <- .Call(C_r_render_page_with_options_ex, doc, as.integer(page),
               as.integer(dpi), as.integer(format), as.double(bg_r),
               as.double(bg_g), as.double(bg_b), as.double(bg_a),
               as.integer(isTRUE(transparent_background)),
               as.integer(isTRUE(render_annotations)), as.integer(jpeg_quality),
               layers)
  new_rendered_image(img)
}

#' Render a rectangular region of a (0-based) page (crop in user-space points).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param crop_x,crop_y,crop_width,crop_height Region in user-space points.
#' @param format Image format (0=PNG 1=JPEG).
#' @return A `pdfoxide_rendered_image`.
#' @export
pdf_render_page_region <- function(doc, page, crop_x, crop_y, crop_width,
                                   crop_height, format = 0L) {
  img <- .Call(C_r_render_page_region, doc, as.integer(page), as.double(crop_x),
               as.double(crop_y), as.double(crop_width), as.double(crop_height),
               as.integer(format))
  new_rendered_image(img)
}

#' Render a (0-based) page to fit inside `w` x `h` pixels (aspect-preserving).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param w,h Bounding box in pixels.
#' @param format Image format.
#' @return A `pdfoxide_rendered_image`.
#' @export
pdf_render_page_fit <- function(doc, page, w, h, format = 0L) {
  img <- .Call(C_r_render_page_fit, doc, as.integer(page), as.integer(w),
               as.integer(h), as.integer(format))
  new_rendered_image(img)
}

#' Render a (0-based) page to a raw premultiplied RGBA8888 pixel buffer.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param dpi Resolution.
#' @return A `pdfoxide_rendered_image` (its `data` is the raw RGBA buffer).
#' @export
pdf_render_page_raw <- function(doc, page, dpi = 150L) {
  img <- .Call(C_r_render_page_raw, doc, as.integer(page), as.integer(dpi))
  new_rendered_image(img)
}

# ── PHASE-7: renderer + estimate ──────────────────────────────────────────────
# A `pdfoxide_renderer` is an external pointer to a native renderer freed by the
# GC finalizer (or now via pdf_renderer_close).

#' Create a reusable renderer.
#' @param dpi Resolution.
#' @param format Image format (0=PNG 1=JPEG).
#' @param quality JPEG quality (1-100).
#' @param anti_alias Enable anti-aliasing?
#' @return A `pdfoxide_renderer` handle.
#' @export
pdf_create_renderer <- function(dpi = 150L, format = 0L, quality = 85L,
                                anti_alias = TRUE) {
  structure(.Call(C_r_create_renderer, as.integer(dpi), as.integer(format),
                  as.integer(quality), isTRUE(anti_alias)),
            class = "pdfoxide_renderer")
}

#' Free a renderer handle now (idempotent).
#' @param renderer A `pdfoxide_renderer`.
#' @export
pdf_renderer_close <- function(renderer) {
  if (!inherits(renderer, "pdfoxide_renderer"))
    stop("pdf_renderer_close: expected a pdfoxide_renderer")
  invisible(.Call(C_r_renderer_close, renderer))
}

#' Estimate the render time (ms) for a (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_estimate_render_time <- function(doc, page) {
  .Call(C_r_estimate_render_time, doc, as.integer(page))
}

# ── PHASE-7: redaction (methods on pdfoxide_editor) ───────────────────────────

#' Queue a redaction rectangle on a (0-based) page (corners + overlay RGB).
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @param x1,y1,x2,y2 Rectangle corners in user-space points.
#' @param r,g,b Overlay fill colour (0-1).
#' @export
pdf_redaction_add <- function(editor, page, x1, y1, x2, y2, r = 0, g = 0, b = 0) {
  invisible(.Call(C_r_redaction_add, editor, as.integer(page), as.double(x1),
                  as.double(y1), as.double(x2), as.double(y2), as.double(r),
                  as.double(g), as.double(b)))
}

#' Number of queued redaction regions for a (0-based) page.
#' @param editor A `pdfoxide_editor`.
#' @param page 0-based page index.
#' @export
pdf_redaction_count <- function(editor, page) {
  .Call(C_r_redaction_count, editor, as.integer(page))
}

#' Destructively apply all queued redactions; returns glyphs removed.
#' @param editor A `pdfoxide_editor`.
#' @param scrub_metadata Also scrub metadata?
#' @param r,g,b Overlay fill colour (0-1).
#' @return The number of glyphs physically removed.
#' @export
pdf_redaction_apply <- function(editor, scrub_metadata = FALSE, r = 0, g = 0,
                                b = 0) {
  .Call(C_r_redaction_apply, editor, isTRUE(scrub_metadata), as.double(r),
        as.double(g), as.double(b))
}

#' Strip document metadata / JavaScript / embedded files; returns count removed.
#' @param editor A `pdfoxide_editor`.
#' @export
pdf_redaction_scrub_metadata <- function(editor) {
  .Call(C_r_redaction_scrub_metadata, editor)
}

# ── PHASE-7: constructors ─────────────────────────────────────────────────────

#' Build a single-page PDF from an image file.
#' @param path Path to an image file.
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_image <- function(path) {
  structure(.Call(C_r_pdf_from_image, path), class = "pdfoxide_pdf")
}

#' Build a single-page PDF from in-memory image bytes.
#' @param bytes A `raw` vector of image bytes.
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_image_bytes <- function(bytes) {
  structure(.Call(C_r_pdf_from_image_bytes, bytes), class = "pdfoxide_pdf")
}

#' Build a PDF from HTML + CSS, optionally with one embedded font.
#' @param html HTML source.
#' @param css CSS source.
#' @param font_bytes Optional `raw` vector of TTF / OTF bytes, or `NULL`.
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_html_css <- function(html, css, font_bytes = NULL) {
  fb <- if (is.null(font_bytes)) NULL else as.raw(font_bytes)
  structure(.Call(C_r_pdf_from_html_css, html, css, fb), class = "pdfoxide_pdf")
}

#' Build a PDF from HTML + CSS with a multi-font cascade.
#'
#' `families` is a character vector of family names; `font_bytes` is a parallel
#' list of `raw` vectors of font bytes (same length).
#' @param html HTML source.
#' @param css CSS source.
#' @param families Character vector of font-family names.
#' @param font_bytes List of `raw` vectors of font bytes (parallel to `families`).
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_html_css_with_fonts <- function(html, css, families, font_bytes) {
  structure(.Call(C_r_pdf_from_html_css_with_fonts, html, css,
                  as.character(families), font_bytes), class = "pdfoxide_pdf")
}

#' Merge multiple PDF files (by path) into a single PDF.
#' @param paths Character vector of input PDF paths.
#' @return A `raw` vector of the merged PDF bytes.
#' @export
pdf_merge <- function(paths) {
  .Call(C_r_pdf_merge, as.character(paths))
}

# ── PHASE-7: page getters ─────────────────────────────────────────────────────

#' Page width in user-space points for a (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_page_get_width <- function(doc, page) {
  .Call(C_r_page_get_width, doc, as.integer(page))
}

#' Page height in user-space points for a (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_page_get_height <- function(doc, page) {
  .Call(C_r_page_get_height, doc, as.integer(page))
}

#' Page rotation in degrees for a (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_page_get_rotation <- function(doc, page) {
  .Call(C_r_page_get_rotation, doc, as.integer(page))
}

#' Extract layout elements for a (0-based) page.
#'
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @return A list of `Element` records, each `list(type=, text=, rect=)` where
#'   `rect` is `list(x=, y=, width=, height=)`.
#' @export
pdf_page_get_elements <- function(doc, page) {
  .Call(C_r_page_get_elements, doc, as.integer(page))
}

# ── PHASE-7: timestamp ────────────────────────────────────────────────────────

#' Add an RFC-3161 timestamp to an existing signature in a PDF.
#' @param pdf_data A `raw` vector of the PDF bytes.
#' @param sig_index 0-based signature index.
#' @param tsa_url The TSA URL.
#' @return A `raw` vector of the timestamped PDF bytes.
#' @export
pdf_add_timestamp <- function(pdf_data, sig_index, tsa_url) {
  .Call(C_r_add_timestamp, as.raw(pdf_data), as.integer(sig_index), tsa_url)
}

# ══ PHASE-8: 100%-coverage closeout ════════════════════════════════════════════

# ── Office round-trip ─────────────────────────────────────────────────────────

#' Open a `pdfoxide_document` from DOCX / PPTX / XLSX bytes.
#' @param bytes A `raw` vector of office-document bytes.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open_from_docx_bytes <- function(bytes) {
  structure(.Call(C_r_doc_open_from_docx_bytes, as.raw(bytes)),
            class = "pdfoxide_document")
}
#' @rdname pdf_open_from_docx_bytes
#' @export
pdf_open_from_pptx_bytes <- function(bytes) {
  structure(.Call(C_r_doc_open_from_pptx_bytes, as.raw(bytes)),
            class = "pdfoxide_document")
}
#' @rdname pdf_open_from_docx_bytes
#' @export
pdf_open_from_xlsx_bytes <- function(bytes) {
  structure(.Call(C_r_doc_open_from_xlsx_bytes, as.raw(bytes)),
            class = "pdfoxide_document")
}

#' Convert the whole document to DOCX / PPTX / XLSX bytes.
#' @param doc A `pdfoxide_document`.
#' @return A `raw` vector of office-document bytes.
#' @export
pdf_to_docx <- function(doc) .Call(C_r_doc_to_docx, doc)
#' @rdname pdf_to_docx
#' @export
pdf_to_pptx <- function(doc) .Call(C_r_doc_to_pptx, doc)
#' @rdname pdf_to_docx
#' @export
pdf_to_xlsx <- function(doc) .Call(C_r_doc_to_xlsx, doc)

# ── In-rect extractors ────────────────────────────────────────────────────────

#' Extract reading-order text within a rectangle on a (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param x,y,width,height The rectangle in user-space points.
#' @export
pdf_extract_text_in_rect <- function(doc, page, x, y, width, height) {
  .Call(C_r_doc_extract_text_in_rect, doc, as.integer(page),
        as.double(x), as.double(y), as.double(width), as.double(height))
}
#' Extract words within a rectangle on a (0-based) page.
#' @inheritParams pdf_extract_text_in_rect
#' @return A list of `Word` records (`list(text=, bbox=)`).
#' @export
pdf_extract_words_in_rect <- function(doc, page, x, y, width, height) {
  .Call(C_r_doc_extract_words_in_rect, doc, as.integer(page),
        as.double(x), as.double(y), as.double(width), as.double(height))
}
#' Extract text lines within a rectangle on a (0-based) page.
#' @inheritParams pdf_extract_text_in_rect
#' @return A list of `TextLine` records.
#' @export
pdf_extract_lines_in_rect <- function(doc, page, x, y, width, height) {
  .Call(C_r_doc_extract_lines_in_rect, doc, as.integer(page),
        as.double(x), as.double(y), as.double(width), as.double(height))
}
#' Extract tables within a rectangle on a (0-based) page.
#' @inheritParams pdf_extract_text_in_rect
#' @return A list of `Table` records.
#' @export
pdf_extract_tables_in_rect <- function(doc, page, x, y, width, height) {
  .Call(C_r_doc_extract_tables_in_rect, doc, as.integer(page),
        as.double(x), as.double(y), as.double(width), as.double(height))
}
#' Extract images within a rectangle on a (0-based) page.
#' @inheritParams pdf_extract_text_in_rect
#' @return A list of `Image` records.
#' @export
pdf_extract_images_in_rect <- function(doc, page, x, y, width, height) {
  .Call(C_r_doc_extract_images_in_rect, doc, as.integer(page),
        as.double(x), as.double(y), as.double(width), as.double(height))
}

# ── Auto extraction + classification ──────────────────────────────────────────

#' Extract all text from the document (whole-document, reading order).
#' @param doc A `pdfoxide_document`.
#' @export
pdf_extract_all_text <- function(doc) .Call(C_r_doc_extract_all_text, doc)

#' Auto-routed (text-vs-OCR) text extraction for one (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_extract_text_auto <- function(doc, page) {
  .Call(C_r_doc_extract_text_auto, doc, as.integer(page))
}

#' Rich per-page auto extraction (JSON `PageExtraction`).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param options_json `{}`-tolerant `AutoExtractOptions` JSON, or `NULL`.
#' @export
pdf_extract_page_auto <- function(doc, page, options_json = NULL) {
  .Call(C_r_doc_extract_page_auto, doc, as.integer(page), options_json)
}

#' Cheap per-page text-vs-OCR classification (JSON `PageClassification`).
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_classify_page <- function(doc, page) {
  .Call(C_r_doc_classify_page, doc, as.integer(page))
}

#' Cheap whole-document classification (JSON `DocumentClassification`).
#' @param doc A `pdfoxide_document`.
#' @export
pdf_classify_document <- function(doc) .Call(C_r_doc_classify_document, doc)

# ── Header / footer / artifact removal ────────────────────────────────────────

#' Remove repeating headers / footers / artifacts across the document.
#' @param doc A `pdfoxide_document`.
#' @param threshold Repetition threshold (0..1).
#' @return The number of affected pages.
#' @export
pdf_remove_headers <- function(doc, threshold = 0.5) {
  .Call(C_r_doc_remove_headers, doc, as.double(threshold))
}
#' @rdname pdf_remove_headers
#' @export
pdf_remove_footers <- function(doc, threshold = 0.5) {
  .Call(C_r_doc_remove_footers, doc, as.double(threshold))
}
#' @rdname pdf_remove_headers
#' @export
pdf_remove_artifacts <- function(doc, threshold = 0.5) {
  .Call(C_r_doc_remove_artifacts, doc, as.double(threshold))
}

#' Erase the header / footer / artifacts on a single (0-based) page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_erase_header <- function(doc, page) {
  .Call(C_r_doc_erase_header, doc, as.integer(page))
}
#' @rdname pdf_erase_header
#' @export
pdf_erase_footer <- function(doc, page) {
  .Call(C_r_doc_erase_footer, doc, as.integer(page))
}
#' @rdname pdf_erase_header
#' @export
pdf_erase_artifacts <- function(doc, page) {
  .Call(C_r_doc_erase_artifacts, doc, as.integer(page))
}

# ── Forms ─────────────────────────────────────────────────────────────────────

#' List the document's AcroForm fields.
#' @param doc A `pdfoxide_document`.
#' @return A list of `FormField` records
#'   (`list(name=, type=, value=, readonly=, required=)`).
#' @export
pdf_get_form_fields <- function(doc) .Call(C_r_doc_get_form_fields, doc)

#' Export form data as FDF (0) or XFDF (1) bytes.
#' @param doc A `pdfoxide_document`.
#' @param format_type 0=FDF, 1=XFDF.
#' @return A `raw` vector.
#' @export
pdf_export_form_data_to_bytes <- function(doc, format_type = 0L) {
  .Call(C_r_doc_export_form_data_to_bytes, doc, as.integer(format_type))
}
#' Import form data from a file path into the document.
#' @param doc A `pdfoxide_document`.
#' @param data_path Path to FDF/XFDF data.
#' @export
pdf_import_form_data <- function(doc, data_path) {
  invisible(.Call(C_r_doc_import_form_data, doc, data_path))
}
#' Import form fields from a file into the document.
#' @param doc A `pdfoxide_document`.
#' @param filename Path to the form file.
#' @return `TRUE` on success.
#' @export
pdf_form_import_from_file <- function(doc, filename) {
  .Call(C_r_form_import_from_file, doc, filename)
}
#' Import FDF / XFDF bytes into an editor's form.
#' @param editor A `pdfoxide_editor`.
#' @param bytes A `raw` vector of FDF/XFDF.
#' @export
pdf_editor_import_fdf_bytes <- function(editor, bytes) {
  invisible(.Call(C_r_editor_import_fdf_bytes, editor, as.raw(bytes)))
}
#' @rdname pdf_editor_import_fdf_bytes
#' @export
pdf_editor_import_xfdf_bytes <- function(editor, bytes) {
  invisible(.Call(C_r_editor_import_xfdf_bytes, editor, as.raw(bytes)))
}

# ── Document structure / metadata ─────────────────────────────────────────────

#' Document outline (bookmarks) as JSON.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_get_outline <- function(doc) .Call(C_r_doc_get_outline, doc)
#' Page labels as JSON.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_get_page_labels <- function(doc) .Call(C_r_doc_get_page_labels, doc)
#' XMP metadata as JSON.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_get_xmp_metadata <- function(doc) .Call(C_r_doc_get_xmp_metadata, doc)
#' Current source bytes of the document (after any in-place conversion).
#' @param doc A `pdfoxide_document`.
#' @return A `raw` vector.
#' @export
pdf_get_source_bytes <- function(doc) .Call(C_r_doc_get_source_bytes, doc)
#' Whether the document carries XFA forms.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_has_xfa <- function(doc) .Call(C_r_doc_has_xfa, doc)
#' Plan a bookmark-based split (JSON array of segments).
#' @param doc A `pdfoxide_document`.
#' @param options_json `{}`-tolerant options JSON, or `NULL`.
#' @export
pdf_plan_split_by_bookmarks <- function(doc, options_json = NULL) {
  .Call(C_r_doc_plan_split_by_bookmarks, doc, options_json)
}
#' Number of pages of a built `pdfoxide_pdf` (not a document).
#' @param pdf A `pdfoxide_pdf`.
#' @export
pdf_get_page_count <- function(pdf) .Call(C_r_pdf_get_page_count, pdf)

# ── Signatures on a document ──────────────────────────────────────────────────

#' Sign the document in place with a loaded certificate.
#' @param doc A `pdfoxide_document`.
#' @param certificate A `pdfoxide_certificate`.
#' @param reason,location Optional strings.
#' @export
pdf_sign <- function(doc, certificate, reason = NULL, location = NULL) {
  invisible(.Call(C_r_doc_sign, doc, certificate, reason, location))
}
#' Verify all signatures in the document.
#' @param doc A `pdfoxide_document`.
#' @return An integer status (1 valid / 0 invalid / -1 unknown).
#' @export
pdf_verify_all_signatures <- function(doc) {
  .Call(C_r_doc_verify_all_signatures, doc)
}
#' Whether the document carries a document-scoped archival timestamp.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_has_timestamp <- function(doc) .Call(C_r_doc_has_timestamp, doc)

# ── Annotation extras (page + index addressed) ────────────────────────────────

#' Annotation color (packed 0xRRGGBB) for the i-th annotation on a page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page.
#' @param index 0-based annotation index.
#' @export
pdf_annotation_get_color <- function(doc, page, index) {
  .Call(C_r_annotation_get_color, doc, as.integer(page), as.integer(index))
}
#' Annotation creation date (epoch seconds).
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_get_creation_date <- function(doc, page, index) {
  .Call(C_r_annotation_get_creation_date, doc, as.integer(page), as.integer(index))
}
#' Annotation modification date (epoch seconds).
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_get_modification_date <- function(doc, page, index) {
  .Call(C_r_annotation_get_modification_date, doc, as.integer(page), as.integer(index))
}
#' Whether the annotation is hidden.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_is_hidden <- function(doc, page, index) {
  .Call(C_r_annotation_is_hidden, doc, as.integer(page), as.integer(index))
}
#' Whether the annotation is marked deleted.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_is_marked_deleted <- function(doc, page, index) {
  .Call(C_r_annotation_is_marked_deleted, doc, as.integer(page), as.integer(index))
}
#' Whether the annotation is printable.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_is_printable <- function(doc, page, index) {
  .Call(C_r_annotation_is_printable, doc, as.integer(page), as.integer(index))
}
#' Whether the annotation is read-only.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_annotation_is_read_only <- function(doc, page, index) {
  .Call(C_r_annotation_is_read_only, doc, as.integer(page), as.integer(index))
}
#' URI of a link annotation.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_link_annotation_get_uri <- function(doc, page, index) {
  .Call(C_r_link_annotation_get_uri, doc, as.integer(page), as.integer(index))
}
#' Icon name of a text annotation.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_text_annotation_get_icon_name <- function(doc, page, index) {
  .Call(C_r_text_annotation_get_icon_name, doc, as.integer(page), as.integer(index))
}
#' Number of quad points of a highlight annotation.
#' @inheritParams pdf_annotation_get_color
#' @export
pdf_highlight_annotation_quad_points_count <- function(doc, page, index) {
  .Call(C_r_highlight_annotation_quad_points_count, doc, as.integer(page), as.integer(index))
}
#' The `quad_index`-th quad point (8 numbers: x1,y1..x4,y4) of a highlight.
#' @inheritParams pdf_annotation_get_color
#' @param quad_index 0-based quad index.
#' @export
pdf_highlight_annotation_quad_point <- function(doc, page, index, quad_index) {
  .Call(C_r_highlight_annotation_quad_point, doc, as.integer(page),
        as.integer(index), as.integer(quad_index))
}
#' All annotations on a page serialized to JSON.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_annotations_to_json <- function(doc, page) {
  .Call(C_r_annotations_to_json, doc, as.integer(page))
}

# ── Element / font / search JSON accessors ────────────────────────────────────

#' Font size of the i-th embedded font on a page.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page.
#' @param index 0-based font index.
#' @export
pdf_font_get_size <- function(doc, page, index) {
  .Call(C_r_font_get_size, doc, as.integer(page), as.integer(index))
}
#' Embedded fonts of a page serialized to JSON.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_fonts_to_json <- function(doc, page) {
  .Call(C_r_fonts_to_json, doc, as.integer(page))
}
#' Layout elements of a page serialized to JSON.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @export
pdf_elements_to_json <- function(doc, page) {
  .Call(C_r_elements_to_json, doc, as.integer(page))
}
#' Search results for a term on a page serialized to JSON.
#' @param doc A `pdfoxide_document`.
#' @param page 0-based page index.
#' @param term Search term.
#' @param case_sensitive Logical.
#' @export
pdf_search_results_to_json <- function(doc, page, term, case_sensitive = FALSE) {
  .Call(C_r_search_results_to_json, doc, as.integer(page), term,
        as.logical(case_sensitive))
}

# ── Crypto / FIPS / governance ────────────────────────────────────────────────

#' Active cryptographic provider name.
#' @export
pdf_crypto_active_provider <- function() .Call(C_r_crypto_active_provider)
#' Whether the FIPS provider was compiled in.
#' @export
pdf_crypto_fips_available <- function() .Call(C_r_crypto_fips_available)
#' Install the FIPS provider (returns a status code).
#' @export
pdf_crypto_use_fips <- function() .Call(C_r_crypto_use_fips)
#' Install the runtime crypto policy from its grammar string.
#' @param spec The policy spec string.
#' @return A status code (0 = success).
#' @export
pdf_crypto_set_policy <- function(spec) .Call(C_r_crypto_set_policy, spec)
#' The active crypto policy as its canonical grammar string.
#' @export
pdf_crypto_policy <- function() .Call(C_r_crypto_policy)
#' Comma-joined tokens of algorithms exercised so far (governance inventory).
#' @export
pdf_crypto_inventory <- function() .Call(C_r_crypto_inventory)
#' CycloneDX Cryptographic Bill of Materials (JSON).
#' @export
pdf_crypto_cbom <- function() .Call(C_r_crypto_cbom)

# ── Models / config ───────────────────────────────────────────────────────────

#' Air-gapped OCR model manifest (JSON).
#' @export
pdf_model_manifest <- function() .Call(C_r_model_manifest)
#' Whether this build can download OCR models.
#' @export
pdf_prefetch_available <- function() .Call(C_r_prefetch_available)
#' Prefetch OCR models for the given comma-separated languages.
#' @param languages_csv Comma-separated language codes, or `NULL` for English.
#' @return The model cache directory path.
#' @export
pdf_prefetch_models <- function(languages_csv = NULL) {
  .Call(C_r_prefetch_models, languages_csv)
}
#' Set the global content-stream operator cap. Returns the previous cap.
#' @param limit New cap (negative restores the default).
#' @export
pdf_set_max_ops_per_stream <- function(limit) {
  .Call(C_r_set_max_ops_per_stream, as.double(limit))
}
#' Toggle preservation of unmapped (U+FFFD) glyphs. Returns the previous value.
#' @param preserve 1 to preserve, 0 to filter.
#' @export
pdf_set_preserve_unmapped_glyphs <- function(preserve) {
  .Call(C_r_set_preserve_unmapped_glyphs, as.integer(preserve))
}
#' Convert the document to PDF/A in place.
#' @param doc A `pdfoxide_document`.
#' @param level The PDF/A level.
#' @return `TRUE` on success.
#' @export
pdf_convert_to_pdf_a <- function(doc, level = 2L) {
  .Call(C_r_doc_convert_to_pdf_a, doc, as.integer(level))
}
