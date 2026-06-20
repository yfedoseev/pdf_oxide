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
#' @param pdf A `pdfoxide_pdf`. @param path Output path.
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
#' @param path Path to a PDF. @param password The document password.
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
#' @param doc A `pdfoxide_document`. @param page 0-based page index.
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
#' @param doc A `pdfoxide_document`. @param password The document password.
#' @return A logical scalar.
#' @export
pdf_authenticate <- function(doc, password) {
  .Call(C_r_doc_authenticate, doc, password)
}

# ── Phase-1 element extraction ────────────────────────────────────────────────

#' Extract positioned characters for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Char` records, each `list(character=, bbox=, font_name=,
#'   font_size=)` where `bbox` is `list(x=, y=, width=, height=)` and
#'   `character` is the Unicode codepoint as an integer.
#' @export
pdf_extract_chars <- function(doc, page) {
  .Call(C_r_doc_extract_chars, doc, as.integer(page))
}

#' Extract positioned words for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Word` records, each `list(text=, bbox=, font_name=,
#'   font_size=, bold=)`.
#' @export
pdf_extract_words <- function(doc, page) {
  .Call(C_r_doc_extract_words, doc, as.integer(page))
}

#' Extract reading-order text lines for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `TextLine` records, each `list(text=, bbox=, word_count=)`.
#' @export
pdf_extract_text_lines <- function(doc, page) {
  .Call(C_r_doc_extract_text_lines, doc, as.integer(page))
}

#' Extract detected tables for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
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
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Font` records, each `list(name=, type=, encoding=,
#'   embedded=, subset=)`.
#' @export
pdf_embedded_fonts <- function(doc, page) {
  .Call(C_r_doc_embedded_fonts, doc, as.integer(page))
}

#' Extract embedded images for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Image` records, each `list(width=, height=,
#'   bits_per_component=, format=, colorspace=, data=)` where `data` is a `raw`
#'   vector of the image bytes.
#' @export
pdf_embedded_images <- function(doc, page) {
  .Call(C_r_doc_embedded_images, doc, as.integer(page))
}

#' Extract annotations for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Annotation` records, each `list(type=, subtype=, content=,
#'   author=, rect=, border_width=)` where `rect` is `list(x=, y=, width=,
#'   height=)`.
#' @export
pdf_page_annotations <- function(doc, page) {
  .Call(C_r_doc_page_annotations, doc, as.integer(page))
}

#' Extract vector paths for one (0-based) page.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @return A list of `Path` records, each `list(bbox=, stroke_width=, has_stroke=,
#'   has_fill=, operation_count=)`.
#' @export
pdf_extract_paths <- function(doc, page) {
  .Call(C_r_doc_extract_paths, doc, as.integer(page))
}

#' Search a single (0-based) page for a term.
#'
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @param term The search term. @param case_sensitive Whether to match case.
#' @return A list of `SearchResult` records, each `list(text=, page=, bbox=)`.
#' @export
pdf_search <- function(doc, page, term, case_sensitive = FALSE) {
  .Call(C_r_doc_search, doc, as.integer(page), term,
        isTRUE(case_sensitive))
}

#' Search the whole document for a term.
#'
#' @param doc A `pdfoxide_document`. @param term The search term.
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
#' @param doc A `pdfoxide_document`. @param page 0-based page index (required).
#' @param zoom Scale factor (`render_page_zoom`). @param size Largest side in
#'   pixels (`render_page_thumbnail`). @param format Image format (`0` = PNG).
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
#' @param image A `pdfoxide_rendered_image`. @param path Output file path.
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
#' @param doc A `pdfoxide_document`. @param index 0-based page index (required).
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
#' @param editor A `pdfoxide_editor`. @param value New producer string.
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
#' @param editor A `pdfoxide_editor`. @param value Raw PDF date string.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @export
pdf_editor_delete_page <- function(editor, page) {
  invisible(.Call(C_r_editor_delete_page, editor, as.integer(page)))
}

#' Move a page from one (0-based) index to another.
#' @param editor A `pdfoxide_editor`. @param from,to 0-based page indices.
#' @export
pdf_editor_move_page <- function(editor, from, to) {
  invisible(.Call(C_r_editor_move_page, editor, as.integer(from), as.integer(to)))
}

#' Rotate a single (0-based) page by `degrees` (additive).
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @param degrees Degrees to rotate.
#' @export
pdf_editor_rotate_page_by <- function(editor, page, degrees) {
  invisible(.Call(C_r_editor_rotate_page_by, editor, as.integer(page),
                  as.integer(degrees)))
}

#' Rotate all pages by `degrees` (additive).
#' @param editor A `pdfoxide_editor`. @param degrees Degrees to rotate.
#' @export
pdf_editor_rotate_all_pages <- function(editor, degrees) {
  invisible(.Call(C_r_editor_rotate_all_pages, editor, as.integer(degrees)))
}

#' Get / set the absolute rotation of a (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param left,right,top,bottom Margins.
#' @export
pdf_editor_crop_margins <- function(editor, left, right, top, bottom) {
  invisible(.Call(C_r_editor_crop_margins, editor, as.double(left),
                  as.double(right), as.double(top), as.double(bottom)))
}

#' Get / set the CropBox for a (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @export
pdf_editor_apply_page_redactions <- function(editor, page) {
  invisible(.Call(C_r_editor_apply_page_redactions, editor, as.integer(page)))
}

#' Whether a (0-based) page is marked for redaction.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @return A logical scalar.
#' @export
pdf_editor_is_page_marked_for_redaction <- function(editor, page) {
  .Call(C_r_editor_is_page_marked_for_redaction, editor, as.integer(page))
}

#' Remove the redaction mark from a (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @export
pdf_editor_unmark_page_for_redaction <- function(editor, page) {
  invisible(.Call(C_r_editor_unmark_page_for_redaction, editor, as.integer(page)))
}

#' Erase a single rectangular region on a (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @export
pdf_editor_flatten_forms_on_page <- function(editor, page) {
  invisible(.Call(C_r_editor_flatten_forms_on_page, editor, as.integer(page)))
}

#' Set a form field value.
#' @param editor A `pdfoxide_editor`. @param name Field name. @param value Value.
#' @export
pdf_editor_set_form_field_value <- function(editor, name, value) {
  invisible(.Call(C_r_editor_set_form_field_value, editor, name, value))
}

#' Flatten annotations on a single (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
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
#' @param editor A `pdfoxide_editor`. @param index 0-based warning index.
#' @export
pdf_editor_flatten_warning <- function(editor, index) {
  .Call(C_r_editor_flatten_warning, editor, as.integer(index))
}

#' Whether a (0-based) page is marked for annotation-flatten.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @return A logical scalar.
#' @export
pdf_editor_is_page_marked_for_flatten <- function(editor, page) {
  .Call(C_r_editor_is_page_marked_for_flatten, editor, as.integer(page))
}

#' Remove the flatten mark from a (0-based) page.
#' @param editor A `pdfoxide_editor`. @param page 0-based page index.
#' @export
pdf_editor_unmark_page_for_flatten <- function(editor, page) {
  invisible(.Call(C_r_editor_unmark_page_for_flatten, editor, as.integer(page)))
}

#' Merge pages from another PDF file into this document.
#' @param editor A `pdfoxide_editor`. @param source_path Path to the source PDF.
#' @export
pdf_editor_merge_from <- function(editor, source_path) {
  invisible(.Call(C_r_editor_merge_from, editor, source_path))
}

#' Merge pages from an in-memory PDF (raw vector) into this document.
#' @param editor A `pdfoxide_editor`. @param bytes A `raw` vector.
#' @export
pdf_editor_merge_from_bytes <- function(editor, bytes) {
  invisible(.Call(C_r_editor_merge_from_bytes, editor, bytes))
}

#' Convert the document to PDF/A in place.
#'
#' `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
#' @param editor A `pdfoxide_editor`. @param level PDF/A conformance level.
#' @export
pdf_editor_convert_to_pdf_a <- function(editor, level) {
  invisible(.Call(C_r_editor_convert_to_pdf_a, editor, as.integer(level)))
}

#' Embed a file attachment into the document.
#' @param editor A `pdfoxide_editor`. @param name Attachment name.
#' @param bytes A `raw` vector of the file contents.
#' @export
pdf_editor_embed_file <- function(editor, name, bytes) {
  invisible(.Call(C_r_editor_embed_file, editor, name, bytes))
}

#' Extract a subset of (0-based) pages to a new in-memory PDF.
#' @param editor A `pdfoxide_editor`. @param pages Integer vector of 0-based pages.
#' @return A `raw` vector.
#' @export
pdf_editor_extract_pages_to_bytes <- function(editor, pages) {
  .Call(C_r_editor_extract_pages_to_bytes, editor, as.integer(pages))
}

#' Save the edited document to a path.
#' @param editor A `pdfoxide_editor`. @param path Output path.
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
#' @param editor A `pdfoxide_editor`. @param path Output path.
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
