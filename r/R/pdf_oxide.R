#' pdf_oxide — idiomatic R bindings for fast PDF text/Markdown/HTML extraction.
#'
#' Wraps the pdf_oxide C ABI. Handles are external pointers freed by the GC.
#' Page indices are 0-based to match the underlying engine.
#'
#' @useDynLib pdfoxide, .registration = TRUE
#' @keywords internal
"_PACKAGE"

# ── Pdf builder ───────────────────────────────────────────────────────────────

#' Build a PDF from Markdown / HTML / plain text.
#' @param markdown,html,text Source string.
#' @return A `pdfoxide_pdf` handle.
#' @export
pdf_from_markdown <- function(markdown) {
  structure(.Call("r_pdf_from_markdown", markdown), class = "pdfoxide_pdf")
}
#' @rdname pdf_from_markdown
#' @export
pdf_from_html <- function(html) {
  structure(.Call("r_pdf_from_html", html), class = "pdfoxide_pdf")
}
#' @rdname pdf_from_markdown
#' @export
pdf_from_text <- function(text) {
  structure(.Call("r_pdf_from_text", text), class = "pdfoxide_pdf")
}

#' Save a built PDF to a path.
#' @param pdf A `pdfoxide_pdf`. @param path Output path.
#' @export
pdf_save <- function(pdf, path) {
  invisible(.Call("r_pdf_save", pdf, path))
}

#' Serialize a built PDF to a raw vector.
#' @param pdf A `pdfoxide_pdf`.
#' @return A `raw` vector.
#' @export
pdf_save_to_bytes <- function(pdf) {
  .Call("r_pdf_save_to_bytes", pdf)
}

# ── Document ──────────────────────────────────────────────────────────────────

#' Open a PDF document for extraction.
#' @param path Path to a PDF. @param password Optional password.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open <- function(path, password = NULL) {
  h <- if (is.null(password)) {
    .Call("r_doc_open", path)
  } else {
    .Call("r_doc_open_with_password", path, password)
  }
  structure(h, class = "pdfoxide_document")
}

#' Open a PDF document from a raw vector.
#' @param bytes A `raw` vector.
#' @return A `pdfoxide_document` handle.
#' @export
pdf_open_bytes <- function(bytes) {
  structure(.Call("r_doc_open_from_bytes", bytes), class = "pdfoxide_document")
}

#' Number of pages.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_page_count <- function(doc) .Call("r_doc_page_count", doc)

#' PDF version as `c(major, minor)`.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_version <- function(doc) .Call("r_doc_version", doc)

#' Whether the document is encrypted.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_is_encrypted <- function(doc) .Call("r_doc_is_encrypted", doc)

#' Whether the document has a logical structure tree (tagged PDF).
#' @param doc A `pdfoxide_document`.
#' @export
pdf_has_structure_tree <- function(doc) .Call("r_doc_has_structure_tree", doc)

#' Extract reading-order text for one (0-based) page.
#' @param doc A `pdfoxide_document`. @param page 0-based page index.
#' @export
pdf_extract_text <- function(doc, page = 0L) {
  .Call("r_doc_extract_text", doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_plain_text <- function(doc, page = 0L) {
  .Call("r_doc_to_plain_text", doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_markdown <- function(doc, page = 0L) {
  .Call("r_doc_to_markdown", doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_to_html <- function(doc, page = 0L) {
  .Call("r_doc_to_html", doc, as.integer(page))
}
#' @rdname pdf_extract_text
#' @export
pdf_extract_structured_json <- function(doc, page = 0L) {
  .Call("r_doc_extract_structured_json", doc, as.integer(page))
}

#' Markdown for the whole document.
#' @param doc A `pdfoxide_document`.
#' @export
pdf_to_markdown_all <- function(doc) .Call("r_doc_to_markdown_all", doc)
