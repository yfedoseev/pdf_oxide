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
