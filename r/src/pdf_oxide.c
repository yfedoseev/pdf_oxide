/* pdf_oxide R binding — C shim bridging R's .Call interface to the C ABI.
 *
 * Handles (PdfDocument*, Pdf*) are wrapped in R external pointers with
 * finalizers so the GC frees them. C strings returned by the core are copied
 * into R character vectors and freed via free_string. Non-success C-ABI error
 * codes are raised as R errors. */
#include <R.h>
#include <Rinternals.h>
#include <stdint.h>
#include <string.h>

#include <pdf_oxide_c/pdf_oxide.h>

/* ── external-pointer finalizers ─────────────────────────────────────────── */
static void doc_finalizer(SEXP ext) {
    PdfDocument *h = (PdfDocument *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_document_free(h);
        R_ClearExternalPtr(ext);
    }
}
static void pdf_finalizer(SEXP ext) {
    Pdf *h = (Pdf *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_free(h);
        R_ClearExternalPtr(ext);
    }
}

static SEXP wrap_doc(PdfDocument *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, doc_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static SEXP wrap_pdf(Pdf *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, pdf_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}

static PdfDocument *doc_ptr(SEXP ext) {
    PdfDocument *h = (PdfDocument *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: document handle is closed");
    return h;
}
static Pdf *pdf_ptr(SEXP ext) {
    Pdf *h = (Pdf *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: pdf handle is closed");
    return h;
}

static SEXP take_string(char *s, int32_t code, const char *op) {
    if (s == NULL) Rf_error("pdf_oxide: %s failed (error code %d)", op, code);
    SEXP out = PROTECT(Rf_mkString(s));
    free_string(s);
    UNPROTECT(1);
    return out;
}

/* ── Pdf builder ─────────────────────────────────────────────────────────── */
SEXP r_pdf_from_markdown(SEXP md) {
    int32_t code = 0;
    Pdf *h = pdf_from_markdown(CHAR(STRING_ELT(md, 0)), &code);
    if (!h) Rf_error("pdf_oxide: from_markdown failed (error code %d)", code);
    return wrap_pdf(h);
}
SEXP r_pdf_from_html(SEXP html) {
    int32_t code = 0;
    Pdf *h = pdf_from_html(CHAR(STRING_ELT(html, 0)), &code);
    if (!h) Rf_error("pdf_oxide: from_html failed (error code %d)", code);
    return wrap_pdf(h);
}
SEXP r_pdf_from_text(SEXP text) {
    int32_t code = 0;
    Pdf *h = pdf_from_text(CHAR(STRING_ELT(text, 0)), &code);
    if (!h) Rf_error("pdf_oxide: from_text failed (error code %d)", code);
    return wrap_pdf(h);
}
SEXP r_pdf_save(SEXP ext, SEXP path) {
    int32_t code = 0;
    if (pdf_save(pdf_ptr(ext), CHAR(STRING_ELT(path, 0)), &code) != 0)
        Rf_error("pdf_oxide: save failed (error code %d)", code);
    return R_NilValue;
}
SEXP r_pdf_save_to_bytes(SEXP ext) {
    int32_t code = 0, len = 0;
    uint8_t *p = pdf_save_to_bytes(pdf_ptr(ext), &len, &code);
    if (!p) Rf_error("pdf_oxide: save_to_bytes failed (error code %d)", code);
    R_xlen_t n = len < 0 ? 0 : (R_xlen_t)len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    free_bytes(p);
    UNPROTECT(1);
    return out;
}

/* ── Document ────────────────────────────────────────────────────────────── */
SEXP r_doc_open(SEXP path) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open(CHAR(STRING_ELT(path, 0)), &code);
    if (!h) Rf_error("pdf_oxide: open failed (error code %d)", code);
    return wrap_doc(h);
}
SEXP r_doc_open_from_bytes(SEXP raw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) Rf_error("pdf_oxide: open_from_bytes failed (error code %d)", code);
    return wrap_doc(h);
}
SEXP r_doc_open_with_password(SEXP path, SEXP pw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_with_password(
        CHAR(STRING_ELT(path, 0)), CHAR(STRING_ELT(pw, 0)), &code);
    if (!h) Rf_error("pdf_oxide: open_with_password failed (error code %d)", code);
    return wrap_doc(h);
}
SEXP r_doc_page_count(SEXP ext) {
    int32_t code = 0;
    int32_t n = pdf_document_get_page_count(doc_ptr(ext), &code);
    if (n < 0) Rf_error("pdf_oxide: page_count failed (error code %d)", code);
    return Rf_ScalarInteger(n);
}
SEXP r_doc_version(SEXP ext) {
    uint8_t maj = 0, min = 0;
    pdf_document_get_version(doc_ptr(ext), &maj, &min);
    SEXP out = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(out)[0] = maj;
    INTEGER(out)[1] = min;
    UNPROTECT(1);
    return out;
}
SEXP r_doc_is_encrypted(SEXP ext) {
    return Rf_ScalarLogical(pdf_document_is_encrypted(doc_ptr(ext)));
}
SEXP r_doc_has_structure_tree(SEXP ext) {
    return Rf_ScalarLogical(pdf_document_has_structure_tree(doc_ptr(ext)));
}
SEXP r_doc_extract_text(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_extract_text(doc_ptr(ext), Rf_asInteger(page), &code), code,
        "extract_text");
}
SEXP r_doc_to_plain_text(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_to_plain_text(doc_ptr(ext), Rf_asInteger(page), &code), code,
        "to_plain_text");
}
SEXP r_doc_to_markdown(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_to_markdown(doc_ptr(ext), Rf_asInteger(page), &code), code,
        "to_markdown");
}
SEXP r_doc_to_html(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_to_html(doc_ptr(ext), Rf_asInteger(page), &code), code,
        "to_html");
}
SEXP r_doc_to_markdown_all(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_to_markdown_all(doc_ptr(ext), &code), code,
                       "to_markdown_all");
}
SEXP r_doc_extract_structured_json(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(pdf_document_extract_structured_to_json(
                           doc_ptr(ext), Rf_asInteger(page), &code),
                       code, "extract_structured_json");
}

/* ── Native routine registration (R Writing-R-Extensions §5.4) ──────────────
 * Backs `useDynLib(pdfoxide, .registration = TRUE, .fixes = "C_")` so R resolves
 * each .Call via a registered symbol object rather than a runtime string lookup,
 * and `R CMD check` reports no missing-registration NOTE. */
#include <R_ext/Rdynload.h>

#define CDEF(name, n) {#name, (DL_FUNC) &name, n}
static const R_CallMethodDef CallEntries[] = {
    CDEF(r_pdf_from_markdown, 1),
    CDEF(r_pdf_from_html, 1),
    CDEF(r_pdf_from_text, 1),
    CDEF(r_pdf_save, 2),
    CDEF(r_pdf_save_to_bytes, 1),
    CDEF(r_doc_open, 1),
    CDEF(r_doc_open_from_bytes, 1),
    CDEF(r_doc_open_with_password, 2),
    CDEF(r_doc_page_count, 1),
    CDEF(r_doc_version, 1),
    CDEF(r_doc_is_encrypted, 1),
    CDEF(r_doc_has_structure_tree, 1),
    CDEF(r_doc_extract_text, 2),
    CDEF(r_doc_to_plain_text, 2),
    CDEF(r_doc_to_markdown, 2),
    CDEF(r_doc_to_html, 2),
    CDEF(r_doc_to_markdown_all, 1),
    CDEF(r_doc_extract_structured_json, 2),
    {NULL, NULL, 0}
};

void R_init_pdfoxide(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
