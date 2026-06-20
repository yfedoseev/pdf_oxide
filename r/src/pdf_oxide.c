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

/* Raise a classed R condition carrying both the C-ABI `code` and the `op`
 * (class "pdfoxide_error"), so callers get the same {code, op} payload the other
 * bindings expose — not a bare message string. */
static void pdfox_raise(int32_t code, const char *op) {
    char msg[256];
    snprintf(msg, sizeof msg, "pdf_oxide: %s failed (error code %d)", op, code);
    SEXP cond = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_VECTOR_ELT(cond, 0, Rf_mkString(msg));       SET_STRING_ELT(nms, 0, Rf_mkChar("message"));
    SET_VECTOR_ELT(cond, 1, R_NilValue);             SET_STRING_ELT(nms, 1, Rf_mkChar("call"));
    SET_VECTOR_ELT(cond, 2, Rf_ScalarInteger(code)); SET_STRING_ELT(nms, 2, Rf_mkChar("code"));
    SET_VECTOR_ELT(cond, 3, Rf_mkString(op));        SET_STRING_ELT(nms, 3, Rf_mkChar("op"));
    Rf_setAttrib(cond, R_NamesSymbol, nms);
    SEXP cls = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(cls, 0, Rf_mkChar("pdfoxide_error"));
    SET_STRING_ELT(cls, 1, Rf_mkChar("error"));
    SET_STRING_ELT(cls, 2, Rf_mkChar("condition"));
    Rf_classgets(cond, cls);
    SEXP call = PROTECT(Rf_lang2(Rf_install("stop"), cond));
    Rf_eval(call, R_BaseEnv);
    UNPROTECT(4); /* not reached */
}

static SEXP take_string(char *s, int32_t code, const char *op) {
    if (s == NULL) pdfox_raise(code, op);
    SEXP out = PROTECT(Rf_mkString(s));
    free_string(s);
    UNPROTECT(1);
    return out;
}

/* ── Pdf builder ─────────────────────────────────────────────────────────── */
SEXP r_pdf_from_markdown(SEXP md) {
    int32_t code = 0;
    Pdf *h = pdf_from_markdown(CHAR(STRING_ELT(md, 0)), &code);
    if (!h) pdfox_raise(code, "from_markdown");
    return wrap_pdf(h);
}
SEXP r_pdf_from_html(SEXP html) {
    int32_t code = 0;
    Pdf *h = pdf_from_html(CHAR(STRING_ELT(html, 0)), &code);
    if (!h) pdfox_raise(code, "from_html");
    return wrap_pdf(h);
}
SEXP r_pdf_from_text(SEXP text) {
    int32_t code = 0;
    Pdf *h = pdf_from_text(CHAR(STRING_ELT(text, 0)), &code);
    if (!h) pdfox_raise(code, "from_text");
    return wrap_pdf(h);
}
SEXP r_pdf_save(SEXP ext, SEXP path) {
    int32_t code = 0;
    if (pdf_save(pdf_ptr(ext), CHAR(STRING_ELT(path, 0)), &code) != 0)
        pdfox_raise(code, "save");
    return R_NilValue;
}
SEXP r_pdf_save_to_bytes(SEXP ext) {
    int32_t code = 0, len = 0;
    uint8_t *p = pdf_save_to_bytes(pdf_ptr(ext), &len, &code);
    if (!p) pdfox_raise(code, "save_to_bytes");
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
    if (!h) pdfox_raise(code, "open");
    return wrap_doc(h);
}
SEXP r_doc_open_from_bytes(SEXP raw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "open_from_bytes");
    return wrap_doc(h);
}
SEXP r_doc_open_with_password(SEXP path, SEXP pw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_with_password(
        CHAR(STRING_ELT(path, 0)), CHAR(STRING_ELT(pw, 0)), &code);
    if (!h) pdfox_raise(code, "open_with_password");
    return wrap_doc(h);
}
SEXP r_doc_page_count(SEXP ext) {
    int32_t code = 0;
    int32_t n = pdf_document_get_page_count(doc_ptr(ext), &code);
    if (n < 0) pdfox_raise(code, "page_count");
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
SEXP r_doc_to_html_all(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_to_html_all(doc_ptr(ext), &code), code,
                       "to_html_all");
}
SEXP r_doc_to_plain_text_all(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_to_plain_text_all(doc_ptr(ext), &code), code,
                       "to_plain_text_all");
}
/* authenticate returns false for a wrong password WITHOUT an error; the bool is
 * the result. We only raise if the C-ABI signals a real failure via error_code,
 * matching how the other bindings treat this method. */
SEXP r_doc_authenticate(SEXP ext, SEXP pw) {
    int32_t code = 0;
    bool ok = pdf_document_authenticate(doc_ptr(ext), CHAR(STRING_ELT(pw, 0)),
                                        &code);
    if (!ok && code != 0) pdfox_raise(code, "authenticate");
    return Rf_ScalarLogical(ok);
}
SEXP r_doc_extract_structured_json(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(pdf_document_extract_structured_to_json(
                           doc_ptr(ext), Rf_asInteger(page), &code),
                       code, "extract_structured_json");
}

/* Explicit, idempotent close: free the native handle now and clear the external
 * pointer so the GC finalizer is a no-op and later use raises "handle is closed". */
SEXP r_doc_close(SEXP ext) {
    PdfDocument *h = (PdfDocument *)R_ExternalPtrAddr(ext);
    if (h) { pdf_document_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
SEXP r_pdf_close(SEXP ext) {
    Pdf *h = (Pdf *)R_ExternalPtrAddr(ext);
    if (h) { pdf_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
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
    CDEF(r_doc_to_html_all, 1),
    CDEF(r_doc_to_plain_text_all, 1),
    CDEF(r_doc_authenticate, 2),
    CDEF(r_doc_extract_structured_json, 2),
    CDEF(r_doc_close, 1),
    CDEF(r_pdf_close, 1),
    {NULL, NULL, 0}
};

void R_init_pdfoxide(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
