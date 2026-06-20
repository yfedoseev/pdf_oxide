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

/* ── Phase-1 element extraction ──────────────────────────────────────────────
 * Each returns a list of records (one named list per element) so callers get a
 * data.frame-able structure. The C-ABI LIST handle is freed once with the
 * matching *_list_free after every element has been read; owned char* fields are
 * copied into R strings and freed individually via free_string. */

/* Build a 4-element numeric Bbox list `list(x=, y=, width=, height=)`. */
static SEXP make_bbox(float x, float y, float w, float h) {
    SEXP bb = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_VECTOR_ELT(bb, 0, Rf_ScalarReal(x)); SET_STRING_ELT(nms, 0, Rf_mkChar("x"));
    SET_VECTOR_ELT(bb, 1, Rf_ScalarReal(y)); SET_STRING_ELT(nms, 1, Rf_mkChar("y"));
    SET_VECTOR_ELT(bb, 2, Rf_ScalarReal(w)); SET_STRING_ELT(nms, 2, Rf_mkChar("width"));
    SET_VECTOR_ELT(bb, 3, Rf_ScalarReal(h)); SET_STRING_ELT(nms, 3, Rf_mkChar("height"));
    Rf_setAttrib(bb, R_NamesSymbol, nms);
    UNPROTECT(2);
    return bb;
}

SEXP r_doc_extract_chars(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiCharList *list =
        pdf_document_extract_chars(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "extract_chars");
    int32_t n = pdf_oxide_char_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        uint32_t cp = pdf_oxide_char_get_char(list, i, &code);
        if (code != 0) { pdf_oxide_char_list_free(list); pdfox_raise(code, "extract_chars"); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_char_get_bbox(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { pdf_oxide_char_list_free(list); pdfox_raise(code, "extract_chars"); }
        code = 0;
        char *fn = pdf_oxide_char_get_font_name(list, i, &code);
        if (!fn) { pdf_oxide_char_list_free(list); pdfox_raise(code, "extract_chars"); }
        code = 0;
        float fs = pdf_oxide_char_get_font_size(list, i, &code);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 4));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
        SET_VECTOR_ELT(rec, 0, Rf_ScalarInteger((int)cp));      SET_STRING_ELT(nms, 0, Rf_mkChar("character"));
        SET_VECTOR_ELT(rec, 1, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 1, Rf_mkChar("bbox"));
        SEXP fnstr = PROTECT(Rf_mkChar(fn)); free_string(fn);
        SET_VECTOR_ELT(rec, 2, Rf_ScalarString(fnstr));         SET_STRING_ELT(nms, 2, Rf_mkChar("font_name"));
        SET_VECTOR_ELT(rec, 3, Rf_ScalarReal(fs));              SET_STRING_ELT(nms, 3, Rf_mkChar("font_size"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_char_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_extract_words(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiWordList *list =
        pdf_document_extract_words(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "extract_words");
    int32_t n = pdf_oxide_word_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *txt = pdf_oxide_word_get_text(list, i, &code);
        if (!txt) { pdf_oxide_word_list_free(list); pdfox_raise(code, "extract_words"); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_word_get_bbox(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { free_string(txt); pdf_oxide_word_list_free(list); pdfox_raise(code, "extract_words"); }
        code = 0;
        char *fn = pdf_oxide_word_get_font_name(list, i, &code);
        if (!fn) { free_string(txt); pdf_oxide_word_list_free(list); pdfox_raise(code, "extract_words"); }
        code = 0;
        float fs = pdf_oxide_word_get_font_size(list, i, &code);
        code = 0;
        bool bold = pdf_oxide_word_is_bold(list, i, &code);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 5));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 5));
        SEXP txtstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(txtstr));        SET_STRING_ELT(nms, 0, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 1, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 1, Rf_mkChar("bbox"));
        SEXP fnstr = PROTECT(Rf_mkChar(fn)); free_string(fn);
        SET_VECTOR_ELT(rec, 2, Rf_ScalarString(fnstr));         SET_STRING_ELT(nms, 2, Rf_mkChar("font_name"));
        SET_VECTOR_ELT(rec, 3, Rf_ScalarReal(fs));              SET_STRING_ELT(nms, 3, Rf_mkChar("font_size"));
        SET_VECTOR_ELT(rec, 4, Rf_ScalarLogical(bold));         SET_STRING_ELT(nms, 4, Rf_mkChar("bold"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(4);
    }
    pdf_oxide_word_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_extract_text_lines(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiTextLineList *list =
        pdf_document_extract_text_lines(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "extract_text_lines");
    int32_t n = pdf_oxide_line_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *txt = pdf_oxide_line_get_text(list, i, &code);
        if (!txt) { pdf_oxide_line_list_free(list); pdfox_raise(code, "extract_text_lines"); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_line_get_bbox(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { free_string(txt); pdf_oxide_line_list_free(list); pdfox_raise(code, "extract_text_lines"); }
        code = 0;
        int32_t wc = pdf_oxide_line_get_word_count(list, i, &code);
        if (code != 0) { free_string(txt); pdf_oxide_line_list_free(list); pdfox_raise(code, "extract_text_lines"); }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 3));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
        SEXP txtstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(txtstr));        SET_STRING_ELT(nms, 0, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 1, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 1, Rf_mkChar("bbox"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarInteger(wc));           SET_STRING_ELT(nms, 2, Rf_mkChar("word_count"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_line_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_extract_tables(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiTableList *list =
        pdf_document_extract_tables(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "extract_tables");
    int32_t n = pdf_oxide_table_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        int32_t rows = pdf_oxide_table_get_row_count(list, i, &code);
        if (code != 0) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables"); }
        code = 0;
        int32_t cols = pdf_oxide_table_get_col_count(list, i, &code);
        if (code != 0) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables"); }
        code = 0;
        bool hdr = pdf_oxide_table_has_header(list, i, &code);
        if (code != 0) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables"); }
        if (rows < 0) rows = 0;
        if (cols < 0) cols = 0;
        /* cells: a rows×cols character matrix (column-major as R expects). */
        SEXP cells = PROTECT(Rf_allocMatrix(STRSXP, rows, cols));
        for (int32_t r = 0; r < rows; r++) {
            for (int32_t c = 0; c < cols; c++) {
                code = 0;
                char *cell = pdf_oxide_table_get_cell_text(list, i, r, c, &code);
                if (!cell) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables"); }
                SET_STRING_ELT(cells, r + c * rows, Rf_mkChar(cell));
                free_string(cell);
            }
        }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 4));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
        SET_VECTOR_ELT(rec, 0, Rf_ScalarInteger(rows));         SET_STRING_ELT(nms, 0, Rf_mkChar("row_count"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarInteger(cols));         SET_STRING_ELT(nms, 1, Rf_mkChar("col_count"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarLogical(hdr));          SET_STRING_ELT(nms, 2, Rf_mkChar("has_header"));
        SET_VECTOR_ELT(rec, 3, cells);                          SET_STRING_ELT(nms, 3, Rf_mkChar("cells"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_table_list_free(list);
    UNPROTECT(1);
    return out;
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
    CDEF(r_doc_extract_chars, 2),
    CDEF(r_doc_extract_words, 2),
    CDEF(r_doc_extract_text_lines, 2),
    CDEF(r_doc_extract_tables, 2),
    CDEF(r_doc_close, 1),
    CDEF(r_pdf_close, 1),
    {NULL, NULL, 0}
};

void R_init_pdfoxide(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
