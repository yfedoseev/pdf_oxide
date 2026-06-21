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

/* ── Phase-2 element extraction ──────────────────────────────────────────────
 * Same marshalling contract as Phase-1: open the C-ABI LIST handle, read each
 * record into a named R list, copy owned char* fields with free_string, free the
 * whole list once with the matching *_(list_)free, then return the R list. */

SEXP r_doc_embedded_fonts(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiFontList *list =
        pdf_document_get_embedded_fonts(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "embedded_fonts");
    int32_t n = pdf_oxide_font_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *name = pdf_oxide_font_get_name(list, i, &code);
        if (!name) { pdf_oxide_font_list_free(list); pdfox_raise(code, "embedded_fonts"); }
        code = 0;
        char *type = pdf_oxide_font_get_type(list, i, &code);
        if (!type) { free_string(name); pdf_oxide_font_list_free(list); pdfox_raise(code, "embedded_fonts"); }
        code = 0;
        char *enc = pdf_oxide_font_get_encoding(list, i, &code);
        if (!enc) { free_string(name); free_string(type); pdf_oxide_font_list_free(list); pdfox_raise(code, "embedded_fonts"); }
        code = 0;
        bool emb = pdf_oxide_font_is_embedded(list, i, &code) != 0;
        code = 0;
        bool sub = pdf_oxide_font_is_subset(list, i, &code) != 0;
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 5));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 5));
        SEXP nstr = PROTECT(Rf_mkChar(name)); free_string(name);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(nstr));          SET_STRING_ELT(nms, 0, Rf_mkChar("name"));
        SEXP tstr = PROTECT(Rf_mkChar(type)); free_string(type);
        SET_VECTOR_ELT(rec, 1, Rf_ScalarString(tstr));          SET_STRING_ELT(nms, 1, Rf_mkChar("type"));
        SEXP estr = PROTECT(Rf_mkChar(enc)); free_string(enc);
        SET_VECTOR_ELT(rec, 2, Rf_ScalarString(estr));          SET_STRING_ELT(nms, 2, Rf_mkChar("encoding"));
        SET_VECTOR_ELT(rec, 3, Rf_ScalarLogical(emb));          SET_STRING_ELT(nms, 3, Rf_mkChar("embedded"));
        SET_VECTOR_ELT(rec, 4, Rf_ScalarLogical(sub));          SET_STRING_ELT(nms, 4, Rf_mkChar("subset"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(5);
    }
    pdf_oxide_font_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_embedded_images(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiImageList *list =
        pdf_document_get_embedded_images(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "embedded_images");
    int32_t n = pdf_oxide_image_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        int32_t w = pdf_oxide_image_get_width(list, i, &code);
        if (code != 0) { pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        code = 0;
        int32_t h = pdf_oxide_image_get_height(list, i, &code);
        if (code != 0) { pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        code = 0;
        int32_t bpc = pdf_oxide_image_get_bits_per_component(list, i, &code);
        if (code != 0) { pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        code = 0;
        char *fmt = pdf_oxide_image_get_format(list, i, &code);
        if (!fmt) { pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        code = 0;
        char *cs = pdf_oxide_image_get_colorspace(list, i, &code);
        if (!cs) { free_string(fmt); pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        code = 0;
        int32_t dlen = 0;
        uint8_t *data = pdf_oxide_image_get_data(list, i, &dlen, &code);
        if (!data) { free_string(fmt); free_string(cs); pdf_oxide_image_list_free(list); pdfox_raise(code, "embedded_images"); }
        R_xlen_t dn = dlen < 0 ? 0 : (R_xlen_t)dlen;
        SEXP raw = PROTECT(Rf_allocVector(RAWSXP, dn));
        if (dn) memcpy(RAW(raw), data, (size_t)dn);
        free_bytes(data);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 6));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 6));
        SET_VECTOR_ELT(rec, 0, Rf_ScalarInteger(w));            SET_STRING_ELT(nms, 0, Rf_mkChar("width"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarInteger(h));            SET_STRING_ELT(nms, 1, Rf_mkChar("height"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarInteger(bpc));          SET_STRING_ELT(nms, 2, Rf_mkChar("bits_per_component"));
        SEXP fstr = PROTECT(Rf_mkChar(fmt)); free_string(fmt);
        SET_VECTOR_ELT(rec, 3, Rf_ScalarString(fstr));          SET_STRING_ELT(nms, 3, Rf_mkChar("format"));
        SEXP csstr = PROTECT(Rf_mkChar(cs)); free_string(cs);
        SET_VECTOR_ELT(rec, 4, Rf_ScalarString(csstr));         SET_STRING_ELT(nms, 4, Rf_mkChar("colorspace"));
        SET_VECTOR_ELT(rec, 5, raw);                            SET_STRING_ELT(nms, 5, Rf_mkChar("data"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(4);
    }
    pdf_oxide_image_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_page_annotations(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiAnnotationList *list =
        pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "page_annotations");
    int32_t n = pdf_oxide_annotation_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *type = pdf_oxide_annotation_get_type(list, i, &code);
        if (!type) { pdf_oxide_annotation_list_free(list); pdfox_raise(code, "page_annotations"); }
        code = 0;
        char *subtype = pdf_oxide_annotation_get_subtype(list, i, &code);
        if (!subtype) { free_string(type); pdf_oxide_annotation_list_free(list); pdfox_raise(code, "page_annotations"); }
        code = 0;
        char *content = pdf_oxide_annotation_get_content(list, i, &code);
        if (!content) { free_string(type); free_string(subtype); pdf_oxide_annotation_list_free(list); pdfox_raise(code, "page_annotations"); }
        code = 0;
        char *author = pdf_oxide_annotation_get_author(list, i, &code);
        if (!author) { free_string(type); free_string(subtype); free_string(content); pdf_oxide_annotation_list_free(list); pdfox_raise(code, "page_annotations"); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_annotation_get_rect(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { free_string(type); free_string(subtype); free_string(content); free_string(author); pdf_oxide_annotation_list_free(list); pdfox_raise(code, "page_annotations"); }
        code = 0;
        float bw = pdf_oxide_annotation_get_border_width(list, i, &code);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 6));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 6));
        SEXP tstr = PROTECT(Rf_mkChar(type)); free_string(type);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(tstr));          SET_STRING_ELT(nms, 0, Rf_mkChar("type"));
        SEXP ststr = PROTECT(Rf_mkChar(subtype)); free_string(subtype);
        SET_VECTOR_ELT(rec, 1, Rf_ScalarString(ststr));         SET_STRING_ELT(nms, 1, Rf_mkChar("subtype"));
        SEXP cstr = PROTECT(Rf_mkChar(content)); free_string(content);
        SET_VECTOR_ELT(rec, 2, Rf_ScalarString(cstr));          SET_STRING_ELT(nms, 2, Rf_mkChar("content"));
        SEXP astr = PROTECT(Rf_mkChar(author)); free_string(author);
        SET_VECTOR_ELT(rec, 3, Rf_ScalarString(astr));          SET_STRING_ELT(nms, 3, Rf_mkChar("author"));
        SET_VECTOR_ELT(rec, 4, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 4, Rf_mkChar("rect"));
        SET_VECTOR_ELT(rec, 5, Rf_ScalarReal(bw));              SET_STRING_ELT(nms, 5, Rf_mkChar("border_width"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(5);
    }
    pdf_oxide_annotation_list_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_extract_paths(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiPathList *list =
        pdf_document_extract_paths(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "extract_paths");
    int32_t n = pdf_oxide_path_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_path_get_bbox(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { pdf_oxide_path_list_free(list); pdfox_raise(code, "extract_paths"); }
        code = 0;
        float sw = pdf_oxide_path_get_stroke_width(list, i, &code);
        if (code != 0) { pdf_oxide_path_list_free(list); pdfox_raise(code, "extract_paths"); }
        code = 0;
        bool stroke = pdf_oxide_path_has_stroke(list, i, &code);
        code = 0;
        bool fill = pdf_oxide_path_has_fill(list, i, &code);
        code = 0;
        int32_t opc = pdf_oxide_path_get_operation_count(list, i, &code);
        if (code != 0) { pdf_oxide_path_list_free(list); pdfox_raise(code, "extract_paths"); }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 5));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 5));
        SET_VECTOR_ELT(rec, 0, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 0, Rf_mkChar("bbox"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarReal(sw));              SET_STRING_ELT(nms, 1, Rf_mkChar("stroke_width"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarLogical(stroke));       SET_STRING_ELT(nms, 2, Rf_mkChar("has_stroke"));
        SET_VECTOR_ELT(rec, 3, Rf_ScalarLogical(fill));         SET_STRING_ELT(nms, 3, Rf_mkChar("has_fill"));
        SET_VECTOR_ELT(rec, 4, Rf_ScalarInteger(opc));          SET_STRING_ELT(nms, 4, Rf_mkChar("operation_count"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_path_list_free(list);
    UNPROTECT(1);
    return out;
}

/* Build a SearchResult R list from an FfiSearchResults handle (shared by the
 * page-scoped and document-wide search entry points). Frees the handle. */
static SEXP search_results_to_list(FfiSearchResults *list, const char *op) {
    int32_t code = 0;
    int32_t n = pdf_oxide_search_result_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *txt = pdf_oxide_search_result_get_text(list, i, &code);
        if (!txt) { pdf_oxide_search_result_free(list); pdfox_raise(code, op); }
        code = 0;
        int32_t pg = pdf_oxide_search_result_get_page(list, i, &code);
        if (code != 0) { free_string(txt); pdf_oxide_search_result_free(list); pdfox_raise(code, op); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_search_result_get_bbox(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { free_string(txt); pdf_oxide_search_result_free(list); pdfox_raise(code, op); }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 3));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
        SEXP tstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(tstr));          SET_STRING_ELT(nms, 0, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarInteger(pg));           SET_STRING_ELT(nms, 1, Rf_mkChar("page"));
        SET_VECTOR_ELT(rec, 2, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 2, Rf_mkChar("bbox"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_search_result_free(list);
    UNPROTECT(1);
    return out;
}

SEXP r_doc_search(SEXP ext, SEXP page, SEXP term, SEXP case_sensitive) {
    int32_t code = 0;
    FfiSearchResults *list = pdf_document_search_page(
        doc_ptr(ext), Rf_asInteger(page), CHAR(STRING_ELT(term, 0)),
        Rf_asLogical(case_sensitive) == TRUE, &code);
    if (!list) pdfox_raise(code, "search");
    return search_results_to_list(list, "search");
}

SEXP r_doc_search_all(SEXP ext, SEXP term, SEXP case_sensitive) {
    int32_t code = 0;
    FfiSearchResults *list = pdf_document_search_all(
        doc_ptr(ext), CHAR(STRING_ELT(term, 0)),
        Rf_asLogical(case_sensitive) == TRUE, &code);
    if (!list) pdfox_raise(code, "search_all");
    return search_results_to_list(list, "search_all");
}

/* ── Phase-3 page rendering ───────────────────────────────────────────────────
 * The FfiRenderedImage handle is wrapped in its own external pointer (with a
 * finalizer that calls pdf_rendered_image_free) so the GC frees it. The R-level
 * RenderedImage model reads width/height/data eagerly from the live handle and
 * keeps the handle so save(path) can call pdf_save_rendered_image on it. */
static void rendered_image_finalizer(SEXP ext) {
    FfiRenderedImage *h = (FfiRenderedImage *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_rendered_image_free(h);
        R_ClearExternalPtr(ext);
    }
}
static SEXP wrap_rendered_image(FfiRenderedImage *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, rendered_image_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiRenderedImage *rendered_image_ptr(SEXP ext) {
    FfiRenderedImage *h = (FfiRenderedImage *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: rendered image handle is closed");
    return h;
}

SEXP r_doc_render_page(SEXP ext, SEXP page, SEXP format) {
    int32_t code = 0;
    FfiRenderedImage *img =
        pdf_render_page(doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(format), &code);
    if (!img) pdfox_raise(code, "render_page");
    return wrap_rendered_image(img);
}
SEXP r_doc_render_page_zoom(SEXP ext, SEXP page, SEXP zoom, SEXP format) {
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_zoom(
        doc_ptr(ext), Rf_asInteger(page), (float)Rf_asReal(zoom),
        Rf_asInteger(format), &code);
    if (!img) pdfox_raise(code, "render_page_zoom");
    return wrap_rendered_image(img);
}
SEXP r_doc_render_page_thumbnail(SEXP ext, SEXP page, SEXP size, SEXP format) {
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_thumbnail(
        doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(size),
        Rf_asInteger(format), &code);
    if (!img) pdfox_raise(code, "render_page_thumbnail");
    return wrap_rendered_image(img);
}
SEXP r_rendered_image_width(SEXP ext) {
    int32_t code = 0;
    int32_t w = pdf_get_rendered_image_width(rendered_image_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "rendered_image_width");
    return Rf_ScalarInteger(w);
}
SEXP r_rendered_image_height(SEXP ext) {
    int32_t code = 0;
    int32_t h = pdf_get_rendered_image_height(rendered_image_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "rendered_image_height");
    return Rf_ScalarInteger(h);
}
SEXP r_rendered_image_data(SEXP ext) {
    int32_t code = 0, len = 0;
    uint8_t *p = pdf_get_rendered_image_data(rendered_image_ptr(ext), &len, &code);
    if (!p) pdfox_raise(code, "rendered_image_data");
    R_xlen_t n = len < 0 ? 0 : (R_xlen_t)len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    free_bytes(p);
    UNPROTECT(1);
    return out;
}
SEXP r_rendered_image_save(SEXP ext, SEXP path) {
    int32_t code = 0;
    if (pdf_save_rendered_image(rendered_image_ptr(ext),
                                CHAR(STRING_ELT(path, 0)), &code) != 0)
        pdfox_raise(code, "save_rendered_image");
    return R_NilValue;
}
/* Explicit, idempotent free of a rendered-image handle. */
SEXP r_rendered_image_close(SEXP ext) {
    FfiRenderedImage *h = (FfiRenderedImage *)R_ExternalPtrAddr(ext);
    if (h) { pdf_rendered_image_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
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

/* ── DocumentEditor ──────────────────────────────────────────────────────────
 * Mirrors the PdfDocument/Pdf handle pattern: a static open factory wraps the
 * owned DocumentEditor* in an R external pointer with a finalizer that calls
 * document_editor_free, the same pdfox_raise error helper, take_string +
 * free_string for owned char* returns, free_bytes for owned uint8* returns, and
 * an explicit idempotent close. int32 status codes: 0 = success; a non-zero
 * status OR a set error_code is raised. is_* queries are exposed as logicals. */
static void editor_finalizer(SEXP ext) {
    DocumentEditor *h = (DocumentEditor *)R_ExternalPtrAddr(ext);
    if (h) {
        document_editor_free(h);
        R_ClearExternalPtr(ext);
    }
}
static SEXP wrap_editor(DocumentEditor *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, editor_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static DocumentEditor *editor_ptr(SEXP ext) {
    DocumentEditor *h = (DocumentEditor *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: editor handle is closed");
    return h;
}

/* Open / construct */
SEXP r_editor_open(SEXP path) {
    int32_t code = 0;
    DocumentEditor *h = document_editor_open(CHAR(STRING_ELT(path, 0)), &code);
    if (!h) pdfox_raise(code, "editor_open");
    return wrap_editor(h);
}
SEXP r_editor_open_from_bytes(SEXP raw) {
    int32_t code = 0;
    DocumentEditor *h =
        document_editor_open_from_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "editor_open_from_bytes");
    return wrap_editor(h);
}

/* Inspection */
SEXP r_editor_is_modified(SEXP ext) {
    return Rf_ScalarLogical(document_editor_is_modified(editor_ptr(ext)));
}
SEXP r_editor_source_path(SEXP ext) {
    int32_t code = 0;
    return take_string(document_editor_get_source_path(editor_ptr(ext), &code),
                       code, "editor_source_path");
}
SEXP r_editor_version(SEXP ext) {
    uint8_t maj = 0, min = 0;
    document_editor_get_version(editor_ptr(ext), &maj, &min);
    SEXP out = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(out)[0] = maj;
    INTEGER(out)[1] = min;
    UNPROTECT(1);
    return out;
}
SEXP r_editor_page_count(SEXP ext) {
    int32_t code = 0;
    int32_t n = document_editor_get_page_count(editor_ptr(ext), &code);
    if (n < 0) pdfox_raise(code, "editor_page_count");
    return Rf_ScalarInteger(n);
}

/* Metadata */
SEXP r_editor_get_producer(SEXP ext) {
    int32_t code = 0;
    return take_string(document_editor_get_producer(editor_ptr(ext), &code),
                       code, "editor_get_producer");
}
SEXP r_editor_set_producer(SEXP ext, SEXP value) {
    int32_t code = 0;
    if (document_editor_set_producer(editor_ptr(ext),
                                     CHAR(STRING_ELT(value, 0)), &code) != 0)
        pdfox_raise(code, "editor_set_producer");
    return R_NilValue;
}
SEXP r_editor_get_creation_date(SEXP ext) {
    int32_t code = 0;
    return take_string(document_editor_get_creation_date(editor_ptr(ext), &code),
                       code, "editor_get_creation_date");
}
SEXP r_editor_set_creation_date(SEXP ext, SEXP date_str) {
    int32_t code = 0;
    if (document_editor_set_creation_date(editor_ptr(ext),
                                          CHAR(STRING_ELT(date_str, 0)),
                                          &code) != 0)
        pdfox_raise(code, "editor_set_creation_date");
    return R_NilValue;
}

/* Page operations */
SEXP r_editor_delete_page(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_delete_page(editor_ptr(ext), Rf_asInteger(page),
                                    &code) != 0)
        pdfox_raise(code, "editor_delete_page");
    return R_NilValue;
}
SEXP r_editor_move_page(SEXP ext, SEXP from, SEXP to) {
    int32_t code = 0;
    if (document_editor_move_page(editor_ptr(ext), Rf_asInteger(from),
                                  Rf_asInteger(to), &code) != 0)
        pdfox_raise(code, "editor_move_page");
    return R_NilValue;
}
SEXP r_editor_rotate_page_by(SEXP ext, SEXP page, SEXP degrees) {
    int32_t code = 0;
    if (document_editor_rotate_page_by(editor_ptr(ext), (uintptr_t)Rf_asInteger(page),
                                       Rf_asInteger(degrees), &code) != 0)
        pdfox_raise(code, "editor_rotate_page_by");
    return R_NilValue;
}
SEXP r_editor_rotate_all_pages(SEXP ext, SEXP degrees) {
    int32_t code = 0;
    if (document_editor_rotate_all_pages(editor_ptr(ext), Rf_asInteger(degrees),
                                         &code) != 0)
        pdfox_raise(code, "editor_rotate_all_pages");
    return R_NilValue;
}
SEXP r_editor_set_page_rotation(SEXP ext, SEXP page, SEXP degrees) {
    int32_t code = 0;
    if (document_editor_set_page_rotation(editor_ptr(ext), Rf_asInteger(page),
                                          Rf_asInteger(degrees), &code) != 0)
        pdfox_raise(code, "editor_set_page_rotation");
    return R_NilValue;
}
SEXP r_editor_get_page_rotation(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t deg = document_editor_get_page_rotation(editor_ptr(ext),
                                                    Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "editor_get_page_rotation");
    return Rf_ScalarInteger(deg);
}
SEXP r_editor_crop_margins(SEXP ext, SEXP left, SEXP right, SEXP top, SEXP bottom) {
    int32_t code = 0;
    if (document_editor_crop_margins(editor_ptr(ext), (float)Rf_asReal(left),
                                     (float)Rf_asReal(right), (float)Rf_asReal(top),
                                     (float)Rf_asReal(bottom), &code) != 0)
        pdfox_raise(code, "editor_crop_margins");
    return R_NilValue;
}

/* Box geometry — get returns a Bbox list(x, y, width, height) of doubles. */
static SEXP make_bbox_d(double x, double y, double w, double h) {
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
SEXP r_editor_get_page_crop_box(SEXP ext, SEXP page) {
    int32_t code = 0;
    double x = 0, y = 0, w = 0, h = 0;
    if (document_editor_get_page_crop_box(editor_ptr(ext),
                                          (uintptr_t)Rf_asInteger(page),
                                          &x, &y, &w, &h, &code) != 0)
        pdfox_raise(code, "editor_get_page_crop_box");
    return make_bbox_d(x, y, w, h);
}
SEXP r_editor_set_page_crop_box(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (document_editor_set_page_crop_box(editor_ptr(ext),
                                          (uintptr_t)Rf_asInteger(page),
                                          Rf_asReal(x), Rf_asReal(y),
                                          Rf_asReal(w), Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "editor_set_page_crop_box");
    return R_NilValue;
}
SEXP r_editor_get_page_media_box(SEXP ext, SEXP page) {
    int32_t code = 0;
    double x = 0, y = 0, w = 0, h = 0;
    if (document_editor_get_page_media_box(editor_ptr(ext),
                                           (uintptr_t)Rf_asInteger(page),
                                           &x, &y, &w, &h, &code) != 0)
        pdfox_raise(code, "editor_get_page_media_box");
    return make_bbox_d(x, y, w, h);
}
SEXP r_editor_set_page_media_box(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (document_editor_set_page_media_box(editor_ptr(ext),
                                           (uintptr_t)Rf_asInteger(page),
                                           Rf_asReal(x), Rf_asReal(y),
                                           Rf_asReal(w), Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "editor_set_page_media_box");
    return R_NilValue;
}

/* Redaction */
SEXP r_editor_apply_all_redactions(SEXP ext) {
    int32_t code = 0;
    if (document_editor_apply_all_redactions(editor_ptr(ext), &code) != 0)
        pdfox_raise(code, "editor_apply_all_redactions");
    return R_NilValue;
}
SEXP r_editor_apply_page_redactions(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_apply_page_redactions(editor_ptr(ext),
                                              (uintptr_t)Rf_asInteger(page),
                                              &code) != 0)
        pdfox_raise(code, "editor_apply_page_redactions");
    return R_NilValue;
}
SEXP r_editor_is_page_marked_for_redaction(SEXP ext, SEXP page) {
    int32_t r = document_editor_is_page_marked_for_redaction(
        editor_ptr(ext), (uintptr_t)Rf_asInteger(page));
    if (r < 0) pdfox_raise(r, "editor_is_page_marked_for_redaction");
    return Rf_ScalarLogical(r == 1);
}
SEXP r_editor_unmark_page_for_redaction(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_unmark_page_for_redaction(editor_ptr(ext),
                                                  (uintptr_t)Rf_asInteger(page),
                                                  &code) != 0)
        pdfox_raise(code, "editor_unmark_page_for_redaction");
    return R_NilValue;
}

/* Erase regions */
SEXP r_editor_erase_region(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (document_editor_erase_region(editor_ptr(ext), Rf_asInteger(page),
                                     (float)Rf_asReal(x), (float)Rf_asReal(y),
                                     (float)Rf_asReal(w), (float)Rf_asReal(h),
                                     &code) != 0)
        pdfox_raise(code, "editor_erase_region");
    return R_NilValue;
}
/* `rects` is a flat numeric vector of [x, y, w, h] quads (length must be 4*N). */
SEXP r_editor_erase_regions(SEXP ext, SEXP page, SEXP rects) {
    int32_t code = 0;
    R_xlen_t total = XLENGTH(rects);
    uintptr_t count = (uintptr_t)(total / 4);
    SEXP r = PROTECT(Rf_coerceVector(rects, REALSXP));
    int32_t rc = document_editor_erase_regions(editor_ptr(ext),
                                               (uintptr_t)Rf_asInteger(page),
                                               REAL(r), count, &code);
    UNPROTECT(1);
    if (rc != 0) pdfox_raise(code, "editor_erase_regions");
    return R_NilValue;
}
SEXP r_editor_clear_erase_regions(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_clear_erase_regions(editor_ptr(ext),
                                            (uintptr_t)Rf_asInteger(page),
                                            &code) != 0)
        pdfox_raise(code, "editor_clear_erase_regions");
    return R_NilValue;
}

/* Flatten — forms */
SEXP r_editor_flatten_forms(SEXP ext) {
    int32_t code = 0;
    if (document_editor_flatten_forms(editor_ptr(ext), &code) != 0)
        pdfox_raise(code, "editor_flatten_forms");
    return R_NilValue;
}
SEXP r_editor_flatten_forms_on_page(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_flatten_forms_on_page(editor_ptr(ext), Rf_asInteger(page),
                                              &code) != 0)
        pdfox_raise(code, "editor_flatten_forms_on_page");
    return R_NilValue;
}
SEXP r_editor_set_form_field_value(SEXP ext, SEXP name, SEXP value) {
    int32_t code = 0;
    if (document_editor_set_form_field_value(editor_ptr(ext),
                                             CHAR(STRING_ELT(name, 0)),
                                             CHAR(STRING_ELT(value, 0)),
                                             &code) != 0)
        pdfox_raise(code, "editor_set_form_field_value");
    return R_NilValue;
}

/* Flatten — annotations */
SEXP r_editor_flatten_annotations(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_flatten_annotations(editor_ptr(ext), Rf_asInteger(page),
                                            &code) != 0)
        pdfox_raise(code, "editor_flatten_annotations");
    return R_NilValue;
}
SEXP r_editor_flatten_all_annotations(SEXP ext) {
    int32_t code = 0;
    if (document_editor_flatten_all_annotations(editor_ptr(ext), &code) != 0)
        pdfox_raise(code, "editor_flatten_all_annotations");
    return R_NilValue;
}
SEXP r_editor_flatten_warnings_count(SEXP ext) {
    return Rf_ScalarInteger(
        document_editor_flatten_warnings_count(editor_ptr(ext)));
}
SEXP r_editor_flatten_warning(SEXP ext, SEXP index) {
    int32_t code = 0;
    return take_string(document_editor_flatten_warning(editor_ptr(ext),
                                                       Rf_asInteger(index), &code),
                       code, "editor_flatten_warning");
}
SEXP r_editor_is_page_marked_for_flatten(SEXP ext, SEXP page) {
    int32_t r = document_editor_is_page_marked_for_flatten(
        editor_ptr(ext), (uintptr_t)Rf_asInteger(page));
    if (r < 0) pdfox_raise(r, "editor_is_page_marked_for_flatten");
    return Rf_ScalarLogical(r == 1);
}
SEXP r_editor_unmark_page_for_flatten(SEXP ext, SEXP page) {
    int32_t code = 0;
    if (document_editor_unmark_page_for_flatten(editor_ptr(ext),
                                                (uintptr_t)Rf_asInteger(page),
                                                &code) != 0)
        pdfox_raise(code, "editor_unmark_page_for_flatten");
    return R_NilValue;
}

/* Merge / convert / embed / extract */
SEXP r_editor_merge_from(SEXP ext, SEXP source_path) {
    int32_t code = 0;
    if (document_editor_merge_from(editor_ptr(ext),
                                   CHAR(STRING_ELT(source_path, 0)), &code) != 0)
        pdfox_raise(code, "editor_merge_from");
    return R_NilValue;
}
SEXP r_editor_merge_from_bytes(SEXP ext, SEXP raw) {
    int32_t code = 0;
    if (document_editor_merge_from_bytes(editor_ptr(ext), RAW(raw),
                                         (uintptr_t)XLENGTH(raw), &code) != 0)
        pdfox_raise(code, "editor_merge_from_bytes");
    return R_NilValue;
}
SEXP r_editor_convert_to_pdf_a(SEXP ext, SEXP level) {
    int32_t code = 0;
    if (document_editor_convert_to_pdf_a(editor_ptr(ext), Rf_asInteger(level),
                                         &code) != 0)
        pdfox_raise(code, "editor_convert_to_pdf_a");
    return R_NilValue;
}
SEXP r_editor_embed_file(SEXP ext, SEXP name, SEXP raw) {
    int32_t code = 0;
    if (document_editor_embed_file(editor_ptr(ext), CHAR(STRING_ELT(name, 0)),
                                   RAW(raw), (uintptr_t)XLENGTH(raw), &code) != 0)
        pdfox_raise(code, "editor_embed_file");
    return R_NilValue;
}
SEXP r_editor_extract_pages_to_bytes(SEXP ext, SEXP pages) {
    int32_t code = 0;
    SEXP p = PROTECT(Rf_coerceVector(pages, INTSXP));
    uintptr_t out_len = 0;
    uint8_t *buf = document_editor_extract_pages_to_bytes(
        editor_ptr(ext), INTEGER(p), (uintptr_t)XLENGTH(p), &out_len, &code);
    UNPROTECT(1);
    if (!buf) pdfox_raise(code, "editor_extract_pages_to_bytes");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}

/* Save */
SEXP r_editor_save(SEXP ext, SEXP path) {
    int32_t code = 0;
    if (document_editor_save(editor_ptr(ext), CHAR(STRING_ELT(path, 0)),
                             &code) != 0)
        pdfox_raise(code, "editor_save");
    return R_NilValue;
}
SEXP r_editor_save_to_bytes(SEXP ext) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = document_editor_save_to_bytes(editor_ptr(ext), &out_len, &code);
    if (!buf) pdfox_raise(code, "editor_save_to_bytes");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}
SEXP r_editor_save_to_bytes_with_options(SEXP ext, SEXP compress,
                                         SEXP garbage_collect, SEXP linearize) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = document_editor_save_to_bytes_with_options(
        editor_ptr(ext), Rf_asLogical(compress) == TRUE,
        Rf_asLogical(garbage_collect) == TRUE, Rf_asLogical(linearize) == TRUE,
        &out_len, &code);
    if (!buf) pdfox_raise(code, "editor_save_to_bytes_with_options");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}
SEXP r_editor_save_encrypted(SEXP ext, SEXP path, SEXP user_pw, SEXP owner_pw) {
    int32_t code = 0;
    if (document_editor_save_encrypted(editor_ptr(ext), CHAR(STRING_ELT(path, 0)),
                                       CHAR(STRING_ELT(user_pw, 0)),
                                       CHAR(STRING_ELT(owner_pw, 0)), &code) != 0)
        pdfox_raise(code, "editor_save_encrypted");
    return R_NilValue;
}
SEXP r_editor_save_encrypted_to_bytes(SEXP ext, SEXP user_pw, SEXP owner_pw) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = document_editor_save_encrypted_to_bytes(
        editor_ptr(ext), CHAR(STRING_ELT(user_pw, 0)),
        CHAR(STRING_ELT(owner_pw, 0)), &out_len, &code);
    if (!buf) pdfox_raise(code, "editor_save_encrypted_to_bytes");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}

/* Explicit, idempotent close. */
SEXP r_editor_close(SEXP ext) {
    DocumentEditor *h = (DocumentEditor *)R_ExternalPtrAddr(ext);
    if (h) { document_editor_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── PDF creation builder API ────────────────────────────────────────────────
 * Three owned native handle types, each wrapped in an R external pointer with a
 * GC finalizer, mirroring the PdfDocument/Pdf/DocumentEditor pattern above:
 *
 *   FfiDocumentBuilder  — pdf_document_builder_create / _free
 *   FfiPageBuilder      — documentBuilder.page()/letterPage()/a4Page() / _free
 *   EmbeddedFont        — pdf_embedded_font_from_file/_from_bytes / _free
 *
 * int32 returns are status codes (0 = success); a non-zero return OR a set
 * error_code raises a classed pdfoxide_error via pdfox_raise. Owned uint8*
 * buffers are copied into RAWSXP and released with free_bytes.
 *
 * Ownership subtleties carried over from the header:
 *  - pdf_page_builder_done CONSUMES the page handle (do not _free after); we
 *    clear the external pointer on success so the finalizer is a no-op.
 *  - pdf_document_builder_register_embedded_font CONSUMES the font handle on
 *    success; we clear the font external pointer so it is never double-freed.
 *  - build/save/to_bytes_encrypted consume builder STATE only — the wrapper is
 *    still freed by the finalizer (pdf_document_builder_free). */
static void doc_builder_finalizer(SEXP ext) {
    FfiDocumentBuilder *h = (FfiDocumentBuilder *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_document_builder_free(h);
        R_ClearExternalPtr(ext);
    }
}
static SEXP wrap_doc_builder(FfiDocumentBuilder *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, doc_builder_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiDocumentBuilder *doc_builder_ptr(SEXP ext) {
    FfiDocumentBuilder *h = (FfiDocumentBuilder *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: document-builder handle is closed");
    return h;
}

static void page_builder_finalizer(SEXP ext) {
    FfiPageBuilder *h = (FfiPageBuilder *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_page_builder_free(h);
        R_ClearExternalPtr(ext);
    }
}
static SEXP wrap_page_builder(FfiPageBuilder *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, page_builder_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiPageBuilder *page_builder_ptr(SEXP ext) {
    FfiPageBuilder *h = (FfiPageBuilder *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: page-builder handle is closed");
    return h;
}

static void embedded_font_finalizer(SEXP ext) {
    EmbeddedFont *h = (EmbeddedFont *)R_ExternalPtrAddr(ext);
    if (h) {
        pdf_embedded_font_free(h);
        R_ClearExternalPtr(ext);
    }
}
static SEXP wrap_embedded_font(EmbeddedFont *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, embedded_font_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static EmbeddedFont *embedded_font_ptr(SEXP ext) {
    EmbeddedFont *h = (EmbeddedFont *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: embedded-font handle is closed");
    return h;
}

/* Marshal an R character vector into a heap array of `const char*` pointing at
 * the (R-owned) CHARSXP buffers. The returned array must be R_Free'd; the
 * pointed-at strings stay valid while `vec` is protected by the caller. */
static const char **strvec_to_c(SEXP vec, uintptr_t *out_n) {
    R_xlen_t n = (vec == R_NilValue) ? 0 : XLENGTH(vec);
    *out_n = (uintptr_t)n;
    if (n == 0) return NULL;
    const char **arr = (const char **)R_alloc((size_t)n, sizeof(const char *));
    for (R_xlen_t i = 0; i < n; i++) arr[i] = CHAR(STRING_ELT(vec, i));
    return arr;
}

/* ── EmbeddedFont ── */
SEXP r_embedded_font_from_file(SEXP path) {
    int32_t code = 0;
    EmbeddedFont *h = pdf_embedded_font_from_file(CHAR(STRING_ELT(path, 0)), &code);
    if (!h) pdfox_raise(code, "embedded_font_from_file");
    return wrap_embedded_font(h);
}
SEXP r_embedded_font_from_bytes(SEXP raw, SEXP name) {
    int32_t code = 0;
    const char *nm = (name == R_NilValue) ? NULL : CHAR(STRING_ELT(name, 0));
    EmbeddedFont *h = pdf_embedded_font_from_bytes(
        RAW(raw), (uintptr_t)XLENGTH(raw), nm, &code);
    if (!h) pdfox_raise(code, "embedded_font_from_bytes");
    return wrap_embedded_font(h);
}
SEXP r_embedded_font_close(SEXP ext) {
    EmbeddedFont *h = (EmbeddedFont *)R_ExternalPtrAddr(ext);
    if (h) { pdf_embedded_font_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── DocumentBuilder ── */
SEXP r_builder_create(void) {
    int32_t code = 0;
    FfiDocumentBuilder *h = pdf_document_builder_create(&code);
    if (!h) pdfox_raise(code, "builder_create");
    return wrap_doc_builder(h);
}
SEXP r_builder_close(SEXP ext) {
    FfiDocumentBuilder *h = (FfiDocumentBuilder *)R_ExternalPtrAddr(ext);
    if (h) { pdf_document_builder_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* metadata + catalog setters (all int32 status) */
#define BUILDER_STR_SETTER(rname, cfn, op)                                     \
    SEXP rname(SEXP ext, SEXP value) {                                         \
        int32_t code = 0;                                                      \
        if (cfn(doc_builder_ptr(ext), CHAR(STRING_ELT(value, 0)), &code) != 0) \
            pdfox_raise(code, op);                                             \
        return R_NilValue;                                                     \
    }
BUILDER_STR_SETTER(r_builder_set_title,    pdf_document_builder_set_title,    "builder_set_title")
BUILDER_STR_SETTER(r_builder_set_author,   pdf_document_builder_set_author,   "builder_set_author")
BUILDER_STR_SETTER(r_builder_set_subject,  pdf_document_builder_set_subject,  "builder_set_subject")
BUILDER_STR_SETTER(r_builder_set_keywords, pdf_document_builder_set_keywords, "builder_set_keywords")
BUILDER_STR_SETTER(r_builder_set_creator,  pdf_document_builder_set_creator,  "builder_set_creator")
BUILDER_STR_SETTER(r_builder_on_open,      pdf_document_builder_on_open,      "builder_on_open")
BUILDER_STR_SETTER(r_builder_language,     pdf_document_builder_language,     "builder_language")

SEXP r_builder_tagged_pdf_ua1(SEXP ext) {
    int32_t code = 0;
    if (pdf_document_builder_tagged_pdf_ua1(doc_builder_ptr(ext), &code) != 0)
        pdfox_raise(code, "builder_tagged_pdf_ua1");
    return R_NilValue;
}
SEXP r_builder_role_map(SEXP ext, SEXP custom, SEXP standard) {
    int32_t code = 0;
    if (pdf_document_builder_role_map(doc_builder_ptr(ext),
                                      CHAR(STRING_ELT(custom, 0)),
                                      CHAR(STRING_ELT(standard, 0)), &code) != 0)
        pdfox_raise(code, "builder_role_map");
    return R_NilValue;
}
/* Consumes the font handle on success — clear its external pointer so the GC
 * finalizer (and any explicit close) becomes a no-op and we never double-free. */
SEXP r_builder_register_embedded_font(SEXP ext, SEXP name, SEXP font_ext) {
    int32_t code = 0;
    if (pdf_document_builder_register_embedded_font(
            doc_builder_ptr(ext), CHAR(STRING_ELT(name, 0)),
            embedded_font_ptr(font_ext), &code) != 0)
        pdfox_raise(code, "builder_register_embedded_font");
    R_ClearExternalPtr(font_ext); /* ownership transferred to the builder */
    return R_NilValue;
}

/* page factories — return a wrapped FfiPageBuilder */
SEXP r_builder_a4_page(SEXP ext) {
    int32_t code = 0;
    FfiPageBuilder *p = pdf_document_builder_a4_page(doc_builder_ptr(ext), &code);
    if (!p) pdfox_raise(code, "builder_a4_page");
    return wrap_page_builder(p);
}
SEXP r_builder_letter_page(SEXP ext) {
    int32_t code = 0;
    FfiPageBuilder *p = pdf_document_builder_letter_page(doc_builder_ptr(ext), &code);
    if (!p) pdfox_raise(code, "builder_letter_page");
    return wrap_page_builder(p);
}
SEXP r_builder_page(SEXP ext, SEXP width, SEXP height) {
    int32_t code = 0;
    FfiPageBuilder *p = pdf_document_builder_page(
        doc_builder_ptr(ext), (float)Rf_asReal(width), (float)Rf_asReal(height),
        &code);
    if (!p) pdfox_raise(code, "builder_page");
    return wrap_page_builder(p);
}

/* build / save (builder STATE consumed; wrapper still freed by finalizer) */
SEXP r_builder_build(SEXP ext) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = pdf_document_builder_build(doc_builder_ptr(ext), &out_len, &code);
    if (!buf) pdfox_raise(code, "builder_build");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}
SEXP r_builder_save(SEXP ext, SEXP path) {
    int32_t code = 0;
    if (pdf_document_builder_save(doc_builder_ptr(ext),
                                  CHAR(STRING_ELT(path, 0)), &code) != 0)
        pdfox_raise(code, "builder_save");
    return R_NilValue;
}
SEXP r_builder_save_encrypted(SEXP ext, SEXP path, SEXP user_pw, SEXP owner_pw) {
    int32_t code = 0;
    if (pdf_document_builder_save_encrypted(
            doc_builder_ptr(ext), CHAR(STRING_ELT(path, 0)),
            CHAR(STRING_ELT(user_pw, 0)), CHAR(STRING_ELT(owner_pw, 0)),
            &code) != 0)
        pdfox_raise(code, "builder_save_encrypted");
    return R_NilValue;
}
SEXP r_builder_to_bytes_encrypted(SEXP ext, SEXP user_pw, SEXP owner_pw) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = pdf_document_builder_to_bytes_encrypted(
        doc_builder_ptr(ext), CHAR(STRING_ELT(user_pw, 0)),
        CHAR(STRING_ELT(owner_pw, 0)), &out_len, &code);
    if (!buf) pdfox_raise(code, "builder_to_bytes_encrypted");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), buf, (size_t)n);
    free_bytes(buf);
    UNPROTECT(1);
    return out;
}

/* ── PageBuilder ── fluent ops (all int32 status unless noted) */
SEXP r_page_font(SEXP ext, SEXP name, SEXP size) {
    int32_t code = 0;
    if (pdf_page_builder_font(page_builder_ptr(ext), CHAR(STRING_ELT(name, 0)),
                              (float)Rf_asReal(size), &code) != 0)
        pdfox_raise(code, "page_font");
    return R_NilValue;
}
SEXP r_page_at(SEXP ext, SEXP x, SEXP y) {
    int32_t code = 0;
    if (pdf_page_builder_at(page_builder_ptr(ext), (float)Rf_asReal(x),
                            (float)Rf_asReal(y), &code) != 0)
        pdfox_raise(code, "page_at");
    return R_NilValue;
}

/* one-string-arg fluent ops sharing the same shape */
#define PAGE_STR_OP(rname, cfn, op)                                            \
    SEXP rname(SEXP ext, SEXP text) {                                          \
        int32_t code = 0;                                                      \
        if (cfn(page_builder_ptr(ext), CHAR(STRING_ELT(text, 0)), &code) != 0) \
            pdfox_raise(code, op);                                             \
        return R_NilValue;                                                     \
    }
PAGE_STR_OP(r_page_text,           pdf_page_builder_text,           "page_text")
PAGE_STR_OP(r_page_paragraph,      pdf_page_builder_paragraph,      "page_paragraph")
PAGE_STR_OP(r_page_link_url,       pdf_page_builder_link_url,       "page_link_url")
PAGE_STR_OP(r_page_link_named,     pdf_page_builder_link_named,     "page_link_named")
PAGE_STR_OP(r_page_link_javascript,pdf_page_builder_link_javascript,"page_link_javascript")
PAGE_STR_OP(r_page_on_open,        pdf_page_builder_on_open,        "page_on_open")
PAGE_STR_OP(r_page_on_close,       pdf_page_builder_on_close,       "page_on_close")
PAGE_STR_OP(r_page_field_keystroke,pdf_page_builder_field_keystroke,"page_field_keystroke")
PAGE_STR_OP(r_page_field_format,   pdf_page_builder_field_format,   "page_field_format")
PAGE_STR_OP(r_page_field_validate, pdf_page_builder_field_validate, "page_field_validate")
PAGE_STR_OP(r_page_field_calculate,pdf_page_builder_field_calculate,"page_field_calculate")
PAGE_STR_OP(r_page_sticky_note,    pdf_page_builder_sticky_note,    "page_sticky_note")
PAGE_STR_OP(r_page_watermark,      pdf_page_builder_watermark,      "page_watermark")
PAGE_STR_OP(r_page_stamp,          pdf_page_builder_stamp,          "page_stamp")
PAGE_STR_OP(r_page_inline,         pdf_page_builder_inline,         "page_inline")
PAGE_STR_OP(r_page_inline_bold,    pdf_page_builder_inline_bold,    "page_inline_bold")
PAGE_STR_OP(r_page_inline_italic,  pdf_page_builder_inline_italic,  "page_inline_italic")

/* no-arg fluent ops sharing the same shape */
#define PAGE_VOID_OP(rname, cfn, op)                                           \
    SEXP rname(SEXP ext) {                                                     \
        int32_t code = 0;                                                      \
        if (cfn(page_builder_ptr(ext), &code) != 0) pdfox_raise(code, op);     \
        return R_NilValue;                                                     \
    }
PAGE_VOID_OP(r_page_horizontal_rule,        pdf_page_builder_horizontal_rule,        "page_horizontal_rule")
PAGE_VOID_OP(r_page_watermark_confidential, pdf_page_builder_watermark_confidential, "page_watermark_confidential")
PAGE_VOID_OP(r_page_watermark_draft,        pdf_page_builder_watermark_draft,        "page_watermark_draft")
PAGE_VOID_OP(r_page_new_page_same_size,     pdf_page_builder_new_page_same_size,     "page_new_page_same_size")
PAGE_VOID_OP(r_page_newline,                pdf_page_builder_newline,                "page_newline")

SEXP r_page_heading(SEXP ext, SEXP level, SEXP text) {
    int32_t code = 0;
    if (pdf_page_builder_heading(page_builder_ptr(ext),
                                 (uint8_t)Rf_asInteger(level),
                                 CHAR(STRING_ELT(text, 0)), &code) != 0)
        pdfox_raise(code, "page_heading");
    return R_NilValue;
}
SEXP r_page_space(SEXP ext, SEXP points) {
    int32_t code = 0;
    if (pdf_page_builder_space(page_builder_ptr(ext), (float)Rf_asReal(points),
                               &code) != 0)
        pdfox_raise(code, "page_space");
    return R_NilValue;
}
SEXP r_page_link_page(SEXP ext, SEXP page_index) {
    int32_t code = 0;
    if (pdf_page_builder_link_page(page_builder_ptr(ext),
                                   (uintptr_t)Rf_asInteger(page_index), &code) != 0)
        pdfox_raise(code, "page_link_page");
    return R_NilValue;
}

/* RGB-colour decorations sharing the (r, g, b) shape */
#define PAGE_RGB_OP(rname, cfn, op)                                            \
    SEXP rname(SEXP ext, SEXP r, SEXP g, SEXP b) {                             \
        int32_t code = 0;                                                      \
        if (cfn(page_builder_ptr(ext), (float)Rf_asReal(r),                   \
                (float)Rf_asReal(g), (float)Rf_asReal(b), &code) != 0)        \
            pdfox_raise(code, op);                                             \
        return R_NilValue;                                                     \
    }
PAGE_RGB_OP(r_page_highlight, pdf_page_builder_highlight, "page_highlight")
PAGE_RGB_OP(r_page_underline, pdf_page_builder_underline, "page_underline")
PAGE_RGB_OP(r_page_strikeout, pdf_page_builder_strikeout, "page_strikeout")
PAGE_RGB_OP(r_page_squiggly,  pdf_page_builder_squiggly,  "page_squiggly")

SEXP r_page_inline_color(SEXP ext, SEXP r, SEXP g, SEXP b, SEXP text) {
    int32_t code = 0;
    if (pdf_page_builder_inline_color(page_builder_ptr(ext), (float)Rf_asReal(r),
                                      (float)Rf_asReal(g), (float)Rf_asReal(b),
                                      CHAR(STRING_ELT(text, 0)), &code) != 0)
        pdfox_raise(code, "page_inline_color");
    return R_NilValue;
}
SEXP r_page_sticky_note_at(SEXP ext, SEXP x, SEXP y, SEXP text) {
    int32_t code = 0;
    if (pdf_page_builder_sticky_note_at(page_builder_ptr(ext), (float)Rf_asReal(x),
                                        (float)Rf_asReal(y),
                                        CHAR(STRING_ELT(text, 0)), &code) != 0)
        pdfox_raise(code, "page_sticky_note_at");
    return R_NilValue;
}
SEXP r_page_freetext(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h, SEXP text) {
    int32_t code = 0;
    if (pdf_page_builder_freetext(page_builder_ptr(ext), (float)Rf_asReal(x),
                                  (float)Rf_asReal(y), (float)Rf_asReal(w),
                                  (float)Rf_asReal(h), CHAR(STRING_ELT(text, 0)),
                                  &code) != 0)
        pdfox_raise(code, "page_freetext");
    return R_NilValue;
}
SEXP r_page_footnote(SEXP ext, SEXP ref_mark, SEXP note_text) {
    int32_t code = 0;
    if (pdf_page_builder_footnote(page_builder_ptr(ext),
                                  CHAR(STRING_ELT(ref_mark, 0)),
                                  CHAR(STRING_ELT(note_text, 0)), &code) != 0)
        pdfox_raise(code, "page_footnote");
    return R_NilValue;
}
SEXP r_page_columns(SEXP ext, SEXP column_count, SEXP gap_pt, SEXP text) {
    int32_t code = 0;
    if (pdf_page_builder_columns(page_builder_ptr(ext),
                                 (uint32_t)Rf_asInteger(column_count),
                                 (float)Rf_asReal(gap_pt),
                                 CHAR(STRING_ELT(text, 0)), &code) != 0)
        pdfox_raise(code, "page_columns");
    return R_NilValue;
}

/* form fields */
SEXP r_page_text_field(SEXP ext, SEXP name, SEXP x, SEXP y, SEXP w, SEXP h,
                       SEXP default_value) {
    int32_t code = 0;
    const char *dv = (default_value == R_NilValue) ? NULL
                                                   : CHAR(STRING_ELT(default_value, 0));
    if (pdf_page_builder_text_field(page_builder_ptr(ext),
                                    CHAR(STRING_ELT(name, 0)), (float)Rf_asReal(x),
                                    (float)Rf_asReal(y), (float)Rf_asReal(w),
                                    (float)Rf_asReal(h), dv, &code) != 0)
        pdfox_raise(code, "page_text_field");
    return R_NilValue;
}
SEXP r_page_checkbox(SEXP ext, SEXP name, SEXP x, SEXP y, SEXP w, SEXP h,
                     SEXP checked) {
    int32_t code = 0;
    if (pdf_page_builder_checkbox(page_builder_ptr(ext),
                                  CHAR(STRING_ELT(name, 0)), (float)Rf_asReal(x),
                                  (float)Rf_asReal(y), (float)Rf_asReal(w),
                                  (float)Rf_asReal(h),
                                  Rf_asLogical(checked) == TRUE ? 1 : 0,
                                  &code) != 0)
        pdfox_raise(code, "page_checkbox");
    return R_NilValue;
}
SEXP r_page_combo_box(SEXP ext, SEXP name, SEXP x, SEXP y, SEXP w, SEXP h,
                      SEXP options, SEXP selected) {
    int32_t code = 0;
    uintptr_t n = 0;
    const char **opts = strvec_to_c(options, &n);
    const char *sel = (selected == R_NilValue) ? NULL : CHAR(STRING_ELT(selected, 0));
    if (pdf_page_builder_combo_box(page_builder_ptr(ext),
                                   CHAR(STRING_ELT(name, 0)), (float)Rf_asReal(x),
                                   (float)Rf_asReal(y), (float)Rf_asReal(w),
                                   (float)Rf_asReal(h), opts, n, sel, &code) != 0)
        pdfox_raise(code, "page_combo_box");
    return R_NilValue;
}
SEXP r_page_radio_group(SEXP ext, SEXP name, SEXP values, SEXP xs, SEXP ys,
                        SEXP ws, SEXP hs, SEXP selected) {
    int32_t code = 0;
    uintptr_t n = 0;
    const char **vals = strvec_to_c(values, &n);
    SEXP xr = PROTECT(Rf_coerceVector(xs, REALSXP));
    SEXP yr = PROTECT(Rf_coerceVector(ys, REALSXP));
    SEXP wr = PROTECT(Rf_coerceVector(ws, REALSXP));
    SEXP hr = PROTECT(Rf_coerceVector(hs, REALSXP));
    /* C ABI wants float arrays — narrow the doubles into stack/heap floats. */
    float *fx = (float *)R_alloc((size_t)n, sizeof(float));
    float *fy = (float *)R_alloc((size_t)n, sizeof(float));
    float *fw = (float *)R_alloc((size_t)n, sizeof(float));
    float *fh = (float *)R_alloc((size_t)n, sizeof(float));
    for (uintptr_t i = 0; i < n; i++) {
        fx[i] = (float)REAL(xr)[i]; fy[i] = (float)REAL(yr)[i];
        fw[i] = (float)REAL(wr)[i]; fh[i] = (float)REAL(hr)[i];
    }
    const char *sel = (selected == R_NilValue) ? NULL : CHAR(STRING_ELT(selected, 0));
    int32_t rc = pdf_page_builder_radio_group(page_builder_ptr(ext),
                                              CHAR(STRING_ELT(name, 0)), vals,
                                              fx, fy, fw, fh, n, sel, &code);
    UNPROTECT(4);
    if (rc != 0) pdfox_raise(code, "page_radio_group");
    return R_NilValue;
}
SEXP r_page_push_button(SEXP ext, SEXP name, SEXP x, SEXP y, SEXP w, SEXP h,
                        SEXP caption) {
    int32_t code = 0;
    if (pdf_page_builder_push_button(page_builder_ptr(ext),
                                     CHAR(STRING_ELT(name, 0)), (float)Rf_asReal(x),
                                     (float)Rf_asReal(y), (float)Rf_asReal(w),
                                     (float)Rf_asReal(h),
                                     CHAR(STRING_ELT(caption, 0)), &code) != 0)
        pdfox_raise(code, "page_push_button");
    return R_NilValue;
}
SEXP r_page_signature_field(SEXP ext, SEXP name, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_page_builder_signature_field(page_builder_ptr(ext),
                                         CHAR(STRING_ELT(name, 0)),
                                         (float)Rf_asReal(x), (float)Rf_asReal(y),
                                         (float)Rf_asReal(w), (float)Rf_asReal(h),
                                         &code) != 0)
        pdfox_raise(code, "page_signature_field");
    return R_NilValue;
}

/* barcodes */
SEXP r_page_barcode_1d(SEXP ext, SEXP barcode_type, SEXP data, SEXP x, SEXP y,
                       SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_page_builder_barcode_1d(page_builder_ptr(ext),
                                    Rf_asInteger(barcode_type),
                                    CHAR(STRING_ELT(data, 0)), (float)Rf_asReal(x),
                                    (float)Rf_asReal(y), (float)Rf_asReal(w),
                                    (float)Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "page_barcode_1d");
    return R_NilValue;
}
SEXP r_page_barcode_qr(SEXP ext, SEXP data, SEXP x, SEXP y, SEXP size) {
    int32_t code = 0;
    if (pdf_page_builder_barcode_qr(page_builder_ptr(ext),
                                    CHAR(STRING_ELT(data, 0)), (float)Rf_asReal(x),
                                    (float)Rf_asReal(y), (float)Rf_asReal(size),
                                    &code) != 0)
        pdfox_raise(code, "page_barcode_qr");
    return R_NilValue;
}

/* images */
SEXP r_page_image(SEXP ext, SEXP raw, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_page_builder_image(page_builder_ptr(ext), RAW(raw),
                               (uintptr_t)XLENGTH(raw), (float)Rf_asReal(x),
                               (float)Rf_asReal(y), (float)Rf_asReal(w),
                               (float)Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "page_image");
    return R_NilValue;
}
SEXP r_page_image_with_alt(SEXP ext, SEXP raw, SEXP x, SEXP y, SEXP w, SEXP h,
                           SEXP alt_text) {
    int32_t code = 0;
    if (pdf_page_builder_image_with_alt(page_builder_ptr(ext), RAW(raw),
                                        (uintptr_t)XLENGTH(raw), (float)Rf_asReal(x),
                                        (float)Rf_asReal(y), (float)Rf_asReal(w),
                                        (float)Rf_asReal(h),
                                        CHAR(STRING_ELT(alt_text, 0)), &code) != 0)
        pdfox_raise(code, "page_image_with_alt");
    return R_NilValue;
}
SEXP r_page_image_artifact(SEXP ext, SEXP raw, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_page_builder_image_artifact(page_builder_ptr(ext), RAW(raw),
                                        (uintptr_t)XLENGTH(raw), (float)Rf_asReal(x),
                                        (float)Rf_asReal(y), (float)Rf_asReal(w),
                                        (float)Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "page_image_artifact");
    return R_NilValue;
}

/* vector graphics */
SEXP r_page_rect(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_page_builder_rect(page_builder_ptr(ext), (float)Rf_asReal(x),
                              (float)Rf_asReal(y), (float)Rf_asReal(w),
                              (float)Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "page_rect");
    return R_NilValue;
}
SEXP r_page_filled_rect(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h, SEXP r, SEXP g,
                        SEXP b) {
    int32_t code = 0;
    if (pdf_page_builder_filled_rect(page_builder_ptr(ext), (float)Rf_asReal(x),
                                     (float)Rf_asReal(y), (float)Rf_asReal(w),
                                     (float)Rf_asReal(h), (float)Rf_asReal(r),
                                     (float)Rf_asReal(g), (float)Rf_asReal(b),
                                     &code) != 0)
        pdfox_raise(code, "page_filled_rect");
    return R_NilValue;
}
SEXP r_page_line(SEXP ext, SEXP x1, SEXP y1, SEXP x2, SEXP y2) {
    int32_t code = 0;
    if (pdf_page_builder_line(page_builder_ptr(ext), (float)Rf_asReal(x1),
                              (float)Rf_asReal(y1), (float)Rf_asReal(x2),
                              (float)Rf_asReal(y2), &code) != 0)
        pdfox_raise(code, "page_line");
    return R_NilValue;
}
SEXP r_page_stroke_rect(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h, SEXP width,
                        SEXP r, SEXP g, SEXP b) {
    int32_t code = 0;
    if (pdf_page_builder_stroke_rect(page_builder_ptr(ext), (float)Rf_asReal(x),
                                     (float)Rf_asReal(y), (float)Rf_asReal(w),
                                     (float)Rf_asReal(h), (float)Rf_asReal(width),
                                     (float)Rf_asReal(r), (float)Rf_asReal(g),
                                     (float)Rf_asReal(b), &code) != 0)
        pdfox_raise(code, "page_stroke_rect");
    return R_NilValue;
}
SEXP r_page_stroke_line(SEXP ext, SEXP x1, SEXP y1, SEXP x2, SEXP y2, SEXP width,
                        SEXP r, SEXP g, SEXP b) {
    int32_t code = 0;
    if (pdf_page_builder_stroke_line(page_builder_ptr(ext), (float)Rf_asReal(x1),
                                     (float)Rf_asReal(y1), (float)Rf_asReal(x2),
                                     (float)Rf_asReal(y2), (float)Rf_asReal(width),
                                     (float)Rf_asReal(r), (float)Rf_asReal(g),
                                     (float)Rf_asReal(b), &code) != 0)
        pdfox_raise(code, "page_stroke_line");
    return R_NilValue;
}
/* Narrow an R numeric `dash` vector into a heap float array (NULL when empty). */
static const float *dash_to_c(SEXP dash, uintptr_t *out_n) {
    R_xlen_t n = (dash == R_NilValue) ? 0 : XLENGTH(dash);
    *out_n = (uintptr_t)n;
    if (n == 0) return NULL;
    SEXP d = PROTECT(Rf_coerceVector(dash, REALSXP));
    float *arr = (float *)R_alloc((size_t)n, sizeof(float));
    for (R_xlen_t i = 0; i < n; i++) arr[i] = (float)REAL(d)[i];
    UNPROTECT(1);
    return arr;
}
SEXP r_page_stroke_rect_dashed(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h,
                               SEXP width, SEXP r, SEXP g, SEXP b, SEXP dash,
                               SEXP phase) {
    int32_t code = 0;
    uintptr_t n_dash = 0;
    const float *darr = dash_to_c(dash, &n_dash);
    if (pdf_page_builder_stroke_rect_dashed(
            page_builder_ptr(ext), (float)Rf_asReal(x), (float)Rf_asReal(y),
            (float)Rf_asReal(w), (float)Rf_asReal(h), (float)Rf_asReal(width),
            (float)Rf_asReal(r), (float)Rf_asReal(g), (float)Rf_asReal(b),
            darr, n_dash, (float)Rf_asReal(phase), &code) != 0)
        pdfox_raise(code, "page_stroke_rect_dashed");
    return R_NilValue;
}
SEXP r_page_stroke_line_dashed(SEXP ext, SEXP x1, SEXP y1, SEXP x2, SEXP y2,
                               SEXP width, SEXP r, SEXP g, SEXP b, SEXP dash,
                               SEXP phase) {
    int32_t code = 0;
    uintptr_t n_dash = 0;
    const float *darr = dash_to_c(dash, &n_dash);
    if (pdf_page_builder_stroke_line_dashed(
            page_builder_ptr(ext), (float)Rf_asReal(x1), (float)Rf_asReal(y1),
            (float)Rf_asReal(x2), (float)Rf_asReal(y2), (float)Rf_asReal(width),
            (float)Rf_asReal(r), (float)Rf_asReal(g), (float)Rf_asReal(b),
            darr, n_dash, (float)Rf_asReal(phase), &code) != 0)
        pdfox_raise(code, "page_stroke_line_dashed");
    return R_NilValue;
}
SEXP r_page_text_in_rect(SEXP ext, SEXP x, SEXP y, SEXP w, SEXP h, SEXP text,
                         SEXP align) {
    int32_t code = 0;
    if (pdf_page_builder_text_in_rect(page_builder_ptr(ext), (float)Rf_asReal(x),
                                      (float)Rf_asReal(y), (float)Rf_asReal(w),
                                      (float)Rf_asReal(h),
                                      CHAR(STRING_ELT(text, 0)),
                                      Rf_asInteger(align), &code) != 0)
        pdfox_raise(code, "page_text_in_rect");
    return R_NilValue;
}

/* table — `cells` is a row-major character vector of length n_rows*n_columns;
 * `widths` length n_columns numeric; `aligns` length n_columns integer. */
SEXP r_page_table(SEXP ext, SEXP n_columns, SEXP widths, SEXP aligns,
                  SEXP n_rows, SEXP cells, SEXP has_header) {
    int32_t code = 0;
    uintptr_t ncol = (uintptr_t)Rf_asInteger(n_columns);
    uintptr_t nrow = (uintptr_t)Rf_asInteger(n_rows);
    uintptr_t ncells = 0;
    const char **cellp = strvec_to_c(cells, &ncells);
    uintptr_t wn = 0;
    const float *wp = dash_to_c(widths, &wn);
    SEXP ai = PROTECT(Rf_coerceVector(aligns, INTSXP));
    int32_t rc = pdf_page_builder_table(page_builder_ptr(ext), ncol, wp,
                                        INTEGER(ai), nrow, cellp,
                                        Rf_asLogical(has_header) == TRUE ? 1 : 0,
                                        &code);
    UNPROTECT(1);
    if (rc != 0) pdfox_raise(code, "page_table");
    return R_NilValue;
}

/* streaming table */
SEXP r_page_streaming_table_begin(SEXP ext, SEXP n_columns, SEXP headers,
                                  SEXP widths, SEXP aligns, SEXP repeat_header) {
    int32_t code = 0;
    uintptr_t ncol = (uintptr_t)Rf_asInteger(n_columns);
    uintptr_t hn = 0;
    const char **hp = strvec_to_c(headers, &hn);
    uintptr_t wn = 0;
    const float *wp = dash_to_c(widths, &wn);
    SEXP ai = PROTECT(Rf_coerceVector(aligns, INTSXP));
    int32_t rc = pdf_page_builder_streaming_table_begin(
        page_builder_ptr(ext), ncol, hp, wp, INTEGER(ai),
        Rf_asLogical(repeat_header) == TRUE ? 1 : 0, &code);
    UNPROTECT(1);
    if (rc != 0) pdfox_raise(code, "page_streaming_table_begin");
    return R_NilValue;
}
SEXP r_page_streaming_table_begin_v2(SEXP ext, SEXP n_columns, SEXP headers,
                                     SEXP widths, SEXP aligns, SEXP repeat_header,
                                     SEXP mode, SEXP sample_rows,
                                     SEXP min_col_width_pt, SEXP max_col_width_pt,
                                     SEXP max_rowspan) {
    int32_t code = 0;
    uintptr_t ncol = (uintptr_t)Rf_asInteger(n_columns);
    uintptr_t hn = 0;
    const char **hp = strvec_to_c(headers, &hn);
    uintptr_t wn = 0;
    const float *wp = dash_to_c(widths, &wn);
    SEXP ai = PROTECT(Rf_coerceVector(aligns, INTSXP));
    int32_t rc = pdf_page_builder_streaming_table_begin_v2(
        page_builder_ptr(ext), ncol, hp, wp, INTEGER(ai),
        Rf_asLogical(repeat_header) == TRUE ? 1 : 0, Rf_asInteger(mode),
        (uintptr_t)Rf_asInteger(sample_rows), (float)Rf_asReal(min_col_width_pt),
        (float)Rf_asReal(max_col_width_pt), (uintptr_t)Rf_asInteger(max_rowspan),
        &code);
    UNPROTECT(1);
    if (rc != 0) pdfox_raise(code, "page_streaming_table_begin_v2");
    return R_NilValue;
}
SEXP r_page_streaming_table_set_batch_size(SEXP ext, SEXP batch_size) {
    int32_t code = 0;
    if (pdf_page_builder_streaming_table_set_batch_size(
            page_builder_ptr(ext), (uintptr_t)Rf_asInteger(batch_size), &code) != 0)
        pdfox_raise(code, "page_streaming_table_set_batch_size");
    return R_NilValue;
}
SEXP r_page_streaming_table_pending_row_count(SEXP ext) {
    return Rf_ScalarReal(
        (double)pdf_page_builder_streaming_table_pending_row_count(page_builder_ptr(ext)));
}
SEXP r_page_streaming_table_batch_count(SEXP ext) {
    return Rf_ScalarReal(
        (double)pdf_page_builder_streaming_table_batch_count(page_builder_ptr(ext)));
}
SEXP r_page_streaming_table_flush(SEXP ext) {
    int32_t code = 0;
    if (pdf_page_builder_streaming_table_flush(page_builder_ptr(ext), &code) != 0)
        pdfox_raise(code, "page_streaming_table_flush");
    return R_NilValue;
}
SEXP r_page_streaming_table_push_row(SEXP ext, SEXP cells) {
    int32_t code = 0;
    uintptr_t n = 0;
    const char **cp = strvec_to_c(cells, &n);
    if (pdf_page_builder_streaming_table_push_row(page_builder_ptr(ext), n, cp,
                                                  &code) != 0)
        pdfox_raise(code, "page_streaming_table_push_row");
    return R_NilValue;
}
SEXP r_page_streaming_table_push_row_v2(SEXP ext, SEXP cells, SEXP rowspans) {
    int32_t code = 0;
    uintptr_t n = 0;
    const char **cp = strvec_to_c(cells, &n);
    const uintptr_t *rsp = NULL;
    if (rowspans != R_NilValue && XLENGTH(rowspans) > 0) {
        SEXP ri = PROTECT(Rf_coerceVector(rowspans, INTSXP));
        R_xlen_t rn = XLENGTH(ri);
        uintptr_t *rs = (uintptr_t *)R_alloc((size_t)rn, sizeof(uintptr_t));
        for (R_xlen_t i = 0; i < rn; i++) rs[i] = (uintptr_t)INTEGER(ri)[i];
        rsp = rs;
        UNPROTECT(1);
    }
    if (pdf_page_builder_streaming_table_push_row_v2(page_builder_ptr(ext), n, cp,
                                                     rsp, &code) != 0)
        pdfox_raise(code, "page_streaming_table_push_row_v2");
    return R_NilValue;
}
SEXP r_page_streaming_table_finish(SEXP ext) {
    int32_t code = 0;
    if (pdf_page_builder_streaming_table_finish(page_builder_ptr(ext), &code) != 0)
        pdfox_raise(code, "page_streaming_table_finish");
    return R_NilValue;
}

/* done — CONSUMES the page handle; clear the external pointer on success so the
 * finalizer (and any explicit close) is a no-op (no double-free). */
SEXP r_page_done(SEXP ext) {
    int32_t code = 0;
    if (pdf_page_builder_done(page_builder_ptr(ext), &code) != 0)
        pdfox_raise(code, "page_done");
    R_ClearExternalPtr(ext);
    return R_NilValue;
}
/* Explicit, idempotent free of an uncommitted page handle. */
SEXP r_page_close(SEXP ext) {
    FfiPageBuilder *h = (FfiPageBuilder *)R_ExternalPtrAddr(ext);
    if (h) { pdf_page_builder_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── PHASE-6 digital signatures / PKI / timestamps / TSA / validation ─────────
 * Five opaque native handle families, each wrapped in an R external pointer with
 * a GC finalizer (or an explicit idempotent close), mirroring the
 * PdfDocument/Pdf/DocumentEditor pattern above:
 *
 *   Certificate (void*)    — pdf_certificate_load_* / pdf_certificate_free
 *   SignatureInfo          — pdf_document_get_signature / pdf_signature_free
 *   Timestamp (void*)      — pdf_timestamp_parse / pdf_timestamp_free
 *   TsaClient (void*)      — pdf_tsa_client_create / pdf_tsa_client_free
 *   Dss (void*)            — pdf_document_get_dss / pdf_dss_free
 *   PdfA/Ua/X results      — pdf_validate_* / pdf_pdf_*_results_free
 *
 * Owned char* returns use take_string + free_string; owned uint8* buffers are
 * copied into RAWSXP and released with free_bytes; const uint8* returns (the
 * timestamp token / message-imprint, owned by the live handle) are COPIED but
 * NOT freed. Errors raise the same classed pdfoxide_error via pdfox_raise. */

/* Certificate (opaque void*) */
static void certificate_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_certificate_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_certificate(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, certificate_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static void *certificate_ptr(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: certificate handle is closed");
    return h;
}

/* SignatureInfo */
static void signature_finalizer(SEXP ext) {
    FfiSignatureInfo *h = (FfiSignatureInfo *)R_ExternalPtrAddr(ext);
    if (h) { pdf_signature_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_signature(FfiSignatureInfo *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, signature_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiSignatureInfo *signature_ptr(SEXP ext) {
    FfiSignatureInfo *h = (FfiSignatureInfo *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: signature handle is closed");
    return h;
}

/* Timestamp (opaque void*) */
static void timestamp_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_timestamp_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_timestamp(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, timestamp_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static void *timestamp_ptr(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: timestamp handle is closed");
    return h;
}

/* TsaClient (opaque void*) */
static void tsa_client_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_tsa_client_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_tsa_client(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, tsa_client_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static void *tsa_client_ptr(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: TSA client handle is closed");
    return h;
}

/* Dss (opaque void*) */
static void dss_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_dss_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_dss(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, dss_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static void *dss_ptr(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: DSS handle is closed");
    return h;
}

/* PdfA / Ua / X result handles */
static void pdf_a_results_finalizer(SEXP ext) {
    FfiPdfAResults *h = (FfiPdfAResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_a_results_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_pdf_a_results(FfiPdfAResults *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, pdf_a_results_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiPdfAResults *pdf_a_results_ptr(SEXP ext) {
    FfiPdfAResults *h = (FfiPdfAResults *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: PDF/A results handle is closed");
    return h;
}
static void ua_results_finalizer(SEXP ext) {
    FfiUaResults *h = (FfiUaResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_ua_results_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_ua_results(FfiUaResults *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, ua_results_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiUaResults *ua_results_ptr(SEXP ext) {
    FfiUaResults *h = (FfiUaResults *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: PDF/UA results handle is closed");
    return h;
}
static void pdf_x_results_finalizer(SEXP ext) {
    FfiPdfXResults *h = (FfiPdfXResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_x_results_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_pdf_x_results(FfiPdfXResults *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, pdf_x_results_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiPdfXResults *pdf_x_results_ptr(SEXP ext) {
    FfiPdfXResults *h = (FfiPdfXResults *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: PDF/X results handle is closed");
    return h;
}

/* Copy an owned uint8* buffer into a RAWSXP and free_bytes it (NULL ⇒ raise). */
static SEXP take_bytes(uint8_t *p, uintptr_t len, int32_t code, const char *op) {
    if (!p) pdfox_raise(code, op);
    R_xlen_t n = (R_xlen_t)len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    free_bytes(p);
    UNPROTECT(1);
    return out;
}

/* ── log level ── */
SEXP r_oxide_set_log_level(SEXP level) {
    pdf_oxide_set_log_level(Rf_asInteger(level));
    return R_NilValue;
}
SEXP r_oxide_get_log_level(void) {
    return Rf_ScalarInteger(pdf_oxide_get_log_level());
}

/* ── Certificate ── */
SEXP r_certificate_load_from_bytes(SEXP raw, SEXP password) {
    int32_t code = 0;
    const char *pw = (password == R_NilValue) ? NULL : CHAR(STRING_ELT(password, 0));
    void *h = pdf_certificate_load_from_bytes(RAW(raw), (int32_t)XLENGTH(raw),
                                              pw, &code);
    if (!h) pdfox_raise(code, "certificate_load_from_bytes");
    return wrap_certificate(h);
}
SEXP r_certificate_load_from_pem(SEXP cert_pem, SEXP key_pem) {
    int32_t code = 0;
    void *h = pdf_certificate_load_from_pem(CHAR(STRING_ELT(cert_pem, 0)),
                                            CHAR(STRING_ELT(key_pem, 0)), &code);
    if (!h) pdfox_raise(code, "certificate_load_from_pem");
    return wrap_certificate(h);
}
SEXP r_certificate_get_subject(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_certificate_get_subject(certificate_ptr(ext), &code),
                       code, "certificate_get_subject");
}
SEXP r_certificate_get_issuer(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_certificate_get_issuer(certificate_ptr(ext), &code),
                       code, "certificate_get_issuer");
}
SEXP r_certificate_get_serial(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_certificate_get_serial(certificate_ptr(ext), &code),
                       code, "certificate_get_serial");
}
SEXP r_certificate_get_validity(SEXP ext) {
    int32_t code = 0;
    int64_t nb = 0, na = 0;
    pdf_certificate_get_validity(certificate_ptr(ext), &nb, &na, &code);
    if (code != 0) pdfox_raise(code, "certificate_get_validity");
    SEXP out = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(out)[0] = (double)nb;
    REAL(out)[1] = (double)na;
    UNPROTECT(1);
    return out;
}
SEXP r_certificate_is_valid(SEXP ext) {
    int32_t code = 0;
    int32_t r = pdf_certificate_is_valid(certificate_ptr(ext), &code);
    if (r < 0) pdfox_raise(code, "certificate_is_valid");
    return Rf_ScalarLogical(r == 1);
}
SEXP r_certificate_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_certificate_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── Signing (return owned byte buffers via free_bytes) ── */
SEXP r_sign_bytes(SEXP pdf, SEXP cert_ext, SEXP reason, SEXP location) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    const char *rs = (reason == R_NilValue) ? NULL : CHAR(STRING_ELT(reason, 0));
    const char *loc = (location == R_NilValue) ? NULL : CHAR(STRING_ELT(location, 0));
    uint8_t *buf = pdf_sign_bytes(RAW(pdf), (uintptr_t)XLENGTH(pdf),
                                  certificate_ptr(cert_ext), rs, loc, &out_len,
                                  &code);
    return take_bytes(buf, out_len, code, "sign_bytes");
}

/* Marshal an R list of raw vectors into a parallel (const uint8* const*, lens[])
 * pair via R_alloc (valid while the SEXP list is protected by the caller). An
 * empty / NULL list yields (NULL, NULL, 0). */
static const uint8_t *const *rawlist_to_c(SEXP lst, const uintptr_t **out_lens,
                                          uintptr_t *out_n) {
    R_xlen_t n = (lst == R_NilValue) ? 0 : XLENGTH(lst);
    *out_n = (uintptr_t)n;
    if (n == 0) { *out_lens = NULL; return NULL; }
    const uint8_t **ptrs = (const uint8_t **)R_alloc((size_t)n, sizeof(uint8_t *));
    uintptr_t *lens = (uintptr_t *)R_alloc((size_t)n, sizeof(uintptr_t));
    for (R_xlen_t i = 0; i < n; i++) {
        SEXP el = VECTOR_ELT(lst, i);
        ptrs[i] = (el == R_NilValue) ? NULL : RAW(el);
        lens[i] = (el == R_NilValue) ? 0 : (uintptr_t)XLENGTH(el);
    }
    *out_lens = lens;
    return ptrs;
}

SEXP r_sign_bytes_pades(SEXP pdf, SEXP cert_ext, SEXP level, SEXP tsa_url,
                        SEXP reason, SEXP location, SEXP certs, SEXP crls,
                        SEXP ocsps) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    const char *url = (tsa_url == R_NilValue) ? NULL : CHAR(STRING_ELT(tsa_url, 0));
    const char *rs = (reason == R_NilValue) ? NULL : CHAR(STRING_ELT(reason, 0));
    const char *loc = (location == R_NilValue) ? NULL : CHAR(STRING_ELT(location, 0));
    const uintptr_t *cert_lens, *crl_lens, *ocsp_lens;
    uintptr_t n_certs, n_crls, n_ocsps;
    const uint8_t *const *cp = rawlist_to_c(certs, &cert_lens, &n_certs);
    const uint8_t *const *rp = rawlist_to_c(crls, &crl_lens, &n_crls);
    const uint8_t *const *op = rawlist_to_c(ocsps, &ocsp_lens, &n_ocsps);
    uint8_t *buf = pdf_sign_bytes_pades(
        RAW(pdf), (uintptr_t)XLENGTH(pdf), certificate_ptr(cert_ext),
        Rf_asInteger(level), url, rs, loc, cp, cert_lens, n_certs, rp, crl_lens,
        n_crls, op, ocsp_lens, n_ocsps, &out_len, &code);
    return take_bytes(buf, out_len, code, "sign_bytes_pades");
}

SEXP r_sign_bytes_pades_opts(SEXP pdf, SEXP cert_ext, SEXP level, SEXP tsa_url,
                             SEXP reason, SEXP location, SEXP certs, SEXP crls,
                             SEXP ocsps) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    const char *url = (tsa_url == R_NilValue) ? NULL : CHAR(STRING_ELT(tsa_url, 0));
    const char *rs = (reason == R_NilValue) ? NULL : CHAR(STRING_ELT(reason, 0));
    const char *loc = (location == R_NilValue) ? NULL : CHAR(STRING_ELT(location, 0));
    const uintptr_t *cert_lens, *crl_lens, *ocsp_lens;
    uintptr_t n_certs, n_crls, n_ocsps;
    const uint8_t *const *cp = rawlist_to_c(certs, &cert_lens, &n_certs);
    const uint8_t *const *rp = rawlist_to_c(crls, &crl_lens, &n_crls);
    const uint8_t *const *op = rawlist_to_c(ocsps, &ocsp_lens, &n_ocsps);
    PadesSignOptionsC options;
    options.certificate_handle = certificate_ptr(cert_ext);
    options.certs = cp;       options.cert_lens = cert_lens; options.n_certs = n_certs;
    options.crls = rp;        options.crl_lens = crl_lens;   options.n_crls = n_crls;
    options.ocsps = op;       options.ocsp_lens = ocsp_lens; options.n_ocsps = n_ocsps;
    options.tsa_url = url;     options.reason = rs;           options.location = loc;
    options.level = Rf_asInteger(level);
    uint8_t *buf = pdf_sign_bytes_pades_opts(RAW(pdf), (uintptr_t)XLENGTH(pdf),
                                             &options, &out_len, &code);
    return take_bytes(buf, out_len, code, "sign_bytes_pades_opts");
}

/* ── SignatureInfo ── */
SEXP r_doc_signature_count(SEXP ext) {
    int32_t code = 0;
    int32_t n = pdf_document_get_signature_count(doc_ptr(ext), &code);
    if (n < 0) pdfox_raise(code, "signature_count");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_get_signature(SEXP ext, SEXP index) {
    int32_t code = 0;
    void *h = pdf_document_get_signature(doc_ptr(ext), Rf_asInteger(index), &code);
    if (!h) pdfox_raise(code, "get_signature");
    return wrap_signature((FfiSignatureInfo *)h);
}
SEXP r_signature_get_signer_name(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_signature_get_signer_name(signature_ptr(ext), &code),
                       code, "signature_get_signer_name");
}
SEXP r_signature_get_signing_reason(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_signature_get_signing_reason(signature_ptr(ext), &code),
                       code, "signature_get_signing_reason");
}
SEXP r_signature_get_signing_location(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_signature_get_signing_location(signature_ptr(ext), &code),
                       code, "signature_get_signing_location");
}
SEXP r_signature_get_signing_time(SEXP ext) {
    int32_t code = 0;
    int64_t t = pdf_signature_get_signing_time(signature_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "signature_get_signing_time");
    return Rf_ScalarReal((double)t);
}
SEXP r_signature_get_certificate(SEXP ext) {
    int32_t code = 0;
    void *h = pdf_signature_get_certificate(signature_ptr(ext), &code);
    if (!h) pdfox_raise(code, "signature_get_certificate");
    return wrap_certificate(h);
}
SEXP r_signature_get_pades_level(SEXP ext) {
    int32_t code = 0;
    int32_t lvl = pdf_signature_get_pades_level(signature_ptr(ext), &code);
    if (lvl < 0) pdfox_raise(code, "signature_get_pades_level");
    return Rf_ScalarInteger(lvl);
}
SEXP r_signature_has_timestamp(SEXP ext) {
    int32_t code = 0;
    bool r = pdf_signature_has_timestamp(signature_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "signature_has_timestamp");
    return Rf_ScalarLogical(r);
}
SEXP r_signature_get_timestamp(SEXP ext) {
    int32_t code = 0;
    void *h = pdf_signature_get_timestamp(signature_ptr(ext), &code);
    if (!h) pdfox_raise(code, "signature_get_timestamp");
    return wrap_timestamp(h);
}
SEXP r_signature_add_timestamp(SEXP ext, SEXP ts_ext) {
    int32_t code = 0;
    bool r = pdf_signature_add_timestamp(signature_ptr(ext), timestamp_ptr(ts_ext),
                                         &code);
    if (code != 0) pdfox_raise(code, "signature_add_timestamp");
    return Rf_ScalarLogical(r);
}
SEXP r_signature_verify(SEXP ext) {
    int32_t code = 0;
    int32_t r = pdf_signature_verify(signature_ptr(ext), &code);
    return Rf_ScalarInteger(r);
}
SEXP r_signature_verify_detached(SEXP ext, SEXP pdf) {
    int32_t code = 0;
    int32_t r = pdf_signature_verify_detached(signature_ptr(ext), RAW(pdf),
                                              (uintptr_t)XLENGTH(pdf), &code);
    return Rf_ScalarInteger(r);
}
SEXP r_signature_close(SEXP ext) {
    FfiSignatureInfo *h = (FfiSignatureInfo *)R_ExternalPtrAddr(ext);
    if (h) { pdf_signature_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── Timestamp ── */
SEXP r_timestamp_parse(SEXP raw) {
    int32_t code = 0;
    void *h = pdf_timestamp_parse(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "timestamp_parse");
    return wrap_timestamp(h);
}
/* const uint8* returns are owned by the live handle: COPY, do NOT free_bytes. */
SEXP r_timestamp_get_token(SEXP ext) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    const uint8_t *p = pdf_timestamp_get_token(timestamp_ptr(ext), &out_len, &code);
    if (!p) pdfox_raise(code, "timestamp_get_token");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    UNPROTECT(1);
    return out;
}
SEXP r_timestamp_get_message_imprint(SEXP ext) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    const uint8_t *p =
        pdf_timestamp_get_message_imprint(timestamp_ptr(ext), &out_len, &code);
    if (!p) pdfox_raise(code, "timestamp_get_message_imprint");
    R_xlen_t n = (R_xlen_t)out_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    UNPROTECT(1);
    return out;
}
SEXP r_timestamp_get_time(SEXP ext) {
    int32_t code = 0;
    int64_t t = pdf_timestamp_get_time(timestamp_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "timestamp_get_time");
    return Rf_ScalarReal((double)t);
}
SEXP r_timestamp_get_serial(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_timestamp_get_serial(timestamp_ptr(ext), &code), code,
                       "timestamp_get_serial");
}
SEXP r_timestamp_get_tsa_name(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_timestamp_get_tsa_name(timestamp_ptr(ext), &code), code,
                       "timestamp_get_tsa_name");
}
SEXP r_timestamp_get_policy_oid(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_timestamp_get_policy_oid(timestamp_ptr(ext), &code), code,
                       "timestamp_get_policy_oid");
}
SEXP r_timestamp_get_hash_algorithm(SEXP ext) {
    int32_t code = 0;
    int32_t a = pdf_timestamp_get_hash_algorithm(timestamp_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "timestamp_get_hash_algorithm");
    return Rf_ScalarInteger(a);
}
SEXP r_timestamp_verify(SEXP ext) {
    int32_t code = 0;
    bool r = pdf_timestamp_verify(timestamp_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "timestamp_verify");
    return Rf_ScalarLogical(r);
}
SEXP r_timestamp_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_timestamp_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── TSA client ── */
SEXP r_tsa_client_create(SEXP url, SEXP username, SEXP password, SEXP timeout,
                         SEXP hash_algo, SEXP use_nonce, SEXP cert_req) {
    int32_t code = 0;
    const char *un = (username == R_NilValue) ? NULL : CHAR(STRING_ELT(username, 0));
    const char *pw = (password == R_NilValue) ? NULL : CHAR(STRING_ELT(password, 0));
    void *h = pdf_tsa_client_create(CHAR(STRING_ELT(url, 0)), un, pw,
                                    Rf_asInteger(timeout), Rf_asInteger(hash_algo),
                                    Rf_asLogical(use_nonce) == TRUE,
                                    Rf_asLogical(cert_req) == TRUE, &code);
    if (!h) pdfox_raise(code, "tsa_client_create");
    return wrap_tsa_client(h);
}
SEXP r_tsa_request_timestamp(SEXP ext, SEXP data) {
    int32_t code = 0;
    void *h = pdf_tsa_request_timestamp(tsa_client_ptr(ext), RAW(data),
                                        (uintptr_t)XLENGTH(data), &code);
    if (!h) pdfox_raise(code, "tsa_request_timestamp");
    return wrap_timestamp(h);
}
SEXP r_tsa_request_timestamp_hash(SEXP ext, SEXP hash, SEXP hash_algo) {
    int32_t code = 0;
    void *h = pdf_tsa_request_timestamp_hash(tsa_client_ptr(ext), RAW(hash),
                                             (uintptr_t)XLENGTH(hash),
                                             Rf_asInteger(hash_algo), &code);
    if (!h) pdfox_raise(code, "tsa_request_timestamp_hash");
    return wrap_timestamp(h);
}
SEXP r_tsa_client_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_tsa_client_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── DSS ── */
SEXP r_doc_get_dss(SEXP ext) {
    int32_t code = 0;
    void *h = pdf_document_get_dss(doc_ptr(ext), &code);
    /* NULL with code==0 means "no DSS" (not an error): return NULL handle. */
    if (!h) { if (code != 0) pdfox_raise(code, "get_dss"); return R_NilValue; }
    return wrap_dss(h);
}
SEXP r_dss_cert_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_dss_cert_count(dss_ptr(ext)));
}
SEXP r_dss_crl_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_dss_crl_count(dss_ptr(ext)));
}
SEXP r_dss_ocsp_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_dss_ocsp_count(dss_ptr(ext)));
}
SEXP r_dss_vri_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_dss_vri_count(dss_ptr(ext)));
}
SEXP r_dss_get_cert(SEXP ext, SEXP index) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = pdf_dss_get_cert(dss_ptr(ext), Rf_asInteger(index), &out_len,
                                    &code);
    return take_bytes(buf, out_len, code, "dss_get_cert");
}
SEXP r_dss_get_crl(SEXP ext, SEXP index) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = pdf_dss_get_crl(dss_ptr(ext), Rf_asInteger(index), &out_len,
                                   &code);
    return take_bytes(buf, out_len, code, "dss_get_crl");
}
SEXP r_dss_get_ocsp(SEXP ext, SEXP index) {
    int32_t code = 0;
    uintptr_t out_len = 0;
    uint8_t *buf = pdf_dss_get_ocsp(dss_ptr(ext), Rf_asInteger(index), &out_len,
                                    &code);
    return take_bytes(buf, out_len, code, "dss_get_ocsp");
}
SEXP r_dss_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_dss_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── Validation (Document → result handle) ── */
SEXP r_validate_pdf_a(SEXP ext, SEXP level) {
    int32_t code = 0;
    FfiPdfAResults *h = pdf_validate_pdf_a_level(doc_ptr(ext), Rf_asInteger(level),
                                                 &code);
    if (!h) pdfox_raise(code, "validate_pdf_a");
    return wrap_pdf_a_results(h);
}
SEXP r_validate_pdf_ua(SEXP ext, SEXP level) {
    int32_t code = 0;
    FfiUaResults *h = pdf_validate_pdf_ua(doc_ptr(ext), Rf_asInteger(level), &code);
    if (!h) pdfox_raise(code, "validate_pdf_ua");
    return wrap_ua_results(h);
}
SEXP r_validate_pdf_x(SEXP ext, SEXP level) {
    int32_t code = 0;
    FfiPdfXResults *h = pdf_validate_pdf_x_level(doc_ptr(ext), Rf_asInteger(level),
                                                 &code);
    if (!h) pdfox_raise(code, "validate_pdf_x");
    return wrap_pdf_x_results(h);
}
/* PDF/A results */
SEXP r_pdf_a_is_compliant(SEXP ext) {
    int32_t code = 0;
    bool r = pdf_pdf_a_is_compliant(pdf_a_results_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "pdf_a_is_compliant");
    return Rf_ScalarLogical(r);
}
SEXP r_pdf_a_error_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_pdf_a_error_count(pdf_a_results_ptr(ext)));
}
SEXP r_pdf_a_warning_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_pdf_a_warning_count(pdf_a_results_ptr(ext)));
}
SEXP r_pdf_a_get_error(SEXP ext, SEXP index) {
    int32_t code = 0;
    return take_string(pdf_pdf_a_get_error(pdf_a_results_ptr(ext),
                                           Rf_asInteger(index), &code),
                       code, "pdf_a_get_error");
}
SEXP r_pdf_a_results_close(SEXP ext) {
    FfiPdfAResults *h = (FfiPdfAResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_a_results_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
/* PDF/UA results */
SEXP r_pdf_ua_is_accessible(SEXP ext) {
    int32_t code = 0;
    bool r = pdf_pdf_ua_is_accessible(ua_results_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "pdf_ua_is_accessible");
    return Rf_ScalarLogical(r);
}
SEXP r_pdf_ua_error_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_pdf_ua_error_count(ua_results_ptr(ext)));
}
SEXP r_pdf_ua_warning_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_pdf_ua_warning_count(ua_results_ptr(ext)));
}
SEXP r_pdf_ua_get_error(SEXP ext, SEXP index) {
    int32_t code = 0;
    return take_string(pdf_pdf_ua_get_error(ua_results_ptr(ext),
                                            Rf_asInteger(index), &code),
                       code, "pdf_ua_get_error");
}
SEXP r_pdf_ua_get_warning(SEXP ext, SEXP index) {
    int32_t code = 0;
    return take_string(pdf_pdf_ua_get_warning(ua_results_ptr(ext),
                                              Rf_asInteger(index), &code),
                       code, "pdf_ua_get_warning");
}
SEXP r_pdf_ua_get_stats(SEXP ext) {
    int32_t code = 0;
    int32_t s = 0, im = 0, t = 0, f = 0, an = 0, pg = 0;
    bool ok = pdf_pdf_ua_get_stats(ua_results_ptr(ext), &s, &im, &t, &f, &an, &pg,
                                   &code);
    if (!ok && code != 0) pdfox_raise(code, "pdf_ua_get_stats");
    SEXP out = PROTECT(Rf_allocVector(INTSXP, 6));
    INTEGER(out)[0] = s; INTEGER(out)[1] = im; INTEGER(out)[2] = t;
    INTEGER(out)[3] = f; INTEGER(out)[4] = an; INTEGER(out)[5] = pg;
    UNPROTECT(1);
    return out;
}
SEXP r_pdf_ua_results_close(SEXP ext) {
    FfiUaResults *h = (FfiUaResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_ua_results_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
/* PDF/X results */
SEXP r_pdf_x_is_compliant(SEXP ext) {
    int32_t code = 0;
    bool r = pdf_pdf_x_is_compliant(pdf_x_results_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "pdf_x_is_compliant");
    return Rf_ScalarLogical(r);
}
SEXP r_pdf_x_error_count(SEXP ext) {
    return Rf_ScalarInteger(pdf_pdf_x_error_count(pdf_x_results_ptr(ext)));
}
SEXP r_pdf_x_get_error(SEXP ext, SEXP index) {
    int32_t code = 0;
    return take_string(pdf_pdf_x_get_error(pdf_x_results_ptr(ext),
                                           Rf_asInteger(index), &code),
                       code, "pdf_x_get_error");
}
SEXP r_pdf_x_results_close(SEXP ext) {
    FfiPdfXResults *h = (FfiPdfXResults *)R_ExternalPtrAddr(ext);
    if (h) { pdf_pdf_x_results_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}

/* ── PHASE-7 barcodes / OCR / render variants / redaction / constructors /
 * page getters / elements / timestamp ────────────────────────────────────────
 * Reuses the established handle pattern: FfiBarcodeImage is wrapped in its own
 * external pointer with a GC finalizer (pdf_barcode_free) + an explicit
 * idempotent close; FfiElementList is wrapped similarly (pdf_oxide_elements_free)
 * with full per-element accessors exposed as an R list. OCR engine / renderer are
 * opaque void* handles. Redaction is exposed as methods on the existing
 * DocumentEditor wrapper. RenderedImage returns reuse wrap_rendered_image. char*
 * returns use take_string + free_string; owned uint8* buffers use take_bytes
 * (free_bytes); errors raise the same classed pdfoxide_error via pdfox_raise. */

/* ── Barcode (FfiBarcodeImage opaque handle) ── */
static void barcode_finalizer(SEXP ext) {
    FfiBarcodeImage *h = (FfiBarcodeImage *)R_ExternalPtrAddr(ext);
    if (h) { pdf_barcode_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_barcode(FfiBarcodeImage *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, barcode_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static FfiBarcodeImage *barcode_ptr(SEXP ext) {
    FfiBarcodeImage *h = (FfiBarcodeImage *)R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: barcode handle is closed");
    return h;
}

SEXP r_generate_qr_code(SEXP data, SEXP error_correction, SEXP size_px) {
    int32_t code = 0;
    FfiBarcodeImage *h = pdf_generate_qr_code(CHAR(STRING_ELT(data, 0)),
                                              Rf_asInteger(error_correction),
                                              Rf_asInteger(size_px), &code);
    if (!h) pdfox_raise(code, "generate_qr_code");
    return wrap_barcode(h);
}
SEXP r_generate_barcode(SEXP data, SEXP format, SEXP size_px) {
    int32_t code = 0;
    FfiBarcodeImage *h = pdf_generate_barcode(CHAR(STRING_ELT(data, 0)),
                                              Rf_asInteger(format),
                                              Rf_asInteger(size_px), &code);
    if (!h) pdfox_raise(code, "generate_barcode");
    return wrap_barcode(h);
}
SEXP r_barcode_get_data(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_barcode_get_data(barcode_ptr(ext), &code), code,
                       "barcode_get_data");
}
SEXP r_barcode_get_format(SEXP ext) {
    int32_t code = 0;
    int32_t f = pdf_barcode_get_format(barcode_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "barcode_get_format");
    return Rf_ScalarInteger(f);
}
SEXP r_barcode_get_confidence(SEXP ext) {
    int32_t code = 0;
    float c = pdf_barcode_get_confidence(barcode_ptr(ext), &code);
    if (code != 0) pdfox_raise(code, "barcode_get_confidence");
    return Rf_ScalarReal((double)c);
}
SEXP r_barcode_get_image_png(SEXP ext, SEXP size_px) {
    int32_t code = 0, len = 0;
    uint8_t *p = pdf_barcode_get_image_png(barcode_ptr(ext), Rf_asInteger(size_px),
                                           &len, &code);
    if (!p) pdfox_raise(code, "barcode_get_image_png");
    R_xlen_t n = len < 0 ? 0 : (R_xlen_t)len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    if (n) memcpy(RAW(out), p, (size_t)n);
    free_bytes(p);
    UNPROTECT(1);
    return out;
}
SEXP r_barcode_get_svg(SEXP ext, SEXP size_px) {
    int32_t code = 0;
    return take_string(pdf_barcode_get_svg(barcode_ptr(ext), Rf_asInteger(size_px),
                                           &code), code, "barcode_get_svg");
}
SEXP r_barcode_close(SEXP ext) {
    FfiBarcodeImage *h = (FfiBarcodeImage *)R_ExternalPtrAddr(ext);
    if (h) { pdf_barcode_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
SEXP r_editor_add_barcode_to_page(SEXP ext, SEXP page, SEXP bc, SEXP x, SEXP y,
                                  SEXP w, SEXP h) {
    int32_t code = 0;
    if (pdf_add_barcode_to_page(editor_ptr(ext), Rf_asInteger(page),
                                barcode_ptr(bc), (float)Rf_asReal(x),
                                (float)Rf_asReal(y), (float)Rf_asReal(w),
                                (float)Rf_asReal(h), &code) != 0)
        pdfox_raise(code, "add_barcode_to_page");
    return R_NilValue;
}

/* ── OCR engine (opaque void*) ── */
static void ocr_engine_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_ocr_engine_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_ocr_engine(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, ocr_engine_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
static void *ocr_engine_ptr(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (!h) Rf_error("pdf_oxide: OCR engine handle is closed");
    return h;
}
SEXP r_ocr_engine_create(SEXP det_path, SEXP rec_path, SEXP dict_path) {
    int32_t code = 0;
    void *h = pdf_ocr_engine_create(CHAR(STRING_ELT(det_path, 0)),
                                    CHAR(STRING_ELT(rec_path, 0)),
                                    CHAR(STRING_ELT(dict_path, 0)), &code);
    if (!h) pdfox_raise(code, "ocr_engine_create");
    return wrap_ocr_engine(h);
}
SEXP r_ocr_engine_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_ocr_engine_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
SEXP r_ocr_page_needs_ocr(SEXP ext, SEXP page) {
    int32_t code = 0;
    bool r = pdf_ocr_page_needs_ocr(doc_ptr(ext), Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "ocr_page_needs_ocr");
    return Rf_ScalarLogical(r);
}
/* engine may be NULL (native-only extraction). */
SEXP r_ocr_extract_text(SEXP ext, SEXP page, SEXP engine) {
    int32_t code = 0;
    const void *eng = (engine == R_NilValue) ? NULL : ocr_engine_ptr(engine);
    return take_string(pdf_ocr_extract_text(doc_ptr(ext), Rf_asInteger(page), eng,
                                            &code), code, "ocr_extract_text");
}

/* ── Render variants (all reuse wrap_rendered_image) ── */
SEXP r_render_page_with_options(SEXP ext, SEXP page, SEXP dpi, SEXP format,
                                SEXP bg_r, SEXP bg_g, SEXP bg_b, SEXP bg_a,
                                SEXP transparent_bg, SEXP render_annots,
                                SEXP jpeg_quality) {
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_with_options(
        doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(dpi), Rf_asInteger(format),
        (float)Rf_asReal(bg_r), (float)Rf_asReal(bg_g), (float)Rf_asReal(bg_b),
        (float)Rf_asReal(bg_a), Rf_asInteger(transparent_bg),
        Rf_asInteger(render_annots), Rf_asInteger(jpeg_quality), &code);
    if (!img) pdfox_raise(code, "render_page_with_options");
    return wrap_rendered_image(img);
}
SEXP r_render_page_with_options_ex(SEXP ext, SEXP page, SEXP dpi, SEXP format,
                                   SEXP bg_r, SEXP bg_g, SEXP bg_b, SEXP bg_a,
                                   SEXP transparent_bg, SEXP render_annots,
                                   SEXP jpeg_quality, SEXP excluded_layers) {
    int32_t code = 0;
    uintptr_t n = 0;
    const char **layers = strvec_to_c(excluded_layers, &n);
    FfiRenderedImage *img = pdf_render_page_with_options_ex(
        doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(dpi), Rf_asInteger(format),
        (float)Rf_asReal(bg_r), (float)Rf_asReal(bg_g), (float)Rf_asReal(bg_b),
        (float)Rf_asReal(bg_a), Rf_asInteger(transparent_bg),
        Rf_asInteger(render_annots), Rf_asInteger(jpeg_quality),
        (const char *const *)layers, n, &code);
    if (!img) pdfox_raise(code, "render_page_with_options_ex");
    return wrap_rendered_image(img);
}
SEXP r_render_page_region(SEXP ext, SEXP page, SEXP crop_x, SEXP crop_y,
                          SEXP crop_w, SEXP crop_h, SEXP format) {
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_region(
        doc_ptr(ext), Rf_asInteger(page), (float)Rf_asReal(crop_x),
        (float)Rf_asReal(crop_y), (float)Rf_asReal(crop_w),
        (float)Rf_asReal(crop_h), Rf_asInteger(format), &code);
    if (!img) pdfox_raise(code, "render_page_region");
    return wrap_rendered_image(img);
}
SEXP r_render_page_fit(SEXP ext, SEXP page, SEXP w, SEXP h, SEXP format) {
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_fit(
        doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(w), Rf_asInteger(h),
        Rf_asInteger(format), &code);
    if (!img) pdfox_raise(code, "render_page_fit");
    return wrap_rendered_image(img);
}
/* raw render also surfaces the (out_width, out_height) it sets. */
SEXP r_render_page_raw(SEXP ext, SEXP page, SEXP dpi) {
    int32_t code = 0, ow = 0, oh = 0;
    FfiRenderedImage *img = pdf_render_page_raw(
        doc_ptr(ext), Rf_asInteger(page), Rf_asInteger(dpi), &ow, &oh, &code);
    if (!img) pdfox_raise(code, "render_page_raw");
    return wrap_rendered_image(img);
}

/* ── Renderer (opaque void*) + estimate ── */
static void renderer_finalizer(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_renderer_free(h); R_ClearExternalPtr(ext); }
}
static SEXP wrap_renderer(void *h) {
    SEXP ext = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, renderer_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}
SEXP r_create_renderer(SEXP dpi, SEXP format, SEXP quality, SEXP anti_alias) {
    int32_t code = 0;
    void *h = pdf_create_renderer(Rf_asInteger(dpi), Rf_asInteger(format),
                                  Rf_asInteger(quality),
                                  Rf_asLogical(anti_alias) == TRUE, &code);
    if (!h) pdfox_raise(code, "create_renderer");
    return wrap_renderer(h);
}
SEXP r_renderer_close(SEXP ext) {
    void *h = R_ExternalPtrAddr(ext);
    if (h) { pdf_renderer_free(h); R_ClearExternalPtr(ext); }
    return R_NilValue;
}
SEXP r_estimate_render_time(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t t = pdf_estimate_render_time(doc_ptr(ext), Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "estimate_render_time");
    return Rf_ScalarInteger(t);
}

/* ── Redaction (methods on the existing DocumentEditor wrapper) ── */
SEXP r_redaction_add(SEXP ext, SEXP page, SEXP x1, SEXP y1, SEXP x2, SEXP y2,
                     SEXP r, SEXP g, SEXP b) {
    int32_t code = 0;
    if (pdf_redaction_add(editor_ptr(ext), (uintptr_t)Rf_asInteger(page),
                          Rf_asReal(x1), Rf_asReal(y1), Rf_asReal(x2),
                          Rf_asReal(y2), Rf_asReal(r), Rf_asReal(g), Rf_asReal(b),
                          &code) != 0)
        pdfox_raise(code, "redaction_add");
    return R_NilValue;
}
SEXP r_redaction_count(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t n = pdf_redaction_count(editor_ptr(ext), (uintptr_t)Rf_asInteger(page),
                                    &code);
    if (n < 0) pdfox_raise(code, "redaction_count");
    return Rf_ScalarInteger(n);
}
SEXP r_redaction_apply(SEXP ext, SEXP scrub_metadata, SEXP r, SEXP g, SEXP b) {
    int32_t code = 0;
    int32_t n = pdf_redaction_apply(editor_ptr(ext),
                                    Rf_asLogical(scrub_metadata) == TRUE,
                                    Rf_asReal(r), Rf_asReal(g), Rf_asReal(b),
                                    &code);
    if (n < 0) pdfox_raise(code, "redaction_apply");
    return Rf_ScalarInteger(n);
}
SEXP r_redaction_scrub_metadata(SEXP ext) {
    int32_t code = 0;
    int32_t n = pdf_redaction_scrub_metadata(editor_ptr(ext), &code);
    if (n < 0) pdfox_raise(code, "redaction_scrub_metadata");
    return Rf_ScalarInteger(n);
}

/* ── Constructors (return Pdf*, reuse wrap_pdf) ── */
SEXP r_pdf_from_image(SEXP path) {
    int32_t code = 0;
    Pdf *h = pdf_from_image(CHAR(STRING_ELT(path, 0)), &code);
    if (!h) pdfox_raise(code, "from_image");
    return wrap_pdf(h);
}
SEXP r_pdf_from_image_bytes(SEXP raw) {
    int32_t code = 0;
    Pdf *h = pdf_from_image_bytes(RAW(raw), (int32_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "from_image_bytes");
    return wrap_pdf(h);
}
SEXP r_pdf_from_html_css(SEXP html, SEXP css, SEXP font_bytes) {
    int32_t code = 0;
    const uint8_t *fb = (font_bytes == R_NilValue) ? NULL : RAW(font_bytes);
    uintptr_t fl = (font_bytes == R_NilValue) ? 0 : (uintptr_t)XLENGTH(font_bytes);
    Pdf *h = pdf_from_html_css(CHAR(STRING_ELT(html, 0)),
                               CHAR(STRING_ELT(css, 0)), fb, fl, &code);
    if (!h) pdfox_raise(code, "from_html_css");
    return wrap_pdf(h);
}
SEXP r_pdf_from_html_css_with_fonts(SEXP html, SEXP css, SEXP families,
                                    SEXP font_bytes) {
    int32_t code = 0;
    uintptr_t n_fam = 0;
    const char **fams = strvec_to_c(families, &n_fam);
    const uintptr_t *font_lens;
    uintptr_t n_fonts = 0;
    const uint8_t *const *fonts = rawlist_to_c(font_bytes, &font_lens, &n_fonts);
    uintptr_t count = (n_fam < n_fonts) ? n_fam : n_fonts;
    Pdf *h = pdf_from_html_css_with_fonts(
        CHAR(STRING_ELT(html, 0)), CHAR(STRING_ELT(css, 0)),
        (const char *const *)fams, fonts, font_lens, count, &code);
    if (!h) pdfox_raise(code, "from_html_css_with_fonts");
    return wrap_pdf(h);
}
SEXP r_pdf_merge(SEXP paths) {
    int32_t code = 0, data_len = 0;
    uintptr_t n = 0;
    const char **arr = strvec_to_c(paths, &n);
    uint8_t *p = pdf_merge((const char *const *)arr, (int32_t)n, &data_len, &code);
    if (!p) pdfox_raise(code, "merge");
    R_xlen_t dn = data_len < 0 ? 0 : (R_xlen_t)data_len;
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, dn));
    if (dn) memcpy(RAW(out), p, (size_t)dn);
    free_bytes(p);
    UNPROTECT(1);
    return out;
}

/* ── Page getters (on Document, 0-based page) ── */
SEXP r_page_get_width(SEXP ext, SEXP page) {
    int32_t code = 0;
    float w = pdf_page_get_width(doc_ptr(ext), Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "page_get_width");
    return Rf_ScalarReal((double)w);
}
SEXP r_page_get_height(SEXP ext, SEXP page) {
    int32_t code = 0;
    float h = pdf_page_get_height(doc_ptr(ext), Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "page_get_height");
    return Rf_ScalarReal((double)h);
}
SEXP r_page_get_rotation(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t r = pdf_page_get_rotation(doc_ptr(ext), Rf_asInteger(page), &code);
    if (code != 0) pdfox_raise(code, "page_get_rotation");
    return Rf_ScalarInteger(r);
}

/* ── ElementList (FfiElementList opaque handle, full per-element accessors) ── */
SEXP r_page_get_elements(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiElementList *list =
        pdf_page_get_elements(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "page_get_elements");
    int32_t n = pdf_oxide_element_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *type = pdf_oxide_element_get_type(list, i, &code);
        if (!type) { pdf_oxide_elements_free(list); pdfox_raise(code, "page_get_elements"); }
        code = 0;
        char *txt = pdf_oxide_element_get_text(list, i, &code);
        if (!txt) { free_string(type); pdf_oxide_elements_free(list); pdfox_raise(code, "page_get_elements"); }
        float x = 0, y = 0, w = 0, h = 0;
        code = 0;
        pdf_oxide_element_get_rect(list, i, &x, &y, &w, &h, &code);
        if (code != 0) { free_string(type); free_string(txt); pdf_oxide_elements_free(list); pdfox_raise(code, "page_get_elements"); }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 3));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
        SEXP tstr = PROTECT(Rf_mkChar(type)); free_string(type);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(tstr));          SET_STRING_ELT(nms, 0, Rf_mkChar("type"));
        SEXP txstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 1, Rf_ScalarString(txstr));         SET_STRING_ELT(nms, 1, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 2, make_bbox(x, y, w, h));          SET_STRING_ELT(nms, 2, Rf_mkChar("rect"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(4);
    }
    pdf_oxide_elements_free(list);
    UNPROTECT(1);
    return out;
}

/* ── Timestamp (top-level fn returning bytes via out-params) ── */
SEXP r_add_timestamp(SEXP pdf_data, SEXP sig_index, SEXP tsa_url) {
    int32_t code = 0;
    uint8_t *out_data = NULL;
    uintptr_t out_len = 0;
    const char *url = (tsa_url == R_NilValue) ? NULL : CHAR(STRING_ELT(tsa_url, 0));
    bool ok = pdf_add_timestamp(RAW(pdf_data), (uintptr_t)XLENGTH(pdf_data),
                                Rf_asInteger(sig_index), url, &out_data, &out_len,
                                &code);
    if (!ok || !out_data) pdfox_raise(code, "add_timestamp");
    return take_bytes(out_data, out_len, code, "add_timestamp");
}

/* ══ PHASE-8: 100%-coverage closeout — every remaining C-ABI symbol ══════════
 * Same contracts as the earlier phases: char* returns -> take_string/free_string;
 * owned uint8* -> take_bytes/free_bytes; LIST handles are opened, drained into
 * named R lists with the matching *_list_free, then freed; non-success status or
 * a set error_code raises a classed pdfoxide_error. */

/* ── OFFICE: open-from-bytes (-> Document) ── */
SEXP r_doc_open_from_docx_bytes(SEXP raw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_docx_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "open_from_docx_bytes");
    return wrap_doc(h);
}
SEXP r_doc_open_from_pptx_bytes(SEXP raw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_pptx_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "open_from_pptx_bytes");
    return wrap_doc(h);
}
SEXP r_doc_open_from_xlsx_bytes(SEXP raw) {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_xlsx_bytes(RAW(raw), (uintptr_t)XLENGTH(raw), &code);
    if (!h) pdfox_raise(code, "open_from_xlsx_bytes");
    return wrap_doc(h);
}
/* ── OFFICE: to-office bytes (-> raw via free_bytes) ── */
SEXP r_doc_to_docx(SEXP ext) {
    int32_t code = 0; uintptr_t out_len = 0;
    uint8_t *p = pdf_document_to_docx(doc_ptr(ext), &out_len, &code);
    return take_bytes(p, out_len, code, "to_docx");
}
SEXP r_doc_to_pptx(SEXP ext) {
    int32_t code = 0; uintptr_t out_len = 0;
    uint8_t *p = pdf_document_to_pptx(doc_ptr(ext), &out_len, &code);
    return take_bytes(p, out_len, code, "to_pptx");
}
SEXP r_doc_to_xlsx(SEXP ext) {
    int32_t code = 0; uintptr_t out_len = 0;
    uint8_t *p = pdf_document_to_xlsx(doc_ptr(ext), &out_len, &code);
    return take_bytes(p, out_len, code, "to_xlsx");
}

/* ── IN-RECT extractors (reuse the element list marshalling) ── */
SEXP r_doc_extract_text_in_rect(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    return take_string(
        pdf_document_extract_text_in_rect(doc_ptr(ext), Rf_asInteger(page),
            (float)Rf_asReal(x), (float)Rf_asReal(y), (float)Rf_asReal(w),
            (float)Rf_asReal(h), &code),
        code, "extract_text_in_rect");
}
SEXP r_doc_extract_words_in_rect(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    FfiWordList *list = pdf_document_extract_words_in_rect(doc_ptr(ext),
        Rf_asInteger(page), (float)Rf_asReal(x), (float)Rf_asReal(y),
        (float)Rf_asReal(w), (float)Rf_asReal(h), &code);
    if (!list) pdfox_raise(code, "extract_words_in_rect");
    int32_t n = pdf_oxide_word_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *txt = pdf_oxide_word_get_text(list, i, &code);
        if (!txt) { pdf_oxide_word_list_free(list); pdfox_raise(code, "extract_words_in_rect"); }
        float bx = 0, by = 0, bw = 0, bh = 0;
        code = 0;
        pdf_oxide_word_get_bbox(list, i, &bx, &by, &bw, &bh, &code);
        if (code != 0) { free_string(txt); pdf_oxide_word_list_free(list); pdfox_raise(code, "extract_words_in_rect"); }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 2));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 2));
        SEXP txtstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(txtstr));  SET_STRING_ELT(nms, 0, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 1, make_bbox(bx, by, bw, bh)); SET_STRING_ELT(nms, 1, Rf_mkChar("bbox"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_word_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_doc_extract_lines_in_rect(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    FfiTextLineList *list = pdf_document_extract_lines_in_rect(doc_ptr(ext),
        Rf_asInteger(page), (float)Rf_asReal(x), (float)Rf_asReal(y),
        (float)Rf_asReal(w), (float)Rf_asReal(h), &code);
    if (!list) pdfox_raise(code, "extract_lines_in_rect");
    int32_t n = pdf_oxide_line_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *txt = pdf_oxide_line_get_text(list, i, &code);
        if (!txt) { pdf_oxide_line_list_free(list); pdfox_raise(code, "extract_lines_in_rect"); }
        float bx = 0, by = 0, bw = 0, bh = 0;
        code = 0;
        pdf_oxide_line_get_bbox(list, i, &bx, &by, &bw, &bh, &code);
        if (code != 0) { free_string(txt); pdf_oxide_line_list_free(list); pdfox_raise(code, "extract_lines_in_rect"); }
        code = 0;
        int32_t wc = pdf_oxide_line_get_word_count(list, i, &code);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 3));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
        SEXP txtstr = PROTECT(Rf_mkChar(txt)); free_string(txt);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(txtstr));  SET_STRING_ELT(nms, 0, Rf_mkChar("text"));
        SET_VECTOR_ELT(rec, 1, make_bbox(bx, by, bw, bh)); SET_STRING_ELT(nms, 1, Rf_mkChar("bbox"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarInteger(wc));      SET_STRING_ELT(nms, 2, Rf_mkChar("word_count"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_line_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_doc_extract_tables_in_rect(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    FfiTableList *list = pdf_document_extract_tables_in_rect(doc_ptr(ext),
        Rf_asInteger(page), (float)Rf_asReal(x), (float)Rf_asReal(y),
        (float)Rf_asReal(w), (float)Rf_asReal(h), &code);
    if (!list) pdfox_raise(code, "extract_tables_in_rect");
    int32_t n = pdf_oxide_table_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        int32_t rows = pdf_oxide_table_get_row_count(list, i, &code);
        if (code != 0) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables_in_rect"); }
        code = 0;
        int32_t cols = pdf_oxide_table_get_col_count(list, i, &code);
        if (code != 0) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables_in_rect"); }
        code = 0;
        bool hdr = pdf_oxide_table_has_header(list, i, &code);
        if (rows < 0) rows = 0;
        if (cols < 0) cols = 0;
        SEXP cells = PROTECT(Rf_allocMatrix(STRSXP, rows, cols));
        for (int32_t r = 0; r < rows; r++) {
            for (int32_t c = 0; c < cols; c++) {
                code = 0;
                char *cell = pdf_oxide_table_get_cell_text(list, i, r, c, &code);
                if (!cell) { pdf_oxide_table_list_free(list); pdfox_raise(code, "extract_tables_in_rect"); }
                SET_STRING_ELT(cells, r + c * rows, Rf_mkChar(cell));
                free_string(cell);
            }
        }
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 4));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
        SET_VECTOR_ELT(rec, 0, Rf_ScalarInteger(rows)); SET_STRING_ELT(nms, 0, Rf_mkChar("row_count"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarInteger(cols)); SET_STRING_ELT(nms, 1, Rf_mkChar("col_count"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarLogical(hdr));  SET_STRING_ELT(nms, 2, Rf_mkChar("has_header"));
        SET_VECTOR_ELT(rec, 3, cells);                  SET_STRING_ELT(nms, 3, Rf_mkChar("cells"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(3);
    }
    pdf_oxide_table_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_doc_extract_images_in_rect(SEXP ext, SEXP page, SEXP x, SEXP y, SEXP w, SEXP h) {
    int32_t code = 0;
    FfiImageList *list = pdf_document_extract_images_in_rect(doc_ptr(ext),
        Rf_asInteger(page), (float)Rf_asReal(x), (float)Rf_asReal(y),
        (float)Rf_asReal(w), (float)Rf_asReal(h), &code);
    if (!list) pdfox_raise(code, "extract_images_in_rect");
    int32_t n = pdf_oxide_image_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        int32_t iw = pdf_oxide_image_get_width(list, i, &code);
        if (code != 0) { pdf_oxide_image_list_free(list); pdfox_raise(code, "extract_images_in_rect"); }
        code = 0;
        int32_t ih = pdf_oxide_image_get_height(list, i, &code);
        if (code != 0) { pdf_oxide_image_list_free(list); pdfox_raise(code, "extract_images_in_rect"); }
        code = 0;
        int32_t bpc = pdf_oxide_image_get_bits_per_component(list, i, &code);
        code = 0;
        char *fmt = pdf_oxide_image_get_format(list, i, &code);
        if (!fmt) { pdf_oxide_image_list_free(list); pdfox_raise(code, "extract_images_in_rect"); }
        code = 0;
        char *cs = pdf_oxide_image_get_colorspace(list, i, &code);
        if (!cs) { free_string(fmt); pdf_oxide_image_list_free(list); pdfox_raise(code, "extract_images_in_rect"); }
        code = 0;
        int32_t dlen = 0;
        uint8_t *data = pdf_oxide_image_get_data(list, i, &dlen, &code);
        if (!data) { free_string(fmt); free_string(cs); pdf_oxide_image_list_free(list); pdfox_raise(code, "extract_images_in_rect"); }
        R_xlen_t dn = dlen < 0 ? 0 : (R_xlen_t)dlen;
        SEXP rawd = PROTECT(Rf_allocVector(RAWSXP, dn));
        if (dn) memcpy(RAW(rawd), data, (size_t)dn);
        free_bytes(data);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 6));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 6));
        SET_VECTOR_ELT(rec, 0, Rf_ScalarInteger(iw));   SET_STRING_ELT(nms, 0, Rf_mkChar("width"));
        SET_VECTOR_ELT(rec, 1, Rf_ScalarInteger(ih));   SET_STRING_ELT(nms, 1, Rf_mkChar("height"));
        SET_VECTOR_ELT(rec, 2, Rf_ScalarInteger(bpc));  SET_STRING_ELT(nms, 2, Rf_mkChar("bits_per_component"));
        SEXP fstr = PROTECT(Rf_mkChar(fmt)); free_string(fmt);
        SET_VECTOR_ELT(rec, 3, Rf_ScalarString(fstr));  SET_STRING_ELT(nms, 3, Rf_mkChar("format"));
        SEXP csstr = PROTECT(Rf_mkChar(cs)); free_string(cs);
        SET_VECTOR_ELT(rec, 4, Rf_ScalarString(csstr)); SET_STRING_ELT(nms, 4, Rf_mkChar("colorspace"));
        SET_VECTOR_ELT(rec, 5, rawd);                   SET_STRING_ELT(nms, 5, Rf_mkChar("data"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(4);
    }
    pdf_oxide_image_list_free(list);
    UNPROTECT(1);
    return out;
}

/* ── AUTO extraction + classification (all char* JSON / string) ── */
SEXP r_doc_extract_all_text(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_extract_all_text(doc_ptr(ext), &code), code,
                       "extract_all_text");
}
SEXP r_doc_extract_text_auto(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_extract_text_auto(doc_ptr(ext), Rf_asInteger(page), &code),
        code, "extract_text_auto");
}
SEXP r_doc_extract_page_auto(SEXP ext, SEXP page, SEXP options_json) {
    int32_t code = 0;
    const char *opts = (options_json == R_NilValue) ? NULL : CHAR(STRING_ELT(options_json, 0));
    return take_string(
        pdf_document_extract_page_auto(doc_ptr(ext), Rf_asInteger(page), opts, &code),
        code, "extract_page_auto");
}
SEXP r_doc_classify_page(SEXP ext, SEXP page) {
    int32_t code = 0;
    return take_string(
        pdf_document_classify_page(doc_ptr(ext), Rf_asInteger(page), &code),
        code, "classify_page");
}
SEXP r_doc_classify_document(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_classify_document(doc_ptr(ext), &code), code,
                       "classify_document");
}

/* ── HEADER / FOOTER / ARTIFACT (all int32 status) ── */
SEXP r_doc_remove_headers(SEXP ext, SEXP threshold) {
    int32_t code = 0;
    int32_t n = pdf_document_remove_headers(doc_ptr(ext), (float)Rf_asReal(threshold), &code);
    if (n < 0) pdfox_raise(code, "remove_headers");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_remove_footers(SEXP ext, SEXP threshold) {
    int32_t code = 0;
    int32_t n = pdf_document_remove_footers(doc_ptr(ext), (float)Rf_asReal(threshold), &code);
    if (n < 0) pdfox_raise(code, "remove_footers");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_remove_artifacts(SEXP ext, SEXP threshold) {
    int32_t code = 0;
    int32_t n = pdf_document_remove_artifacts(doc_ptr(ext), (float)Rf_asReal(threshold), &code);
    if (n < 0) pdfox_raise(code, "remove_artifacts");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_erase_header(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t n = pdf_document_erase_header(doc_ptr(ext), Rf_asInteger(page), &code);
    if (n < 0) pdfox_raise(code, "erase_header");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_erase_footer(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t n = pdf_document_erase_footer(doc_ptr(ext), Rf_asInteger(page), &code);
    if (n < 0) pdfox_raise(code, "erase_footer");
    return Rf_ScalarInteger(n);
}
SEXP r_doc_erase_artifacts(SEXP ext, SEXP page) {
    int32_t code = 0;
    int32_t n = pdf_document_erase_artifacts(doc_ptr(ext), Rf_asInteger(page), &code);
    if (n < 0) pdfox_raise(code, "erase_artifacts");
    return Rf_ScalarInteger(n);
}

/* ── FORMS ── */
SEXP r_doc_get_form_fields(SEXP ext) {
    int32_t code = 0;
    FfiFormFieldList *list = pdf_document_get_form_fields(doc_ptr(ext), &code);
    if (!list) pdfox_raise(code, "get_form_fields");
    int32_t n = pdf_oxide_form_field_count(list);
    if (n < 0) n = 0;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (int32_t i = 0; i < n; i++) {
        code = 0;
        char *name = pdf_oxide_form_field_get_name(list, i, &code);
        if (!name) { pdf_oxide_form_field_list_free(list); pdfox_raise(code, "get_form_fields"); }
        code = 0;
        char *type = pdf_oxide_form_field_get_type(list, i, &code);
        if (!type) { free_string(name); pdf_oxide_form_field_list_free(list); pdfox_raise(code, "get_form_fields"); }
        code = 0;
        char *value = pdf_oxide_form_field_get_value(list, i, &code);
        if (!value) { free_string(name); free_string(type); pdf_oxide_form_field_list_free(list); pdfox_raise(code, "get_form_fields"); }
        code = 0;
        bool ro = pdf_oxide_form_field_is_readonly(list, i, &code);
        code = 0;
        bool req = pdf_oxide_form_field_is_required(list, i, &code);
        SEXP rec = PROTECT(Rf_allocVector(VECSXP, 5));
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, 5));
        SEXP nstr = PROTECT(Rf_mkChar(name)); free_string(name);
        SET_VECTOR_ELT(rec, 0, Rf_ScalarString(nstr));  SET_STRING_ELT(nms, 0, Rf_mkChar("name"));
        SEXP tstr = PROTECT(Rf_mkChar(type)); free_string(type);
        SET_VECTOR_ELT(rec, 1, Rf_ScalarString(tstr));  SET_STRING_ELT(nms, 1, Rf_mkChar("type"));
        SEXP vstr = PROTECT(Rf_mkChar(value)); free_string(value);
        SET_VECTOR_ELT(rec, 2, Rf_ScalarString(vstr));  SET_STRING_ELT(nms, 2, Rf_mkChar("value"));
        SET_VECTOR_ELT(rec, 3, Rf_ScalarLogical(ro));   SET_STRING_ELT(nms, 3, Rf_mkChar("readonly"));
        SET_VECTOR_ELT(rec, 4, Rf_ScalarLogical(req));  SET_STRING_ELT(nms, 4, Rf_mkChar("required"));
        Rf_setAttrib(rec, R_NamesSymbol, nms);
        SET_VECTOR_ELT(out, i, rec);
        UNPROTECT(4);
    }
    pdf_oxide_form_field_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_doc_export_form_data_to_bytes(SEXP ext, SEXP format_type) {
    int32_t code = 0; uintptr_t out_len = 0;
    uint8_t *p = pdf_document_export_form_data_to_bytes(doc_ptr(ext),
        Rf_asInteger(format_type), &out_len, &code);
    return take_bytes(p, out_len, code, "export_form_data_to_bytes");
}
SEXP r_doc_import_form_data(SEXP ext, SEXP data_path) {
    int32_t code = 0;
    if (pdf_document_import_form_data(doc_ptr(ext), CHAR(STRING_ELT(data_path, 0)), &code) != 0)
        pdfox_raise(code, "import_form_data");
    return R_NilValue;
}
SEXP r_form_import_from_file(SEXP ext, SEXP filename) {
    int32_t code = 0;
    bool ok = pdf_form_import_from_file(doc_ptr(ext), CHAR(STRING_ELT(filename, 0)), &code);
    if (code != 0) pdfox_raise(code, "form_import_from_file");
    return Rf_ScalarLogical(ok);
}
SEXP r_editor_import_fdf_bytes(SEXP ext, SEXP raw) {
    int32_t code = 0;
    if (pdf_editor_import_fdf_bytes(editor_ptr(ext), RAW(raw),
            (uintptr_t)XLENGTH(raw), &code) != 0)
        pdfox_raise(code, "editor_import_fdf_bytes");
    return R_NilValue;
}
SEXP r_editor_import_xfdf_bytes(SEXP ext, SEXP raw) {
    int32_t code = 0;
    if (pdf_editor_import_xfdf_bytes(editor_ptr(ext), RAW(raw),
            (uintptr_t)XLENGTH(raw), &code) != 0)
        pdfox_raise(code, "editor_import_xfdf_bytes");
    return R_NilValue;
}

/* ── DOC STRUCTURE / METADATA (char* JSON / raw / bool / int) ── */
SEXP r_doc_get_outline(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_get_outline(doc_ptr(ext), &code), code, "get_outline");
}
SEXP r_doc_get_page_labels(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_get_page_labels(doc_ptr(ext), &code), code, "get_page_labels");
}
SEXP r_doc_get_xmp_metadata(SEXP ext) {
    int32_t code = 0;
    return take_string(pdf_document_get_xmp_metadata(doc_ptr(ext), &code), code, "get_xmp_metadata");
}
SEXP r_doc_get_source_bytes(SEXP ext) {
    int32_t code = 0; uintptr_t out_len = 0;
    uint8_t *p = pdf_document_get_source_bytes(doc_ptr(ext), &out_len, &code);
    return take_bytes(p, out_len, code, "get_source_bytes");
}
SEXP r_doc_has_xfa(SEXP ext) {
    return Rf_ScalarLogical(pdf_document_has_xfa(doc_ptr(ext)));
}
SEXP r_doc_plan_split_by_bookmarks(SEXP ext, SEXP options_json) {
    int32_t code = 0;
    const char *opts = (options_json == R_NilValue) ? NULL : CHAR(STRING_ELT(options_json, 0));
    return take_string(
        pdf_document_plan_split_by_bookmarks(doc_ptr(ext), opts, &code),
        code, "plan_split_by_bookmarks");
}
/* pdf_get_page_count operates on a built Pdf* (not a PdfDocument). */
SEXP r_pdf_get_page_count(SEXP ext) {
    int32_t code = 0;
    int32_t n = pdf_get_page_count(pdf_ptr(ext), &code);
    if (n < 0) pdfox_raise(code, "get_page_count");
    return Rf_ScalarInteger(n);
}

/* ── SIGNATURES on document (reuse SignatureInfo / Dss handles) ── */
SEXP r_doc_sign(SEXP ext, SEXP cert_ext, SEXP reason, SEXP location) {
    int32_t code = 0;
    const char *rs = (reason == R_NilValue) ? NULL : CHAR(STRING_ELT(reason, 0));
    const char *loc = (location == R_NilValue) ? NULL : CHAR(STRING_ELT(location, 0));
    if (pdf_document_sign(doc_ptr(ext), certificate_ptr(cert_ext), rs, loc, &code) != 0)
        pdfox_raise(code, "document_sign");
    return R_NilValue;
}
SEXP r_doc_verify_all_signatures(SEXP ext) {
    int32_t code = 0;
    int32_t r = pdf_document_verify_all_signatures(doc_ptr(ext), &code);
    return Rf_ScalarInteger(r);
}
SEXP r_doc_has_timestamp(SEXP ext) {
    int32_t code = 0;
    int32_t r = pdf_document_has_timestamp(doc_ptr(ext), &code);
    if (r < 0) pdfox_raise(code, "has_timestamp");
    return Rf_ScalarLogical(r != 0);
}

/* ── ANNOTATION EXTRAS (extend the FfiAnnotationList accessors) ── */
SEXP r_annotation_get_color(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_get_color");
    code = 0;
    uint32_t color = pdf_oxide_annotation_get_color(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_get_color");
    return Rf_ScalarInteger((int32_t)color);
}
SEXP r_annotation_get_creation_date(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_get_creation_date");
    code = 0;
    int64_t t = pdf_oxide_annotation_get_creation_date(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_get_creation_date");
    return Rf_ScalarReal((double)t);
}
SEXP r_annotation_get_modification_date(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_get_modification_date");
    code = 0;
    int64_t t = pdf_oxide_annotation_get_modification_date(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_get_modification_date");
    return Rf_ScalarReal((double)t);
}
SEXP r_annotation_is_hidden(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_is_hidden");
    code = 0;
    bool r = pdf_oxide_annotation_is_hidden(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_is_hidden");
    return Rf_ScalarLogical(r);
}
SEXP r_annotation_is_marked_deleted(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_is_marked_deleted");
    code = 0;
    bool r = pdf_oxide_annotation_is_marked_deleted(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_is_marked_deleted");
    return Rf_ScalarLogical(r);
}
SEXP r_annotation_is_printable(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_is_printable");
    code = 0;
    bool r = pdf_oxide_annotation_is_printable(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_is_printable");
    return Rf_ScalarLogical(r);
}
SEXP r_annotation_is_read_only(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotation_is_read_only");
    code = 0;
    bool r = pdf_oxide_annotation_is_read_only(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "annotation_is_read_only");
    return Rf_ScalarLogical(r);
}
SEXP r_link_annotation_get_uri(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "link_annotation_get_uri");
    code = 0;
    char *uri = pdf_oxide_link_annotation_get_uri(list, Rf_asInteger(index), &code);
    if (!uri) { pdf_oxide_annotation_list_free(list); pdfox_raise(code, "link_annotation_get_uri"); }
    SEXP out = PROTECT(Rf_mkString(uri));
    free_string(uri);
    pdf_oxide_annotation_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_text_annotation_get_icon_name(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "text_annotation_get_icon_name");
    code = 0;
    char *icon = pdf_oxide_text_annotation_get_icon_name(list, Rf_asInteger(index), &code);
    if (!icon) { pdf_oxide_annotation_list_free(list); pdfox_raise(code, "text_annotation_get_icon_name"); }
    SEXP out = PROTECT(Rf_mkString(icon));
    free_string(icon);
    pdf_oxide_annotation_list_free(list);
    UNPROTECT(1);
    return out;
}
SEXP r_highlight_annotation_quad_points_count(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "highlight_quad_points_count");
    code = 0;
    int32_t n = pdf_oxide_highlight_annotation_get_quad_points_count(list, Rf_asInteger(index), &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "highlight_quad_points_count");
    return Rf_ScalarInteger(n);
}
SEXP r_highlight_annotation_quad_point(SEXP ext, SEXP page, SEXP index, SEXP quad_index) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "highlight_quad_point");
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0, x3 = 0, y3 = 0, x4 = 0, y4 = 0;
    code = 0;
    pdf_oxide_highlight_annotation_get_quad_point(list, Rf_asInteger(index),
        Rf_asInteger(quad_index), &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) pdfox_raise(code, "highlight_quad_point");
    SEXP out = PROTECT(Rf_allocVector(REALSXP, 8));
    REAL(out)[0] = x1; REAL(out)[1] = y1; REAL(out)[2] = x2; REAL(out)[3] = y2;
    REAL(out)[4] = x3; REAL(out)[5] = y3; REAL(out)[6] = x4; REAL(out)[7] = y4;
    UNPROTECT(1);
    return out;
}
SEXP r_annotations_to_json(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "annotations_to_json");
    code = 0;
    char *json = pdf_oxide_annotations_to_json(list, &code);
    pdf_oxide_annotation_list_free(list);
    return take_string(json, code, "annotations_to_json");
}

/* ── ELEMENT / FONT / SEARCH JSON accessors ── */
SEXP r_font_get_size(SEXP ext, SEXP page, SEXP index) {
    int32_t code = 0;
    FfiFontList *list = pdf_document_get_embedded_fonts(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "font_get_size");
    code = 0;
    float sz = pdf_oxide_font_get_size(list, Rf_asInteger(index), &code);
    pdf_oxide_font_list_free(list);
    if (code != 0) pdfox_raise(code, "font_get_size");
    return Rf_ScalarReal((double)sz);
}
SEXP r_fonts_to_json(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiFontList *list = pdf_document_get_embedded_fonts(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "fonts_to_json");
    code = 0;
    char *json = pdf_oxide_fonts_to_json(list, &code);
    pdf_oxide_font_list_free(list);
    return take_string(json, code, "fonts_to_json");
}
SEXP r_elements_to_json(SEXP ext, SEXP page) {
    int32_t code = 0;
    FfiElementList *list = pdf_page_get_elements(doc_ptr(ext), Rf_asInteger(page), &code);
    if (!list) pdfox_raise(code, "elements_to_json");
    code = 0;
    char *json = pdf_oxide_elements_to_json(list, &code);
    pdf_oxide_elements_free(list);
    return take_string(json, code, "elements_to_json");
}
SEXP r_search_results_to_json(SEXP ext, SEXP page, SEXP term, SEXP case_sensitive) {
    int32_t code = 0;
    FfiSearchResults *list = pdf_document_search_page(doc_ptr(ext), Rf_asInteger(page),
        CHAR(STRING_ELT(term, 0)), Rf_asLogical(case_sensitive) == TRUE, &code);
    if (!list) pdfox_raise(code, "search_results_to_json");
    code = 0;
    char *json = pdf_oxide_search_results_to_json(list, &code);
    pdf_oxide_search_result_free(list);
    return take_string(json, code, "search_results_to_json");
}

/* ── CRYPTO / FIPS / governance ── */
SEXP r_crypto_active_provider(void) {
    char *s = pdf_oxide_crypto_active_provider();
    return take_string(s, 0, "crypto_active_provider");
}
SEXP r_crypto_fips_available(void) {
    return Rf_ScalarLogical(pdf_oxide_crypto_fips_available() != 0);
}
SEXP r_crypto_use_fips(void) {
    int32_t r = pdf_oxide_crypto_use_fips();
    return Rf_ScalarInteger(r);
}
SEXP r_crypto_set_policy(SEXP spec) {
    int32_t r = pdf_oxide_crypto_set_policy(CHAR(STRING_ELT(spec, 0)));
    return Rf_ScalarInteger(r);
}
SEXP r_crypto_policy(void) {
    char *s = pdf_oxide_crypto_policy();
    return take_string(s, 0, "crypto_policy");
}
SEXP r_crypto_inventory(void) {
    char *s = pdf_oxide_crypto_inventory();
    return take_string(s, 0, "crypto_inventory");
}
SEXP r_crypto_cbom(void) {
    char *s = pdf_oxide_crypto_cbom();
    return take_string(s, 0, "crypto_cbom");
}

/* ── MODELS / CONFIG ── */
SEXP r_model_manifest(void) {
    char *s = pdf_oxide_model_manifest();
    return take_string(s, 0, "model_manifest");
}
SEXP r_prefetch_available(void) {
    return Rf_ScalarLogical(pdf_oxide_prefetch_available() != 0);
}
SEXP r_prefetch_models(SEXP languages_csv) {
    int32_t code = 0;
    const char *csv = (languages_csv == R_NilValue) ? NULL : CHAR(STRING_ELT(languages_csv, 0));
    return take_string(pdf_oxide_prefetch_models(csv, &code), code, "prefetch_models");
}
SEXP r_set_max_ops_per_stream(SEXP limit) {
    int64_t prev = pdf_oxide_set_max_ops_per_stream((int64_t)Rf_asReal(limit));
    return Rf_ScalarReal((double)prev);
}
SEXP r_set_preserve_unmapped_glyphs(SEXP preserve) {
    int32_t prev = pdf_oxide_set_preserve_unmapped_glyphs(Rf_asInteger(preserve));
    return Rf_ScalarInteger(prev);
}
SEXP r_doc_convert_to_pdf_a(SEXP ext, SEXP level) {
    int32_t code = 0;
    bool ok = pdf_convert_to_pdf_a(doc_ptr(ext), Rf_asInteger(level), &code);
    if (code != 0) pdfox_raise(code, "convert_to_pdf_a");
    return Rf_ScalarLogical(ok);
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
    CDEF(r_doc_embedded_fonts, 2),
    CDEF(r_doc_embedded_images, 2),
    CDEF(r_doc_page_annotations, 2),
    CDEF(r_doc_extract_paths, 2),
    CDEF(r_doc_search, 4),
    CDEF(r_doc_search_all, 3),
    CDEF(r_doc_render_page, 3),
    CDEF(r_doc_render_page_zoom, 4),
    CDEF(r_doc_render_page_thumbnail, 4),
    CDEF(r_rendered_image_width, 1),
    CDEF(r_rendered_image_height, 1),
    CDEF(r_rendered_image_data, 1),
    CDEF(r_rendered_image_save, 2),
    CDEF(r_rendered_image_close, 1),
    CDEF(r_doc_close, 1),
    CDEF(r_pdf_close, 1),
    CDEF(r_editor_open, 1),
    CDEF(r_editor_open_from_bytes, 1),
    CDEF(r_editor_is_modified, 1),
    CDEF(r_editor_source_path, 1),
    CDEF(r_editor_version, 1),
    CDEF(r_editor_page_count, 1),
    CDEF(r_editor_get_producer, 1),
    CDEF(r_editor_set_producer, 2),
    CDEF(r_editor_get_creation_date, 1),
    CDEF(r_editor_set_creation_date, 2),
    CDEF(r_editor_delete_page, 2),
    CDEF(r_editor_move_page, 3),
    CDEF(r_editor_rotate_page_by, 3),
    CDEF(r_editor_rotate_all_pages, 2),
    CDEF(r_editor_set_page_rotation, 3),
    CDEF(r_editor_get_page_rotation, 2),
    CDEF(r_editor_crop_margins, 5),
    CDEF(r_editor_get_page_crop_box, 2),
    CDEF(r_editor_set_page_crop_box, 6),
    CDEF(r_editor_get_page_media_box, 2),
    CDEF(r_editor_set_page_media_box, 6),
    CDEF(r_editor_apply_all_redactions, 1),
    CDEF(r_editor_apply_page_redactions, 2),
    CDEF(r_editor_is_page_marked_for_redaction, 2),
    CDEF(r_editor_unmark_page_for_redaction, 2),
    CDEF(r_editor_erase_region, 6),
    CDEF(r_editor_erase_regions, 3),
    CDEF(r_editor_clear_erase_regions, 2),
    CDEF(r_editor_flatten_forms, 1),
    CDEF(r_editor_flatten_forms_on_page, 2),
    CDEF(r_editor_set_form_field_value, 3),
    CDEF(r_editor_flatten_annotations, 2),
    CDEF(r_editor_flatten_all_annotations, 1),
    CDEF(r_editor_flatten_warnings_count, 1),
    CDEF(r_editor_flatten_warning, 2),
    CDEF(r_editor_is_page_marked_for_flatten, 2),
    CDEF(r_editor_unmark_page_for_flatten, 2),
    CDEF(r_editor_merge_from, 2),
    CDEF(r_editor_merge_from_bytes, 2),
    CDEF(r_editor_convert_to_pdf_a, 2),
    CDEF(r_editor_embed_file, 3),
    CDEF(r_editor_extract_pages_to_bytes, 2),
    CDEF(r_editor_save, 2),
    CDEF(r_editor_save_to_bytes, 1),
    CDEF(r_editor_save_to_bytes_with_options, 4),
    CDEF(r_editor_save_encrypted, 4),
    CDEF(r_editor_save_encrypted_to_bytes, 3),
    CDEF(r_editor_close, 1),
    /* PDF creation builder API */
    CDEF(r_embedded_font_from_file, 1),
    CDEF(r_embedded_font_from_bytes, 2),
    CDEF(r_embedded_font_close, 1),
    CDEF(r_builder_create, 0),
    CDEF(r_builder_close, 1),
    CDEF(r_builder_set_title, 2),
    CDEF(r_builder_set_author, 2),
    CDEF(r_builder_set_subject, 2),
    CDEF(r_builder_set_keywords, 2),
    CDEF(r_builder_set_creator, 2),
    CDEF(r_builder_on_open, 2),
    CDEF(r_builder_language, 2),
    CDEF(r_builder_tagged_pdf_ua1, 1),
    CDEF(r_builder_role_map, 3),
    CDEF(r_builder_register_embedded_font, 3),
    CDEF(r_builder_a4_page, 1),
    CDEF(r_builder_letter_page, 1),
    CDEF(r_builder_page, 3),
    CDEF(r_builder_build, 1),
    CDEF(r_builder_save, 2),
    CDEF(r_builder_save_encrypted, 4),
    CDEF(r_builder_to_bytes_encrypted, 3),
    CDEF(r_page_font, 3),
    CDEF(r_page_at, 3),
    CDEF(r_page_text, 2),
    CDEF(r_page_heading, 3),
    CDEF(r_page_paragraph, 2),
    CDEF(r_page_space, 2),
    CDEF(r_page_horizontal_rule, 1),
    CDEF(r_page_link_url, 2),
    CDEF(r_page_link_page, 2),
    CDEF(r_page_link_named, 2),
    CDEF(r_page_link_javascript, 2),
    CDEF(r_page_on_open, 2),
    CDEF(r_page_on_close, 2),
    CDEF(r_page_field_keystroke, 2),
    CDEF(r_page_field_format, 2),
    CDEF(r_page_field_validate, 2),
    CDEF(r_page_field_calculate, 2),
    CDEF(r_page_highlight, 4),
    CDEF(r_page_underline, 4),
    CDEF(r_page_strikeout, 4),
    CDEF(r_page_squiggly, 4),
    CDEF(r_page_sticky_note, 2),
    CDEF(r_page_sticky_note_at, 4),
    CDEF(r_page_watermark, 2),
    CDEF(r_page_watermark_confidential, 1),
    CDEF(r_page_watermark_draft, 1),
    CDEF(r_page_stamp, 2),
    CDEF(r_page_freetext, 6),
    CDEF(r_page_footnote, 3),
    CDEF(r_page_columns, 4),
    CDEF(r_page_inline, 2),
    CDEF(r_page_inline_bold, 2),
    CDEF(r_page_inline_italic, 2),
    CDEF(r_page_inline_color, 5),
    CDEF(r_page_newline, 1),
    CDEF(r_page_text_field, 7),
    CDEF(r_page_checkbox, 7),
    CDEF(r_page_combo_box, 8),
    CDEF(r_page_radio_group, 8),
    CDEF(r_page_push_button, 7),
    CDEF(r_page_signature_field, 6),
    CDEF(r_page_barcode_1d, 7),
    CDEF(r_page_barcode_qr, 5),
    CDEF(r_page_image, 6),
    CDEF(r_page_image_with_alt, 7),
    CDEF(r_page_image_artifact, 6),
    CDEF(r_page_rect, 5),
    CDEF(r_page_filled_rect, 8),
    CDEF(r_page_line, 5),
    CDEF(r_page_stroke_rect, 9),
    CDEF(r_page_stroke_line, 9),
    CDEF(r_page_stroke_rect_dashed, 11),
    CDEF(r_page_stroke_line_dashed, 11),
    CDEF(r_page_text_in_rect, 7),
    CDEF(r_page_new_page_same_size, 1),
    CDEF(r_page_table, 7),
    CDEF(r_page_streaming_table_begin, 6),
    CDEF(r_page_streaming_table_begin_v2, 11),
    CDEF(r_page_streaming_table_set_batch_size, 2),
    CDEF(r_page_streaming_table_pending_row_count, 1),
    CDEF(r_page_streaming_table_batch_count, 1),
    CDEF(r_page_streaming_table_flush, 1),
    CDEF(r_page_streaming_table_push_row, 2),
    CDEF(r_page_streaming_table_push_row_v2, 3),
    CDEF(r_page_streaming_table_finish, 1),
    CDEF(r_page_done, 1),
    CDEF(r_page_close, 1),
    /* PHASE-6 signatures / PKI / timestamps / TSA / DSS / validation */
    CDEF(r_oxide_set_log_level, 1),
    CDEF(r_oxide_get_log_level, 0),
    CDEF(r_certificate_load_from_bytes, 2),
    CDEF(r_certificate_load_from_pem, 2),
    CDEF(r_certificate_get_subject, 1),
    CDEF(r_certificate_get_issuer, 1),
    CDEF(r_certificate_get_serial, 1),
    CDEF(r_certificate_get_validity, 1),
    CDEF(r_certificate_is_valid, 1),
    CDEF(r_certificate_close, 1),
    CDEF(r_sign_bytes, 4),
    CDEF(r_sign_bytes_pades, 9),
    CDEF(r_sign_bytes_pades_opts, 9),
    CDEF(r_doc_signature_count, 1),
    CDEF(r_doc_get_signature, 2),
    CDEF(r_signature_get_signer_name, 1),
    CDEF(r_signature_get_signing_reason, 1),
    CDEF(r_signature_get_signing_location, 1),
    CDEF(r_signature_get_signing_time, 1),
    CDEF(r_signature_get_certificate, 1),
    CDEF(r_signature_get_pades_level, 1),
    CDEF(r_signature_has_timestamp, 1),
    CDEF(r_signature_get_timestamp, 1),
    CDEF(r_signature_add_timestamp, 2),
    CDEF(r_signature_verify, 1),
    CDEF(r_signature_verify_detached, 2),
    CDEF(r_signature_close, 1),
    CDEF(r_timestamp_parse, 1),
    CDEF(r_timestamp_get_token, 1),
    CDEF(r_timestamp_get_message_imprint, 1),
    CDEF(r_timestamp_get_time, 1),
    CDEF(r_timestamp_get_serial, 1),
    CDEF(r_timestamp_get_tsa_name, 1),
    CDEF(r_timestamp_get_policy_oid, 1),
    CDEF(r_timestamp_get_hash_algorithm, 1),
    CDEF(r_timestamp_verify, 1),
    CDEF(r_timestamp_close, 1),
    CDEF(r_tsa_client_create, 7),
    CDEF(r_tsa_request_timestamp, 2),
    CDEF(r_tsa_request_timestamp_hash, 3),
    CDEF(r_tsa_client_close, 1),
    CDEF(r_doc_get_dss, 1),
    CDEF(r_dss_cert_count, 1),
    CDEF(r_dss_crl_count, 1),
    CDEF(r_dss_ocsp_count, 1),
    CDEF(r_dss_vri_count, 1),
    CDEF(r_dss_get_cert, 2),
    CDEF(r_dss_get_crl, 2),
    CDEF(r_dss_get_ocsp, 2),
    CDEF(r_dss_close, 1),
    CDEF(r_validate_pdf_a, 2),
    CDEF(r_validate_pdf_ua, 2),
    CDEF(r_validate_pdf_x, 2),
    CDEF(r_pdf_a_is_compliant, 1),
    CDEF(r_pdf_a_error_count, 1),
    CDEF(r_pdf_a_warning_count, 1),
    CDEF(r_pdf_a_get_error, 2),
    CDEF(r_pdf_a_results_close, 1),
    CDEF(r_pdf_ua_is_accessible, 1),
    CDEF(r_pdf_ua_error_count, 1),
    CDEF(r_pdf_ua_warning_count, 1),
    CDEF(r_pdf_ua_get_error, 2),
    CDEF(r_pdf_ua_get_warning, 2),
    CDEF(r_pdf_ua_get_stats, 1),
    CDEF(r_pdf_ua_results_close, 1),
    CDEF(r_pdf_x_is_compliant, 1),
    CDEF(r_pdf_x_error_count, 1),
    CDEF(r_pdf_x_get_error, 2),
    CDEF(r_pdf_x_results_close, 1),
    /* PHASE-7 barcodes / OCR / render variants / redaction / constructors /
     * page getters / elements / timestamp */
    CDEF(r_generate_qr_code, 3),
    CDEF(r_generate_barcode, 3),
    CDEF(r_barcode_get_data, 1),
    CDEF(r_barcode_get_format, 1),
    CDEF(r_barcode_get_confidence, 1),
    CDEF(r_barcode_get_image_png, 2),
    CDEF(r_barcode_get_svg, 2),
    CDEF(r_barcode_close, 1),
    CDEF(r_editor_add_barcode_to_page, 7),
    CDEF(r_ocr_engine_create, 3),
    CDEF(r_ocr_engine_close, 1),
    CDEF(r_ocr_page_needs_ocr, 2),
    CDEF(r_ocr_extract_text, 3),
    CDEF(r_render_page_with_options, 11),
    CDEF(r_render_page_with_options_ex, 12),
    CDEF(r_render_page_region, 7),
    CDEF(r_render_page_fit, 5),
    CDEF(r_render_page_raw, 3),
    CDEF(r_create_renderer, 4),
    CDEF(r_renderer_close, 1),
    CDEF(r_estimate_render_time, 2),
    CDEF(r_redaction_add, 9),
    CDEF(r_redaction_count, 2),
    CDEF(r_redaction_apply, 5),
    CDEF(r_redaction_scrub_metadata, 1),
    CDEF(r_pdf_from_image, 1),
    CDEF(r_pdf_from_image_bytes, 1),
    CDEF(r_pdf_from_html_css, 3),
    CDEF(r_pdf_from_html_css_with_fonts, 4),
    CDEF(r_pdf_merge, 1),
    CDEF(r_page_get_width, 2),
    CDEF(r_page_get_height, 2),
    CDEF(r_page_get_rotation, 2),
    CDEF(r_page_get_elements, 2),
    CDEF(r_add_timestamp, 3),
    /* PHASE-8: 100%-coverage closeout */
    CDEF(r_doc_open_from_docx_bytes, 1),
    CDEF(r_doc_open_from_pptx_bytes, 1),
    CDEF(r_doc_open_from_xlsx_bytes, 1),
    CDEF(r_doc_to_docx, 1),
    CDEF(r_doc_to_pptx, 1),
    CDEF(r_doc_to_xlsx, 1),
    CDEF(r_doc_extract_text_in_rect, 6),
    CDEF(r_doc_extract_words_in_rect, 6),
    CDEF(r_doc_extract_lines_in_rect, 6),
    CDEF(r_doc_extract_tables_in_rect, 6),
    CDEF(r_doc_extract_images_in_rect, 6),
    CDEF(r_doc_extract_all_text, 1),
    CDEF(r_doc_extract_text_auto, 2),
    CDEF(r_doc_extract_page_auto, 3),
    CDEF(r_doc_classify_page, 2),
    CDEF(r_doc_classify_document, 1),
    CDEF(r_doc_remove_headers, 2),
    CDEF(r_doc_remove_footers, 2),
    CDEF(r_doc_remove_artifacts, 2),
    CDEF(r_doc_erase_header, 2),
    CDEF(r_doc_erase_footer, 2),
    CDEF(r_doc_erase_artifacts, 2),
    CDEF(r_doc_get_form_fields, 1),
    CDEF(r_doc_export_form_data_to_bytes, 2),
    CDEF(r_doc_import_form_data, 2),
    CDEF(r_form_import_from_file, 2),
    CDEF(r_editor_import_fdf_bytes, 2),
    CDEF(r_editor_import_xfdf_bytes, 2),
    CDEF(r_doc_get_outline, 1),
    CDEF(r_doc_get_page_labels, 1),
    CDEF(r_doc_get_xmp_metadata, 1),
    CDEF(r_doc_get_source_bytes, 1),
    CDEF(r_doc_has_xfa, 1),
    CDEF(r_doc_plan_split_by_bookmarks, 2),
    CDEF(r_pdf_get_page_count, 1),
    CDEF(r_doc_sign, 4),
    CDEF(r_doc_verify_all_signatures, 1),
    CDEF(r_doc_has_timestamp, 1),
    CDEF(r_annotation_get_color, 3),
    CDEF(r_annotation_get_creation_date, 3),
    CDEF(r_annotation_get_modification_date, 3),
    CDEF(r_annotation_is_hidden, 3),
    CDEF(r_annotation_is_marked_deleted, 3),
    CDEF(r_annotation_is_printable, 3),
    CDEF(r_annotation_is_read_only, 3),
    CDEF(r_link_annotation_get_uri, 3),
    CDEF(r_text_annotation_get_icon_name, 3),
    CDEF(r_highlight_annotation_quad_points_count, 3),
    CDEF(r_highlight_annotation_quad_point, 4),
    CDEF(r_annotations_to_json, 2),
    CDEF(r_font_get_size, 3),
    CDEF(r_fonts_to_json, 2),
    CDEF(r_elements_to_json, 2),
    CDEF(r_search_results_to_json, 4),
    CDEF(r_crypto_active_provider, 0),
    CDEF(r_crypto_fips_available, 0),
    CDEF(r_crypto_use_fips, 0),
    CDEF(r_crypto_set_policy, 1),
    CDEF(r_crypto_policy, 0),
    CDEF(r_crypto_inventory, 0),
    CDEF(r_crypto_cbom, 0),
    CDEF(r_model_manifest, 0),
    CDEF(r_prefetch_available, 0),
    CDEF(r_prefetch_models, 1),
    CDEF(r_set_max_ops_per_stream, 1),
    CDEF(r_set_preserve_unmapped_glyphs, 1),
    CDEF(r_doc_convert_to_pdf_a, 2),
    {NULL, NULL, 0}
};

void R_init_pdfoxide(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
