/* pdf_oxide — Elixir NIF over the C ABI.
 *
 * Extraction is CPU-bound, so every text-producing NIF is scheduled on a DIRTY
 * CPU scheduler (ERL_NIF_DIRTY_JOB_CPU_BOUND) — a plain NIF would block the
 * BEAM scheduler. Document/Pdf handles are NIF resources freed by their
 * destructors; returned C strings/buffers become Elixir binaries and the C
 * buffer is freed via free_string; non-success C-ABI codes return
 * {:error, code}. */
#include <erl_nif.h>
#include <string.h>
#include <pdf_oxide_c/pdf_oxide.h>

static ErlNifResourceType *DOC_RES;
static ErlNifResourceType *PDF_RES;
static ErlNifResourceType *IMG_RES;

typedef struct { PdfDocument *h; } DocRes;
typedef struct { Pdf *h; } PdfRes;
typedef struct { FfiRenderedImage *h; } ImgRes;

static void doc_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    DocRes *r = (DocRes *)obj;
    if (r->h) { pdf_document_free(r->h); r->h = NULL; }
}
static void pdf_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    PdfRes *r = (PdfRes *)obj;
    if (r->h) { pdf_free(r->h); r->h = NULL; }
}
static void img_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    ImgRes *r = (ImgRes *)obj;
    if (r->h) { pdf_rendered_image_free(r->h); r->h = NULL; }
}

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    (void)priv; (void)info;
    int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
    DOC_RES = enif_open_resource_type(env, NULL, "pdf_oxide_doc", doc_dtor, flags, NULL);
    PDF_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf", pdf_dtor, flags, NULL);
    IMG_RES = enif_open_resource_type(env, NULL, "pdf_oxide_img", img_dtor, flags, NULL);
    return (DOC_RES && PDF_RES && IMG_RES) ? 0 : 1;
}

static ERL_NIF_TERM err_tuple(ErlNifEnv *env, int32_t code) {
    return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_int(env, code));
}

/* Read an iolist/binary argument into a NUL-terminated C string (caller frees). */
static char *term_to_cstr(ErlNifEnv *env, ERL_NIF_TERM term) {
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, term, &bin)) return NULL;
    char *s = enif_alloc(bin.size + 1);
    if (!s) return NULL;
    memcpy(s, bin.data, bin.size);
    s[bin.size] = '\0';
    return s;
}

/* Wrap a returned C string into {:ok, binary}, freeing it via free_string. */
static ERL_NIF_TERM ok_string(ErlNifEnv *env, char *s, int32_t code) {
    if (!s) return err_tuple(env, code);
    ERL_NIF_TERM bin;
    size_t n = strlen(s);
    unsigned char *buf = enif_make_new_binary(env, n, &bin);
    memcpy(buf, s, n);
    free_string(s);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), bin);
}

/* Take an owned C string into an Elixir binary term, freeing it via
 * free_string. A NULL string (e.g. an empty cell) becomes an empty binary. */
static ERL_NIF_TERM take_string(ErlNifEnv *env, char *s) {
    ERL_NIF_TERM bin;
    if (!s) {
        enif_make_new_binary(env, 0, &bin);
        return bin;
    }
    size_t n = strlen(s);
    unsigned char *buf = enif_make_new_binary(env, n, &bin);
    memcpy(buf, s, n);
    free_string(s);
    return bin;
}

/* ── builder ──────────────────────────────────────────────────────────────── */
#define BUILD_NIF(name, cfn)                                                    \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        char *in = term_to_cstr(env, a[0]);                                    \
        if (!in) return enif_make_badarg(env);                                 \
        int32_t code = 0;                                                      \
        Pdf *h = cfn(in, &code);                                               \
        enif_free(in);                                                         \
        if (!h) return err_tuple(env, code);                                   \
        PdfRes *r = enif_alloc_resource(PDF_RES, sizeof(PdfRes));              \
        r->h = h;                                                              \
        ERL_NIF_TERM term = enif_make_resource(env, r);                        \
        enif_release_resource(r);                                              \
        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);         \
    }
BUILD_NIF(from_markdown, pdf_from_markdown)
BUILD_NIF(from_html, pdf_from_html)
BUILD_NIF(from_text, pdf_from_text)

static ERL_NIF_TERM pdf_save_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfRes *r;
    if (!enif_get_resource(env, a[0], PDF_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    char *path = term_to_cstr(env, a[1]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_save(r->h, path, &code);
    enif_free(path);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pdf_save_bytes_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfRes *r;
    if (!enif_get_resource(env, a[0], PDF_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t len = 0, code = 0;
    uint8_t *p = pdf_save_to_bytes(r->h, &len, &code);
    if (!p) return err_tuple(env, code);
    size_t n = len < 0 ? 0 : (size_t)len;
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, n, &bin);
    memcpy(buf, p, n);
    free_bytes(p);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), bin);
}

/* ── document ─────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM make_doc(ErlNifEnv *env, PdfDocument *h) {
    DocRes *r = enif_alloc_resource(DOC_RES, sizeof(DocRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

static ERL_NIF_TERM doc_open(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *path = term_to_cstr(env, a[0]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    PdfDocument *h = pdf_document_open(path, &code);
    enif_free(path);
    return h ? make_doc(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM doc_open_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_bytes(bin.data, bin.size, &code);
    return h ? make_doc(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM doc_open_pw(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *path = term_to_cstr(env, a[0]);
    char *pw = term_to_cstr(env, a[1]);
    if (!path || !pw) { enif_free(path); enif_free(pw); return enif_make_badarg(env); }
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_with_password(path, pw, &code);
    enif_free(path); enif_free(pw);
    return h ? make_doc(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM doc_page_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t n = pdf_document_get_page_count(r->h, &code);
    if (n < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM doc_version(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    uint8_t maj = 0, min = 0;
    pdf_document_get_version(r->h, &maj, &min);
    return enif_make_tuple2(env, enif_make_int(env, maj), enif_make_int(env, min));
}

static ERL_NIF_TERM doc_is_encrypted(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    return enif_make_atom(env, pdf_document_is_encrypted(r->h) ? "true" : "false");
}

static ERL_NIF_TERM doc_has_tree(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    return enif_make_atom(env, pdf_document_has_structure_tree(r->h) ? "true" : "false");
}

/* page-text family — generated, all dirty CPU-bound. */
#define PAGE_NIF(name, cfn)                                                     \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        DocRes *r;                                                             \
        int page;                                                             \
        if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||             \
            !enif_get_int(env, a[1], &page))                                   \
            return enif_make_badarg(env);                                      \
        if (!r->h) return enif_make_badarg(env);                               \
        int32_t code = 0;                                                      \
        return ok_string(env, cfn(r->h, page, &code), code);                   \
    }
PAGE_NIF(doc_extract_text, pdf_document_extract_text)
PAGE_NIF(doc_to_plain_text, pdf_document_to_plain_text)
PAGE_NIF(doc_to_markdown, pdf_document_to_markdown)
PAGE_NIF(doc_to_html, pdf_document_to_html)
PAGE_NIF(doc_struct_json, pdf_document_extract_structured_to_json)

static ERL_NIF_TERM doc_to_markdown_all(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_document_to_markdown_all(r->h, &code), code);
}

static ERL_NIF_TERM doc_to_html_all(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_document_to_html_all(r->h, &code), code);
}

static ERL_NIF_TERM doc_to_plain_text_all(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_document_to_plain_text_all(r->h, &code), code);
}

/* authenticate returns a plain bool: false is a legitimate "wrong password"
 * outcome, not a failure, so always return {:ok, bool}. */
static ERL_NIF_TERM doc_authenticate(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    char *pw = term_to_cstr(env, a[1]);
    if (!pw) return enif_make_badarg(env);
    int32_t code = 0;
    bool ok = pdf_document_authenticate(r->h, pw, &code);
    enif_free(pw);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, ok ? "true" : "false"));
}

/* ── element extraction (phase 1) ───────────────────────────────────────────
 * Each extractor returns a NULL list on error; the list owns its elements and
 * is freed once via pdf_oxide_X_list_free after every element has been read.
 * Owned char* fields are copied into binaries and freed via free_string. All
 * are dirty CPU-bound (extraction parses page content). */

/* Read the doc resource + page index from args[0]/args[1]. */
#define GET_DOC_PAGE                                                            \
    DocRes *r;                                                                  \
    int page;                                                                   \
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||                  \
        !enif_get_int(env, a[1], &page))                                        \
        return enif_make_badarg(env);                                           \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM doc_extract_chars(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiCharList *list = pdf_document_extract_chars(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_char_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        uint32_t cp = pdf_oxide_char_get_char(list, i, &c);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_char_get_bbox(list, i, &x, &y, &w, &h, &c);
        ERL_NIF_TERM font = take_string(env, pdf_oxide_char_get_font_name(list, i, &c));
        float size = pdf_oxide_char_get_font_size(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple7(env, enif_make_uint(env, cp),
                                             enif_make_double(env, x), enif_make_double(env, y),
                                             enif_make_double(env, w), enif_make_double(env, h),
                                             font, enif_make_double(env, size));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_char_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_extract_words(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiWordList *list = pdf_document_extract_words(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_word_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM text = take_string(env, pdf_oxide_word_get_text(list, i, &c));
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_word_get_bbox(list, i, &x, &y, &w, &h, &c);
        ERL_NIF_TERM font = take_string(env, pdf_oxide_word_get_font_name(list, i, &c));
        float size = pdf_oxide_word_get_font_size(list, i, &c);
        bool bold = pdf_oxide_word_is_bold(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple(env, 8, text,
                                            enif_make_double(env, x), enif_make_double(env, y),
                                            enif_make_double(env, w), enif_make_double(env, h),
                                            font, enif_make_double(env, size),
                                            enif_make_atom(env, bold ? "true" : "false"));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_word_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_extract_text_lines(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiTextLineList *list = pdf_document_extract_text_lines(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_line_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM text = take_string(env, pdf_oxide_line_get_text(list, i, &c));
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_line_get_bbox(list, i, &x, &y, &w, &h, &c);
        int32_t wc = pdf_oxide_line_get_word_count(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple6(env, text,
                                             enif_make_double(env, x), enif_make_double(env, y),
                                             enif_make_double(env, w), enif_make_double(env, h),
                                             enif_make_int(env, wc));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_line_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_extract_tables(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiTableList *list = pdf_document_extract_tables(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_table_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM tables = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        int32_t rows = pdf_oxide_table_get_row_count(list, i, &c);
        int32_t cols = pdf_oxide_table_get_col_count(list, i, &c);
        bool header = pdf_oxide_table_has_header(list, i, &c);
        int32_t rr = rows < 0 ? 0 : rows;
        int32_t cc = cols < 0 ? 0 : cols;
        ERL_NIF_TERM grid = enif_make_list(env, 0);
        for (int32_t row = rr - 1; row >= 0; row--) {
            ERL_NIF_TERM line = enif_make_list(env, 0);
            for (int32_t col = cc - 1; col >= 0; col--) {
                ERL_NIF_TERM cell = take_string(env, pdf_oxide_table_get_cell_text(list, i, row, col, &c));
                line = enif_make_list_cell(env, cell, line);
            }
            grid = enif_make_list_cell(env, line, grid);
        }
        ERL_NIF_TERM item = enif_make_tuple4(env, enif_make_int(env, rr),
                                             enif_make_int(env, cc),
                                             enif_make_atom(env, header ? "true" : "false"),
                                             grid);
        tables = enif_make_list_cell(env, item, tables);
    }
    pdf_oxide_table_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), tables);
}

/* ── element extraction (phase 2) ───────────────────────────────────────────
 * Same shape as phase 1: each extractor returns a NULL list on error; the list
 * owns its elements and is freed once via its *_free after every element has
 * been read. Owned char* fields are copied into binaries and freed via
 * free_string; image bytes are freed via free_bytes. All are dirty CPU-bound. */

static ERL_NIF_TERM doc_embedded_fonts(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiFontList *list = pdf_document_get_embedded_fonts(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_font_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM name = take_string(env, pdf_oxide_font_get_name(list, i, &c));
        ERL_NIF_TERM type = take_string(env, pdf_oxide_font_get_type(list, i, &c));
        ERL_NIF_TERM enc = take_string(env, pdf_oxide_font_get_encoding(list, i, &c));
        bool embedded = pdf_oxide_font_is_embedded(list, i, &c) != 0;
        bool subset = pdf_oxide_font_is_subset(list, i, &c) != 0;
        ERL_NIF_TERM item = enif_make_tuple5(env, name, type, enc,
                                             enif_make_atom(env, embedded ? "true" : "false"),
                                             enif_make_atom(env, subset ? "true" : "false"));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_font_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_embedded_images(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiImageList *list = pdf_document_get_embedded_images(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_image_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        int32_t w = pdf_oxide_image_get_width(list, i, &c);
        int32_t h = pdf_oxide_image_get_height(list, i, &c);
        int32_t bpc = pdf_oxide_image_get_bits_per_component(list, i, &c);
        ERL_NIF_TERM format = take_string(env, pdf_oxide_image_get_format(list, i, &c));
        ERL_NIF_TERM colorspace = take_string(env, pdf_oxide_image_get_colorspace(list, i, &c));
        int32_t dlen = 0;
        uint8_t *p = pdf_oxide_image_get_data(list, i, &dlen, &c);
        size_t dn = (p && dlen > 0) ? (size_t)dlen : 0;
        ERL_NIF_TERM data;
        unsigned char *buf = enif_make_new_binary(env, dn, &data);
        if (dn) memcpy(buf, p, dn);
        if (p) free_bytes(p);
        ERL_NIF_TERM item = enif_make_tuple6(env, enif_make_int(env, w),
                                             enif_make_int(env, h),
                                             enif_make_int(env, bpc),
                                             format, colorspace, data);
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_image_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_page_annotations(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_annotation_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM type = take_string(env, pdf_oxide_annotation_get_type(list, i, &c));
        ERL_NIF_TERM subtype = take_string(env, pdf_oxide_annotation_get_subtype(list, i, &c));
        ERL_NIF_TERM content = take_string(env, pdf_oxide_annotation_get_content(list, i, &c));
        ERL_NIF_TERM author = take_string(env, pdf_oxide_annotation_get_author(list, i, &c));
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_annotation_get_rect(list, i, &x, &y, &w, &h, &c);
        float bw = pdf_oxide_annotation_get_border_width(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple(env, 9, type, subtype, content, author,
                                            enif_make_double(env, x), enif_make_double(env, y),
                                            enif_make_double(env, w), enif_make_double(env, h),
                                            enif_make_double(env, bw));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_annotation_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_extract_paths(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiPathList *list = pdf_document_extract_paths(r->h, page, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_path_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_path_get_bbox(list, i, &x, &y, &w, &h, &c);
        float sw = pdf_oxide_path_get_stroke_width(list, i, &c);
        bool stroke = pdf_oxide_path_has_stroke(list, i, &c);
        bool fill = pdf_oxide_path_has_fill(list, i, &c);
        int32_t ops = pdf_oxide_path_get_operation_count(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple(env, 8,
                                            enif_make_double(env, x), enif_make_double(env, y),
                                            enif_make_double(env, w), enif_make_double(env, h),
                                            enif_make_double(env, sw),
                                            enif_make_atom(env, stroke ? "true" : "false"),
                                            enif_make_atom(env, fill ? "true" : "false"),
                                            enif_make_int(env, ops));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_path_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

/* Build the {:ok, [search-result tuples]} term from an owned results handle and
 * free it via pdf_oxide_search_result_free (NOT _list_free). */
static ERL_NIF_TERM search_results_term(ErlNifEnv *env, FfiSearchResults *list) {
    int32_t n = pdf_oxide_search_result_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM text = take_string(env, pdf_oxide_search_result_get_text(list, i, &c));
        int32_t pg = pdf_oxide_search_result_get_page(list, i, &c);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_search_result_get_bbox(list, i, &x, &y, &w, &h, &c);
        ERL_NIF_TERM item = enif_make_tuple6(env, text, enif_make_int(env, pg),
                                             enif_make_double(env, x), enif_make_double(env, y),
                                             enif_make_double(env, w), enif_make_double(env, h));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_search_result_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

static ERL_NIF_TERM doc_search_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    char *term = term_to_cstr(env, a[2]);
    if (!term) return enif_make_badarg(env);
    bool case_sensitive = enif_is_identical(a[3], enif_make_atom(env, "true"));
    int32_t code = 0;
    FfiSearchResults *list = pdf_document_search_page(r->h, page, term, case_sensitive, &code);
    enif_free(term);
    if (!list) return err_tuple(env, code);
    return search_results_term(env, list);
}

static ERL_NIF_TERM doc_search_all(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    char *term = term_to_cstr(env, a[1]);
    if (!term) return enif_make_badarg(env);
    bool case_sensitive = enif_is_identical(a[2], enif_make_atom(env, "true"));
    int32_t code = 0;
    FfiSearchResults *list = pdf_document_search_all(r->h, term, case_sensitive, &code);
    enif_free(term);
    if (!list) return err_tuple(env, code);
    return search_results_term(env, list);
}

/* ── page rendering (phase 3) ────────────────────────────────────────────────
 * Each render returns an FfiRenderedImage handle (NULL on error). The handle is
 * wrapped in an IMG_RES resource whose destructor frees it via
 * pdf_rendered_image_free; the live handle is kept so save/3 can call
 * pdf_save_rendered_image. width/height/data are read once into the returned
 * tuple; data bytes are copied into a binary and freed via free_bytes. All are
 * dirty CPU-bound (rendering rasterises the page). */

/* Read width/height/data from a rendered-image handle and return
 * {:ok, {ref, width, height, data}}, keeping the handle live in IMG_RES. */
static ERL_NIF_TERM make_rendered_image(ErlNifEnv *env, FfiRenderedImage *h) {
    int32_t c = 0;
    int32_t w = pdf_get_rendered_image_width(h, &c);
    int32_t hgt = pdf_get_rendered_image_height(h, &c);
    int32_t dlen = 0;
    uint8_t *p = pdf_get_rendered_image_data(h, &dlen, &c);
    size_t dn = (p && dlen > 0) ? (size_t)dlen : 0;
    ERL_NIF_TERM data;
    unsigned char *buf = enif_make_new_binary(env, dn, &data);
    if (dn) memcpy(buf, p, dn);
    if (p) free_bytes(p);
    ImgRes *r = enif_alloc_resource(IMG_RES, sizeof(ImgRes));
    r->h = h;
    ERL_NIF_TERM ref = enif_make_resource(env, r);
    enif_release_resource(r);
    ERL_NIF_TERM tuple = enif_make_tuple4(env, ref, enif_make_int(env, w < 0 ? 0 : w),
                                          enif_make_int(env, hgt < 0 ? 0 : hgt), data);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), tuple);
}

static ERL_NIF_TERM doc_render_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, format;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &format))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page(r->h, page_index, format, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

static ERL_NIF_TERM doc_render_page_zoom(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, format;
    double zoom;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_double(env, a[2], &zoom) ||
        !enif_get_int(env, a[3], &format))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page_zoom(r->h, page_index, (float)zoom, format, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

static ERL_NIF_TERM doc_render_page_thumbnail(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, size, format;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &size) ||
        !enif_get_int(env, a[3], &format))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page_thumbnail(r->h, page_index, size, format, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

/* Save a rendered image to a file path via the live handle. Returns :ok or
 * {:error, code}. */
static ERL_NIF_TERM img_save(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ImgRes *r;
    if (!enif_get_resource(env, a[0], IMG_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    char *path = term_to_cstr(env, a[1]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_save_rendered_image(r->h, path, &code);
    enif_free(path);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

/* Explicit, idempotent close: free the native handle now and null it so the GC
 * destructor is a no-op and later use raises (badarg). */
static ERL_NIF_TERM doc_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_document_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM pdf_close_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfRes *r;
    if (!enif_get_resource(env, a[0], PDF_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

#define DIRTY ERL_NIF_DIRTY_JOB_CPU_BOUND
static ErlNifFunc funcs[] = {
    {"from_markdown", 1, from_markdown, DIRTY},
    {"from_html", 1, from_html, DIRTY},
    {"from_text", 1, from_text, DIRTY},
    {"pdf_save", 2, pdf_save_nif, DIRTY},
    {"pdf_save_to_bytes", 1, pdf_save_bytes_nif, DIRTY},
    {"doc_open", 1, doc_open, DIRTY},
    {"doc_open_bytes", 1, doc_open_bytes, DIRTY},
    {"doc_open_pw", 2, doc_open_pw, DIRTY},
    {"doc_page_count", 1, doc_page_count, 0},
    {"doc_version", 1, doc_version, 0},
    {"doc_is_encrypted", 1, doc_is_encrypted, 0},
    {"doc_has_structure_tree", 1, doc_has_tree, 0},
    {"doc_extract_text", 2, doc_extract_text, DIRTY},
    {"doc_to_plain_text", 2, doc_to_plain_text, DIRTY},
    {"doc_to_markdown", 2, doc_to_markdown, DIRTY},
    {"doc_to_html", 2, doc_to_html, DIRTY},
    {"doc_to_markdown_all", 1, doc_to_markdown_all, DIRTY},
    {"doc_to_html_all", 1, doc_to_html_all, DIRTY},
    {"doc_to_plain_text_all", 1, doc_to_plain_text_all, DIRTY},
    {"doc_authenticate", 2, doc_authenticate, DIRTY},
    {"doc_extract_structured_json", 2, doc_struct_json, DIRTY},
    {"doc_extract_chars", 2, doc_extract_chars, DIRTY},
    {"doc_extract_words", 2, doc_extract_words, DIRTY},
    {"doc_extract_text_lines", 2, doc_extract_text_lines, DIRTY},
    {"doc_extract_tables", 2, doc_extract_tables, DIRTY},
    {"doc_embedded_fonts", 2, doc_embedded_fonts, DIRTY},
    {"doc_embedded_images", 2, doc_embedded_images, DIRTY},
    {"doc_page_annotations", 2, doc_page_annotations, DIRTY},
    {"doc_extract_paths", 2, doc_extract_paths, DIRTY},
    {"doc_search_page", 4, doc_search_page, DIRTY},
    {"doc_search_all", 3, doc_search_all, DIRTY},
    {"doc_render_page", 3, doc_render_page, DIRTY},
    {"doc_render_page_zoom", 4, doc_render_page_zoom, DIRTY},
    {"doc_render_page_thumbnail", 4, doc_render_page_thumbnail, DIRTY},
    {"img_save", 2, img_save, DIRTY},
    {"doc_close", 1, doc_close, 0},
    {"pdf_close", 1, pdf_close_nif, 0},
};

ERL_NIF_INIT(Elixir.PdfOxide.Native, funcs, load, NULL, NULL, NULL)
