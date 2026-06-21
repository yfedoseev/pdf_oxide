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
static ErlNifResourceType *EDIT_RES;
static ErlNifResourceType *DBLD_RES;
static ErlNifResourceType *PBLD_RES;
static ErlNifResourceType *FONT_RES;
/* phase 6 — digital signatures / PKI / timestamps / TSA / validation. The
 * native handles are opaque (void* / FfiSignatureInfo* / FfiPdf{A,X}Results* /
 * FfiUaResults*); each is wrapped in its own resource type whose dtor frees it
 * via the matching pdf_*_free entry point, and a closed handle (NULL) raises
 * badarg. */
static ErlNifResourceType *CERT_RES;
static ErlNifResourceType *SIG_RES;
static ErlNifResourceType *TS_RES;
static ErlNifResourceType *TSA_RES;
static ErlNifResourceType *DSS_RES;
static ErlNifResourceType *PDFA_RES;
static ErlNifResourceType *PDFUA_RES;
static ErlNifResourceType *PDFX_RES;
/* phase 7 — barcodes/QR, OCR engine, renderer, element list. Each opaque
 * native handle (FfiBarcodeImage* / void* OCR engine / void* renderer /
 * FfiElementList*) is wrapped in its own resource type whose dtor frees it via
 * its matching pdf_*_free entry point; a closed handle (NULL) raises badarg. */
static ErlNifResourceType *BARCODE_RES;
static ErlNifResourceType *OCR_RES;
static ErlNifResourceType *RENDERER_RES;
static ErlNifResourceType *ELEMS_RES;

typedef struct { PdfDocument *h; } DocRes;
typedef struct { Pdf *h; } PdfRes;
typedef struct { FfiRenderedImage *h; } ImgRes;
typedef struct { DocumentEditor *h; } EditRes;
typedef struct { FfiDocumentBuilder *h; } DBldRes;
typedef struct { FfiPageBuilder *h; } PBldRes;
typedef struct { EmbeddedFont *h; } FontRes;
typedef struct { void *h; } CertRes;
typedef struct { FfiSignatureInfo *h; } SigRes;
typedef struct { void *h; } TsRes;
typedef struct { void *h; } TsaRes;
typedef struct { void *h; } DssRes;
typedef struct { FfiPdfAResults *h; } PdfARes;
typedef struct { FfiUaResults *h; } PdfUaRes;
typedef struct { FfiPdfXResults *h; } PdfXRes;
typedef struct { FfiBarcodeImage *h; } BarcodeRes;
typedef struct { void *h; } OcrRes;
typedef struct { void *h; } RendererRes;
typedef struct { FfiElementList *h; } ElemsRes;

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
static void edit_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    EditRes *r = (EditRes *)obj;
    if (r->h) { document_editor_free(r->h); r->h = NULL; }
}
static void dbld_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    DBldRes *r = (DBldRes *)obj;
    if (r->h) { pdf_document_builder_free(r->h); r->h = NULL; }
}
static void pbld_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    PBldRes *r = (PBldRes *)obj;
    if (r->h) { pdf_page_builder_free(r->h); r->h = NULL; }
}
static void font_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    FontRes *r = (FontRes *)obj;
    if (r->h) { pdf_embedded_font_free(r->h); r->h = NULL; }
}
static void cert_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    CertRes *r = (CertRes *)obj;
    if (r->h) { pdf_certificate_free(r->h); r->h = NULL; }
}
static void sig_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    SigRes *r = (SigRes *)obj;
    if (r->h) { pdf_signature_free(r->h); r->h = NULL; }
}
static void ts_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    TsRes *r = (TsRes *)obj;
    if (r->h) { pdf_timestamp_free(r->h); r->h = NULL; }
}
static void tsa_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    TsaRes *r = (TsaRes *)obj;
    if (r->h) { pdf_tsa_client_free(r->h); r->h = NULL; }
}
static void dss_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    DssRes *r = (DssRes *)obj;
    if (r->h) { pdf_dss_free(r->h); r->h = NULL; }
}
static void pdfa_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    PdfARes *r = (PdfARes *)obj;
    if (r->h) { pdf_pdf_a_results_free(r->h); r->h = NULL; }
}
static void pdfua_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    PdfUaRes *r = (PdfUaRes *)obj;
    if (r->h) { pdf_pdf_ua_results_free(r->h); r->h = NULL; }
}
static void pdfx_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    PdfXRes *r = (PdfXRes *)obj;
    if (r->h) { pdf_pdf_x_results_free(r->h); r->h = NULL; }
}
static void barcode_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    BarcodeRes *r = (BarcodeRes *)obj;
    if (r->h) { pdf_barcode_free(r->h); r->h = NULL; }
}
static void ocr_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    OcrRes *r = (OcrRes *)obj;
    if (r->h) { pdf_ocr_engine_free(r->h); r->h = NULL; }
}
static void renderer_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    RendererRes *r = (RendererRes *)obj;
    if (r->h) { pdf_renderer_free(r->h); r->h = NULL; }
}
static void elems_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    ElemsRes *r = (ElemsRes *)obj;
    if (r->h) { pdf_oxide_elements_free(r->h); r->h = NULL; }
}

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    (void)priv; (void)info;
    int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
    DOC_RES = enif_open_resource_type(env, NULL, "pdf_oxide_doc", doc_dtor, flags, NULL);
    PDF_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf", pdf_dtor, flags, NULL);
    IMG_RES = enif_open_resource_type(env, NULL, "pdf_oxide_img", img_dtor, flags, NULL);
    EDIT_RES = enif_open_resource_type(env, NULL, "pdf_oxide_editor", edit_dtor, flags, NULL);
    DBLD_RES = enif_open_resource_type(env, NULL, "pdf_oxide_doc_builder", dbld_dtor, flags, NULL);
    PBLD_RES = enif_open_resource_type(env, NULL, "pdf_oxide_page_builder", pbld_dtor, flags, NULL);
    FONT_RES = enif_open_resource_type(env, NULL, "pdf_oxide_embedded_font", font_dtor, flags, NULL);
    CERT_RES = enif_open_resource_type(env, NULL, "pdf_oxide_certificate", cert_dtor, flags, NULL);
    SIG_RES = enif_open_resource_type(env, NULL, "pdf_oxide_signature", sig_dtor, flags, NULL);
    TS_RES = enif_open_resource_type(env, NULL, "pdf_oxide_timestamp", ts_dtor, flags, NULL);
    TSA_RES = enif_open_resource_type(env, NULL, "pdf_oxide_tsa_client", tsa_dtor, flags, NULL);
    DSS_RES = enif_open_resource_type(env, NULL, "pdf_oxide_dss", dss_dtor, flags, NULL);
    PDFA_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf_a_results", pdfa_dtor, flags, NULL);
    PDFUA_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf_ua_results", pdfua_dtor, flags, NULL);
    PDFX_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf_x_results", pdfx_dtor, flags, NULL);
    BARCODE_RES = enif_open_resource_type(env, NULL, "pdf_oxide_barcode", barcode_dtor, flags, NULL);
    OCR_RES = enif_open_resource_type(env, NULL, "pdf_oxide_ocr_engine", ocr_dtor, flags, NULL);
    RENDERER_RES = enif_open_resource_type(env, NULL, "pdf_oxide_renderer", renderer_dtor, flags, NULL);
    ELEMS_RES = enif_open_resource_type(env, NULL, "pdf_oxide_element_list", elems_dtor, flags, NULL);
    return (DOC_RES && PDF_RES && IMG_RES && EDIT_RES && DBLD_RES && PBLD_RES && FONT_RES &&
            CERT_RES && SIG_RES && TS_RES && TSA_RES && DSS_RES &&
            PDFA_RES && PDFUA_RES && PDFX_RES &&
            BARCODE_RES && OCR_RES && RENDERER_RES && ELEMS_RES) ? 0 : 1;
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

/* ── document editor ──────────────────────────────────────────────────────────
 * A DocumentEditor handle wraps an owned native editor, freed by the GC dtor
 * (edit_dtor) or eagerly by editor_close/1. Every accessor first reads the
 * EDIT_RES resource and NULL-guards r->h (a closed handle raises badarg). int32
 * C return values are status codes: a non-zero status OR a set error_code is a
 * failure → {:error, code}; bool-ish is_* queries return {:ok, bool}. String
 * returns become binaries (free_string); byte returns become binaries
 * (free_bytes). All are dirty CPU-bound (editing parses/rewrites the file). */

/* Read the editor resource into r and NULL-guard its handle. */
#define GET_EDIT                                                                 \
    EditRes *r;                                                                  \
    if (!enif_get_resource(env, a[0], EDIT_RES, (void **)&r))                    \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM make_editor(ErlNifEnv *env, DocumentEditor *h) {
    EditRes *r = enif_alloc_resource(EDIT_RES, sizeof(EditRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

/* Wrap an owned byte buffer into {:ok, binary}, freeing it via free_bytes. */
static ERL_NIF_TERM ok_bytes(ErlNifEnv *env, uint8_t *p, size_t n, int32_t code) {
    if (!p) return err_tuple(env, code);
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, n, &bin);
    if (n) memcpy(buf, p, n);
    free_bytes(p);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), bin);
}

static ERL_NIF_TERM editor_open(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *path = term_to_cstr(env, a[0]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    DocumentEditor *h = document_editor_open(path, &code);
    enif_free(path);
    return h ? make_editor(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM editor_open_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    DocumentEditor *h = document_editor_open_from_bytes(bin.data, bin.size, &code);
    return h ? make_editor(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM editor_is_modified(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    return enif_make_atom(env, document_editor_is_modified(r->h) ? "true" : "false");
}

static ERL_NIF_TERM editor_source_path(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    return ok_string(env, document_editor_get_source_path(r->h, &code), code);
}

static ERL_NIF_TERM editor_version(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    uint8_t maj = 0, min = 0;
    document_editor_get_version(r->h, &maj, &min);
    return enif_make_tuple2(env, enif_make_int(env, maj), enif_make_int(env, min));
}

static ERL_NIF_TERM editor_page_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    int32_t n = document_editor_get_page_count(r->h, &code);
    if (n < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM editor_get_producer(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    return ok_string(env, document_editor_get_producer(r->h, &code), code);
}

static ERL_NIF_TERM editor_set_producer(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *value = term_to_cstr(env, a[1]);
    if (!value) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_set_producer(r->h, value, &code);
    enif_free(value);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_get_creation_date(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    return ok_string(env, document_editor_get_creation_date(r->h, &code), code);
}

static ERL_NIF_TERM editor_set_creation_date(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *date = term_to_cstr(env, a[1]);
    if (!date) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_set_creation_date(r->h, date, &code);
    enif_free(date);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_save(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *path = term_to_cstr(env, a[1]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_save(r->h, path, &code);
    enif_free(path);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_save_to_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = document_editor_save_to_bytes(r->h, &len, &code);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM editor_save_to_bytes_with_options(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    bool compress = enif_is_identical(a[1], enif_make_atom(env, "true"));
    bool gc = enif_is_identical(a[2], enif_make_atom(env, "true"));
    bool linearize = enif_is_identical(a[3], enif_make_atom(env, "true"));
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = document_editor_save_to_bytes_with_options(r->h, compress, gc, linearize, &len, &code);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM editor_extract_pages_to_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned count = 0;
    if (!enif_get_list_length(env, a[1], &count)) return enif_make_badarg(env);
    int32_t *pages = NULL;
    if (count > 0) {
        pages = enif_alloc(count * sizeof(int32_t));
        if (!pages) return enif_make_badarg(env);
        ERL_NIF_TERM list = a[1], head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            int v;
            if (!enif_get_int(env, head, &v)) { enif_free(pages); return enif_make_badarg(env); }
            pages[i++] = (int32_t)v;
        }
    }
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = document_editor_extract_pages_to_bytes(r->h, pages, (uintptr_t)count, &len, &code);
    if (pages) enif_free(pages);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM editor_convert_to_pdf_a(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int level;
    if (!enif_get_int(env, a[1], &level)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_convert_to_pdf_a(r->h, level, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_save_encrypted_to_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *user_pw = term_to_cstr(env, a[1]);
    char *owner_pw = term_to_cstr(env, a[2]);
    if (!user_pw || !owner_pw) { enif_free(user_pw); enif_free(owner_pw); return enif_make_badarg(env); }
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = document_editor_save_encrypted_to_bytes(r->h, user_pw, owner_pw, &len, &code);
    enif_free(user_pw); enif_free(owner_pw);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM editor_merge_from_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[1], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_merge_from_bytes(r->h, bin.data, bin.size, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_embed_file(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[2], &bin)) { enif_free(name); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = document_editor_embed_file(r->h, name, bin.data, bin.size, &code);
    enif_free(name);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_apply_page_redactions(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_apply_page_redactions(r->h, (uintptr_t)page, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_apply_all_redactions(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    int rc = document_editor_apply_all_redactions(r->h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_rotate_all_pages(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int degrees;
    if (!enif_get_int(env, a[1], &degrees)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_rotate_all_pages(r->h, degrees, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_rotate_page_by(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    int degrees;
    if (!enif_get_ulong(env, a[1], &page) || !enif_get_int(env, a[2], &degrees))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_rotate_page_by(r->h, (uintptr_t)page, degrees, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

/* Build {:ok, {x, y, w, h}} from the four out-params (crop/media box). */
static ERL_NIF_TERM box_getter(ErlNifEnv *env, EditRes *r, unsigned long page,
                               int32_t (*cfn)(DocumentEditor *, uintptr_t, double *,
                                              double *, double *, double *, int32_t *)) {
    double x = 0, y = 0, w = 0, h = 0;
    int32_t code = 0;
    int rc = cfn(r->h, (uintptr_t)page, &x, &y, &w, &h, &code);
    if (rc != 0) return err_tuple(env, code);
    ERL_NIF_TERM box = enif_make_tuple4(env, enif_make_double(env, x), enif_make_double(env, y),
                                        enif_make_double(env, w), enif_make_double(env, h));
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), box);
}

static ERL_NIF_TERM editor_get_media_box(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    return box_getter(env, r, page, document_editor_get_page_media_box);
}

static ERL_NIF_TERM editor_get_crop_box(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    return box_getter(env, r, page, document_editor_get_page_crop_box);
}

/* Set a crop/media box from (page, x, y, w, h). */
static ERL_NIF_TERM box_setter(ErlNifEnv *env, const ERL_NIF_TERM a[], EditRes *r,
                               int32_t (*cfn)(DocumentEditor *, uintptr_t, double,
                                              double, double, double, int32_t *)) {
    unsigned long page;
    double x, y, w, h;
    if (!enif_get_ulong(env, a[1], &page) ||
        !enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = cfn(r->h, (uintptr_t)page, x, y, w, h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_set_media_box(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    return box_setter(env, a, r, document_editor_set_page_media_box);
}

static ERL_NIF_TERM editor_set_crop_box(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    return box_setter(env, a, r, document_editor_set_page_crop_box);
}

static ERL_NIF_TERM editor_erase_regions(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    /* a[2] is a list of {x, y, w, h} tuples flattened into doubles. */
    unsigned count = 0;
    if (!enif_get_list_length(env, a[2], &count)) return enif_make_badarg(env);
    double *rects = NULL;
    if (count > 0) {
        rects = enif_alloc(count * 4 * sizeof(double));
        if (!rects) return enif_make_badarg(env);
        ERL_NIF_TERM list = a[2], head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            int arity;
            const ERL_NIF_TERM *quad;
            if (!enif_get_tuple(env, head, &arity, &quad) || arity != 4) {
                enif_free(rects); return enif_make_badarg(env);
            }
            for (int j = 0; j < 4; j++) {
                if (!enif_get_double(env, quad[j], &rects[i * 4 + j])) {
                    enif_free(rects); return enif_make_badarg(env);
                }
            }
            i++;
        }
    }
    int32_t code = 0;
    int rc = document_editor_erase_regions(r->h, (uintptr_t)page, rects, (uintptr_t)count, &code);
    if (rects) enif_free(rects);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_clear_erase_regions(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_clear_erase_regions(r->h, (uintptr_t)page, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_is_marked_for_flatten(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t v = document_editor_is_page_marked_for_flatten(r->h, (uintptr_t)page);
    if (v < 0) return err_tuple(env, v);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, v == 1 ? "true" : "false"));
}

static ERL_NIF_TERM editor_unmark_for_flatten(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_unmark_page_for_flatten(r->h, (uintptr_t)page, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_is_marked_for_redaction(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t v = document_editor_is_page_marked_for_redaction(r->h, (uintptr_t)page);
    if (v < 0) return err_tuple(env, v);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, v == 1 ? "true" : "false"));
}

static ERL_NIF_TERM editor_unmark_for_redaction(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    unsigned long page;
    if (!enif_get_ulong(env, a[1], &page)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_unmark_page_for_redaction(r->h, (uintptr_t)page, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_delete_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index;
    if (!enif_get_int(env, a[1], &page_index)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_delete_page(r->h, page_index, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_move_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int from, to;
    if (!enif_get_int(env, a[1], &from) || !enif_get_int(env, a[2], &to))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_move_page(r->h, from, to, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_get_page_rotation(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index;
    if (!enif_get_int(env, a[1], &page_index)) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t deg = document_editor_get_page_rotation(r->h, page_index, &code);
    if (deg < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, deg));
}

static ERL_NIF_TERM editor_set_page_rotation(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index, degrees;
    if (!enif_get_int(env, a[1], &page_index) || !enif_get_int(env, a[2], &degrees))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_set_page_rotation(r->h, page_index, degrees, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_erase_region(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index;
    double x, y, w, h;
    if (!enif_get_int(env, a[1], &page_index) ||
        !enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_erase_region(r->h, page_index, (float)x, (float)y, (float)w, (float)h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_flatten_annotations(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index;
    if (!enif_get_int(env, a[1], &page_index)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_flatten_annotations(r->h, page_index, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_flatten_all_annotations(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    int rc = document_editor_flatten_all_annotations(r->h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_crop_margins(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    double left, right, top, bottom;
    if (!enif_get_double(env, a[1], &left) || !enif_get_double(env, a[2], &right) ||
        !enif_get_double(env, a[3], &top) || !enif_get_double(env, a[4], &bottom))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_crop_margins(r->h, (float)left, (float)right, (float)top, (float)bottom, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_merge_from(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *source = term_to_cstr(env, a[1]);
    if (!source) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_merge_from(r->h, source, &code);
    enif_free(source);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_save_encrypted(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *path = term_to_cstr(env, a[1]);
    char *user_pw = term_to_cstr(env, a[2]);
    char *owner_pw = term_to_cstr(env, a[3]);
    if (!path || !user_pw || !owner_pw) {
        enif_free(path); enif_free(user_pw); enif_free(owner_pw);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = document_editor_save_encrypted(r->h, path, user_pw, owner_pw, &code);
    enif_free(path); enif_free(user_pw); enif_free(owner_pw);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_set_form_field_value(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    char *name = term_to_cstr(env, a[1]);
    char *value = term_to_cstr(env, a[2]);
    if (!name || !value) { enif_free(name); enif_free(value); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = document_editor_set_form_field_value(r->h, name, value, &code);
    enif_free(name); enif_free(value);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_flatten_forms(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    int rc = document_editor_flatten_forms(r->h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_flatten_forms_on_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page_index;
    if (!enif_get_int(env, a[1], &page_index)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = document_editor_flatten_forms_on_page(r->h, page_index, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM editor_flatten_warnings_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t n = document_editor_flatten_warnings_count(r->h);
    if (n < 0) return err_tuple(env, n);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM editor_flatten_warning(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, document_editor_flatten_warning(r->h, index, &code), code);
}

/* Explicit, idempotent close: free the native handle now and null it so the GC
 * destructor is a no-op and later use raises (badarg). */
static ERL_NIF_TERM editor_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    EditRes *r;
    if (!enif_get_resource(env, a[0], EDIT_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { document_editor_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── PDF creation builder ──────────────────────────────────────────────────────
 * Three owned handle types, each freed by its GC dtor or eagerly by its
 * *_close/1: a DocumentBuilder (DBLD_RES), a PageBuilder (PBLD_RES) started off a
 * document builder, and an EmbeddedFont (FONT_RES). int32 C returns are status
 * codes (0 = ok; non-zero or a set error_code → {:error, code}); byte returns
 * (build/encrypted bytes) use a uintptr out-len and free_bytes. C string-array
 * args (const char* const*) and float/int32 array args are marshalled from
 * Elixir lists. A successful register_embedded_font CONSUMES the font handle —
 * the wrapper is nulled so neither the dtor nor a later close double-frees it.
 * pdf_page_builder_done likewise consumes the page handle (null on success).
 * All are dirty CPU-bound (building lays out + serialises a PDF). */

/* Read the document-builder resource into r and NULL-guard its handle. */
#define GET_DBLD                                                                  \
    DBldRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], DBLD_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

/* Read the page-builder resource into r and NULL-guard its handle. */
#define GET_PBLD                                                                  \
    PBldRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], PBLD_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

/* Marshal an Elixir list of binaries into a heap array of owned C strings.
 * On success *out points to count cstrings (caller frees each + the array via
 * free_cstr_array); returns 1. Returns 0 (badarg) on any failure. */
static int list_to_cstr_array(ErlNifEnv *env, ERL_NIF_TERM list_term,
                              char ***out, unsigned *out_count) {
    unsigned count = 0;
    if (!enif_get_list_length(env, list_term, &count)) return 0;
    char **arr = NULL;
    if (count > 0) {
        arr = enif_alloc(count * sizeof(char *));
        if (!arr) return 0;
        ERL_NIF_TERM list = list_term, head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            char *s = term_to_cstr(env, head);
            if (!s) {
                for (unsigned k = 0; k < i; k++) enif_free(arr[k]);
                enif_free(arr);
                return 0;
            }
            arr[i++] = s;
        }
    }
    *out = arr;
    *out_count = count;
    return 1;
}

static void free_cstr_array(char **arr, unsigned count) {
    if (!arr) return;
    for (unsigned i = 0; i < count; i++) enif_free(arr[i]);
    enif_free(arr);
}

/* Marshal an Elixir list of numbers into a heap array of floats. Returns 1 on
 * success (*out is NULL when count==0), 0 (badarg) otherwise. Caller frees. */
static int list_to_float_array(ErlNifEnv *env, ERL_NIF_TERM list_term,
                               float **out, unsigned *out_count) {
    unsigned count = 0;
    if (!enif_get_list_length(env, list_term, &count)) return 0;
    float *arr = NULL;
    if (count > 0) {
        arr = enif_alloc(count * sizeof(float));
        if (!arr) return 0;
        ERL_NIF_TERM list = list_term, head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            double v;
            if (!enif_get_double(env, head, &v)) {
                int iv;
                if (!enif_get_int(env, head, &iv)) { enif_free(arr); return 0; }
                v = (double)iv;
            }
            arr[i++] = (float)v;
        }
    }
    *out = arr;
    *out_count = count;
    return 1;
}

/* Marshal an Elixir list of integers into a heap array of int32. */
static int list_to_i32_array(ErlNifEnv *env, ERL_NIF_TERM list_term,
                             int32_t **out, unsigned *out_count) {
    unsigned count = 0;
    if (!enif_get_list_length(env, list_term, &count)) return 0;
    int32_t *arr = NULL;
    if (count > 0) {
        arr = enif_alloc(count * sizeof(int32_t));
        if (!arr) return 0;
        ERL_NIF_TERM list = list_term, head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            int iv;
            if (!enif_get_int(env, head, &iv)) { enif_free(arr); return 0; }
            arr[i++] = (int32_t)iv;
        }
    }
    *out = arr;
    *out_count = count;
    return 1;
}

/* ── EmbeddedFont ─────────────────────────────────────────────────────────── */
static ERL_NIF_TERM make_font(ErlNifEnv *env, EmbeddedFont *h) {
    FontRes *r = enif_alloc_resource(FONT_RES, sizeof(FontRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

static ERL_NIF_TERM font_from_file(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *path = term_to_cstr(env, a[0]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    EmbeddedFont *h = pdf_embedded_font_from_file(path, &code);
    enif_free(path);
    return h ? make_font(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM font_from_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    /* a[1] is the optional name binary; an empty binary means "use PS name". */
    char *name = term_to_cstr(env, a[1]);
    int32_t code = 0;
    const char *name_arg = (name && name[0]) ? name : NULL;
    EmbeddedFont *h = pdf_embedded_font_from_bytes(bin.data, bin.size, name_arg, &code);
    enif_free(name);
    return h ? make_font(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM font_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    FontRes *r;
    if (!enif_get_resource(env, a[0], FONT_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_embedded_font_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── DocumentBuilder ──────────────────────────────────────────────────────── */
static ERL_NIF_TERM make_doc_builder(ErlNifEnv *env, FfiDocumentBuilder *h) {
    DBldRes *r = enif_alloc_resource(DBLD_RES, sizeof(DBldRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

static ERL_NIF_TERM make_page_builder(ErlNifEnv *env, FfiPageBuilder *h) {
    PBldRes *r = enif_alloc_resource(PBLD_RES, sizeof(PBldRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

static ERL_NIF_TERM dbld_create(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    int32_t code = 0;
    FfiDocumentBuilder *h = pdf_document_builder_create(&code);
    return h ? make_doc_builder(env, h) : err_tuple(env, code);
}

/* DocumentBuilder string-setter family (title/author/subject/keywords/creator/
 * on_open/language). Each takes one string arg and returns :ok / {:error}. */
#define DBLD_STR_NIF(name, cfn)                                                  \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        GET_DBLD                                                               \
        char *s = term_to_cstr(env, a[1]);                                    \
        if (!s) return enif_make_badarg(env);                                 \
        int32_t code = 0;                                                      \
        int rc = cfn(r->h, s, &code);                                          \
        enif_free(s);                                                          \
        return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);    \
    }
DBLD_STR_NIF(dbld_set_title, pdf_document_builder_set_title)
DBLD_STR_NIF(dbld_set_author, pdf_document_builder_set_author)
DBLD_STR_NIF(dbld_set_subject, pdf_document_builder_set_subject)
DBLD_STR_NIF(dbld_set_keywords, pdf_document_builder_set_keywords)
DBLD_STR_NIF(dbld_set_creator, pdf_document_builder_set_creator)
DBLD_STR_NIF(dbld_on_open, pdf_document_builder_on_open)
DBLD_STR_NIF(dbld_language, pdf_document_builder_language)

static ERL_NIF_TERM dbld_tagged_pdf_ua1(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    int32_t code = 0;
    int rc = pdf_document_builder_tagged_pdf_ua1(r->h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_role_map(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    char *custom = term_to_cstr(env, a[1]);
    char *standard = term_to_cstr(env, a[2]);
    if (!custom || !standard) { enif_free(custom); enif_free(standard); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = pdf_document_builder_role_map(r->h, custom, standard, &code);
    enif_free(custom); enif_free(standard);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_register_embedded_font(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    FontRes *fr;
    if (!enif_get_resource(env, a[2], FONT_RES, (void **)&fr) || !fr->h) {
        enif_free(name);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_document_builder_register_embedded_font(r->h, name, fr->h, &code);
    enif_free(name);
    /* On success the builder consumed the font: null the wrapper so neither the
     * GC dtor nor a later font_close double-frees it. On error it remains valid. */
    if (rc == 0) { fr->h = NULL; return enif_make_atom(env, "ok"); }
    return err_tuple(env, code);
}

static ERL_NIF_TERM dbld_a4_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    int32_t code = 0;
    FfiPageBuilder *h = pdf_document_builder_a4_page(r->h, &code);
    return h ? make_page_builder(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_letter_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    int32_t code = 0;
    FfiPageBuilder *h = pdf_document_builder_letter_page(r->h, &code);
    return h ? make_page_builder(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    double width, height;
    if (!enif_get_double(env, a[1], &width) || !enif_get_double(env, a[2], &height))
        return enif_make_badarg(env);
    int32_t code = 0;
    FfiPageBuilder *h = pdf_document_builder_page(r->h, (float)width, (float)height, &code);
    return h ? make_page_builder(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_build(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_builder_build(r->h, &len, &code);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM dbld_save(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    char *path = term_to_cstr(env, a[1]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_document_builder_save(r->h, path, &code);
    enif_free(path);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_save_encrypted(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    char *path = term_to_cstr(env, a[1]);
    char *user_pw = term_to_cstr(env, a[2]);
    char *owner_pw = term_to_cstr(env, a[3]);
    if (!path || !user_pw || !owner_pw) {
        enif_free(path); enif_free(user_pw); enif_free(owner_pw);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_document_builder_save_encrypted(r->h, path, user_pw, owner_pw, &code);
    enif_free(path); enif_free(user_pw); enif_free(owner_pw);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM dbld_to_bytes_encrypted(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DBLD
    char *user_pw = term_to_cstr(env, a[1]);
    char *owner_pw = term_to_cstr(env, a[2]);
    if (!user_pw || !owner_pw) { enif_free(user_pw); enif_free(owner_pw); return enif_make_badarg(env); }
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_builder_to_bytes_encrypted(r->h, user_pw, owner_pw, &len, &code);
    enif_free(user_pw); enif_free(owner_pw);
    return ok_bytes(env, p, (size_t)len, code);
}

static ERL_NIF_TERM dbld_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DBldRes *r;
    if (!enif_get_resource(env, a[0], DBLD_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_document_builder_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── PageBuilder ──────────────────────────────────────────────────────────── */

/* PageBuilder no-arg op family (horizontal_rule/watermark_confidential/
 * watermark_draft/newline/new_page_same_size/streaming_table_flush/
 * streaming_table_finish). Each returns :ok / {:error}. */
#define PBLD_NOARG_NIF(name, cfn)                                               \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        GET_PBLD                                                               \
        int32_t code = 0;                                                      \
        int rc = cfn(r->h, &code);                                             \
        return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);    \
    }
PBLD_NOARG_NIF(pbld_horizontal_rule, pdf_page_builder_horizontal_rule)
PBLD_NOARG_NIF(pbld_watermark_confidential, pdf_page_builder_watermark_confidential)
PBLD_NOARG_NIF(pbld_watermark_draft, pdf_page_builder_watermark_draft)
PBLD_NOARG_NIF(pbld_newline, pdf_page_builder_newline)
PBLD_NOARG_NIF(pbld_new_page_same_size, pdf_page_builder_new_page_same_size)
PBLD_NOARG_NIF(pbld_streaming_table_flush, pdf_page_builder_streaming_table_flush)
PBLD_NOARG_NIF(pbld_streaming_table_finish, pdf_page_builder_streaming_table_finish)

/* PageBuilder single-string op family (text/paragraph/link_url/link_named/
 * link_javascript/on_open/on_close/field ops/sticky_note/watermark/stamp/
 * inline/inline_bold/inline_italic). */
#define PBLD_STR_NIF(name, cfn)                                                 \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        GET_PBLD                                                               \
        char *s = term_to_cstr(env, a[1]);                                    \
        if (!s) return enif_make_badarg(env);                                 \
        int32_t code = 0;                                                      \
        int rc = cfn(r->h, s, &code);                                          \
        enif_free(s);                                                          \
        return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);    \
    }
PBLD_STR_NIF(pbld_text, pdf_page_builder_text)
PBLD_STR_NIF(pbld_paragraph, pdf_page_builder_paragraph)
PBLD_STR_NIF(pbld_link_url, pdf_page_builder_link_url)
PBLD_STR_NIF(pbld_link_named, pdf_page_builder_link_named)
PBLD_STR_NIF(pbld_link_javascript, pdf_page_builder_link_javascript)
PBLD_STR_NIF(pbld_on_open, pdf_page_builder_on_open)
PBLD_STR_NIF(pbld_on_close, pdf_page_builder_on_close)
PBLD_STR_NIF(pbld_field_keystroke, pdf_page_builder_field_keystroke)
PBLD_STR_NIF(pbld_field_format, pdf_page_builder_field_format)
PBLD_STR_NIF(pbld_field_validate, pdf_page_builder_field_validate)
PBLD_STR_NIF(pbld_field_calculate, pdf_page_builder_field_calculate)
PBLD_STR_NIF(pbld_sticky_note, pdf_page_builder_sticky_note)
PBLD_STR_NIF(pbld_watermark, pdf_page_builder_watermark)
PBLD_STR_NIF(pbld_stamp, pdf_page_builder_stamp)
PBLD_STR_NIF(pbld_inline, pdf_page_builder_inline)
PBLD_STR_NIF(pbld_inline_bold, pdf_page_builder_inline_bold)
PBLD_STR_NIF(pbld_inline_italic, pdf_page_builder_inline_italic)

/* PageBuilder RGB-decoration family (highlight/underline/strikeout/squiggly):
 * (r, g, b) floats. */
#define PBLD_RGB_NIF(name, cfn)                                                 \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        GET_PBLD                                                               \
        double rr, gg, bb;                                                     \
        if (!enif_get_double(env, a[1], &rr) || !enif_get_double(env, a[2], &gg) || \
            !enif_get_double(env, a[3], &bb))                                  \
            return enif_make_badarg(env);                                      \
        int32_t code = 0;                                                      \
        int rc = cfn(r->h, (float)rr, (float)gg, (float)bb, &code);           \
        return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);    \
    }
PBLD_RGB_NIF(pbld_highlight, pdf_page_builder_highlight)
PBLD_RGB_NIF(pbld_underline, pdf_page_builder_underline)
PBLD_RGB_NIF(pbld_strikeout, pdf_page_builder_strikeout)
PBLD_RGB_NIF(pbld_squiggly, pdf_page_builder_squiggly)

static ERL_NIF_TERM pbld_font(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double size;
    if (!enif_get_double(env, a[2], &size)) { enif_free(name); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = pdf_page_builder_font(r->h, name, (float)size, &code);
    enif_free(name);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_at(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_at(r->h, (float)x, (float)y, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_heading(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    int level;
    if (!enif_get_int(env, a[1], &level)) return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[2]);
    if (!text) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_heading(r->h, (uint8_t)level, text, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_space(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double points;
    if (!enif_get_double(env, a[1], &points)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_space(r->h, (float)points, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_link_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long page_index;
    if (!enif_get_ulong(env, a[1], &page_index)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_link_page(r->h, (uintptr_t)page_index, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_sticky_note_at(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y))
        return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[3]);
    if (!text) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_sticky_note_at(r->h, (float)x, (float)y, text, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_freetext(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h))
        return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[5]);
    if (!text) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_freetext(r->h, (float)x, (float)y, (float)w, (float)h, text, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_inline_color(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double rr, gg, bb;
    if (!enif_get_double(env, a[1], &rr) || !enif_get_double(env, a[2], &gg) ||
        !enif_get_double(env, a[3], &bb))
        return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[4]);
    if (!text) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_inline_color(r->h, (float)rr, (float)gg, (float)bb, text, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_columns(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long column_count;
    double gap;
    if (!enif_get_ulong(env, a[1], &column_count) || !enif_get_double(env, a[2], &gap))
        return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[3]);
    if (!text) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_columns(r->h, (uint32_t)column_count, (float)gap, text, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_footnote(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *ref_mark = term_to_cstr(env, a[1]);
    char *note_text = term_to_cstr(env, a[2]);
    if (!ref_mark || !note_text) { enif_free(ref_mark); enif_free(note_text); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = pdf_page_builder_footnote(r->h, ref_mark, note_text, &code);
    enif_free(ref_mark); enif_free(note_text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_text_field(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h)) {
        enif_free(name); return enif_make_badarg(env);
    }
    char *def = term_to_cstr(env, a[6]);
    const char *def_arg = (def && def[0]) ? def : NULL;
    int32_t code = 0;
    int rc = pdf_page_builder_text_field(r->h, name, (float)x, (float)y, (float)w, (float)h, def_arg, &code);
    enif_free(name); enif_free(def);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_checkbox(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double x, y, w, h;
    int checked;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h) ||
        !enif_get_int(env, a[6], &checked)) {
        enif_free(name); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_checkbox(r->h, name, (float)x, (float)y, (float)w, (float)h, checked, &code);
    enif_free(name);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_combo_box(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h)) {
        enif_free(name); return enif_make_badarg(env);
    }
    char **options = NULL;
    unsigned n_options = 0;
    if (!list_to_cstr_array(env, a[6], &options, &n_options)) { enif_free(name); return enif_make_badarg(env); }
    char *selected = term_to_cstr(env, a[7]);
    const char *sel_arg = (selected && selected[0]) ? selected : NULL;
    int32_t code = 0;
    int rc = pdf_page_builder_combo_box(r->h, name, (float)x, (float)y, (float)w, (float)h,
                                        (const char *const *)options, (uintptr_t)n_options, sel_arg, &code);
    enif_free(name); enif_free(selected);
    free_cstr_array(options, n_options);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_radio_group(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    char **values = NULL;
    float *xs = NULL, *ys = NULL, *ws = NULL, *hs = NULL;
    unsigned n_values = 0, n_xs = 0, n_ys = 0, n_ws = 0, n_hs = 0;
    char *selected = NULL;
    if (!list_to_cstr_array(env, a[2], &values, &n_values)) { enif_free(name); return enif_make_badarg(env); }
    if (!list_to_float_array(env, a[3], &xs, &n_xs) ||
        !list_to_float_array(env, a[4], &ys, &n_ys) ||
        !list_to_float_array(env, a[5], &ws, &n_ws) ||
        !list_to_float_array(env, a[6], &hs, &n_hs)) {
        enif_free(name); free_cstr_array(values, n_values);
        enif_free(xs); enif_free(ys); enif_free(ws); enif_free(hs);
        return enif_make_badarg(env);
    }
    selected = term_to_cstr(env, a[7]);
    const char *sel_arg = (selected && selected[0]) ? selected : NULL;
    int32_t code = 0;
    int rc = pdf_page_builder_radio_group(r->h, name, (const char *const *)values,
                                          xs, ys, ws, hs, (uintptr_t)n_values, sel_arg, &code);
    enif_free(name); enif_free(selected);
    free_cstr_array(values, n_values);
    enif_free(xs); enif_free(ys); enif_free(ws); enif_free(hs);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_push_button(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h)) {
        enif_free(name); return enif_make_badarg(env);
    }
    char *caption = term_to_cstr(env, a[6]);
    if (!caption) { enif_free(name); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = pdf_page_builder_push_button(r->h, name, (float)x, (float)y, (float)w, (float)h, caption, &code);
    enif_free(name); enif_free(caption);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_signature_field(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *name = term_to_cstr(env, a[1]);
    if (!name) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h)) {
        enif_free(name); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_signature_field(r->h, name, (float)x, (float)y, (float)w, (float)h, &code);
    enif_free(name);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_barcode_1d(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    int barcode_type;
    if (!enif_get_int(env, a[1], &barcode_type)) return enif_make_badarg(env);
    char *data = term_to_cstr(env, a[2]);
    if (!data) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[3], &x) || !enif_get_double(env, a[4], &y) ||
        !enif_get_double(env, a[5], &w) || !enif_get_double(env, a[6], &h)) {
        enif_free(data); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_barcode_1d(r->h, barcode_type, data, (float)x, (float)y, (float)w, (float)h, &code);
    enif_free(data);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_barcode_qr(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char *data = term_to_cstr(env, a[1]);
    if (!data) return enif_make_badarg(env);
    double x, y, size;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &size)) {
        enif_free(data); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_barcode_qr(r->h, data, (float)x, (float)y, (float)size, &code);
    enif_free(data);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

/* PageBuilder image op family (image / image_artifact): bytes + (x,y,w,h). */
#define PBLD_IMAGE_NIF(name, cfn)                                              \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {\
        (void)argc;                                                            \
        GET_PBLD                                                               \
        ErlNifBinary bin;                                                      \
        if (!enif_inspect_binary(env, a[1], &bin)) return enif_make_badarg(env); \
        double x, y, w, h;                                                     \
        if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) || \
            !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h)) \
            return enif_make_badarg(env);                                      \
        int32_t code = 0;                                                      \
        int rc = cfn(r->h, bin.data, bin.size, (float)x, (float)y, (float)w, (float)h, &code); \
        return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);    \
    }
PBLD_IMAGE_NIF(pbld_image, pdf_page_builder_image)
PBLD_IMAGE_NIF(pbld_image_artifact, pdf_page_builder_image_artifact)

static ERL_NIF_TERM pbld_image_with_alt(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[1], &bin)) return enif_make_badarg(env);
    double x, y, w, h;
    if (!enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h))
        return enif_make_badarg(env);
    char *alt = term_to_cstr(env, a[6]);
    if (!alt) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_image_with_alt(r->h, bin.data, bin.size,
                                             (float)x, (float)y, (float)w, (float)h, alt, &code);
    enif_free(alt);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_rect(r->h, (float)x, (float)y, (float)w, (float)h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_filled_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h, rr, gg, bb;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h) ||
        !enif_get_double(env, a[5], &rr) || !enif_get_double(env, a[6], &gg) ||
        !enif_get_double(env, a[7], &bb))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_filled_rect(r->h, (float)x, (float)y, (float)w, (float)h,
                                          (float)rr, (float)gg, (float)bb, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_line(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x1, y1, x2, y2;
    if (!enif_get_double(env, a[1], &x1) || !enif_get_double(env, a[2], &y1) ||
        !enif_get_double(env, a[3], &x2) || !enif_get_double(env, a[4], &y2))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_line(r->h, (float)x1, (float)y1, (float)x2, (float)y2, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_stroke_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h, width, rr, gg, bb;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h) ||
        !enif_get_double(env, a[5], &width) || !enif_get_double(env, a[6], &rr) ||
        !enif_get_double(env, a[7], &gg) || !enif_get_double(env, a[8], &bb))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_stroke_rect(r->h, (float)x, (float)y, (float)w, (float)h,
                                          (float)width, (float)rr, (float)gg, (float)bb, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_stroke_line(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x1, y1, x2, y2, width, rr, gg, bb;
    if (!enif_get_double(env, a[1], &x1) || !enif_get_double(env, a[2], &y1) ||
        !enif_get_double(env, a[3], &x2) || !enif_get_double(env, a[4], &y2) ||
        !enif_get_double(env, a[5], &width) || !enif_get_double(env, a[6], &rr) ||
        !enif_get_double(env, a[7], &gg) || !enif_get_double(env, a[8], &bb))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_stroke_line(r->h, (float)x1, (float)y1, (float)x2, (float)y2,
                                          (float)width, (float)rr, (float)gg, (float)bb, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_stroke_rect_dashed(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h, width, rr, gg, bb, phase;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h) ||
        !enif_get_double(env, a[5], &width) || !enif_get_double(env, a[6], &rr) ||
        !enif_get_double(env, a[7], &gg) || !enif_get_double(env, a[8], &bb) ||
        !enif_get_double(env, a[10], &phase))
        return enif_make_badarg(env);
    float *dash = NULL;
    unsigned n_dash = 0;
    if (!list_to_float_array(env, a[9], &dash, &n_dash)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_stroke_rect_dashed(r->h, (float)x, (float)y, (float)w, (float)h,
                                                 (float)width, (float)rr, (float)gg, (float)bb,
                                                 dash, (uintptr_t)n_dash, (float)phase, &code);
    if (dash) enif_free(dash);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_stroke_line_dashed(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x1, y1, x2, y2, width, rr, gg, bb, phase;
    if (!enif_get_double(env, a[1], &x1) || !enif_get_double(env, a[2], &y1) ||
        !enif_get_double(env, a[3], &x2) || !enif_get_double(env, a[4], &y2) ||
        !enif_get_double(env, a[5], &width) || !enif_get_double(env, a[6], &rr) ||
        !enif_get_double(env, a[7], &gg) || !enif_get_double(env, a[8], &bb) ||
        !enif_get_double(env, a[10], &phase))
        return enif_make_badarg(env);
    float *dash = NULL;
    unsigned n_dash = 0;
    if (!list_to_float_array(env, a[9], &dash, &n_dash)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_stroke_line_dashed(r->h, (float)x1, (float)y1, (float)x2, (float)y2,
                                                 (float)width, (float)rr, (float)gg, (float)bb,
                                                 dash, (uintptr_t)n_dash, (float)phase, &code);
    if (dash) enif_free(dash);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_text_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    double x, y, w, h;
    int align;
    if (!enif_get_double(env, a[1], &x) || !enif_get_double(env, a[2], &y) ||
        !enif_get_double(env, a[3], &w) || !enif_get_double(env, a[4], &h))
        return enif_make_badarg(env);
    char *text = term_to_cstr(env, a[5]);
    if (!text) return enif_make_badarg(env);
    if (!enif_get_int(env, a[6], &align)) { enif_free(text); return enif_make_badarg(env); }
    int32_t code = 0;
    int rc = pdf_page_builder_text_in_rect(r->h, (float)x, (float)y, (float)w, (float)h, text, align, &code);
    enif_free(text);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_table(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long n_columns, n_rows;
    int has_header;
    if (!enif_get_ulong(env, a[1], &n_columns) || !enif_get_ulong(env, a[4], &n_rows) ||
        !enif_get_int(env, a[6], &has_header))
        return enif_make_badarg(env);
    float *widths = NULL;
    int32_t *aligns = NULL;
    char **cells = NULL;
    unsigned n_widths = 0, n_aligns = 0, n_cells = 0;
    if (!list_to_float_array(env, a[2], &widths, &n_widths)) return enif_make_badarg(env);
    if (!list_to_i32_array(env, a[3], &aligns, &n_aligns)) { enif_free(widths); return enif_make_badarg(env); }
    if (!list_to_cstr_array(env, a[5], &cells, &n_cells)) {
        enif_free(widths); enif_free(aligns); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_table(r->h, (uintptr_t)n_columns, widths, aligns, (uintptr_t)n_rows,
                                    (const char *const *)cells, has_header, &code);
    enif_free(widths); enif_free(aligns);
    free_cstr_array(cells, n_cells);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_streaming_table_begin(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long n_columns;
    int repeat_header;
    if (!enif_get_ulong(env, a[1], &n_columns) || !enif_get_int(env, a[5], &repeat_header))
        return enif_make_badarg(env);
    char **headers = NULL;
    float *widths = NULL;
    int32_t *aligns = NULL;
    unsigned n_headers = 0, n_widths = 0, n_aligns = 0;
    if (!list_to_cstr_array(env, a[2], &headers, &n_headers)) return enif_make_badarg(env);
    if (!list_to_float_array(env, a[3], &widths, &n_widths)) { free_cstr_array(headers, n_headers); return enif_make_badarg(env); }
    if (!list_to_i32_array(env, a[4], &aligns, &n_aligns)) {
        free_cstr_array(headers, n_headers); enif_free(widths); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_streaming_table_begin(r->h, (uintptr_t)n_columns,
                                                    (const char *const *)headers, widths, aligns,
                                                    repeat_header, &code);
    free_cstr_array(headers, n_headers); enif_free(widths); enif_free(aligns);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_streaming_table_begin_v2(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long n_columns, sample_rows, max_rowspan;
    int repeat_header, mode;
    double min_w, max_w;
    if (!enif_get_ulong(env, a[1], &n_columns) || !enif_get_int(env, a[5], &repeat_header) ||
        !enif_get_int(env, a[6], &mode) || !enif_get_ulong(env, a[7], &sample_rows) ||
        !enif_get_double(env, a[8], &min_w) || !enif_get_double(env, a[9], &max_w) ||
        !enif_get_ulong(env, a[10], &max_rowspan))
        return enif_make_badarg(env);
    char **headers = NULL;
    float *widths = NULL;
    int32_t *aligns = NULL;
    unsigned n_headers = 0, n_widths = 0, n_aligns = 0;
    if (!list_to_cstr_array(env, a[2], &headers, &n_headers)) return enif_make_badarg(env);
    if (!list_to_float_array(env, a[3], &widths, &n_widths)) { free_cstr_array(headers, n_headers); return enif_make_badarg(env); }
    if (!list_to_i32_array(env, a[4], &aligns, &n_aligns)) {
        free_cstr_array(headers, n_headers); enif_free(widths); return enif_make_badarg(env);
    }
    int32_t code = 0;
    int rc = pdf_page_builder_streaming_table_begin_v2(r->h, (uintptr_t)n_columns,
                                                       (const char *const *)headers, widths, aligns,
                                                       repeat_header, mode, (uintptr_t)sample_rows,
                                                       (float)min_w, (float)max_w,
                                                       (uintptr_t)max_rowspan, &code);
    free_cstr_array(headers, n_headers); enif_free(widths); enif_free(aligns);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_streaming_table_set_batch_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    unsigned long batch_size;
    if (!enif_get_ulong(env, a[1], &batch_size)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_streaming_table_set_batch_size(r->h, (uintptr_t)batch_size, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_streaming_table_pending_row_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    uintptr_t n = pdf_page_builder_streaming_table_pending_row_count(r->h);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_uint64(env, (ErlNifUInt64)n));
}

static ERL_NIF_TERM pbld_streaming_table_batch_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    uintptr_t n = pdf_page_builder_streaming_table_batch_count(r->h);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_uint64(env, (ErlNifUInt64)n));
}

static ERL_NIF_TERM pbld_streaming_table_push_row(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char **cells = NULL;
    unsigned n_cells = 0;
    if (!list_to_cstr_array(env, a[1], &cells, &n_cells)) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_page_builder_streaming_table_push_row(r->h, (uintptr_t)n_cells,
                                                       (const char *const *)cells, &code);
    free_cstr_array(cells, n_cells);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM pbld_streaming_table_push_row_v2(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    char **cells = NULL;
    unsigned n_cells = 0;
    if (!list_to_cstr_array(env, a[1], &cells, &n_cells)) return enif_make_badarg(env);
    /* a[2] is a list of rowspans (ints) or an empty list for "all rowspan=1". */
    unsigned n_spans = 0;
    if (!enif_get_list_length(env, a[2], &n_spans)) { free_cstr_array(cells, n_cells); return enif_make_badarg(env); }
    uintptr_t *spans = NULL;
    if (n_spans > 0) {
        spans = enif_alloc(n_spans * sizeof(uintptr_t));
        if (!spans) { free_cstr_array(cells, n_cells); return enif_make_badarg(env); }
        ERL_NIF_TERM list = a[2], head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            unsigned long v;
            if (!enif_get_ulong(env, head, &v)) {
                enif_free(spans); free_cstr_array(cells, n_cells); return enif_make_badarg(env);
            }
            spans[i++] = (uintptr_t)v;
        }
    }
    int32_t code = 0;
    int rc = pdf_page_builder_streaming_table_push_row_v2(r->h, (uintptr_t)n_cells,
                                                          (const char *const *)cells, spans, &code);
    if (spans) enif_free(spans);
    free_cstr_array(cells, n_cells);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

/* Commit the page's buffered ops to its parent builder. CONSUMES the handle:
 * on success the wrapper is nulled so the dtor / close are no-ops. */
static ERL_NIF_TERM pbld_done(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_PBLD
    int32_t code = 0;
    int rc = pdf_page_builder_done(r->h, &code);
    if (rc == 0) { r->h = NULL; return enif_make_atom(env, "ok"); }
    return err_tuple(env, code);
}

static ERL_NIF_TERM pbld_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PBldRes *r;
    if (!enif_get_resource(env, a[0], PBLD_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_page_builder_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── phase 6: digital signatures / PKI / timestamps / TSA / validation ────────
 * Certificate / SignatureInfo / Timestamp / TsaClient / Dss / PDF-{A,UA,X}
 * result handles are opaque, each wrapped in its own resource type and freed by
 * its dtor (or eagerly by *_close/1). A loaded handle is returned as
 * {:ok, ref}; a NULL native handle becomes {:error, code}. String returns are
 * taken via free_string (ok_string), byte returns via free_bytes (ok_bytes);
 * const byte returns (timestamp token / message imprint) are COPIED and NOT
 * freed. Crypto/network ops are dirty CPU-bound. */

/* Wrap an owned opaque handle into {:ok, ref} on RES, or {:error, code} when
 * NULL. */
#define MAKE_HANDLE(RES, RESTYPE, FIELD)                                         \
    do {                                                                        \
        RESTYPE *rr = enif_alloc_resource(RES, sizeof(RESTYPE));                \
        rr->h = (FIELD);                                                        \
        ERL_NIF_TERM tt = enif_make_resource(env, rr);                         \
        enif_release_resource(rr);                                             \
        return enif_make_tuple2(env, enif_make_atom(env, "ok"), tt);           \
    } while (0)

/* Read an owned const-byte buffer (len via out param) into {:ok, binary}
 * WITHOUT freeing it (the buffer is owned by the parent handle). */
static ERL_NIF_TERM copy_const_bytes(ErlNifEnv *env, const uint8_t *p, uintptr_t n, int32_t code) {
    if (!p) return err_tuple(env, code);
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, (size_t)n, &bin);
    if (n) memcpy(buf, p, (size_t)n);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), bin);
}

/* ── Certificate ──────────────────────────────────────────────────────────── */
static ERL_NIF_TERM cert_load_from_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    char *pw = term_to_cstr(env, a[1]);
    if (!pw) return enif_make_badarg(env);
    int32_t code = 0;
    void *h = pdf_certificate_load_from_bytes(bin.data, (int32_t)bin.size, pw, &code);
    enif_free(pw);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(CERT_RES, CertRes, h);
}

static ERL_NIF_TERM cert_load_from_pem(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *cert = term_to_cstr(env, a[0]);
    char *key = term_to_cstr(env, a[1]);
    if (!cert || !key) { enif_free(cert); enif_free(key); return enif_make_badarg(env); }
    int32_t code = 0;
    void *h = pdf_certificate_load_from_pem(cert, key, &code);
    enif_free(cert); enif_free(key);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(CERT_RES, CertRes, h);
}

#define GET_CERT                                                                 \
    CertRes *r;                                                                  \
    if (!enif_get_resource(env, a[0], CERT_RES, (void **)&r))                    \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM cert_get_subject(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_CERT
    int32_t code = 0;
    return ok_string(env, pdf_certificate_get_subject(r->h, &code), code);
}
static ERL_NIF_TERM cert_get_issuer(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_CERT
    int32_t code = 0;
    return ok_string(env, pdf_certificate_get_issuer(r->h, &code), code);
}
static ERL_NIF_TERM cert_get_serial(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_CERT
    int32_t code = 0;
    return ok_string(env, pdf_certificate_get_serial(r->h, &code), code);
}
static ERL_NIF_TERM cert_get_validity(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_CERT
    int64_t nb = 0, na = 0;
    int32_t code = 0;
    pdf_certificate_get_validity(r->h, &nb, &na, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_tuple2(env, enif_make_int64(env, nb),
                                             enif_make_int64(env, na)));
}
static ERL_NIF_TERM cert_is_valid(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_CERT
    int32_t code = 0;
    int32_t v = pdf_certificate_is_valid(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM cert_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    CertRes *r;
    if (!enif_get_resource(env, a[0], CERT_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_certificate_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── Signing ──────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM sign_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary pdf;
    CertRes *c;
    if (!enif_inspect_binary(env, a[0], &pdf)) return enif_make_badarg(env);
    if (!enif_get_resource(env, a[1], CERT_RES, (void **)&c)) return enif_make_badarg(env);
    if (!c->h) return enif_make_badarg(env);
    char *reason = term_to_cstr(env, a[2]);
    char *location = term_to_cstr(env, a[3]);
    if (!reason || !location) { enif_free(reason); enif_free(location); return enif_make_badarg(env); }
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_sign_bytes(pdf.data, pdf.size, c->h, reason, location, &out_len, &code);
    enif_free(reason); enif_free(location);
    return ok_bytes(env, p, (size_t)out_len, code);
}

/* Marshal an Elixir list-of-binaries into a parallel
 * (const uint8_t* const*, const uintptr_t*) pair. Returns the count and fills
 * the ptrs and lens out-arrays (caller frees both via enif_free); on a non-binary element the
 * partially-built arrays are freed and -1 is returned. The binary terms keep
 * the data alive for the duration of the call. */
static int marshal_byte_arrays(ErlNifEnv *env, ERL_NIF_TERM list,
                               const uint8_t ***ptrs, uintptr_t **lens) {
    unsigned count = 0;
    if (!enif_get_list_length(env, list, &count)) return -1;
    const uint8_t **pp = NULL;
    uintptr_t *ll = NULL;
    if (count > 0) {
        pp = enif_alloc(count * sizeof(uint8_t *));
        ll = enif_alloc(count * sizeof(uintptr_t));
        if (!pp || !ll) { enif_free(pp); enif_free(ll); return -1; }
        ERL_NIF_TERM cur = list, head;
        unsigned i = 0;
        while (enif_get_list_cell(env, cur, &head, &cur)) {
            ErlNifBinary b;
            if (!enif_inspect_binary(env, head, &b)) { enif_free(pp); enif_free(ll); return -1; }
            pp[i] = b.data;
            ll[i] = b.size;
            i++;
        }
    }
    *ptrs = pp;
    *lens = ll;
    return (int)count;
}

static ERL_NIF_TERM sign_bytes_pades(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    /* args: pdf, cert, level, tsa_url, reason, location, certs, crls, ocsps */
    ErlNifBinary pdf;
    CertRes *c;
    int level;
    if (!enif_inspect_binary(env, a[0], &pdf)) return enif_make_badarg(env);
    if (!enif_get_resource(env, a[1], CERT_RES, (void **)&c)) return enif_make_badarg(env);
    if (!c->h) return enif_make_badarg(env);
    if (!enif_get_int(env, a[2], &level)) return enif_make_badarg(env);
    char *tsa = term_to_cstr(env, a[3]);
    char *reason = term_to_cstr(env, a[4]);
    char *location = term_to_cstr(env, a[5]);
    if (!tsa || !reason || !location) {
        enif_free(tsa); enif_free(reason); enif_free(location);
        return enif_make_badarg(env);
    }
    const uint8_t **certs = NULL, **crls = NULL, **ocsps = NULL;
    uintptr_t *cl = NULL, *rl = NULL, *ol = NULL;
    int nc = marshal_byte_arrays(env, a[6], &certs, &cl);
    int nr = marshal_byte_arrays(env, a[7], &crls, &rl);
    int no = marshal_byte_arrays(env, a[8], &ocsps, &ol);
    if (nc < 0 || nr < 0 || no < 0) {
        enif_free(tsa); enif_free(reason); enif_free(location);
        enif_free(certs); enif_free(cl); enif_free(crls); enif_free(rl);
        enif_free(ocsps); enif_free(ol);
        return enif_make_badarg(env);
    }
    /* An empty tsa_url string means "none" (NULL) for B-B. */
    const char *tsa_arg = (tsa[0] == '\0') ? NULL : tsa;
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_sign_bytes_pades(pdf.data, pdf.size, c->h, level, tsa_arg,
                                      reason, location,
                                      certs, cl, (uintptr_t)nc,
                                      crls, rl, (uintptr_t)nr,
                                      ocsps, ol, (uintptr_t)no,
                                      &out_len, &code);
    enif_free(tsa); enif_free(reason); enif_free(location);
    enif_free(certs); enif_free(cl); enif_free(crls); enif_free(rl);
    enif_free(ocsps); enif_free(ol);
    return ok_bytes(env, p, (size_t)out_len, code);
}

static ERL_NIF_TERM sign_bytes_pades_opts(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    /* args: pdf, cert, level, tsa_url, reason, location, certs, crls, ocsps —
     * same as sign_bytes_pades, but assembled into a PadesSignOptionsC. */
    ErlNifBinary pdf;
    CertRes *c;
    int level;
    if (!enif_inspect_binary(env, a[0], &pdf)) return enif_make_badarg(env);
    if (!enif_get_resource(env, a[1], CERT_RES, (void **)&c)) return enif_make_badarg(env);
    if (!c->h) return enif_make_badarg(env);
    if (!enif_get_int(env, a[2], &level)) return enif_make_badarg(env);
    char *tsa = term_to_cstr(env, a[3]);
    char *reason = term_to_cstr(env, a[4]);
    char *location = term_to_cstr(env, a[5]);
    if (!tsa || !reason || !location) {
        enif_free(tsa); enif_free(reason); enif_free(location);
        return enif_make_badarg(env);
    }
    const uint8_t **certs = NULL, **crls = NULL, **ocsps = NULL;
    uintptr_t *cl = NULL, *rl = NULL, *ol = NULL;
    int nc = marshal_byte_arrays(env, a[6], &certs, &cl);
    int nr = marshal_byte_arrays(env, a[7], &crls, &rl);
    int no = marshal_byte_arrays(env, a[8], &ocsps, &ol);
    if (nc < 0 || nr < 0 || no < 0) {
        enif_free(tsa); enif_free(reason); enif_free(location);
        enif_free(certs); enif_free(cl); enif_free(crls); enif_free(rl);
        enif_free(ocsps); enif_free(ol);
        return enif_make_badarg(env);
    }
    PadesSignOptionsC options;
    memset(&options, 0, sizeof(options));
    options.certificate_handle = c->h;
    options.certs = certs;
    options.cert_lens = cl;
    options.n_certs = (uintptr_t)nc;
    options.crls = crls;
    options.crl_lens = rl;
    options.n_crls = (uintptr_t)nr;
    options.ocsps = ocsps;
    options.ocsp_lens = ol;
    options.n_ocsps = (uintptr_t)no;
    options.tsa_url = (tsa[0] == '\0') ? NULL : tsa;
    options.reason = reason;
    options.location = location;
    options.level = level;
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_sign_bytes_pades_opts(pdf.data, pdf.size, &options, &out_len, &code);
    enif_free(tsa); enif_free(reason); enif_free(location);
    enif_free(certs); enif_free(cl); enif_free(crls); enif_free(rl);
    enif_free(ocsps); enif_free(ol);
    return ok_bytes(env, p, (size_t)out_len, code);
}

/* ── SignatureInfo ────────────────────────────────────────────────────────── */
#define GET_SIG                                                                  \
    SigRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], SIG_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM sig_get_signer_name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    return ok_string(env, pdf_signature_get_signer_name(r->h, &code), code);
}
static ERL_NIF_TERM sig_get_signing_reason(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    return ok_string(env, pdf_signature_get_signing_reason(r->h, &code), code);
}
static ERL_NIF_TERM sig_get_signing_location(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    return ok_string(env, pdf_signature_get_signing_location(r->h, &code), code);
}
static ERL_NIF_TERM sig_get_signing_time(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    int64_t t = pdf_signature_get_signing_time(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int64(env, t));
}
static ERL_NIF_TERM sig_get_certificate(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    void *h = pdf_signature_get_certificate(r->h, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(CERT_RES, CertRes, h);
}
static ERL_NIF_TERM sig_get_pades_level(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    int32_t lvl = pdf_signature_get_pades_level(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, lvl));
}
static ERL_NIF_TERM sig_has_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    bool b = pdf_signature_has_timestamp(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM sig_get_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    void *h = pdf_signature_get_timestamp(r->h, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(TS_RES, TsRes, h);
}
static ERL_NIF_TERM sig_add_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    TsRes *ts;
    if (!enif_get_resource(env, a[1], TS_RES, (void **)&ts)) return enif_make_badarg(env);
    if (!ts->h) return enif_make_badarg(env);
    int32_t code = 0;
    bool b = pdf_signature_add_timestamp(r->h, ts->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM sig_verify(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    int32_t code = 0;
    int32_t v = pdf_signature_verify(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM sig_verify_detached(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_SIG
    ErlNifBinary pdf;
    if (!enif_inspect_binary(env, a[1], &pdf)) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_signature_verify_detached(r->h, pdf.data, pdf.size, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM sig_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    SigRes *r;
    if (!enif_get_resource(env, a[0], SIG_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_signature_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── Timestamp ────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM ts_parse(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    void *h = pdf_timestamp_parse(bin.data, bin.size, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(TS_RES, TsRes, h);
}

#define GET_TS                                                                   \
    TsRes *r;                                                                    \
    if (!enif_get_resource(env, a[0], TS_RES, (void **)&r))                      \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM ts_get_token(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    uintptr_t n = 0;
    int32_t code = 0;
    const uint8_t *p = pdf_timestamp_get_token(r->h, &n, &code);
    return copy_const_bytes(env, p, n, code);
}
static ERL_NIF_TERM ts_get_message_imprint(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    uintptr_t n = 0;
    int32_t code = 0;
    const uint8_t *p = pdf_timestamp_get_message_imprint(r->h, &n, &code);
    return copy_const_bytes(env, p, n, code);
}
static ERL_NIF_TERM ts_get_time(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    int64_t t = pdf_timestamp_get_time(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int64(env, t));
}
static ERL_NIF_TERM ts_get_serial(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    return ok_string(env, pdf_timestamp_get_serial(r->h, &code), code);
}
static ERL_NIF_TERM ts_get_tsa_name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    return ok_string(env, pdf_timestamp_get_tsa_name(r->h, &code), code);
}
static ERL_NIF_TERM ts_get_policy_oid(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    return ok_string(env, pdf_timestamp_get_policy_oid(r->h, &code), code);
}
static ERL_NIF_TERM ts_get_hash_algorithm(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    int32_t v = pdf_timestamp_get_hash_algorithm(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM ts_verify(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TS
    int32_t code = 0;
    bool b = pdf_timestamp_verify(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM ts_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    TsRes *r;
    if (!enif_get_resource(env, a[0], TS_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_timestamp_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── TSA client ───────────────────────────────────────────────────────────── */
static ERL_NIF_TERM tsa_create(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *url = term_to_cstr(env, a[0]);
    char *user = term_to_cstr(env, a[1]);
    char *pw = term_to_cstr(env, a[2]);
    int timeout, hash_algo;
    if (!url || !user || !pw ||
        !enif_get_int(env, a[3], &timeout) ||
        !enif_get_int(env, a[4], &hash_algo)) {
        enif_free(url); enif_free(user); enif_free(pw);
        return enif_make_badarg(env);
    }
    bool use_nonce = enif_is_identical(a[5], enif_make_atom(env, "true"));
    bool cert_req = enif_is_identical(a[6], enif_make_atom(env, "true"));
    /* Empty username/password mean "none" (NULL). */
    const char *user_arg = (user[0] == '\0') ? NULL : user;
    const char *pw_arg = (pw[0] == '\0') ? NULL : pw;
    int32_t code = 0;
    void *h = pdf_tsa_client_create(url, user_arg, pw_arg, timeout, hash_algo,
                                    use_nonce, cert_req, &code);
    enif_free(url); enif_free(user); enif_free(pw);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(TSA_RES, TsaRes, h);
}

#define GET_TSA                                                                  \
    TsaRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], TSA_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM tsa_request_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TSA
    ErlNifBinary data;
    if (!enif_inspect_binary(env, a[1], &data)) return enif_make_badarg(env);
    int32_t code = 0;
    void *h = pdf_tsa_request_timestamp(r->h, data.data, data.size, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(TS_RES, TsRes, h);
}
static ERL_NIF_TERM tsa_request_timestamp_hash(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_TSA
    ErlNifBinary hash;
    int hash_algo;
    if (!enif_inspect_binary(env, a[1], &hash) || !enif_get_int(env, a[2], &hash_algo))
        return enif_make_badarg(env);
    int32_t code = 0;
    void *h = pdf_tsa_request_timestamp_hash(r->h, hash.data, hash.size, hash_algo, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(TS_RES, TsRes, h);
}
static ERL_NIF_TERM tsa_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    TsaRes *r;
    if (!enif_get_resource(env, a[0], TSA_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_tsa_client_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── DSS ──────────────────────────────────────────────────────────────────── */
#define GET_DSS                                                                  \
    DssRes *r;                                                                    \
    if (!enif_get_resource(env, a[0], DSS_RES, (void **)&r))                      \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM dss_cert_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    return enif_make_int(env, pdf_dss_cert_count(r->h));
}
static ERL_NIF_TERM dss_crl_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    return enif_make_int(env, pdf_dss_crl_count(r->h));
}
static ERL_NIF_TERM dss_ocsp_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    return enif_make_int(env, pdf_dss_ocsp_count(r->h));
}
static ERL_NIF_TERM dss_vri_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    return enif_make_int(env, pdf_dss_vri_count(r->h));
}
static ERL_NIF_TERM dss_get_cert(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    uintptr_t n = 0;
    int32_t code = 0;
    uint8_t *p = pdf_dss_get_cert(r->h, index, &n, &code);
    return ok_bytes(env, p, (size_t)n, code);
}
static ERL_NIF_TERM dss_get_crl(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    uintptr_t n = 0;
    int32_t code = 0;
    uint8_t *p = pdf_dss_get_crl(r->h, index, &n, &code);
    return ok_bytes(env, p, (size_t)n, code);
}
static ERL_NIF_TERM dss_get_ocsp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DSS
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    uintptr_t n = 0;
    int32_t code = 0;
    uint8_t *p = pdf_dss_get_ocsp(r->h, index, &n, &code);
    return ok_bytes(env, p, (size_t)n, code);
}
static ERL_NIF_TERM dss_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DssRes *r;
    if (!enif_get_resource(env, a[0], DSS_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_dss_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── Validation: PDF/A, PDF/UA, PDF/X ─────────────────────────────────────── */
static ERL_NIF_TERM validate_pdf_a(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *d;
    int level;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&d) ||
        !enif_get_int(env, a[1], &level))
        return enif_make_badarg(env);
    if (!d->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiPdfAResults *h = pdf_validate_pdf_a_level(d->h, level, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(PDFA_RES, PdfARes, h);
}
static ERL_NIF_TERM validate_pdf_ua(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *d;
    int level;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&d) ||
        !enif_get_int(env, a[1], &level))
        return enif_make_badarg(env);
    if (!d->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiUaResults *h = pdf_validate_pdf_ua(d->h, level, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(PDFUA_RES, PdfUaRes, h);
}
static ERL_NIF_TERM validate_pdf_x(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *d;
    int level;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&d) ||
        !enif_get_int(env, a[1], &level))
        return enif_make_badarg(env);
    if (!d->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiPdfXResults *h = pdf_validate_pdf_x_level(d->h, level, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(PDFX_RES, PdfXRes, h);
}

#define GET_PDFA                                                                 \
    PdfARes *r;                                                                   \
    if (!enif_get_resource(env, a[0], PDFA_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);
#define GET_PDFUA                                                                \
    PdfUaRes *r;                                                                  \
    if (!enif_get_resource(env, a[0], PDFUA_RES, (void **)&r))                    \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);
#define GET_PDFX                                                                 \
    PdfXRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], PDFX_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM pdf_a_is_compliant(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFA
    int32_t code = 0;
    bool b = pdf_pdf_a_is_compliant(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM pdf_a_error_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFA
    return enif_make_int(env, pdf_pdf_a_error_count(r->h));
}
static ERL_NIF_TERM pdf_a_warning_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFA
    return enif_make_int(env, pdf_pdf_a_warning_count(r->h));
}
static ERL_NIF_TERM pdf_a_get_error(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFA
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_pdf_a_get_error(r->h, index, &code), code);
}
static ERL_NIF_TERM pdf_a_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfARes *r;
    if (!enif_get_resource(env, a[0], PDFA_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_pdf_a_results_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM pdf_ua_is_accessible(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    int32_t code = 0;
    bool b = pdf_pdf_ua_is_accessible(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM pdf_ua_error_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    return enif_make_int(env, pdf_pdf_ua_error_count(r->h));
}
static ERL_NIF_TERM pdf_ua_warning_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    return enif_make_int(env, pdf_pdf_ua_warning_count(r->h));
}
static ERL_NIF_TERM pdf_ua_get_error(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_pdf_ua_get_error(r->h, index, &code), code);
}
static ERL_NIF_TERM pdf_ua_get_warning(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_pdf_ua_get_warning(r->h, index, &code), code);
}
static ERL_NIF_TERM pdf_ua_get_stats(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFUA
    int32_t s = 0, im = 0, t = 0, f = 0, an = 0, pg = 0, code = 0;
    bool ok = pdf_pdf_ua_get_stats(r->h, &s, &im, &t, &f, &an, &pg, &code);
    if (!ok) return err_tuple(env, code);
    ERL_NIF_TERM stats = enif_make_tuple6(env, enif_make_int(env, s),
                                          enif_make_int(env, im), enif_make_int(env, t),
                                          enif_make_int(env, f), enif_make_int(env, an),
                                          enif_make_int(env, pg));
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), stats);
}
static ERL_NIF_TERM pdf_ua_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfUaRes *r;
    if (!enif_get_resource(env, a[0], PDFUA_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_pdf_ua_results_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM pdf_x_is_compliant(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFX
    int32_t code = 0;
    bool b = pdf_pdf_x_is_compliant(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}
static ERL_NIF_TERM pdf_x_error_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFX
    return enif_make_int(env, pdf_pdf_x_error_count(r->h));
}
static ERL_NIF_TERM pdf_x_get_error(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_PDFX
    int index;
    if (!enif_get_int(env, a[1], &index)) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_pdf_x_get_error(r->h, index, &code), code);
}
static ERL_NIF_TERM pdf_x_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfXRes *r;
    if (!enif_get_resource(env, a[0], PDFX_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_pdf_x_results_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── log level ────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM oxide_set_log_level(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    int level;
    if (!enif_get_int(env, a[0], &level)) return enif_make_badarg(env);
    pdf_oxide_set_log_level(level);
    return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM oxide_get_log_level(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return enif_make_int(env, pdf_oxide_get_log_level());
}

/* ── phase 7 — barcodes / OCR / render variants / redaction / constructors /
 * page getters / timestamp ────────────────────────────────────────────────────
 * Barcodes are FfiBarcodeImage handles (BARCODE_RES, freed via pdf_barcode_free);
 * the OCR engine and ad-hoc renderer are opaque void handles (OCR_RES/
 * RENDERER_RES). Render variants reuse make_rendered_image / IMG_RES from
 * phase 3. Redaction methods act on an existing editor (GET_EDIT). The
 * from_image and from_html_css constructors return Pdf handles (PDF_RES); page
 * getters read a Document (DOC_RES). All allocation-heavy NIFs are dirty
 * CPU-bound. */

/* Wrap an owned Pdf handle into {:ok, ref} (PDF_RES, GC-freed via pdf_dtor). */
static ERL_NIF_TERM make_pdf(ErlNifEnv *env, Pdf *h) {
    PdfRes *r = enif_alloc_resource(PDF_RES, sizeof(PdfRes));
    r->h = h;
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

/* Read the barcode resource into r and NULL-guard its handle. */
#define GET_BARCODE                                                              \
    BarcodeRes *r;                                                               \
    if (!enif_get_resource(env, a[0], BARCODE_RES, (void **)&r))                 \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM barcode_generate_qr(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *data = term_to_cstr(env, a[0]);
    int ec, size_px;
    if (!data || !enif_get_int(env, a[1], &ec) || !enif_get_int(env, a[2], &size_px)) {
        enif_free(data);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    FfiBarcodeImage *h = pdf_generate_qr_code(data, ec, size_px, &code);
    enif_free(data);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(BARCODE_RES, BarcodeRes, h);
}

static ERL_NIF_TERM barcode_generate(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *data = term_to_cstr(env, a[0]);
    int format, size_px;
    if (!data || !enif_get_int(env, a[1], &format) || !enif_get_int(env, a[2], &size_px)) {
        enif_free(data);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    FfiBarcodeImage *h = pdf_generate_barcode(data, format, size_px, &code);
    enif_free(data);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(BARCODE_RES, BarcodeRes, h);
}

static ERL_NIF_TERM barcode_get_data(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_BARCODE
    int32_t code = 0;
    return ok_string(env, pdf_barcode_get_data(r->h, &code), code);
}

static ERL_NIF_TERM barcode_get_format(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_BARCODE
    int32_t code = 0;
    int32_t v = pdf_barcode_get_format(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}

static ERL_NIF_TERM barcode_get_confidence(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_BARCODE
    int32_t code = 0;
    float v = pdf_barcode_get_confidence(r->h, &code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_double(env, v));
}

static ERL_NIF_TERM barcode_get_image_png(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    BarcodeRes *r;
    int size_px;
    if (!enif_get_resource(env, a[0], BARCODE_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &size_px))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t out_len = 0, code = 0;
    uint8_t *p = pdf_barcode_get_image_png(r->h, size_px, &out_len, &code);
    size_t n = (p && out_len > 0) ? (size_t)out_len : 0;
    return ok_bytes(env, p, n, code);
}

static ERL_NIF_TERM barcode_get_svg(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    BarcodeRes *r;
    int size_px;
    if (!enif_get_resource(env, a[0], BARCODE_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &size_px))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_barcode_get_svg(r->h, size_px, &code), code);
}

static ERL_NIF_TERM barcode_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    BarcodeRes *r;
    if (!enif_get_resource(env, a[0], BARCODE_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_barcode_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM editor_add_barcode_to_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    BarcodeRes *b;
    int page;
    double x, y, w, h;
    if (!enif_get_int(env, a[1], &page) ||
        !enif_get_resource(env, a[2], BARCODE_RES, (void **)&b) ||
        !enif_get_double(env, a[3], &x) || !enif_get_double(env, a[4], &y) ||
        !enif_get_double(env, a[5], &w) || !enif_get_double(env, a[6], &h))
        return enif_make_badarg(env);
    if (!b->h) return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_add_barcode_to_page(r->h, page, b->h, (float)x, (float)y,
                                     (float)w, (float)h, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

/* ── OCR ──────────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM ocr_engine_create(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *det = term_to_cstr(env, a[0]);
    char *rec = term_to_cstr(env, a[1]);
    char *dict = term_to_cstr(env, a[2]);
    if (!det || !rec || !dict) {
        enif_free(det); enif_free(rec); enif_free(dict);
        return enif_make_badarg(env);
    }
    int32_t code = 0;
    void *h = pdf_ocr_engine_create(det, rec, dict, &code);
    enif_free(det); enif_free(rec); enif_free(dict);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(OCR_RES, OcrRes, h);
}

static ERL_NIF_TERM ocr_engine_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    OcrRes *r;
    if (!enif_get_resource(env, a[0], OCR_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_ocr_engine_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM ocr_page_needs_ocr(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    bool b = pdf_ocr_page_needs_ocr(r->h, page, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, b ? "true" : "false"));
}

/* engine arg may be the atom :nil (use native extraction only) or an OCR_RES. */
static ERL_NIF_TERM ocr_extract_text(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    const void *engine = NULL;
    if (!enif_is_identical(a[2], enif_make_atom(env, "nil"))) {
        OcrRes *e;
        if (!enif_get_resource(env, a[2], OCR_RES, (void **)&e)) return enif_make_badarg(env);
        if (!e->h) return enif_make_badarg(env);
        engine = e->h;
    }
    int32_t code = 0;
    return ok_string(env, pdf_ocr_extract_text(r->h, page, engine, &code), code);
}

/* ── render variants ──────────────────────────────────────────────────────── */
static ERL_NIF_TERM doc_render_page_with_options(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, dpi, format, transparent, render_annots, jpeg_quality;
    double bg_r, bg_g, bg_b, bg_a;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &dpi) ||
        !enif_get_int(env, a[3], &format) ||
        !enif_get_double(env, a[4], &bg_r) || !enif_get_double(env, a[5], &bg_g) ||
        !enif_get_double(env, a[6], &bg_b) || !enif_get_double(env, a[7], &bg_a) ||
        !enif_get_int(env, a[8], &transparent) ||
        !enif_get_int(env, a[9], &render_annots) ||
        !enif_get_int(env, a[10], &jpeg_quality))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page_with_options(
        r->h, page_index, dpi, format, (float)bg_r, (float)bg_g, (float)bg_b,
        (float)bg_a, transparent, render_annots, jpeg_quality, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

static ERL_NIF_TERM doc_render_page_with_options_ex(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, dpi, format, transparent, render_annots, jpeg_quality;
    double bg_r, bg_g, bg_b, bg_a;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &dpi) ||
        !enif_get_int(env, a[3], &format) ||
        !enif_get_double(env, a[4], &bg_r) || !enif_get_double(env, a[5], &bg_g) ||
        !enif_get_double(env, a[6], &bg_b) || !enif_get_double(env, a[7], &bg_a) ||
        !enif_get_int(env, a[8], &transparent) ||
        !enif_get_int(env, a[9], &render_annots) ||
        !enif_get_int(env, a[10], &jpeg_quality))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    /* Marshal a[11] (list of strings) into a NULL-terminated-string array. */
    unsigned count = 0;
    if (!enif_get_list_length(env, a[11], &count)) return enif_make_badarg(env);
    char **layers = NULL;
    if (count > 0) {
        layers = enif_alloc(count * sizeof(char *));
        if (!layers) return enif_make_badarg(env);
        ERL_NIF_TERM list = a[11], head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            char *s = term_to_cstr(env, head);
            if (!s) {
                for (unsigned k = 0; k < i; k++) enif_free(layers[k]);
                enif_free(layers);
                return enif_make_badarg(env);
            }
            layers[i++] = s;
        }
    }
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page_with_options_ex(
        r->h, page_index, dpi, format, (float)bg_r, (float)bg_g, (float)bg_b,
        (float)bg_a, transparent, render_annots, jpeg_quality,
        (const char *const *)layers, (uintptr_t)count, &code);
    if (layers) {
        for (unsigned k = 0; k < count; k++) enif_free(layers[k]);
        enif_free(layers);
    }
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

static ERL_NIF_TERM doc_render_page_region(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, format;
    double cx, cy, cw, ch;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_double(env, a[2], &cx) || !enif_get_double(env, a[3], &cy) ||
        !enif_get_double(env, a[4], &cw) || !enif_get_double(env, a[5], &ch) ||
        !enif_get_int(env, a[6], &format))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *h = pdf_render_page_region(r->h, page_index, (float)cx, (float)cy,
                                                 (float)cw, (float)ch, format, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

static ERL_NIF_TERM doc_render_page_fit(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, w, h, format;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &w) || !enif_get_int(env, a[3], &h) ||
        !enif_get_int(env, a[4], &format))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    FfiRenderedImage *img = pdf_render_page_fit(r->h, page_index, w, h, format, &code);
    if (!img) return err_tuple(env, code);
    return make_rendered_image(env, img);
}

static ERL_NIF_TERM doc_render_page_raw(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_index, dpi;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_index) ||
        !enif_get_int(env, a[2], &dpi))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t out_w = 0, out_h = 0, code = 0;
    FfiRenderedImage *h = pdf_render_page_raw(r->h, page_index, dpi, &out_w, &out_h, &code);
    if (!h) return err_tuple(env, code);
    return make_rendered_image(env, h);
}

/* ── ad-hoc renderer ──────────────────────────────────────────────────────── */
static ERL_NIF_TERM renderer_create(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    int dpi, format, quality;
    if (!enif_get_int(env, a[0], &dpi) || !enif_get_int(env, a[1], &format) ||
        !enif_get_int(env, a[2], &quality))
        return enif_make_badarg(env);
    bool anti_alias = enif_is_identical(a[3], enif_make_atom(env, "true"));
    int32_t code = 0;
    void *h = pdf_create_renderer(dpi, format, quality, anti_alias, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(RENDERER_RES, RendererRes, h);
}

static ERL_NIF_TERM renderer_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    RendererRes *r;
    if (!enif_get_resource(env, a[0], RENDERER_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_renderer_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM doc_estimate_render_time(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    int32_t v = pdf_estimate_render_time(r->h, page, &code);
    if (v < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}

/* ── redaction (on a DocumentEditor) ──────────────────────────────────────── */
static ERL_NIF_TERM redaction_add(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page;
    double x1, y1, x2, y2, rr, gg, bb;
    if (!enif_get_int(env, a[1], &page) ||
        !enif_get_double(env, a[2], &x1) || !enif_get_double(env, a[3], &y1) ||
        !enif_get_double(env, a[4], &x2) || !enif_get_double(env, a[5], &y2) ||
        !enif_get_double(env, a[6], &rr) || !enif_get_double(env, a[7], &gg) ||
        !enif_get_double(env, a[8], &bb))
        return enif_make_badarg(env);
    int32_t code = 0;
    int rc = pdf_redaction_add(r->h, (uintptr_t)page, x1, y1, x2, y2, rr, gg, bb, &code);
    return rc == 0 ? enif_make_atom(env, "ok") : err_tuple(env, code);
}

static ERL_NIF_TERM redaction_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int page;
    if (!enif_get_int(env, a[1], &page)) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t n = pdf_redaction_count(r->h, (uintptr_t)page, &code);
    if (n < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM redaction_apply(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    bool scrub = enif_is_identical(a[1], enif_make_atom(env, "true"));
    double rr, gg, bb;
    if (!enif_get_double(env, a[2], &rr) || !enif_get_double(env, a[3], &gg) ||
        !enif_get_double(env, a[4], &bb))
        return enif_make_badarg(env);
    int32_t code = 0;
    int32_t n = pdf_redaction_apply(r->h, scrub, rr, gg, bb, &code);
    if (n < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM redaction_scrub_metadata(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_EDIT
    int32_t code = 0;
    int32_t n = pdf_redaction_scrub_metadata(r->h, &code);
    if (n < 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

/* ── constructors ─────────────────────────────────────────────────────────── */
static ERL_NIF_TERM pdf_from_image_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *path = term_to_cstr(env, a[0]);
    if (!path) return enif_make_badarg(env);
    int32_t code = 0;
    Pdf *h = pdf_from_image(path, &code);
    enif_free(path);
    return h ? make_pdf(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM pdf_from_image_bytes_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    Pdf *h = pdf_from_image_bytes(bin.data, (int32_t)bin.size, &code);
    return h ? make_pdf(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM pdf_from_html_css_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *html = term_to_cstr(env, a[0]);
    char *css = term_to_cstr(env, a[1]);
    ErlNifBinary font;
    if (!html || !css || !enif_inspect_binary(env, a[2], &font)) {
        enif_free(html); enif_free(css);
        return enif_make_badarg(env);
    }
    const uint8_t *fb = font.size ? font.data : NULL;
    int32_t code = 0;
    Pdf *h = pdf_from_html_css(html, css, fb, (uintptr_t)font.size, &code);
    enif_free(html); enif_free(css);
    return h ? make_pdf(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM pdf_from_html_css_with_fonts_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *html = term_to_cstr(env, a[0]);
    char *css = term_to_cstr(env, a[1]);
    if (!html || !css) { enif_free(html); enif_free(css); return enif_make_badarg(env); }
    /* Parallel lists: a[2] families (strings), a[3] font binaries. */
    unsigned fcount = 0, bcount = 0;
    if (!enif_get_list_length(env, a[2], &fcount) ||
        !enif_get_list_length(env, a[3], &bcount) || fcount != bcount) {
        enif_free(html); enif_free(css);
        return enif_make_badarg(env);
    }
    char **families = NULL;
    uint8_t **font_bytes = NULL;
    uintptr_t *font_lens = NULL;
    if (fcount > 0) {
        families = enif_alloc(fcount * sizeof(char *));
        font_bytes = enif_alloc(fcount * sizeof(uint8_t *));
        font_lens = enif_alloc(fcount * sizeof(uintptr_t));
        if (!families || !font_bytes || !font_lens) {
            enif_free(families); enif_free(font_bytes); enif_free(font_lens);
            enif_free(html); enif_free(css);
            return enif_make_badarg(env);
        }
        ERL_NIF_TERM flist = a[2], blist = a[3], fhead, bhead;
        unsigned i = 0;
        while (enif_get_list_cell(env, flist, &fhead, &flist) &&
               enif_get_list_cell(env, blist, &bhead, &blist)) {
            char *fam = term_to_cstr(env, fhead);
            ErlNifBinary fb;
            if (!fam || !enif_inspect_binary(env, bhead, &fb)) {
                enif_free(fam);
                for (unsigned k = 0; k < i; k++) { enif_free(families[k]); enif_free(font_bytes[k]); }
                enif_free(families); enif_free(font_bytes); enif_free(font_lens);
                enif_free(html); enif_free(css);
                return enif_make_badarg(env);
            }
            /* Copy the font bytes into a stable buffer the FFI can read. */
            uint8_t *copy = enif_alloc(fb.size ? fb.size : 1);
            if (fb.size) memcpy(copy, fb.data, fb.size);
            families[i] = fam;
            font_bytes[i] = copy;
            font_lens[i] = (uintptr_t)fb.size;
            i++;
        }
    }
    int32_t code = 0;
    Pdf *h = pdf_from_html_css_with_fonts(html, css, (const char *const *)families,
                                          (const uint8_t *const *)font_bytes,
                                          font_lens, (uintptr_t)fcount, &code);
    if (fcount > 0) {
        for (unsigned k = 0; k < fcount; k++) { enif_free(families[k]); enif_free(font_bytes[k]); }
        enif_free(families); enif_free(font_bytes); enif_free(font_lens);
    }
    enif_free(html); enif_free(css);
    return h ? make_pdf(env, h) : err_tuple(env, code);
}

static ERL_NIF_TERM pdf_merge_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    unsigned count = 0;
    if (!enif_get_list_length(env, a[0], &count)) return enif_make_badarg(env);
    char **paths = NULL;
    if (count > 0) {
        paths = enif_alloc(count * sizeof(char *));
        if (!paths) return enif_make_badarg(env);
        ERL_NIF_TERM list = a[0], head;
        unsigned i = 0;
        while (enif_get_list_cell(env, list, &head, &list)) {
            char *s = term_to_cstr(env, head);
            if (!s) {
                for (unsigned k = 0; k < i; k++) enif_free(paths[k]);
                enif_free(paths);
                return enif_make_badarg(env);
            }
            paths[i++] = s;
        }
    }
    int32_t data_len = 0, code = 0;
    uint8_t *p = pdf_merge((const char *const *)paths, (int32_t)count, &data_len, &code);
    if (paths) {
        for (unsigned k = 0; k < count; k++) enif_free(paths[k]);
        enif_free(paths);
    }
    size_t n = (p && data_len > 0) ? (size_t)data_len : 0;
    return ok_bytes(env, p, n, code);
}

/* ── page getters (on a Document) ─────────────────────────────────────────── */
static ERL_NIF_TERM page_get_width(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    float v = pdf_page_get_width(r->h, page, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_double(env, v));
}

static ERL_NIF_TERM page_get_height(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    float v = pdf_page_get_height(r->h, page, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_double(env, v));
}

static ERL_NIF_TERM page_get_rotation(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    int32_t v = pdf_page_get_rotation(r->h, page, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}

/* Return an opaque element-list handle (ELEMS_RES, freed via *_close or GC). */
static ERL_NIF_TERM page_get_elements(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    GET_DOC_PAGE
    int32_t code = 0;
    FfiElementList *h = pdf_page_get_elements(r->h, page, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(ELEMS_RES, ElemsRes, h);
}

static ERL_NIF_TERM elements_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t n = pdf_oxide_element_count(r->h);
    if (n < 0) n = 0;
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, n));
}

static ERL_NIF_TERM elements_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r)) return enif_make_badarg(env);
    if (r->h) { pdf_oxide_elements_free(r->h); r->h = NULL; }
    return enif_make_atom(env, "ok");
}

/* ── timestamp ────────────────────────────────────────────────────────────── */
static ERL_NIF_TERM add_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary pdf;
    int sig_index;
    if (!enif_inspect_binary(env, a[0], &pdf) || !enif_get_int(env, a[1], &sig_index))
        return enif_make_badarg(env);
    char *tsa_url = term_to_cstr(env, a[2]);
    if (!tsa_url) return enif_make_badarg(env);
    uint8_t *out_data = NULL;
    uintptr_t out_len = 0;
    int32_t code = 0;
    bool ok = pdf_add_timestamp(pdf.data, (uintptr_t)pdf.size, sig_index, tsa_url,
                                &out_data, &out_len, &code);
    enif_free(tsa_url);
    if (!ok) { if (out_data) free_bytes(out_data); return err_tuple(env, code); }
    return ok_bytes(env, out_data, (size_t)out_len, code);
}

/* ── phase 8: final coverage — office I/O, in-rect/auto extraction, header/
 * footer/artifact erase, forms, doc structure/metadata, doc-level signatures,
 * annotation/element/JSON accessors, crypto/FIPS, models/config, PDF/A. The
 * list-shaped accessors (form fields, fonts/annotations/elements/search JSON,
 * per-annotation extras) re-extract the owning native list, read from it, then
 * free it via its matching *_free — matching the inline list-then-free pattern
 * the phase-1/2 extractors already use, so no new long-lived resource type is
 * needed. */

/* A document-only handle fetch (no page arg). */
#define GET_DOC                                                                  \
    DocRes *r;                                                                   \
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r))                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

/* GET_DOC plus a 0-based page int in a[1] (own name to avoid GET_DOC_PAGE's
 * `page` var when both are not needed). */
#define GET_DOC_AND_PAGE                                                         \
    DocRes *r;                                                                   \
    int page_idx;                                                                \
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||                   \
        !enif_get_int(env, a[1], &page_idx))                                     \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

/* ── office: open from docx/pptx/xlsx bytes ───────────────────────────────── */
static ERL_NIF_TERM doc_open_from_docx_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_docx_bytes(bin.data, bin.size, &code);
    return h ? make_doc(env, h) : err_tuple(env, code);
}
static ERL_NIF_TERM doc_open_from_pptx_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_pptx_bytes(bin.data, bin.size, &code);
    return h ? make_doc(env, h) : err_tuple(env, code);
}
static ERL_NIF_TERM doc_open_from_xlsx_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, a[0], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_xlsx_bytes(bin.data, bin.size, &code);
    return h ? make_doc(env, h) : err_tuple(env, code);
}

/* ── office: export to docx/pptx/xlsx bytes ───────────────────────────────── */
static ERL_NIF_TERM doc_to_docx(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_to_docx(r->h, &out_len, &code);
    return ok_bytes(env, p, (size_t)out_len, code);
}
static ERL_NIF_TERM doc_to_pptx(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_to_pptx(r->h, &out_len, &code);
    return ok_bytes(env, p, (size_t)out_len, code);
}
static ERL_NIF_TERM doc_to_xlsx(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_to_xlsx(r->h, &out_len, &code);
    return ok_bytes(env, p, (size_t)out_len, code);
}

/* ── in-rect extractors ───────────────────────────────────────────────────── */
#define GET_DOC_RECT                                                             \
    DocRes *r;                                                                   \
    int page_idx;                                                                \
    double x, y, w, h;                                                           \
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||                   \
        !enif_get_int(env, a[1], &page_idx) ||                                   \
        !enif_get_double(env, a[2], &x) || !enif_get_double(env, a[3], &y) ||    \
        !enif_get_double(env, a[4], &w) || !enif_get_double(env, a[5], &h))      \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);

static ERL_NIF_TERM doc_extract_text_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_RECT
    int32_t code = 0;
    return ok_string(env, pdf_document_extract_text_in_rect(r->h, page_idx,
                          (float)x, (float)y, (float)w, (float)h, &code), code);
}
static ERL_NIF_TERM doc_extract_words_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_RECT
    int32_t code = 0;
    FfiWordList *list = pdf_document_extract_words_in_rect(r->h, page_idx,
                            (float)x, (float)y, (float)w, (float)h, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_word_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM text = take_string(env, pdf_oxide_word_get_text(list, i, &c));
        float bx = 0, by = 0, bw = 0, bh = 0;
        pdf_oxide_word_get_bbox(list, i, &bx, &by, &bw, &bh, &c);
        ERL_NIF_TERM font = take_string(env, pdf_oxide_word_get_font_name(list, i, &c));
        float size = pdf_oxide_word_get_font_size(list, i, &c);
        bool bold = pdf_oxide_word_is_bold(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple(env, 8, text,
                                            enif_make_double(env, bx), enif_make_double(env, by),
                                            enif_make_double(env, bw), enif_make_double(env, bh),
                                            font, enif_make_double(env, size),
                                            enif_make_atom(env, bold ? "true" : "false"));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_word_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}
static ERL_NIF_TERM doc_extract_lines_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_RECT
    int32_t code = 0;
    FfiTextLineList *list = pdf_document_extract_lines_in_rect(r->h, page_idx,
                                (float)x, (float)y, (float)w, (float)h, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_line_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM text = take_string(env, pdf_oxide_line_get_text(list, i, &c));
        float bx = 0, by = 0, bw = 0, bh = 0;
        pdf_oxide_line_get_bbox(list, i, &bx, &by, &bw, &bh, &c);
        int32_t wc = pdf_oxide_line_get_word_count(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple6(env, text,
                                             enif_make_double(env, bx), enif_make_double(env, by),
                                             enif_make_double(env, bw), enif_make_double(env, bh),
                                             enif_make_int(env, wc));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_line_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}
static ERL_NIF_TERM doc_extract_tables_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_RECT
    int32_t code = 0;
    FfiTableList *list = pdf_document_extract_tables_in_rect(r->h, page_idx,
                             (float)x, (float)y, (float)w, (float)h, &code);
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
static ERL_NIF_TERM doc_extract_images_in_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_RECT
    int32_t code = 0;
    FfiImageList *list = pdf_document_extract_images_in_rect(r->h, page_idx,
                             (float)x, (float)y, (float)w, (float)h, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_image_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        int32_t iw = pdf_oxide_image_get_width(list, i, &c);
        int32_t ih = pdf_oxide_image_get_height(list, i, &c);
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
        ERL_NIF_TERM item = enif_make_tuple6(env, enif_make_int(env, iw),
                                             enif_make_int(env, ih),
                                             enif_make_int(env, bpc),
                                             format, colorspace, data);
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_image_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}

/* ── auto extraction / classification ─────────────────────────────────────── */
static ERL_NIF_TERM doc_extract_text_auto(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t code = 0;
    return ok_string(env, pdf_document_extract_text_auto(r->h, page_idx, &code), code);
}
static ERL_NIF_TERM doc_extract_all_text(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    return ok_string(env, pdf_document_extract_all_text(r->h, &code), code);
}
static ERL_NIF_TERM doc_extract_page_auto(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    char *opts = term_to_cstr(env, a[2]);
    if (!opts) return enif_make_badarg(env);
    int32_t code = 0;
    char *s = pdf_document_extract_page_auto(r->h, page_idx,
                                             opts[0] == '\0' ? NULL : opts, &code);
    enif_free(opts);
    return ok_string(env, s, code);
}
static ERL_NIF_TERM doc_classify_page(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t code = 0;
    return ok_string(env, pdf_document_classify_page(r->h, page_idx, &code), code);
}
static ERL_NIF_TERM doc_classify_document(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    return ok_string(env, pdf_document_classify_document(r->h, &code), code);
}

/* ── header / footer / artifact erase + remove ────────────────────────────── */
static ERL_NIF_TERM doc_erase_header(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t code = 0;
    int32_t v = pdf_document_erase_header(r->h, page_idx, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_erase_footer(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t code = 0;
    int32_t v = pdf_document_erase_footer(r->h, page_idx, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_erase_artifacts(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t code = 0;
    int32_t v = pdf_document_erase_artifacts(r->h, page_idx, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_remove_headers(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    double threshold;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_double(env, a[1], &threshold))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_document_remove_headers(r->h, (float)threshold, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_remove_footers(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    double threshold;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_double(env, a[1], &threshold))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_document_remove_footers(r->h, (float)threshold, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_remove_artifacts(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    double threshold;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_double(env, a[1], &threshold))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_document_remove_artifacts(r->h, (float)threshold, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}

/* ── forms ────────────────────────────────────────────────────────────────── */
/* Re-extract the form-field list, marshal every field into a list of tuples,
 * then free the list. An empty document yields an empty list (not an error). */
static ERL_NIF_TERM doc_get_form_fields(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    FfiFormFieldList *list = pdf_document_get_form_fields(r->h, &code);
    if (!list) return err_tuple(env, code);
    int32_t n = pdf_oxide_form_field_count(list);
    if (n < 0) n = 0;
    ERL_NIF_TERM items = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t c = 0;
        ERL_NIF_TERM name = take_string(env, pdf_oxide_form_field_get_name(list, i, &c));
        ERL_NIF_TERM value = take_string(env, pdf_oxide_form_field_get_value(list, i, &c));
        ERL_NIF_TERM type = take_string(env, pdf_oxide_form_field_get_type(list, i, &c));
        bool ro = pdf_oxide_form_field_is_readonly(list, i, &c);
        bool req = pdf_oxide_form_field_is_required(list, i, &c);
        ERL_NIF_TERM item = enif_make_tuple5(env, name, value, type,
                                             enif_make_atom(env, ro ? "true" : "false"),
                                             enif_make_atom(env, req ? "true" : "false"));
        items = enif_make_list_cell(env, item, items);
    }
    pdf_oxide_form_field_list_free(list);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), items);
}
static ERL_NIF_TERM doc_export_form_data_to_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int format_type;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &format_type))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_export_form_data_to_bytes(r->h, format_type, &out_len, &code);
    return ok_bytes(env, p, (size_t)out_len, code);
}
static ERL_NIF_TERM doc_import_form_data(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    char *data_path = term_to_cstr(env, a[1]);
    if (!data_path) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_document_import_form_data(r->h, data_path, &code);
    enif_free(data_path);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM editor_import_fdf_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    EditRes *r;
    ErlNifBinary bin;
    if (!enif_get_resource(env, a[0], EDIT_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, a[1], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_editor_import_fdf_bytes(r->h, bin.data, bin.size, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM editor_import_xfdf_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    EditRes *r;
    ErlNifBinary bin;
    if (!enif_get_resource(env, a[0], EDIT_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, a[1], &bin)) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_editor_import_xfdf_bytes(r->h, bin.data, bin.size, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM form_import_from_file(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    char *filename = term_to_cstr(env, a[1]);
    if (!filename) return enif_make_badarg(env);
    int32_t code = 0;
    bool ok = pdf_form_import_from_file(r->h, filename, &code);
    enif_free(filename);
    if (!ok) return err_tuple(env, code);
    return enif_make_atom(env, "ok");
}

/* ── document structure / metadata ────────────────────────────────────────── */
static ERL_NIF_TERM doc_get_outline(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    return ok_string(env, pdf_document_get_outline(r->h, &code), code);
}
static ERL_NIF_TERM doc_get_page_labels(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    return ok_string(env, pdf_document_get_page_labels(r->h, &code), code);
}
static ERL_NIF_TERM doc_get_xmp_metadata(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    return ok_string(env, pdf_document_get_xmp_metadata(r->h, &code), code);
}
static ERL_NIF_TERM doc_get_source_bytes(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    uintptr_t out_len = 0;
    int32_t code = 0;
    uint8_t *p = pdf_document_get_source_bytes(r->h, &out_len, &code);
    return ok_bytes(env, p, (size_t)out_len, code);
}
static ERL_NIF_TERM doc_has_xfa(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    bool b = pdf_document_has_xfa(r->h);
    return enif_make_atom(env, b ? "true" : "false");
}
static ERL_NIF_TERM doc_get_page_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    PdfRes *r;
    if (!enif_get_resource(env, a[0], PDF_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    int32_t v = pdf_get_page_count(r->h, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_plan_split_by_bookmarks(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    char *opts = term_to_cstr(env, a[1]);
    if (!opts) return enif_make_badarg(env);
    int32_t code = 0;
    char *s = pdf_document_plan_split_by_bookmarks(r->h, opts[0] == '\0' ? NULL : opts, &code);
    enif_free(opts);
    return ok_string(env, s, code);
}

/* ── document-level signatures ────────────────────────────────────────────── */
static ERL_NIF_TERM doc_sign(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    CertRes *c;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    if (!enif_get_resource(env, a[1], CERT_RES, (void **)&c)) return enif_make_badarg(env);
    if (!c->h) return enif_make_badarg(env);
    char *reason = term_to_cstr(env, a[2]);
    char *location = term_to_cstr(env, a[3]);
    if (!reason || !location) { enif_free(reason); enif_free(location); return enif_make_badarg(env); }
    int32_t code = 0;
    int32_t v = pdf_document_sign(r->h, c->h,
                                  reason[0] == '\0' ? NULL : reason,
                                  location[0] == '\0' ? NULL : location, &code);
    enif_free(reason); enif_free(location);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_get_signature_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    int32_t v = pdf_document_get_signature_count(r->h, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_get_signature(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int index;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    void *h = pdf_document_get_signature(r->h, index, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(SIG_RES, SigRes, (FfiSignatureInfo *)h);
}
static ERL_NIF_TERM doc_verify_all_signatures(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    int32_t v = pdf_document_verify_all_signatures(r->h, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM doc_has_timestamp(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    int32_t v = pdf_document_has_timestamp(r->h, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, v != 0 ? "true" : "false"));
}
static ERL_NIF_TERM doc_get_dss(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    int32_t code = 0;
    void *h = pdf_document_get_dss(r->h, &code);
    if (!h) return err_tuple(env, code);
    MAKE_HANDLE(DSS_RES, DssRes, h);
}

/* ── annotation extras ────────────────────────────────────────────────────── */
/* Re-extract the page's annotation list, read the per-index accessor, then free
 * it. index out of range / wrong subtype surfaces as {:error, code}. */
#define ANNOT_ACCESSOR_PROLOGUE                                                  \
    DocRes *r;                                                                   \
    int page_idx, index;                                                         \
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||                   \
        !enif_get_int(env, a[1], &page_idx) ||                                   \
        !enif_get_int(env, a[2], &index))                                        \
        return enif_make_badarg(env);                                            \
    if (!r->h) return enif_make_badarg(env);                                     \
    int32_t lcode = 0;                                                           \
    FfiAnnotationList *list = pdf_document_get_page_annotations(r->h, page_idx, &lcode); \
    if (!list) return err_tuple(env, lcode);

static ERL_NIF_TERM annot_get_color(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    uint32_t v = pdf_oxide_annotation_get_color(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_uint(env, v));
}
static ERL_NIF_TERM annot_get_creation_date(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    int64_t v = pdf_oxide_annotation_get_creation_date(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int64(env, v));
}
static ERL_NIF_TERM annot_get_modification_date(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    int64_t v = pdf_oxide_annotation_get_modification_date(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int64(env, v));
}
static ERL_NIF_TERM annot_is_hidden(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    bool v = pdf_oxide_annotation_is_hidden(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, v ? "true" : "false"));
}
static ERL_NIF_TERM annot_is_marked_deleted(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    bool v = pdf_oxide_annotation_is_marked_deleted(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, v ? "true" : "false"));
}
static ERL_NIF_TERM annot_is_printable(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    bool v = pdf_oxide_annotation_is_printable(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, v ? "true" : "false"));
}
static ERL_NIF_TERM annot_is_read_only(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    bool v = pdf_oxide_annotation_is_read_only(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, v ? "true" : "false"));
}
static ERL_NIF_TERM annot_link_get_uri(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    char *s = pdf_oxide_link_annotation_get_uri(list, index, &code);
    ERL_NIF_TERM res = ok_string(env, s, code);
    pdf_oxide_annotation_list_free(list);
    return res;
}
static ERL_NIF_TERM annot_text_get_icon_name(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    char *s = pdf_oxide_text_annotation_get_icon_name(list, index, &code);
    ERL_NIF_TERM res = ok_string(env, s, code);
    pdf_oxide_annotation_list_free(list);
    return res;
}
static ERL_NIF_TERM annot_highlight_quad_points_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; ANNOT_ACCESSOR_PROLOGUE
    int32_t code = 0;
    int32_t v = pdf_oxide_highlight_annotation_get_quad_points_count(list, index, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_int(env, v));
}
static ERL_NIF_TERM annot_highlight_quad_point(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_idx, index, quad_index;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_idx) ||
        !enif_get_int(env, a[2], &index) ||
        !enif_get_int(env, a[3], &quad_index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t lcode = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(r->h, page_idx, &lcode);
    if (!list) return err_tuple(env, lcode);
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0, x3 = 0, y3 = 0, x4 = 0, y4 = 0;
    int32_t code = 0;
    pdf_oxide_highlight_annotation_get_quad_point(list, index, quad_index,
                                                  &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &code);
    pdf_oxide_annotation_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_tuple(env, 8,
                                            enif_make_double(env, x1), enif_make_double(env, y1),
                                            enif_make_double(env, x2), enif_make_double(env, y2),
                                            enif_make_double(env, x3), enif_make_double(env, y3),
                                            enif_make_double(env, x4), enif_make_double(env, y4)));
}
static ERL_NIF_TERM annotations_to_json(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t lcode = 0;
    FfiAnnotationList *list = pdf_document_get_page_annotations(r->h, page_idx, &lcode);
    if (!list) return err_tuple(env, lcode);
    int32_t code = 0;
    char *s = pdf_oxide_annotations_to_json(list, &code);
    ERL_NIF_TERM res = ok_string(env, s, code);
    pdf_oxide_annotation_list_free(list);
    return res;
}

/* ── element / font / search JSON accessors ───────────────────────────────── */
static ERL_NIF_TERM element_get_type(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    int index;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_oxide_element_get_type(r->h, index, &code), code);
}
static ERL_NIF_TERM element_get_text(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    int index;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_oxide_element_get_text(r->h, index, &code), code);
}
static ERL_NIF_TERM element_get_rect(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    int index;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    float x = 0, y = 0, w = 0, h = 0;
    int32_t code = 0;
    pdf_oxide_element_get_rect(r->h, index, &x, &y, &w, &h, &code);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_tuple4(env, enif_make_double(env, x), enif_make_double(env, y),
                                             enif_make_double(env, w), enif_make_double(env, h)));
}
static ERL_NIF_TERM elements_to_json(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ElemsRes *r;
    if (!enif_get_resource(env, a[0], ELEMS_RES, (void **)&r)) return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    return ok_string(env, pdf_oxide_elements_to_json(r->h, &code), code);
}
/* Font size + fonts JSON re-extract the page's font list, read, then free. */
static ERL_NIF_TERM font_get_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int page_idx, index;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &page_idx) ||
        !enif_get_int(env, a[2], &index))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t lcode = 0;
    FfiFontList *list = pdf_document_get_embedded_fonts(r->h, page_idx, &lcode);
    if (!list) return err_tuple(env, lcode);
    int32_t code = 0;
    float v = pdf_oxide_font_get_size(list, index, &code);
    pdf_oxide_font_list_free(list);
    if (code != 0) return err_tuple(env, code);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_double(env, v));
}
static ERL_NIF_TERM fonts_to_json(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC_AND_PAGE
    int32_t lcode = 0;
    FfiFontList *list = pdf_document_get_embedded_fonts(r->h, page_idx, &lcode);
    if (!list) return err_tuple(env, lcode);
    int32_t code = 0;
    char *s = pdf_oxide_fonts_to_json(list, &code);
    ERL_NIF_TERM res = ok_string(env, s, code);
    pdf_oxide_font_list_free(list);
    return res;
}
static ERL_NIF_TERM search_results_to_json(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; GET_DOC
    char *term = term_to_cstr(env, a[1]);
    if (!term) return enif_make_badarg(env);
    int cs;
    if (!enif_get_int(env, a[2], &cs)) { enif_free(term); return enif_make_badarg(env); }
    int32_t lcode = 0;
    FfiSearchResults *list = pdf_document_search_all(r->h, term, cs != 0, &lcode);
    enif_free(term);
    if (!list) return err_tuple(env, lcode);
    int32_t code = 0;
    char *s = pdf_oxide_search_results_to_json(list, &code);
    ERL_NIF_TERM res = ok_string(env, s, code);
    pdf_oxide_search_result_free(list);
    return res;
}

/* ── crypto / FIPS ────────────────────────────────────────────────────────── */
static ERL_NIF_TERM crypto_active_provider(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return ok_string(env, pdf_oxide_crypto_active_provider(), -1);
}
static ERL_NIF_TERM crypto_cbom(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return ok_string(env, pdf_oxide_crypto_cbom(), -1);
}
static ERL_NIF_TERM crypto_inventory(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return ok_string(env, pdf_oxide_crypto_inventory(), -1);
}
static ERL_NIF_TERM crypto_policy(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return ok_string(env, pdf_oxide_crypto_policy(), -1);
}
static ERL_NIF_TERM crypto_fips_available(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return enif_make_int(env, pdf_oxide_crypto_fips_available());
}
static ERL_NIF_TERM crypto_use_fips(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return enif_make_int(env, pdf_oxide_crypto_use_fips());
}
static ERL_NIF_TERM crypto_set_policy(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *spec = term_to_cstr(env, a[0]);
    if (!spec) return enif_make_badarg(env);
    int32_t v = pdf_oxide_crypto_set_policy(spec);
    enif_free(spec);
    return enif_make_int(env, v);
}

/* ── models / config ──────────────────────────────────────────────────────── */
static ERL_NIF_TERM model_manifest(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return ok_string(env, pdf_oxide_model_manifest(), -1);
}
static ERL_NIF_TERM prefetch_available(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc; (void)a;
    return enif_make_int(env, pdf_oxide_prefetch_available());
}
static ERL_NIF_TERM prefetch_models(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    char *langs = term_to_cstr(env, a[0]);
    if (!langs) return enif_make_badarg(env);
    int32_t code = 0;
    char *s = pdf_oxide_prefetch_models(langs[0] == '\0' ? NULL : langs, &code);
    enif_free(langs);
    return ok_string(env, s, code);
}
static ERL_NIF_TERM set_max_ops_per_stream(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    ErlNifSInt64 limit;
    if (!enif_get_int64(env, a[0], &limit)) return enif_make_badarg(env);
    return enif_make_int64(env, pdf_oxide_set_max_ops_per_stream((int64_t)limit));
}
static ERL_NIF_TERM set_preserve_unmapped_glyphs(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    int preserve;
    if (!enif_get_int(env, a[0], &preserve)) return enif_make_badarg(env);
    return enif_make_int(env, pdf_oxide_set_preserve_unmapped_glyphs(preserve));
}

/* ── PDF/A conversion (on a Document) ─────────────────────────────────────── */
static ERL_NIF_TERM doc_convert_to_pdf_a(ErlNifEnv *env, int argc, const ERL_NIF_TERM a[]) {
    (void)argc;
    DocRes *r;
    int level;
    if (!enif_get_resource(env, a[0], DOC_RES, (void **)&r) ||
        !enif_get_int(env, a[1], &level))
        return enif_make_badarg(env);
    if (!r->h) return enif_make_badarg(env);
    int32_t code = 0;
    bool ok = pdf_convert_to_pdf_a(r->h, level, &code);
    if (!ok) return err_tuple(env, code);
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
    {"editor_open", 1, editor_open, DIRTY},
    {"editor_open_bytes", 1, editor_open_bytes, DIRTY},
    {"editor_is_modified", 1, editor_is_modified, 0},
    {"editor_source_path", 1, editor_source_path, 0},
    {"editor_version", 1, editor_version, 0},
    {"editor_page_count", 1, editor_page_count, 0},
    {"editor_get_producer", 1, editor_get_producer, 0},
    {"editor_set_producer", 2, editor_set_producer, 0},
    {"editor_get_creation_date", 1, editor_get_creation_date, 0},
    {"editor_set_creation_date", 2, editor_set_creation_date, 0},
    {"editor_save", 2, editor_save, DIRTY},
    {"editor_save_to_bytes", 1, editor_save_to_bytes, DIRTY},
    {"editor_save_to_bytes_with_options", 4, editor_save_to_bytes_with_options, DIRTY},
    {"editor_extract_pages_to_bytes", 2, editor_extract_pages_to_bytes, DIRTY},
    {"editor_convert_to_pdf_a", 2, editor_convert_to_pdf_a, DIRTY},
    {"editor_save_encrypted_to_bytes", 3, editor_save_encrypted_to_bytes, DIRTY},
    {"editor_merge_from_bytes", 2, editor_merge_from_bytes, DIRTY},
    {"editor_embed_file", 3, editor_embed_file, DIRTY},
    {"editor_apply_page_redactions", 2, editor_apply_page_redactions, DIRTY},
    {"editor_apply_all_redactions", 1, editor_apply_all_redactions, DIRTY},
    {"editor_rotate_all_pages", 2, editor_rotate_all_pages, DIRTY},
    {"editor_rotate_page_by", 3, editor_rotate_page_by, DIRTY},
    {"editor_get_media_box", 2, editor_get_media_box, 0},
    {"editor_set_media_box", 6, editor_set_media_box, 0},
    {"editor_get_crop_box", 2, editor_get_crop_box, 0},
    {"editor_set_crop_box", 6, editor_set_crop_box, 0},
    {"editor_erase_regions", 3, editor_erase_regions, DIRTY},
    {"editor_clear_erase_regions", 2, editor_clear_erase_regions, 0},
    {"editor_is_marked_for_flatten", 2, editor_is_marked_for_flatten, 0},
    {"editor_unmark_for_flatten", 2, editor_unmark_for_flatten, 0},
    {"editor_is_marked_for_redaction", 2, editor_is_marked_for_redaction, 0},
    {"editor_unmark_for_redaction", 2, editor_unmark_for_redaction, 0},
    {"editor_delete_page", 2, editor_delete_page, DIRTY},
    {"editor_move_page", 3, editor_move_page, DIRTY},
    {"editor_get_page_rotation", 2, editor_get_page_rotation, 0},
    {"editor_set_page_rotation", 3, editor_set_page_rotation, DIRTY},
    {"editor_erase_region", 6, editor_erase_region, DIRTY},
    {"editor_flatten_annotations", 2, editor_flatten_annotations, DIRTY},
    {"editor_flatten_all_annotations", 1, editor_flatten_all_annotations, DIRTY},
    {"editor_crop_margins", 5, editor_crop_margins, DIRTY},
    {"editor_merge_from", 2, editor_merge_from, DIRTY},
    {"editor_save_encrypted", 4, editor_save_encrypted, DIRTY},
    {"editor_set_form_field_value", 3, editor_set_form_field_value, DIRTY},
    {"editor_flatten_forms", 1, editor_flatten_forms, DIRTY},
    {"editor_flatten_forms_on_page", 2, editor_flatten_forms_on_page, DIRTY},
    {"editor_flatten_warnings_count", 1, editor_flatten_warnings_count, 0},
    {"editor_flatten_warning", 2, editor_flatten_warning, 0},
    {"editor_close", 1, editor_close, 0},
    /* PDF creation builder — embedded font */
    {"font_from_file", 1, font_from_file, DIRTY},
    {"font_from_bytes", 2, font_from_bytes, DIRTY},
    {"font_close", 1, font_close, 0},
    /* PDF creation builder — document builder */
    {"dbld_create", 0, dbld_create, 0},
    {"dbld_set_title", 2, dbld_set_title, 0},
    {"dbld_set_author", 2, dbld_set_author, 0},
    {"dbld_set_subject", 2, dbld_set_subject, 0},
    {"dbld_set_keywords", 2, dbld_set_keywords, 0},
    {"dbld_set_creator", 2, dbld_set_creator, 0},
    {"dbld_on_open", 2, dbld_on_open, 0},
    {"dbld_language", 2, dbld_language, 0},
    {"dbld_tagged_pdf_ua1", 1, dbld_tagged_pdf_ua1, 0},
    {"dbld_role_map", 3, dbld_role_map, 0},
    {"dbld_register_embedded_font", 3, dbld_register_embedded_font, DIRTY},
    {"dbld_a4_page", 1, dbld_a4_page, 0},
    {"dbld_letter_page", 1, dbld_letter_page, 0},
    {"dbld_page", 3, dbld_page, 0},
    {"dbld_build", 1, dbld_build, DIRTY},
    {"dbld_save", 2, dbld_save, DIRTY},
    {"dbld_save_encrypted", 4, dbld_save_encrypted, DIRTY},
    {"dbld_to_bytes_encrypted", 3, dbld_to_bytes_encrypted, DIRTY},
    {"dbld_close", 1, dbld_close, 0},
    /* PDF creation builder — page builder */
    {"pbld_font", 3, pbld_font, 0},
    {"pbld_at", 3, pbld_at, 0},
    {"pbld_text", 2, pbld_text, 0},
    {"pbld_heading", 3, pbld_heading, 0},
    {"pbld_paragraph", 2, pbld_paragraph, 0},
    {"pbld_space", 2, pbld_space, 0},
    {"pbld_horizontal_rule", 1, pbld_horizontal_rule, 0},
    {"pbld_link_url", 2, pbld_link_url, 0},
    {"pbld_link_page", 2, pbld_link_page, 0},
    {"pbld_link_named", 2, pbld_link_named, 0},
    {"pbld_link_javascript", 2, pbld_link_javascript, 0},
    {"pbld_on_open", 2, pbld_on_open, 0},
    {"pbld_on_close", 2, pbld_on_close, 0},
    {"pbld_field_keystroke", 2, pbld_field_keystroke, 0},
    {"pbld_field_format", 2, pbld_field_format, 0},
    {"pbld_field_validate", 2, pbld_field_validate, 0},
    {"pbld_field_calculate", 2, pbld_field_calculate, 0},
    {"pbld_highlight", 4, pbld_highlight, 0},
    {"pbld_underline", 4, pbld_underline, 0},
    {"pbld_strikeout", 4, pbld_strikeout, 0},
    {"pbld_squiggly", 4, pbld_squiggly, 0},
    {"pbld_sticky_note", 2, pbld_sticky_note, 0},
    {"pbld_sticky_note_at", 4, pbld_sticky_note_at, 0},
    {"pbld_watermark", 2, pbld_watermark, 0},
    {"pbld_watermark_confidential", 1, pbld_watermark_confidential, 0},
    {"pbld_watermark_draft", 1, pbld_watermark_draft, 0},
    {"pbld_stamp", 2, pbld_stamp, 0},
    {"pbld_freetext", 6, pbld_freetext, 0},
    {"pbld_text_field", 7, pbld_text_field, 0},
    {"pbld_checkbox", 7, pbld_checkbox, 0},
    {"pbld_combo_box", 8, pbld_combo_box, 0},
    {"pbld_radio_group", 8, pbld_radio_group, 0},
    {"pbld_push_button", 7, pbld_push_button, 0},
    {"pbld_signature_field", 6, pbld_signature_field, 0},
    {"pbld_footnote", 3, pbld_footnote, 0},
    {"pbld_columns", 4, pbld_columns, 0},
    {"pbld_inline", 2, pbld_inline, 0},
    {"pbld_inline_bold", 2, pbld_inline_bold, 0},
    {"pbld_inline_italic", 2, pbld_inline_italic, 0},
    {"pbld_inline_color", 5, pbld_inline_color, 0},
    {"pbld_newline", 1, pbld_newline, 0},
    {"pbld_barcode_1d", 7, pbld_barcode_1d, 0},
    {"pbld_barcode_qr", 5, pbld_barcode_qr, 0},
    {"pbld_image", 6, pbld_image, DIRTY},
    {"pbld_image_with_alt", 7, pbld_image_with_alt, DIRTY},
    {"pbld_image_artifact", 6, pbld_image_artifact, DIRTY},
    {"pbld_rect", 5, pbld_rect, 0},
    {"pbld_filled_rect", 8, pbld_filled_rect, 0},
    {"pbld_line", 5, pbld_line, 0},
    {"pbld_stroke_rect", 9, pbld_stroke_rect, 0},
    {"pbld_stroke_line", 9, pbld_stroke_line, 0},
    {"pbld_stroke_rect_dashed", 11, pbld_stroke_rect_dashed, 0},
    {"pbld_stroke_line_dashed", 11, pbld_stroke_line_dashed, 0},
    {"pbld_text_in_rect", 7, pbld_text_in_rect, 0},
    {"pbld_new_page_same_size", 1, pbld_new_page_same_size, 0},
    {"pbld_table", 7, pbld_table, DIRTY},
    {"pbld_streaming_table_begin", 6, pbld_streaming_table_begin, 0},
    {"pbld_streaming_table_begin_v2", 11, pbld_streaming_table_begin_v2, 0},
    {"pbld_streaming_table_set_batch_size", 2, pbld_streaming_table_set_batch_size, 0},
    {"pbld_streaming_table_pending_row_count", 1, pbld_streaming_table_pending_row_count, 0},
    {"pbld_streaming_table_batch_count", 1, pbld_streaming_table_batch_count, 0},
    {"pbld_streaming_table_push_row", 2, pbld_streaming_table_push_row, 0},
    {"pbld_streaming_table_push_row_v2", 3, pbld_streaming_table_push_row_v2, 0},
    {"pbld_streaming_table_flush", 1, pbld_streaming_table_flush, 0},
    {"pbld_streaming_table_finish", 1, pbld_streaming_table_finish, 0},
    {"pbld_done", 1, pbld_done, DIRTY},
    {"pbld_close", 1, pbld_close, 0},
    /* phase 6 — certificate */
    {"cert_load_from_bytes", 2, cert_load_from_bytes, DIRTY},
    {"cert_load_from_pem", 2, cert_load_from_pem, DIRTY},
    {"cert_get_subject", 1, cert_get_subject, 0},
    {"cert_get_issuer", 1, cert_get_issuer, 0},
    {"cert_get_serial", 1, cert_get_serial, 0},
    {"cert_get_validity", 1, cert_get_validity, 0},
    {"cert_is_valid", 1, cert_is_valid, 0},
    {"cert_close", 1, cert_close, 0},
    /* phase 6 — signing */
    {"sign_bytes", 4, sign_bytes, DIRTY},
    {"sign_bytes_pades", 9, sign_bytes_pades, DIRTY},
    {"sign_bytes_pades_opts", 9, sign_bytes_pades_opts, DIRTY},
    /* phase 6 — signature info */
    {"sig_get_signer_name", 1, sig_get_signer_name, 0},
    {"sig_get_signing_reason", 1, sig_get_signing_reason, 0},
    {"sig_get_signing_location", 1, sig_get_signing_location, 0},
    {"sig_get_signing_time", 1, sig_get_signing_time, 0},
    {"sig_get_certificate", 1, sig_get_certificate, 0},
    {"sig_get_pades_level", 1, sig_get_pades_level, 0},
    {"sig_has_timestamp", 1, sig_has_timestamp, 0},
    {"sig_get_timestamp", 1, sig_get_timestamp, 0},
    {"sig_add_timestamp", 2, sig_add_timestamp, 0},
    {"sig_verify", 1, sig_verify, DIRTY},
    {"sig_verify_detached", 2, sig_verify_detached, DIRTY},
    {"sig_close", 1, sig_close, 0},
    /* phase 6 — timestamp */
    {"ts_parse", 1, ts_parse, DIRTY},
    {"ts_get_token", 1, ts_get_token, 0},
    {"ts_get_message_imprint", 1, ts_get_message_imprint, 0},
    {"ts_get_time", 1, ts_get_time, 0},
    {"ts_get_serial", 1, ts_get_serial, 0},
    {"ts_get_tsa_name", 1, ts_get_tsa_name, 0},
    {"ts_get_policy_oid", 1, ts_get_policy_oid, 0},
    {"ts_get_hash_algorithm", 1, ts_get_hash_algorithm, 0},
    {"ts_verify", 1, ts_verify, DIRTY},
    {"ts_close", 1, ts_close, 0},
    /* phase 6 — TSA client */
    {"tsa_create", 7, tsa_create, DIRTY},
    {"tsa_request_timestamp", 2, tsa_request_timestamp, DIRTY},
    {"tsa_request_timestamp_hash", 3, tsa_request_timestamp_hash, DIRTY},
    {"tsa_close", 1, tsa_close, 0},
    /* phase 6 — DSS */
    {"dss_cert_count", 1, dss_cert_count, 0},
    {"dss_crl_count", 1, dss_crl_count, 0},
    {"dss_ocsp_count", 1, dss_ocsp_count, 0},
    {"dss_vri_count", 1, dss_vri_count, 0},
    {"dss_get_cert", 2, dss_get_cert, 0},
    {"dss_get_crl", 2, dss_get_crl, 0},
    {"dss_get_ocsp", 2, dss_get_ocsp, 0},
    {"dss_close", 1, dss_close, 0},
    /* phase 6 — validation */
    {"validate_pdf_a", 2, validate_pdf_a, DIRTY},
    {"validate_pdf_ua", 2, validate_pdf_ua, DIRTY},
    {"validate_pdf_x", 2, validate_pdf_x, DIRTY},
    {"pdf_a_is_compliant", 1, pdf_a_is_compliant, 0},
    {"pdf_a_error_count", 1, pdf_a_error_count, 0},
    {"pdf_a_warning_count", 1, pdf_a_warning_count, 0},
    {"pdf_a_get_error", 2, pdf_a_get_error, 0},
    {"pdf_a_close", 1, pdf_a_close, 0},
    {"pdf_ua_is_accessible", 1, pdf_ua_is_accessible, 0},
    {"pdf_ua_error_count", 1, pdf_ua_error_count, 0},
    {"pdf_ua_warning_count", 1, pdf_ua_warning_count, 0},
    {"pdf_ua_get_error", 2, pdf_ua_get_error, 0},
    {"pdf_ua_get_warning", 2, pdf_ua_get_warning, 0},
    {"pdf_ua_get_stats", 1, pdf_ua_get_stats, 0},
    {"pdf_ua_close", 1, pdf_ua_close, 0},
    {"pdf_x_is_compliant", 1, pdf_x_is_compliant, 0},
    {"pdf_x_error_count", 1, pdf_x_error_count, 0},
    {"pdf_x_get_error", 2, pdf_x_get_error, 0},
    {"pdf_x_close", 1, pdf_x_close, 0},
    /* phase 6 — log level */
    {"oxide_set_log_level", 1, oxide_set_log_level, 0},
    {"oxide_get_log_level", 0, oxide_get_log_level, 0},
    /* phase 7 — barcodes / QR */
    {"barcode_generate_qr", 3, barcode_generate_qr, DIRTY},
    {"barcode_generate", 3, barcode_generate, DIRTY},
    {"barcode_get_data", 1, barcode_get_data, 0},
    {"barcode_get_format", 1, barcode_get_format, 0},
    {"barcode_get_confidence", 1, barcode_get_confidence, 0},
    {"barcode_get_image_png", 2, barcode_get_image_png, DIRTY},
    {"barcode_get_svg", 2, barcode_get_svg, DIRTY},
    {"barcode_close", 1, barcode_close, 0},
    {"editor_add_barcode_to_page", 7, editor_add_barcode_to_page, DIRTY},
    /* phase 7 — OCR */
    {"ocr_engine_create", 3, ocr_engine_create, DIRTY},
    {"ocr_engine_close", 1, ocr_engine_close, 0},
    {"ocr_page_needs_ocr", 2, ocr_page_needs_ocr, DIRTY},
    {"ocr_extract_text", 3, ocr_extract_text, DIRTY},
    /* phase 7 — render variants */
    {"doc_render_page_with_options", 11, doc_render_page_with_options, DIRTY},
    {"doc_render_page_with_options_ex", 12, doc_render_page_with_options_ex, DIRTY},
    {"doc_render_page_region", 7, doc_render_page_region, DIRTY},
    {"doc_render_page_fit", 5, doc_render_page_fit, DIRTY},
    {"doc_render_page_raw", 3, doc_render_page_raw, DIRTY},
    {"renderer_create", 4, renderer_create, 0},
    {"renderer_close", 1, renderer_close, 0},
    {"doc_estimate_render_time", 2, doc_estimate_render_time, DIRTY},
    /* phase 7 — redaction */
    {"redaction_add", 9, redaction_add, DIRTY},
    {"redaction_count", 2, redaction_count, 0},
    {"redaction_apply", 5, redaction_apply, DIRTY},
    {"redaction_scrub_metadata", 1, redaction_scrub_metadata, DIRTY},
    /* phase 7 — constructors */
    {"pdf_from_image", 1, pdf_from_image_nif, DIRTY},
    {"pdf_from_image_bytes", 1, pdf_from_image_bytes_nif, DIRTY},
    {"pdf_from_html_css", 3, pdf_from_html_css_nif, DIRTY},
    {"pdf_from_html_css_with_fonts", 4, pdf_from_html_css_with_fonts_nif, DIRTY},
    {"pdf_merge", 1, pdf_merge_nif, DIRTY},
    /* phase 7 — page getters */
    {"page_get_width", 2, page_get_width, 0},
    {"page_get_height", 2, page_get_height, 0},
    {"page_get_rotation", 2, page_get_rotation, 0},
    {"page_get_elements", 2, page_get_elements, DIRTY},
    {"elements_count", 1, elements_count, 0},
    {"elements_close", 1, elements_close, 0},
    /* phase 7 — timestamp */
    {"add_timestamp", 3, add_timestamp, DIRTY},
    /* phase 8 — office I/O */
    {"doc_open_from_docx_bytes", 1, doc_open_from_docx_bytes, DIRTY},
    {"doc_open_from_pptx_bytes", 1, doc_open_from_pptx_bytes, DIRTY},
    {"doc_open_from_xlsx_bytes", 1, doc_open_from_xlsx_bytes, DIRTY},
    {"doc_to_docx", 1, doc_to_docx, DIRTY},
    {"doc_to_pptx", 1, doc_to_pptx, DIRTY},
    {"doc_to_xlsx", 1, doc_to_xlsx, DIRTY},
    /* phase 8 — in-rect extractors */
    {"doc_extract_text_in_rect", 6, doc_extract_text_in_rect, DIRTY},
    {"doc_extract_words_in_rect", 6, doc_extract_words_in_rect, DIRTY},
    {"doc_extract_lines_in_rect", 6, doc_extract_lines_in_rect, DIRTY},
    {"doc_extract_tables_in_rect", 6, doc_extract_tables_in_rect, DIRTY},
    {"doc_extract_images_in_rect", 6, doc_extract_images_in_rect, DIRTY},
    /* phase 8 — auto extraction / classification */
    {"doc_extract_text_auto", 2, doc_extract_text_auto, DIRTY},
    {"doc_extract_all_text", 1, doc_extract_all_text, DIRTY},
    {"doc_extract_page_auto", 3, doc_extract_page_auto, DIRTY},
    {"doc_classify_page", 2, doc_classify_page, DIRTY},
    {"doc_classify_document", 1, doc_classify_document, DIRTY},
    /* phase 8 — header / footer / artifact */
    {"doc_erase_header", 2, doc_erase_header, DIRTY},
    {"doc_erase_footer", 2, doc_erase_footer, DIRTY},
    {"doc_erase_artifacts", 2, doc_erase_artifacts, DIRTY},
    {"doc_remove_headers", 2, doc_remove_headers, DIRTY},
    {"doc_remove_footers", 2, doc_remove_footers, DIRTY},
    {"doc_remove_artifacts", 2, doc_remove_artifacts, DIRTY},
    /* phase 8 — forms */
    {"doc_get_form_fields", 1, doc_get_form_fields, DIRTY},
    {"doc_export_form_data_to_bytes", 2, doc_export_form_data_to_bytes, DIRTY},
    {"doc_import_form_data", 2, doc_import_form_data, DIRTY},
    {"editor_import_fdf_bytes", 2, editor_import_fdf_bytes, DIRTY},
    {"editor_import_xfdf_bytes", 2, editor_import_xfdf_bytes, DIRTY},
    {"form_import_from_file", 2, form_import_from_file, DIRTY},
    /* phase 8 — document structure / metadata */
    {"doc_get_outline", 1, doc_get_outline, DIRTY},
    {"doc_get_page_labels", 1, doc_get_page_labels, DIRTY},
    {"doc_get_xmp_metadata", 1, doc_get_xmp_metadata, DIRTY},
    {"doc_get_source_bytes", 1, doc_get_source_bytes, DIRTY},
    {"doc_has_xfa", 1, doc_has_xfa, 0},
    {"doc_get_page_count", 1, doc_get_page_count, 0},
    {"doc_plan_split_by_bookmarks", 2, doc_plan_split_by_bookmarks, DIRTY},
    /* phase 8 — document-level signatures */
    {"doc_sign", 4, doc_sign, DIRTY},
    {"doc_get_signature_count", 1, doc_get_signature_count, 0},
    {"doc_get_signature", 2, doc_get_signature, 0},
    {"doc_verify_all_signatures", 1, doc_verify_all_signatures, DIRTY},
    {"doc_has_timestamp", 1, doc_has_timestamp, 0},
    {"doc_get_dss", 1, doc_get_dss, 0},
    /* phase 8 — annotation extras */
    {"annot_get_color", 3, annot_get_color, DIRTY},
    {"annot_get_creation_date", 3, annot_get_creation_date, DIRTY},
    {"annot_get_modification_date", 3, annot_get_modification_date, DIRTY},
    {"annot_is_hidden", 3, annot_is_hidden, DIRTY},
    {"annot_is_marked_deleted", 3, annot_is_marked_deleted, DIRTY},
    {"annot_is_printable", 3, annot_is_printable, DIRTY},
    {"annot_is_read_only", 3, annot_is_read_only, DIRTY},
    {"annot_link_get_uri", 3, annot_link_get_uri, DIRTY},
    {"annot_text_get_icon_name", 3, annot_text_get_icon_name, DIRTY},
    {"annot_highlight_quad_points_count", 3, annot_highlight_quad_points_count, DIRTY},
    {"annot_highlight_quad_point", 4, annot_highlight_quad_point, DIRTY},
    {"annotations_to_json", 2, annotations_to_json, DIRTY},
    /* phase 8 — element / font / search JSON accessors */
    {"element_get_type", 2, element_get_type, 0},
    {"element_get_text", 2, element_get_text, 0},
    {"element_get_rect", 2, element_get_rect, 0},
    {"elements_to_json", 1, elements_to_json, DIRTY},
    {"font_get_size", 3, font_get_size, DIRTY},
    {"fonts_to_json", 2, fonts_to_json, DIRTY},
    {"search_results_to_json", 3, search_results_to_json, DIRTY},
    /* phase 8 — crypto / FIPS */
    {"crypto_active_provider", 0, crypto_active_provider, 0},
    {"crypto_cbom", 0, crypto_cbom, 0},
    {"crypto_inventory", 0, crypto_inventory, 0},
    {"crypto_policy", 0, crypto_policy, 0},
    {"crypto_fips_available", 0, crypto_fips_available, 0},
    {"crypto_use_fips", 0, crypto_use_fips, 0},
    {"crypto_set_policy", 1, crypto_set_policy, 0},
    /* phase 8 — models / config */
    {"model_manifest", 0, model_manifest, 0},
    {"prefetch_available", 0, prefetch_available, 0},
    {"prefetch_models", 1, prefetch_models, DIRTY},
    {"set_max_ops_per_stream", 1, set_max_ops_per_stream, 0},
    {"set_preserve_unmapped_glyphs", 1, set_preserve_unmapped_glyphs, 0},
    /* phase 8 — PDF/A conversion */
    {"doc_convert_to_pdf_a", 2, doc_convert_to_pdf_a, DIRTY},
};

ERL_NIF_INIT(Elixir.PdfOxide.Native, funcs, load, NULL, NULL, NULL)
