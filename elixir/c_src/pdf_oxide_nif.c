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

typedef struct { PdfDocument *h; } DocRes;
typedef struct { Pdf *h; } PdfRes;

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

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    (void)priv; (void)info;
    int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
    DOC_RES = enif_open_resource_type(env, NULL, "pdf_oxide_doc", doc_dtor, flags, NULL);
    PDF_RES = enif_open_resource_type(env, NULL, "pdf_oxide_pdf", pdf_dtor, flags, NULL);
    return (DOC_RES && PDF_RES) ? 0 : 1;
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
    {"doc_extract_structured_json", 2, doc_struct_json, DIRTY},
    {"doc_close", 1, doc_close, 0},
    {"pdf_close", 1, pdf_close_nif, 0},
};

ERL_NIF_INIT(Elixir.PdfOxide.Native, funcs, load, NULL, NULL, NULL)
