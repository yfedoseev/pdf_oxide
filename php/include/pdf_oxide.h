

typedef struct Color Color;

typedef struct DocumentEditor DocumentEditor;

typedef struct EmbeddedFont EmbeddedFont;

typedef struct FfiAnnotationList FfiAnnotationList;

typedef struct FfiBarcodeImage FfiBarcodeImage;

typedef struct FfiCharList FfiCharList;

typedef struct FfiDocumentBuilder FfiDocumentBuilder;

typedef struct FfiElementList FfiElementList;

typedef struct FfiFontList FfiFontList;

typedef struct FfiFormFieldList FfiFormFieldList;

typedef struct FfiImageList FfiImageList;

typedef struct FfiPageBuilder FfiPageBuilder;

typedef struct FfiPathList FfiPathList;

typedef struct FfiPdfAResults FfiPdfAResults;

typedef struct FfiPdfXResults FfiPdfXResults;

typedef struct FfiRenderedImage FfiRenderedImage;

typedef struct FfiRenderedImage FfiRenderedImage;

typedef struct FfiSearchResults FfiSearchResults;

typedef struct FfiSignatureInfo FfiSignatureInfo;

typedef struct FfiSignatureInfo FfiSignatureInfo;

typedef struct FfiTableList FfiTableList;

typedef struct FfiTextLineList FfiTextLineList;

typedef struct FfiUaResults FfiUaResults;

typedef struct FfiWordList FfiWordList;

typedef struct Pdf Pdf;

typedef struct PdfDocument PdfDocument;

typedef struct VerticalMetrics VerticalMetrics;

typedef struct {
    const void *certificate_handle;
    const uint8_t *const *certs;
    const uintptr_t *cert_lens;
    uintptr_t n_certs;
    const uint8_t *const *crls;
    const uintptr_t *crl_lens;
    uintptr_t n_crls;
    const uint8_t *const *ocsps;
    const uintptr_t *ocsp_lens;
    uintptr_t n_ocsps;
    const char *tsa_url;
    const char *reason;
    const char *location;
    int32_t level;
} PadesSignOptionsC;

typedef uint32_t NodeId;

typedef uint32_t BoxId;

void pdf_oxide_set_log_level(int32_t level);

int32_t pdf_oxide_get_log_level(void);

int64_t pdf_oxide_set_max_ops_per_stream(int64_t limit);

int32_t pdf_oxide_set_preserve_unmapped_glyphs(int32_t preserve);

char *pdf_oxide_crypto_active_provider(void);

int32_t pdf_oxide_crypto_fips_available(void);

int32_t pdf_oxide_crypto_use_fips(void);

int32_t pdf_oxide_crypto_set_policy(const char *spec);

char *pdf_oxide_crypto_policy(void);

char *pdf_oxide_crypto_inventory(void);

char *pdf_oxide_crypto_cbom(void);

void free_string(char *ptr);

void free_bytes(uint8_t *ptr);

PdfDocument *pdf_document_open(const char *path, int32_t *error_code);

void pdf_document_free(PdfDocument *handle);

int32_t pdf_document_get_page_count(PdfDocument *handle, int32_t *error_code);

void pdf_document_get_version(const PdfDocument *handle, uint8_t *major, uint8_t *minor);

bool pdf_document_has_structure_tree(PdfDocument *handle);

char *pdf_document_extract_text(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_extract_structured_to_json(PdfDocument *handle,
                                              int32_t page_index,
                                              int32_t *error_code);

char *pdf_document_to_markdown(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_to_html(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_to_plain_text(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_to_markdown_all(PdfDocument *handle, int32_t *error_code);

uint8_t *pdf_document_to_docx(PdfDocument *handle, uintptr_t *out_len, int32_t *error_code);

uint8_t *pdf_document_to_pptx(PdfDocument *handle, uintptr_t *out_len, int32_t *error_code);

uint8_t *pdf_document_to_xlsx(PdfDocument *handle, uintptr_t *out_len, int32_t *error_code);

PdfDocument *pdf_document_open_from_docx_bytes(const uint8_t *data,
                                               uintptr_t len,
                                               int32_t *error_code);

PdfDocument *pdf_document_open_from_pptx_bytes(const uint8_t *data,
                                               uintptr_t len,
                                               int32_t *error_code);

PdfDocument *pdf_document_open_from_xlsx_bytes(const uint8_t *data,
                                               uintptr_t len,
                                               int32_t *error_code);

DocumentEditor *document_editor_open(const char *path,
                                     int32_t *error_code);

void document_editor_free(DocumentEditor *handle);

bool document_editor_is_modified(const DocumentEditor *handle);

char *document_editor_get_source_path(const DocumentEditor *handle, int32_t *error_code);

void document_editor_get_version(const DocumentEditor *handle, uint8_t *major, uint8_t *minor);

int32_t document_editor_get_page_count(DocumentEditor *handle, int32_t *error_code);

char *document_editor_get_producer(DocumentEditor *handle, int32_t *error_code);

int32_t document_editor_set_producer(DocumentEditor *handle,
                                     const char *value,
                                     int32_t *error_code);

char *document_editor_get_creation_date(DocumentEditor *handle, int32_t *error_code);

int32_t document_editor_set_creation_date(DocumentEditor *handle,
                                          const char *date_str,
                                          int32_t *error_code);

int32_t document_editor_save(DocumentEditor *handle, const char *path, int32_t *error_code);

DocumentEditor *document_editor_open_from_bytes(const uint8_t *data,
                                                uintptr_t len,
                                                int32_t *error_code);

uint8_t *document_editor_save_to_bytes(DocumentEditor *handle,
                                       uintptr_t *out_len,
                                       int32_t *error_code);

uint8_t *document_editor_save_to_bytes_with_options(DocumentEditor *handle,
                                                    bool compress,
                                                    bool garbage_collect,
                                                    bool linearize,
                                                    uintptr_t *out_len,
                                                    int32_t *error_code);

uint8_t *document_editor_extract_pages_to_bytes(DocumentEditor *handle,
                                                const int32_t *pages,
                                                uintptr_t count,
                                                uintptr_t *out_len,
                                                int32_t *error_code);

int32_t document_editor_convert_to_pdf_a(DocumentEditor *handle,
                                         int32_t level,
                                         int32_t *error_code);

uint8_t *document_editor_save_encrypted_to_bytes(DocumentEditor *handle,
                                                 const char *user_password,
                                                 const char *owner_password,
                                                 uintptr_t *out_len,
                                                 int32_t *error_code);

int32_t document_editor_merge_from_bytes(DocumentEditor *handle,
                                         const uint8_t *data,
                                         uintptr_t len,
                                         int32_t *error_code);

int32_t document_editor_embed_file(DocumentEditor *handle,
                                   const char *name,
                                   const uint8_t *data,
                                   uintptr_t len,
                                   int32_t *error_code);

int32_t document_editor_apply_page_redactions(DocumentEditor *handle,
                                              uintptr_t page,
                                              int32_t *error_code);

int32_t document_editor_apply_all_redactions(DocumentEditor *handle, int32_t *error_code);

int32_t pdf_redaction_add(DocumentEditor *handle,
                          uintptr_t page,
                          double x1,
                          double y1,
                          double x2,
                          double y2,
                          double r,
                          double g,
                          double b,
                          int32_t *error_code);

int32_t pdf_redaction_count(DocumentEditor *handle, uintptr_t page, int32_t *error_code);

int32_t pdf_redaction_apply(DocumentEditor *handle,
                            bool scrub_metadata,
                            double r,
                            double g,
                            double b,
                            int32_t *error_code);

int32_t pdf_redaction_scrub_metadata(DocumentEditor *handle, int32_t *error_code);

int32_t document_editor_rotate_all_pages(DocumentEditor *handle,
                                         int32_t degrees,
                                         int32_t *error_code);

int32_t document_editor_rotate_page_by(DocumentEditor *handle,
                                       uintptr_t page,
                                       int32_t degrees,
                                       int32_t *error_code);

int32_t document_editor_get_page_media_box(DocumentEditor *handle,
                                           uintptr_t page,
                                           double *x,
                                           double *y,
                                           double *w,
                                           double *h,
                                           int32_t *error_code);

int32_t document_editor_set_page_media_box(DocumentEditor *handle,
                                           uintptr_t page,
                                           double x,
                                           double y,
                                           double w,
                                           double h,
                                           int32_t *error_code);

int32_t document_editor_get_page_crop_box(DocumentEditor *handle,
                                          uintptr_t page,
                                          double *x,
                                          double *y,
                                          double *w,
                                          double *h,
                                          int32_t *error_code);

int32_t document_editor_set_page_crop_box(DocumentEditor *handle,
                                          uintptr_t page,
                                          double x,
                                          double y,
                                          double w,
                                          double h,
                                          int32_t *error_code);

int32_t document_editor_erase_regions(DocumentEditor *handle,
                                      uintptr_t page,
                                      const double *rects,
                                      uintptr_t rects_count,
                                      int32_t *error_code);

int32_t document_editor_clear_erase_regions(DocumentEditor *handle,
                                            uintptr_t page,
                                            int32_t *error_code);

int32_t document_editor_is_page_marked_for_flatten(const DocumentEditor *handle, uintptr_t page);

int32_t document_editor_unmark_page_for_flatten(DocumentEditor *handle,
                                                uintptr_t page,
                                                int32_t *error_code);

int32_t document_editor_is_page_marked_for_redaction(const DocumentEditor *handle, uintptr_t page);

int32_t document_editor_unmark_page_for_redaction(DocumentEditor *handle,
                                                  uintptr_t page,
                                                  int32_t *error_code);

Pdf *pdf_from_markdown(const char *markdown, int32_t *error_code);

Pdf *pdf_from_html(const char *html, int32_t *error_code);

Pdf *pdf_from_text(const char *text, int32_t *error_code);

int32_t pdf_save(Pdf *handle, const char *path, int32_t *error_code);

uint8_t *pdf_save_to_bytes(Pdf *handle, int32_t *data_len, int32_t *error_code);

int32_t pdf_get_page_count(Pdf *handle, int32_t *error_code);

void pdf_free(Pdf *handle);

FfiSearchResults *pdf_document_search_page(PdfDocument *handle,
                                           int32_t page_index,
                                           const char *search_term,
                                           bool case_sensitive,
                                           int32_t *error_code);

FfiSearchResults *pdf_document_search_all(PdfDocument *handle,
                                          const char *search_term,
                                          bool case_sensitive,
                                          int32_t *error_code);

int32_t pdf_oxide_search_result_count(const FfiSearchResults *results);

char *pdf_oxide_search_result_get_text(const FfiSearchResults *results,
                                       int32_t index,
                                       int32_t *error_code);

int32_t pdf_oxide_search_result_get_page(const FfiSearchResults *results,
                                         int32_t index,
                                         int32_t *error_code);

void pdf_oxide_search_result_get_bbox(const FfiSearchResults *results,
                                      int32_t index,
                                      float *x,
                                      float *y,
                                      float *width,
                                      float *height,
                                      int32_t *error_code);

void pdf_oxide_search_result_free(FfiSearchResults *handle);

FfiFontList *pdf_document_get_embedded_fonts(PdfDocument *handle,
                                             int32_t page_index,
                                             int32_t *error_code);

int32_t pdf_oxide_font_count(const FfiFontList *fonts);

char *pdf_oxide_font_get_name(const FfiFontList *fonts, int32_t index, int32_t *error_code);

char *pdf_oxide_font_get_type(const FfiFontList *fonts, int32_t index, int32_t *error_code);

char *pdf_oxide_font_get_encoding(const FfiFontList *fonts, int32_t index, int32_t *error_code);

int32_t pdf_oxide_font_is_embedded(const FfiFontList *fonts, int32_t index, int32_t *error_code);

int32_t pdf_oxide_font_is_subset(const FfiFontList *fonts, int32_t index, int32_t *error_code);

float pdf_oxide_font_get_size(const FfiFontList *_fonts, int32_t _index, int32_t *error_code);

void pdf_oxide_font_list_free(FfiFontList *handle);

FfiImageList *pdf_document_get_embedded_images(PdfDocument *handle,
                                               int32_t page_index,
                                               int32_t *error_code);

int32_t pdf_oxide_image_count(const FfiImageList *images);

int32_t pdf_oxide_image_get_width(const FfiImageList *images, int32_t index, int32_t *error_code);

int32_t pdf_oxide_image_get_height(const FfiImageList *images, int32_t index, int32_t *error_code);

char *pdf_oxide_image_get_format(const FfiImageList *images, int32_t index, int32_t *error_code);

char *pdf_oxide_image_get_colorspace(const FfiImageList *images,
                                     int32_t index,
                                     int32_t *error_code);

int32_t pdf_oxide_image_get_bits_per_component(const FfiImageList *images,
                                               int32_t index,
                                               int32_t *error_code);

uint8_t *pdf_oxide_image_get_data(const FfiImageList *images,
                                  int32_t index,
                                  int32_t *data_len,
                                  int32_t *error_code);

void pdf_oxide_image_list_free(FfiImageList *handle);

FfiAnnotationList *pdf_document_get_page_annotations(PdfDocument *handle,
                                                     int32_t page_index,
                                                     int32_t *error_code);

int32_t pdf_oxide_annotation_count(const FfiAnnotationList *annotations);

char *pdf_oxide_annotation_get_type(const FfiAnnotationList *annotations,
                                    int32_t index,
                                    int32_t *error_code);

char *pdf_oxide_annotation_get_content(const FfiAnnotationList *annotations,
                                       int32_t index,
                                       int32_t *error_code);

void pdf_oxide_annotation_get_rect(const FfiAnnotationList *annotations,
                                   int32_t index,
                                   float *x,
                                   float *y,
                                   float *width,
                                   float *height,
                                   int32_t *error_code);

void pdf_oxide_annotation_list_free(FfiAnnotationList *handle);

char *pdf_oxide_annotation_get_subtype(const FfiAnnotationList *annotations,
                                       int32_t index,
                                       int32_t *error_code);

bool pdf_oxide_annotation_is_marked_deleted(const FfiAnnotationList *_annotations,
                                            int32_t _index,
                                            int32_t *error_code);

int64_t pdf_oxide_annotation_get_creation_date(const FfiAnnotationList *annotations,
                                               int32_t index,
                                               int32_t *error_code);

int64_t pdf_oxide_annotation_get_modification_date(const FfiAnnotationList *annotations,
                                                   int32_t index,
                                                   int32_t *error_code);

char *pdf_oxide_annotation_get_author(const FfiAnnotationList *annotations,
                                      int32_t index,
                                      int32_t *error_code);

float pdf_oxide_annotation_get_border_width(const FfiAnnotationList *annotations,
                                            int32_t index,
                                            int32_t *error_code);

uint32_t pdf_oxide_annotation_get_color(const FfiAnnotationList *annotations,
                                        int32_t index,
                                        int32_t *error_code);

bool pdf_oxide_annotation_is_hidden(const FfiAnnotationList *annotations,
                                    int32_t index,
                                    int32_t *error_code);

bool pdf_oxide_annotation_is_printable(const FfiAnnotationList *annotations,
                                       int32_t index,
                                       int32_t *error_code);

bool pdf_oxide_annotation_is_read_only(const FfiAnnotationList *annotations,
                                       int32_t index,
                                       int32_t *error_code);

char *pdf_oxide_link_annotation_get_uri(const FfiAnnotationList *annotations,
                                        int32_t index,
                                        int32_t *error_code);

char *pdf_oxide_text_annotation_get_icon_name(const FfiAnnotationList *_annotations,
                                              int32_t _index,
                                              int32_t *error_code);

int32_t pdf_oxide_highlight_annotation_get_quad_points_count(const FfiAnnotationList *annotations,
                                                             int32_t index,
                                                             int32_t *error_code);

void pdf_oxide_highlight_annotation_get_quad_point(const FfiAnnotationList *annotations,
                                                   int32_t index,
                                                   int32_t quad_index,
                                                   float *x1,
                                                   float *y1,
                                                   float *x2,
                                                   float *y2,
                                                   float *x3,
                                                   float *y3,
                                                   float *x4,
                                                   float *y4,
                                                   int32_t *error_code);

float pdf_page_get_width(PdfDocument *handle, int32_t page_index, int32_t *error_code);

float pdf_page_get_height(PdfDocument *handle, int32_t page_index, int32_t *error_code);

int32_t pdf_page_get_rotation(PdfDocument *_handle, int32_t _page_index, int32_t *error_code);

FfiElementList *pdf_page_get_elements(PdfDocument *handle, int32_t page_index, int32_t *error_code);

int32_t pdf_oxide_element_count(const FfiElementList *elements);

char *pdf_oxide_element_get_type(const FfiElementList *_elements,
                                 int32_t _index,
                                 int32_t *error_code);

char *pdf_oxide_element_get_text(const FfiElementList *elements,
                                 int32_t index,
                                 int32_t *error_code);

void pdf_oxide_element_get_rect(const FfiElementList *elements,
                                int32_t index,
                                float *x,
                                float *y,
                                float *width,
                                float *height,
                                int32_t *error_code);

void pdf_oxide_elements_free(FfiElementList *handle);

FfiBarcodeImage *pdf_generate_qr_code(const char *data,
                                      int32_t error_correction,
                                      int32_t size_px,
                                      int32_t *error_code);

FfiBarcodeImage *pdf_generate_barcode(const char *data,
                                      int32_t format,
                                      int32_t size_px,
                                      int32_t *error_code);

uint8_t *pdf_barcode_get_image_png(const FfiBarcodeImage *barcode_handle,
                                   int32_t _size_px,
                                   int32_t *out_len,
                                   int32_t *error_code);

char *pdf_barcode_get_svg(const FfiBarcodeImage *barcode_handle,
                          int32_t _size_px,
                          int32_t *error_code);

int32_t pdf_add_barcode_to_page(DocumentEditor *document_handle,
                                int32_t page_index,
                                const FfiBarcodeImage *barcode_handle,
                                float x,
                                float y,
                                float width,
                                float height,
                                int32_t *error_code);

int32_t pdf_barcode_get_format(const FfiBarcodeImage *barcode_handle, int32_t *error_code);

char *pdf_barcode_get_data(const FfiBarcodeImage *barcode_handle, int32_t *error_code);

float pdf_barcode_get_confidence(const FfiBarcodeImage *_barcode_handle, int32_t *error_code);

void pdf_barcode_free(FfiBarcodeImage *handle);

void *pdf_certificate_load_from_bytes(const uint8_t *cert_bytes,
                                      int32_t cert_len,
                                      const char *password,
                                      int32_t *error_code);

void *pdf_certificate_load_from_pem(const char *cert_pem, const char *key_pem, int32_t *error_code);

int32_t pdf_document_sign(void *document_handle,
                          const void *certificate_handle,
                          const char *reason,
                          const char *location,
                          int32_t *error_code);

uint8_t *pdf_sign_bytes(const uint8_t *pdf_data,
                        uintptr_t pdf_len,
                        const void *certificate_handle,
                        const char *reason,
                        const char *location,
                        uintptr_t *out_len,
                        int32_t *error_code);

int32_t pdf_document_get_signature_count(const void *document_handle, int32_t *error_code);

void *pdf_document_get_signature(const void *document_handle, int32_t index, int32_t *error_code);

int32_t pdf_signature_verify(const void *signature_handle, int32_t *error_code);

int32_t pdf_signature_verify_detached(const void *signature_handle,
                                      const uint8_t *pdf_data,
                                      uintptr_t pdf_len,
                                      int32_t *error_code);

int32_t pdf_document_verify_all_signatures(const void *_document_handle, int32_t *error_code);

char *pdf_signature_get_signer_name(const FfiSignatureInfo *sig, int32_t *error_code);

int64_t pdf_signature_get_signing_time(const FfiSignatureInfo *sig, int32_t *error_code);

char *pdf_signature_get_signing_reason(const FfiSignatureInfo *sig, int32_t *error_code);

char *pdf_signature_get_signing_location(const FfiSignatureInfo *sig, int32_t *error_code);

void *pdf_signature_get_certificate(const void *sig, int32_t *error_code);

char *pdf_certificate_get_subject(const void *cert, int32_t *error_code);

char *pdf_certificate_get_issuer(const void *cert, int32_t *error_code);

char *pdf_certificate_get_serial(const void *cert, int32_t *error_code);

void pdf_certificate_get_validity(const void *cert,
                                  int64_t *not_before,
                                  int64_t *not_after,
                                  int32_t *error_code);

int32_t pdf_certificate_is_valid(const void *cert, int32_t *error_code);

void pdf_signature_free(FfiSignatureInfo *handle);

uint8_t *pdf_sign_bytes_pades(const uint8_t *pdf_data,
                              uintptr_t pdf_len,
                              const void *certificate_handle,
                              int32_t level,
                              const char *tsa_url,
                              const char *reason,
                              const char *location,
                              const uint8_t *const *certs,
                              const uintptr_t *cert_lens,
                              uintptr_t n_certs,
                              const uint8_t *const *crls,
                              const uintptr_t *crl_lens,
                              uintptr_t n_crls,
                              const uint8_t *const *ocsps,
                              const uintptr_t *ocsp_lens,
                              uintptr_t n_ocsps,
                              uintptr_t *out_len,
                              int32_t *error_code);

uint8_t *pdf_sign_bytes_pades_opts(const uint8_t *pdf_data,
                                   uintptr_t pdf_len,
                                   const PadesSignOptionsC *options,
                                   uintptr_t *out_len,
                                   int32_t *error_code);

int32_t pdf_signature_get_pades_level(const void *signature_handle, int32_t *error_code);

void *pdf_document_get_dss(const void *document_handle, int32_t *error_code);

int32_t pdf_document_has_timestamp(const void *document_handle, int32_t *error_code);

int32_t pdf_dss_cert_count(const void *_dss);

int32_t pdf_dss_crl_count(const void *_dss);

int32_t pdf_dss_ocsp_count(const void *_dss);

int32_t pdf_dss_vri_count(const void *_dss);

uint8_t *pdf_dss_get_cert(const void *_dss,
                          int32_t _index,
                          uintptr_t *_out_len,
                          int32_t *error_code);

uint8_t *pdf_dss_get_crl(const void *_dss,
                         int32_t _index,
                         uintptr_t *_out_len,
                         int32_t *error_code);

uint8_t *pdf_dss_get_ocsp(const void *_dss,
                          int32_t _index,
                          uintptr_t *_out_len,
                          int32_t *error_code);

void pdf_dss_free(void *dss);

void pdf_certificate_free(void *handle);

int32_t pdf_estimate_render_time(const void *_doc, int32_t _page_index, int32_t *error_code);

void *pdf_create_renderer(int32_t _dpi,
                          int32_t _format,
                          int32_t _quality,
                          bool _anti_alias,
                          int32_t *error_code);

FfiRenderedImage *pdf_render_page(PdfDocument *doc,
                                  int32_t page_index,
                                  int32_t format,
                                  int32_t *error_code);

FfiRenderedImage *pdf_render_page_with_options(PdfDocument *doc,
                                               int32_t page_index,
                                               int32_t dpi,
                                               int32_t format,
                                               float bg_r,
                                               float bg_g,
                                               float bg_b,
                                               float bg_a,
                                               int32_t transparent_background,
                                               int32_t render_annotations,
                                               int32_t jpeg_quality,
                                               int32_t *error_code);

FfiRenderedImage *pdf_render_page_with_options_ex(PdfDocument *doc,
                                                  int32_t page_index,
                                                  int32_t dpi,
                                                  int32_t format,
                                                  float bg_r,
                                                  float bg_g,
                                                  float bg_b,
                                                  float bg_a,
                                                  int32_t transparent_background,
                                                  int32_t render_annotations,
                                                  int32_t jpeg_quality,
                                                  const char *const *excluded_layers,
                                                  uintptr_t excluded_layers_count,
                                                  int32_t *error_code);

FfiRenderedImage *pdf_render_page_region(PdfDocument *doc,
                                         int32_t page_index,
                                         float crop_x,
                                         float crop_y,
                                         float crop_width,
                                         float crop_height,
                                         int32_t format,
                                         int32_t *error_code);

FfiRenderedImage *pdf_render_page_zoom(PdfDocument *doc,
                                       int32_t page_index,
                                       float zoom,
                                       int32_t format,
                                       int32_t *error_code);

FfiRenderedImage *pdf_render_page_fit(PdfDocument *doc,
                                      int32_t page_index,
                                      int32_t w,
                                      int32_t h,
                                      int32_t format,
                                      int32_t *error_code);

FfiRenderedImage *pdf_render_page_thumbnail(PdfDocument *doc,
                                            int32_t page_index,
                                            int32_t _size,
                                            int32_t format,
                                            int32_t *error_code);

FfiRenderedImage *pdf_render_page_raw(PdfDocument *doc,
                                      int32_t page_index,
                                      int32_t dpi,
                                      int32_t *out_width,
                                      int32_t *out_height,
                                      int32_t *error_code);

int32_t pdf_get_rendered_image_width(const FfiRenderedImage *img, int32_t *error_code);

int32_t pdf_get_rendered_image_height(const FfiRenderedImage *img, int32_t *error_code);

uint8_t *pdf_get_rendered_image_data(const FfiRenderedImage *img,
                                     int32_t *data_len,
                                     int32_t *error_code);

int32_t pdf_save_rendered_image(const FfiRenderedImage *img,
                                const char *file_path,
                                int32_t *error_code);

void pdf_rendered_image_free(FfiRenderedImage *handle);

void pdf_renderer_free(void *_handle);

void *pdf_tsa_client_create(const char *url,
                            const char *username,
                            const char *password,
                            int32_t timeout,
                            int32_t hash_algo,
                            bool use_nonce,
                            bool cert_req,
                            int32_t *error_code);

void pdf_tsa_client_free(void *client);

void *pdf_tsa_request_timestamp(const void *client,
                                const uint8_t *data,
                                uintptr_t data_len,
                                int32_t *error_code);

void *pdf_tsa_request_timestamp_hash(const void *client,
                                     const uint8_t *hash,
                                     uintptr_t hash_len,
                                     int32_t hash_algo,
                                     int32_t *error_code);

void *pdf_timestamp_parse(const uint8_t *bytes, uintptr_t len, int32_t *error_code);

const uint8_t *pdf_timestamp_get_token(const void *ts, uintptr_t *out_len, int32_t *error_code);

int64_t pdf_timestamp_get_time(const void *ts, int32_t *error_code);

char *pdf_timestamp_get_serial(const void *ts, int32_t *error_code);

char *pdf_timestamp_get_tsa_name(const void *ts, int32_t *error_code);

char *pdf_timestamp_get_policy_oid(const void *ts, int32_t *error_code);

int32_t pdf_timestamp_get_hash_algorithm(const void *ts, int32_t *error_code);

const uint8_t *pdf_timestamp_get_message_imprint(const void *ts,
                                                 uintptr_t *out_len,
                                                 int32_t *error_code);

bool pdf_timestamp_verify(const void *ts, int32_t *error_code);

void pdf_timestamp_free(void *ts);

bool pdf_signature_add_timestamp(const void *_sig, const void *_ts, int32_t *error_code);

bool pdf_signature_has_timestamp(const void *_sig, int32_t *error_code);

void *pdf_signature_get_timestamp(const void *_sig, int32_t *error_code);

bool pdf_add_timestamp(const uint8_t *_pdf_data,
                       uintptr_t _pdf_len,
                       int32_t _sig_index,
                       const char *_tsa_url,
                       uint8_t **_out_data,
                       uintptr_t *_out_len,
                       int32_t *error_code);

FfiUaResults *pdf_validate_pdf_ua(PdfDocument *document, int32_t level, int32_t *error_code);

bool pdf_pdf_ua_is_accessible(const FfiUaResults *results, int32_t *error_code);

int32_t pdf_pdf_ua_error_count(const FfiUaResults *results);

char *pdf_pdf_ua_get_error(const FfiUaResults *results, int32_t index, int32_t *error_code);

int32_t pdf_pdf_ua_warning_count(const FfiUaResults *results);

char *pdf_pdf_ua_get_warning(const FfiUaResults *results, int32_t index, int32_t *error_code);

bool pdf_pdf_ua_get_stats(const FfiUaResults *results,
                          int32_t *out_struct,
                          int32_t *out_images,
                          int32_t *out_tables,
                          int32_t *out_forms,
                          int32_t *out_annotations,
                          int32_t *out_pages,
                          int32_t *error_code);

void pdf_pdf_ua_results_free(FfiUaResults *results);

bool pdf_form_import_from_file(const void *_document, const char *_filename, int32_t *error_code);

int32_t pdf_document_import_form_data(const void *_document,
                                      const char *_data_path,
                                      int32_t *error_code);

int32_t pdf_editor_import_fdf_bytes(const void *_document,
                                    const uint8_t *_data,
                                    uintptr_t _data_len,
                                    int32_t *error_code);

int32_t pdf_editor_import_xfdf_bytes(const void *_document,
                                     const uint8_t *_data,
                                     uintptr_t _data_len,
                                     int32_t *error_code);

uint8_t *pdf_document_export_form_data_to_bytes(PdfDocument *document,
                                                int32_t format_type,
                                                uintptr_t *out_len,
                                                int32_t *error_code);

PdfDocument *pdf_document_open_from_bytes(const uint8_t *data, uintptr_t len, int32_t *error_code);

PdfDocument *pdf_document_open_with_password(const char *path,
                                             const char *password,
                                             int32_t *error_code);

bool pdf_document_is_encrypted(const PdfDocument *handle);

bool pdf_document_authenticate(PdfDocument *handle, const char *password, int32_t *error_code);

char *pdf_document_extract_all_text(PdfDocument *handle, int32_t *error_code);

char *pdf_document_to_html_all(PdfDocument *handle, int32_t *error_code);

char *pdf_document_to_plain_text_all(PdfDocument *handle, int32_t *error_code);

FfiCharList *pdf_document_extract_chars(PdfDocument *handle,
                                        int32_t page_index,
                                        int32_t *error_code);

int32_t pdf_oxide_char_count(const FfiCharList *chars);

uint32_t pdf_oxide_char_get_char(const FfiCharList *chars, int32_t index, int32_t *error_code);

void pdf_oxide_char_get_bbox(const FfiCharList *chars,
                             int32_t index,
                             float *x,
                             float *y,
                             float *w,
                             float *h,
                             int32_t *error_code);

char *pdf_oxide_char_get_font_name(const FfiCharList *chars, int32_t index, int32_t *error_code);

float pdf_oxide_char_get_font_size(const FfiCharList *chars, int32_t index, int32_t *error_code);

void pdf_oxide_char_list_free(FfiCharList *handle);

FfiWordList *pdf_document_extract_words(PdfDocument *handle,
                                        int32_t page_index,
                                        int32_t *error_code);

int32_t pdf_oxide_word_count(const FfiWordList *words);

char *pdf_oxide_word_get_text(const FfiWordList *words, int32_t index, int32_t *error_code);

void pdf_oxide_word_get_bbox(const FfiWordList *words,
                             int32_t index,
                             float *x,
                             float *y,
                             float *w,
                             float *h,
                             int32_t *error_code);

char *pdf_oxide_word_get_font_name(const FfiWordList *words, int32_t index, int32_t *error_code);

float pdf_oxide_word_get_font_size(const FfiWordList *words, int32_t index, int32_t *error_code);

bool pdf_oxide_word_is_bold(const FfiWordList *words, int32_t index, int32_t *error_code);

void pdf_oxide_word_list_free(FfiWordList *handle);

FfiTextLineList *pdf_document_extract_text_lines(PdfDocument *handle,
                                                 int32_t page_index,
                                                 int32_t *error_code);

int32_t pdf_oxide_line_count(const FfiTextLineList *lines);

char *pdf_oxide_line_get_text(const FfiTextLineList *lines, int32_t index, int32_t *error_code);

void pdf_oxide_line_get_bbox(const FfiTextLineList *lines,
                             int32_t index,
                             float *x,
                             float *y,
                             float *w,
                             float *h,
                             int32_t *error_code);

int32_t pdf_oxide_line_get_word_count(const FfiTextLineList *lines,
                                      int32_t index,
                                      int32_t *error_code);

void pdf_oxide_line_list_free(FfiTextLineList *handle);

FfiTableList *pdf_document_extract_tables(PdfDocument *handle,
                                          int32_t page_index,
                                          int32_t *error_code);

int32_t pdf_oxide_table_count(const FfiTableList *tables);

int32_t pdf_oxide_table_get_row_count(const FfiTableList *tables,
                                      int32_t index,
                                      int32_t *error_code);

int32_t pdf_oxide_table_get_col_count(const FfiTableList *tables,
                                      int32_t index,
                                      int32_t *error_code);

char *pdf_oxide_table_get_cell_text(const FfiTableList *tables,
                                    int32_t table_index,
                                    int32_t row,
                                    int32_t col,
                                    int32_t *error_code);

bool pdf_oxide_table_has_header(const FfiTableList *tables, int32_t index, int32_t *error_code);

void pdf_oxide_table_list_free(FfiTableList *handle);

char *pdf_document_extract_text_in_rect(PdfDocument *handle,
                                        int32_t page_index,
                                        float x,
                                        float y,
                                        float w,
                                        float h,
                                        int32_t *error_code);

FfiWordList *pdf_document_extract_words_in_rect(PdfDocument *handle,
                                                int32_t page_index,
                                                float x,
                                                float y,
                                                float w,
                                                float h,
                                                int32_t *error_code);

FfiTextLineList *pdf_document_extract_lines_in_rect(PdfDocument *handle,
                                                    int32_t page_index,
                                                    float x,
                                                    float y,
                                                    float w,
                                                    float h,
                                                    int32_t *error_code);

FfiTableList *pdf_document_extract_tables_in_rect(PdfDocument *handle,
                                                  int32_t page_index,
                                                  float x,
                                                  float y,
                                                  float w,
                                                  float h,
                                                  int32_t *error_code);

FfiImageList *pdf_document_extract_images_in_rect(PdfDocument *handle,
                                                  int32_t page_index,
                                                  float x,
                                                  float y,
                                                  float w,
                                                  float h,
                                                  int32_t *error_code);

FfiFormFieldList *pdf_document_get_form_fields(PdfDocument *handle, int32_t *error_code);

int32_t pdf_oxide_form_field_count(const FfiFormFieldList *fields);

char *pdf_oxide_form_field_get_name(const FfiFormFieldList *fields,
                                    int32_t index,
                                    int32_t *error_code);

char *pdf_oxide_form_field_get_type(const FfiFormFieldList *fields,
                                    int32_t index,
                                    int32_t *error_code);

char *pdf_oxide_form_field_get_value(const FfiFormFieldList *fields,
                                     int32_t index,
                                     int32_t *error_code);

bool pdf_oxide_form_field_is_readonly(const FfiFormFieldList *fields,
                                      int32_t index,
                                      int32_t *error_code);

bool pdf_oxide_form_field_is_required(const FfiFormFieldList *fields,
                                      int32_t index,
                                      int32_t *error_code);

void pdf_oxide_form_field_list_free(FfiFormFieldList *handle);

bool pdf_document_has_xfa(PdfDocument *handle);

int32_t pdf_document_remove_headers(PdfDocument *handle, float threshold, int32_t *error_code);

int32_t pdf_document_remove_footers(PdfDocument *handle, float threshold, int32_t *error_code);

int32_t pdf_document_remove_artifacts(PdfDocument *handle, float threshold, int32_t *error_code);

int32_t pdf_document_erase_header(PdfDocument *handle, int32_t page_index, int32_t *error_code);

int32_t pdf_document_erase_footer(PdfDocument *handle, int32_t page_index, int32_t *error_code);

int32_t pdf_document_erase_artifacts(PdfDocument *handle, int32_t page_index, int32_t *error_code);

int32_t document_editor_delete_page(DocumentEditor *handle,
                                    int32_t page_index,
                                    int32_t *error_code);

int32_t document_editor_move_page(DocumentEditor *handle,
                                  int32_t from,
                                  int32_t to,
                                  int32_t *error_code);

int32_t document_editor_get_page_rotation(DocumentEditor *handle,
                                          int32_t page,
                                          int32_t *error_code);

int32_t document_editor_set_page_rotation(DocumentEditor *handle,
                                          int32_t page,
                                          int32_t degrees,
                                          int32_t *error_code);

int32_t document_editor_erase_region(DocumentEditor *handle,
                                     int32_t page,
                                     float x,
                                     float y,
                                     float w,
                                     float h,
                                     int32_t *error_code);

int32_t document_editor_flatten_annotations(DocumentEditor *handle,
                                            int32_t page,
                                            int32_t *error_code);

int32_t document_editor_flatten_all_annotations(DocumentEditor *handle, int32_t *error_code);

int32_t document_editor_crop_margins(DocumentEditor *handle,
                                     float left,
                                     float right,
                                     float top,
                                     float bottom,
                                     int32_t *error_code);

int32_t document_editor_merge_from(DocumentEditor *handle,
                                   const char *source_path,
                                   int32_t *error_code);

Pdf *pdf_from_image(const char *path, int32_t *error_code);

Pdf *pdf_from_image_bytes(const uint8_t *data, int32_t data_len, int32_t *error_code);

uint8_t *pdf_merge(const char *const *paths,
                   int32_t path_count,
                   int32_t *data_len,
                   int32_t *error_code);

FfiPdfAResults *pdf_validate_pdf_a_level(PdfDocument *document, int32_t level, int32_t *error_code);

bool pdf_convert_to_pdf_a(PdfDocument *document, int32_t level, int32_t *error_code);

uint8_t *pdf_document_get_source_bytes(PdfDocument *document,
                                       uintptr_t *out_len,
                                       int32_t *error_code);

bool pdf_pdf_a_is_compliant(const FfiPdfAResults *results, int32_t *error_code);

int32_t pdf_pdf_a_error_count(const FfiPdfAResults *results);

char *pdf_pdf_a_get_error(const FfiPdfAResults *results, int32_t index, int32_t *error_code);

int32_t pdf_pdf_a_warning_count(const FfiPdfAResults *results);

void pdf_pdf_a_results_free(FfiPdfAResults *results);

FfiPdfXResults *pdf_validate_pdf_x_level(PdfDocument *document, int32_t level, int32_t *error_code);

bool pdf_pdf_x_is_compliant(const FfiPdfXResults *results, int32_t *error_code);

int32_t pdf_pdf_x_error_count(const FfiPdfXResults *results);

char *pdf_pdf_x_get_error(const FfiPdfXResults *results, int32_t index, int32_t *error_code);

void pdf_pdf_x_results_free(FfiPdfXResults *results);

int32_t document_editor_save_encrypted(DocumentEditor *handle,
                                       const char *path,
                                       const char *user_password,
                                       const char *owner_password,
                                       int32_t *error_code);

FfiPathList *pdf_document_extract_paths(PdfDocument *handle,
                                        int32_t page_index,
                                        int32_t *error_code);

int32_t pdf_oxide_path_count(const FfiPathList *paths);

void pdf_oxide_path_get_bbox(const FfiPathList *paths,
                             int32_t index,
                             float *x,
                             float *y,
                             float *w,
                             float *h,
                             int32_t *error_code);

float pdf_oxide_path_get_stroke_width(const FfiPathList *paths, int32_t index, int32_t *error_code);

bool pdf_oxide_path_has_stroke(const FfiPathList *paths, int32_t index, int32_t *error_code);

bool pdf_oxide_path_has_fill(const FfiPathList *paths, int32_t index, int32_t *error_code);

int32_t pdf_oxide_path_get_operation_count(const FfiPathList *paths,
                                           int32_t index,
                                           int32_t *error_code);

void pdf_oxide_path_list_free(FfiPathList *handle);

char *pdf_document_get_page_labels(PdfDocument *handle, int32_t *error_code);

char *pdf_document_get_xmp_metadata(PdfDocument *handle, int32_t *error_code);

char *pdf_document_get_outline(PdfDocument *handle, int32_t *error_code);

char *pdf_document_plan_split_by_bookmarks(PdfDocument *handle,
                                           const char *options_json,
                                           int32_t *error_code);

char *pdf_document_classify_page(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_classify_document(PdfDocument *handle, int32_t *error_code);

char *pdf_document_extract_text_auto(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *pdf_document_extract_page_auto(PdfDocument *handle,
                                     int32_t page_index,
                                     const char *options_json,
                                     int32_t *error_code);

char *pdf_oxide_prefetch_models(const char *languages_csv, int32_t *error_code);

char *pdf_oxide_model_manifest(void);

int32_t pdf_oxide_prefetch_available(void);

int32_t document_editor_set_form_field_value(DocumentEditor *handle,
                                             const char *name,
                                             const char *value,
                                             int32_t *error_code);

int32_t document_editor_flatten_forms(DocumentEditor *handle, int32_t *error_code);

int32_t document_editor_flatten_forms_on_page(DocumentEditor *handle,
                                              int32_t page_index,
                                              int32_t *error_code);

int32_t document_editor_flatten_warnings_count(const DocumentEditor *handle);

char *document_editor_flatten_warning(const DocumentEditor *handle,
                                      int32_t index,
                                      int32_t *error_code);

PdfDocument *PdfDocumentOpen(const char *path, int32_t *error_code);

void PdfDocumentFree(PdfDocument *handle);

int32_t PdfDocumentGetPageCount(PdfDocument *handle, int32_t *error_code);

char *PdfDocumentExtractText(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *PdfDocumentToMarkdown(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *PdfDocumentToHtml(PdfDocument *handle, int32_t page_index, int32_t *error_code);

char *PdfDocumentToPlainText(PdfDocument *handle, int32_t page_index, int32_t *error_code);

Pdf *PdfFromMarkdown(const char *markdown, int32_t *error_code);

Pdf *PdfFromHtml(const char *html, int32_t *error_code);

Pdf *PdfFromText(const char *text, int32_t *error_code);

int32_t PdfSave(Pdf *handle, const char *path, int32_t *error_code);

uint8_t *PdfSaveToBytes(Pdf *handle, int32_t *data_len, int32_t *error_code);

void PdfFree(Pdf *handle);

DocumentEditor *DocumentEditorOpen(const char *path, int32_t *error_code);

void DocumentEditorFree(DocumentEditor *handle);

int32_t DocumentEditorSave(DocumentEditor *handle, const char *path, int32_t *error_code);

int32_t DocumentEditorSetTitle(DocumentEditor *handle, const char *value, int32_t *error_code);

int32_t DocumentEditorSetAuthor(DocumentEditor *handle, const char *value, int32_t *error_code);

void FreeString(char *ptr);

void FreeBytes(uint8_t *ptr);

char *AllocString(const char *s);

char *pdf_oxide_fonts_to_json(const FfiFontList *fonts, int32_t *error_code);

char *pdf_oxide_annotations_to_json(const FfiAnnotationList *annotations, int32_t *error_code);

char *pdf_oxide_elements_to_json(const FfiElementList *elements, int32_t *error_code);

char *pdf_oxide_search_results_to_json(const FfiSearchResults *results, int32_t *error_code);

void *pdf_ocr_engine_create(const char *det_model_path,
                            const char *rec_model_path,
                            const char *dict_path,
                            int32_t *error_code);

void pdf_ocr_engine_free(void *engine);

bool pdf_ocr_page_needs_ocr(PdfDocument *doc, int32_t page_index, int32_t *error_code);

char *pdf_ocr_extract_text(PdfDocument *doc,
                           int32_t page_index,
                           const void *engine,
                           int32_t *error_code);

EmbeddedFont *pdf_embedded_font_from_file(const char *path, int32_t *error_code);

EmbeddedFont *pdf_embedded_font_from_bytes(const uint8_t *data,
                                           uintptr_t len,
                                           const char *name,
                                           int32_t *error_code);

void pdf_embedded_font_free(EmbeddedFont *handle);

FfiDocumentBuilder *pdf_document_builder_create(int32_t *error_code);

void pdf_document_builder_free(FfiDocumentBuilder *handle);

int32_t pdf_document_builder_set_title(FfiDocumentBuilder *handle,
                                       const char *title,
                                       int32_t *error_code);

int32_t pdf_document_builder_set_author(FfiDocumentBuilder *handle,
                                        const char *author,
                                        int32_t *error_code);

int32_t pdf_document_builder_set_subject(FfiDocumentBuilder *handle,
                                         const char *subject,
                                         int32_t *error_code);

int32_t pdf_document_builder_set_keywords(FfiDocumentBuilder *handle,
                                          const char *keywords,
                                          int32_t *error_code);

int32_t pdf_document_builder_set_creator(FfiDocumentBuilder *handle,
                                         const char *creator,
                                         int32_t *error_code);

int32_t pdf_document_builder_on_open(FfiDocumentBuilder *handle,
                                     const char *script,
                                     int32_t *error_code);

int32_t pdf_document_builder_tagged_pdf_ua1(FfiDocumentBuilder *handle, int32_t *error_code);

int32_t pdf_document_builder_language(FfiDocumentBuilder *handle,
                                      const char *lang,
                                      int32_t *error_code);

int32_t pdf_document_builder_role_map(FfiDocumentBuilder *handle,
                                      const char *custom,
                                      const char *standard,
                                      int32_t *error_code);

int32_t pdf_document_builder_register_embedded_font(FfiDocumentBuilder *handle,
                                                    const char *name,
                                                    EmbeddedFont *font,
                                                    int32_t *error_code);

FfiPageBuilder *pdf_document_builder_a4_page(FfiDocumentBuilder *handle, int32_t *error_code);

FfiPageBuilder *pdf_document_builder_letter_page(FfiDocumentBuilder *handle, int32_t *error_code);

FfiPageBuilder *pdf_document_builder_page(FfiDocumentBuilder *handle,
                                          float width,
                                          float height,
                                          int32_t *error_code);

int32_t pdf_page_builder_font(FfiPageBuilder *handle,
                              const char *name,
                              float size,
                              int32_t *error_code);

int32_t pdf_page_builder_at(FfiPageBuilder *handle, float x, float y, int32_t *error_code);

int32_t pdf_page_builder_text(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_heading(FfiPageBuilder *handle,
                                 uint8_t level,
                                 const char *text,
                                 int32_t *error_code);

int32_t pdf_page_builder_paragraph(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_space(FfiPageBuilder *handle, float points, int32_t *error_code);

int32_t pdf_page_builder_horizontal_rule(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_link_url(FfiPageBuilder *handle, const char *url, int32_t *error_code);

int32_t pdf_page_builder_link_page(FfiPageBuilder *handle, uintptr_t page, int32_t *error_code);

int32_t pdf_page_builder_link_named(FfiPageBuilder *handle,
                                    const char *destination,
                                    int32_t *error_code);

int32_t pdf_page_builder_link_javascript(FfiPageBuilder *handle,
                                         const char *script,
                                         int32_t *error_code);

int32_t pdf_page_builder_on_open(FfiPageBuilder *handle, const char *script, int32_t *error_code);

int32_t pdf_page_builder_on_close(FfiPageBuilder *handle, const char *script, int32_t *error_code);

int32_t pdf_page_builder_field_keystroke(FfiPageBuilder *handle,
                                         const char *script,
                                         int32_t *error_code);

int32_t pdf_page_builder_field_format(FfiPageBuilder *handle,
                                      const char *script,
                                      int32_t *error_code);

int32_t pdf_page_builder_field_validate(FfiPageBuilder *handle,
                                        const char *script,
                                        int32_t *error_code);

int32_t pdf_page_builder_field_calculate(FfiPageBuilder *handle,
                                         const char *script,
                                         int32_t *error_code);

int32_t pdf_page_builder_highlight(FfiPageBuilder *handle,
                                   float r,
                                   float g,
                                   float b,
                                   int32_t *error_code);

int32_t pdf_page_builder_underline(FfiPageBuilder *handle,
                                   float r,
                                   float g,
                                   float b,
                                   int32_t *error_code);

int32_t pdf_page_builder_strikeout(FfiPageBuilder *handle,
                                   float r,
                                   float g,
                                   float b,
                                   int32_t *error_code);

int32_t pdf_page_builder_squiggly(FfiPageBuilder *handle,
                                  float r,
                                  float g,
                                  float b,
                                  int32_t *error_code);

int32_t pdf_page_builder_sticky_note(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_sticky_note_at(FfiPageBuilder *handle,
                                        float x,
                                        float y,
                                        const char *text,
                                        int32_t *error_code);

int32_t pdf_page_builder_watermark(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_watermark_confidential(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_watermark_draft(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_stamp(FfiPageBuilder *handle, const char *type_name, int32_t *error_code);

int32_t pdf_page_builder_freetext(FfiPageBuilder *handle,
                                  float x,
                                  float y,
                                  float w,
                                  float h,
                                  const char *text,
                                  int32_t *error_code);

int32_t pdf_page_builder_text_field(FfiPageBuilder *handle,
                                    const char *name,
                                    float x,
                                    float y,
                                    float w,
                                    float h,
                                    const char *default_value,
                                    int32_t *error_code);

int32_t pdf_page_builder_checkbox(FfiPageBuilder *handle,
                                  const char *name,
                                  float x,
                                  float y,
                                  float w,
                                  float h,
                                  int32_t checked,
                                  int32_t *error_code);

int32_t pdf_page_builder_combo_box(FfiPageBuilder *handle,
                                   const char *name,
                                   float x,
                                   float y,
                                   float w,
                                   float h,
                                   const char *const *options,
                                   uintptr_t options_count,
                                   const char *selected,
                                   int32_t *error_code);

int32_t pdf_page_builder_radio_group(FfiPageBuilder *handle,
                                     const char *name,
                                     const char *const *values,
                                     const float *xs,
                                     const float *ys,
                                     const float *ws,
                                     const float *hs,
                                     uintptr_t count,
                                     const char *selected,
                                     int32_t *error_code);

int32_t pdf_page_builder_push_button(FfiPageBuilder *handle,
                                     const char *name,
                                     float x,
                                     float y,
                                     float w,
                                     float h,
                                     const char *caption,
                                     int32_t *error_code);

int32_t pdf_page_builder_signature_field(FfiPageBuilder *handle,
                                         const char *name,
                                         float x,
                                         float y,
                                         float w,
                                         float h,
                                         int32_t *error_code);

int32_t pdf_page_builder_footnote(FfiPageBuilder *handle,
                                  const char *ref_mark,
                                  const char *note_text,
                                  int32_t *error_code);

int32_t pdf_page_builder_columns(FfiPageBuilder *handle,
                                 uint32_t column_count,
                                 float gap_pt,
                                 const char *text,
                                 int32_t *error_code);

int32_t pdf_page_builder_inline(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_inline_bold(FfiPageBuilder *handle, const char *text, int32_t *error_code);

int32_t pdf_page_builder_inline_italic(FfiPageBuilder *handle,
                                       const char *text,
                                       int32_t *error_code);

int32_t pdf_page_builder_inline_color(FfiPageBuilder *handle,
                                      float r,
                                      float g,
                                      float b,
                                      const char *text,
                                      int32_t *error_code);

int32_t pdf_page_builder_newline(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_barcode_1d(FfiPageBuilder *handle,
                                    int32_t barcode_type,
                                    const char *data,
                                    float x,
                                    float y,
                                    float w,
                                    float h,
                                    int32_t *error_code);

int32_t pdf_page_builder_barcode_qr(FfiPageBuilder *handle,
                                    const char *data,
                                    float x,
                                    float y,
                                    float size,
                                    int32_t *error_code);

int32_t pdf_page_builder_image(FfiPageBuilder *handle,
                               const uint8_t *bytes,
                               uintptr_t len,
                               float x,
                               float y,
                               float w,
                               float h,
                               int32_t *error_code);

int32_t pdf_page_builder_image_with_alt(FfiPageBuilder *handle,
                                        const uint8_t *bytes,
                                        uintptr_t len,
                                        float x,
                                        float y,
                                        float w,
                                        float h,
                                        const char *alt_text,
                                        int32_t *error_code);

int32_t pdf_page_builder_image_artifact(FfiPageBuilder *handle,
                                        const uint8_t *bytes,
                                        uintptr_t len,
                                        float x,
                                        float y,
                                        float w,
                                        float h,
                                        int32_t *error_code);

int32_t pdf_page_builder_rect(FfiPageBuilder *handle,
                              float x,
                              float y,
                              float w,
                              float h,
                              int32_t *error_code);

int32_t pdf_page_builder_filled_rect(FfiPageBuilder *handle,
                                     float x,
                                     float y,
                                     float w,
                                     float h,
                                     float r,
                                     float g,
                                     float b,
                                     int32_t *error_code);

int32_t pdf_page_builder_line(FfiPageBuilder *handle,
                              float x1,
                              float y1,
                              float x2,
                              float y2,
                              int32_t *error_code);

int32_t pdf_page_builder_stroke_rect(FfiPageBuilder *handle,
                                     float x,
                                     float y,
                                     float w,
                                     float h,
                                     float width,
                                     float r,
                                     float g,
                                     float b,
                                     int32_t *error_code);

int32_t pdf_page_builder_stroke_line(FfiPageBuilder *handle,
                                     float x1,
                                     float y1,
                                     float x2,
                                     float y2,
                                     float width,
                                     float r,
                                     float g,
                                     float b,
                                     int32_t *error_code);

int32_t pdf_page_builder_stroke_rect_dashed(FfiPageBuilder *handle,
                                            float x,
                                            float y,
                                            float w,
                                            float h,
                                            float width,
                                            float r,
                                            float g,
                                            float b,
                                            const float *dash_array,
                                            uintptr_t n_dash,
                                            float phase,
                                            int32_t *error_code);

int32_t pdf_page_builder_stroke_line_dashed(FfiPageBuilder *handle,
                                            float x1,
                                            float y1,
                                            float x2,
                                            float y2,
                                            float width,
                                            float r,
                                            float g,
                                            float b,
                                            const float *dash_array,
                                            uintptr_t n_dash,
                                            float phase,
                                            int32_t *error_code);

int32_t pdf_page_builder_text_in_rect(FfiPageBuilder *handle,
                                      float x,
                                      float y,
                                      float w,
                                      float h,
                                      const char *text,
                                      int32_t align,
                                      int32_t *error_code);

int32_t pdf_page_builder_new_page_same_size(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_table(FfiPageBuilder *handle,
                               uintptr_t n_columns,
                               const float *widths,
                               const int32_t *aligns,
                               uintptr_t n_rows,
                               const char *const *cell_strings,
                               int32_t has_header,
                               int32_t *error_code);

int32_t pdf_page_builder_streaming_table_begin(FfiPageBuilder *handle,
                                               uintptr_t n_columns,
                                               const char *const *headers,
                                               const float *widths,
                                               const int32_t *aligns,
                                               int32_t repeat_header,
                                               int32_t *error_code);

int32_t pdf_page_builder_streaming_table_begin_v2(FfiPageBuilder *handle,
                                                  uintptr_t n_columns,
                                                  const char *const *headers,
                                                  const float *widths,
                                                  const int32_t *aligns,
                                                  int32_t repeat_header,
                                                  int32_t mode,
                                                  uintptr_t sample_rows,
                                                  float min_col_width_pt,
                                                  float max_col_width_pt,
                                                  uintptr_t max_rowspan,
                                                  int32_t *error_code);

int32_t pdf_page_builder_streaming_table_set_batch_size(FfiPageBuilder *handle,
                                                        uintptr_t batch_size,
                                                        int32_t *error_code);

uintptr_t pdf_page_builder_streaming_table_pending_row_count(FfiPageBuilder *handle);

uintptr_t pdf_page_builder_streaming_table_batch_count(FfiPageBuilder *handle);

int32_t pdf_page_builder_streaming_table_flush(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_streaming_table_push_row(FfiPageBuilder *handle,
                                                  uintptr_t n_cells,
                                                  const char *const *cells,
                                                  int32_t *error_code);

int32_t pdf_page_builder_streaming_table_push_row_v2(FfiPageBuilder *handle,
                                                     uintptr_t n_cells,
                                                     const char *const *cells,
                                                     const uintptr_t *rowspans,
                                                     int32_t *error_code);

int32_t pdf_page_builder_streaming_table_finish(FfiPageBuilder *handle, int32_t *error_code);

int32_t pdf_page_builder_done(FfiPageBuilder *handle, int32_t *error_code);

void pdf_page_builder_free(FfiPageBuilder *handle);

uint8_t *pdf_document_builder_build(FfiDocumentBuilder *handle,
                                    uintptr_t *out_len,
                                    int32_t *error_code);

int32_t pdf_document_builder_save(FfiDocumentBuilder *handle,
                                  const char *path,
                                  int32_t *error_code);

int32_t pdf_document_builder_save_encrypted(FfiDocumentBuilder *handle,
                                            const char *path,
                                            const char *user_password,
                                            const char *owner_password,
                                            int32_t *error_code);

uint8_t *pdf_document_builder_to_bytes_encrypted(FfiDocumentBuilder *handle,
                                                 const char *user_password,
                                                 const char *owner_password,
                                                 uintptr_t *out_len,
                                                 int32_t *error_code);

Pdf *pdf_from_html_css(const char *html,
                       const char *css,
                       const uint8_t *font_bytes,
                       uintptr_t font_len,
                       int32_t *error_code);

Pdf *pdf_from_html_css_with_fonts(const char *html,
                                  const char *css,
                                  const char *const *families,
                                  const uint8_t *const *font_bytes,
                                  const uintptr_t *font_lens,
                                  uintptr_t count,
                                  int32_t *error_code);
