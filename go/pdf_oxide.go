//go:build cgo

package pdfoxide

// Linking configuration — v0.3.31 onwards.
//
// This file declares only the C function prototypes (the CGo preamble below).
// `#cgo LDFLAGS` directives live in SEPARATE files so we can pick the right
// link path without committing native libraries to the module:
//
//   cgo_dev.go     — built under `//go:build pdf_oxide_dev`. Points at the
//                    Cargo workspace target/ dir. Used inside the monorepo
//                    after `cargo build --release --lib`.
//   cgo_flags.go   — OPTIONAL, generated locally by `cmd/install` for users
//                    who want a committed-per-machine file. Not shipped.
//   (no file)      — consumer exports CGO_CFLAGS / CGO_LDFLAGS after running
//                    `go run github.com/yfedoseev/pdf_oxide/go/cmd/install`.
//                    The installer prints the exact values to export.
//
// Background: static-linking the rustc-produced `libpdf_oxide.a` was adding
// ~310 MB to git history per release (6 platforms × ~50 MB). Shipping the
// archives via GitHub Releases and downloading on demand removes that bloat
// without changing the final binary's runtime characteristics (still a
// self-contained Go binary — no `LD_LIBRARY_PATH` needed).
//
// Regenerate the system-library list per target via:
//   cargo rustc --release --lib --target <triple> -- --print native-static-libs
// The exact list is baked into cmd/install/main.go.
//
// Windows ARM64: still dynamic — must ship pdf_oxide.dll alongside the exe.

/*
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

extern void* pdf_document_open(const char* path, int* error_code);
extern void pdf_document_free(void* handle);
extern int32_t pdf_document_get_page_count(void* handle, int* error_code);
extern void pdf_document_get_version(const void* handle, uint8_t* major, uint8_t* minor);
extern bool pdf_document_has_structure_tree(void* handle);
extern char* pdf_document_extract_text(void* handle, int32_t page_index, int* error_code);
extern char* pdf_document_to_markdown(void* handle, int32_t page_index, int* error_code);
extern char* pdf_document_to_html(void* handle, int32_t page_index, int* error_code);
extern char* pdf_document_to_plain_text(void* handle, int32_t page_index, int* error_code);
extern char* pdf_document_to_markdown_all(void* handle, int* error_code);

// Document Editor FFI declarations
extern void* document_editor_open(const char* path, int* error_code);
extern void document_editor_free(void* handle);
extern bool document_editor_is_modified(const void* handle);
extern char* document_editor_get_source_path(const void* handle, int* error_code);
extern void document_editor_get_version(const void* handle, uint8_t* major, uint8_t* minor);
extern int32_t document_editor_get_page_count(void* handle, int* error_code);
extern char* document_editor_get_title(const void* handle, int* error_code);
extern int document_editor_set_title(void* handle, const char* title, int* error_code);
extern char* document_editor_get_author(const void* handle, int* error_code);
extern int document_editor_set_author(void* handle, const char* author, int* error_code);
extern char* document_editor_get_subject(const void* handle, int* error_code);
extern int document_editor_set_subject(void* handle, const char* subject, int* error_code);
extern char* document_editor_get_producer(const void* handle, int* error_code);
extern int document_editor_set_producer(void* handle, const char* producer, int* error_code);
extern char* document_editor_get_creation_date(const void* handle, int* error_code);
extern int document_editor_set_creation_date(void* handle, const char* date_str, int* error_code);
extern int document_editor_save(void* handle, const char* path, int* error_code);

// PDF Creator FFI declarations
extern void* pdf_from_markdown(const char* markdown, int* error_code);
extern void* pdf_from_html(const char* html, int* error_code);
extern void* pdf_from_text(const char* text, int* error_code);
extern int pdf_save(void* handle, const char* path, int* error_code);
extern void* pdf_save_to_bytes(void* handle, int* data_len, int* error_code);
extern int32_t pdf_get_page_count(void* handle, int* error_code);
extern void pdf_free(void* handle);

// Search FFI declarations
extern void* pdf_document_search_page(void* handle, int32_t page_index, const char* search_term, bool case_sensitive, int* error_code);
extern void* pdf_document_search_all(void* handle, const char* search_term, bool case_sensitive, int* error_code);
extern char* pdf_oxide_search_results_to_json(const void* results, int* error_code);
extern void pdf_oxide_search_result_free(void* handle);

// JSON bulk extractors (one FFI crossing per list, vs N*M per-field calls)
extern char* pdf_oxide_fonts_to_json(const void* fonts, int* error_code);
extern char* pdf_oxide_annotations_to_json(const void* annotations, int* error_code);
extern char* pdf_oxide_elements_to_json(const void* elements, int* error_code);

// Font extraction FFI declarations
extern void* pdf_document_get_embedded_fonts(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_font_count(const void* fonts);
extern char* pdf_oxide_font_get_name(const void* fonts, int32_t index, int* error_code);
extern char* pdf_oxide_font_get_type(const void* fonts, int32_t index, int* error_code);
extern char* pdf_oxide_font_get_encoding(const void* fonts, int32_t index, int* error_code);
extern int pdf_oxide_font_is_embedded(const void* fonts, int32_t index, int* error_code);
extern int pdf_oxide_font_is_subset(const void* fonts, int32_t index, int* error_code);
extern float pdf_oxide_font_get_size(const void* fonts, int32_t index, int* error_code);
extern void pdf_oxide_font_list_free(void* handle);

// Image extraction FFI declarations
extern void* pdf_document_get_embedded_images(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_image_count(const void* images);
extern int32_t pdf_oxide_image_get_width(const void* images, int32_t index, int* error_code);
extern int32_t pdf_oxide_image_get_height(const void* images, int32_t index, int* error_code);
extern char* pdf_oxide_image_get_format(const void* images, int32_t index, int* error_code);
extern char* pdf_oxide_image_get_colorspace(const void* images, int32_t index, int* error_code);
extern int32_t pdf_oxide_image_get_bits_per_component(const void* images, int32_t index, int* error_code);
extern void* pdf_oxide_image_get_data(const void* images, int32_t index, int* data_len, int* error_code);
extern void pdf_oxide_image_list_free(void* handle);

// Annotation extraction FFI declarations
extern void* pdf_document_get_page_annotations(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_annotation_count(const void* annotations);
extern char* pdf_oxide_annotation_get_type(const void* annotations, int32_t index, int* error_code);
extern char* pdf_oxide_annotation_get_content(const void* annotations, int32_t index, int* error_code);
extern void pdf_oxide_annotation_get_rect(const void* annotations, int32_t index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_oxide_annotation_list_free(void* handle);

// Page operations FFI declarations
extern float pdf_page_get_width(void* handle, int32_t page_index, int* error_code);
extern float pdf_page_get_height(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_page_get_rotation(void* handle, int32_t page_index, int* error_code);
extern void pdf_page_get_media_box(void* handle, int32_t page_index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_page_get_crop_box(void* handle, int32_t page_index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_page_get_art_box(void* handle, int32_t page_index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_page_get_bleed_box(void* handle, int32_t page_index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_page_get_trim_box(void* handle, int32_t page_index, float* x, float* y, float* width, float* height, int* error_code);
extern void* pdf_page_get_elements(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_element_count(const void* elements);
extern char* pdf_oxide_element_get_type(const void* elements, int32_t index, int* error_code);
extern char* pdf_oxide_element_get_text(const void* elements, int32_t index, int* error_code);
extern void pdf_oxide_element_get_rect(const void* elements, int32_t index, float* x, float* y, float* width, float* height, int* error_code);
extern void pdf_oxide_elements_free(void* handle);

// Advanced annotation FFI declarations
extern char* pdf_oxide_annotation_get_subtype(const void* annotations, int32_t index, int* error_code);
extern bool pdf_oxide_annotation_is_marked_deleted(const void* annotations, int32_t index, int* error_code);
extern int64_t pdf_oxide_annotation_get_creation_date(const void* annotations, int32_t index, int* error_code);
extern int64_t pdf_oxide_annotation_get_modification_date(const void* annotations, int32_t index, int* error_code);
extern char* pdf_oxide_annotation_get_author(const void* annotations, int32_t index, int* error_code);
extern float pdf_oxide_annotation_get_border_width(const void* annotations, int32_t index, int* error_code);
extern uint32_t pdf_oxide_annotation_get_color(const void* annotations, int32_t index, int* error_code);
extern bool pdf_oxide_annotation_is_hidden(const void* annotations, int32_t index, int* error_code);
extern bool pdf_oxide_annotation_is_printable(const void* annotations, int32_t index, int* error_code);
extern bool pdf_oxide_annotation_is_read_only(const void* annotations, int32_t index, int* error_code);
extern char* pdf_oxide_link_annotation_get_uri(const void* annotations, int32_t index, int* error_code);
extern char* pdf_oxide_text_annotation_get_icon_name(const void* annotations, int32_t index, int* error_code);
extern int32_t pdf_oxide_highlight_annotation_get_quad_points_count(const void* annotations, int32_t index, int* error_code);
extern void pdf_oxide_highlight_annotation_get_quad_point(const void* annotations, int32_t index, int32_t quad_index, float* x1, float* y1, float* x2, float* y2, float* x3, float* y3, float* x4, float* y4, int* error_code);

// Barcodes FFI declarations (9 functions)
extern void* pdf_generate_qr_code(const char* data, int error_correction, int32_t size_px, int* error_code);
extern void* pdf_generate_barcode(const char* data, int format, int32_t size_px, int* error_code);
extern uint8_t* pdf_barcode_get_image_png(const void* barcode_handle, int32_t size_px, int32_t* out_len, int* error_code);
extern char* pdf_barcode_get_svg(const void* barcode_handle, int32_t size_px, int* error_code);
extern int pdf_add_barcode_to_page(void* document_handle, int32_t page_index, const void* barcode_handle, float x, float y, float width, float height, int* error_code);
extern int pdf_barcode_get_format(const void* barcode_handle, int* error_code);
extern char* pdf_barcode_get_data(const void* barcode_handle, int* error_code);
extern float pdf_barcode_get_confidence(const void* barcode_handle, int* error_code);
extern void pdf_barcode_free(void* handle);

// Signatures FFI declarations (19 functions)
extern void* pdf_certificate_load_from_bytes(const uint8_t* cert_bytes, int32_t cert_len, const char* password, int* error_code);
extern void* pdf_certificate_load_from_pem(const char* cert_pem, const char* key_pem, int* error_code);
extern int pdf_document_sign(void* document_handle, const void* certificate_handle, const char* reason, const char* location, int* error_code);
extern int32_t pdf_document_get_signature_count(const void* document_handle, int* error_code);
extern void* pdf_document_get_signature(const void* document_handle, int32_t index, int* error_code);
extern int pdf_signature_verify(const void* signature_handle, int* error_code);
extern int pdf_signature_verify_detached(const void* signature_handle, const unsigned char* pdf_data, size_t pdf_len, int* error_code);
extern int pdf_document_verify_all_signatures(const void* document_handle, int* error_code);
extern char* pdf_signature_get_signer_name(const void* signature_handle, int* error_code);
extern int64_t pdf_signature_get_signing_time(const void* signature_handle, int* error_code);
extern char* pdf_signature_get_signing_reason(const void* signature_handle, int* error_code);
extern char* pdf_signature_get_signing_location(const void* signature_handle, int* error_code);
extern void* pdf_signature_get_certificate(const void* signature_handle, int* error_code);
extern char* pdf_certificate_get_subject(const void* certificate_handle, int* error_code);
extern char* pdf_certificate_get_issuer(const void* certificate_handle, int* error_code);
extern char* pdf_certificate_get_serial(const void* certificate_handle, int* error_code);
extern void pdf_certificate_get_validity(const void* certificate_handle, int64_t* not_before, int64_t* not_after, int* error_code);
extern int pdf_certificate_is_valid(const void* certificate_handle, int* error_code);
extern void pdf_signature_free(void* handle);
extern void pdf_certificate_free(void* handle);
extern uint8_t* pdf_sign_bytes(const uint8_t* pdf_data, size_t pdf_len, const void* certificate_handle, const char* reason, const char* location, size_t* out_len, int* error_code);

// Rendering FFI declarations (21 functions)
extern int32_t pdf_estimate_render_time(const void* document_handle, int32_t page_index, int* error_code);
extern void* pdf_create_renderer(int32_t dpi, int32_t format, int32_t quality, bool anti_alias, int* error_code);
extern void* pdf_render_page(void* document_handle, int32_t page_index, int32_t format, int* error_code);
extern void* pdf_render_page_region(void* document_handle, int32_t page_index, float crop_x, float crop_y, float crop_width, float crop_height, int32_t format, int* error_code);
extern void* pdf_render_page_zoom(void* document_handle, int32_t page_index, float zoom_level, int32_t format, int* error_code);
extern void* pdf_render_page_fit(void* document_handle, int32_t page_index, int32_t fit_width, int32_t fit_height, int32_t format, int* error_code);
extern void* pdf_render_page_thumbnail(void* document_handle, int32_t page_index, int32_t thumbnail_size, int32_t format, int* error_code);
extern void* pdf_render_page_with_options(void* document_handle, int32_t page_index, int32_t dpi, int32_t format, float bg_r, float bg_g, float bg_b, float bg_a, int32_t transparent_background, int32_t render_annotations, int32_t jpeg_quality, int* error_code);
extern int32_t pdf_get_rendered_image_width(const void* image_handle, int* error_code);
extern int32_t pdf_get_rendered_image_height(const void* image_handle, int* error_code);
extern void* pdf_get_rendered_image_data(const void* image_handle, int32_t* data_len, int* error_code);
extern int pdf_save_rendered_image(const void* image_handle, const char* file_path, int* error_code);
extern void pdf_rendered_image_free(void* handle);
extern void* pdf_render_page_raw(void* document_handle, int32_t page_index, int32_t dpi, int32_t* out_width, int32_t* out_height, int* error_code);
extern void pdf_renderer_free(void* handle);

// TSA (Time Stamp Authority) FFI declarations
extern void* pdf_timestamp_parse(const uint8_t* bytes, size_t len, int* error_code);
extern void* pdf_tsa_client_create(const char* url, const char* username, const char* password, int32_t timeout, int32_t hash_algo, bool use_nonce, bool cert_req, int* error_code);
extern void pdf_tsa_client_free(void* client);
extern void* pdf_tsa_request_timestamp(const void* client, const uint8_t* data, size_t data_len, int* error_code);
extern void* pdf_tsa_request_timestamp_hash(const void* client, const uint8_t* hash, size_t hash_len, int32_t hash_algo, int* error_code);
extern const uint8_t* pdf_timestamp_get_token(const void* timestamp, size_t* out_len, int* error_code);
extern int64_t pdf_timestamp_get_time(const void* timestamp, int* error_code);
extern char* pdf_timestamp_get_serial(const void* timestamp, int* error_code);
extern char* pdf_timestamp_get_tsa_name(const void* timestamp, int* error_code);
extern char* pdf_timestamp_get_policy_oid(const void* timestamp, int* error_code);
extern int32_t pdf_timestamp_get_hash_algorithm(const void* timestamp, int* error_code);
extern const uint8_t* pdf_timestamp_get_message_imprint(const void* timestamp, size_t* out_len, int* error_code);
extern bool pdf_timestamp_verify(const void* timestamp, int* error_code);
extern void pdf_timestamp_free(void* timestamp);
extern bool pdf_signature_add_timestamp(const void* signature, const void* timestamp, int* error_code);
extern bool pdf_signature_has_timestamp(const void* signature, int* error_code);
extern void* pdf_signature_get_timestamp(const void* signature, int* error_code);
extern bool pdf_add_timestamp(const uint8_t* pdf_data, size_t pdf_len, int32_t signature_index, const char* tsa_url, uint8_t** out_data, size_t* out_len, int* error_code);

// PDF/UA Validation FFI declarations
extern void* pdf_validate_pdf_ua(const void* document, int32_t level, int* error_code);
extern bool pdf_pdf_ua_is_accessible(const void* results, int* error_code);
extern int32_t pdf_pdf_ua_error_count(const void* results);
extern char* pdf_pdf_ua_get_error(const void* results, int32_t index, int* error_code);
extern int32_t pdf_pdf_ua_warning_count(const void* results);
extern void* pdf_pdf_ua_get_warning(const void* results, int32_t index, int* error_code);
extern bool pdf_pdf_ua_get_stats(const void* results, int32_t* out_struct, int32_t* out_images, int32_t* out_tables, int32_t* out_forms, int32_t* out_annotations, int32_t* out_pages, int* error_code);
extern void pdf_pdf_ua_results_free(void* results);

// FDF/XFDF Import/Export FFI declarations
extern bool pdf_form_import_from_file(const void* document, const char* filename, int* error_code);
extern int32_t pdf_document_import_form_data(const void* document, const char* data_path, int* error_code);
extern int32_t pdf_editor_import_fdf_bytes(const void* document, const uint8_t* data, size_t data_len, int* error_code);
extern int32_t pdf_editor_import_xfdf_bytes(const void* document, const uint8_t* data, size_t data_len, int* error_code);
extern uint8_t* pdf_document_export_form_data_to_bytes(const void* document, int32_t format_type, size_t* out_len, int* error_code);

// New FFI functions (v0.3.24)
extern void* pdf_document_open_from_bytes(const uint8_t* data, size_t len, int* error_code);
extern void* pdf_document_open_with_password(const char* path, const char* password, int* error_code);
extern bool pdf_document_is_encrypted(const void* handle);
extern bool pdf_document_authenticate(void* handle, const char* password, int* error_code);
extern char* pdf_document_extract_all_text(void* handle, int* error_code);
extern char* pdf_document_to_html_all(void* handle, int* error_code);
extern char* pdf_document_to_plain_text_all(void* handle, int* error_code);

// Granular extraction
extern void* pdf_document_extract_chars(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_char_count(const void* chars);
extern uint32_t pdf_oxide_char_get_char(const void* chars, int32_t index, int* error_code);
extern void pdf_oxide_char_get_bbox(const void* chars, int32_t index, float* x, float* y, float* w, float* h, int* error_code);
extern char* pdf_oxide_char_get_font_name(const void* chars, int32_t index, int* error_code);
extern float pdf_oxide_char_get_font_size(const void* chars, int32_t index, int* error_code);
extern void pdf_oxide_char_list_free(void* handle);

extern void* pdf_document_extract_words(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_word_count(const void* words);
extern char* pdf_oxide_word_get_text(const void* words, int32_t index, int* error_code);
extern void pdf_oxide_word_get_bbox(const void* words, int32_t index, float* x, float* y, float* w, float* h, int* error_code);
extern char* pdf_oxide_word_get_font_name(const void* words, int32_t index, int* error_code);
extern float pdf_oxide_word_get_font_size(const void* words, int32_t index, int* error_code);
extern bool pdf_oxide_word_is_bold(const void* words, int32_t index, int* error_code);
extern void pdf_oxide_word_list_free(void* handle);

extern void* pdf_document_extract_text_lines(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_line_count(const void* lines);
extern char* pdf_oxide_line_get_text(const void* lines, int32_t index, int* error_code);
extern void pdf_oxide_line_get_bbox(const void* lines, int32_t index, float* x, float* y, float* w, float* h, int* error_code);
extern int32_t pdf_oxide_line_get_word_count(const void* lines, int32_t index, int* error_code);
extern void pdf_oxide_line_list_free(void* handle);

extern void* pdf_document_extract_tables(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_table_count(const void* tables);
extern int32_t pdf_oxide_table_get_row_count(const void* tables, int32_t index, int* error_code);
extern int32_t pdf_oxide_table_get_col_count(const void* tables, int32_t index, int* error_code);
extern char* pdf_oxide_table_get_cell_text(const void* tables, int32_t table_index, int32_t row, int32_t col, int* error_code);
extern bool pdf_oxide_table_has_header(const void* tables, int32_t index, int* error_code);
extern void pdf_oxide_table_list_free(void* handle);

// Region extraction
extern char* pdf_document_extract_text_in_rect(void* handle, int32_t page_index, float x, float y, float w, float h, int* error_code);
extern void* pdf_document_extract_words_in_rect(void* handle, int32_t page_index, float x, float y, float w, float h, int* error_code);
extern void* pdf_document_extract_images_in_rect(void* handle, int32_t page_index, float x, float y, float w, float h, int* error_code);

// Forms
extern void* pdf_document_get_form_fields(void* handle, int* error_code);
extern int32_t pdf_oxide_form_field_count(const void* fields);
extern char* pdf_oxide_form_field_get_name(const void* fields, int32_t index, int* error_code);
extern char* pdf_oxide_form_field_get_type(const void* fields, int32_t index, int* error_code);
extern char* pdf_oxide_form_field_get_value(const void* fields, int32_t index, int* error_code);
extern bool pdf_oxide_form_field_is_readonly(const void* fields, int32_t index, int* error_code);
extern bool pdf_oxide_form_field_is_required(const void* fields, int32_t index, int* error_code);
extern void pdf_oxide_form_field_list_free(void* handle);
extern bool pdf_document_has_xfa(void* handle);

// Artifact removal
extern int32_t pdf_document_remove_headers(void* handle, float threshold, int* error_code);
extern int32_t pdf_document_remove_footers(void* handle, float threshold, int* error_code);
extern int32_t pdf_document_remove_artifacts(void* handle, float threshold, int* error_code);
extern int32_t pdf_document_erase_header(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_document_erase_footer(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_document_erase_artifacts(void* handle, int32_t page_index, int* error_code);

// Editor: page operations
extern int32_t document_editor_delete_page(void* handle, int32_t page_index, int* error_code);
extern int32_t document_editor_move_page(void* handle, int32_t from, int32_t to, int* error_code);
extern int32_t document_editor_get_page_rotation(void* handle, int32_t page, int* error_code);
extern int32_t document_editor_set_page_rotation(void* handle, int32_t page, int32_t degrees, int* error_code);
extern int32_t document_editor_erase_region(void* handle, int32_t page, float x, float y, float w, float h, int* error_code);
extern int32_t document_editor_flatten_annotations(void* handle, int32_t page, int* error_code);
extern int32_t document_editor_flatten_all_annotations(void* handle, int* error_code);
extern int32_t document_editor_crop_margins(void* handle, float left, float right, float top, float bottom, int* error_code);
extern int32_t document_editor_merge_from(void* handle, const char* source_path, int* error_code);
extern int32_t document_editor_save_encrypted(void* handle, const char* path, const char* user_password, const char* owner_password, int* error_code);

// Editor: new functions (v0.3.39)
extern void* document_editor_open_from_bytes(const uint8_t* data, size_t len, int* error_code);
extern uint8_t* document_editor_save_to_bytes(void* handle, size_t* out_len, int* error_code);
extern uint8_t* document_editor_save_to_bytes_with_options(void* handle, bool compress, bool garbage_collect, bool linearize, size_t* out_len, int* error_code);

// Editor: v0.3.40 additions
extern void* document_editor_extract_pages_to_bytes(void* handle, int32_t* pages, size_t count, size_t* out_len, int* error_code);
extern int document_editor_convert_to_pdf_a(void* handle, int32_t level, int* error_code);
extern void* document_editor_save_encrypted_to_bytes(void* handle, const char* user_password, const char* owner_password, size_t* out_len, int* error_code);
extern char* document_editor_get_keywords(const void* handle, int* error_code);
extern int   document_editor_set_keywords(void* handle, const char* keywords, int* error_code);
extern int32_t document_editor_merge_from_bytes(void* handle, const uint8_t* data, size_t len, int* error_code);
extern int   document_editor_embed_file(void* handle, const char* name, const uint8_t* data, size_t len, int* error_code);
extern int   document_editor_apply_page_redactions(void* handle, size_t page, int* error_code);
extern int   document_editor_apply_all_redactions(void* handle, int* error_code);
extern int   document_editor_rotate_all_pages(void* handle, int32_t degrees, int* error_code);
extern int   document_editor_rotate_page_by(void* handle, size_t page, int32_t degrees, int* error_code);
extern int   document_editor_get_page_media_box(void* handle, size_t page, double* x, double* y, double* w, double* h, int* error_code);
extern int   document_editor_set_page_media_box(void* handle, size_t page, double x, double y, double w, double h, int* error_code);
extern int   document_editor_get_page_crop_box(void* handle, size_t page, double* x, double* y, double* w, double* h, int* error_code);
extern int   document_editor_set_page_crop_box(void* handle, size_t page, double x, double y, double w, double h, int* error_code);
extern int   document_editor_erase_regions(void* handle, size_t page, const double* rects, size_t rects_count, int* error_code);
extern int   document_editor_clear_erase_regions(void* handle, size_t page, int* error_code);
extern int32_t document_editor_is_page_marked_for_flatten(const void* handle, size_t page);
extern int   document_editor_unmark_page_for_flatten(void* handle, size_t page, int* error_code);
extern int32_t document_editor_is_page_marked_for_redaction(const void* handle, size_t page);
extern int   document_editor_unmark_page_for_redaction(void* handle, size_t page, int* error_code);

// PDF creation extras
extern void* pdf_from_image(const char* path, int* error_code);
extern void* pdf_from_image_bytes(const uint8_t* data, int32_t data_len, int* error_code);
extern void* pdf_merge(const char** paths, int32_t path_count, int32_t* data_len, int* error_code);

// Compliance
extern void* pdf_validate_pdf_a_level(void* document, int32_t level, int* error_code);
extern bool pdf_pdf_a_is_compliant(const void* results, int* error_code);
extern int32_t pdf_pdf_a_error_count(const void* results);
extern int32_t pdf_pdf_a_warning_count(const void* results);
extern char* pdf_pdf_a_get_error(const void* results, int32_t index, int* error_code);
extern void pdf_pdf_a_results_free(void* results);

extern void* pdf_validate_pdf_x_level(void* document, int32_t level, int* error_code);
extern bool pdf_pdf_x_is_compliant(const void* results, int* error_code);
extern int32_t pdf_pdf_x_error_count(const void* results);
extern char* pdf_pdf_x_get_error(const void* results, int32_t index, int* error_code);
extern void pdf_pdf_x_results_free(void* results);

// Paths, labels, XMP, outline
extern void* pdf_document_extract_paths(void* handle, int32_t page_index, int* error_code);
extern int32_t pdf_oxide_path_count(const void* paths);
extern void pdf_oxide_path_get_bbox(const void* paths, int32_t index, float* x, float* y, float* w, float* h, int* error_code);
extern float pdf_oxide_path_get_stroke_width(const void* paths, int32_t index, int* error_code);
extern bool pdf_oxide_path_has_stroke(const void* paths, int32_t index, int* error_code);
extern bool pdf_oxide_path_has_fill(const void* paths, int32_t index, int* error_code);
extern int32_t pdf_oxide_path_get_operation_count(const void* paths, int32_t index, int* error_code);
extern void pdf_oxide_path_list_free(void* handle);
extern char* pdf_document_get_page_labels(void* handle, int* error_code);
extern char* pdf_document_get_xmp_metadata(void* handle, int* error_code);
extern char* pdf_document_get_outline(void* handle, int* error_code);

// Rendering
extern void* pdf_render_page(void* doc, int32_t page_index, int32_t format, int* error_code);
extern void* pdf_render_page_zoom(void* doc, int32_t page_index, float zoom, int32_t format, int* error_code);
extern void* pdf_render_page_fit(void* doc, int32_t page_index, int32_t fit_width, int32_t fit_height, int32_t format, int* error_code);
extern void* pdf_render_page_thumbnail(void* doc, int32_t page_index, int32_t size, int32_t format, int* error_code);
extern int32_t pdf_get_rendered_image_width(const void* img, int* error_code);
extern int32_t pdf_get_rendered_image_height(const void* img, int* error_code);
extern void* pdf_get_rendered_image_data(const void* img, int32_t* data_len, int* error_code);
extern int pdf_save_rendered_image(const void* img, const char* file_path, int* error_code);
extern void pdf_rendered_image_free(void* handle);
extern void* pdf_render_page_raw(void* doc, int32_t page_index, int32_t dpi, int32_t* out_width, int32_t* out_height, int* error_code);

// Barcodes
extern void* pdf_generate_qr_code(const char* data, int error_correction, int32_t size_px, int* error_code);
extern void* pdf_generate_barcode(const char* data, int format, int32_t size_px, int* error_code);
extern uint8_t* pdf_barcode_get_image_png(const void* barcode_handle, int32_t size_px, int32_t* out_len, int* error_code);
extern char* pdf_barcode_get_data(const void* barcode_handle, int* error_code);
extern int pdf_barcode_get_format(const void* barcode_handle, int* error_code);
extern void pdf_barcode_free(void* handle);

// Signatures
extern void* pdf_certificate_load_from_bytes(const uint8_t* cert_bytes, int32_t cert_len, const char* password, int* error_code);
extern void* pdf_certificate_load_from_pem(const char* cert_pem, const char* key_pem, int* error_code);
extern void pdf_certificate_free(void* handle);
extern char* pdf_signature_get_signer_name(const void* sig, int* error_code);
extern char* pdf_signature_get_signing_reason(const void* sig, int* error_code);
extern char* pdf_signature_get_signing_location(const void* sig, int* error_code);
extern void pdf_signature_free(void* handle);
extern uint8_t* pdf_sign_bytes(const uint8_t* pdf_data, size_t pdf_len, const void* certificate_handle, const char* reason, const char* location, size_t* out_len, int* error_code);

// Form mutation + flatten
extern int32_t document_editor_set_form_field_value(void* handle, const char* name, const char* value, int* error_code);
extern int32_t document_editor_flatten_forms(void* handle, int* error_code);
extern int32_t document_editor_flatten_forms_on_page(void* handle, int32_t page_index, int* error_code);
extern int32_t document_editor_flatten_warnings_count(const void* handle);
extern char*   document_editor_flatten_warning(const void* handle, int32_t index, int* error_code);

// Region extraction extras
extern void* pdf_document_extract_lines_in_rect(void* handle, int32_t page_index, float x, float y, float w, float h, int* error_code);
extern void* pdf_document_extract_tables_in_rect(void* handle, int32_t page_index, float x, float y, float w, float h, int* error_code);

// Logging
extern void pdf_oxide_set_log_level(int level);
extern int pdf_oxide_get_log_level();

// Crypto provider (issue #236)
extern char* pdf_oxide_crypto_active_provider();
extern int pdf_oxide_crypto_fips_available();
extern int pdf_oxide_crypto_use_fips();

// OCR (v0.3.27 — FFI bridge wrapping src/ocr::OcrEngine)
// Returns _ERR_UNSUPPORTED when the Rust crate was built without --features ocr.
extern void* pdf_ocr_engine_create(const char* det_model_path, const char* rec_model_path, const char* dict_path, int* error_code);
extern void pdf_ocr_engine_free(void* engine);
extern bool pdf_ocr_page_needs_ocr(void* document, int32_t page_index, int* error_code);
extern char* pdf_ocr_extract_text(void* document, int32_t page_index, const void* engine, int* error_code);

extern void free_string(char* ptr);
extern void free_bytes(void* ptr);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
	"unsafe"
)

// PdfDocument represents an open PDF document.
//
// A *PdfDocument is safe to share across goroutines. The internal lock
// serializes every method call (including reads) because the underlying
// native PDF parser is not reentrant — concurrent native-side reads could
// return spurious parse errors. The previous RWMutex design was
// downgraded to an exclusive lock after TestConcurrentReads flaked with
// "parse error (code 5)" under `go test -race`.
type PdfDocument struct {
	mu     sync.RWMutex
	handle unsafe.Pointer
	closed bool
}

// ffiError wraps a C.int FFI error code into a fully populated *Error. It is
// the canonical cgo-backend constructor for every error returned from the FFI
// boundary. Sentinels, Error struct, and sentinelForCode live in types.go.
func ffiError(errorCode C.int) error {
	return ffiErrorFromInt(int(errorCode))
}

// Open opens a PDF document from file path
func Open(path string) (*PdfDocument, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorCode C.int
	handle := C.pdf_document_open(cPath, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to open document: %w", ErrInternal)
	}

	return &PdfDocument{
		handle: handle,
		closed: false,
	}, nil
}

// Close closes the document and releases resources.
// It is safe to call Close multiple times.
func (doc *PdfDocument) Close() error {
	doc.mu.Lock()
	defer doc.mu.Unlock()
	if !doc.closed && doc.handle != nil {
		C.pdf_document_free(doc.handle)
		doc.closed = true
		doc.handle = nil
	}
	return nil
}

// acquireRead takes the exclusive lock and checks the document is open.
// (Name kept for the read-side API surface, but semantically an exclusive
// lock — see PdfDocument doc comment for why.) Caller must call
// doc.mu.Unlock() when done.
func (doc *PdfDocument) acquireRead() error {
	doc.mu.Lock()
	if doc.closed {
		doc.mu.Unlock()
		return ErrDocumentClosed
	}
	return nil
}

// PageCount returns the number of pages in the document
func (doc *PdfDocument) PageCount() (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	count := C.pdf_document_get_page_count(doc.handle, &errorCode)

	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}

	return int(count), nil
}

// Version returns the PDF version as (major, minor). It returns an error
// (wrapping ErrDocumentClosed) if the document has been closed.
func (doc *PdfDocument) Version() (major, minor uint8, err error) {
	if err := doc.acquireRead(); err != nil {
		return 0, 0, err
	}
	defer doc.mu.Unlock()
	var cmajor, cminor C.uint8_t
	C.pdf_document_get_version(doc.handle, &cmajor, &cminor)
	return uint8(cmajor), uint8(cminor), nil
}

// HasStructureTree reports whether the document has a Tagged PDF structure
// tree. It returns an error (wrapping ErrDocumentClosed) if the document has
// been closed.
func (doc *PdfDocument) HasStructureTree() (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	return bool(C.pdf_document_has_structure_tree(doc.handle)), nil
}

// ExtractText extracts plain text from a page
func (doc *PdfDocument) ExtractText(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()

	if pageIndex < 0 {
		return "", ErrInvalidPageIndex
	}

	var errorCode C.int
	cText := C.pdf_document_extract_text(doc.handle, C.int32_t(pageIndex), &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cText == nil {
		return "", ErrInternal
	}

	text := C.GoString(cText)
	C.free_string(cText)

	return text, nil
}

// ToMarkdown converts a page to Markdown format
func (doc *PdfDocument) ToMarkdown(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	cMarkdown := C.pdf_document_to_markdown(doc.handle, C.int32_t(pageIndex), &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	markdown := C.GoString(cMarkdown)
	C.free_string(cMarkdown)

	return markdown, nil
}

// ToHtml converts a page to HTML format
func (doc *PdfDocument) ToHtml(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	cHtml := C.pdf_document_to_html(doc.handle, C.int32_t(pageIndex), &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	html := C.GoString(cHtml)
	C.free_string(cHtml)

	return html, nil
}

// ToPlainText converts a page to plain text format
func (doc *PdfDocument) ToPlainText(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	cText := C.pdf_document_to_plain_text(doc.handle, C.int32_t(pageIndex), &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	text := C.GoString(cText)
	C.free_string(cText)

	return text, nil
}

// ToMarkdownAll converts all pages to Markdown format
func (doc *PdfDocument) ToMarkdownAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	cMarkdown := C.pdf_document_to_markdown_all(doc.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	markdown := C.GoString(cMarkdown)
	C.free_string(cMarkdown)

	return markdown, nil
}

// IsClosed returns whether the document is closed
func (doc *PdfDocument) IsClosed() bool {
	doc.mu.Lock()
	defer doc.mu.Unlock()
	return doc.closed
}

// DocumentEditor represents a PDF document editor for modifying metadata and
// properties. It is safe for concurrent use by multiple goroutines — all
// public methods acquire an internal RWMutex.
type DocumentEditor struct {
	mu     sync.RWMutex
	handle unsafe.Pointer
	closed bool
}

// acquireRead takes the editor's exclusive lock and verifies the editor
// is not closed. On success the caller MUST defer editor.mu.Unlock(). On
// failure the lock is released automatically and an error is returned.
func (editor *DocumentEditor) acquireRead() error {
	editor.mu.Lock()
	if editor.closed {
		editor.mu.Unlock()
		return ErrEditorClosed
	}
	return nil
}

// acquireWrite takes the editor's write lock and verifies the editor is not
// closed. On success the caller MUST defer editor.mu.Unlock().
func (editor *DocumentEditor) acquireWrite() error {
	editor.mu.Lock()
	if editor.closed {
		editor.mu.Unlock()
		return ErrEditorClosed
	}
	return nil
}

// OpenEditor opens a PDF document for editing metadata
func OpenEditor(path string) (*DocumentEditor, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorCode C.int
	handle := C.document_editor_open(cPath, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to open document editor: %w", ErrInternal)
	}

	return &DocumentEditor{
		handle: handle,
		closed: false,
	}, nil
}

// Close closes the editor and releases resources. Safe to call multiple times.
func (editor *DocumentEditor) Close() {
	editor.mu.Lock()
	defer editor.mu.Unlock()
	if !editor.closed && editor.handle != nil {
		C.document_editor_free(editor.handle)
		editor.closed = true
		editor.handle = nil
	}
}

// IsModified reports whether the document has been modified since opening.
// It returns an error (wrapping ErrEditorClosed) if the editor has been closed.
func (editor *DocumentEditor) IsModified() (bool, error) {
	if err := editor.acquireRead(); err != nil {
		return false, err
	}
	defer editor.mu.Unlock()
	return bool(C.document_editor_is_modified(editor.handle)), nil
}

// SourcePath returns the source file path of the document.
func (editor *DocumentEditor) SourcePath() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cPath := C.document_editor_get_source_path(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	path := C.GoString(cPath)
	C.free_string(cPath)
	return path, nil
}

// Version returns the PDF version as (major, minor). It returns an error
// (wrapping ErrEditorClosed) if the editor has been closed.
func (editor *DocumentEditor) Version() (major, minor uint8, err error) {
	if err := editor.acquireRead(); err != nil {
		return 0, 0, err
	}
	defer editor.mu.Unlock()
	var cmajor, cminor C.uint8_t
	C.document_editor_get_version(editor.handle, &cmajor, &cminor)
	return uint8(cmajor), uint8(cminor), nil
}

// PageCount returns the number of pages in the document
func (editor *DocumentEditor) PageCount() (int, error) {
	if err := editor.acquireRead(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	count := C.document_editor_get_page_count(editor.handle, &errorCode)

	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}

	return int(count), nil
}

// Title returns the document title
func (editor *DocumentEditor) Title() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cTitle := C.document_editor_get_title(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cTitle == nil {
		return "", nil
	}

	title := C.GoString(cTitle)
	C.free_string(cTitle)
	return title, nil
}

// SetTitle sets the document title
func (editor *DocumentEditor) SetTitle(title string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cTitle := C.CString(title)
	defer C.free(unsafe.Pointer(cTitle))

	var errorCode C.int
	C.document_editor_set_title(editor.handle, cTitle, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// Author returns the document author
func (editor *DocumentEditor) Author() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cAuthor := C.document_editor_get_author(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cAuthor == nil {
		return "", nil
	}

	author := C.GoString(cAuthor)
	C.free_string(cAuthor)
	return author, nil
}

// SetAuthor sets the document author
func (editor *DocumentEditor) SetAuthor(author string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cAuthor := C.CString(author)
	defer C.free(unsafe.Pointer(cAuthor))

	var errorCode C.int
	C.document_editor_set_author(editor.handle, cAuthor, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// Subject returns the document subject
func (editor *DocumentEditor) Subject() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cSubject := C.document_editor_get_subject(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cSubject == nil {
		return "", nil
	}

	subject := C.GoString(cSubject)
	C.free_string(cSubject)
	return subject, nil
}

// SetSubject sets the document subject
func (editor *DocumentEditor) SetSubject(subject string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cSubject := C.CString(subject)
	defer C.free(unsafe.Pointer(cSubject))

	var errorCode C.int
	C.document_editor_set_subject(editor.handle, cSubject, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// Producer returns the document producer
func (editor *DocumentEditor) Producer() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cProducer := C.document_editor_get_producer(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cProducer == nil {
		return "", nil
	}

	producer := C.GoString(cProducer)
	C.free_string(cProducer)
	return producer, nil
}

// SetProducer sets the document producer
func (editor *DocumentEditor) SetProducer(producer string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cProducer := C.CString(producer)
	defer C.free(unsafe.Pointer(cProducer))

	var errorCode C.int
	C.document_editor_set_producer(editor.handle, cProducer, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// CreationDate returns the document creation date
func (editor *DocumentEditor) CreationDate() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()

	var errorCode C.int
	cDate := C.document_editor_get_creation_date(editor.handle, &errorCode)

	if errorCode != 0 {
		return "", ffiError(errorCode)
	}

	if cDate == nil {
		return "", nil
	}

	date := C.GoString(cDate)
	C.free_string(cDate)
	return date, nil
}

// SetCreationDate sets the document creation date
func (editor *DocumentEditor) SetCreationDate(dateStr string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cDate := C.CString(dateStr)
	defer C.free(unsafe.Pointer(cDate))

	var errorCode C.int
	C.document_editor_set_creation_date(editor.handle, cDate, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// Save saves the edited document to a file
func (editor *DocumentEditor) Save(path string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorCode C.int
	C.document_editor_save(editor.handle, cPath, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// PdfCreator represents a PDF document being created or built
type PdfCreator struct {
	handle unsafe.Pointer
	closed bool
}

// FromMarkdown creates a new PDF from markdown content
//
// The returned PdfCreator must be closed with Close() when done.
//
// Example:
//
//	pdf, err := FromMarkdown("# Hello World\n\nThis is a PDF from markdown.")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer pdf.Close()
//
//	err = pdf.Save("output.pdf")
func FromMarkdown(markdown string) (*PdfCreator, error) {
	cMarkdown := C.CString(markdown)
	defer C.free(unsafe.Pointer(cMarkdown))

	var errorCode C.int
	handle := C.pdf_from_markdown(cMarkdown, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to create PDF from markdown: %w", ErrInternal)
	}

	return &PdfCreator{handle: handle}, nil
}

// FromHtml creates a new PDF from HTML content
//
// The returned PdfCreator must be closed with Close() when done.
//
// Example:
//
//	pdf, err := FromHtml("<h1>Hello World</h1><p>This is a PDF from HTML.</p>")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer pdf.Close()
//
//	err = pdf.Save("output.pdf")
func FromHtml(html string) (*PdfCreator, error) {
	cHtml := C.CString(html)
	defer C.free(unsafe.Pointer(cHtml))

	var errorCode C.int
	handle := C.pdf_from_html(cHtml, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to create PDF from HTML: %w", ErrInternal)
	}

	return &PdfCreator{handle: handle}, nil
}

// FromText creates a new PDF from plain text content
//
// The returned PdfCreator must be closed with Close() when done.
//
// Example:
//
//	pdf, err := FromText("Hello World\n\nThis is a PDF from plain text.")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer pdf.Close()
//
//	err = pdf.Save("output.pdf")
func FromText(text string) (*PdfCreator, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var errorCode C.int
	handle := C.pdf_from_text(cText, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to create PDF from text: %w", ErrInternal)
	}

	return &PdfCreator{handle: handle}, nil
}

// Save writes the PDF to a file
//
// Returns an error if the file cannot be written or the PDF is invalid.
func (pdf *PdfCreator) Save(path string) error {
	if pdf.closed {
		return ErrCreatorClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorCode C.int
	C.pdf_save(pdf.handle, cPath, &errorCode)

	if errorCode != 0 {
		return ffiError(errorCode)
	}

	return nil
}

// SaveToBytes returns the PDF as a byte slice
//
// The caller is responsible for managing the returned byte slice.
func (pdf *PdfCreator) SaveToBytes() ([]byte, error) {
	if pdf.closed {
		return nil, ErrCreatorClosed
	}

	var dataLen C.int
	var errorCode C.int

	ptr := C.pdf_save_to_bytes(pdf.handle, &dataLen, &errorCode)

	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	if ptr == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to save PDF to bytes: %w", ErrInternal)
	}

	// Convert C bytes to Go slice
	bytes := C.GoBytes(ptr, dataLen)
	// Make a copy since we'll free the original
	result := make([]byte, len(bytes))
	copy(result, bytes)

	// Free the original buffer
	C.free_bytes(ptr)

	return result, nil
}

// PageCount returns the number of pages in the PDF
func (pdf *PdfCreator) PageCount() (int, error) {
	if pdf.closed {
		return 0, ErrCreatorClosed
	}

	var errorCode C.int
	count := C.pdf_get_page_count(pdf.handle, &errorCode)

	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}

	return int(count), nil
}

// Close releases the resources associated with the PDF
//
// After calling Close(), the PdfCreator cannot be used.
func (pdf *PdfCreator) Close() {
	if pdf.closed {
		return
	}

	if pdf.handle != nil {
		C.pdf_free(pdf.handle)
		pdf.handle = nil
	}

	pdf.closed = true
}

// SearchResult, Font, Annotation, Element structs live in types.go so
// both the cgo and purego backends share them.

// SearchPage searches for text on a specific page and returns all matches.
// All marshaling logic lives on the Rust side in `pdf_oxide_search_results_to_json`;
// the Go layer makes exactly one FFI call per search.
func (doc *PdfDocument) SearchPage(pageIndex int, searchTerm string, caseSensitive bool) ([]SearchResult, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	cSearchTerm := C.CString(searchTerm)
	defer C.free(unsafe.Pointer(cSearchTerm))

	var errorCode C.int
	handle := C.pdf_document_search_page(doc.handle, C.int32_t(pageIndex), cSearchTerm, C.bool(caseSensitive), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	defer C.pdf_oxide_search_result_free(handle)

	return decodeSearchResults(handle)
}

// SearchAll searches for text across the entire document and returns all matches.
func (doc *PdfDocument) SearchAll(searchTerm string, caseSensitive bool) ([]SearchResult, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	cSearchTerm := C.CString(searchTerm)
	defer C.free(unsafe.Pointer(cSearchTerm))

	var errorCode C.int
	handle := C.pdf_document_search_all(doc.handle, cSearchTerm, C.bool(caseSensitive), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	defer C.pdf_oxide_search_result_free(handle)

	return decodeSearchResults(handle)
}

func decodeSearchResults(handle unsafe.Pointer) ([]SearchResult, error) {
	var errorCode C.int
	cJSON := C.pdf_oxide_search_results_to_json(handle, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if cJSON == nil {
		return []SearchResult{}, nil
	}
	defer C.free_string(cJSON)

	var results []SearchResult
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &results); err != nil {
		return nil, fmt.Errorf("pdf_oxide: failed to decode search results: %w", err)
	}
	return results, nil
}

// Fonts returns all fonts used or embedded in the given page. Marshaling is
// done entirely on the Rust side via `pdf_oxide_fonts_to_json`; the Go layer
// makes exactly one FFI call per page.
func (doc *PdfDocument) Fonts(pageIndex int) ([]Font, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	handle := C.pdf_document_get_embedded_fonts(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to get fonts: %w", ErrInternal)
	}
	defer C.pdf_oxide_font_list_free(handle)

	var jsonErr C.int
	cJSON := C.pdf_oxide_fonts_to_json(handle, &jsonErr)
	if jsonErr != 0 {
		return nil, ffiError(jsonErr)
	}
	if cJSON == nil {
		return []Font{}, nil
	}
	defer C.free_string(cJSON)

	var fonts []Font
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &fonts); err != nil {
		return nil, fmt.Errorf("pdf_oxide: failed to decode fonts: %w", err)
	}
	return fonts, nil
}

// Image represents an image embedded in a PDF page.
type Image struct {
	Width            int
	Height           int
	Format           string
	Colorspace       string
	BitsPerComponent int
	Data             []byte
}

// Images returns all images embedded in the given page.
func (doc *PdfDocument) Images(pageIndex int) ([]Image, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	handle := C.pdf_document_get_embedded_images(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to get images: %w", ErrInternal)
	}
	defer C.pdf_oxide_image_list_free(handle)

	count := int(C.pdf_oxide_image_count(handle))
	images := make([]Image, 0, count)
	for i := 0; i < count; i++ {
		img, err := readImageAt(handle, C.int32_t(i))
		if err != nil {
			return nil, err
		}
		images = append(images, img)
	}
	return images, nil
}

func readImageAt(handle unsafe.Pointer, index C.int32_t) (Image, error) {
	var wErr C.int
	width := int(C.pdf_oxide_image_get_width(handle, index, &wErr))
	if wErr != 0 {
		return Image{}, ffiError(wErr)
	}

	var hErr C.int
	height := int(C.pdf_oxide_image_get_height(handle, index, &hErr))
	if hErr != 0 {
		return Image{}, ffiError(hErr)
	}

	var fErr C.int
	cFormat := C.pdf_oxide_image_get_format(handle, index, &fErr)
	if fErr != 0 {
		return Image{}, ffiError(fErr)
	}
	defer C.free_string(cFormat)

	var csErr C.int
	cColorspace := C.pdf_oxide_image_get_colorspace(handle, index, &csErr)
	if csErr != 0 {
		return Image{}, ffiError(csErr)
	}
	defer C.free_string(cColorspace)

	var bpcErr C.int
	bits := int(C.pdf_oxide_image_get_bits_per_component(handle, index, &bpcErr))
	if bpcErr != 0 {
		return Image{}, ffiError(bpcErr)
	}

	var dataLen C.int
	var dataErr C.int
	dataPtr := C.pdf_oxide_image_get_data(handle, index, &dataLen, &dataErr)
	if dataErr != 0 {
		return Image{}, ffiError(dataErr)
	}
	data := C.GoBytes(dataPtr, dataLen)
	imageCopy := make([]byte, len(data))
	copy(imageCopy, data)
	C.free_bytes(dataPtr)

	return Image{
		Width:            width,
		Height:           height,
		Format:           C.GoString(cFormat),
		Colorspace:       C.GoString(cColorspace),
		BitsPerComponent: bits,
		Data:             imageCopy,
	}, nil
}

// Annotations returns all annotations on the given page with full details.
// Marshaling is done entirely on the Rust side via `pdf_oxide_annotations_to_json`.
func (doc *PdfDocument) Annotations(pageIndex int) ([]Annotation, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	handle := C.pdf_document_get_page_annotations(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to get annotations: %w", ErrInternal)
	}
	defer C.pdf_oxide_annotation_list_free(handle)

	var jsonErr C.int
	cJSON := C.pdf_oxide_annotations_to_json(handle, &jsonErr)
	if jsonErr != 0 {
		return nil, ffiError(jsonErr)
	}
	if cJSON == nil {
		return []Annotation{}, nil
	}
	defer C.free_string(cJSON)

	var anns []Annotation
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &anns); err != nil {
		return nil, fmt.Errorf("pdf_oxide: failed to decode annotations: %w", err)
	}
	return anns, nil
}

// Rect represents a rectangular region.
type Rect struct {
	X      float32
	Y      float32
	Width  float32
	Height float32
}

// PageInfo contains information about a PDF page.
type PageInfo struct {
	Width    float32
	Height   float32
	Rotation int
	MediaBox Rect
	CropBox  Rect
	ArtBox   Rect
	BleedBox Rect
	TrimBox  Rect
}

// PageInfo retrieves dimensions and boxes for a specific page.
func (doc *PdfDocument) PageInfo(pageIndex int) (*PageInfo, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int

	width := float32(C.pdf_page_get_width(doc.handle, C.int32_t(pageIndex), &errorCode))
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	height := float32(C.pdf_page_get_height(doc.handle, C.int32_t(pageIndex), &errorCode))
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	rotation := int(C.pdf_page_get_rotation(doc.handle, C.int32_t(pageIndex), &errorCode))
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	var mbX, mbY, mbW, mbH C.float
	C.pdf_page_get_media_box(doc.handle, C.int32_t(pageIndex), &mbX, &mbY, &mbW, &mbH, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	var cbX, cbY, cbW, cbH C.float
	C.pdf_page_get_crop_box(doc.handle, C.int32_t(pageIndex), &cbX, &cbY, &cbW, &cbH, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	var abX, abY, abW, abH C.float
	C.pdf_page_get_art_box(doc.handle, C.int32_t(pageIndex), &abX, &abY, &abW, &abH, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	var blbX, blbY, blbW, blbH C.float
	C.pdf_page_get_bleed_box(doc.handle, C.int32_t(pageIndex), &blbX, &blbY, &blbW, &blbH, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	var tbX, tbY, tbW, tbH C.float
	C.pdf_page_get_trim_box(doc.handle, C.int32_t(pageIndex), &tbX, &tbY, &tbW, &tbH, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}

	return &PageInfo{
		Width:    width,
		Height:   height,
		Rotation: rotation,
		MediaBox: Rect{X: float32(mbX), Y: float32(mbY), Width: float32(mbW), Height: float32(mbH)},
		CropBox:  Rect{X: float32(cbX), Y: float32(cbY), Width: float32(cbW), Height: float32(cbH)},
		ArtBox:   Rect{X: float32(abX), Y: float32(abY), Width: float32(abW), Height: float32(abH)},
		BleedBox: Rect{X: float32(blbX), Y: float32(blbY), Width: float32(blbW), Height: float32(blbH)},
		TrimBox:  Rect{X: float32(tbX), Y: float32(tbY), Width: float32(tbW), Height: float32(tbH)},
	}, nil
}

// PageElements returns all text-span elements on the given page. Marshaling
// is done entirely on the Rust side via `pdf_oxide_elements_to_json`.
func (doc *PdfDocument) PageElements(pageIndex int) ([]Element, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	var errorCode C.int
	handle := C.pdf_page_get_elements(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to get elements: %w", ErrInternal)
	}
	defer C.pdf_oxide_elements_free(handle)

	var jsonErr C.int
	cJSON := C.pdf_oxide_elements_to_json(handle, &jsonErr)
	if jsonErr != 0 {
		return nil, ffiError(jsonErr)
	}
	if cJSON == nil {
		return []Element{}, nil
	}
	defer C.free_string(cJSON)

	var elements []Element
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &elements); err != nil {
		return nil, fmt.Errorf("pdf_oxide: failed to decode elements: %w", err)
	}
	return elements, nil
}

// Metadata holds document metadata fields that ApplyMetadata can set in one
// call. Empty string fields are treated as "do not change". Use the individual
// SetTitle/SetAuthor/... setters if you need to distinguish empty-but-set from
// not-set semantics.
type Metadata struct {
	Title        string
	Author       string
	Subject      string
	Producer     string
	CreationDate string
}

// ApplyMetadata writes every non-empty field of m to the document. If any
// underlying setter returns an error, ApplyMetadata stops and returns it —
// previously applied fields are not rolled back.
func (editor *DocumentEditor) ApplyMetadata(m Metadata) error {
	if m.Title != "" {
		if err := editor.SetTitle(m.Title); err != nil {
			return err
		}
	}
	if m.Author != "" {
		if err := editor.SetAuthor(m.Author); err != nil {
			return err
		}
	}
	if m.Subject != "" {
		if err := editor.SetSubject(m.Subject); err != nil {
			return err
		}
	}
	if m.Producer != "" {
		if err := editor.SetProducer(m.Producer); err != nil {
			return err
		}
	}
	if m.CreationDate != "" {
		if err := editor.SetCreationDate(m.CreationDate); err != nil {
			return err
		}
	}
	return nil
}

// ================================================================
// New v0.3.24 methods
// ================================================================

// OpenFromBytes opens a PDF document from a byte slice
func OpenFromBytes(data []byte) (*PdfDocument, error) {
	if len(data) == 0 {
		return nil, ErrEmptyContent
	}
	var errorCode C.int
	handle := C.pdf_document_open_from_bytes((*C.uint8_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &PdfDocument{handle: handle, closed: false}, nil
}

// OpenWithPassword opens a PDF document with a password
func OpenWithPassword(path, password string) (*PdfDocument, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cPwd := C.CString(password)
	defer C.free(unsafe.Pointer(cPwd))
	var errorCode C.int
	handle := C.pdf_document_open_with_password(cPath, cPwd, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &PdfDocument{handle: handle, closed: false}, nil
}

// IsEncrypted checks if the document is encrypted
func (doc *PdfDocument) IsEncrypted() bool {
	if err := doc.acquireRead(); err != nil {
		return false
	}
	defer doc.mu.Unlock()
	return bool(C.pdf_document_is_encrypted(doc.handle))
}

// Authenticate authenticates with a password
func (doc *PdfDocument) Authenticate(password string) (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	cPwd := C.CString(password)
	defer C.free(unsafe.Pointer(cPwd))
	var errorCode C.int
	ok := C.pdf_document_authenticate(doc.handle, cPwd, &errorCode)
	if errorCode != 0 {
		return false, ffiError(errorCode)
	}
	return bool(ok), nil
}

// ExtractAllText extracts text from all pages
func (doc *PdfDocument) ExtractAllText() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_extract_all_text(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// ToHtmlAll converts all pages to HTML
func (doc *PdfDocument) ToHtmlAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_to_html_all(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// ToPlainTextAll converts all pages to plain text
func (doc *PdfDocument) ToPlainTextAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_to_plain_text_all(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// Word represents a word with position info
type Word struct {
	Text                string
	X, Y, Width, Height float32
	FontName            string
	FontSize            float32
	IsBold              bool
}

// ExtractWords extracts words with bounding boxes from a page
func (doc *PdfDocument) ExtractWords(pageIndex int) ([]Word, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_words(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_word_list_free(handle)
	count := int(C.pdf_oxide_word_count(handle))
	words := make([]Word, 0, count)
	for i := 0; i < count; i++ {
		var x, y, w, h C.float
		C.pdf_oxide_word_get_bbox(handle, C.int32_t(i), &x, &y, &w, &h, &errorCode)
		cText := C.pdf_oxide_word_get_text(handle, C.int32_t(i), &errorCode)
		text := C.GoString(cText)
		C.free_string(cText)
		cFont := C.pdf_oxide_word_get_font_name(handle, C.int32_t(i), &errorCode)
		font := C.GoString(cFont)
		C.free_string(cFont)
		words = append(words, Word{
			Text: text, X: float32(x), Y: float32(y), Width: float32(w), Height: float32(h),
			FontName: font,
			FontSize: float32(C.pdf_oxide_word_get_font_size(handle, C.int32_t(i), &errorCode)),
			IsBold:   bool(C.pdf_oxide_word_is_bold(handle, C.int32_t(i), &errorCode)),
		})
	}
	return words, nil
}

// TextLine represents a line of text
type TextLine struct {
	Text                string
	X, Y, Width, Height float32
	WordCount           int
}

// ExtractTextLines extracts text lines from a page
func (doc *PdfDocument) ExtractTextLines(pageIndex int) ([]TextLine, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_text_lines(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_line_list_free(handle)
	count := int(C.pdf_oxide_line_count(handle))
	lines := make([]TextLine, 0, count)
	for i := 0; i < count; i++ {
		var x, y, w, h C.float
		C.pdf_oxide_line_get_bbox(handle, C.int32_t(i), &x, &y, &w, &h, &errorCode)
		cText := C.pdf_oxide_line_get_text(handle, C.int32_t(i), &errorCode)
		text := C.GoString(cText)
		C.free_string(cText)
		lines = append(lines, TextLine{
			Text: text, X: float32(x), Y: float32(y), Width: float32(w), Height: float32(h),
			WordCount: int(C.pdf_oxide_line_get_word_count(handle, C.int32_t(i), &errorCode)),
		})
	}
	return lines, nil
}

// Table represents an extracted table
type Table struct {
	RowCount  int
	ColCount  int
	HasHeader bool
	handle    unsafe.Pointer
	index     int
}

// ExtractTables extracts tables from a page
func (doc *PdfDocument) ExtractTables(pageIndex int) ([]Table, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_tables(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	count := int(C.pdf_oxide_table_count(handle))
	tables := make([]Table, 0, count)
	for i := 0; i < count; i++ {
		tables = append(tables, Table{
			RowCount:  int(C.pdf_oxide_table_get_row_count(handle, C.int32_t(i), &errorCode)),
			ColCount:  int(C.pdf_oxide_table_get_col_count(handle, C.int32_t(i), &errorCode)),
			HasHeader: bool(C.pdf_oxide_table_has_header(handle, C.int32_t(i), &errorCode)),
			handle:    handle,
			index:     i,
		})
	}
	return tables, nil
}

// CellText returns the text of a cell at (row, col)
func (t *Table) CellText(row, col int) string {
	var errorCode C.int
	cText := C.pdf_oxide_table_get_cell_text(t.handle, C.int32_t(t.index), C.int32_t(row), C.int32_t(col), &errorCode)
	if cText == nil {
		return ""
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text
}

// ExtractTextInRect extracts text from a rectangular region
func (doc *PdfDocument) ExtractTextInRect(pageIndex int, x, y, w, h float32) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_extract_text_in_rect(doc.handle, C.int32_t(pageIndex), C.float(x), C.float(y), C.float(w), C.float(h), &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// FormField represents a form field
type FormField struct {
	Name     string
	Type     string
	Value    string
	ReadOnly bool
	Required bool
}

// FormFields returns all form fields in the document.
func (doc *PdfDocument) FormFields() ([]FormField, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_get_form_fields(doc.handle, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_form_field_list_free(handle)
	count := int(C.pdf_oxide_form_field_count(handle))
	fields := make([]FormField, 0, count)
	for i := 0; i < count; i++ {
		cName := C.pdf_oxide_form_field_get_name(handle, C.int32_t(i), &errorCode)
		name := C.GoString(cName)
		C.free_string(cName)
		cType := C.pdf_oxide_form_field_get_type(handle, C.int32_t(i), &errorCode)
		ftype := C.GoString(cType)
		C.free_string(cType)
		cVal := C.pdf_oxide_form_field_get_value(handle, C.int32_t(i), &errorCode)
		val := ""
		if cVal != nil {
			val = C.GoString(cVal)
			C.free_string(cVal)
		}
		fields = append(fields, FormField{
			Name: name, Type: ftype, Value: val,
			ReadOnly: bool(C.pdf_oxide_form_field_is_readonly(handle, C.int32_t(i), &errorCode)),
			Required: bool(C.pdf_oxide_form_field_is_required(handle, C.int32_t(i), &errorCode)),
		})
	}
	return fields, nil
}

// HasXfa checks if the document has XFA forms
func (doc *PdfDocument) HasXfa() bool {
	if err := doc.acquireRead(); err != nil {
		return false
	}
	defer doc.mu.Unlock()
	return bool(C.pdf_document_has_xfa(doc.handle))
}

// OcrEngine wraps the Rust OcrEngine for text recognition.
// Requires the Rust crate to be built with --features ocr;
// when the feature is off, NewOcrEngine returns ErrUnsupported.
type OcrEngine struct {
	handle unsafe.Pointer
}

// NewOcrEngine creates an OCR engine from model file paths.
// detModelPath: DBNet++ detection model (.onnx)
// recModelPath: SVTR recognition model (.onnx)
// dictPath:     character dictionary (.txt)
func NewOcrEngine(detModelPath, recModelPath, dictPath string) (*OcrEngine, error) {
	cDet := C.CString(detModelPath)
	cRec := C.CString(recModelPath)
	cDict := C.CString(dictPath)
	defer C.free(unsafe.Pointer(cDet))
	defer C.free(unsafe.Pointer(cRec))
	defer C.free(unsafe.Pointer(cDict))
	var errorCode C.int
	handle := C.pdf_ocr_engine_create(cDet, cRec, cDict, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	return &OcrEngine{handle: handle}, nil
}

// Close frees the OCR engine resources.
func (e *OcrEngine) Close() {
	if e.handle != nil {
		C.pdf_ocr_engine_free(e.handle)
		e.handle = nil
	}
}

// NeedsOcr checks whether a page would benefit from OCR
// (e.g. scanned image with no text layer).
func (doc *PdfDocument) NeedsOcr(pageIndex int) (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	result := bool(C.pdf_ocr_page_needs_ocr(doc.handle, C.int32_t(pageIndex), &errorCode))
	if errorCode != 0 {
		return false, ffiError(errorCode)
	}
	return result, nil
}

// ExtractTextWithOcr runs OCR on a page and returns the recognized text.
// engine may be nil, in which case the Rust side returns an error.
func (doc *PdfDocument) ExtractTextWithOcr(pageIndex int, engine *OcrEngine) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var enginePtr unsafe.Pointer
	if engine != nil {
		enginePtr = engine.handle
	}
	var errorCode C.int
	cText := C.pdf_ocr_extract_text(doc.handle, C.int32_t(pageIndex), enginePtr, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	if cText == nil {
		return "", nil
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// RemoveHeaders removes repeated headers. Returns count removed.
func (doc *PdfDocument) RemoveHeaders(threshold float32) (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	n := C.pdf_document_remove_headers(doc.handle, C.float(threshold), &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// RemoveFooters removes repeated footers. Returns count removed.
func (doc *PdfDocument) RemoveFooters(threshold float32) (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	n := C.pdf_document_remove_footers(doc.handle, C.float(threshold), &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// RemoveArtifacts removes headers and footers. Returns count removed.
func (doc *PdfDocument) RemoveArtifacts(threshold float32) (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	n := C.pdf_document_remove_artifacts(doc.handle, C.float(threshold), &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// ExportFormData exports form data as FDF (format=0) or XFDF (format=1)
func (doc *PdfDocument) ExportFormData(format int) ([]byte, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	var outLen C.size_t
	data := C.pdf_document_export_form_data_to_bytes(doc.handle, C.int32_t(format), &outLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if data == nil {
		return []byte{}, nil
	}
	bytes := C.GoBytes(unsafe.Pointer(data), C.int(outLen))
	C.free_bytes(unsafe.Pointer(data))
	return bytes, nil
}

// --- DocumentEditor new methods ---

// DeletePage removes a page by index
func (editor *DocumentEditor) DeletePage(pageIndex int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_delete_page(editor.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// MovePage moves a page from one index to another
func (editor *DocumentEditor) MovePage(from, to int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_move_page(editor.handle, C.int32_t(from), C.int32_t(to), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// PageRotation returns the rotation of a page in degrees
func (editor *DocumentEditor) PageRotation(pageIndex int) (int, error) {
	if err := editor.acquireRead(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	r := C.document_editor_get_page_rotation(editor.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(r), nil
}

// SetPageRotation sets page rotation (0, 90, 180, 270)
func (editor *DocumentEditor) SetPageRotation(pageIndex, degrees int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_set_page_rotation(editor.handle, C.int32_t(pageIndex), C.int32_t(degrees), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// EraseRegion erases a rectangular region on a page
func (editor *DocumentEditor) EraseRegion(pageIndex int, x, y, w, h float32) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_erase_region(editor.handle, C.int32_t(pageIndex), C.float(x), C.float(y), C.float(w), C.float(h), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// FlattenAnnotations flattens annotations on a specific page
func (editor *DocumentEditor) FlattenAnnotations(pageIndex int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_flatten_annotations(editor.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// FlattenAllAnnotations flattens all annotations
func (editor *DocumentEditor) FlattenAllAnnotations() error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_flatten_all_annotations(editor.handle, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// CropMargins crops margins on all pages
func (editor *DocumentEditor) CropMargins(left, right, top, bottom float32) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_crop_margins(editor.handle, C.float(left), C.float(right), C.float(top), C.float(bottom), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// MergeFrom merges pages from another PDF. Returns pages added.
func (editor *DocumentEditor) MergeFrom(sourcePath string) (int, error) {
	if err := editor.acquireWrite(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	cPath := C.CString(sourcePath)
	defer C.free(unsafe.Pointer(cPath))
	var errorCode C.int
	n := C.document_editor_merge_from(editor.handle, cPath, &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// SaveEncrypted saves the document with AES-256 encryption
func (editor *DocumentEditor) SaveEncrypted(path, userPassword, ownerPassword string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cUser := C.CString(userPassword)
	defer C.free(unsafe.Pointer(cUser))
	cOwner := C.CString(ownerPassword)
	defer C.free(unsafe.Pointer(cOwner))
	var errorCode C.int
	C.document_editor_save_encrypted(editor.handle, cPath, cUser, cOwner, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// OpenEditorFromBytes opens a PDF for editing from an in-memory byte slice.
func OpenEditorFromBytes(data []byte) (*DocumentEditor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("pdf_oxide: data must not be empty: %w", ErrInvalidPath)
	}
	var errorCode C.int
	handle := C.document_editor_open_from_bytes((*C.uint8_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to open editor from bytes: %w", ErrInternal)
	}
	return &DocumentEditor{
		handle: handle,
	}, nil
}

// SaveToBytes saves the edited document to an in-memory byte slice.
func (editor *DocumentEditor) SaveToBytes() ([]byte, error) {
	if err := editor.acquireWrite(); err != nil {
		return nil, err
	}
	defer editor.mu.Unlock()
	var outLen C.size_t
	var errorCode C.int
	ptr := C.document_editor_save_to_bytes(editor.handle, &outLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if ptr == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to save editor to bytes: %w", ErrInternal)
	}
	result := C.GoBytes(unsafe.Pointer(ptr), C.int(outLen))
	C.free_bytes(unsafe.Pointer(ptr))
	return result, nil
}

// SaveToBytesWithOptions saves the edited document to bytes with compression / GC / linearize flags.
func (editor *DocumentEditor) SaveToBytesWithOptions(compress, garbageCollect, linearize bool) ([]byte, error) {
	if err := editor.acquireWrite(); err != nil {
		return nil, err
	}
	defer editor.mu.Unlock()
	var outLen C.size_t
	var errorCode C.int
	ptr := C.document_editor_save_to_bytes_with_options(
		editor.handle,
		C.bool(compress),
		C.bool(garbageCollect),
		C.bool(linearize),
		&outLen,
		&errorCode,
	)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if ptr == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to save editor to bytes with options: %w", ErrInternal)
	}
	result := C.GoBytes(unsafe.Pointer(ptr), C.int(outLen))
	C.free_bytes(unsafe.Pointer(ptr))
	return result, nil
}

// ExtractPagesToBytes returns a new PDF containing only the specified 0-based page indices.
func (editor *DocumentEditor) ExtractPagesToBytes(pageIndices []int) ([]byte, error) {
	if len(pageIndices) == 0 {
		return nil, fmt.Errorf("pdf_oxide: ExtractPagesToBytes: no pages specified")
	}
	if err := editor.acquireWrite(); err != nil {
		return nil, err
	}
	defer editor.mu.Unlock()
	pages := make([]C.int32_t, len(pageIndices))
	for i, p := range pageIndices {
		pages[i] = C.int32_t(p)
	}
	var outLen C.size_t
	var errorCode C.int
	ptr := C.document_editor_extract_pages_to_bytes(editor.handle, &pages[0], C.size_t(len(pages)), &outLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if ptr == nil {
		return nil, fmt.Errorf("pdf_oxide: ExtractPagesToBytes failed: %w", ErrInternal)
	}
	result := C.GoBytes(unsafe.Pointer(ptr), C.int(outLen))
	C.free_bytes(unsafe.Pointer(ptr))
	return result, nil
}

// ConvertToPdfA converts the document to PDF/A in-place.
// level: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u
func (editor *DocumentEditor) ConvertToPdfA(level int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_convert_to_pdf_a(editor.handle, C.int32_t(level), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// SaveEncryptedToBytes saves the document with AES-256 encryption and returns the bytes.
// ownerPassword defaults to userPassword if empty.
func (editor *DocumentEditor) SaveEncryptedToBytes(userPassword, ownerPassword string) ([]byte, error) {
	if ownerPassword == "" {
		ownerPassword = userPassword
	}
	if err := editor.acquireWrite(); err != nil {
		return nil, err
	}
	defer editor.mu.Unlock()
	cUser := C.CString(userPassword)
	defer C.free(unsafe.Pointer(cUser))
	cOwner := C.CString(ownerPassword)
	defer C.free(unsafe.Pointer(cOwner))
	var outLen C.size_t
	var errorCode C.int
	ptr := C.document_editor_save_encrypted_to_bytes(editor.handle, cUser, cOwner, &outLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if ptr == nil {
		return nil, fmt.Errorf("pdf_oxide: SaveEncryptedToBytes failed: %w", ErrInternal)
	}
	result := C.GoBytes(unsafe.Pointer(ptr), C.int(outLen))
	C.free_bytes(unsafe.Pointer(ptr))
	return result, nil
}

// Keywords returns the document keywords metadata.
func (editor *DocumentEditor) Keywords() (string, error) {
	if err := editor.acquireRead(); err != nil {
		return "", err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	cKw := C.document_editor_get_keywords(editor.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	if cKw == nil {
		return "", nil
	}
	s := C.GoString(cKw)
	C.free_string(cKw)
	return s, nil
}

// SetKeywords sets the document keywords metadata.
func (editor *DocumentEditor) SetKeywords(keywords string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	cKw := C.CString(keywords)
	defer C.free(unsafe.Pointer(cKw))
	var errorCode C.int
	C.document_editor_set_keywords(editor.handle, cKw, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// MergeFromBytes merges pages from an in-memory PDF byte slice. Returns pages added.
func (editor *DocumentEditor) MergeFromBytes(data []byte) (int, error) {
	if err := editor.acquireWrite(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	if len(data) == 0 {
		return 0, fmt.Errorf("pdf_oxide: data must not be empty: %w", ErrInvalidPath)
	}
	var errorCode C.int
	n := C.document_editor_merge_from_bytes(
		editor.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		&errorCode,
	)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// EmbedFile embeds a file attachment into the document.
func (editor *DocumentEditor) EmbedFile(name string, data []byte) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var dataPtr *C.uint8_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	var errorCode C.int
	C.document_editor_embed_file(editor.handle, cName, dataPtr, C.size_t(len(data)), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// ApplyPageRedactions burns in redaction annotations on a single page.
func (editor *DocumentEditor) ApplyPageRedactions(page int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_apply_page_redactions(editor.handle, C.size_t(page), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// ApplyAllRedactions burns in all pending redaction annotations across the document.
func (editor *DocumentEditor) ApplyAllRedactions() error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_apply_all_redactions(editor.handle, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// RotateAllPages rotates every page by degrees (additive, not absolute).
func (editor *DocumentEditor) RotateAllPages(degrees int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_rotate_all_pages(editor.handle, C.int32_t(degrees), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// RotatePageBy rotates a single page by degrees (additive, not absolute).
func (editor *DocumentEditor) RotatePageBy(page, degrees int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_rotate_page_by(editor.handle, C.size_t(page), C.int32_t(degrees), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// GetPageMediaBox returns the MediaBox of a page as [x, y, w, h].
func (editor *DocumentEditor) GetPageMediaBox(page int) (x, y, w, h float64, err error) {
	if err = editor.acquireWrite(); err != nil {
		return
	}
	defer editor.mu.Unlock()
	var cx, cy, cw, ch C.double
	var errorCode C.int
	C.document_editor_get_page_media_box(editor.handle, C.size_t(page), &cx, &cy, &cw, &ch, &errorCode)
	if errorCode != 0 {
		err = ffiError(errorCode)
		return
	}
	return float64(cx), float64(cy), float64(cw), float64(ch), nil
}

// SetPageMediaBox sets the MediaBox of a page.
func (editor *DocumentEditor) SetPageMediaBox(page int, x, y, w, h float64) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_set_page_media_box(editor.handle, C.size_t(page), C.double(x), C.double(y), C.double(w), C.double(h), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// GetPageCropBox returns the CropBox of a page as [x, y, w, h]. Returns zeros if no CropBox is set.
func (editor *DocumentEditor) GetPageCropBox(page int) (x, y, w, h float64, err error) {
	if err = editor.acquireWrite(); err != nil {
		return
	}
	defer editor.mu.Unlock()
	var cx, cy, cw, ch C.double
	var errorCode C.int
	C.document_editor_get_page_crop_box(editor.handle, C.size_t(page), &cx, &cy, &cw, &ch, &errorCode)
	if errorCode != 0 {
		err = ffiError(errorCode)
		return
	}
	return float64(cx), float64(cy), float64(cw), float64(ch), nil
}

// SetPageCropBox sets the CropBox of a page.
func (editor *DocumentEditor) SetPageCropBox(page int, x, y, w, h float64) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_set_page_crop_box(editor.handle, C.size_t(page), C.double(x), C.double(y), C.double(w), C.double(h), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// EraseRegions erases multiple rectangular regions on a page.
// Each element of rects is [x, y, w, h] in PDF user space.
func (editor *DocumentEditor) EraseRegions(page int, rects [][4]float64) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	if len(rects) == 0 {
		return nil
	}
	flat := make([]C.double, len(rects)*4)
	for i, r := range rects {
		flat[i*4+0] = C.double(r[0])
		flat[i*4+1] = C.double(r[1])
		flat[i*4+2] = C.double(r[2])
		flat[i*4+3] = C.double(r[3])
	}
	var errorCode C.int
	C.document_editor_erase_regions(editor.handle, C.size_t(page), &flat[0], C.size_t(len(rects)), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// ClearEraseRegions removes all pending erase-region entries for a page.
func (editor *DocumentEditor) ClearEraseRegions(page int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_clear_erase_regions(editor.handle, C.size_t(page), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// IsPageMarkedForFlatten returns true if the page is marked for annotation-flatten.
func (editor *DocumentEditor) IsPageMarkedForFlatten(page int) bool {
	if err := editor.acquireRead(); err != nil {
		return false
	}
	defer editor.mu.Unlock()
	return C.document_editor_is_page_marked_for_flatten(editor.handle, C.size_t(page)) == 1
}

// UnmarkPageForFlatten removes the flatten mark from a page.
func (editor *DocumentEditor) UnmarkPageForFlatten(page int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_unmark_page_for_flatten(editor.handle, C.size_t(page), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// IsPageMarkedForRedaction returns true if the page is marked for redaction.
func (editor *DocumentEditor) IsPageMarkedForRedaction(page int) bool {
	if err := editor.acquireRead(); err != nil {
		return false
	}
	defer editor.mu.Unlock()
	return C.document_editor_is_page_marked_for_redaction(editor.handle, C.size_t(page)) == 1
}

// UnmarkPageForRedaction removes the redaction mark from a page.
func (editor *DocumentEditor) UnmarkPageForRedaction(page int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_unmark_page_for_redaction(editor.handle, C.size_t(page), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// PdfAResult holds PDF/A compliance validation results.
type PdfAResult struct {
	Compliant bool
	Errors    []string
	Warnings  []string
}

// ValidatePdfA validates PDF/A compliance. Level: 0=A1b, 1=A1a, 2=A2b, 3=A2a, 4=A2u, 5=A3b
func (doc *PdfDocument) ValidatePdfA(level int) (*PdfAResult, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	results := C.pdf_validate_pdf_a_level(doc.handle, C.int32_t(level), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_pdf_a_results_free(results)

	result := &PdfAResult{
		Compliant: bool(C.pdf_pdf_a_is_compliant(results, &errorCode)),
	}

	errCount := int(C.pdf_pdf_a_error_count(results))
	result.Errors = make([]string, 0, errCount)
	for i := 0; i < errCount; i++ {
		cErr := C.pdf_pdf_a_get_error(results, C.int32_t(i), &errorCode)
		if cErr != nil {
			result.Errors = append(result.Errors, C.GoString(cErr))
			C.free_string(cErr)
		}
	}

	warnCount := int(C.pdf_pdf_a_warning_count(results))
	result.Warnings = make([]string, 0, warnCount)
	for i := 0; i < warnCount; i++ {
		// Warnings use the same accessor as errors for now (API returns warnings via error list after errors)
		cWarn := C.pdf_pdf_a_get_error(results, C.int32_t(errCount+i), &errorCode)
		if cWarn != nil {
			result.Warnings = append(result.Warnings, C.GoString(cWarn))
			C.free_string(cWarn)
		}
	}

	return result, nil
}

// ValidatePdfUa validates PDF/UA accessibility
func (doc *PdfDocument) ValidatePdfUa() (bool, []string, error) {
	if err := doc.acquireRead(); err != nil {
		return false, nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	results := C.pdf_validate_pdf_ua(doc.handle, C.int32_t(1), &errorCode)
	if errorCode != 0 {
		return false, nil, ffiError(errorCode)
	}
	defer C.pdf_pdf_ua_results_free(results)
	compliant := bool(C.pdf_pdf_ua_is_accessible(results, &errorCode))
	errCount := int(C.pdf_pdf_ua_error_count(results))
	errors := make([]string, 0, errCount)
	for i := 0; i < errCount; i++ {
		cErr := C.pdf_pdf_ua_get_error(results, C.int32_t(i), &errorCode)
		if cErr != nil {
			errors = append(errors, C.GoString(cErr))
			C.free_string(cErr)
		}
	}
	return compliant, errors, nil
}

// Char represents a single character with position
type Char struct {
	Char                rune
	X, Y, Width, Height float32
	FontName            string
	FontSize            float32
}

// ExtractChars extracts individual characters from a page
func (doc *PdfDocument) ExtractChars(pageIndex int) ([]Char, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_chars(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_char_list_free(handle)
	count := int(C.pdf_oxide_char_count(handle))
	chars := make([]Char, 0, count)
	for i := 0; i < count; i++ {
		var x, y, w, h C.float
		C.pdf_oxide_char_get_bbox(handle, C.int32_t(i), &x, &y, &w, &h, &errorCode)
		ch := C.pdf_oxide_char_get_char(handle, C.int32_t(i), &errorCode)
		cFont := C.pdf_oxide_char_get_font_name(handle, C.int32_t(i), &errorCode)
		font := C.GoString(cFont)
		C.free_string(cFont)
		chars = append(chars, Char{
			Char: rune(ch), X: float32(x), Y: float32(y), Width: float32(w), Height: float32(h),
			FontName: font, FontSize: float32(C.pdf_oxide_char_get_font_size(handle, C.int32_t(i), &errorCode)),
		})
	}
	return chars, nil
}

// Path represents a vector path/shape
type Path struct {
	X, Y, W, H     float32
	StrokeWidth    float32
	HasStroke      bool
	HasFill        bool
	OperationCount int
}

// ExtractPaths extracts vector paths from a page
func (doc *PdfDocument) ExtractPaths(pageIndex int) ([]Path, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_paths(doc.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_path_list_free(handle)
	count := int(C.pdf_oxide_path_count(handle))
	paths := make([]Path, 0, count)
	for i := 0; i < count; i++ {
		var x, y, w, h C.float
		C.pdf_oxide_path_get_bbox(handle, C.int32_t(i), &x, &y, &w, &h, &errorCode)
		paths = append(paths, Path{
			X: float32(x), Y: float32(y), W: float32(w), H: float32(h),
			StrokeWidth:    float32(C.pdf_oxide_path_get_stroke_width(handle, C.int32_t(i), &errorCode)),
			HasStroke:      bool(C.pdf_oxide_path_has_stroke(handle, C.int32_t(i), &errorCode)),
			HasFill:        bool(C.pdf_oxide_path_has_fill(handle, C.int32_t(i), &errorCode)),
			OperationCount: int(C.pdf_oxide_path_get_operation_count(handle, C.int32_t(i), &errorCode)),
		})
	}
	return paths, nil
}

// PageLabels returns the document's page-label map serialised as a JSON string.
func (doc *PdfDocument) PageLabels() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_get_page_labels(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// XmpMetadata returns the document's XMP metadata serialised as a JSON string.
func (doc *PdfDocument) XmpMetadata() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_get_xmp_metadata(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// Outline returns the document outline / bookmarks serialised as a JSON string.
func (doc *PdfDocument) Outline() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	cText := C.pdf_document_get_outline(doc.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	text := C.GoString(cText)
	C.free_string(cText)
	return text, nil
}

// FromImage creates a PDF from an image file
func FromImage(path string) (*PdfCreator, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var errorCode C.int
	handle := C.pdf_from_image(cPath, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &PdfCreator{handle: handle, closed: false}, nil
}

// Merge merges multiple PDF files. Returns the merged PDF bytes.
func Merge(paths []string) ([]byte, error) {
	if len(paths) == 0 {
		return nil, ErrEmptyContent
	}
	cPaths := make([]*C.char, len(paths))
	for i, p := range paths {
		cPaths[i] = C.CString(p)
		defer C.free(unsafe.Pointer(cPaths[i]))
	}
	var errorCode C.int
	var dataLen C.int32_t
	data := C.pdf_merge((**C.char)(unsafe.Pointer(&cPaths[0])), C.int32_t(len(paths)), &dataLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if data == nil {
		return []byte{}, nil
	}
	bytes := C.GoBytes(unsafe.Pointer(data), C.int(dataLen))
	C.free_bytes(unsafe.Pointer(data))
	return bytes, nil
}

// ValidatePdfX validates PDF/X compliance. Level: 0=X1a2001, 1=X32002, 2=X4
func (doc *PdfDocument) ValidatePdfX(level int) (bool, []string, error) {
	if err := doc.acquireRead(); err != nil {
		return false, nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	results := C.pdf_validate_pdf_x_level(doc.handle, C.int32_t(level), &errorCode)
	if errorCode != 0 {
		return false, nil, ffiError(errorCode)
	}
	defer C.pdf_pdf_x_results_free(results)
	compliant := bool(C.pdf_pdf_x_is_compliant(results, &errorCode))
	errCount := int(C.pdf_pdf_x_error_count(results))
	errors := make([]string, 0, errCount)
	for i := 0; i < errCount; i++ {
		cErr := C.pdf_pdf_x_get_error(results, C.int32_t(i), &errorCode)
		if cErr != nil {
			errors = append(errors, C.GoString(cErr))
			C.free_string(cErr)
		}
	}
	return compliant, errors, nil
}

// FromImageBytes creates a PDF from image bytes
func FromImageBytes(data []byte) (*PdfCreator, error) {
	if len(data) == 0 {
		return nil, ErrEmptyContent
	}
	var errorCode C.int
	handle := C.pdf_from_image_bytes((*C.uint8_t)(unsafe.Pointer(&data[0])), C.int32_t(len(data)), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &PdfCreator{handle: handle, closed: false}, nil
}

// ExtractWordsInRect extracts words from a rectangular region
func (doc *PdfDocument) ExtractWordsInRect(pageIndex int, x, y, w, h float32) ([]Word, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_words_in_rect(doc.handle, C.int32_t(pageIndex), C.float(x), C.float(y), C.float(w), C.float(h), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	defer C.pdf_oxide_word_list_free(handle)
	count := int(C.pdf_oxide_word_count(handle))
	words := make([]Word, 0, count)
	for i := 0; i < count; i++ {
		var bx, by, bw, bh C.float
		C.pdf_oxide_word_get_bbox(handle, C.int32_t(i), &bx, &by, &bw, &bh, &errorCode)
		cText := C.pdf_oxide_word_get_text(handle, C.int32_t(i), &errorCode)
		text := C.GoString(cText)
		C.free_string(cText)
		words = append(words, Word{Text: text, X: float32(bx), Y: float32(by), Width: float32(bw), Height: float32(bh)})
	}
	return words, nil
}

// ExtractImagesInRect extracts images from a rectangular region
func (doc *PdfDocument) ExtractImagesInRect(pageIndex int, x, y, w, h float32) (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_document_extract_images_in_rect(doc.handle, C.int32_t(pageIndex), C.float(x), C.float(y), C.float(w), C.float(h), &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	count := int(C.pdf_oxide_image_count(handle))
	C.pdf_oxide_image_list_free(handle)
	return count, nil
}

// SetFormFieldValue sets a form field value on the editor
func (editor *DocumentEditor) SetFormFieldValue(name, value string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cVal := C.CString(value)
	defer C.free(unsafe.Pointer(cVal))
	var errorCode C.int
	C.document_editor_set_form_field_value(editor.handle, cName, cVal, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// FlattenForms flattens all form fields into page content
func (editor *DocumentEditor) FlattenForms() error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_flatten_forms(editor.handle, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// FlattenFormsOnPage flattens form fields on a specific page
func (editor *DocumentEditor) FlattenFormsOnPage(pageIndex int) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var errorCode C.int
	C.document_editor_flatten_forms_on_page(editor.handle, C.int32_t(pageIndex), &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// FlattenWarnings returns warnings collected during the last form-flattening
// save. Each entry names a widget that had no /AP appearance stream; flattening
// it produces a blank rectangle.
func (editor *DocumentEditor) FlattenWarnings() []string {
	if err := editor.acquireRead(); err != nil {
		return nil
	}
	defer editor.mu.RUnlock()
	count := int(C.document_editor_flatten_warnings_count(editor.handle))
	if count <= 0 {
		return nil
	}
	result := make([]string, 0, count)
	for i := 0; i < count; i++ {
		var errorCode C.int
		ptr := C.document_editor_flatten_warning(editor.handle, C.int32_t(i), &errorCode)
		if ptr != nil {
			result = append(result, C.GoString(ptr))
			C.free_string(ptr)
		}
	}
	return result
}

// ================================================================
// Rendering
// ================================================================

// RenderedImage holds a rendered page image
type RenderedImage struct {
	handle unsafe.Pointer
	Width  int
	Height int
}

// RenderFormat selects the output image format.
type RenderFormat int32

const (
	// RenderFormatPng selects PNG output (supports transparency).
	RenderFormatPng RenderFormat = 0
	// RenderFormatJpeg selects JPEG output (honours JpegQuality).
	RenderFormatJpeg RenderFormat = 1
)

// RenderOptions mirrors Rust's RenderOptions
// (src/rendering/page_renderer.rs:41). Fields are individually
// zero-safe: a zero-value RenderOptions{} renders at 150 DPI PNG with
// an opaque white background, annotations on, JPEG quality 85 (same
// defaults as Rust).
type RenderOptions struct {
	// Dpi is the resolution in dots per inch. 0 => default 150.
	Dpi int
	// Format selects PNG or JPEG. Zero = PNG.
	Format RenderFormat
	// Background is RGBA in [0.0, 1.0]. Zero value is the alpha-0
	// transparent (unusual default); if all four channels are zero
	// we substitute the Rust default of opaque white so Go callers
	// get intuitive behaviour without having to fill the struct.
	Background [4]float32
	// TransparentBackground drops the background fill entirely.
	// When true, overrides Background and matches Rust's
	// `Option::None` on the background field.
	TransparentBackground bool
	// RenderAnnotations toggles the annotation layer. Zero value
	// (false) maps to Rust's default of true.
	RenderAnnotations bool
	// JpegQuality is 1..=100. 0 => default 85. Only applies when
	// Format is RenderFormatJpeg.
	JpegQuality int
	// renderAnnotationsSet is an internal sentinel letting callers
	// set RenderAnnotations=false deliberately without having to
	// pass the whole struct through a constructor.
	renderAnnotationsSet bool
}

// WithAnnotationsOff returns a copy of opts with annotation rendering
// disabled. Use this instead of setting RenderAnnotations=false
// directly, otherwise a zero-value struct cannot be distinguished
// from an unspecified field.
func (opts RenderOptions) WithAnnotationsOff() RenderOptions {
	opts.RenderAnnotations = false
	opts.renderAnnotationsSet = true
	return opts
}

// WithAnnotationsOn mirrors WithAnnotationsOff.
func (opts RenderOptions) WithAnnotationsOn() RenderOptions {
	opts.RenderAnnotations = true
	opts.renderAnnotationsSet = true
	return opts
}

// RenderPageWithOptions renders a page with the full RenderOptions
// surface — DPI, format, background colour or transparency,
// annotation toggle, and JPEG quality. Mirrors Python's expanded
// `render_page` keywords and the C#
// `RenderPage(int, RenderOptions)` overload.
//
// Returns an error (wrapping ErrInvalidPath or equivalent) on bad
// options, matching other Go FFI error paths.
func (doc *PdfDocument) RenderPageWithOptions(pageIndex int, opts RenderOptions) (*RenderedImage, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()

	dpi := opts.Dpi
	if dpi == 0 {
		dpi = 150
	}
	jpegQuality := opts.JpegQuality
	if jpegQuality == 0 {
		jpegQuality = 85
	}
	// Apply "zero-value means Rust default (opaque white)" trick.
	bg := opts.Background
	if bg == [4]float32{0, 0, 0, 0} && !opts.TransparentBackground {
		bg = [4]float32{1, 1, 1, 1}
	}
	renderAnnots := int32(1)
	if opts.renderAnnotationsSet {
		if !opts.RenderAnnotations {
			renderAnnots = 0
		}
	}
	transparent := int32(0)
	if opts.TransparentBackground {
		transparent = 1
	}

	var errorCode C.int
	handle := C.pdf_render_page_with_options(
		doc.handle,
		C.int32_t(pageIndex),
		C.int32_t(dpi),
		C.int32_t(opts.Format),
		C.float(bg[0]), C.float(bg[1]), C.float(bg[2]), C.float(bg[3]),
		C.int32_t(transparent),
		C.int32_t(renderAnnots),
		C.int32_t(jpegQuality),
		&errorCode,
	)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	w := int(C.pdf_get_rendered_image_width(handle, &errorCode))
	h := int(C.pdf_get_rendered_image_height(handle, &errorCode))
	return &RenderedImage{handle: handle, Width: w, Height: h}, nil
}

// RenderPage renders a page to an image. format: 0=PNG, 1=JPEG
func (doc *PdfDocument) RenderPage(pageIndex int, format int) (*RenderedImage, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_render_page(doc.handle, C.int32_t(pageIndex), C.int32_t(format), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	w := int(C.pdf_get_rendered_image_width(handle, &errorCode))
	h := int(C.pdf_get_rendered_image_height(handle, &errorCode))
	return &RenderedImage{handle: handle, Width: w, Height: h}, nil
}

// RenderPageZoom renders a page with zoom factor
func (doc *PdfDocument) RenderPageZoom(pageIndex int, zoom float32, format int) (*RenderedImage, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_render_page_zoom(doc.handle, C.int32_t(pageIndex), C.float(zoom), C.int32_t(format), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	w := int(C.pdf_get_rendered_image_width(handle, &errorCode))
	h := int(C.pdf_get_rendered_image_height(handle, &errorCode))
	return &RenderedImage{handle: handle, Width: w, Height: h}, nil
}

// RenderPageFit renders a page to fit inside a fitWidth × fitHeight pixel
// box, preserving aspect ratio. Picks the largest DPI such that both
// rendered dimensions are ≤ the target box, so the output may be smaller
// than the requested box on one axis. Issue #448.
func (doc *PdfDocument) RenderPageFit(pageIndex, fitWidth, fitHeight, format int) (*RenderedImage, error) {
	if fitWidth <= 0 || fitHeight <= 0 {
		return nil, fmt.Errorf("RenderPageFit: fitWidth and fitHeight must be > 0, got %dx%d", fitWidth, fitHeight)
	}
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_render_page_fit(doc.handle, C.int32_t(pageIndex), C.int32_t(fitWidth), C.int32_t(fitHeight), C.int32_t(format), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	w := int(C.pdf_get_rendered_image_width(handle, &errorCode))
	h := int(C.pdf_get_rendered_image_height(handle, &errorCode))
	return &RenderedImage{handle: handle, Width: w, Height: h}, nil
}

// RenderThumbnail renders a page thumbnail
func (doc *PdfDocument) RenderThumbnail(pageIndex int, size int, format int) (*RenderedImage, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	handle := C.pdf_render_page_thumbnail(doc.handle, C.int32_t(pageIndex), C.int32_t(size), C.int32_t(format), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if handle == nil {
		return nil, ErrInternal
	}
	w := int(C.pdf_get_rendered_image_width(handle, &errorCode))
	h := int(C.pdf_get_rendered_image_height(handle, &errorCode))
	return &RenderedImage{handle: handle, Width: w, Height: h}, nil
}

// RgbaPixmap holds raw premultiplied RGBA8888 pixel data returned by RenderPageRaw.
// Layout: row-major, top-left origin, 4 bytes (R,G,B,A) per pixel.
// len(Data) == Width * Height * 4.
type RgbaPixmap struct {
	Data          []byte
	Width, Height int
}

// RenderPageRaw renders a page as raw premultiplied RGBA8888 pixels without
// PNG/JPEG encoding overhead. Useful for direct handoff to image-processing
// pipelines. Premultiplied alpha is the native PDF compositing format
// (spec §11 transparency model).
func (doc *PdfDocument) RenderPageRaw(pageIndex, dpi int) (RgbaPixmap, error) {
	if dpi <= 0 {
		return RgbaPixmap{}, fmt.Errorf("RenderPageRaw: dpi must be > 0, got %d", dpi)
	}
	if err := doc.acquireRead(); err != nil {
		return RgbaPixmap{}, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	var outW, outH C.int32_t
	handle := C.pdf_render_page_raw(doc.handle, C.int32_t(pageIndex), C.int32_t(dpi), &outW, &outH, &errorCode)
	if errorCode != 0 {
		return RgbaPixmap{}, ffiError(errorCode)
	}
	if handle == nil {
		return RgbaPixmap{}, ErrInternal
	}
	defer C.pdf_rendered_image_free(handle)
	var dataLen C.int32_t
	dataPtr := C.pdf_get_rendered_image_data(handle, &dataLen, &errorCode)
	if dataPtr == nil {
		return RgbaPixmap{}, fmt.Errorf("RenderPageRaw: no pixel data returned")
	}
	data := C.GoBytes(unsafe.Pointer(dataPtr), C.int(dataLen))
	C.free_bytes(unsafe.Pointer(dataPtr))
	return RgbaPixmap{Data: data, Width: int(outW), Height: int(outH)}, nil
}

// Data returns the raw image bytes
func (img *RenderedImage) Data() []byte {
	var errorCode C.int
	var dataLen C.int32_t
	data := C.pdf_get_rendered_image_data(img.handle, &dataLen, &errorCode)
	if data == nil {
		return nil
	}
	bytes := C.GoBytes(unsafe.Pointer(data), C.int(dataLen))
	C.free_bytes(unsafe.Pointer(data))
	return bytes
}

// SaveToFile saves the rendered image to a file
func (img *RenderedImage) SaveToFile(path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var errorCode C.int
	C.pdf_save_rendered_image(img.handle, cPath, &errorCode)
	if errorCode != 0 {
		return ffiError(errorCode)
	}
	return nil
}

// Close releases the rendered image resources
func (img *RenderedImage) Close() {
	if img.handle != nil {
		C.pdf_rendered_image_free(img.handle)
		img.handle = nil
	}
}

// ================================================================
// Barcodes
// ================================================================

// BarcodeImage holds a generated barcode
type BarcodeImage struct {
	handle unsafe.Pointer
}

// GenerateQRCode generates a QR code image. errorCorrection: 0=Low, 1=Medium, 2=Quartile, 3=High
func GenerateQRCode(data string, errorCorrection int, sizePx int) (*BarcodeImage, error) {
	cData := C.CString(data)
	defer C.free(unsafe.Pointer(cData))
	var errorCode C.int
	handle := C.pdf_generate_qr_code(cData, C.int(errorCorrection), C.int32_t(sizePx), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &BarcodeImage{handle: handle}, nil
}

// GenerateBarcode generates a 1D barcode. format: 0=Code128, 1=Code39, 2=EAN13, etc.
func GenerateBarcode(data string, format int, sizePx int) (*BarcodeImage, error) {
	cData := C.CString(data)
	defer C.free(unsafe.Pointer(cData))
	var errorCode C.int
	handle := C.pdf_generate_barcode(cData, C.int(format), C.int32_t(sizePx), &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &BarcodeImage{handle: handle}, nil
}

// PNGData returns the barcode rendered as PNG bytes, or an error if the
// native layer failed to produce the image.
func (bc *BarcodeImage) PNGData() ([]byte, error) {
	if bc.handle == nil {
		return nil, ErrInternal
	}
	var outLen C.int32_t
	var errorCode C.int
	ptr := C.pdf_barcode_get_image_png(bc.handle, 0, &outLen, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if ptr == nil || outLen <= 0 {
		return nil, ErrInternal
	}
	// Copy the C buffer into a Go-owned slice and free the C allocation.
	data := C.GoBytes(unsafe.Pointer(ptr), C.int(outLen))
	C.free_bytes(unsafe.Pointer(ptr))
	return data, nil
}

// SourceData returns the original data encoded in the barcode
func (bc *BarcodeImage) SourceData() string {
	var errorCode C.int
	cData := C.pdf_barcode_get_data(bc.handle, &errorCode)
	if cData == nil {
		return ""
	}
	data := C.GoString(cData)
	C.free_string(cData)
	return data
}

// SVGData returns the barcode rendered as a vector SVG string.
func (bc *BarcodeImage) SVGData() (string, error) {
	if bc.handle == nil {
		return "", ErrInternal
	}
	var errorCode C.int
	cSvg := C.pdf_barcode_get_svg(bc.handle, 0, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	if cSvg == nil {
		return "", ErrInternal
	}
	svg := C.GoString(cSvg)
	C.free_string(cSvg)
	return svg, nil
}

// Close releases barcode resources
func (bc *BarcodeImage) Close() {
	if bc.handle != nil {
		C.pdf_barcode_free(bc.handle)
		bc.handle = nil
	}
}

// ================================================================
// Signatures
// ================================================================

// Certificate holds a loaded signing certificate
type Certificate struct {
	handle unsafe.Pointer
}

// LoadCertificate loads a PKCS#12 certificate from bytes
func LoadCertificate(data []byte, password string) (*Certificate, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("pdf_oxide: certificate data is empty: %w", ErrEmptyContent)
	}
	cPwd := C.CString(password)
	defer C.free(unsafe.Pointer(cPwd))
	var errorCode C.int
	handle := C.pdf_certificate_load_from_bytes((*C.uint8_t)(unsafe.Pointer(&data[0])), C.int32_t(len(data)), cPwd, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &Certificate{handle: handle}, nil
}

// LoadCertificateFromPem loads signing credentials from PEM-encoded certificate and private key strings.
// certPem must begin "-----BEGIN CERTIFICATE-----"; keyPem must begin
// "-----BEGIN PRIVATE KEY-----" (PKCS#8) or "-----BEGIN RSA PRIVATE KEY-----" (PKCS#1).
func LoadCertificateFromPem(certPem, keyPem string) (*Certificate, error) {
	cCert := C.CString(certPem)
	defer C.free(unsafe.Pointer(cCert))
	cKey := C.CString(keyPem)
	defer C.free(unsafe.Pointer(cKey))
	var errorCode C.int
	handle := C.pdf_certificate_load_from_pem(cCert, cKey, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	return &Certificate{handle: handle}, nil
}

// Close releases certificate resources
func (cert *Certificate) Close() {
	if cert.handle != nil {
		C.pdf_certificate_free(cert.handle)
		cert.handle = nil
	}
}

// SignPdfBytes applies a CMS/PKCS#7 detached signature to pdfData and returns
// the signed PDF as a new byte slice. The Certificate must have been loaded
// with a private key (e.g. via LoadCertificateFromPem or LoadCertificateFromBytes).
// reason and location are optional; pass empty strings to omit them.
func SignPdfBytes(pdfData []byte, cert *Certificate, reason, location string) ([]byte, error) {
	if cert == nil || cert.handle == nil {
		return nil, ErrInternal
	}
	if len(pdfData) == 0 {
		return nil, ErrEmptyContent
	}
	var reasonPtr, locationPtr *C.char
	if reason != "" {
		cs := C.CString(reason)
		defer C.free(unsafe.Pointer(cs))
		reasonPtr = cs
	}
	if location != "" {
		cs := C.CString(location)
		defer C.free(unsafe.Pointer(cs))
		locationPtr = cs
	}
	var outLen C.size_t
	var errorCode C.int
	out := C.pdf_sign_bytes(
		(*C.uint8_t)(unsafe.Pointer(&pdfData[0])),
		C.size_t(len(pdfData)),
		cert.handle,
		reasonPtr,
		locationPtr,
		&outLen,
		&errorCode,
	)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	if out == nil {
		return nil, ErrInternal
	}
	result := C.GoBytes(unsafe.Pointer(out), C.int(outLen))
	C.free_bytes(unsafe.Pointer(out))
	return result, nil
}

// certReadString is the shared body for Subject / Issuer / Serial — each FFI
// call returns a `*C.char` that must be copied to Go memory and freed.
func (cert *Certificate) certReadString(fn func(unsafe.Pointer, *C.int) *C.char) (string, error) {
	if cert.handle == nil {
		return "", ErrInternal
	}
	var errorCode C.int
	cStr := fn(cert.handle, &errorCode)
	if errorCode != 0 {
		return "", ffiError(errorCode)
	}
	if cStr == nil {
		return "", nil
	}
	defer C.free_string(cStr)
	return C.GoString(cStr), nil
}

// Subject returns the certificate's subject distinguished name (e.g.
// "CN=Example Corp, O=Example, C=US"). Returns an error if the certificate
// has been closed or the native call fails.
func (cert *Certificate) Subject() (string, error) {
	return cert.certReadString(func(h unsafe.Pointer, ec *C.int) *C.char {
		return C.pdf_certificate_get_subject(h, ec)
	})
}

// Issuer returns the issuer distinguished name — the DN of the CA that
// signed this certificate (self-signed certs have Issuer == Subject).
func (cert *Certificate) Issuer() (string, error) {
	return cert.certReadString(func(h unsafe.Pointer, ec *C.int) *C.char {
		return C.pdf_certificate_get_issuer(h, ec)
	})
}

// Serial returns the certificate's serial number as a hex-encoded string.
func (cert *Certificate) Serial() (string, error) {
	return cert.certReadString(func(h unsafe.Pointer, ec *C.int) *C.char {
		return C.pdf_certificate_get_serial(h, ec)
	})
}

// Validity returns the certificate's notBefore and notAfter times as Unix
// epoch seconds, wrapped in time.Time. Callers comparing against time.Now()
// get "is this cert currently time-valid" — IsValid() does that check.
func (cert *Certificate) Validity() (notBefore, notAfter time.Time, err error) {
	if cert.handle == nil {
		return time.Time{}, time.Time{}, ErrInternal
	}
	var nb, na C.int64_t
	var errorCode C.int
	C.pdf_certificate_get_validity(cert.handle, &nb, &na, &errorCode)
	if errorCode != 0 {
		return time.Time{}, time.Time{}, ffiError(errorCode)
	}
	return time.Unix(int64(nb), 0).UTC(), time.Unix(int64(na), 0).UTC(), nil
}

// IsValid reports whether the certificate is currently within its validity
// window (notBefore ≤ now ≤ notAfter). It does NOT verify the signature
// chain, trust-root, or revocation — this is a time-window check only.
func (cert *Certificate) IsValid() (bool, error) {
	if cert.handle == nil {
		return false, ErrInternal
	}
	var errorCode C.int
	rc := C.pdf_certificate_is_valid(cert.handle, &errorCode)
	if errorCode != 0 {
		return false, ffiError(errorCode)
	}
	return rc != 0, nil
}

// SignatureInfo holds extracted signature information
type SignatureInfo struct {
	handle     unsafe.Pointer
	SignerName string
	Reason     string
	Location   string
}

// Close releases signature info resources
func (sig *SignatureInfo) Close() {
	if sig.handle != nil {
		C.pdf_signature_free(sig.handle)
		sig.handle = nil
	}
}

// Signature is a live handle to an existing PDF digital signature
// returned by PdfDocument.Signatures. Close() must be called to
// release the underlying native handle. Cryptographic Verify()
// surfaces as ErrUnsupportedFeature until the Rust CMS
// signature-verification path lands.
type Signature struct {
	handle unsafe.Pointer
}

// SignatureCount returns the number of existing digital signatures in
// the document. Returns 0 when the PDF has no AcroForm or no signed
// signature fields (not an error).
func (doc *PdfDocument) SignatureCount() (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	n := C.pdf_document_get_signature_count(doc.handle, &errorCode)
	if errorCode != 0 {
		return 0, ffiError(errorCode)
	}
	return int(n), nil
}

// Signatures returns a snapshot of every signature currently on the
// document. Each Signature must be Close()d by the caller.
func (doc *PdfDocument) Signatures() ([]*Signature, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var errorCode C.int
	n := C.pdf_document_get_signature_count(doc.handle, &errorCode)
	if errorCode != 0 {
		return nil, ffiError(errorCode)
	}
	out := make([]*Signature, 0, n)
	for i := C.int32_t(0); i < n; i++ {
		var e C.int
		h := C.pdf_document_get_signature(doc.handle, i, &e)
		if e != 0 {
			for _, s := range out {
				s.Close()
			}
			return nil, ffiError(e)
		}
		if h == nil {
			for _, s := range out {
				s.Close()
			}
			return nil, fmt.Errorf("pdf_oxide: pdf_document_get_signature(%d) returned null", i)
		}
		out = append(out, &Signature{handle: h})
	}
	return out, nil
}

// SignerName returns the /Name entry on the signature dictionary, or
// an empty string if absent.
func (s *Signature) SignerName() (string, error) {
	return sigReadString(s.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_signature_get_signer_name(h, e)
	})
}

// Reason returns the /Reason entry on the signature dictionary, or an
// empty string if absent.
func (s *Signature) Reason() (string, error) {
	return sigReadString(s.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_signature_get_signing_reason(h, e)
	})
}

// Location returns the /Location entry on the signature dictionary, or
// an empty string if absent.
func (s *Signature) Location() (string, error) {
	return sigReadString(s.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_signature_get_signing_location(h, e)
	})
}

// SigningTime returns the signing time parsed from the /M entry as a
// Unix epoch. Returns 0 when the /M entry is absent or unparseable.
func (s *Signature) SigningTime() (int64, error) {
	if s.handle == nil {
		return 0, ErrDocumentClosed
	}
	var e C.int
	t := C.pdf_signature_get_signing_time(s.handle, &e)
	if e != 0 {
		return 0, ffiError(e)
	}
	return int64(t), nil
}

// Verify runs the RFC 5652 §5.4 signer-attributes RSA-PKCS#1 v1.5
// crypto check against the certificate embedded in the signature's
// CMS blob. Today it covers SHA-1 / SHA-256 / SHA-384 / SHA-512 —
// the padding used by essentially every PDF signature in the wild.
//
// A true return proves the signer held the private key matching the
// embedded certificate and that the signed-attribute bundle is
// authentic. It does NOT verify the messageDigest attribute against
// the document's byte-range content hash — call VerifyDetached for
// that end-to-end check.
//
// Returns ErrUnsupportedFeature for RSA-PSS, ECDSA, unknown digest
// OIDs, or CMS blobs missing signed_attrs.
func (s *Signature) Verify() (bool, error) {
	if s.handle == nil {
		return false, ErrDocumentClosed
	}
	var e C.int
	r := C.pdf_signature_verify(s.handle, &e)
	if e != 0 {
		return false, ffiError(e)
	}
	return r == 1, nil
}

// VerifyDetached runs both the signer-attributes RSA-PKCS#1 v1.5
// crypto check AND the RFC 5652 §11.2 messageDigest attribute check
// against the portion of pdfData that this signature covers (extracted
// via the signature's /ByteRange).
//
// pdfData must be the full PDF file. A true result proves the signer
// is authentic AND the document bytes under the ByteRange have not
// been altered since signing. A false result means either the signer
// check failed (wrong key / tampered attributes) or the content was
// modified.
//
// Returns ErrUnsupportedFeature for RSA-PSS, ECDSA, unknown digest
// OIDs, or CMS blobs missing signed_attrs / messageDigest.
func (s *Signature) VerifyDetached(pdfData []byte) (bool, error) {
	if s.handle == nil {
		return false, ErrDocumentClosed
	}
	var e C.int
	var dataPtr *C.uchar
	if len(pdfData) > 0 {
		dataPtr = (*C.uchar)(unsafe.Pointer(&pdfData[0]))
	}
	r := C.pdf_signature_verify_detached(
		s.handle,
		dataPtr,
		C.size_t(len(pdfData)),
		&e,
	)
	if e != 0 {
		return false, ffiError(e)
	}
	return r == 1, nil
}

// Close releases the underlying native signature handle. Safe to call
// more than once.
func (s *Signature) Close() {
	if s.handle != nil {
		C.pdf_signature_free(s.handle)
		s.handle = nil
	}
}

// TimestampHashAlgorithm matches the Rust `HashAlgorithm` enum / FFI
// contract: 1=SHA-1, 2=SHA-256, 3=SHA-384, 4=SHA-512, 0=Unknown.
type TimestampHashAlgorithm int32

// TimestampHashAlgorithm constants mirror the Rust `HashAlgorithm` enum /
// FFI contract. Use these when creating or inspecting a Timestamp.
const (
	TimestampHashUnknown TimestampHashAlgorithm = 0
	TimestampHashSha1    TimestampHashAlgorithm = 1
	TimestampHashSha256  TimestampHashAlgorithm = 2
	TimestampHashSha384  TimestampHashAlgorithm = 3
	TimestampHashSha512  TimestampHashAlgorithm = 4
)

// Timestamp is an RFC 3161 timestamp parsed from a DER TimeStampToken
// or a bare TSTInfo. Close() must be called to release the native
// handle.
type Timestamp struct {
	handle unsafe.Pointer
}

// ParseTimestamp parses a DER-encoded RFC 3161 TimeStampToken (or a
// bare TSTInfo SEQUENCE) into a Timestamp.
func ParseTimestamp(data []byte) (*Timestamp, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("pdf_oxide: timestamp data is empty: %w", ErrEmptyContent)
	}
	var e C.int
	h := C.pdf_timestamp_parse((*C.uint8_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)), &e)
	if e != 0 {
		return nil, ffiError(e)
	}
	if h == nil {
		return nil, fmt.Errorf("pdf_oxide: pdf_timestamp_parse returned null")
	}
	return &Timestamp{handle: h}, nil
}

// Time returns the generation time of the timestamp as a Unix epoch.
func (t *Timestamp) Time() (int64, error) {
	if t.handle == nil {
		return 0, ErrDocumentClosed
	}
	var e C.int
	ts := C.pdf_timestamp_get_time(t.handle, &e)
	if e != 0 {
		return 0, ffiError(e)
	}
	return int64(ts), nil
}

// Serial returns the serial number as a hex string (no 0x prefix).
func (t *Timestamp) Serial() (string, error) {
	return sigReadString(t.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_timestamp_get_serial(h, e)
	})
}

// PolicyOid returns the TSA policy OID in dotted-decimal form.
func (t *Timestamp) PolicyOid() (string, error) {
	return sigReadString(t.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_timestamp_get_policy_oid(h, e)
	})
}

// TsaName returns the TSA name declared in the TSTInfo, or "" if
// absent.
func (t *Timestamp) TsaName() (string, error) {
	return sigReadString(t.handle, func(h unsafe.Pointer, e *C.int) *C.char {
		return C.pdf_timestamp_get_tsa_name(h, e)
	})
}

// HashAlgorithm returns the hash algorithm of the message imprint.
func (t *Timestamp) HashAlgorithm() (TimestampHashAlgorithm, error) {
	if t.handle == nil {
		return TimestampHashUnknown, ErrDocumentClosed
	}
	var e C.int
	code := C.pdf_timestamp_get_hash_algorithm(t.handle, &e)
	if e != 0 {
		return TimestampHashUnknown, ffiError(e)
	}
	return TimestampHashAlgorithm(int32(code)), nil
}

// MessageImprint returns the raw message-imprint hash bytes.
func (t *Timestamp) MessageImprint() ([]byte, error) {
	if t.handle == nil {
		return nil, ErrDocumentClosed
	}
	var e C.int
	var outLen C.size_t
	p := C.pdf_timestamp_get_message_imprint(t.handle, &outLen, &e)
	if e != 0 {
		return nil, ffiError(e)
	}
	if p == nil || outLen == 0 {
		return nil, nil
	}
	return C.GoBytes(unsafe.Pointer(p), C.int(outLen)), nil
}

// Verify cryptographically verifies the timestamp. Currently returns
// ErrUnsupportedFeature until the Rust CMS signer verification path
// lands.
func (t *Timestamp) Verify() (bool, error) {
	if t.handle == nil {
		return false, ErrDocumentClosed
	}
	var e C.int
	ok := C.pdf_timestamp_verify(t.handle, &e)
	if e != 0 {
		return false, ffiError(e)
	}
	return bool(ok), nil
}

// Close releases the native timestamp handle. Safe to call more than
// once.
func (t *Timestamp) Close() {
	if t.handle != nil {
		C.pdf_timestamp_free(t.handle)
		t.handle = nil
	}
}

// TsaClientOptions configures a TsaClient. Only Url is required;
// everything else mirrors the Rust-core TsaClientConfig defaults.
type TsaClientOptions struct {
	URL            string
	Username       string // optional
	Password       string // optional
	TimeoutSeconds int32  // 0 falls back to 30s
	HashAlgorithm  TimestampHashAlgorithm
	UseNonce       bool
	CertReq        bool
}

// NewTsaClientOptions returns options with Rust-core-matching defaults.
func NewTsaClientOptions(url string) TsaClientOptions {
	return TsaClientOptions{
		URL:            url,
		TimeoutSeconds: 30,
		HashAlgorithm:  TimestampHashSha256,
		UseNonce:       true,
		CertReq:        true,
	}
}

// TsaClient is an RFC 3161 TSA HTTP client.
// Only linked when pdf_oxide was built with the `tsa-client` Rust-core
// feature; otherwise the FFI entry returns ErrUnsupportedFeature.
type TsaClient struct {
	handle unsafe.Pointer
}

// NewTsaClient builds a TSA client. Network is not touched until
// RequestTimestamp / RequestTimestampHash is called.
func NewTsaClient(opts TsaClientOptions) (*TsaClient, error) {
	if opts.URL == "" {
		return nil, fmt.Errorf("pdf_oxide: TSA url must not be empty")
	}
	cURL := C.CString(opts.URL)
	defer C.free(unsafe.Pointer(cURL))
	cUser := C.CString(opts.Username)
	defer C.free(unsafe.Pointer(cUser))
	cPass := C.CString(opts.Password)
	defer C.free(unsafe.Pointer(cPass))
	var e C.int
	h := C.pdf_tsa_client_create(
		cURL, cUser, cPass,
		C.int32_t(opts.TimeoutSeconds),
		C.int32_t(opts.HashAlgorithm),
		C.bool(opts.UseNonce),
		C.bool(opts.CertReq),
		&e,
	)
	if e != 0 {
		return nil, ffiError(e)
	}
	if h == nil {
		return nil, fmt.Errorf("pdf_oxide: pdf_tsa_client_create returned null")
	}
	return &TsaClient{handle: h}, nil
}

// RequestTimestamp hashes data with the configured algorithm and
// requests a timestamp for the digest.
func (c *TsaClient) RequestTimestamp(data []byte) (*Timestamp, error) {
	if c.handle == nil {
		return nil, ErrDocumentClosed
	}
	var e C.int
	var dataPtr *C.uint8_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	h := C.pdf_tsa_request_timestamp(c.handle, dataPtr, C.size_t(len(data)), &e)
	if e != 0 {
		return nil, ffiError(e)
	}
	if h == nil {
		return nil, fmt.Errorf("pdf_oxide: pdf_tsa_request_timestamp returned null")
	}
	return &Timestamp{handle: h}, nil
}

// RequestTimestampHash requests a timestamp for a pre-computed hash.
func (c *TsaClient) RequestTimestampHash(hash []byte, algo TimestampHashAlgorithm) (*Timestamp, error) {
	if c.handle == nil {
		return nil, ErrDocumentClosed
	}
	var e C.int
	var hashPtr *C.uint8_t
	if len(hash) > 0 {
		hashPtr = (*C.uint8_t)(unsafe.Pointer(&hash[0]))
	}
	h := C.pdf_tsa_request_timestamp_hash(
		c.handle, hashPtr, C.size_t(len(hash)), C.int32_t(algo), &e,
	)
	if e != 0 {
		return nil, ffiError(e)
	}
	if h == nil {
		return nil, fmt.Errorf("pdf_oxide: pdf_tsa_request_timestamp_hash returned null")
	}
	return &Timestamp{handle: h}, nil
}

// Close releases the client handle. Safe to call more than once.
func (c *TsaClient) Close() {
	if c.handle != nil {
		C.pdf_tsa_client_free(c.handle)
		c.handle = nil
	}
}

func sigReadString(h unsafe.Pointer, fn func(unsafe.Pointer, *C.int) *C.char) (string, error) {
	if h == nil {
		return "", ErrDocumentClosed
	}
	var e C.int
	p := fn(h, &e)
	if e != 0 {
		return "", ffiError(e)
	}
	if p == nil {
		return "", nil
	}
	defer C.free_string(p)
	return C.GoString(p), nil
}

// ================================================================
// io.Reader support
// ================================================================

// OpenReader opens a PDF document from an io.Reader by reading all bytes.
// This is the idiomatic Go way to load PDFs from HTTP responses, archives,
// or any other stream source without writing to a temporary file.
func OpenReader(r io.Reader) (*PdfDocument, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}
	return OpenFromBytes(data)
}

// ================================================================
// Logging — LogLevel type and constants live in types.go (tag-free)
// so both backends share them.
// ================================================================

// SetLogLevel sets the global log level for the pdf_oxide library.
// Use the LogLevel constants (LogOff, LogError, LogWarn, LogInfo, LogDebug, LogTrace).
func SetLogLevel(level LogLevel) {
	C.pdf_oxide_set_log_level(C.int(level))
}

// GetLogLevel returns the current log level of the pdf_oxide library.
func GetLogLevel() LogLevel {
	return LogLevel(C.pdf_oxide_get_log_level())
}

// ================================================================
// Crypto provider (issue #236)
// ================================================================

// ErrFipsNotCompiled is returned by UseFipsCryptoProvider when the
// native pdf_oxide library was built without the fips feature.
var ErrFipsNotCompiled = errors.New("FIPS provider not compiled in; rebuild native lib with --features fips")

// ErrCryptoProviderAlreadySet is returned by UseFipsCryptoProvider
// when a cryptographic provider has already been installed for the
// process. The set-once policy is intentional — see
// docs/CRYPTO_PROVIDERS.md.
var ErrCryptoProviderAlreadySet = errors.New("cryptographic provider already installed")

// ActiveCryptoProvider returns the name of the currently active
// cryptographic provider — "rust-crypto" for the default permissive
// provider, or "aws-lc-rs" once UseFipsCryptoProvider has been
// called.
func ActiveCryptoProvider() string {
	cstr := C.pdf_oxide_crypto_active_provider()
	if cstr == nil {
		return "unknown"
	}
	defer C.free_string(cstr)
	return C.GoString(cstr)
}

// IsFipsCryptoAvailable reports whether the FIPS-validated aws-lc-rs
// provider was compiled into the native library. Build the lib with
// --features fips to enable.
func IsFipsCryptoAvailable() bool {
	return C.pdf_oxide_crypto_fips_available() != 0
}

// UseFipsCryptoProvider installs the FIPS-validated aws-lc-rs
// provider as the process-wide active cryptographic backend. Must
// be called before any PDF operation that uses crypto.
func UseFipsCryptoProvider() error {
	switch C.pdf_oxide_crypto_use_fips() {
	case 0:
		return nil
	case 1:
		return ErrFipsNotCompiled
	case 2:
		return ErrCryptoProviderAlreadySet
	default:
		return fmt.Errorf("pdf_oxide_crypto_use_fips returned unknown error code")
	}
}

// ================================================================
// Page — a lightweight handle for a single page of a PdfDocument.
// ================================================================

// Page represents a single page of a PdfDocument.
// All methods dispatch to the parent document.
type Page struct {
	doc   *PdfDocument
	Index int
}

// Page returns a handle to the page at the given zero-based index.
func (doc *PdfDocument) Page(index int) (*Page, error) {
	count, err := doc.PageCount()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= count {
		return nil, fmt.Errorf("page index %d out of range [0, %d)", index, count)
	}
	return &Page{doc: doc, Index: index}, nil
}

// Pages returns all pages as a slice. Enables range loops.
func (doc *PdfDocument) Pages() ([]*Page, error) {
	count, err := doc.PageCount()
	if err != nil {
		return nil, err
	}
	pages := make([]*Page, count)
	for i := 0; i < count; i++ {
		pages[i] = &Page{doc: doc, Index: i}
	}
	return pages, nil
}

// Text extracts plain text from the page. Wrapper around PdfDocument.ExtractText.
func (p *Page) Text() (string, error) { return p.doc.ExtractText(p.Index) }

// Markdown renders the page as Markdown.
func (p *Page) Markdown() (string, error) { return p.doc.ToMarkdown(p.Index) }

// Html renders the page as HTML.
func (p *Page) Html() (string, error) { return p.doc.ToHtml(p.Index) }

// PlainText returns the page's stripped plain-text form.
func (p *Page) PlainText() (string, error) { return p.doc.ToPlainText(p.Index) }

// Chars returns the individual character records for the page.
func (p *Page) Chars() ([]Char, error) { return p.doc.ExtractChars(p.Index) }

// Words returns word-level records for the page.
func (p *Page) Words() ([]Word, error) { return p.doc.ExtractWords(p.Index) }

// Lines returns text-line records for the page.
func (p *Page) Lines() ([]TextLine, error) { return p.doc.ExtractTextLines(p.Index) }

// Tables returns detected tables on the page.
func (p *Page) Tables() ([]Table, error) { return p.doc.ExtractTables(p.Index) }

// Images returns embedded images on the page.
func (p *Page) Images() ([]Image, error) { return p.doc.Images(p.Index) }

// Paths returns vector paths on the page.
func (p *Page) Paths() ([]Path, error) { return p.doc.ExtractPaths(p.Index) }

// Fonts returns the fonts referenced by the page.
func (p *Page) Fonts() ([]Font, error) { return p.doc.Fonts(p.Index) }

// Annotations returns the page's annotations.
func (p *Page) Annotations() ([]Annotation, error) { return p.doc.Annotations(p.Index) }

// Info returns the page's metadata (size, rotation, …).
func (p *Page) Info() (*PageInfo, error) { return p.doc.PageInfo(p.Index) }

// Search runs a text search across the page. Case-sensitive if cs is true.
func (p *Page) Search(term string, cs bool) ([]SearchResult, error) {
	return p.doc.SearchPage(p.Index, term, cs)
}

// NeedsOcr reports whether the page appears to be scanned image content.
func (p *Page) NeedsOcr() (bool, error) { return p.doc.NeedsOcr(p.Index) }

// TextWithOcr runs OCR on the page using the supplied engine.
func (p *Page) TextWithOcr(engine *OcrEngine) (string, error) {
	return p.doc.ExtractTextWithOcr(p.Index, engine)
}
