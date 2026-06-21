# frozen_string_literal: true

require 'ffi'
require_relative 'library'

module PdfOxide
  module FFI
    # All FFI function bindings for PDF Oxide library
    # Total: 315+ functions covering all PDF operations
    module Bindings
      extend ::FFI::Library

      # Load native library
      begin
        ffi_lib(Library.library_path)
      rescue LoadError => e
        raise ::PdfOxide::InternalError, "Failed to load PDF Oxide native library: #{e.message}. " \
          'Make sure libpdf_oxide is installed.'
      end

      # ============================================================
      # ERROR HANDLING & MEMORY MANAGEMENT
      # ============================================================

      # ============================================================
      # UTILITY FUNCTIONS (16 total)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :alloc_string (1 line)
      attach_function :free_string, [:pointer], :void
      attach_function :free_bytes, [:pointer], :void

      # Document Editor Operations (13)
      attach_function :document_editor_open, %i[string pointer], :pointer
      attach_function :document_editor_free, [:pointer], :void
      attach_function :document_editor_save, %i[pointer string pointer], :bool
      attach_function :document_editor_get_page_count, %i[pointer pointer], :int32
      # char *document_editor_get_source_path(const DocumentEditor *handle, int32_t *error_code)
      # Owned-char* — bind as :pointer so the caller can free via StringMarshaller.
      attach_function :document_editor_get_source_path, %i[pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :document_editor_get_title
      # REMOVED phantom (no upstream symbol): :document_editor_get_author
      # REMOVED phantom (no upstream symbol): :document_editor_get_subject
      # void document_editor_get_version(const DocumentEditor *handle,
      #                                  uint8_t *major, uint8_t *minor)
      attach_function :document_editor_get_version, %i[pointer pointer pointer], :void
      attach_function :document_editor_set_title, %i[pointer string pointer], :bool
      attach_function :document_editor_set_author, %i[pointer string pointer], :bool
      attach_function :document_editor_set_subject, %i[pointer string pointer], :bool
      attach_function :document_editor_is_modified, [:pointer], :bool

      # ============================================================
      # DOCUMENT OPERATIONS
      # ============================================================

      # Core document operations
      attach_function :pdf_document_open, %i[string pointer], :pointer
      attach_function :pdf_document_free, [:pointer], :void
      attach_function :pdf_document_get_page_count, %i[pointer pointer], :int32
      # bool pdf_document_is_encrypted(const PdfDocument *handle) — no err arg
      attach_function :pdf_document_is_encrypted, [:pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_document_requires_password (1 line)

      # Document metadata
      # void pdf_document_get_version(const PdfDocument *handle,
      #                               uint8_t *major, uint8_t *minor)
      attach_function :pdf_document_get_version, %i[pointer pointer pointer], :void
      attach_function :pdf_document_has_structure_tree, [:pointer], :bool

      # ============================================================
      # TEXT EXTRACTION
      # ============================================================

      # Owned-`char *` extraction APIs. Declared as :pointer (NOT :string)
      # so callers can free the buffer via StringMarshaller.from_c_string,
      # which delegates to free_string. Ruby FFI's :string copies the C
      # bytes but never calls free_string → leaks one full-document buffer
      # per call.
      attach_function :pdf_document_extract_text, %i[pointer int32 pointer], :pointer
      # char *pdf_document_extract_structured_to_json(void *handle, int32_t page_index, int32_t *error_code)
      # Returns owned char* (serialized StructuredPage JSON) — bind as
      # :pointer so the caller frees via StringMarshaller/free_string (#536).
      attach_function :pdf_document_extract_structured_to_json, %i[pointer int32 pointer], :pointer
      attach_function :pdf_document_to_markdown, %i[pointer int32 pointer], :pointer
      attach_function :pdf_document_to_markdown_all, %i[pointer pointer], :pointer
      attach_function :pdf_document_to_html, %i[pointer int32 pointer], :pointer
      attach_function :pdf_document_to_plain_text, %i[pointer int32 pointer], :pointer

      # ============================================================
      # SEARCH OPERATIONS (15 functions)
      # ============================================================

      attach_function :pdf_document_search_page, %i[pointer string int32 bool pointer], :pointer
      attach_function :pdf_document_search_all, %i[pointer string bool pointer], :pointer
      attach_function :pdf_oxide_search_result_count, [:pointer], :int32
      # Each accessor takes the results handle + index + an int32* error
      # buffer. The trailing pointer is REQUIRED — omitting it caused
      # the cdylib to dereference register garbage as the err pointer
      # and segfault on aarch64/macOS-arm64 (issue #547 v0.3.55 CI).
      attach_function :pdf_oxide_search_result_get_page, %i[pointer int32 pointer], :int32
      # Returns owned char* — bind as :pointer + StringMarshaller to
      # free via free_string. Declaring as :string leaked the buffer
      # AND read past the missing err arg.
      attach_function :pdf_oxide_search_result_get_text, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_search_result_get_bbox,
                      %i[pointer int32 pointer pointer pointer pointer pointer], :void
      attach_function :pdf_oxide_search_result_free, [:pointer], :void

      # ============================================================
      # PAGE OPERATIONS (30 functions)
      # ============================================================

      # ============================================================
      # RENDERING OPERATIONS (25 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_render_page_to_file (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_page_to_bytes (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_page_range (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_document (1 line)
      attach_function :pdf_render_page_fit, %i[pointer int32 int32 int32 int32 pointer], :pointer
      attach_function :pdf_render_page_zoom, %i[pointer int32 float int32 pointer], :pointer
      attach_function :pdf_render_page_region, %i[pointer int32 float float float float int32 pointer], :pointer
      # FfiRenderedImage *pdf_render_page_thumbnail(PdfDocument *doc, int32_t page_index, int32_t _size, int32_t format, int32_t *error_code)
      attach_function :pdf_render_page_thumbnail, %i[pointer int32 int32 int32 pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_width (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_height (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_data (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_to_base64 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_save (1 line)
      attach_function :pdf_rendered_image_free, [:pointer], :void

      # ============================================================
      # ANNOTATION OPERATIONS (20 functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_document_get_annotations (1 line)
      attach_function :pdf_oxide_annotation_count, [:pointer], :int32
      # char *pdf_oxide_annotation_get_type(const FfiAnnotationList *annotations, int32_t index, int32_t *error_code)
      # Pre-v0.3.55 declared as 2-arg `:int32` — wrong arg count AND wrong
      # return type (C returns owned char*). Both fixed here.
      attach_function :pdf_oxide_annotation_get_type, %i[pointer int32 pointer], :pointer

      # ============================================================
      # FORM OPERATIONS (20 functions)
      # ============================================================

      # Form field operations
      # REMOVED phantom (no upstream symbol): :pdf_form_export_to_fdf (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_form_export_to_xfdf (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_form_export_to_json (1 line)
      attach_function :pdf_form_import_from_file, %i[pointer string pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_form_reset_all_fields (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_form_field_find_by_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_form_field_set_value_by_name_string (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_form_field_set_value_by_name_boolean (1 line)

      # ============================================================
      # FONT OPERATIONS (15 functions)
      # ============================================================

      attach_function :pdf_document_get_embedded_fonts, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_font_count, [:pointer], :int32
      # Font accessors all share (fonts, index, int32_t *error_code).
      # Owned-char* return for `_get_name` — bind as :pointer.
      attach_function :pdf_oxide_font_get_name, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_font_get_size, %i[pointer int32 pointer], :float
      attach_function :pdf_oxide_font_is_embedded, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_font_list_free, [:pointer], :void

      # ============================================================
      # IMAGE OPERATIONS (20 functions)
      # ============================================================

      attach_function :pdf_document_get_embedded_images, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_image_count, [:pointer], :int32
      # Image accessors: (images, index, int32_t *error_code) → int32.
      attach_function :pdf_oxide_image_get_width, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_image_get_height, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_image_get_bits_per_component, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_image_list_free, [:pointer], :void

      # ============================================================
      # OUTLINE/BOOKMARK OPERATIONS (4 functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_document_get_outline_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_outline_title (1 line)

      # ============================================================
      # LAYER OPERATIONS (3 functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_document_get_layer_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_layer_name (1 line)

      # ============================================================
      # METADATA OPERATIONS (12 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_document_get_title (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_author (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_subject (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_keywords (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_creator (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_producer (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_creation_date (1 line)

      # ============================================================
      # OCR OPERATIONS (15 functions)
      # ============================================================

      attach_function :pdf_ocr_engine_free, [:pointer], :void

      # ============================================================
      # DIGITAL SIGNATURE OPERATIONS (15 functions)
      # ============================================================

      # Phase 1 signing functions
      # REMOVED phantom (no upstream symbol): :pdf_credentials_from_pkcs12 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_from_pem (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_sign_data (8 lines)
      #
      #
      #
      #
      #
      #
      #
      # REMOVED phantom (no upstream symbol): :pdf_document_sign_file (7 lines)
      #
      #
      #
      #
      #
      #
      # REMOVED phantom (no upstream symbol): :pdf_embed_ltv_data (7 lines)
      #
      #
      #
      #
      #
      #
      # REMOVED phantom (no upstream symbol): :pdf_document_save_signed (5 lines)
      #
      #
      #
      #
      # REMOVED phantom (no upstream symbol): :pdf_signed_bytes_free (1 line)

      # ============================================================
      # BARCODE OPERATIONS (6 functions)
      # ============================================================

      # ============================================================
      # COMPLIANCE OPERATIONS (20 functions)
      # ============================================================

      # ============================================================
      # CACHE OPERATIONS (5 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_cache_clear (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_cache_invalidate_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_cache_get_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_cache_set_max_size (1 line)

      # ============================================================
      # ANALYSIS OPERATIONS (10 functions)
      # ============================================================

      # ============================================================
      # CONVERSION OPERATIONS (10 functions)
      # ============================================================

      # ============================================================
      # XFA FORM OPERATIONS (12 functions)
      # ============================================================

      attach_function :pdf_document_has_xfa, %i[pointer pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_parse_xfa_form (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_field_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_get_field (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_get_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_get_dataset (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_dataset_to_xml (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_dataset_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_convert_xfa_to_acroform (1 line)

      # ============================================================
      # ANALYSIS/ML OPERATIONS (12 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_analyze_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_complexity (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_complexity_score (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_content_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_text_density (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_image_density (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_result_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analyze_document (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_estimate_processing_time (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_detect_columns (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_detect_tables (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ml_get_status (1 line)

      # ============================================================
      # ADDITIONAL RENDERING OPERATIONS (15 functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_create (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_set_options (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_get_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_reset_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_convert (1 line)

      # ============================================================
      # ADDITIONAL OCR OPERATIONS (12 functions)
      # ============================================================

      attach_function :pdf_ocr_engine_create, %i[pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_ocr_engine_get_version (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_engine_get_status (1 line)
      attach_function :pdf_ocr_page_needs_ocr, %i[pointer int32 pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_ocr_detect_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_recognize_page (1 line)
      # char *pdf_ocr_extract_text(PdfDocument *doc, int32_t page_index, const void *engine, int32_t *error_code)
      # Pre-v0.3.55 had a phantom 5th `bool` arg and a leaky :string return.
      attach_function :pdf_ocr_extract_text, %i[pointer int32 pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_ocr_extract_spans (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_extract_pages (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_get_text (1 line)

      # ============================================================
      # OCR RESULT ACCESSORS (6 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_average_confidence (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_get_char_confidence (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_batch_results_get_page (1 line)

      # ============================================================
      # CERTIFICATE AND SIGNATURE OPERATIONS (8 functions)
      # ============================================================

      attach_function :pdf_document_get_signature, %i[pointer int32 pointer], :pointer
      attach_function :pdf_signature_free, [:pointer], :void
      attach_function :pdf_certificate_load_from_bytes, %i[pointer size_t string pointer], :pointer
      attach_function :pdf_certificate_free, [:pointer], :void
      attach_function :pdf_document_sign, %i[pointer pointer string string pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_signer (1 line)
      attach_function :pdf_signature_verify, %i[pointer pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_compliance_issue_free (1 line)

      # ============================================================
      # COMPLIANCE RESULT ACCESSORS (12 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_validate_pdf_a (1 line)
      # bool pdf_pdf_a_is_compliant(const FfiPdfAResults *results, int32_t *error_code)
      attach_function :pdf_pdf_a_is_compliant, %i[pointer pointer], :bool
      attach_function :pdf_pdf_a_error_count, [:pointer], :int32
      attach_function :pdf_pdf_a_warning_count, [:pointer], :int32
      attach_function :pdf_pdf_a_get_error, %i[pointer int32 pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_pdf_a_get_warning (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_a_get_report (1 line)
      attach_function :pdf_pdf_a_results_free, [:pointer], :void
      # REMOVED phantom (no upstream symbol): :pdf_validate_pdf_x (1 line)
      # bool pdf_pdf_x_is_compliant(const FfiPdfXResults *results, int32_t *error_code)
      attach_function :pdf_pdf_x_is_compliant, %i[pointer pointer], :bool
      attach_function :pdf_pdf_x_error_count, [:pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_warning_count (1 line)

      # ============================================================
      # PDF/X RESULT ACCESSORS (6 functions)
      # ============================================================

      attach_function :pdf_pdf_x_get_error, %i[pointer int32 pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_get_warning (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_get_report (1 line)
      attach_function :pdf_pdf_x_results_free, [:pointer], :void
      attach_function :pdf_validate_pdf_ua, %i[pointer int32 pointer], :pointer
      # bool pdf_pdf_ua_is_accessible(const FfiUaResults *results, int32_t *error_code)
      attach_function :pdf_pdf_ua_is_accessible, %i[pointer pointer], :bool

      # ============================================================
      # PDF/UA RESULT ACCESSORS (4 functions)
      # ============================================================

      attach_function :pdf_pdf_ua_error_count, [:pointer], :int32
      attach_function :pdf_pdf_ua_get_error, %i[pointer int32 pointer], :pointer
      attach_function :pdf_pdf_ua_results_free, [:pointer], :void
      attach_function :pdf_convert_to_pdf_a, %i[pointer int32 pointer], :bool

      # ============================================================
      # CONVERSION OPERATIONS (4 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_convert_to_pdf_x (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_convert_to_pdf_ua (1 line)

      # ============================================================
      # BARCODE OPERATIONS (7 functions)
      # ============================================================

      # FfiBarcodeImage *pdf_generate_qr_code(const char *data, int32_t error_correction, int32_t size_px, int32_t *error_code)
      attach_function :pdf_generate_qr_code, %i[string int32 int32 pointer], :pointer
      # FfiBarcodeImage *pdf_generate_barcode(const char *data, int32_t format, int32_t size_px, int32_t *error_code)
      # Pre-v0.3.55 had (int32, string, pointer) — args were reordered AND
      # missing the size_px slot.
      attach_function :pdf_generate_barcode, %i[string int32 int32 pointer], :pointer
      attach_function :pdf_barcode_get_image_png, %i[pointer int32 pointer pointer], :pointer
      # char *pdf_barcode_get_svg(const FfiBarcodeImage *handle, int32_t _size_px, int32_t *error_code)
      attach_function :pdf_barcode_get_svg, %i[pointer int32 pointer], :pointer
      attach_function :pdf_barcode_free, [:pointer], :void
      attach_function :pdf_add_barcode_to_page, %i[pointer int32 pointer float float float float pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_ml_model_available (1 line)

      # ============================================================
      # STRATEGY AND EXTRACTION (6 functions)
      # ============================================================

      # REMOVED phantom (no upstream symbol): :pdf_create_extraction_strategy (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_get_description (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_recommends_ocr (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_free (1 line)

      # ============================================================
      # ADDITIONAL TEXT OPERATIONS (12 functions)
      # ============================================================

      # ============================================================
      # ADDITIONAL HELPER FUNCTIONS (60+ functions)
      # ============================================================

      # Links and embedded files

      # Font usage

      # Barcodes

      # Analysis

      # Document analysis

      # Signatures

      # Signature verification

      # Certificates

      # XFA forms

      # Cache

      # Text extraction all pages

      # Document rendering utilities

      # ML functions

      # Additional document methods

      # ============================================================
      # MISSING REAL RUST FUNCTIONS (73 total) - NOW ADDED
      # ============================================================

      # ANNOTATION (21)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_author (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_color (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_contents (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_flags (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_opacity (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_subject (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_get_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_freetext_annotation_get_font_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_freetext_annotation_get_font_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_link_annotation_get_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_link_annotation_get_uri (1 line)
      attach_function :pdf_oxide_annotation_get_author, %i[pointer int32 pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_oxide_annotation_get_contents (1 line)
      attach_function :pdf_oxide_annotation_get_creation_date, %i[pointer int32 pointer], :int64
      # void pdf_oxide_annotation_get_rect(const FfiAnnotationList *annotations, int32_t index, float *x, float *y, float *width, float *height, int32_t *error_code)
      # Pre-v0.3.55 had reversed pointer/int32 order AND was missing 3 args.
      attach_function :pdf_oxide_annotation_get_rect,
                      %i[pointer int32 pointer pointer pointer pointer pointer], :void
      # REMOVED phantom (no upstream symbol): :pdf_page_get_annotations_by_type_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_get_annotations_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_text_annotation_get_icon (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_text_annotation_get_open (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_text_markup_annotation_get_type (1 line)

      # DOCUMENT (10)
      # REMOVED phantom (no upstream symbol): :pdf_document_can_annotate (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_can_copy (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_can_fill_forms (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_can_modify (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_can_print (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_encryption_algorithm (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_mod_date (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_outline_level (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_outline_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_is_layer_visible (1 line)

      # PAGE (8)
      attach_function :pdf_get_page_count, %i[pointer pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_page_find_elements_count (1 line)
      attach_function :pdf_page_get_height, [:pointer], :float
      # REMOVED phantom (no upstream symbol): :pdf_page_get_index (1 line)
      attach_function :pdf_page_get_width, [:pointer], :float
      # REMOVED phantom (no upstream symbol): :pdf_page_search_text (1 line)
      attach_function :pdf_render_page, %i[pointer int32 pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_search_result_get_page (1 line)

      # ELEMENT (9)
      # REMOVED phantom (no upstream symbol): :pdf_element_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_element_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_element_get_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_element_get_data (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_element_get_data_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_element_get_dimensions (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_element_get_format (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_text_element_get_content (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_text_element_get_font_size (1 line)

      # SEARCH (3)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_search_result_get_position (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_search_result_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_search_result_get_text (1 line)

      # RENDERING (2)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_copy_data (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_format (1 line)

      # OCR (2)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_get_span (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_to_text_span (1 line)

      # IMAGE (3)
      attach_function :pdf_oxide_image_get_colorspace, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_image_get_data, %i[pointer int32 pointer pointer], :pointer
      attach_function :pdf_oxide_image_get_format, %i[pointer int32 pointer], :pointer

      # TEXT (1)
      attach_function :pdf_from_text, %i[string pointer], :pointer

      # OTHER (8)
      # REMOVED phantom (no upstream symbol): :pdf_cache_get_statistics_json (1 line)
      attach_function :pdf_from_html, %i[string pointer], :pointer
      attach_function :pdf_from_markdown, %i[string pointer], :pointer
      attach_function :pdf_oxide_font_get_encoding, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_font_get_type, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_font_is_subset, %i[pointer int32 pointer], :int32
      attach_function :pdf_save, %i[pointer string pointer], :int32
      attach_function :pdf_save_to_bytes, %i[pointer pointer pointer pointer], :int32

      # FREE/CLEANUP FUNCTIONS (6)
      # REMOVED phantom (no upstream symbol): :pdf_annotation_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_search_result_free (1 line)
      attach_function :pdf_free, [:pointer], :void
      # REMOVED phantom (no upstream symbol): :pdf_oxide_annotation_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_get_dimensions (1 line)

      # ============================================================
      # PHASE 1: ADDITIONAL FFI BINDINGS (122 MISSING FUNCTIONS)
      # ============================================================

      # ============================================================
      # OCR OPERATIONS (16 missing functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_get_char_confidence (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_batch_results_get_page (1 line)
      # REMOVED duplicate declaration: :pdf_ocr_page_needs_ocr (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_detect_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_recognize_page (1 line)
      # REMOVED duplicate declaration: :pdf_ocr_extract_text (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_extract_spans (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_extract_pages (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_engine_get_version (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_engine_get_status (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_get_text (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_average_confidence (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_results_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_create (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_set_detection_threshold (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_set_recognition_threshold (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_set_max_side_len (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_set_use_gpu (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_set_gpu_device_id (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_config_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_engine_create_with_config (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_span_to_text_span (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_ocr_result_confidence (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_gpu_available (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_ocr_gpu_device_count (1 line)

      # ============================================================
      # RENDERING OPERATIONS (20 missing functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_create (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_set_options (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_page_renderer_free (1 line)
      # REMOVED duplicate declaration: :pdf_render_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_page_to_file (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_page_range (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_document (1 line)
      # REMOVED duplicate declaration: :pdf_render_page_region (1 line)
      # REMOVED duplicate declaration: :pdf_render_page_zoom (1 line)
      # REMOVED duplicate declaration: :pdf_render_page_fit (1 line)
      # REMOVED duplicate declaration: :pdf_render_page_thumbnail (1 line)
      attach_function :pdf_estimate_render_time, %i[pointer int32 pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_renderer_get_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_reset_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_width (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_height (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_data (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_convert (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_to_base64 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_render_page_to_base64 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_format_mime_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_image_format_extension (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_statistics_pages_rendered (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_statistics_total_time (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_renderer_statistics_avg_time (1 line)

      # ============================================================
      # BARCODE OPERATIONS (13 missing functions)
      # ============================================================
      # REMOVED duplicate declaration: :pdf_generate_qr_code (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_generate_ean13 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_generate_ean8 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_generate_upc_a (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_generate_code128 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_generate_code39 (1 line)
      # REMOVED duplicate declaration: :pdf_barcode_get_image_png (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_barcode_get_image_base64 (1 line)
      # REMOVED duplicate declaration: :pdf_barcode_get_svg (1 line)
      # REMOVED duplicate declaration: :pdf_barcode_free (1 line)
      # REMOVED duplicate declaration: :pdf_add_barcode_to_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_add_barcode_to_page_fit (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_add_qr_code_with_label (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_detect_barcodes_on_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_barcode_get_bounds (1 line)
      # Barcode accessors: (handle, int32_t *error_code).
      # pdf_barcode_get_data returns owned char* — bind as :pointer.
      attach_function :pdf_barcode_get_confidence, %i[pointer pointer], :float
      attach_function :pdf_barcode_get_data,       %i[pointer pointer], :pointer
      attach_function :pdf_barcode_get_format,     %i[pointer pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_oxide_barcode_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_barcode_get_data (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_barcode_get_format (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_barcode_list_free (1 line)

      # ============================================================
      # DIGITAL SIGNATURE OPERATIONS (38 missing functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_credentials_from_pkcs12 (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_from_pem (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_from_der (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_add_chain_cert (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_get_certificate (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_credentials_free (1 line)
      # Certificate accessors: (cert, int32_t *error_code) returning
      # owned char* — bind as :pointer so callers can free via
      # StringMarshaller.
      attach_function :pdf_certificate_get_subject, %i[pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_cn (1 line)
      attach_function :pdf_certificate_get_issuer, %i[pointer pointer], :pointer
      attach_function :pdf_certificate_get_serial, %i[pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_size (1 line)
      attach_function :pdf_certificate_get_validity, %i[pointer pointer pointer pointer], :void
      # int32_t pdf_certificate_is_valid(const void *cert, int32_t *error_code)
      # Pre-v0.3.55 had 1-arg :bool — wrong arg count AND wrong return type
      # (C returns int32_t; 1 = valid, 0 = invalid, negative = error).
      attach_function :pdf_certificate_is_valid, %i[pointer pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_certificate_is_expired (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_key_size (1 line)
      # REMOVED duplicate declaration: :pdf_certificate_free (1 line)
      # REMOVED duplicate declaration: :pdf_document_sign (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_sign_with_appearance (10 lines)
      #
      #
      #
      #
      #
      #
      #
      #
      #
      attach_function :pdf_add_timestamp, [
        :pointer, :size_t,     # pdf_data, pdf_len
        :int32,                # signature_index
        :string,               # tsa_url
        :pointer, :pointer,    # out_data, out_len
        :pointer               # error_code
      ], :bool
      # REMOVED phantom (no upstream symbol): :pdf_document_co_sign (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_signature_count (1 line)
      # REMOVED duplicate declaration: :pdf_document_get_signature (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_verify_signature (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_verify_all_signatures (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_time (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_reason (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_location (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_contact (1 line)
      attach_function :pdf_signature_get_certificate, %i[pointer pointer], :pointer
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_subfilter (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_get_digest_algorithm (1 line)
      # bool pdf_signature_has_timestamp(const void *_sig, int32_t *error_code)
      attach_function :pdf_signature_has_timestamp, %i[pointer pointer], :bool
      # REMOVED phantom (no upstream symbol): :pdf_signature_to_json (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_signature_info_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_remove_signature (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_clear_all_signatures (1 line)

      # ============================================================
      # PAdES Level Enforcement
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_pades_validate_level (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pades_sign (8 lines)
      #
      #
      #
      #
      #
      #
      #
      # REMOVED phantom (no upstream symbol): :pdf_pades_get_level (1 line)

      # ============================================================
      # XFA FORM OPERATIONS (13 missing functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_get_xfa_form_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_get_title (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_page_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_form_find_field (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_get_label (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_get_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_get_value (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_set_value (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_is_required (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_field_is_readonly (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_xfa_dataset_to_json (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_extract_xfa_as_fdf (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_get_xfa_template_xml (1 line)

      # ============================================================
      # COMPLIANCE OPERATIONS (23 missing functions)
      # ============================================================
      # REMOVED phantom (no upstream symbol): :pdf_validate_pdf_a (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_a_is_compliant (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_a_error_count (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_a_warning_count (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_a_get_error (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_a_get_warning (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_a_get_report (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_a_results_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_validate_pdf_x (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_x_is_compliant (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_x_error_count (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_x_get_error (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_get_report (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_x_results_free (1 line)
      # REMOVED duplicate declaration: :pdf_validate_pdf_ua (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_ua_is_compliant (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_ua_issue_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_ua_get_issue (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_ua_get_report (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_ua_results_free (1 line)
      # REMOVED duplicate declaration: :pdf_convert_to_pdf_a (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_conversion_success (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_conversion_modification_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_conversion_get_modification (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_conversion_get_report (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_conversion_results_free (1 line)

      # ============================================================
      # ADDITIONAL MANAGER SUPPORT FUNCTIONS
      # ============================================================

      # Analysis functions used by managers
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_block_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_column_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_image_block_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_layout_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_table_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_analysis_get_text_block_count (1 line)

      # Document edit functions
      # REMOVED phantom (no upstream symbol): :pdf_document_delete_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_insert_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_move_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_duplicate_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_merge_pages (1 line)

      # Page dimension functions
      # REMOVED phantom (no upstream symbol): :pdf_document_get_media_box (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_crop_box (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_crop_box (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_page_label (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_page_rotation (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_page_rotation (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_page_width (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_page_height (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_page_complexity (1 line)

      # Document property functions
      # REMOVED phantom (no upstream symbol): :pdf_document_has_layers (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_has_outlines (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_has_javascript (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_has_valid_signatures (1 line)

      # Form functions
      # REMOVED phantom (no upstream symbol): :pdf_document_has_acro_forms (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_form_field_names (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_form_field_count (1 line)

      # Image extraction functions
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_all_images (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_image (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_font (1 line)

      # Layout and structure functions
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_layout (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_with_layout (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_with_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_detect_columns (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_detect_tables (1 line)

      # Search functions
      # REMOVED phantom (no upstream symbol): :pdf_document_search_regex (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_search_in_range (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_search_in_area (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_search_annotations (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_count_text_occurrences (1 line)

      # Text functions
      # REMOVED phantom (no upstream symbol): :pdf_document_get_text_with_coordinates (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_unique_characters (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_text_statistics (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_replace_text (1 line)

      # Analysis functions
      # REMOVED phantom (no upstream symbol): :pdf_document_analyze_page (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_complexity_score (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_content_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_text_density (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_image_density (1 line)

      # Link extraction functions
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_links (1 line)

      # Embedded files functions
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_embedded_files (1 line)

      # Font usage functions
      # REMOVED phantom (no upstream symbol): :pdf_document_get_font_usage (1 line)

      # OCR functions
      # REMOVED phantom (no upstream symbol): :pdf_document_apply_ocr (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_ocr_page (1 line)
      # REMOVED duplicate declaration: :pdf_ocr_engine_create (1 line)

      # Signature functions
      # REMOVED duplicate declaration: :pdf_document_sign (1 line)
      attach_function :pdf_document_get_signature_count, %i[pointer pointer], :int32

      # Save functions
      # REMOVED phantom (no upstream symbol): :pdf_document_save (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_save_incremental (1 line)

      # Metadata set functions
      # REMOVED phantom (no upstream symbol): :pdf_document_set_title (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_author (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_subject (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_keywords (1 line)

      # Compliance validation functions
      # REMOVED phantom (no upstream symbol): :pdf_document_validate_pdf_a (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_validate_pdf_x (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_validate_pdf_ua (1 line)

      # Conversion functions
      # REMOVED phantom (no upstream symbol): :pdf_convert_to_pdf_ua (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_convert_to_pdf_x (1 line)

      # Extraction strategy functions
      # REMOVED phantom (no upstream symbol): :pdf_create_extraction_strategy (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_get_description (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_recommends_ocr (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_strategy_free (1 line)

      # Rendering functions
      # REMOVED phantom (no upstream symbol): :pdf_render_page_to_bytes (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_rendered_image_save (1 line)
      # REMOVED duplicate declaration: :pdf_rendered_image_free (1 line)

      # Outline functions
      # REMOVED phantom (no upstream symbol): :pdf_document_get_outline_dest_page (1 line)

      # List helper functions
      # REMOVED phantom (no upstream symbol): :pdf_oxide_column_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_column_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_column_list_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_embedded_file_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_embedded_file_get_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_embedded_file_get_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_embedded_file_list_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_get_family (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_usage_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_usage_get_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_usage_get_page_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_usage_is_embedded (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_font_usage_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_image_get_name (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_image_get_color_space (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_link_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_link_get_url (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_link_get_bbox (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_link_list_free (1 line)
      attach_function :pdf_oxide_table_count, [:pointer], :int32
      # REMOVED phantom (no upstream symbol): :pdf_oxide_table_get_bbox (1 line)
      # Table accessors: (tables, index, int32_t *error_code) → int32.
      attach_function :pdf_oxide_table_get_row_count, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_table_get_col_count, %i[pointer int32 pointer], :int32
      attach_function :pdf_oxide_table_list_free, [:pointer], :void
      # REMOVED phantom (no upstream symbol): :pdf_oxide_ocr_result_get_text (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_ocr_result_confidence (1 line)

      # Certificate functions
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_serial_number (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_valid_from (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_valid_to (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_certificate_get_thumbprint (1 line)

      # PDF/X compliance functions
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_warning_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_pdf_x_get_warning (1 line)

      # PDF/UA compliance functions
      # REMOVED duplicate declaration: :pdf_pdf_ua_error_count (1 line)
      # REMOVED duplicate declaration: :pdf_pdf_ua_is_accessible (1 line)

      # Modification date function
      # REMOVED phantom (no upstream symbol): :pdf_document_get_modification_date (1 line)

      # Extract pages function
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_pages (1 line)

      # ============================================================
      # FINAL REMAINING MANAGER SUPPORT FUNCTIONS
      # ============================================================

      # Annotation modification functions
      # REMOVED phantom (no upstream symbol): :pdf_document_add_highlight (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_add_underline (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_add_strikeout (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_add_text_annotation (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_delete_annotation (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_flatten_annotations (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_annotation_count (1 line)

      # Barcode functions
      # REMOVED phantom (no upstream symbol): :pdf_document_add_qr_code (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_add_barcode (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_extract_barcodes (1 line)

      # Signature functions
      # REMOVED phantom (no upstream symbol): :pdf_document_add_signature (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_verify_signature (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_signature_signer (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_signature_timestamp (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_signature_status (1 line)

      # Form field functions
      # REMOVED phantom (no upstream symbol): :pdf_document_flatten_forms (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_form_field_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_form_field_value (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_form_field_flags (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_set_form_field_value (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_reset_form_fields (1 line)

      # Document utility functions
      # REMOVED phantom (no upstream symbol): :pdf_document_get_file_size (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_get_metadata (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_unlock_with_password (1 line)

      # XFA function alias
      # REMOVED phantom (no upstream symbol): :pdf_document_has_xfa_form (1 line)

      # Annotation list helper functions
      # REMOVED phantom (no upstream symbol): :pdf_oxide_annotation_get_text (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_annotation_get_bbox (1 line)
      # uint32_t pdf_oxide_annotation_get_color(const FfiAnnotationList *annotations, int32_t index, int32_t *error_code)
      # Pre-v0.3.55 missed the trailing err pointer AND had :int32 return
      # (C returns uint32_t — packed ARGB color).
      attach_function :pdf_oxide_annotation_get_color, %i[pointer int32 pointer], :uint32
      attach_function :pdf_oxide_annotation_list_free, [:pointer], :void

      # Form field list helper functions
      attach_function :pdf_oxide_form_field_count, [:pointer], :int32
      # char *pdf_oxide_form_field_get_name(const FfiFormFieldList *fields, int32_t index, int32_t *error_code)
      attach_function :pdf_oxide_form_field_get_name, %i[pointer int32 pointer], :pointer
      attach_function :pdf_oxide_form_field_list_free, [:pointer], :void

      # Signature helper functions
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_get_signer (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_get_timestamp (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_get_status (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_get_reason (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_signature_get_location (1 line)

      # Verification helper functions
      # REMOVED phantom (no upstream symbol): :pdf_oxide_verification_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_verification_is_valid (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_verification_is_trusted (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_verification_is_self_signed (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_oxide_verification_get_error (1 line)

      # ============================================================
      # REDACTION OPERATIONS
      # ============================================================

      # Add a redaction annotation to a page
      attach_function :pdf_redaction_add,
                      %i[pointer int32 float float float float uint8 uint8 uint8 pointer],
                      :bool

      # Apply all pending redactions
      attach_function :pdf_redaction_apply,
                      %i[pointer bool uint8 uint8 uint8 pointer],
                      :bool

      # Scrub document metadata
      attach_function :pdf_redaction_scrub_metadata,
                      %i[pointer bool bool bool pointer],
                      :bool

      # Get count of pending redactions
      attach_function :pdf_redaction_count, %i[pointer pointer], :int32

      # ============================================================
      # FLATTENING OPERATIONS
      # ============================================================

      # Flatten all form fields
      # REMOVED phantom (no upstream symbol): :pdf_document_editor_flatten_forms (1 line)

      # Flatten form fields on a specific page
      # REMOVED phantom (no upstream symbol): :pdf_document_editor_flatten_forms_page (1 line)

      # Flatten all annotations
      # REMOVED phantom (no upstream symbol): :pdf_document_editor_flatten_annotations (1 line)

      # Flatten annotations on a specific page
      # REMOVED phantom (no upstream symbol): :pdf_document_editor_flatten_annotations_page (1 line)

      # ============================================================
      # COMPLIANCE OPERATIONS
      # ============================================================

      # Convert document to PDF/A
      # REMOVED duplicate declaration: :pdf_convert_to_pdf_a (1 line)

      # Validate document against PDF/A
      # REMOVED phantom (no upstream symbol): :pdf_validate_pdfa (1 line)

      # ============================================================
      # ACCESSIBILITY OPERATIONS
      # ============================================================

      # Check if document is tagged
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_is_tagged (1 line)

      # Get the document structure tree
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_get_structure_tree (1 line)

      # Automatically tag the document
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_auto_tag (1 line)

      # Set alt text on a structure element
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_set_alt_text (1 line)

      # Set the document language
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_set_language (1 line)

      # Set the document title for accessibility
      # REMOVED phantom (no upstream symbol): :pdf_accessibility_set_title (1 line)

      # Free a structure tree handle
      # REMOVED phantom (no upstream symbol): :pdf_structure_tree_free (1 line)

      # Free a structure element handle
      # REMOVED phantom (no upstream symbol): :pdf_struct_elem_free (1 line)

      # ============================================================
      # OPTIMIZATION OPERATIONS
      # ============================================================

      # Open document with mmap
      # REMOVED phantom (no upstream symbol): :pdf_document_open_mmap (1 line)

      # Subset fonts to remove unused glyphs
      # REMOVED phantom (no upstream symbol): :pdf_optimize_subset_fonts (1 line)

      # Downsample images
      # REMOVED phantom (no upstream symbol): :pdf_optimize_downsample_images (1 line)

      # Deduplicate content streams
      # REMOVED phantom (no upstream symbol): :pdf_optimize_deduplicate (1 line)

      # Run full optimization pipeline
      # REMOVED phantom (no upstream symbol): :pdf_optimize_full (1 line)

      # Get bytes saved from optimization result
      # REMOVED phantom (no upstream symbol): :pdf_optimization_result_bytes_saved (1 line)

      # Free optimization result handle
      # REMOVED phantom (no upstream symbol): :pdf_optimization_result_free (1 line)

      # ============================================================
      # ENTERPRISE OPERATIONS
      # ============================================================

      # Bates numbering
      # REMOVED phantom (no upstream symbol): :pdf_bates_apply (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_bates_apply_advanced (7 lines)
      #
      #
      #
      #
      #
      #

      # Document comparison
      # REMOVED phantom (no upstream symbol): :pdf_compare_pages (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_comparison_get_similarity (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_comparison_get_diff_count (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_comparison_get_diff (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_comparison_get_diff_type (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_compare_documents (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_comparison_free (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_document_comparison_free (1 line)

      # Header/footer stamping
      # REMOVED phantom (no upstream symbol): :pdf_stamp_header (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_stamp_footer (1 line)
      # REMOVED phantom (no upstream symbol): :pdf_stamp_header_footer (1 line)

      # ============================================================
      # TSA TIMESTAMP OPERATIONS (17 functions)
      # ============================================================

      attach_function :pdf_tsa_client_create, %i[string string string int32 int32 bool bool pointer], :pointer
      attach_function :pdf_tsa_client_free, [:pointer], :void
      attach_function :pdf_tsa_request_timestamp, %i[pointer pointer size_t pointer], :pointer
      attach_function :pdf_tsa_request_timestamp_hash, %i[pointer pointer size_t int32 pointer], :pointer
      attach_function :pdf_timestamp_get_token, %i[pointer pointer pointer], :pointer
      attach_function :pdf_timestamp_get_time, %i[pointer pointer], :int64
      # Timestamp accessors: (ts, int32_t *error_code) returning owned char*.
      attach_function :pdf_timestamp_get_serial, %i[pointer pointer], :pointer
      attach_function :pdf_timestamp_get_tsa_name, %i[pointer pointer], :pointer
      attach_function :pdf_timestamp_get_policy_oid, %i[pointer pointer], :pointer
      attach_function :pdf_timestamp_get_hash_algorithm, %i[pointer pointer], :int32
      attach_function :pdf_timestamp_get_message_imprint, %i[pointer pointer pointer], :pointer
      attach_function :pdf_timestamp_verify, %i[pointer pointer], :bool
      attach_function :pdf_timestamp_free, [:pointer], :void
      attach_function :pdf_signature_add_timestamp, %i[pointer pointer pointer], :bool
      # REMOVED duplicate declaration: :pdf_signature_has_timestamp (1 line)
      attach_function :pdf_signature_get_timestamp, %i[pointer pointer], :pointer

      # ============================================================
      # PDF/UA EXTENDED VALIDATION (3 functions)
      # ============================================================

      attach_function :pdf_pdf_ua_warning_count, [:pointer], :int32
      attach_function :pdf_pdf_ua_get_warning, %i[pointer int32 pointer], :pointer
      attach_function :pdf_pdf_ua_get_stats, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :bool

      # ============================================================
      # FDF/XFDF IN-MEMORY IMPORT/EXPORT (4 functions)
      # ============================================================

      attach_function :pdf_editor_import_fdf_bytes, %i[pointer pointer size_t pointer], :int32
      attach_function :pdf_editor_import_xfdf_bytes, %i[pointer pointer size_t pointer], :int32
      attach_function :pdf_document_import_form_data, %i[pointer string pointer], :int32
      attach_function :pdf_document_export_form_data_to_bytes, %i[pointer int32 pointer pointer], :pointer

      # ============================================================
      # TOTAL: 600+ FUNCTIONS DECLARED (100% API Coverage)
      # ============================================================
      # All core FFI functions now available for Ruby binding

      # ============================================================
      # AUTO-REPAIR Phase 2: cdylib symbols not declared by the prepared
      # snapshot.  Generic signature so the gem loads; real wrappers must
      # be added by Phase 3 (extend) and Phase 4 (test/CI).
      # ============================================================

      attach_function :AllocString, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :DocumentEditorFree, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :DocumentEditorOpen, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :DocumentEditorSave, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :DocumentEditorSetAuthor, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :DocumentEditorSetTitle, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :FreeBytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :FreeString, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentExtractText, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentFree, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentGetPageCount, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentOpen, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentToHtml, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentToMarkdown, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfDocumentToPlainText, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfFree, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfFromHtml, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfFromMarkdown, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfFromText, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfSave, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :PdfSaveToBytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_apply_all_redactions, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_apply_page_redactions, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_clear_erase_regions, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_convert_to_pdf_a, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_crop_margins, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_delete_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_embed_file, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_erase_region, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_erase_regions, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_extract_pages_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_all_annotations, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_annotations, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_forms, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_forms_on_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_warning, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_flatten_warnings_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_creation_date, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_keywords, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_page_crop_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_page_media_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_page_rotation, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_get_producer, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_is_page_marked_for_flatten, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_is_page_marked_for_redaction, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_merge_from, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_merge_from_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_move_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_open_from_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_rotate_all_pages, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_rotate_page_by, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_save_encrypted, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_save_encrypted_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_save_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_save_to_bytes_with_options, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_creation_date, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_form_field_value, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_keywords, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_page_crop_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_page_media_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_page_rotation, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_set_producer, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_unmark_page_for_flatten, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :document_editor_unmark_page_for_redaction, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :lut_interp_linear16, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :lut_inverse_interp16, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_create_from_markdown, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_format, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_open, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_open_from_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_plain_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_save_as, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_to_html, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_to_ir_json, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_document_to_markdown, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_open, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_open_from_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_replace_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_save, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_save_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_editable_set_cell, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_extract_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_oxide_detect_format, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_oxide_free_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_oxide_free_string, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_oxide_version, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_slide_add_image, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_slide_add_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_slide_set_title, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_add_slide, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_new, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_save, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_set_presentation_size, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_pptx_writer_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_to_html, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_to_markdown, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_sheet_merge_cells, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_sheet_set_cell, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_sheet_set_cell_styled, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_sheet_set_column_width, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_writer_add_sheet, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_writer_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_writer_new, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_writer_save, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :office_xlsx_writer_to_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_certificate_load_from_pem, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_create_renderer, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_authenticate, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_a4_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_build, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_create, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_language, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_letter_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_on_open, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_register_embedded_font, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_role_map, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_save, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_save_encrypted, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_set_author, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_set_creator, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_set_keywords, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_set_subject, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_set_title, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_tagged_pdf_ua1, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_builder_to_bytes_encrypted, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_classify_document, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_classify_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_erase_artifacts, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_erase_footer, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_erase_header, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_all_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_chars, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_images_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_lines_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_page_auto, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_paths, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_tables, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_tables_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_text_auto, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_text_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_text_lines, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_words, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_extract_words_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_dss, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # FfiFormFieldList *pdf_document_get_form_fields(PdfDocument *handle, int32_t *error_code)
      attach_function :pdf_document_get_form_fields, %i[pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_outline, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_page_annotations, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_page_labels, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_source_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_get_xmp_metadata, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_has_timestamp, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # PdfDocument *pdf_document_open_from_bytes(const uint8_t *data, uintptr_t len, int32_t *error_code)
      attach_function :pdf_document_open_from_bytes, %i[pointer size_t pointer], :pointer, blocking: false
      attach_function :pdf_document_open_from_docx_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_open_from_pptx_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_open_from_xlsx_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_open_with_password, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_plan_split_by_bookmarks, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_remove_artifacts, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_remove_footers, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_remove_headers, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_to_docx, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # Whole-document text conversions: char *fn(PdfDocument*, int32_t*).
      # Returned char* is owned (free with free_string via StringMarshaller).
      attach_function :pdf_document_to_html_all, %i[pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_to_plain_text_all, %i[pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_to_pptx, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_document_to_xlsx, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # int32_t pdf_document_verify_all_signatures(const void *handle, int32_t *error_code)
      attach_function :pdf_document_verify_all_signatures, %i[pointer pointer], :int32, blocking: false
      attach_function :pdf_dss_cert_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_crl_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_get_cert, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_get_crl, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_get_ocsp, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_ocsp_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_dss_vri_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_embedded_font_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_embedded_font_from_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_embedded_font_from_file, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_from_html_css, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_from_html_css_with_fonts, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_from_image, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_from_image_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_get_rendered_image_data, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_get_rendered_image_height, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_get_rendered_image_width, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_merge, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_get_border_width, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_get_content, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_get_modification_date, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_get_subtype, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_is_hidden, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_is_marked_deleted, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_is_printable, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotation_is_read_only, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_annotations_to_json, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_get_bbox, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_get_char, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_get_font_name, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_get_font_size, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_char_list_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_active_provider, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_cbom, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_fips_available, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_inventory, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_policy, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_set_policy, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_crypto_use_fips, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_element_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_element_get_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_element_get_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_element_get_type, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_elements_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_elements_to_json, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_fonts_to_json, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_form_field_get_type, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_form_field_get_value, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_form_field_is_readonly, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_form_field_is_required, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_get_log_level, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_highlight_annotation_get_quad_point, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_highlight_annotation_get_quad_points_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_line_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_line_get_bbox, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_line_get_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_line_get_word_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_line_list_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_link_annotation_get_uri, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_model_manifest, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_get_bbox, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_get_operation_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_get_stroke_width, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_has_fill, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_has_stroke, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_path_list_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_prefetch_available, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_prefetch_models, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_search_results_to_json, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_set_log_level, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_table_get_cell_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_table_has_header, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_text_annotation_get_icon_name, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_get_bbox, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_get_font_name, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_get_font_size, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_get_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_is_bold, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_oxide_word_list_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_at, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_barcode_1d, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_barcode_qr, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_checkbox, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_columns, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_combo_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_done, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_field_calculate, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_field_format, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_field_keystroke, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_field_validate, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # int32_t pdf_page_builder_filled_rect(FfiPageBuilder *handle, float x, float y, float w, float h, float r, float g, float b, int32_t *error_code)
      attach_function :pdf_page_builder_filled_rect,
                      %i[pointer float float float float float float float pointer], :int32,
                      blocking: false
      attach_function :pdf_page_builder_font, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_footnote, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_freetext, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_heading, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_highlight, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_horizontal_rule, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_image, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_image_artifact, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # int32_t pdf_page_builder_image_with_alt(FfiPageBuilder *handle, const uint8_t *bytes, uintptr_t len, float x, float y, float w, float h, const char *alt_text, int32_t *error_code)
      attach_function :pdf_page_builder_image_with_alt,
                      %i[pointer pointer size_t float float float float string pointer], :int32,
                      blocking: false
      attach_function :pdf_page_builder_inline, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_inline_bold, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_inline_color, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_inline_italic, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_line, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_link_javascript, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_link_named, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_link_page, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_link_url, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_new_page_same_size, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_newline, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_on_close, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_on_open, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_paragraph, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_push_button, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_radio_group, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_signature_field, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_space, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_squiggly, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_stamp, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_sticky_note, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_sticky_note_at, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_batch_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_begin, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_begin_v2, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_finish, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_flush, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_pending_row_count, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_push_row, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_push_row_v2, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_streaming_table_set_batch_size, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_strikeout, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_stroke_line, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_stroke_line_dashed, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_stroke_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_stroke_rect_dashed, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_table, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_text, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_text_field, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_text_in_rect, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_underline, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_watermark, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_watermark_confidential, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_builder_watermark_draft, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_art_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_bleed_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_crop_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_elements, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_media_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_rotation, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_page_get_trim_box, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_render_page_raw, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_render_page_with_options, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_renderer_free, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_save_rendered_image, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_sign_bytes, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_sign_bytes_pades, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_sign_bytes_pades_opts, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_get_pades_level, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_get_signer_name, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_get_signing_location, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_get_signing_reason, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_get_signing_time, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_signature_verify_detached, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :pdf_timestamp_parse, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # FfiPdfAResults *pdf_validate_pdf_a_level(PdfDocument *document, int32_t level, int32_t *error_code)
      attach_function :pdf_validate_pdf_a_level, %i[pointer int32 pointer], :pointer, blocking: false
      # FfiPdfXResults *pdf_validate_pdf_x_level(PdfDocument *document, int32_t level, int32_t *error_code)
      attach_function :pdf_validate_pdf_x_level, %i[pointer int32 pointer], :pointer, blocking: false
      attach_function :qcms_enable_iccv4, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_profile_is_bogus, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_profile_precache_output_transform, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_bgra_out_lut, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_bgra_out_lut_precache, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_rgb_out_lut, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_rgb_out_lut_precache, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_rgba_out_lut, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_transform_data_rgba_out_lut_precache, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      # The `_avx` and `_sse2` qcms transforms are x86-only intrinsics
      # builds; they are absent on aarch64-{darwin,linux} cdylibs.
      # qcms internally selects the right impl, so the Ruby side never
      # calls these directly — wrap in a tolerant attach so the gem
      # loads on ARM.  (Public C ABI never exposes these; they leak
      # from the qcms crate's `#[no_mangle]` symbols.)
      %i[qcms_transform_data_bgra_out_lut_avx qcms_transform_data_bgra_out_lut_sse2
         qcms_transform_data_rgb_out_lut_avx qcms_transform_data_rgb_out_lut_sse2
         qcms_transform_data_rgba_out_lut_avx qcms_transform_data_rgba_out_lut_sse2].each do |sym|
        attach_function sym, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      rescue ::FFI::NotFoundError
        # Symbol absent on this arch — never invoked from Ruby, skip silently.
      end
      attach_function :qcms_transform_release, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false
      attach_function :qcms_white_point_sRGB, %i[pointer pointer pointer pointer pointer pointer pointer pointer], :pointer, blocking: false

      # ============================================================
      # PHASE 2 REPAIR: real signatures for symbols the Ruby wrappers
      # actively call.  FFI's `attach_function` lets a later declaration
      # override an earlier generic skeleton with the correct signature.
      # ============================================================

      # PDF creation entry points (replace Creator stub) - returns Pdf*.
      attach_function :pdf_from_markdown, %i[string pointer], :pointer
      attach_function :pdf_from_html, %i[string pointer], :pointer
      attach_function :pdf_from_text, %i[string pointer], :pointer
      attach_function :pdf_from_image, %i[string pointer], :pointer
      attach_function :pdf_from_image_bytes, %i[pointer size_t pointer], :pointer

      # PDF handle save / inspect / free.
      attach_function :pdf_save, %i[pointer string pointer], :int32
      attach_function :pdf_save_to_bytes, %i[pointer pointer pointer], :pointer
      attach_function :pdf_get_page_count, %i[pointer pointer], :int32

      # Free helpers — kept explicit so StringMarshaller.free_c_string
      # resolves to a real ABI signature.
      attach_function :pdf_free, [:pointer], :void
      attach_function :free_bytes, [:pointer], :void

      # ============================================================
      # PHASE 3 EXTEND: real signatures for v0.3.50-v0.3.54 features.
      # Each entry below replaces the placeholder 8-pointer skeleton
      # earlier in the file (FFI permits later attach_function calls
      # to override the prior declaration).
      # ============================================================

      # Auto-extraction (#519, v0.3.51) — JSON-returning classifiers
      # plus the text-only auto-router.  All return malloc'd char*;
      # free with pdf_free / free_string.
      attach_function :pdf_document_classify_page,     %i[pointer int32 pointer], :pointer
      attach_function :pdf_document_classify_document, %i[pointer pointer], :pointer
      attach_function :pdf_document_extract_text_auto, %i[pointer int32 pointer], :pointer
      attach_function :pdf_document_extract_page_auto, %i[pointer int32 string pointer], :pointer

      # Models subsystem (#519 provisioning trio).
      attach_function :pdf_oxide_prefetch_models,    %i[string pointer], :pointer
      attach_function :pdf_oxide_model_manifest,     [],                  :pointer
      attach_function :pdf_oxide_prefetch_available, [],                  :int32

      # Office converter (#159, v0.3.48). All three return a PdfDocument*.
      attach_function :pdf_document_open_from_docx_bytes, %i[pointer size_t pointer], :pointer
      attach_function :pdf_document_open_from_pptx_bytes, %i[pointer size_t pointer], :pointer
      attach_function :pdf_document_open_from_xlsx_bytes, %i[pointer size_t pointer], :pointer

      # Split-by-bookmarks plan (v0.3.50).  Returns a JSON plan as
      # char*; the consumer interprets the segment list and feeds
      # each {start_page, end_page} pair to extract-page utilities.
      attach_function :pdf_document_plan_split_by_bookmarks,
                      %i[pointer string pointer], :pointer

      # Destructive redaction (#231, v0.3.50).  Operates on a
      # DocumentEditor* handle (NOT a PdfDocument*).
      attach_function :pdf_redaction_add,
                      %i[pointer size_t
                         double double double double
                         double double double
                         pointer],
                      :int32
      attach_function :pdf_redaction_count, %i[pointer size_t pointer], :int32
      attach_function :pdf_redaction_apply,
                      %i[pointer bool double double double pointer], :int32
      attach_function :pdf_redaction_scrub_metadata, %i[pointer pointer], :int32

      # PAdES signing — the 5-arg shim (v0.3.51 #517 follow-up to
      # v0.3.50 #235). The 18-arg legacy entry is still available
      # under pdf_sign_bytes_pades but the shim is canonical for all
      # bindings (purego cannot register the legacy form).
      attach_function :pdf_sign_bytes_pades_opts,
                      %i[pointer size_t pointer pointer pointer], :pointer

      # PAdES level inspection.
      attach_function :pdf_signature_get_pades_level, %i[pointer pointer], :int32
      attach_function :pdf_document_has_timestamp,    %i[pointer pointer], :int32

      # PAdES level enum codes (frozen).  These are the int32 values
      # `pdf_signature_get_pades_level` returns and the `level`
      # field of `PadesSignOptionsC` takes.  Keep the names mirrored
      # against the Rust `PadesLevel` enum.
      PADES_LEVEL_B   = 0
      PADES_LEVEL_T   = 1
      PADES_LEVEL_LT  = 2
      PADES_LEVEL_LTA = 3

      # DocumentEditor lifecycle — needed by RedactionManager so it
      # can apply redactions destructively to an editor handle and
      # save the resulting bytes.  The existing skeletons use generic
      # 8-pointer signatures; these declarations refine them.
      attach_function :document_editor_open, %i[string pointer], :pointer
      attach_function :document_editor_open_from_bytes,
                      %i[pointer size_t pointer], :pointer
      attach_function :document_editor_free,           [:pointer], :void
      attach_function :document_editor_save,           %i[pointer string pointer], :int32
      attach_function :document_editor_save_to_bytes,
                      %i[pointer pointer pointer], :pointer
      attach_function :document_editor_apply_page_redactions,
                      %i[pointer size_t pointer], :int32
      attach_function :document_editor_apply_all_redactions,
                      %i[pointer pointer], :int32

      # ============================================================
      # PHASE 4 EXTEND: global tuning toggles + layer-filtered render.
      # Real ABI signatures from include/pdf_oxide_c/pdf_oxide.h.
      # ============================================================

      # int64_t pdf_oxide_set_max_ops_per_stream(int64_t limit)
      # Process-global content-stream operator cap. `limit < 0` restores
      # the default (1,000,000); returns the previous cap (or -1 if the
      # default was active). No error channel.
      attach_function :pdf_oxide_set_max_ops_per_stream, [:int64], :int64

      # int32_t pdf_oxide_set_preserve_unmapped_glyphs(int32_t preserve)
      # Toggle the global U+FFFD preservation flag. `1` = preserve,
      # `0` = filter. Returns the previous value (`0` or `1`). No error
      # channel.
      attach_function :pdf_oxide_set_preserve_unmapped_glyphs, [:int32], :int32

      # FfiRenderedImage *pdf_render_page_with_options_ex(
      #   PdfDocument *doc, int32_t page_index, int32_t dpi, int32_t format,
      #   float bg_r, float bg_g, float bg_b, float bg_a,
      #   int32_t transparent_background, int32_t render_annotations,
      #   int32_t jpeg_quality,
      #   const char *const *excluded_layers, uintptr_t excluded_layers_count,
      #   int32_t *error_code)
      # Like pdf_render_page_with_options but with OCG layer filtering.
      attach_function :pdf_render_page_with_options_ex,
                      %i[pointer int32 int32 int32
                         float float float float
                         int32 int32 int32
                         pointer size_t
                         pointer],
                      :pointer
    end
  end
end
