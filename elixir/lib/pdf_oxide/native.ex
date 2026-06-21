defmodule PdfOxide.Native do
  @moduledoc false
  # NIF loader. The compiled NIF (priv/pdf_oxide_nif) bridges to the pdf_oxide
  # C ABI; every text-producing function runs on a dirty CPU scheduler. All
  # functions here are replaced at load time — the bodies raise if the NIF
  # failed to load.
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:pdf_oxide), ~c"pdf_oxide_nif")
    :erlang.load_nif(path, 0)
  end

  defp nif_error, do: :erlang.nif_error(:nif_not_loaded)

  def from_markdown(_md), do: nif_error()
  def from_html(_html), do: nif_error()
  def from_text(_text), do: nif_error()
  def pdf_save(_pdf, _path), do: nif_error()
  def pdf_save_to_bytes(_pdf), do: nif_error()
  def doc_open(_path), do: nif_error()
  def doc_open_bytes(_bytes), do: nif_error()
  def doc_open_pw(_path, _pw), do: nif_error()
  def doc_page_count(_doc), do: nif_error()
  def doc_version(_doc), do: nif_error()
  def doc_is_encrypted(_doc), do: nif_error()
  def doc_has_structure_tree(_doc), do: nif_error()
  def doc_extract_text(_doc, _page), do: nif_error()
  def doc_to_plain_text(_doc, _page), do: nif_error()
  def doc_to_markdown(_doc, _page), do: nif_error()
  def doc_to_html(_doc, _page), do: nif_error()
  def doc_to_markdown_all(_doc), do: nif_error()
  def doc_to_html_all(_doc), do: nif_error()
  def doc_to_plain_text_all(_doc), do: nif_error()
  def doc_authenticate(_doc, _pw), do: nif_error()
  def doc_extract_structured_json(_doc, _page), do: nif_error()
  def doc_extract_chars(_doc, _page), do: nif_error()
  def doc_extract_words(_doc, _page), do: nif_error()
  def doc_extract_text_lines(_doc, _page), do: nif_error()
  def doc_extract_tables(_doc, _page), do: nif_error()
  def doc_embedded_fonts(_doc, _page), do: nif_error()
  def doc_embedded_images(_doc, _page), do: nif_error()
  def doc_page_annotations(_doc, _page), do: nif_error()
  def doc_extract_paths(_doc, _page), do: nif_error()
  def doc_search_page(_doc, _page, _term, _case_sensitive), do: nif_error()
  def doc_search_all(_doc, _term, _case_sensitive), do: nif_error()
  def doc_render_page(_doc, _page_index, _format), do: nif_error()
  def doc_render_page_zoom(_doc, _page_index, _zoom, _format), do: nif_error()
  def doc_render_page_thumbnail(_doc, _page_index, _size, _format), do: nif_error()
  def img_save(_img, _path), do: nif_error()
  def doc_close(_doc), do: nif_error()
  def pdf_close(_pdf), do: nif_error()

  # document editor
  def editor_open(_path), do: nif_error()
  def editor_open_bytes(_bytes), do: nif_error()
  def editor_is_modified(_ed), do: nif_error()
  def editor_source_path(_ed), do: nif_error()
  def editor_version(_ed), do: nif_error()
  def editor_page_count(_ed), do: nif_error()
  def editor_get_producer(_ed), do: nif_error()
  def editor_set_producer(_ed, _value), do: nif_error()
  def editor_get_creation_date(_ed), do: nif_error()
  def editor_set_creation_date(_ed, _date), do: nif_error()
  def editor_save(_ed, _path), do: nif_error()
  def editor_save_to_bytes(_ed), do: nif_error()
  def editor_save_to_bytes_with_options(_ed, _compress, _gc, _linearize), do: nif_error()
  def editor_extract_pages_to_bytes(_ed, _pages), do: nif_error()
  def editor_convert_to_pdf_a(_ed, _level), do: nif_error()
  def editor_save_encrypted_to_bytes(_ed, _user_pw, _owner_pw), do: nif_error()
  def editor_merge_from_bytes(_ed, _bytes), do: nif_error()
  def editor_embed_file(_ed, _name, _bytes), do: nif_error()
  def editor_apply_page_redactions(_ed, _page), do: nif_error()
  def editor_apply_all_redactions(_ed), do: nif_error()
  def editor_rotate_all_pages(_ed, _degrees), do: nif_error()
  def editor_rotate_page_by(_ed, _page, _degrees), do: nif_error()
  def editor_get_media_box(_ed, _page), do: nif_error()
  def editor_set_media_box(_ed, _page, _x, _y, _w, _h), do: nif_error()
  def editor_get_crop_box(_ed, _page), do: nif_error()
  def editor_set_crop_box(_ed, _page, _x, _y, _w, _h), do: nif_error()
  def editor_erase_regions(_ed, _page, _rects), do: nif_error()
  def editor_clear_erase_regions(_ed, _page), do: nif_error()
  def editor_is_marked_for_flatten(_ed, _page), do: nif_error()
  def editor_unmark_for_flatten(_ed, _page), do: nif_error()
  def editor_is_marked_for_redaction(_ed, _page), do: nif_error()
  def editor_unmark_for_redaction(_ed, _page), do: nif_error()
  def editor_delete_page(_ed, _page_index), do: nif_error()
  def editor_move_page(_ed, _from, _to), do: nif_error()
  def editor_get_page_rotation(_ed, _page), do: nif_error()
  def editor_set_page_rotation(_ed, _page, _degrees), do: nif_error()
  def editor_erase_region(_ed, _page, _x, _y, _w, _h), do: nif_error()
  def editor_flatten_annotations(_ed, _page), do: nif_error()
  def editor_flatten_all_annotations(_ed), do: nif_error()
  def editor_crop_margins(_ed, _left, _right, _top, _bottom), do: nif_error()
  def editor_merge_from(_ed, _source_path), do: nif_error()
  def editor_save_encrypted(_ed, _path, _user_pw, _owner_pw), do: nif_error()
  def editor_set_form_field_value(_ed, _name, _value), do: nif_error()
  def editor_flatten_forms(_ed), do: nif_error()
  def editor_flatten_forms_on_page(_ed, _page_index), do: nif_error()
  def editor_flatten_warnings_count(_ed), do: nif_error()
  def editor_flatten_warning(_ed, _index), do: nif_error()
  def editor_close(_ed), do: nif_error()

  # PDF creation builder — embedded font
  def font_from_file(_path), do: nif_error()
  def font_from_bytes(_bytes, _name), do: nif_error()
  def font_close(_font), do: nif_error()

  # PDF creation builder — document builder
  def dbld_create, do: nif_error()
  def dbld_set_title(_db, _title), do: nif_error()
  def dbld_set_author(_db, _author), do: nif_error()
  def dbld_set_subject(_db, _subject), do: nif_error()
  def dbld_set_keywords(_db, _keywords), do: nif_error()
  def dbld_set_creator(_db, _creator), do: nif_error()
  def dbld_on_open(_db, _script), do: nif_error()
  def dbld_language(_db, _lang), do: nif_error()
  def dbld_tagged_pdf_ua1(_db), do: nif_error()
  def dbld_role_map(_db, _custom, _standard), do: nif_error()
  def dbld_register_embedded_font(_db, _name, _font), do: nif_error()
  def dbld_a4_page(_db), do: nif_error()
  def dbld_letter_page(_db), do: nif_error()
  def dbld_page(_db, _width, _height), do: nif_error()
  def dbld_build(_db), do: nif_error()
  def dbld_save(_db, _path), do: nif_error()
  def dbld_save_encrypted(_db, _path, _user_pw, _owner_pw), do: nif_error()
  def dbld_to_bytes_encrypted(_db, _user_pw, _owner_pw), do: nif_error()
  def dbld_close(_db), do: nif_error()

  # PDF creation builder — page builder
  def pbld_font(_pb, _name, _size), do: nif_error()
  def pbld_at(_pb, _x, _y), do: nif_error()
  def pbld_text(_pb, _text), do: nif_error()
  def pbld_heading(_pb, _level, _text), do: nif_error()
  def pbld_paragraph(_pb, _text), do: nif_error()
  def pbld_space(_pb, _points), do: nif_error()
  def pbld_horizontal_rule(_pb), do: nif_error()
  def pbld_link_url(_pb, _url), do: nif_error()
  def pbld_link_page(_pb, _page_index), do: nif_error()
  def pbld_link_named(_pb, _destination), do: nif_error()
  def pbld_link_javascript(_pb, _script), do: nif_error()
  def pbld_on_open(_pb, _script), do: nif_error()
  def pbld_on_close(_pb, _script), do: nif_error()
  def pbld_field_keystroke(_pb, _script), do: nif_error()
  def pbld_field_format(_pb, _script), do: nif_error()
  def pbld_field_validate(_pb, _script), do: nif_error()
  def pbld_field_calculate(_pb, _script), do: nif_error()
  def pbld_highlight(_pb, _r, _g, _b), do: nif_error()
  def pbld_underline(_pb, _r, _g, _b), do: nif_error()
  def pbld_strikeout(_pb, _r, _g, _b), do: nif_error()
  def pbld_squiggly(_pb, _r, _g, _b), do: nif_error()
  def pbld_sticky_note(_pb, _text), do: nif_error()
  def pbld_sticky_note_at(_pb, _x, _y, _text), do: nif_error()
  def pbld_watermark(_pb, _text), do: nif_error()
  def pbld_watermark_confidential(_pb), do: nif_error()
  def pbld_watermark_draft(_pb), do: nif_error()
  def pbld_stamp(_pb, _type_name), do: nif_error()
  def pbld_freetext(_pb, _x, _y, _w, _h, _text), do: nif_error()
  def pbld_text_field(_pb, _name, _x, _y, _w, _h, _default_value), do: nif_error()
  def pbld_checkbox(_pb, _name, _x, _y, _w, _h, _checked), do: nif_error()
  def pbld_combo_box(_pb, _name, _x, _y, _w, _h, _options, _selected), do: nif_error()
  def pbld_radio_group(_pb, _name, _values, _xs, _ys, _ws, _hs, _selected), do: nif_error()
  def pbld_push_button(_pb, _name, _x, _y, _w, _h, _caption), do: nif_error()
  def pbld_signature_field(_pb, _name, _x, _y, _w, _h), do: nif_error()
  def pbld_footnote(_pb, _ref_mark, _note_text), do: nif_error()
  def pbld_columns(_pb, _column_count, _gap_pt, _text), do: nif_error()
  def pbld_inline(_pb, _text), do: nif_error()
  def pbld_inline_bold(_pb, _text), do: nif_error()
  def pbld_inline_italic(_pb, _text), do: nif_error()
  def pbld_inline_color(_pb, _r, _g, _b, _text), do: nif_error()
  def pbld_newline(_pb), do: nif_error()
  def pbld_barcode_1d(_pb, _barcode_type, _data, _x, _y, _w, _h), do: nif_error()
  def pbld_barcode_qr(_pb, _data, _x, _y, _size), do: nif_error()
  def pbld_image(_pb, _bytes, _x, _y, _w, _h), do: nif_error()
  def pbld_image_with_alt(_pb, _bytes, _x, _y, _w, _h, _alt_text), do: nif_error()
  def pbld_image_artifact(_pb, _bytes, _x, _y, _w, _h), do: nif_error()
  def pbld_rect(_pb, _x, _y, _w, _h), do: nif_error()
  def pbld_filled_rect(_pb, _x, _y, _w, _h, _r, _g, _b), do: nif_error()
  def pbld_line(_pb, _x1, _y1, _x2, _y2), do: nif_error()
  def pbld_stroke_rect(_pb, _x, _y, _w, _h, _width, _r, _g, _b), do: nif_error()
  def pbld_stroke_line(_pb, _x1, _y1, _x2, _y2, _width, _r, _g, _b), do: nif_error()

  def pbld_stroke_rect_dashed(_pb, _x, _y, _w, _h, _width, _r, _g, _b, _dash, _phase),
    do: nif_error()

  def pbld_stroke_line_dashed(_pb, _x1, _y1, _x2, _y2, _width, _r, _g, _b, _dash, _phase),
    do: nif_error()

  def pbld_text_in_rect(_pb, _x, _y, _w, _h, _text, _align), do: nif_error()
  def pbld_new_page_same_size(_pb), do: nif_error()

  def pbld_table(_pb, _n_columns, _widths, _aligns, _n_rows, _cell_strings, _has_header),
    do: nif_error()

  def pbld_streaming_table_begin(_pb, _n_columns, _headers, _widths, _aligns, _repeat_header),
    do: nif_error()

  def pbld_streaming_table_begin_v2(
        _pb,
        _n_columns,
        _headers,
        _widths,
        _aligns,
        _repeat_header,
        _mode,
        _sample_rows,
        _min_w,
        _max_w,
        _max_rowspan
      ),
      do: nif_error()

  def pbld_streaming_table_set_batch_size(_pb, _batch_size), do: nif_error()
  def pbld_streaming_table_pending_row_count(_pb), do: nif_error()
  def pbld_streaming_table_batch_count(_pb), do: nif_error()
  def pbld_streaming_table_push_row(_pb, _cells), do: nif_error()
  def pbld_streaming_table_push_row_v2(_pb, _cells, _rowspans), do: nif_error()
  def pbld_streaming_table_flush(_pb), do: nif_error()
  def pbld_streaming_table_finish(_pb), do: nif_error()
  def pbld_done(_pb), do: nif_error()
  def pbld_close(_pb), do: nif_error()

  # phase 6 — digital signatures / PKI / timestamps / TSA / DSS / validation
  def cert_load_from_bytes(_bytes, _password), do: nif_error()
  def cert_load_from_pem(_cert_pem, _key_pem), do: nif_error()
  def cert_get_subject(_cert), do: nif_error()
  def cert_get_issuer(_cert), do: nif_error()
  def cert_get_serial(_cert), do: nif_error()
  def cert_get_validity(_cert), do: nif_error()
  def cert_is_valid(_cert), do: nif_error()
  def cert_close(_cert), do: nif_error()

  def sign_bytes(_pdf, _cert, _reason, _location), do: nif_error()

  def sign_bytes_pades(_pdf, _cert, _level, _tsa_url, _reason, _location, _certs, _crls, _ocsps),
    do: nif_error()

  def sign_bytes_pades_opts(
        _pdf,
        _cert,
        _level,
        _tsa_url,
        _reason,
        _location,
        _certs,
        _crls,
        _ocsps
      ),
      do: nif_error()

  def sig_get_signer_name(_sig), do: nif_error()
  def sig_get_signing_reason(_sig), do: nif_error()
  def sig_get_signing_location(_sig), do: nif_error()
  def sig_get_signing_time(_sig), do: nif_error()
  def sig_get_certificate(_sig), do: nif_error()
  def sig_get_pades_level(_sig), do: nif_error()
  def sig_has_timestamp(_sig), do: nif_error()
  def sig_get_timestamp(_sig), do: nif_error()
  def sig_add_timestamp(_sig, _ts), do: nif_error()
  def sig_verify(_sig), do: nif_error()
  def sig_verify_detached(_sig, _pdf), do: nif_error()
  def sig_close(_sig), do: nif_error()

  def ts_parse(_bytes), do: nif_error()
  def ts_get_token(_ts), do: nif_error()
  def ts_get_message_imprint(_ts), do: nif_error()
  def ts_get_time(_ts), do: nif_error()
  def ts_get_serial(_ts), do: nif_error()
  def ts_get_tsa_name(_ts), do: nif_error()
  def ts_get_policy_oid(_ts), do: nif_error()
  def ts_get_hash_algorithm(_ts), do: nif_error()
  def ts_verify(_ts), do: nif_error()
  def ts_close(_ts), do: nif_error()

  def tsa_create(_url, _username, _password, _timeout, _hash_algo, _use_nonce, _cert_req),
    do: nif_error()

  def tsa_request_timestamp(_client, _data), do: nif_error()
  def tsa_request_timestamp_hash(_client, _hash, _hash_algo), do: nif_error()
  def tsa_close(_client), do: nif_error()

  def dss_cert_count(_dss), do: nif_error()
  def dss_crl_count(_dss), do: nif_error()
  def dss_ocsp_count(_dss), do: nif_error()
  def dss_vri_count(_dss), do: nif_error()
  def dss_get_cert(_dss, _index), do: nif_error()
  def dss_get_crl(_dss, _index), do: nif_error()
  def dss_get_ocsp(_dss, _index), do: nif_error()
  def dss_close(_dss), do: nif_error()

  def validate_pdf_a(_doc, _level), do: nif_error()
  def validate_pdf_ua(_doc, _level), do: nif_error()
  def validate_pdf_x(_doc, _level), do: nif_error()
  def pdf_a_is_compliant(_results), do: nif_error()
  def pdf_a_error_count(_results), do: nif_error()
  def pdf_a_warning_count(_results), do: nif_error()
  def pdf_a_get_error(_results, _index), do: nif_error()
  def pdf_a_close(_results), do: nif_error()
  def pdf_ua_is_accessible(_results), do: nif_error()
  def pdf_ua_error_count(_results), do: nif_error()
  def pdf_ua_warning_count(_results), do: nif_error()
  def pdf_ua_get_error(_results, _index), do: nif_error()
  def pdf_ua_get_warning(_results, _index), do: nif_error()
  def pdf_ua_get_stats(_results), do: nif_error()
  def pdf_ua_close(_results), do: nif_error()
  def pdf_x_is_compliant(_results), do: nif_error()
  def pdf_x_error_count(_results), do: nif_error()
  def pdf_x_get_error(_results, _index), do: nif_error()
  def pdf_x_close(_results), do: nif_error()

  def oxide_set_log_level(_level), do: nif_error()
  def oxide_get_log_level, do: nif_error()

  # phase 7 — barcodes / QR / OCR / render variants / redaction / constructors /
  # page getters / timestamp
  def barcode_generate_qr(_data, _error_correction, _size_px), do: nif_error()
  def barcode_generate(_data, _format, _size_px), do: nif_error()
  def barcode_get_data(_barcode), do: nif_error()
  def barcode_get_format(_barcode), do: nif_error()
  def barcode_get_confidence(_barcode), do: nif_error()
  def barcode_get_image_png(_barcode, _size_px), do: nif_error()
  def barcode_get_svg(_barcode, _size_px), do: nif_error()
  def barcode_close(_barcode), do: nif_error()
  def editor_add_barcode_to_page(_ed, _page, _barcode, _x, _y, _w, _h), do: nif_error()

  def ocr_engine_create(_det_model_path, _rec_model_path, _dict_path), do: nif_error()
  def ocr_engine_close(_engine), do: nif_error()
  def ocr_page_needs_ocr(_doc, _page), do: nif_error()
  def ocr_extract_text(_doc, _page, _engine), do: nif_error()

  def doc_render_page_with_options(
        _doc,
        _page,
        _dpi,
        _format,
        _bg_r,
        _bg_g,
        _bg_b,
        _bg_a,
        _transparent,
        _render_annotations,
        _jpeg_quality
      ),
      do: nif_error()

  def doc_render_page_with_options_ex(
        _doc,
        _page,
        _dpi,
        _format,
        _bg_r,
        _bg_g,
        _bg_b,
        _bg_a,
        _transparent,
        _render_annotations,
        _jpeg_quality,
        _excluded_layers
      ),
      do: nif_error()

  def doc_render_page_region(_doc, _page, _crop_x, _crop_y, _crop_w, _crop_h, _format),
    do: nif_error()

  def doc_render_page_fit(_doc, _page, _w, _h, _format), do: nif_error()
  def doc_render_page_raw(_doc, _page, _dpi), do: nif_error()
  def renderer_create(_dpi, _format, _quality, _anti_alias), do: nif_error()
  def renderer_close(_renderer), do: nif_error()
  def doc_estimate_render_time(_doc, _page), do: nif_error()

  def redaction_add(_ed, _page, _x1, _y1, _x2, _y2, _r, _g, _b), do: nif_error()
  def redaction_count(_ed, _page), do: nif_error()
  def redaction_apply(_ed, _scrub_metadata, _r, _g, _b), do: nif_error()
  def redaction_scrub_metadata(_ed), do: nif_error()

  def pdf_from_image(_path), do: nif_error()
  def pdf_from_image_bytes(_data), do: nif_error()
  def pdf_from_html_css(_html, _css, _font_bytes), do: nif_error()
  def pdf_from_html_css_with_fonts(_html, _css, _families, _font_bytes), do: nif_error()
  def pdf_merge(_paths), do: nif_error()

  def page_get_width(_doc, _page), do: nif_error()
  def page_get_height(_doc, _page), do: nif_error()
  def page_get_rotation(_doc, _page), do: nif_error()
  def page_get_elements(_doc, _page), do: nif_error()
  def elements_count(_elements), do: nif_error()
  def elements_close(_elements), do: nif_error()

  def add_timestamp(_pdf_data, _sig_index, _tsa_url), do: nif_error()

  # phase 8 — office I/O
  def doc_open_from_docx_bytes(_bytes), do: nif_error()
  def doc_open_from_pptx_bytes(_bytes), do: nif_error()
  def doc_open_from_xlsx_bytes(_bytes), do: nif_error()
  def doc_to_docx(_doc), do: nif_error()
  def doc_to_pptx(_doc), do: nif_error()
  def doc_to_xlsx(_doc), do: nif_error()

  # phase 8 — in-rect extractors
  def doc_extract_text_in_rect(_doc, _page, _x, _y, _w, _h), do: nif_error()
  def doc_extract_words_in_rect(_doc, _page, _x, _y, _w, _h), do: nif_error()
  def doc_extract_lines_in_rect(_doc, _page, _x, _y, _w, _h), do: nif_error()
  def doc_extract_tables_in_rect(_doc, _page, _x, _y, _w, _h), do: nif_error()
  def doc_extract_images_in_rect(_doc, _page, _x, _y, _w, _h), do: nif_error()

  # phase 8 — auto extraction / classification
  def doc_extract_text_auto(_doc, _page), do: nif_error()
  def doc_extract_all_text(_doc), do: nif_error()
  def doc_extract_page_auto(_doc, _page, _options_json), do: nif_error()
  def doc_classify_page(_doc, _page), do: nif_error()
  def doc_classify_document(_doc), do: nif_error()

  # phase 8 — header / footer / artifact
  def doc_erase_header(_doc, _page), do: nif_error()
  def doc_erase_footer(_doc, _page), do: nif_error()
  def doc_erase_artifacts(_doc, _page), do: nif_error()
  def doc_remove_headers(_doc, _threshold), do: nif_error()
  def doc_remove_footers(_doc, _threshold), do: nif_error()
  def doc_remove_artifacts(_doc, _threshold), do: nif_error()

  # phase 8 — forms
  def doc_get_form_fields(_doc), do: nif_error()
  def doc_export_form_data_to_bytes(_doc, _format_type), do: nif_error()
  def doc_import_form_data(_doc, _data_path), do: nif_error()
  def editor_import_fdf_bytes(_ed, _bytes), do: nif_error()
  def editor_import_xfdf_bytes(_ed, _bytes), do: nif_error()
  def form_import_from_file(_doc, _filename), do: nif_error()

  # phase 8 — document structure / metadata
  def doc_get_outline(_doc), do: nif_error()
  def doc_get_page_labels(_doc), do: nif_error()
  def doc_get_xmp_metadata(_doc), do: nif_error()
  def doc_get_source_bytes(_doc), do: nif_error()
  def doc_has_xfa(_doc), do: nif_error()
  def doc_get_page_count(_pdf), do: nif_error()
  def doc_plan_split_by_bookmarks(_doc, _options_json), do: nif_error()

  # phase 8 — document-level signatures
  def doc_sign(_doc, _cert, _reason, _location), do: nif_error()
  def doc_get_signature_count(_doc), do: nif_error()
  def doc_get_signature(_doc, _index), do: nif_error()
  def doc_verify_all_signatures(_doc), do: nif_error()
  def doc_has_timestamp(_doc), do: nif_error()
  def doc_get_dss(_doc), do: nif_error()

  # phase 8 — annotation extras
  def annot_get_color(_doc, _page, _index), do: nif_error()
  def annot_get_creation_date(_doc, _page, _index), do: nif_error()
  def annot_get_modification_date(_doc, _page, _index), do: nif_error()
  def annot_is_hidden(_doc, _page, _index), do: nif_error()
  def annot_is_marked_deleted(_doc, _page, _index), do: nif_error()
  def annot_is_printable(_doc, _page, _index), do: nif_error()
  def annot_is_read_only(_doc, _page, _index), do: nif_error()
  def annot_link_get_uri(_doc, _page, _index), do: nif_error()
  def annot_text_get_icon_name(_doc, _page, _index), do: nif_error()
  def annot_highlight_quad_points_count(_doc, _page, _index), do: nif_error()
  def annot_highlight_quad_point(_doc, _page, _index, _quad_index), do: nif_error()
  def annotations_to_json(_doc, _page), do: nif_error()

  # phase 8 — element / font / search JSON accessors
  def element_get_type(_elements, _index), do: nif_error()
  def element_get_text(_elements, _index), do: nif_error()
  def element_get_rect(_elements, _index), do: nif_error()
  def elements_to_json(_elements), do: nif_error()
  def font_get_size(_doc, _page, _index), do: nif_error()
  def fonts_to_json(_doc, _page), do: nif_error()
  def search_results_to_json(_doc, _term, _case_sensitive), do: nif_error()

  # phase 8 — crypto / FIPS
  def crypto_active_provider, do: nif_error()
  def crypto_cbom, do: nif_error()
  def crypto_inventory, do: nif_error()
  def crypto_policy, do: nif_error()
  def crypto_fips_available, do: nif_error()
  def crypto_use_fips, do: nif_error()
  def crypto_set_policy(_spec), do: nif_error()

  # phase 8 — models / config
  def model_manifest, do: nif_error()
  def prefetch_available, do: nif_error()
  def prefetch_models(_languages_csv), do: nif_error()
  def set_max_ops_per_stream(_limit), do: nif_error()
  def set_preserve_unmapped_glyphs(_preserve), do: nif_error()

  # phase 8 — PDF/A conversion
  def doc_convert_to_pdf_a(_doc, _level), do: nif_error()
end
