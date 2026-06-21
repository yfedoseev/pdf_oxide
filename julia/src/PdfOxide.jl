# pdf_oxide — idiomatic Julia bindings over the C ABI via ccall.
#
# Loads the native cdylib (libpdf_oxide) at runtime; handles are wrapped in
# mutable structs with finalizers; C strings/buffers are copied into Julia and
# freed via free_string; non-success C-ABI error codes throw PdfOxideError.
#
# API surface mirrors the other language bindings; coverage is asserted by
# test/runtests.jl (one test per public method).
module PdfOxide

export PdfDocument, PdfPage, Pdf, PdfOxideError, PdfVersion
export open_document, open_from_bytes, open_with_password
export page_count, version, is_encrypted, has_structure_tree
export extract_text,
    to_plain_text, to_markdown, to_html, to_markdown_all, extract_structured_json
export to_html_all, to_plain_text_all, authenticate, page, text, markdown, html, plain_text
export from_markdown, from_html, from_text, save, to_bytes, close!
export Bbox, Char, Word, TextLine, Table
export extract_chars, extract_words, extract_text_lines, extract_tables, cell
export Font, Image, Annotation, Path, SearchResult
export embedded_fonts, embedded_images, page_annotations, extract_paths, search, search_all
export RenderedImage,
    render_page,
    renderPage,
    render_page_zoom,
    renderPageZoom,
    render_page_thumbnail,
    renderPageThumbnail
export DocumentEditor
export open_editor, open_editor_from_bytes, is_modified, get_source_path
export get_producer, set_producer, get_creation_date, set_creation_date
export save_to_bytes, save_to_bytes_with_options, extract_pages_to_bytes
export convert_to_pdf_a, save_encrypted_to_bytes, save_encrypted
export merge_from_bytes, merge_from, embed_file
export apply_page_redactions, apply_all_redactions
export rotate_all_pages, rotate_page_by, get_page_rotation, set_page_rotation
export delete_page, move_page
export get_page_media_box, set_page_media_box, get_page_crop_box, set_page_crop_box
export crop_margins, erase_region, erase_regions, clear_erase_regions
export is_page_marked_for_flatten, unmark_page_for_flatten
export is_page_marked_for_redaction, unmark_page_for_redaction
export flatten_annotations, flatten_all_annotations
export set_form_field_value, flatten_forms, flatten_forms_on_page
export flatten_warnings_count, flatten_warning
# PDF creation builder API.
export DocumentBuilder, PageBuilder, EmbeddedFont
export embedded_font_from_file, embedded_font_from_bytes
export set_title, set_author, set_subject, set_keywords, set_creator
export on_open, language, role_map, register_embedded_font, tagged_pdf_ua1
export a4_page, letter_page, build, save_encrypted_builder, to_bytes_encrypted
export font, at, heading, paragraph, space, horizontal_rule
export link_url, link_page, link_named, link_javascript, on_close
export field_keystroke, field_format, field_validate, field_calculate
export highlight, underline, strikeout, squiggly
export sticky_note, sticky_note_at, watermark, watermark_confidential, watermark_draft
export stamp, freetext, text_field, checkbox, combo_box, radio_group, push_button
export signature_field, footnote, columns
export inline, inline_bold, inline_italic, inline_color, newline
export barcode_1d, barcode_qr, image, image_with_alt, image_artifact
export rect, filled_rect, line, stroke_rect, stroke_line
export stroke_rect_dashed, stroke_line_dashed, text_in_rect, new_page_same_size, table
export streaming_table_begin, streaming_table_begin_v2, streaming_table_set_batch_size
export streaming_table_pending_row_count, streaming_table_batch_count
export streaming_table_flush, streaming_table_push_row, streaming_table_push_row_v2
export streaming_table_finish, done
# Phase-6: digital signatures / PKI / timestamps / TSA / DSS / validation.
export set_log_level, get_log_level
export Certificate, certificate_load_from_bytes, certificate_load_from_pem
export certificate_get_subject, certificate_get_issuer, certificate_get_serial
export certificate_get_validity, certificate_is_valid
export sign_bytes, sign_bytes_pades, sign_bytes_pades_opts, PadesSignOptionsC
export SignatureInfo
export signature_get_signer_name, signature_get_signing_reason
export signature_get_signing_location, signature_get_signing_time
export signature_get_certificate, signature_get_pades_level
export signature_has_timestamp, signature_get_timestamp, signature_add_timestamp
export signature_verify, signature_verify_detached
export Timestamp, timestamp_parse, timestamp_get_token, timestamp_get_message_imprint
export timestamp_get_time, timestamp_get_serial, timestamp_get_tsa_name
export timestamp_get_policy_oid, timestamp_get_hash_algorithm, timestamp_verify
export TsaClient, tsa_client_create, tsa_request_timestamp, tsa_request_timestamp_hash
export Dss, document_get_dss
export dss_cert_count, dss_crl_count, dss_ocsp_count, dss_vri_count
export dss_get_cert, dss_get_crl, dss_get_ocsp
export PdfAResults, UaResults, PdfXResults
export validate_pdf_a, validate_pdf_ua, validate_pdf_x
export validatePdfA, validatePdfUa, validatePdfX
export is_compliant, is_accessible, errors, warnings, ua_stats
export pdf_a_error_count, pdf_a_warning_count, pdf_ua_error_count
export pdf_ua_warning_count, pdf_x_error_count
# Phase-7: barcodes/QR, OCR, render variants, redaction, image/HTML-CSS/merge
# constructors, page getters, and timestamp.
export Barcode, generate_qr_code, generate_barcode
export barcode_get_data, barcode_get_format, barcode_get_confidence
export barcode_get_image_png, barcode_get_svg, add_barcode_to_page
export OcrEngine, ocr_engine_create, page_needs_ocr, ocr_extract_text
export render_page_with_options, render_page_with_options_ex
export render_page_region, render_page_fit, render_page_raw
export Renderer, create_renderer, estimate_render_time
export redaction_add, redaction_count, redaction_apply, redaction_scrub_metadata
export from_image, from_image_bytes, from_html_css, from_html_css_with_fonts, merge_pdfs
export page_get_width, page_get_height, page_get_rotation
export ElementList, page_get_elements, element_count
export add_timestamp
# Phase-8 (final coverage): office I/O, in-rect extractors, auto extraction,
# header/footer/artifact removal, forms, doc structure/metadata, doc-level
# signatures, annotation extras, element/JSON accessors, crypto/FIPS, models.
export open_from_docx_bytes, open_from_pptx_bytes, open_from_xlsx_bytes
export to_docx, to_pptx, to_xlsx
export extract_text_in_rect, extract_words_in_rect, extract_lines_in_rect
export extract_tables_in_rect, extract_images_in_rect
export extract_text_auto, extract_all_text, extract_page_auto
export classify_page, classify_document
export erase_header, erase_footer, erase_artifacts
export remove_headers, remove_footers, remove_artifacts
export FormField, get_form_fields, form_field_count
export form_field_name, form_field_value, form_field_type
export form_field_is_readonly, form_field_is_required
export export_form_data_to_bytes, import_form_data
export import_fdf_bytes, import_xfdf_bytes, form_import_from_file
export get_outline, get_page_labels, get_xmp_metadata, get_source_bytes
export has_xfa, get_page_count, plan_split_by_bookmarks
export sign, get_signature, get_signature_count, verify_all_signatures
export has_timestamp, document_convert_to_pdf_a
export annotation_get_color, annotation_creation_date, annotation_modification_date
export annotation_is_hidden, annotation_is_marked_deleted, annotation_is_printable
export annotation_is_read_only
export highlight_quad_points_count, highlight_quad_point
export link_annotation_uri, text_annotation_icon_name, annotations_to_json
export element_type, element_text, element_rect, elements_to_json
export fonts_to_json, font_size, search_results_to_json
export crypto_active_provider, crypto_cbom, crypto_fips_available, crypto_inventory
export crypto_policy, crypto_set_policy, crypto_use_fips
export model_manifest, prefetch_available, prefetch_models
export set_max_ops_per_stream, set_preserve_unmapped_glyphs

# Native library resolution: PDF_OXIDE_LIB_PATH (full path) -> PDF_OXIDE_LIB_DIR
# -> common build dirs -> bare name (system loader).
function _libpath()
    p = get(ENV, "PDF_OXIDE_LIB_PATH", "")
    !isempty(p) && isfile(p) && return p
    name =
        Sys.isapple() ? "libpdf_oxide.dylib" :
        Sys.iswindows() ? "pdf_oxide.dll" : "libpdf_oxide.so"
    for dir in (get(ENV, "PDF_OXIDE_LIB_DIR", ""), "../target/release", "target/release")
        isempty(dir) && continue
        cand = joinpath(dir, name)
        isfile(cand) && return cand
    end
    return name  # let the system loader find it
end

const LIB = _libpath()

"""Thrown on any non-success C-ABI error code."""
struct PdfOxideError <: Exception
    code::Int32
    op::String
end
Base.showerror(io::IO, e::PdfOxideError) =
    print(io, "PdfOxideError: $(e.op) failed (error code $(e.code))")

"""PDF version with named `major` / `minor` fields."""
struct PdfVersion
    major::Int
    minor::Int
end
Base.show(io::IO, v::PdfVersion) = print(io, "$(v.major).$(v.minor)")

# Copy a C string return into a Julia String and free it via free_string.
function _take_string(ptr::Ptr{UInt8}, code::Int32, op::String)
    ptr == C_NULL && throw(PdfOxideError(code, op))
    s = unsafe_string(ptr)
    ccall((:free_string, LIB), Cvoid, (Ptr{UInt8},), ptr)
    return s
end

# ── Document ──────────────────────────────────────────────────────────────────
mutable struct PdfDocument
    handle::Ptr{Cvoid}
    function PdfDocument(h::Ptr{Cvoid})
        d = new(h)
        finalizer(close!, d)
        return d
    end
end

"""Free the native handle now (idempotent; also runs at finalization)."""
function close!(d::PdfDocument)
    if d.handle != C_NULL
        ccall((:pdf_document_free, LIB), Cvoid, (Ptr{Cvoid},), d.handle)
        d.handle = C_NULL
    end
    return nothing
end

_doc(d::PdfDocument) = (d.handle == C_NULL && error("PdfDocument is closed"); d.handle)

"""Open a PDF from a filesystem path (optionally password-protected)."""
function open_document(
    path::AbstractString;
    password::Union{Nothing,AbstractString} = nothing,
)
    code = Ref{Int32}(0)
    h = if password === nothing
        ccall((:pdf_document_open, LIB), Ptr{Cvoid}, (Cstring, Ref{Int32}), path, code)
    else
        ccall(
            (:pdf_document_open_with_password, LIB),
            Ptr{Cvoid},
            (Cstring, Cstring, Ref{Int32}),
            path,
            password,
            code,
        )
    end
    h == C_NULL && throw(PdfOxideError(code[], "open_document"))
    return PdfDocument(h)
end

"""Open a PDF from an in-memory byte vector."""
function open_from_bytes(data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_document_open_from_bytes, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Csize_t, Ref{Int32}),
        data,
        length(data),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "open_from_bytes"))
    return PdfDocument(h)
end

open_with_password(path::AbstractString, password::AbstractString) =
    open_document(path; password = password)

"""Number of pages."""
function page_count(d::PdfDocument)
    code = Ref{Int32}(0)
    n = ccall(
        (:pdf_document_get_page_count, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    n < 0 && throw(PdfOxideError(code[], "page_count"))
    return Int(n)
end

"""PDF version as `(major, minor)`."""
function version(d::PdfDocument)
    maj = Ref{UInt8}(0);
    min = Ref{UInt8}(0)
    ccall(
        (:pdf_document_get_version, LIB),
        Cvoid,
        (Ptr{Cvoid}, Ref{UInt8}, Ref{UInt8}),
        _doc(d),
        maj,
        min,
    )
    return PdfVersion(Int(maj[]), Int(min[]))
end

is_encrypted(d::PdfDocument) =
    ccall((:pdf_document_is_encrypted, LIB), Bool, (Ptr{Cvoid},), _doc(d))
has_structure_tree(d::PdfDocument) =
    ccall((:pdf_document_has_structure_tree, LIB), Bool, (Ptr{Cvoid},), _doc(d))

# Per-page text extractors. Generated with @eval so each ccall references its C
# function name as a LITERAL symbol (ccall forbids a variable function name).
for (jl_fn, c_fn) in (
    (:extract_text, :pdf_document_extract_text),
    (:to_plain_text, :pdf_document_to_plain_text),
    (:to_markdown, :pdf_document_to_markdown),
    (:to_html, :pdf_document_to_html),
    (:extract_structured_json, :pdf_document_extract_structured_to_json),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _doc(d),
            Int32(page),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""Markdown for the whole document."""
function to_markdown_all(d::PdfDocument)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_to_markdown_all, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    return _take_string(ptr, code[], "to_markdown_all")
end

"""HTML for the whole document."""
function to_html_all(d::PdfDocument)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_to_html_all, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    return _take_string(ptr, code[], "to_html_all")
end

"""Plain text for the whole document."""
function to_plain_text_all(d::PdfDocument)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_to_plain_text_all, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    return _take_string(ptr, code[], "to_plain_text_all")
end

"""
Authenticate against an encrypted document's password. Returns `true`/`false`
(a wrong password is not an error). Only a set C-ABI error code throws.
"""
function authenticate(d::PdfDocument, password::AbstractString)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_document_authenticate, LIB),
        Bool,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _doc(d),
        password,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "authenticate"))
    return ok
end

# ── Element extraction ────────────────────────────────────────────────────────
# Value type for an axis-aligned bounding box (PDF user-space units).
struct Bbox
    x::Float64
    y::Float64
    width::Float64
    height::Float64
end

"""A single extracted glyph: its `character` (codepoint), `bbox`, `font_name`, `font_size`."""
struct Char
    character::UInt32
    bbox::Bbox
    font_name::String
    font_size::Float64
end

"""An extracted word with `text`, `bbox`, `font_name`, `font_size`, `bold`."""
struct Word
    text::String
    bbox::Bbox
    font_name::String
    font_size::Float64
    bold::Bool
end

"""An extracted text line with `text`, `bbox`, `word_count`."""
struct TextLine
    text::String
    bbox::Bbox
    word_count::Int
end

"""An extracted table with `row_count`, `col_count`, `has_header`, and `cells`."""
struct Table
    row_count::Int
    col_count::Int
    has_header::Bool
    cells::Matrix{String}
end

"""Cell text at (0-based) `row`, `col`."""
cell(t::Table, row::Integer, col::Integer) = t.cells[Int(row)+1, Int(col)+1]

# Read a list bbox out-param into a Bbox value.
# bbox readers — one per C function, generated with @eval so each ccall uses a
# LITERAL symbol (ccall forbids a variable function name).
for (jl_fn, c_fn) in (
    (:_bbox_char, :pdf_oxide_char_get_bbox),
    (:_bbox_word, :pdf_oxide_word_get_bbox),
    (:_bbox_line, :pdf_oxide_line_get_bbox),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        x = Ref{Float32}(0);
        y = Ref{Float32}(0)
        w = Ref{Float32}(0);
        h = Ref{Float32}(0)
        code = Ref{Int32}(0)
        ccall(
            ($(QuoteNode(c_fn)), LIB),
            Cvoid,
            (
                Ptr{Cvoid},
                Int32,
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Int32},
            ),
            list,
            Int32(index),
            x,
            y,
            w,
            h,
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return Bbox(Float64(x[]), Float64(y[]), Float64(w[]), Float64(h[]))
    end
end

# Element-list openers — one per entry point (NULL on error -> throw).
for (jl_fn, c_fn) in (
    (:_open_chars, :pdf_document_extract_chars),
    (:_open_words, :pdf_document_extract_words),
    (:_open_lines, :pdf_document_extract_text_lines),
    (:_open_tables, :pdf_document_extract_tables),
)
    @eval function $jl_fn(d::PdfDocument, page::Integer, op::String)
        code = Ref{Int32}(0)
        list = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _doc(d),
            Int32(page),
            code,
        )
        list == C_NULL && throw(PdfOxideError(code[], op))
        return list
    end
end

"""Extract glyphs from a (0-based) page as a `Vector{Char}`."""
function extract_chars(d::PdfDocument, page::Integer)
    list = _open_chars(d, page, "extract_chars")
    try
        n = ccall((:pdf_oxide_char_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Char}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            code = Ref{Int32}(0)
            cp = ccall(
                (:pdf_oxide_char_get_char, LIB),
                UInt32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                code,
            )
            code[] != 0 && throw(PdfOxideError(code[], "extract_chars"))
            bb = _bbox_char(list, i, "extract_chars")
            fcode = Ref{Int32}(0)
            fptr = ccall(
                (:pdf_oxide_char_get_font_name, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                fcode,
            )
            font = _take_string(fptr, fcode[], "extract_chars")
            scode = Ref{Int32}(0)
            fs = ccall(
                (:pdf_oxide_char_get_font_size, LIB),
                Float32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                scode,
            )
            scode[] != 0 && throw(PdfOxideError(scode[], "extract_chars"))
            out[i+1] = Char(cp, bb, font, Float64(fs))
        end
        return out
    finally
        ccall((:pdf_oxide_char_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Extract words from a (0-based) page as a `Vector{Word}`."""
function extract_words(d::PdfDocument, page::Integer)
    list = _open_words(d, page, "extract_words")
    try
        n = ccall((:pdf_oxide_word_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Word}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            tcode = Ref{Int32}(0)
            tptr = ccall(
                (:pdf_oxide_word_get_text, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                tcode,
            )
            txt = _take_string(tptr, tcode[], "extract_words")
            bb = _bbox_word(list, i, "extract_words")
            fcode = Ref{Int32}(0)
            fptr = ccall(
                (:pdf_oxide_word_get_font_name, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                fcode,
            )
            font = _take_string(fptr, fcode[], "extract_words")
            scode = Ref{Int32}(0)
            fs = ccall(
                (:pdf_oxide_word_get_font_size, LIB),
                Float32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                scode,
            )
            scode[] != 0 && throw(PdfOxideError(scode[], "extract_words"))
            bcode = Ref{Int32}(0)
            bold = ccall(
                (:pdf_oxide_word_is_bold, LIB),
                Bool,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                bcode,
            )
            bcode[] != 0 && throw(PdfOxideError(bcode[], "extract_words"))
            out[i+1] = Word(txt, bb, font, Float64(fs), bold)
        end
        return out
    finally
        ccall((:pdf_oxide_word_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Extract text lines from a (0-based) page as a `Vector{TextLine}`."""
function extract_text_lines(d::PdfDocument, page::Integer)
    list = _open_lines(d, page, "extract_text_lines")
    try
        n = ccall((:pdf_oxide_line_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{TextLine}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            tcode = Ref{Int32}(0)
            tptr = ccall(
                (:pdf_oxide_line_get_text, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                tcode,
            )
            txt = _take_string(tptr, tcode[], "extract_text_lines")
            bb = _bbox_line(list, i, "extract_text_lines")
            wcode = Ref{Int32}(0)
            wc = ccall(
                (:pdf_oxide_line_get_word_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                wcode,
            )
            wcode[] != 0 && throw(PdfOxideError(wcode[], "extract_text_lines"))
            out[i+1] = TextLine(txt, bb, Int(wc))
        end
        return out
    finally
        ccall((:pdf_oxide_line_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Extract tables from a (0-based) page as a `Vector{Table}`."""
function extract_tables(d::PdfDocument, page::Integer)
    list = _open_tables(d, page, "extract_tables")
    try
        n = ccall((:pdf_oxide_table_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Table}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            rcode = Ref{Int32}(0)
            rows = ccall(
                (:pdf_oxide_table_get_row_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                rcode,
            )
            rcode[] != 0 && throw(PdfOxideError(rcode[], "extract_tables"))
            ccode = Ref{Int32}(0)
            cols = ccall(
                (:pdf_oxide_table_get_col_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                ccode,
            )
            ccode[] != 0 && throw(PdfOxideError(ccode[], "extract_tables"))
            hcode = Ref{Int32}(0)
            hdr = ccall(
                (:pdf_oxide_table_has_header, LIB),
                Bool,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                hcode,
            )
            hcode[] != 0 && throw(PdfOxideError(hcode[], "extract_tables"))
            nr = rows < 0 ? 0 : Int(rows)
            nc = cols < 0 ? 0 : Int(cols)
            cells = Matrix{String}(undef, nr, nc)
            for r = 0:(nr-1), c = 0:(nc-1)
                xcode = Ref{Int32}(0)
                cptr = ccall(
                    (:pdf_oxide_table_get_cell_text, LIB),
                    Ptr{UInt8},
                    (Ptr{Cvoid}, Int32, Int32, Int32, Ref{Int32}),
                    list,
                    Int32(i),
                    Int32(r),
                    Int32(c),
                    xcode,
                )
                cells[r+1, c+1] = _take_string(cptr, xcode[], "extract_tables")
            end
            out[i+1] = Table(nr, nc, hdr, cells)
        end
        return out
    finally
        ccall((:pdf_oxide_table_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# ── Phase-2 extraction (fonts, images, annotations, paths, search) ──────────────
"""An embedded font with `name`, `type`, `encoding`, `embedded`, `subset`."""
struct Font
    name::String
    type::String
    encoding::String
    embedded::Bool
    subset::Bool
end

"""An embedded image with `width`, `height`, `bitsPerComponent`, `format`, `colorspace`, `data`."""
struct Image
    width::Int
    height::Int
    bitsPerComponent::Int
    format::String
    colorspace::String
    data::Vector{UInt8}
end

"""An annotation with `type`, `subtype`, `content`, `author`, `rect` (Bbox), `borderWidth`."""
struct Annotation
    type::String
    subtype::String
    content::String
    author::String
    rect::Bbox
    borderWidth::Float64
end

"""A vector path with `bbox` (Bbox), `strokeWidth`, `hasStroke`, `hasFill`, `operationCount`."""
struct Path
    bbox::Bbox
    strokeWidth::Float64
    hasStroke::Bool
    hasFill::Bool
    operationCount::Int
end

"""A search hit with `text`, `page`, `bbox` (Bbox)."""
struct SearchResult
    text::String
    page::Int
    bbox::Bbox
end

# bbox readers for the Phase-2 lists — one per C function, generated with @eval so
# each ccall uses a LITERAL symbol (ccall forbids a variable function name).
for (jl_fn, c_fn) in (
    (:_bbox_annotation, :pdf_oxide_annotation_get_rect),
    (:_bbox_path, :pdf_oxide_path_get_bbox),
    (:_bbox_search, :pdf_oxide_search_result_get_bbox),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        x = Ref{Float32}(0);
        y = Ref{Float32}(0)
        w = Ref{Float32}(0);
        h = Ref{Float32}(0)
        code = Ref{Int32}(0)
        ccall(
            ($(QuoteNode(c_fn)), LIB),
            Cvoid,
            (
                Ptr{Cvoid},
                Int32,
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Int32},
            ),
            list,
            Int32(index),
            x,
            y,
            w,
            h,
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return Bbox(Float64(x[]), Float64(y[]), Float64(w[]), Float64(h[]))
    end
end

# Phase-2 list openers — one per entry point (NULL on error -> throw).
for (jl_fn, c_fn) in (
    (:_open_fonts, :pdf_document_get_embedded_fonts),
    (:_open_images, :pdf_document_get_embedded_images),
    (:_open_annotations, :pdf_document_get_page_annotations),
    (:_open_paths, :pdf_document_extract_paths),
)
    @eval function $jl_fn(d::PdfDocument, page::Integer, op::String)
        code = Ref{Int32}(0)
        list = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _doc(d),
            Int32(page),
            code,
        )
        list == C_NULL && throw(PdfOxideError(code[], op))
        return list
    end
end

# Small string accessor helper for index-addressed lists, generated with @eval so
# each ccall references its C function name as a LITERAL symbol.
for (jl_fn, c_fn) in (
    (:_str_font_name, :pdf_oxide_font_get_name),
    (:_str_font_type, :pdf_oxide_font_get_type),
    (:_str_font_encoding, :pdf_oxide_font_get_encoding),
    (:_str_image_format, :pdf_oxide_image_get_format),
    (:_str_image_colorspace, :pdf_oxide_image_get_colorspace),
    (:_str_annotation_type, :pdf_oxide_annotation_get_type),
    (:_str_annotation_subtype, :pdf_oxide_annotation_get_subtype),
    (:_str_annotation_content, :pdf_oxide_annotation_get_content),
    (:_str_annotation_author, :pdf_oxide_annotation_get_author),
    (:_str_search_text, :pdf_oxide_search_result_get_text),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        return _take_string(ptr, code[], op)
    end
end

# Int32 accessor helper, generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:_i32_image_width, :pdf_oxide_image_get_width),
    (:_i32_image_height, :pdf_oxide_image_get_height),
    (:_i32_image_bpc, :pdf_oxide_image_get_bits_per_component),
    (:_i32_font_is_embedded, :pdf_oxide_font_is_embedded),
    (:_i32_font_is_subset, :pdf_oxide_font_is_subset),
    (:_i32_path_op_count, :pdf_oxide_path_get_operation_count),
    (:_i32_search_page, :pdf_oxide_search_result_get_page),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        v = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return Int(v)
    end
end

# Float32 accessor helper, generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:_f32_annotation_border_width, :pdf_oxide_annotation_get_border_width),
    (:_f32_path_stroke_width, :pdf_oxide_path_get_stroke_width),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        v = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Float32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return Float64(v)
    end
end

# Bool accessor helper, generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:_bool_path_has_stroke, :pdf_oxide_path_has_stroke),
    (:_bool_path_has_fill, :pdf_oxide_path_has_fill),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        v = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Bool,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return v
    end
end

"""Embedded fonts on a (0-based) page as a `Vector{Font}`."""
function embedded_fonts(d::PdfDocument, page::Integer)
    list = _open_fonts(d, page, "embedded_fonts")
    try
        n = ccall((:pdf_oxide_font_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Font}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            name = _str_font_name(list, i, "embedded_fonts")
            typ = _str_font_type(list, i, "embedded_fonts")
            enc = _str_font_encoding(list, i, "embedded_fonts")
            emb = _i32_font_is_embedded(list, i, "embedded_fonts") != 0
            sub = _i32_font_is_subset(list, i, "embedded_fonts") != 0
            out[i+1] = Font(name, typ, enc, emb, sub)
        end
        return out
    finally
        ccall((:pdf_oxide_font_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Embedded images on a (0-based) page as a `Vector{Image}`."""
function embedded_images(d::PdfDocument, page::Integer)
    list = _open_images(d, page, "embedded_images")
    try
        n = ccall((:pdf_oxide_image_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Image}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            w = _i32_image_width(list, i, "embedded_images")
            h = _i32_image_height(list, i, "embedded_images")
            bpc = _i32_image_bpc(list, i, "embedded_images")
            fmt = _str_image_format(list, i, "embedded_images")
            cs = _str_image_colorspace(list, i, "embedded_images")
            dlen = Ref{Int32}(0);
            dcode = Ref{Int32}(0)
            dptr = ccall(
                (:pdf_oxide_image_get_data, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}, Ref{Int32}),
                list,
                Int32(i),
                dlen,
                dcode,
            )
            data = if dptr == C_NULL
                dcode[] != 0 && throw(PdfOxideError(dcode[], "embedded_images"))
                UInt8[]
            else
                m = dlen[] < 0 ? 0 : Int(dlen[])
                bytes = copy(unsafe_wrap(Array, dptr, m))
                # Raw byte buffers free via free_bytes, not free_string.
                ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), dptr)
                bytes
            end
            out[i+1] = Image(w, h, bpc, fmt, cs, data)
        end
        return out
    finally
        ccall((:pdf_oxide_image_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Annotations on a (0-based) page as a `Vector{Annotation}`."""
function page_annotations(d::PdfDocument, page::Integer)
    list = _open_annotations(d, page, "page_annotations")
    try
        n = ccall((:pdf_oxide_annotation_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Annotation}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            typ = _str_annotation_type(list, i, "page_annotations")
            sub = _str_annotation_subtype(list, i, "page_annotations")
            content = _str_annotation_content(list, i, "page_annotations")
            author = _str_annotation_author(list, i, "page_annotations")
            rect = _bbox_annotation(list, i, "page_annotations")
            bw = _f32_annotation_border_width(list, i, "page_annotations")
            out[i+1] = Annotation(typ, sub, content, author, rect, bw)
        end
        return out
    finally
        ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Vector paths on a (0-based) page as a `Vector{Path}`."""
function extract_paths(d::PdfDocument, page::Integer)
    list = _open_paths(d, page, "extract_paths")
    try
        n = ccall((:pdf_oxide_path_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Path}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            bb = _bbox_path(list, i, "extract_paths")
            sw = _f32_path_stroke_width(list, i, "extract_paths")
            hs = _bool_path_has_stroke(list, i, "extract_paths")
            hf = _bool_path_has_fill(list, i, "extract_paths")
            oc = _i32_path_op_count(list, i, "extract_paths")
            out[i+1] = Path(bb, sw, hs, hf, oc)
        end
        return out
    finally
        ccall((:pdf_oxide_path_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# Shared marshaller for the two search entry points: count -> per-index
# accessors -> pdf_oxide_search_result_free (NOT _list_free).
function _search_results(list::Ptr{Cvoid}, op::String)
    try
        n = ccall((:pdf_oxide_search_result_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{SearchResult}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            txt = _str_search_text(list, i, op)
            pg = _i32_search_page(list, i, op)
            bb = _bbox_search(list, i, op)
            out[i+1] = SearchResult(txt, pg, bb)
        end
        return out
    finally
        ccall((:pdf_oxide_search_result_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Search a single (0-based) page for `term`; returns a `Vector{SearchResult}`."""
function search(d::PdfDocument, page::Integer, term::AbstractString, caseSensitive::Bool)
    code = Ref{Int32}(0)
    list = ccall(
        (:pdf_document_search_page, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Cstring, Bool, Ref{Int32}),
        _doc(d),
        Int32(page),
        term,
        caseSensitive,
        code,
    )
    list == C_NULL && throw(PdfOxideError(code[], "search"))
    return _search_results(list, "search")
end

"""Search the whole document for `term`; returns a `Vector{SearchResult}`."""
function search_all(d::PdfDocument, term::AbstractString, caseSensitive::Bool)
    code = Ref{Int32}(0)
    list = ccall(
        (:pdf_document_search_all, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cstring, Bool, Ref{Int32}),
        _doc(d),
        term,
        caseSensitive,
        code,
    )
    list == C_NULL && throw(PdfOxideError(code[], "search_all"))
    return _search_results(list, "search_all")
end

# ── Page ────────────────────────────────────────────────────────────────────────
# A lightweight view over one (0-based) page. Holds a strong reference to its
# PdfDocument so the native handle outlives the page.
struct PdfPage
    doc::PdfDocument
    index::Int32
    PdfPage(doc::PdfDocument, index::Integer) = new(doc, Int32(index))
end

"""A 0-based page view over the document; `index` is required."""
page(d::PdfDocument, index::Integer) = PdfPage(d, index)

# Per-page accessors delegate to the document extractors with the stored index.
for (jl_fn, doc_fn) in (
    (:text, :extract_text),
    (:markdown, :to_markdown),
    (:html, :to_html),
    (:plain_text, :to_plain_text),
    (:extract_chars, :extract_chars),
    (:extract_words, :extract_words),
    (:extract_text_lines, :extract_text_lines),
    (:extract_tables, :extract_tables),
    (:embedded_fonts, :embedded_fonts),
    (:embedded_images, :embedded_images),
    (:page_annotations, :page_annotations),
    (:extract_paths, :extract_paths),
)
    @eval $jl_fn(p::PdfPage) = $doc_fn(p.doc, p.index)
end

# Per-page search delegates carry the term + case-sensitivity arguments.
search(p::PdfPage, term::AbstractString, caseSensitive::Bool) =
    search(p.doc, p.index, term, caseSensitive)

# Per-page render delegates forward to the document renderers with the stored index.
render_page(p::PdfPage, format::Integer = 0) = render_page(p.doc, p.index, format)
render_page_zoom(p::PdfPage, zoom::Real, format::Integer = 0) =
    render_page_zoom(p.doc, p.index, zoom, format)
render_page_thumbnail(p::PdfPage, size::Integer, format::Integer = 0) =
    render_page_thumbnail(p.doc, p.index, size, format)

# ── Phase-3 page rendering ──────────────────────────────────────────────────────
# A rendered raster of one page. Owns the native FfiRenderedImage handle so
# `save(img, path)` can delegate to pdf_save_rendered_image; width/height/data
# are read eagerly (data copied out and the C buffer freed via free_bytes). The
# handle is released on close!/finalization.
mutable struct RenderedImage
    handle::Ptr{Cvoid}
    width::Int
    height::Int
    data::Vector{UInt8}
    function RenderedImage(h::Ptr{Cvoid})
        code = Ref{Int32}(0)
        w = ccall(
            (:pdf_get_rendered_image_width, LIB),
            Int32,
            (Ptr{Cvoid}, Ref{Int32}),
            h,
            code,
        )
        if code[] != 0
            ccall((:pdf_rendered_image_free, LIB), Cvoid, (Ptr{Cvoid},), h)
            throw(PdfOxideError(code[], "render"))
        end
        hcode = Ref{Int32}(0)
        ht = ccall(
            (:pdf_get_rendered_image_height, LIB),
            Int32,
            (Ptr{Cvoid}, Ref{Int32}),
            h,
            hcode,
        )
        if hcode[] != 0
            ccall((:pdf_rendered_image_free, LIB), Cvoid, (Ptr{Cvoid},), h)
            throw(PdfOxideError(hcode[], "render"))
        end
        dlen = Ref{Int32}(0);
        dcode = Ref{Int32}(0)
        dptr = ccall(
            (:pdf_get_rendered_image_data, LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}, Ref{Int32}),
            h,
            dlen,
            dcode,
        )
        data = if dptr == C_NULL
            if dcode[] != 0
                ccall((:pdf_rendered_image_free, LIB), Cvoid, (Ptr{Cvoid},), h)
                throw(PdfOxideError(dcode[], "render"))
            end
            UInt8[]
        else
            m = dlen[] < 0 ? 0 : Int(dlen[])
            bytes = copy(unsafe_wrap(Array, dptr, m))
            # Encoded image bytes free via free_bytes, not free_string.
            ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), dptr)
            bytes
        end
        img = new(h, Int(w), Int(ht), data)
        finalizer(close!, img)
        return img
    end
end

"""Free the native rendered-image handle now (idempotent; also runs at finalization)."""
function close!(img::RenderedImage)
    if img.handle != C_NULL
        ccall((:pdf_rendered_image_free, LIB), Cvoid, (Ptr{Cvoid},), img.handle)
        img.handle = C_NULL
    end
    return nothing
end

"""Save the rendered image to `path` (format inferred by the native encoder)."""
function save(img::RenderedImage, path::AbstractString)
    img.handle == C_NULL && error("RenderedImage is closed")
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_save_rendered_image, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        img.handle,
        path,
        code,
    )
    rc != 0 && throw(PdfOxideError(code[], "save_rendered_image"))
    return nothing
end

"""Render a (0-based) page to a `RenderedImage`. `format` is 0=PNG (default)."""
function render_page(d::PdfDocument, pageIndex::Integer, format::Integer = 0)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Int32, Ref{Int32}),
        _doc(d),
        Int32(pageIndex),
        Int32(format),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page"))
    return RenderedImage(h)
end

"""Render a (0-based) page at a zoom factor. `format` is 0=PNG (default)."""
function render_page_zoom(
    d::PdfDocument,
    pageIndex::Integer,
    zoom::Real,
    format::Integer = 0,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page_zoom, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Float32, Int32, Ref{Int32}),
        _doc(d),
        Int32(pageIndex),
        Float32(zoom),
        Int32(format),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page_zoom"))
    return RenderedImage(h)
end

"""Render a (0-based) page as a thumbnail fitting `size` pixels. `format` is 0=PNG (default)."""
function render_page_thumbnail(
    d::PdfDocument,
    pageIndex::Integer,
    size::Integer,
    format::Integer = 0,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page_thumbnail, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Int32, Int32, Ref{Int32}),
        _doc(d),
        Int32(pageIndex),
        Int32(size),
        Int32(format),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page_thumbnail"))
    return RenderedImage(h)
end

# camelCase aliases matching the cross-binding naming convention.
const renderPage = render_page
const renderPageZoom = render_page_zoom
const renderPageThumbnail = render_page_thumbnail

# ── Pdf builder ───────────────────────────────────────────────────────────────
mutable struct Pdf
    handle::Ptr{Cvoid}
    function Pdf(h::Ptr{Cvoid})
        p = new(h)
        finalizer(close!, p)
        return p
    end
end

function close!(p::Pdf)
    if p.handle != C_NULL
        ccall((:pdf_free, LIB), Cvoid, (Ptr{Cvoid},), p.handle)
        p.handle = C_NULL
    end
    return nothing
end

_pdf(p::Pdf) = (p.handle == C_NULL && error("Pdf is closed"); p.handle)

# Builders. Generated with @eval so each ccall uses a LITERAL C function name.
for (jl_fn, c_fn) in (
    (:from_markdown, :pdf_from_markdown),
    (:from_html, :pdf_from_html),
    (:from_text, :pdf_from_text),
)
    op = String(jl_fn)
    @eval function $jl_fn(input::AbstractString)
        code = Ref{Int32}(0)
        h = ccall(($(QuoteNode(c_fn)), LIB), Ptr{Cvoid}, (Cstring, Ref{Int32}), input, code)
        h == C_NULL && throw(PdfOxideError(code[], $op))
        return Pdf(h)
    end
end

"""Write the built PDF to a path."""
function save(p::Pdf, path::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_save, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _pdf(p),
        path,
        code,
    )
    rc != 0 && throw(PdfOxideError(code[], "save"))
    return nothing
end

"""Serialize the built PDF to a `Vector{UInt8}`."""
function to_bytes(p::Pdf)
    len = Ref{Int32}(0);
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_save_to_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}, Ref{Int32}),
        _pdf(p),
        len,
        code,
    )
    ptr == C_NULL && throw(PdfOxideError(code[], "to_bytes"))
    n = len[] < 0 ? 0 : Int(len[])
    out = copy(unsafe_wrap(Array, ptr, n))
    # Raw byte buffers free via free_bytes, not free_string (which does strlen).
    ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), ptr)
    return out
end

# ── DocumentEditor ──────────────────────────────────────────────────────────────
# Mutable editing handle over the C ABI's DocumentEditor. Mirrors the
# PdfDocument/Pdf pattern: an owned native handle freed on close!/finalization;
# the same PdfOxideError helpers, _take_string, free_bytes byte-take, double/
# uint8 out-param helpers, and a closed-handle guard. Methods use snake_case
# (Julia idiom); page indices are 0-based.
mutable struct DocumentEditor
    handle::Ptr{Cvoid}
    function DocumentEditor(h::Ptr{Cvoid})
        e = new(h)
        finalizer(close!, e)
        return e
    end
end

"""Free the native editor handle now (idempotent; also runs at finalization)."""
function close!(e::DocumentEditor)
    if e.handle != C_NULL
        ccall((:document_editor_free, LIB), Cvoid, (Ptr{Cvoid},), e.handle)
        e.handle = C_NULL
    end
    return nothing
end

_editor(e::DocumentEditor) =
    (e.handle == C_NULL && error("DocumentEditor is closed"); e.handle)

# Copy a raw byte buffer return (uintptr_t out-len) into a Julia Vector and free
# it via free_bytes (NOT free_string, which would strlen).
function _take_bytes_uptr(ptr::Ptr{UInt8}, len::Csize_t, code::Int32, op::String)
    ptr == C_NULL && throw(PdfOxideError(code, op))
    n = Int(len)
    out = copy(unsafe_wrap(Array, ptr, n < 0 ? 0 : n))
    ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), ptr)
    return out
end

"""Open a PDF for editing from a filesystem path."""
function open_editor(path::AbstractString)
    code = Ref{Int32}(0)
    h = ccall((:document_editor_open, LIB), Ptr{Cvoid}, (Cstring, Ref{Int32}), path, code)
    h == C_NULL && throw(PdfOxideError(code[], "open_editor"))
    return DocumentEditor(h)
end

"""Open a PDF for editing from an in-memory byte vector."""
function open_editor_from_bytes(data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    h = ccall(
        (:document_editor_open_from_bytes, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Csize_t, Ref{Int32}),
        data,
        Csize_t(length(data)),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "open_editor_from_bytes"))
    return DocumentEditor(h)
end

"""Whether the editor has unsaved modifications (bool)."""
is_modified(e::DocumentEditor) =
    ccall((:document_editor_is_modified, LIB), Bool, (Ptr{Cvoid},), _editor(e))

"""Source path the editor was opened from."""
function get_source_path(e::DocumentEditor)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_get_source_path, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    return _take_string(ptr, code[], "get_source_path")
end

"""PDF version as `(major, minor)`."""
function version(e::DocumentEditor)
    maj = Ref{UInt8}(0)
    min = Ref{UInt8}(0)
    ccall(
        (:document_editor_get_version, LIB),
        Cvoid,
        (Ptr{Cvoid}, Ref{UInt8}, Ref{UInt8}),
        _editor(e),
        maj,
        min,
    )
    return PdfVersion(Int(maj[]), Int(min[]))
end

"""Number of pages."""
function page_count(e::DocumentEditor)
    code = Ref{Int32}(0)
    n = ccall(
        (:document_editor_get_page_count, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    n < 0 && throw(PdfOxideError(code[], "page_count"))
    return Int(n)
end

"""Producer from `/Info.Producer`."""
function get_producer(e::DocumentEditor)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_get_producer, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    return _take_string(ptr, code[], "get_producer")
end

"""Set the `/Info.Producer` value."""
function set_producer(e::DocumentEditor, value::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_set_producer, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _editor(e),
        value,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "set_producer"))
    return nothing
end

"""Creation date from `/Info.CreationDate` (raw PDF date string)."""
function get_creation_date(e::DocumentEditor)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_get_creation_date, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    return _take_string(ptr, code[], "get_creation_date")
end

"""Set the `/Info.CreationDate` value (raw PDF date string)."""
function set_creation_date(e::DocumentEditor, date_str::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_set_creation_date, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _editor(e),
        date_str,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "set_creation_date"))
    return nothing
end

"""Save the edited document to a filesystem path."""
function save(e::DocumentEditor, path::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_save, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _editor(e),
        path,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "save"))
    return nothing
end

"""Serialize the edited document to a `Vector{UInt8}`."""
function save_to_bytes(e::DocumentEditor)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_save_to_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Csize_t}, Ref{Int32}),
        _editor(e),
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "save_to_bytes")
end

"""Serialize with compress / garbage-collect / linearize options."""
function save_to_bytes_with_options(
    e::DocumentEditor,
    compress::Bool,
    garbage_collect::Bool,
    linearize::Bool,
)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_save_to_bytes_with_options, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Bool, Bool, Bool, Ref{Csize_t}, Ref{Int32}),
        _editor(e),
        compress,
        garbage_collect,
        linearize,
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "save_to_bytes_with_options")
end

"""Extract a subset of (0-based) `pages` to a new in-memory PDF (`Vector{UInt8}`)."""
function extract_pages_to_bytes(e::DocumentEditor, pages::AbstractVector{<:Integer})
    arr = Int32[Int32(p) for p in pages]
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_extract_pages_to_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ptr{Int32}, Csize_t, Ref{Csize_t}, Ref{Int32}),
        _editor(e),
        arr,
        Csize_t(length(arr)),
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "extract_pages_to_bytes")
end

"""Convert to PDF/A in-place. `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u."""
function convert_to_pdf_a(e::DocumentEditor, level::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_convert_to_pdf_a, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(level),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "convert_to_pdf_a"))
    return nothing
end

"""Save with AES-256 encryption to a `Vector{UInt8}`."""
function save_encrypted_to_bytes(
    e::DocumentEditor,
    user_password::AbstractString,
    owner_password::AbstractString,
)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_save_encrypted_to_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Cstring, Cstring, Ref{Csize_t}, Ref{Int32}),
        _editor(e),
        user_password,
        owner_password,
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "save_encrypted_to_bytes")
end

"""Save with AES-256 encryption to a filesystem path."""
function save_encrypted(
    e::DocumentEditor,
    path::AbstractString,
    user_password::AbstractString,
    owner_password::AbstractString,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_save_encrypted, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Cstring, Cstring, Ref{Int32}),
        _editor(e),
        path,
        user_password,
        owner_password,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "save_encrypted"))
    return nothing
end

"""Merge pages from an in-memory PDF byte buffer into this document."""
function merge_from_bytes(e::DocumentEditor, data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_merge_from_bytes, LIB),
        Int32,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Int32}),
        _editor(e),
        data,
        Csize_t(length(data)),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "merge_from_bytes"))
    return nothing
end

"""Merge pages from a PDF on disk into this document."""
function merge_from(e::DocumentEditor, source_path::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_merge_from, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _editor(e),
        source_path,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "merge_from"))
    return nothing
end

"""Embed a file attachment (`name`, `data` bytes) into the document."""
function embed_file(e::DocumentEditor, name::AbstractString, data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_embed_file, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ptr{UInt8}, Csize_t, Ref{Int32}),
        _editor(e),
        name,
        data,
        Csize_t(length(data)),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "embed_file"))
    return nothing
end

"""Apply (burn in) redactions on a single (0-based) page."""
function apply_page_redactions(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_apply_page_redactions, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "apply_page_redactions"))
    return nothing
end

"""Apply all pending redactions across the document."""
function apply_all_redactions(e::DocumentEditor)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_apply_all_redactions, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "apply_all_redactions"))
    return nothing
end

"""Rotate all pages by `degrees` (relative)."""
function rotate_all_pages(e::DocumentEditor, degrees::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_rotate_all_pages, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(degrees),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "rotate_all_pages"))
    return nothing
end

"""Rotate a single (0-based) page by `degrees` (additive)."""
function rotate_page_by(e::DocumentEditor, page::Integer, degrees::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_rotate_page_by, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Int32, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        Int32(degrees),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "rotate_page_by"))
    return nothing
end

"""Absolute rotation (degrees) of a (0-based) page."""
function get_page_rotation(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    v = ccall(
        (:document_editor_get_page_rotation, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(page),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "get_page_rotation"))
    return Int(v)
end

"""Set the absolute rotation (degrees) of a (0-based) page."""
function set_page_rotation(e::DocumentEditor, page::Integer, degrees::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_set_page_rotation, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Int32, Ref{Int32}),
        _editor(e),
        Int32(page),
        Int32(degrees),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "set_page_rotation"))
    return nothing
end

"""Delete a (0-based) page."""
function delete_page(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_delete_page, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "delete_page"))
    return nothing
end

"""Move a page from (0-based) `from` to `to`."""
function move_page(e::DocumentEditor, from::Integer, to::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_move_page, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Int32, Ref{Int32}),
        _editor(e),
        Int32(from),
        Int32(to),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "move_page"))
    return nothing
end

# MediaBox/CropBox getters return a Bbox via double out-params. Generated with
# @eval so each ccall references its C function name as a LITERAL symbol.
for (jl_fn, c_fn) in (
    (:get_page_media_box, :document_editor_get_page_media_box),
    (:get_page_crop_box, :document_editor_get_page_crop_box),
)
    op = String(jl_fn)
    @eval function $jl_fn(e::DocumentEditor, page::Integer)
        x = Ref{Float64}(0)
        y = Ref{Float64}(0)
        w = Ref{Float64}(0)
        h = Ref{Float64}(0)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (
                Ptr{Cvoid},
                Csize_t,
                Ref{Float64},
                Ref{Float64},
                Ref{Float64},
                Ref{Float64},
                Ref{Int32},
            ),
            _editor(e),
            Csize_t(page),
            x,
            y,
            w,
            h,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return Bbox(x[], y[], w[], h[])
    end
end

# MediaBox/CropBox setters take a Bbox's components. Generated with @eval
# (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:set_page_media_box, :document_editor_set_page_media_box),
    (:set_page_crop_box, :document_editor_set_page_crop_box),
)
    op = String(jl_fn)
    @eval function $jl_fn(
        e::DocumentEditor,
        page::Integer,
        x::Real,
        y::Real,
        w::Real,
        h::Real,
    )
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Csize_t, Float64, Float64, Float64, Float64, Ref{Int32}),
            _editor(e),
            Csize_t(page),
            Float64(x),
            Float64(y),
            Float64(w),
            Float64(h),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return nothing
    end
end

"""Crop all pages by `left`/`right`/`top`/`bottom` margins (page user-space)."""
function crop_margins(e::DocumentEditor, left::Real, right::Real, top::Real, bottom::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_crop_margins, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Ref{Int32}),
        _editor(e),
        Float32(left),
        Float32(right),
        Float32(top),
        Float32(bottom),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "crop_margins"))
    return nothing
end

"""Erase one rectangular region (floats) on a (0-based) page."""
function erase_region(e::DocumentEditor, page::Integer, x::Real, y::Real, w::Real, h::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_erase_region, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Float32, Float32, Float32, Float32, Ref{Int32}),
        _editor(e),
        Int32(page),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "erase_region"))
    return nothing
end

"""
Erase multiple rectangular regions on a (0-based) page. `rects` is a vector of
`(x, y, w, h)` tuples, flattened to a contiguous `Float64` quad array.
"""
function erase_regions(
    e::DocumentEditor,
    page::Integer,
    rects::AbstractVector{<:NTuple{4,<:Real}},
)
    flat = Vector{Float64}(undef, 4 * length(rects))
    for (i, r) in enumerate(rects)
        base = 4 * (i - 1)
        flat[base+1] = Float64(r[1])
        flat[base+2] = Float64(r[2])
        flat[base+3] = Float64(r[3])
        flat[base+4] = Float64(r[4])
    end
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_erase_regions, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        flat,
        Csize_t(length(rects)),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "erase_regions"))
    return nothing
end

"""Clear all pending erase-region entries for a (0-based) page."""
function clear_erase_regions(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_clear_erase_regions, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "clear_erase_regions"))
    return nothing
end

"""Whether a (0-based) page is marked for annotation-flatten (bool)."""
function is_page_marked_for_flatten(e::DocumentEditor, page::Integer)
    rc = ccall(
        (:document_editor_is_page_marked_for_flatten, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t),
        _editor(e),
        Csize_t(page),
    )
    rc < 0 && throw(PdfOxideError(rc, "is_page_marked_for_flatten"))
    return rc == 1
end

"""Remove the flatten mark from a (0-based) page."""
function unmark_page_for_flatten(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_unmark_page_for_flatten, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "unmark_page_for_flatten"))
    return nothing
end

"""Whether a (0-based) page is marked for redaction (bool)."""
function is_page_marked_for_redaction(e::DocumentEditor, page::Integer)
    rc = ccall(
        (:document_editor_is_page_marked_for_redaction, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t),
        _editor(e),
        Csize_t(page),
    )
    rc < 0 && throw(PdfOxideError(rc, "is_page_marked_for_redaction"))
    return rc == 1
end

"""Remove the redaction mark from a (0-based) page."""
function unmark_page_for_redaction(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_unmark_page_for_redaction, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "unmark_page_for_redaction"))
    return nothing
end

"""Flatten annotations on a single (0-based) page."""
function flatten_annotations(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_flatten_annotations, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "flatten_annotations"))
    return nothing
end

"""Flatten annotations on all pages."""
function flatten_all_annotations(e::DocumentEditor)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_flatten_all_annotations, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "flatten_all_annotations"))
    return nothing
end

"""Set a form field value (UTF-8) on the document."""
function set_form_field_value(
    e::DocumentEditor,
    name::AbstractString,
    value::AbstractString,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_set_form_field_value, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Cstring, Ref{Int32}),
        _editor(e),
        name,
        value,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "set_form_field_value"))
    return nothing
end

"""Flatten all forms (bake form values into page content)."""
function flatten_forms(e::DocumentEditor)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_flatten_forms, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "flatten_forms"))
    return nothing
end

"""Flatten forms on a single (0-based) page."""
function flatten_forms_on_page(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:document_editor_flatten_forms_on_page, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(page),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "flatten_forms_on_page"))
    return nothing
end

"""Number of warnings collected during the last form-flattening save."""
function flatten_warnings_count(e::DocumentEditor)
    n = ccall(
        (:document_editor_flatten_warnings_count, LIB),
        Int32,
        (Ptr{Cvoid},),
        _editor(e),
    )
    n < 0 && throw(PdfOxideError(n, "flatten_warnings_count"))
    return Int(n)
end

"""The `index`-th (0-based) flatten warning string."""
function flatten_warning(e::DocumentEditor, index::Integer)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:document_editor_flatten_warning, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _editor(e),
        Int32(index),
        code,
    )
    return _take_string(ptr, code[], "flatten_warning")
end

# ── PDF creation builder ──────────────────────────────────────────────────────
# Three owned native handles over the C ABI's builder API, mirroring the
# PdfDocument/Pdf/DocumentEditor pattern: each wraps an owned native pointer,
# uses the same PdfOxideError helpers, _take_string / free_bytes byte-take, and a
# closed-handle guard; freed on close!/finalization. Methods use snake_case
# (Julia idiom); page indices are 0-based.
#
#   EmbeddedFont    — a loaded TTF/OTF font handle (pdf_embedded_font_*).
#   DocumentBuilder — accumulates metadata, fonts and pages (pdf_document_builder_*).
#   PageBuilder     — fluent page content ops (pdf_page_builder_*).
#
# Ownership note: register_embedded_font CONSUMES the font handle on success — the
# wrapper nulls its pointer so close!/finalizer will not double-free it. Likewise
# done() CONSUMES the page handle (wrapper nulled). build()/save() consume the
# builder STATE but not the wrapper handle, which must still be freed.

# EmbeddedFont — owned font handle. Freed via pdf_embedded_font_free UNLESS a
# successful register_embedded_font took ownership (then handle is nulled here).
mutable struct EmbeddedFont
    handle::Ptr{Cvoid}
    function EmbeddedFont(h::Ptr{Cvoid})
        f = new(h)
        finalizer(close!, f)
        return f
    end
end

"""Free the native font handle now (idempotent; no-op once the builder took ownership)."""
function close!(f::EmbeddedFont)
    if f.handle != C_NULL
        ccall((:pdf_embedded_font_free, LIB), Cvoid, (Ptr{Cvoid},), f.handle)
        f.handle = C_NULL
    end
    return nothing
end

_font(f::EmbeddedFont) = (f.handle == C_NULL && error("EmbeddedFont is closed"); f.handle)

"""Load a TTF/OTF font from a filesystem path."""
function embedded_font_from_file(path::AbstractString)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_embedded_font_from_file, LIB),
        Ptr{Cvoid},
        (Cstring, Ref{Int32}),
        path,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "embedded_font_from_file"))
    return EmbeddedFont(h)
end

"""Load a font from a byte buffer. `name` may be empty to use the PostScript name."""
function embedded_font_from_bytes(
    data::AbstractVector{UInt8};
    name::Union{Nothing,AbstractString} = nothing,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_embedded_font_from_bytes, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Csize_t, Cstring, Ref{Int32}),
        data,
        Csize_t(length(data)),
        name === nothing ? C_NULL : name,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "embedded_font_from_bytes"))
    return EmbeddedFont(h)
end

# DocumentBuilder — accumulates metadata/fonts/pages then builds the PDF.
mutable struct DocumentBuilder
    handle::Ptr{Cvoid}
    function DocumentBuilder(h::Ptr{Cvoid})
        b = new(h)
        finalizer(close!, b)
        return b
    end
end

"""Free the native builder handle now (idempotent; also runs at finalization)."""
function close!(b::DocumentBuilder)
    if b.handle != C_NULL
        ccall((:pdf_document_builder_free, LIB), Cvoid, (Ptr{Cvoid},), b.handle)
        b.handle = C_NULL
    end
    return nothing
end

_builder(b::DocumentBuilder) =
    (b.handle == C_NULL && error("DocumentBuilder is closed"); b.handle)

"""Create a new in-memory PDF builder."""
function DocumentBuilder()
    code = Ref{Int32}(0)
    h = ccall((:pdf_document_builder_create, LIB), Ptr{Cvoid}, (Ref{Int32},), code)
    h == C_NULL && throw(PdfOxideError(code[], "DocumentBuilder"))
    return DocumentBuilder(h)
end

# Single-string-arg metadata setters — one per C function, generated with @eval
# so each ccall references its C function name as a LITERAL symbol.
for (jl_fn, c_fn) in (
    (:set_title, :pdf_document_builder_set_title),
    (:set_author, :pdf_document_builder_set_author),
    (:set_subject, :pdf_document_builder_set_subject),
    (:set_keywords, :pdf_document_builder_set_keywords),
    (:set_creator, :pdf_document_builder_set_creator),
    (:on_open, :pdf_document_builder_on_open),
    (:language, :pdf_document_builder_language),
)
    op = String(jl_fn)
    @eval function $jl_fn(b::DocumentBuilder, value::AbstractString)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Cstring, Ref{Int32}),
            _builder(b),
            value,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return b
    end
end

"""Enable PDF/UA-1 tagged-PDF mode (opt-in)."""
function tagged_pdf_ua1(b::DocumentBuilder)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_builder_tagged_pdf_ua1, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _builder(b),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "tagged_pdf_ua1"))
    return b
end

"""Add a role-map entry mapping a `custom` structure type to a `standard` one."""
function role_map(b::DocumentBuilder, custom::AbstractString, standard::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_builder_role_map, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Cstring, Ref{Int32}),
        _builder(b),
        custom,
        standard,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "role_map"))
    return b
end

"""
Register a loaded `EmbeddedFont` under `name`. **Consumes** the font on success —
the wrapper's handle is nulled so it will not be double-freed.
"""
function register_embedded_font(b::DocumentBuilder, name::AbstractString, f::EmbeddedFont)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_builder_register_embedded_font, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Ref{Int32}),
        _builder(b),
        name,
        _font(f),
        code,
    )
    if rc != 0 || code[] != 0
        # On error the font is NOT consumed; leave the wrapper intact to free later.
        throw(PdfOxideError(code[], "register_embedded_font"))
    end
    # Success: the builder took ownership; do NOT free the font handle again.
    f.handle = C_NULL
    return b
end

# Page-openers — one per entry point (NULL on error -> throw). Each opens a new
# PageBuilder bound to this builder.
for (jl_fn, c_fn) in (
    (:a4_page, :pdf_document_builder_a4_page),
    (:letter_page, :pdf_document_builder_letter_page),
)
    op = String(jl_fn)
    @eval function $jl_fn(b::DocumentBuilder)
        code = Ref{Int32}(0)
        h = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Ref{Int32}),
            _builder(b),
            code,
        )
        h == C_NULL && throw(PdfOxideError(code[], $op))
        return PageBuilder(h)
    end
end

"""Start a page with custom dimensions in PDF points (72 pt = 1 inch)."""
function page(b::DocumentBuilder, width::Real, height::Real)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_document_builder_page, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Float32, Float32, Ref{Int32}),
        _builder(b),
        Float32(width),
        Float32(height),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "page"))
    return PageBuilder(h)
end

"""Build the PDF and return the bytes (`Vector{UInt8}`). The builder must still be closed."""
function build(b::DocumentBuilder)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_builder_build, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Csize_t}, Ref{Int32}),
        _builder(b),
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "build")
end

"""Build and save the PDF to `path`."""
function save(b::DocumentBuilder, path::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_builder_save, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _builder(b),
        path,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "save"))
    return nothing
end

"""Build and save with AES-256 encryption to `path`."""
function save_encrypted_builder(
    b::DocumentBuilder,
    path::AbstractString,
    user_password::AbstractString,
    owner_password::AbstractString,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_builder_save_encrypted, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Cstring, Cstring, Ref{Int32}),
        _builder(b),
        path,
        user_password,
        owner_password,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "save_encrypted_builder"))
    return nothing
end

"""Build encrypted bytes (AES-256) and return them (`Vector{UInt8}`)."""
function to_bytes_encrypted(
    b::DocumentBuilder,
    user_password::AbstractString,
    owner_password::AbstractString,
)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_builder_to_bytes_encrypted, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Cstring, Cstring, Ref{Csize_t}, Ref{Int32}),
        _builder(b),
        user_password,
        owner_password,
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "to_bytes_encrypted")
end

# PageBuilder — fluent page content ops. Owns its native page handle; freed via
# pdf_page_builder_free UNLESS done() consumed it (then handle is nulled here).
mutable struct PageBuilder
    handle::Ptr{Cvoid}
    function PageBuilder(h::Ptr{Cvoid})
        p = new(h)
        finalizer(close!, p)
        return p
    end
end

"""Drop the native page handle now (idempotent; no-op once done() consumed it)."""
function close!(p::PageBuilder)
    if p.handle != C_NULL
        ccall((:pdf_page_builder_free, LIB), Cvoid, (Ptr{Cvoid},), p.handle)
        p.handle = C_NULL
    end
    return nothing
end

_pagebuilder(p::PageBuilder) =
    (p.handle == C_NULL && error("PageBuilder is closed"); p.handle)

# No-arg fluent ops (status int + error_code) — generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:horizontal_rule, :pdf_page_builder_horizontal_rule),
    (:newline, :pdf_page_builder_newline),
    (:watermark_confidential, :pdf_page_builder_watermark_confidential),
    (:watermark_draft, :pdf_page_builder_watermark_draft),
    (:new_page_same_size, :pdf_page_builder_new_page_same_size),
    (:streaming_table_flush, :pdf_page_builder_streaming_table_flush),
    (:streaming_table_finish, :pdf_page_builder_streaming_table_finish),
)
    op = String(jl_fn)
    @eval function $jl_fn(p::PageBuilder)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Ref{Int32}),
            _pagebuilder(p),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return p
    end
end

# Single-string-arg fluent ops — generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:text, :pdf_page_builder_text),
    (:paragraph, :pdf_page_builder_paragraph),
    (:link_url, :pdf_page_builder_link_url),
    (:link_named, :pdf_page_builder_link_named),
    (:link_javascript, :pdf_page_builder_link_javascript),
    (:on_open, :pdf_page_builder_on_open),
    (:on_close, :pdf_page_builder_on_close),
    (:field_keystroke, :pdf_page_builder_field_keystroke),
    (:field_format, :pdf_page_builder_field_format),
    (:field_validate, :pdf_page_builder_field_validate),
    (:field_calculate, :pdf_page_builder_field_calculate),
    (:sticky_note, :pdf_page_builder_sticky_note),
    (:watermark, :pdf_page_builder_watermark),
    (:stamp, :pdf_page_builder_stamp),
    (:inline, :pdf_page_builder_inline),
    (:inline_bold, :pdf_page_builder_inline_bold),
    (:inline_italic, :pdf_page_builder_inline_italic),
)
    op = String(jl_fn)
    @eval function $jl_fn(p::PageBuilder, value::AbstractString)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Cstring, Ref{Int32}),
            _pagebuilder(p),
            value,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return p
    end
end

# RGB-triplet fluent ops (r, g, b) — generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in (
    (:highlight, :pdf_page_builder_highlight),
    (:underline, :pdf_page_builder_underline),
    (:strikeout, :pdf_page_builder_strikeout),
    (:squiggly, :pdf_page_builder_squiggly),
)
    op = String(jl_fn)
    @eval function $jl_fn(p::PageBuilder, r::Real, g::Real, b_::Real)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Float32, Float32, Float32, Ref{Int32}),
            _pagebuilder(p),
            Float32(r),
            Float32(g),
            Float32(b_),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return p
    end
end

# Single-float-arg fluent ops — generated with @eval (LITERAL ccall symbol).
for (jl_fn, c_fn) in ((:space, :pdf_page_builder_space),)
    op = String(jl_fn)
    @eval function $jl_fn(p::PageBuilder, value::Real)
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Float32, Ref{Int32}),
            _pagebuilder(p),
            Float32(value),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], $op))
        return p
    end
end

"""Set the font + size for subsequent text on this page."""
function font(p::PageBuilder, name::AbstractString, size::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_font, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Ref{Int32}),
        _pagebuilder(p),
        name,
        Float32(size),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "font"))
    return p
end

"""Move the cursor to absolute `(x, y)` (PDF points, from lower-left)."""
function at(p::PageBuilder, x::Real, y::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_at, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "at"))
    return p
end

"""Emit a heading at the given `level` (1–6) with `text`."""
function heading(p::PageBuilder, level::Integer, text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_heading, LIB),
        Int32,
        (Ptr{Cvoid}, UInt8, Cstring, Ref{Int32}),
        _pagebuilder(p),
        UInt8(level),
        text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "heading"))
    return p
end

"""Link the previous text to an internal (0-based) page index."""
function link_page(p::PageBuilder, page_index::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_link_page, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _pagebuilder(p),
        Csize_t(page_index),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "link_page"))
    return p
end

"""Place a free-standing sticky note at absolute `(x, y)` with `text`."""
function sticky_note_at(p::PageBuilder, x::Real, y::Real, text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_sticky_note_at, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "sticky_note_at"))
    return p
end

"""Place a free-flowing text annotation inside rect `(x, y, w, h)`."""
function freetext(p::PageBuilder, x::Real, y::Real, w::Real, h::Real, text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_freetext, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "freetext"))
    return p
end

"""Add a single-line text form field. `default_value` may be empty for blank."""
function text_field(
    p::PageBuilder,
    name::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real;
    default_value::Union{Nothing,AbstractString} = nothing,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_text_field, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Float32, Float32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        name,
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        default_value === nothing ? C_NULL : default_value,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "text_field"))
    return p
end

"""Add a checkbox form field. `checked` toggles the initially-ticked state."""
function checkbox(
    p::PageBuilder,
    name::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    checked::Bool,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_checkbox, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Float32, Float32, Float32, Int32, Ref{Int32}),
        _pagebuilder(p),
        name,
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        Int32(checked ? 1 : 0),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "checkbox"))
    return p
end

"""Add a dropdown combo-box with a fixed list of `options`. `selected` may be empty."""
function combo_box(
    p::PageBuilder,
    name::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    options::AbstractVector{<:AbstractString};
    selected::Union{Nothing,AbstractString} = nothing,
)
    # Marshal the Julia String list to a C array of NUL-terminated pointers.
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in options]
    GC.@preserve options cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_combo_box, LIB),
            Int32,
            (
                Ptr{Cvoid},
                Cstring,
                Float32,
                Float32,
                Float32,
                Float32,
                Ptr{Cstring},
                Csize_t,
                Cstring,
                Ref{Int32},
            ),
            _pagebuilder(p),
            name,
            Float32(x),
            Float32(y),
            Float32(w),
            Float32(h),
            cstrs,
            Csize_t(length(cstrs)),
            selected === nothing ? C_NULL : selected,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "combo_box"))
    end
    return p
end

"""
Add a radio-button group. `values`/`xs`/`ys`/`ws`/`hs` are parallel arrays
describing each button's export value and rect. `selected` may be empty.
"""
function radio_group(
    p::PageBuilder,
    name::AbstractString,
    values::AbstractVector{<:AbstractString},
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ws::AbstractVector{<:Real},
    hs::AbstractVector{<:Real};
    selected::Union{Nothing,AbstractString} = nothing,
)
    n = length(values)
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in values]
    fxs = Float32[Float32(v) for v in xs]
    fys = Float32[Float32(v) for v in ys]
    fws = Float32[Float32(v) for v in ws]
    fhs = Float32[Float32(v) for v in hs]
    GC.@preserve values cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_radio_group, LIB),
            Int32,
            (
                Ptr{Cvoid},
                Cstring,
                Ptr{Cstring},
                Ptr{Float32},
                Ptr{Float32},
                Ptr{Float32},
                Ptr{Float32},
                Csize_t,
                Cstring,
                Ref{Int32},
            ),
            _pagebuilder(p),
            name,
            cstrs,
            fxs,
            fys,
            fws,
            fhs,
            Csize_t(n),
            selected === nothing ? C_NULL : selected,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "radio_group"))
    end
    return p
end

"""Add a clickable push button with a visible `caption`."""
function push_button(
    p::PageBuilder,
    name::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    caption::AbstractString,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_push_button, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Float32, Float32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        name,
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        caption,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "push_button"))
    return p
end

"""Add an unsigned signature placeholder field at rect `(x, y, w, h)`."""
function signature_field(
    p::PageBuilder,
    name::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_signature_field, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        name,
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "signature_field"))
    return p
end

"""Add a footnote: `ref_mark` inline superscript + `note_text` body at page end."""
function footnote(p::PageBuilder, ref_mark::AbstractString, note_text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_footnote, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Cstring, Ref{Int32}),
        _pagebuilder(p),
        ref_mark,
        note_text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "footnote"))
    return p
end

"""Lay out `text` across `column_count` balanced columns with `gap_pt` between them."""
function columns(p::PageBuilder, column_count::Integer, gap_pt::Real, text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_columns, LIB),
        Int32,
        (Ptr{Cvoid}, UInt32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        UInt32(column_count),
        Float32(gap_pt),
        text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "columns"))
    return p
end

"""Inline colored run (RGB 0.0–1.0) of `text`."""
function inline_color(p::PageBuilder, r::Real, g::Real, b_::Real, text::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_inline_color, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Cstring, Ref{Int32}),
        _pagebuilder(p),
        Float32(r),
        Float32(g),
        Float32(b_),
        text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "inline_color"))
    return p
end

"""
Place a 1-D barcode image. `barcode_type`: 0=Code128 1=Code39 2=EAN13 3=EAN8
4=UPCA 5=ITF 6=Code93 7=Codabar.
"""
function barcode_1d(
    p::PageBuilder,
    barcode_type::Integer,
    data::AbstractString,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_barcode_1d, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Cstring, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        Int32(barcode_type),
        data,
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "barcode_1d"))
    return p
end

"""Place a QR-code image (square `size × size` points)."""
function barcode_qr(p::PageBuilder, data::AbstractString, x::Real, y::Real, size::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_barcode_qr, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        data,
        Float32(x),
        Float32(y),
        Float32(size),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "barcode_qr"))
    return p
end

"""Embed an image at rect `(x, y, w, h)`. `bytes` is raw JPEG/PNG/WebP data."""
function image(
    p::PageBuilder,
    bytes::AbstractVector{UInt8},
    x::Real,
    y::Real,
    w::Real,
    h::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_image, LIB),
        Int32,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        bytes,
        Csize_t(length(bytes)),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "image"))
    return p
end

"""Embed an image at rect `(x, y, w, h)` with accessibility `alt_text`."""
function image_with_alt(
    p::PageBuilder,
    bytes::AbstractVector{UInt8},
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    alt_text::AbstractString,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_image_with_alt, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Ptr{UInt8},
            Csize_t,
            Float32,
            Float32,
            Float32,
            Float32,
            Cstring,
            Ref{Int32},
        ),
        _pagebuilder(p),
        bytes,
        Csize_t(length(bytes)),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        alt_text,
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "image_with_alt"))
    return p
end

"""Embed a decorative image at rect `(x, y, w, h)` as an /Artifact (no alt text)."""
function image_artifact(
    p::PageBuilder,
    bytes::AbstractVector{UInt8},
    x::Real,
    y::Real,
    w::Real,
    h::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_image_artifact, LIB),
        Int32,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        bytes,
        Csize_t(length(bytes)),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "image_artifact"))
    return p
end

"""Draw a stroked rectangle outline (1pt black) at `(x, y, w, h)`."""
function rect(p::PageBuilder, x::Real, y::Real, w::Real, h::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_rect, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "rect"))
    return p
end

"""Draw a filled rectangle at `(x, y, w, h)` in RGB colour (channels 0–1)."""
function filled_rect(
    p::PageBuilder,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    r::Real,
    g::Real,
    b_::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_filled_rect, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Ref{Int32},
        ),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        Float32(r),
        Float32(g),
        Float32(b_),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "filled_rect"))
    return p
end

"""Draw a 1pt black line from `(x1, y1)` to `(x2, y2)`."""
function line(p::PageBuilder, x1::Real, y1::Real, x2::Real, y2::Real)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_line, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Ref{Int32}),
        _pagebuilder(p),
        Float32(x1),
        Float32(y1),
        Float32(x2),
        Float32(y2),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "line"))
    return p
end

"""Stroke a rectangle at `(x, y, w, h)` with `width`pt stroke in RGB colour (0–1)."""
function stroke_rect(
    p::PageBuilder,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    width::Real,
    r::Real,
    g::Real,
    b_::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_stroke_rect, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Ref{Int32},
        ),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        Float32(width),
        Float32(r),
        Float32(g),
        Float32(b_),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "stroke_rect"))
    return p
end

"""Stroke a line from `(x1, y1)` to `(x2, y2)` with `width`pt stroke in RGB (0–1)."""
function stroke_line(
    p::PageBuilder,
    x1::Real,
    y1::Real,
    x2::Real,
    y2::Real,
    width::Real,
    r::Real,
    g::Real,
    b_::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_stroke_line, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Ref{Int32},
        ),
        _pagebuilder(p),
        Float32(x1),
        Float32(y1),
        Float32(x2),
        Float32(y2),
        Float32(width),
        Float32(r),
        Float32(g),
        Float32(b_),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "stroke_line"))
    return p
end

"""Stroke a dashed rectangle. `dash_array` is alternating on/off lengths (pt); `phase` is the offset."""
function stroke_rect_dashed(
    p::PageBuilder,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    width::Real,
    r::Real,
    g::Real,
    b_::Real,
    dash_array::AbstractVector{<:Real},
    phase::Real,
)
    dash = Float32[Float32(v) for v in dash_array]
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_stroke_rect_dashed, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Ptr{Float32},
            Csize_t,
            Float32,
            Ref{Int32},
        ),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        Float32(width),
        Float32(r),
        Float32(g),
        Float32(b_),
        isempty(dash) ? C_NULL : dash,
        Csize_t(length(dash)),
        Float32(phase),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "stroke_rect_dashed"))
    return p
end

"""Stroke a dashed line. `dash_array` is alternating on/off lengths (pt); `phase` is the offset."""
function stroke_line_dashed(
    p::PageBuilder,
    x1::Real,
    y1::Real,
    x2::Real,
    y2::Real,
    width::Real,
    r::Real,
    g::Real,
    b_::Real,
    dash_array::AbstractVector{<:Real},
    phase::Real,
)
    dash = Float32[Float32(v) for v in dash_array]
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_stroke_line_dashed, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Ptr{Float32},
            Csize_t,
            Float32,
            Ref{Int32},
        ),
        _pagebuilder(p),
        Float32(x1),
        Float32(y1),
        Float32(x2),
        Float32(y2),
        Float32(width),
        Float32(r),
        Float32(g),
        Float32(b_),
        isempty(dash) ? C_NULL : dash,
        Csize_t(length(dash)),
        Float32(phase),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "stroke_line_dashed"))
    return p
end

"""Draw `text` inside rect `(x, y, w, h)` with `align` (0=Left, 1=Center, 2=Right)."""
function text_in_rect(
    p::PageBuilder,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
    text::AbstractString,
    align::Integer,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_text_in_rect, LIB),
        Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Cstring, Int32, Ref{Int32}),
        _pagebuilder(p),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        text,
        Int32(align),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "text_in_rect"))
    return p
end

"""
Emit a buffered table. `aligns` encode 0=Left/1=Center/2=Right; `cell_strings`
is a row-major matrix (`n_rows × n_columns`). `has_header` promotes the first row.
"""
function table(
    p::PageBuilder,
    n_columns::Integer,
    widths::AbstractVector{<:Real},
    aligns::AbstractVector{<:Integer},
    n_rows::Integer,
    cell_strings::AbstractMatrix{<:AbstractString},
    has_header::Bool,
)
    fw = Float32[Float32(v) for v in widths]
    al = Int32[Int32(v) for v in aligns]
    # Flatten the matrix row-major: cell_strings[row*n_columns + col].
    flat = Vector{String}(undef, Int(n_rows) * Int(n_columns))
    for r = 0:(Int(n_rows)-1), c = 0:(Int(n_columns)-1)
        flat[r*Int(n_columns)+c+1] = String(cell_strings[r+1, c+1])
    end
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in flat]
    GC.@preserve flat cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_table, LIB),
            Int32,
            (
                Ptr{Cvoid},
                Csize_t,
                Ptr{Float32},
                Ptr{Int32},
                Csize_t,
                Ptr{Cstring},
                Int32,
                Ref{Int32},
            ),
            _pagebuilder(p),
            Csize_t(n_columns),
            fw,
            al,
            Csize_t(n_rows),
            cstrs,
            Int32(has_header ? 1 : 0),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "table"))
    end
    return p
end

"""
Open a streaming table. `headers`/`widths`/`aligns` are parallel length-`n_columns`
arrays (aligns: 0=Left/1=Center/2=Right). `repeat_header` repeats the header per page.
"""
function streaming_table_begin(
    p::PageBuilder,
    headers::AbstractVector{<:AbstractString},
    widths::AbstractVector{<:Real},
    aligns::AbstractVector{<:Integer},
    repeat_header::Bool,
)
    n = length(headers)
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in headers]
    fw = Float32[Float32(v) for v in widths]
    al = Int32[Int32(v) for v in aligns]
    GC.@preserve headers cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_streaming_table_begin, LIB),
            Int32,
            (
                Ptr{Cvoid},
                Csize_t,
                Ptr{Cstring},
                Ptr{Float32},
                Ptr{Int32},
                Int32,
                Ref{Int32},
            ),
            _pagebuilder(p),
            Csize_t(n),
            cstrs,
            fw,
            al,
            Int32(repeat_header ? 1 : 0),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "streaming_table_begin"))
    end
    return p
end

"""
Open a streaming table with a column-width `mode` (0=Fixed, 1=Sample, 2=AutoAll).
`max_rowspan` ≥2 enables rowspans; 0/1 disables.
"""
function streaming_table_begin_v2(
    p::PageBuilder,
    headers::AbstractVector{<:AbstractString},
    widths::AbstractVector{<:Real},
    aligns::AbstractVector{<:Integer},
    repeat_header::Bool,
    mode::Integer,
    sample_rows::Integer,
    min_col_width_pt::Real,
    max_col_width_pt::Real,
    max_rowspan::Integer,
)
    n = length(headers)
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in headers]
    fw = Float32[Float32(v) for v in widths]
    al = Int32[Int32(v) for v in aligns]
    GC.@preserve headers cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_streaming_table_begin_v2, LIB),
            Int32,
            (
                Ptr{Cvoid},
                Csize_t,
                Ptr{Cstring},
                Ptr{Float32},
                Ptr{Int32},
                Int32,
                Int32,
                Csize_t,
                Float32,
                Float32,
                Csize_t,
                Ref{Int32},
            ),
            _pagebuilder(p),
            Csize_t(n),
            cstrs,
            fw,
            al,
            Int32(repeat_header ? 1 : 0),
            Int32(mode),
            Csize_t(sample_rows),
            Float32(min_col_width_pt),
            Float32(max_col_width_pt),
            Csize_t(max_rowspan),
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "streaming_table_begin_v2"))
    end
    return p
end

"""Set the batch size for the currently-open streaming table (0 defaults to 256)."""
function streaming_table_set_batch_size(p::PageBuilder, batch_size::Integer)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_streaming_table_set_batch_size, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _pagebuilder(p),
        Csize_t(batch_size),
        code,
    )
    (rc != 0 || code[] != 0) &&
        throw(PdfOxideError(code[], "streaming_table_set_batch_size"))
    return p
end

"""Rows pushed since the last batch boundary (0 if no table is open)."""
streaming_table_pending_row_count(p::PageBuilder) = Int(
    ccall(
        (:pdf_page_builder_streaming_table_pending_row_count, LIB),
        Csize_t,
        (Ptr{Cvoid},),
        _pagebuilder(p),
    ),
)

"""Number of complete batches recorded so far (0 if no table is open)."""
streaming_table_batch_count(p::PageBuilder) = Int(
    ccall(
        (:pdf_page_builder_streaming_table_batch_count, LIB),
        Csize_t,
        (Ptr{Cvoid},),
        _pagebuilder(p),
    ),
)

"""Push one row of `cells` into the open streaming table (all rowspan=1)."""
function streaming_table_push_row(p::PageBuilder, cells::AbstractVector{<:AbstractString})
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in cells]
    GC.@preserve cells cstrs begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_streaming_table_push_row, LIB),
            Int32,
            (Ptr{Cvoid}, Csize_t, Ptr{Cstring}, Ref{Int32}),
            _pagebuilder(p),
            Csize_t(length(cstrs)),
            cstrs,
            code,
        )
        (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "streaming_table_push_row"))
    end
    return p
end

"""
Push one row of `cells` with per-cell `rowspans` (1=normal, ≥2=span). Pass an
empty `rowspans` to treat every cell as rowspan=1.
"""
function streaming_table_push_row_v2(
    p::PageBuilder,
    cells::AbstractVector{<:AbstractString},
    rowspans::AbstractVector{<:Integer},
)
    cstrs = [Base.unsafe_convert(Cstring, Base.cconvert(Cstring, s)) for s in cells]
    spans = Csize_t[Csize_t(v) for v in rowspans]
    GC.@preserve cells cstrs spans begin
        code = Ref{Int32}(0)
        rc = ccall(
            (:pdf_page_builder_streaming_table_push_row_v2, LIB),
            Int32,
            (Ptr{Cvoid}, Csize_t, Ptr{Cstring}, Ptr{Csize_t}, Ref{Int32}),
            _pagebuilder(p),
            Csize_t(length(cstrs)),
            cstrs,
            isempty(spans) ? C_NULL : spans,
            code,
        )
        (rc != 0 || code[] != 0) &&
            throw(PdfOxideError(code[], "streaming_table_push_row_v2"))
    end
    return p
end

"""
Commit this page's buffered operations to its parent builder. **Consumes** the
page handle — the wrapper's pointer is nulled so it will not be double-freed.
"""
function done(p::PageBuilder)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_page_builder_done, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _pagebuilder(p),
        code,
    )
    if rc != 0 || code[] != 0
        throw(PdfOxideError(code[], "done"))
    end
    # Success: the page was consumed by the parent builder; do NOT free it again.
    p.handle = C_NULL
    return nothing
end

# ── Phase-6 digital signatures / PKI / timestamps / TSA / validation ────────────
# Mirrors the established binding patterns: opaque native handles wrapped in
# mutable structs freed on close!/finalization; the PdfOxideError code helpers;
# _take_string (free_string), free_bytes byte-take, and closed-handle guards.
# All ccalls reference their C symbol as a LITERAL (generated families use
# @eval + QuoteNode). Methods use snake_case (Julia idiom); indices are 0-based.

# Copy a `const uint8_t *` return (out-len out-param) into a Julia Vector WITHOUT
# freeing it — the timestamp token/message-imprint getters return a borrowed
# pointer into the handle's storage (NOT an owned buffer), so free_bytes here
# would corrupt the handle.
function _copy_const_bytes(ptr::Ptr{UInt8}, len::Csize_t, code::Int32, op::String)
    ptr == C_NULL && throw(PdfOxideError(code, op))
    n = Int(len)
    return copy(unsafe_wrap(Array, ptr, n < 0 ? 0 : n))
end

# ── Logging ─────────────────────────────────────────────────────────────────────
"""Set the global log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace)."""
set_log_level(level::Integer) =
    ccall((:pdf_oxide_set_log_level, LIB), Cvoid, (Int32,), Int32(level))

"""Get the current global log level (0-5)."""
get_log_level() = Int(ccall((:pdf_oxide_get_log_level, LIB), Int32, ()))

# ── Certificate ───────────────────────────────────────────────────────────────
"""Signing credentials (certificate + private key) over the C ABI."""
mutable struct Certificate
    handle::Ptr{Cvoid}
    function Certificate(h::Ptr{Cvoid})
        c = new(h)
        finalizer(close!, c)
        return c
    end
end

"""Free the native certificate handle now (idempotent; also runs at finalization)."""
function close!(c::Certificate)
    if c.handle != C_NULL
        ccall((:pdf_certificate_free, LIB), Cvoid, (Ptr{Cvoid},), c.handle)
        c.handle = C_NULL
    end
    return nothing
end

_cert(c::Certificate) = (c.handle == C_NULL && error("Certificate is closed"); c.handle)

"""Load signing credentials from PKCS#12 / PFX bytes (optionally password-protected)."""
function certificate_load_from_bytes(
    data::AbstractVector{UInt8},
    password::AbstractString = "",
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_certificate_load_from_bytes, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Int32, Cstring, Ref{Int32}),
        data,
        Int32(length(data)),
        password,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "certificate_load_from_bytes"))
    return Certificate(h)
end

"""Load signing credentials from PEM-encoded certificate + private-key strings."""
function certificate_load_from_pem(cert_pem::AbstractString, key_pem::AbstractString)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_certificate_load_from_pem, LIB),
        Ptr{Cvoid},
        (Cstring, Cstring, Ref{Int32}),
        cert_pem,
        key_pem,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "certificate_load_from_pem"))
    return Certificate(h)
end

# String accessors over a const void* certificate pointer (free_string return).
for (jl_fn, c_fn) in (
    (:certificate_get_subject, :pdf_certificate_get_subject),
    (:certificate_get_issuer, :pdf_certificate_get_issuer),
    (:certificate_get_serial, :pdf_certificate_get_serial),
)
    op = String(jl_fn)
    @eval function $jl_fn(c::Certificate)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            _cert(c),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""Certificate validity window as `(not_before, not_after)` Unix epoch seconds."""
function certificate_get_validity(c::Certificate)
    nb = Ref{Int64}(0)
    na = Ref{Int64}(0)
    code = Ref{Int32}(0)
    ccall(
        (:pdf_certificate_get_validity, LIB),
        Cvoid,
        (Ptr{Cvoid}, Ref{Int64}, Ref{Int64}, Ref{Int32}),
        _cert(c),
        nb,
        na,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "certificate_get_validity"))
    return (Int(nb[]), Int(na[]))
end

"""Whether the certificate is currently valid (not expired / not-yet-valid)."""
function certificate_is_valid(c::Certificate)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_certificate_is_valid, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _cert(c),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "certificate_is_valid"))
    return v != 0
end

# ── Signing (top-level: return owned byte buffers via free_bytes) ─────────────────
"""Sign raw PDF bytes with `cert`; returns the signed PDF as a `Vector{UInt8}`."""
function sign_bytes(
    pdf::AbstractVector{UInt8},
    cert::Certificate,
    reason::AbstractString,
    location::AbstractString,
)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_sign_bytes, LIB),
        Ptr{UInt8},
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, Cstring, Cstring, Ref{Csize_t}, Ref{Int32}),
        pdf,
        Csize_t(length(pdf)),
        _cert(cert),
        reason,
        location,
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "sign_bytes")
end

# Marshal one parallel byte-array-array (Vec of buffers) into the
# (const uint8* const*, const uintptr* lens) pair the C ABI expects. The
# returned GC-roots keep the inner Julia buffers alive across the ccall.
function _marshal_byte_arrays(arrs::AbstractVector{<:AbstractVector{UInt8}})
    bufs = [Vector{UInt8}(a) for a in arrs]
    ptrs = Ptr{UInt8}[pointer(b) for b in bufs]
    lens = Csize_t[Csize_t(length(b)) for b in bufs]
    return (ptrs, lens, Csize_t(length(bufs)), bufs)
end

"""
Sign raw PDF bytes at a PAdES baseline `level` (0=B-B 1=B-T 2=B-LT). `tsa_url`
is the RFC 3161 timestamp source (required for level >= 1). The three parallel
byte-array lists carry the B-LT revocation material (DER certs / CRLs / OCSPs).
Returns the signed PDF as a `Vector{UInt8}`.
"""
function sign_bytes_pades(
    pdf::AbstractVector{UInt8},
    cert::Certificate,
    level::Integer,
    tsa_url::Union{Nothing,AbstractString},
    reason::AbstractString,
    location::AbstractString;
    certs::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
    crls::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
    ocsps::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
)
    cp, cl, cn, _kc = _marshal_byte_arrays(certs)
    rp, rl, rn, _kr = _marshal_byte_arrays(crls)
    op, ol, on, _ko = _marshal_byte_arrays(ocsps)
    tsa = tsa_url === nothing ? C_NULL : tsa_url
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = GC.@preserve _kc _kr _ko cp cl rp rl op ol ccall(
        (:pdf_sign_bytes_pades, LIB),
        Ptr{UInt8},
        (
            Ptr{UInt8},
            Csize_t,
            Ptr{Cvoid},
            Int32,
            Cstring,
            Cstring,
            Cstring,
            Ptr{Ptr{UInt8}},
            Ptr{Csize_t},
            Csize_t,
            Ptr{Ptr{UInt8}},
            Ptr{Csize_t},
            Csize_t,
            Ptr{Ptr{UInt8}},
            Ptr{Csize_t},
            Csize_t,
            Ref{Csize_t},
            Ref{Int32},
        ),
        pdf,
        Csize_t(length(pdf)),
        _cert(cert),
        Int32(level),
        tsa,
        reason,
        location,
        cp,
        cl,
        cn,
        rp,
        rl,
        rn,
        op,
        ol,
        on,
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "sign_bytes_pades")
end

# Mirror of the C `PadesSignOptionsC` #[repr(C)] struct (field order/types exact).
struct PadesSignOptionsC
    certificate_handle::Ptr{Cvoid}
    certs::Ptr{Ptr{UInt8}}
    cert_lens::Ptr{Csize_t}
    n_certs::Csize_t
    crls::Ptr{Ptr{UInt8}}
    crl_lens::Ptr{Csize_t}
    n_crls::Csize_t
    ocsps::Ptr{Ptr{UInt8}}
    ocsp_lens::Ptr{Csize_t}
    n_ocsps::Csize_t
    tsa_url::Cstring
    reason::Cstring
    location::Cstring
    level::Int32
end

"""
Struct-options variant of [`sign_bytes_pades`] — builds the `PadesSignOptionsC`
struct and delegates to `pdf_sign_bytes_pades_opts`. Behaviour is identical.
Returns the signed PDF as a `Vector{UInt8}`.
"""
function sign_bytes_pades_opts(
    pdf::AbstractVector{UInt8},
    cert::Certificate,
    level::Integer,
    tsa_url::Union{Nothing,AbstractString},
    reason::AbstractString,
    location::AbstractString;
    certs::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
    crls::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
    ocsps::AbstractVector{<:AbstractVector{UInt8}} = Vector{UInt8}[],
)
    cp, cl, cn, _kc = _marshal_byte_arrays(certs)
    rp, rl, rn, _kr = _marshal_byte_arrays(crls)
    op, ol, on, _ko = _marshal_byte_arrays(ocsps)
    # NUL-terminated C strings kept alive for the ccall duration.
    tsa_c =
        tsa_url === nothing ? UInt8[0] : Vector{UInt8}(codeunits(string(tsa_url) * "\0"))
    reason_c = Vector{UInt8}(codeunits(string(reason) * "\0"))
    location_c = Vector{UInt8}(codeunits(string(location) * "\0"))
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = GC.@preserve _kc _kr _ko cp cl rp rl op ol tsa_c reason_c location_c begin
        # Empty parallel arrays marshal to a NULL pointer (count 0).
        ptr_or_null(a) = isempty(a) ? Ptr{Ptr{UInt8}}(0) : pointer(a)
        lens_or_null(a) = isempty(a) ? Ptr{Csize_t}(0) : pointer(a)
        opts = PadesSignOptionsC(
            _cert(cert),
            ptr_or_null(cp),
            lens_or_null(cl),
            cn,
            ptr_or_null(rp),
            lens_or_null(rl),
            rn,
            ptr_or_null(op),
            lens_or_null(ol),
            on,
            tsa_url === nothing ? Cstring(C_NULL) : Cstring(pointer(tsa_c)),
            Cstring(pointer(reason_c)),
            Cstring(pointer(location_c)),
            Int32(level),
        )
        ref = Ref(opts)
        GC.@preserve ref ccall(
            (:pdf_sign_bytes_pades_opts, LIB),
            Ptr{UInt8},
            (Ptr{UInt8}, Csize_t, Ptr{PadesSignOptionsC}, Ref{Csize_t}, Ref{Int32}),
            pdf,
            Csize_t(length(pdf)),
            ref,
            len,
            code,
        )
    end
    return _take_bytes_uptr(ptr, len[], code[], "sign_bytes_pades_opts")
end

# ── SignatureInfo ─────────────────────────────────────────────────────────────
"""A digital signature extracted from a PDF (FfiSignatureInfo over the C ABI)."""
mutable struct SignatureInfo
    handle::Ptr{Cvoid}
    function SignatureInfo(h::Ptr{Cvoid})
        s = new(h)
        finalizer(close!, s)
        return s
    end
end

"""Free the native signature handle now (idempotent; also runs at finalization)."""
function close!(s::SignatureInfo)
    if s.handle != C_NULL
        ccall((:pdf_signature_free, LIB), Cvoid, (Ptr{Cvoid},), s.handle)
        s.handle = C_NULL
    end
    return nothing
end

_sig(s::SignatureInfo) = (s.handle == C_NULL && error("SignatureInfo is closed"); s.handle)

# String accessors over the signature handle (free_string return).
for (jl_fn, c_fn) in (
    (:signature_get_signer_name, :pdf_signature_get_signer_name),
    (:signature_get_signing_reason, :pdf_signature_get_signing_reason),
    (:signature_get_signing_location, :pdf_signature_get_signing_location),
)
    op = String(jl_fn)
    @eval function $jl_fn(s::SignatureInfo)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            _sig(s),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""Signing time as Unix epoch seconds."""
function signature_get_signing_time(s::SignatureInfo)
    code = Ref{Int32}(0)
    t = ccall(
        (:pdf_signature_get_signing_time, LIB),
        Int64,
        (Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "signature_get_signing_time"))
    return Int(t)
end

"""Signer's `Certificate` (owned; free via close!)."""
function signature_get_certificate(s::SignatureInfo)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_signature_get_certificate, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "signature_get_certificate"))
    return Certificate(h)
end

"""PAdES level code of the signature, or throws on error."""
function signature_get_pades_level(s::SignatureInfo)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_signature_get_pades_level, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "signature_get_pades_level"))
    return Int(v)
end

"""Whether the signature carries an embedded RFC 3161 timestamp."""
function signature_has_timestamp(s::SignatureInfo)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_signature_has_timestamp, LIB),
        Bool,
        (Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "signature_has_timestamp"))
    return v
end

"""The signature's embedded `Timestamp` (owned; free via close!)."""
function signature_get_timestamp(s::SignatureInfo)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_signature_get_timestamp, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "signature_get_timestamp"))
    return Timestamp(h)
end

"""Attach `ts` to the signature; returns `true` on success."""
function signature_add_timestamp(s::SignatureInfo, ts)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_signature_add_timestamp, LIB),
        Bool,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Int32}),
        _sig(s),
        _ts(ts),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "signature_add_timestamp"))
    return v
end

"""Run the signer-attributes crypto check. Returns 1=valid 0=invalid -1=unknown."""
function signature_verify(s::SignatureInfo)
    code = Ref{Int32}(0)
    v = ccall((:pdf_signature_verify, LIB), Int32, (Ptr{Cvoid}, Ref{Int32}), _sig(s), code)
    code[] != 0 && throw(PdfOxideError(code[], "signature_verify"))
    return Int(v)
end

"""End-to-end verify against the full `pdf` bytes. Returns 1=valid 0=invalid -1=unknown."""
function signature_verify_detached(s::SignatureInfo, pdf::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_signature_verify_detached, LIB),
        Int32,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Int32}),
        _sig(s),
        pdf,
        Csize_t(length(pdf)),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "signature_verify_detached"))
    return Int(v)
end

# ── Timestamp ─────────────────────────────────────────────────────────────────
"""A parsed RFC 3161 timestamp token (owned handle over the C ABI)."""
mutable struct Timestamp
    handle::Ptr{Cvoid}
    function Timestamp(h::Ptr{Cvoid})
        t = new(h)
        finalizer(close!, t)
        return t
    end
end

"""Free the native timestamp handle now (idempotent; also runs at finalization)."""
function close!(t::Timestamp)
    if t.handle != C_NULL
        ccall((:pdf_timestamp_free, LIB), Cvoid, (Ptr{Cvoid},), t.handle)
        t.handle = C_NULL
    end
    return nothing
end

_ts(t::Timestamp) = (t.handle == C_NULL && error("Timestamp is closed"); t.handle)

"""Parse a DER-encoded RFC 3161 TimeStampToken (or bare TSTInfo) into a `Timestamp`."""
function timestamp_parse(data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_timestamp_parse, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Csize_t, Ref{Int32}),
        data,
        Csize_t(length(data)),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "timestamp_parse"))
    return Timestamp(h)
end

# const uint8_t* + out-len getters — COPY the borrowed bytes, do NOT free_bytes.
for (jl_fn, c_fn) in (
    (:timestamp_get_token, :pdf_timestamp_get_token),
    (:timestamp_get_message_imprint, :pdf_timestamp_get_message_imprint),
)
    op = String(jl_fn)
    @eval function $jl_fn(t::Timestamp)
        len = Ref{Csize_t}(0)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Csize_t}, Ref{Int32}),
            _ts(t),
            len,
            code,
        )
        return _copy_const_bytes(ptr, len[], code[], $op)
    end
end

"""Timestamp time as Unix epoch seconds."""
function timestamp_get_time(t::Timestamp)
    code = Ref{Int32}(0)
    v = ccall((:pdf_timestamp_get_time, LIB), Int64, (Ptr{Cvoid}, Ref{Int32}), _ts(t), code)
    code[] != 0 && throw(PdfOxideError(code[], "timestamp_get_time"))
    return Int(v)
end

# String getters over the timestamp handle (free_string return).
for (jl_fn, c_fn) in (
    (:timestamp_get_serial, :pdf_timestamp_get_serial),
    (:timestamp_get_tsa_name, :pdf_timestamp_get_tsa_name),
    (:timestamp_get_policy_oid, :pdf_timestamp_get_policy_oid),
)
    op = String(jl_fn)
    @eval function $jl_fn(t::Timestamp)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            _ts(t),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""Digest algorithm code of the timestamp's message imprint."""
function timestamp_get_hash_algorithm(t::Timestamp)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_timestamp_get_hash_algorithm, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _ts(t),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "timestamp_get_hash_algorithm"))
    return Int(v)
end

"""Whether the timestamp token verifies (bool)."""
function timestamp_verify(t::Timestamp)
    code = Ref{Int32}(0)
    v = ccall((:pdf_timestamp_verify, LIB), Bool, (Ptr{Cvoid}, Ref{Int32}), _ts(t), code)
    code[] != 0 && throw(PdfOxideError(code[], "timestamp_verify"))
    return v
end

# ── TSA client ────────────────────────────────────────────────────────────────
"""An RFC 3161 Time-Stamping Authority client (owned handle over the C ABI)."""
mutable struct TsaClient
    handle::Ptr{Cvoid}
    function TsaClient(h::Ptr{Cvoid})
        t = new(h)
        finalizer(close!, t)
        return t
    end
end

"""Free the native TSA-client handle now (idempotent; also runs at finalization)."""
function close!(t::TsaClient)
    if t.handle != C_NULL
        ccall((:pdf_tsa_client_free, LIB), Cvoid, (Ptr{Cvoid},), t.handle)
        t.handle = C_NULL
    end
    return nothing
end

_tsa(t::TsaClient) = (t.handle == C_NULL && error("TsaClient is closed"); t.handle)

"""Create a TSA client for `url` (optional basic-auth, timeout, hash algo, nonce, cert-req)."""
function tsa_client_create(
    url::AbstractString;
    username::Union{Nothing,AbstractString} = nothing,
    password::Union{Nothing,AbstractString} = nothing,
    timeout::Integer = 30,
    hash_algo::Integer = 0,
    use_nonce::Bool = true,
    cert_req::Bool = true,
)
    code = Ref{Int32}(0)
    u = username === nothing ? C_NULL : username
    p = password === nothing ? C_NULL : password
    h = ccall(
        (:pdf_tsa_client_create, LIB),
        Ptr{Cvoid},
        (Cstring, Cstring, Cstring, Int32, Int32, Bool, Bool, Ref{Int32}),
        url,
        u,
        p,
        Int32(timeout),
        Int32(hash_algo),
        use_nonce,
        cert_req,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "tsa_client_create"))
    return TsaClient(h)
end

"""Request a timestamp over `data`; returns a `Timestamp` (performs network I/O)."""
function tsa_request_timestamp(t::TsaClient, data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_tsa_request_timestamp, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Int32}),
        _tsa(t),
        data,
        Csize_t(length(data)),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "tsa_request_timestamp"))
    return Timestamp(h)
end

"""Request a timestamp over a precomputed `hash`; returns a `Timestamp`."""
function tsa_request_timestamp_hash(
    t::TsaClient,
    hash::AbstractVector{UInt8},
    hash_algo::Integer,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_tsa_request_timestamp_hash, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Int32, Ref{Int32}),
        _tsa(t),
        hash,
        Csize_t(length(hash)),
        Int32(hash_algo),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "tsa_request_timestamp_hash"))
    return Timestamp(h)
end

# ── DSS (Document Security Store) ────────────────────────────────────────────────
"""A document `/DSS` (validation material) over the C ABI."""
mutable struct Dss
    handle::Ptr{Cvoid}
    function Dss(h::Ptr{Cvoid})
        d = new(h)
        finalizer(close!, d)
        return d
    end
end

"""Free the native DSS handle now (idempotent; also runs at finalization)."""
function close!(d::Dss)
    if d.handle != C_NULL
        ccall((:pdf_dss_free, LIB), Cvoid, (Ptr{Cvoid},), d.handle)
        d.handle = C_NULL
    end
    return nothing
end

_dss(d::Dss) = (d.handle == C_NULL && error("Dss is closed"); d.handle)

"""
Read the document `/DSS` into a `Dss`, or `nothing` when the document has no DSS
(not an error). The `Dss` handle is owned and freed via close!.
"""
function document_get_dss(doc::PdfDocument)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_document_get_dss, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(doc),
        code,
    )
    if h == C_NULL
        code[] != 0 && throw(PdfOxideError(code[], "document_get_dss"))
        return nothing
    end
    return Dss(h)
end

# Count accessors (no error out-param).
for (jl_fn, c_fn) in (
    (:dss_cert_count, :pdf_dss_cert_count),
    (:dss_crl_count, :pdf_dss_crl_count),
    (:dss_ocsp_count, :pdf_dss_ocsp_count),
    (:dss_vri_count, :pdf_dss_vri_count),
)
    @eval $jl_fn(d::Dss) =
        Int(ccall(($(QuoteNode(c_fn)), LIB), Int32, (Ptr{Cvoid},), _dss(d)))
end

# Indexed DER getters returning owned buffers (free via free_bytes).
for (jl_fn, c_fn) in (
    (:dss_get_cert, :pdf_dss_get_cert),
    (:dss_get_crl, :pdf_dss_get_crl),
    (:dss_get_ocsp, :pdf_dss_get_ocsp),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::Dss, index::Integer)
        len = Ref{Csize_t}(0)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Int32, Ref{Csize_t}, Ref{Int32}),
            _dss(d),
            Int32(index),
            len,
            code,
        )
        return _take_bytes_uptr(ptr, len[], code[], $op)
    end
end

# ── Validation (PDF/A, PDF/UA, PDF/X) ────────────────────────────────────────────
"""PDF/A validation result (FfiPdfAResults, freed on close!/finalization)."""
mutable struct PdfAResults
    handle::Ptr{Cvoid}
    function PdfAResults(h::Ptr{Cvoid})
        r = new(h)
        finalizer(close!, r)
        return r
    end
end
function close!(r::PdfAResults)
    if r.handle != C_NULL
        ccall((:pdf_pdf_a_results_free, LIB), Cvoid, (Ptr{Cvoid},), r.handle)
        r.handle = C_NULL
    end
    return nothing
end
_pdfa(r::PdfAResults) = (r.handle == C_NULL && error("PdfAResults is closed"); r.handle)

"""PDF/UA validation result (FfiUaResults, freed on close!/finalization)."""
mutable struct UaResults
    handle::Ptr{Cvoid}
    function UaResults(h::Ptr{Cvoid})
        r = new(h)
        finalizer(close!, r)
        return r
    end
end
function close!(r::UaResults)
    if r.handle != C_NULL
        ccall((:pdf_pdf_ua_results_free, LIB), Cvoid, (Ptr{Cvoid},), r.handle)
        r.handle = C_NULL
    end
    return nothing
end
_ua(r::UaResults) = (r.handle == C_NULL && error("UaResults is closed"); r.handle)

"""PDF/X validation result (FfiPdfXResults, freed on close!/finalization)."""
mutable struct PdfXResults
    handle::Ptr{Cvoid}
    function PdfXResults(h::Ptr{Cvoid})
        r = new(h)
        finalizer(close!, r)
        return r
    end
end
function close!(r::PdfXResults)
    if r.handle != C_NULL
        ccall((:pdf_pdf_x_results_free, LIB), Cvoid, (Ptr{Cvoid},), r.handle)
        r.handle = C_NULL
    end
    return nothing
end
_pdfx(r::PdfXResults) = (r.handle == C_NULL && error("PdfXResults is closed"); r.handle)

"""Validate the document against PDF/A at `level`; returns a `PdfAResults`."""
function validate_pdf_a(doc::PdfDocument, level::Integer)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_validate_pdf_a_level, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(doc),
        Int32(level),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "validate_pdf_a"))
    return PdfAResults(h)
end

"""Validate the document against PDF/UA at `level`; returns a `UaResults`."""
function validate_pdf_ua(doc::PdfDocument, level::Integer)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_validate_pdf_ua, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(doc),
        Int32(level),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "validate_pdf_ua"))
    return UaResults(h)
end

"""Validate the document against PDF/X at `level`; returns a `PdfXResults`."""
function validate_pdf_x(doc::PdfDocument, level::Integer)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_validate_pdf_x_level, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(doc),
        Int32(level),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "validate_pdf_x"))
    return PdfXResults(h)
end

"""Whether the document is PDF/A compliant (bool)."""
function is_compliant(r::PdfAResults)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_pdf_a_is_compliant, LIB),
        Bool,
        (Ptr{Cvoid}, Ref{Int32}),
        _pdfa(r),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "is_compliant"))
    return v
end

"""Whether the document is PDF/UA accessible (bool)."""
function is_accessible(r::UaResults)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_pdf_ua_is_accessible, LIB),
        Bool,
        (Ptr{Cvoid}, Ref{Int32}),
        _ua(r),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "is_accessible"))
    return v
end

"""Whether the document is PDF/X compliant (bool)."""
function is_compliant(r::PdfXResults)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_pdf_x_is_compliant, LIB),
        Bool,
        (Ptr{Cvoid}, Ref{Int32}),
        _pdfx(r),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "is_compliant"))
    return v
end

# Per-result error/warning collectors. Each builds a Vector{String} from the
# count + indexed getter, mirroring the index-addressed list pattern.
# Generated with @eval so each ccall references its C function as a LITERAL
# symbol (ccall forbids a variable function name). One collector per (count,get).
for (count_fn, get_fn) in (
    (:pdf_pdf_a_error_count, :pdf_pdf_a_get_error),
    (:pdf_pdf_a_warning_count, :pdf_pdf_a_get_error),
    (:pdf_pdf_ua_error_count, :pdf_pdf_ua_get_error),
    (:pdf_pdf_ua_warning_count, :pdf_pdf_ua_get_warning),
    (:pdf_pdf_x_error_count, :pdf_pdf_x_get_error),
)
    fname = Symbol("_collect_", count_fn)
    @eval function $fname(handle, op::String)
        n = ccall(($(QuoteNode(count_fn)), LIB), Int32, (Ptr{Cvoid},), handle)
        out = String[]
        for i = 0:(Int(n)-1)
            code = Ref{Int32}(0)
            ptr = ccall(
                ($(QuoteNode(get_fn)), LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                handle,
                Int32(i),
                code,
            )
            push!(out, _take_string(ptr, code[], op))
        end
        return out
    end
end

"""PDF/A error messages as a `Vector{String}`."""
errors(r::PdfAResults) = _collect_pdf_pdf_a_error_count(_pdfa(r), "errors")
"""PDF/A warning messages as a `Vector{String}`."""
warnings(r::PdfAResults) = _collect_pdf_pdf_a_warning_count(_pdfa(r), "warnings")
"""PDF/UA error messages as a `Vector{String}`."""
errors(r::UaResults) = _collect_pdf_pdf_ua_error_count(_ua(r), "errors")
"""PDF/UA warning messages as a `Vector{String}`."""
warnings(r::UaResults) = _collect_pdf_pdf_ua_warning_count(_ua(r), "warnings")
"""PDF/X error messages as a `Vector{String}`."""
errors(r::PdfXResults) = _collect_pdf_pdf_x_error_count(_pdfx(r), "errors")
"""PDF/X has no warning channel; returns an empty `Vector{String}`."""
warnings(::PdfXResults) = String[]

"""PDF/A error count (Int)."""
pdf_a_error_count(r::PdfAResults) =
    Int(ccall((:pdf_pdf_a_error_count, LIB), Int32, (Ptr{Cvoid},), _pdfa(r)))
"""PDF/A warning count (Int)."""
pdf_a_warning_count(r::PdfAResults) =
    Int(ccall((:pdf_pdf_a_warning_count, LIB), Int32, (Ptr{Cvoid},), _pdfa(r)))
"""PDF/UA error count (Int)."""
pdf_ua_error_count(r::UaResults) =
    Int(ccall((:pdf_pdf_ua_error_count, LIB), Int32, (Ptr{Cvoid},), _ua(r)))
"""PDF/UA warning count (Int)."""
pdf_ua_warning_count(r::UaResults) =
    Int(ccall((:pdf_pdf_ua_warning_count, LIB), Int32, (Ptr{Cvoid},), _ua(r)))
"""PDF/X error count (Int)."""
pdf_x_error_count(r::PdfXResults) =
    Int(ccall((:pdf_pdf_x_error_count, LIB), Int32, (Ptr{Cvoid},), _pdfx(r)))

"""
PDF/UA accessibility-element statistics: a `NamedTuple` of element counts
`(structure, images, tables, forms, annotations, pages)`.
"""
function ua_stats(r::UaResults)
    s = Ref{Int32}(0)
    im = Ref{Int32}(0)
    tb = Ref{Int32}(0)
    fm = Ref{Int32}(0)
    an = Ref{Int32}(0)
    pg = Ref{Int32}(0)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_pdf_ua_get_stats, LIB),
        Bool,
        (
            Ptr{Cvoid},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
        ),
        _ua(r),
        s,
        im,
        tb,
        fm,
        an,
        pg,
        code,
    )
    (code[] != 0 || !ok) && throw(PdfOxideError(code[], "ua_stats"))
    return (
        structure = Int(s[]),
        images = Int(im[]),
        tables = Int(tb[]),
        forms = Int(fm[]),
        annotations = Int(an[]),
        pages = Int(pg[]),
    )
end

# Document-level convenience aliases mirroring the cross-binding naming.
"""Validate `doc` against PDF/A `level` (alias of [`validate_pdf_a`])."""
validatePdfA(doc::PdfDocument, level::Integer) = validate_pdf_a(doc, level)
"""Validate `doc` against PDF/UA `level` (alias of [`validate_pdf_ua`])."""
validatePdfUa(doc::PdfDocument, level::Integer) = validate_pdf_ua(doc, level)
"""Validate `doc` against PDF/X `level` (alias of [`validate_pdf_x`])."""
validatePdfX(doc::PdfDocument, level::Integer) = validate_pdf_x(doc, level)

# ── Phase-7: barcodes / OCR / render variants / redaction / constructors /
#    page getters / timestamp ───────────────────────────────────────────────────
# Same patterns as earlier phases: opaque native handles wrapped in mutable
# structs with finalizers and a closed-handle guard; C strings copied out and
# freed via free_string (_take_string); raw byte buffers copied and freed via
# free_bytes; non-success C-ABI error codes throw PdfOxideError. RenderedImage and
# DocumentEditor types from earlier phases are reused. Page indices are 0-based.

# ── Barcodes / QR ───────────────────────────────────────────────────────────────
# A generated or decoded barcode/QR image. Owns the native FfiBarcodeImage handle
# (freed via pdf_barcode_free on close!/finalization); accessors copy out of it.
mutable struct Barcode
    handle::Ptr{Cvoid}
    function Barcode(h::Ptr{Cvoid})
        b = new(h)
        finalizer(close!, b)
        return b
    end
end

"""Free the native barcode handle now (idempotent; also runs at finalization)."""
function close!(b::Barcode)
    if b.handle != C_NULL
        ccall((:pdf_barcode_free, LIB), Cvoid, (Ptr{Cvoid},), b.handle)
        b.handle = C_NULL
    end
    return nothing
end

_barcode(b::Barcode) = (b.handle == C_NULL && error("Barcode is closed"); b.handle)

"""Generate a QR code from `data`. `error_correction`/`size_px` are passthrough ints."""
function generate_qr_code(
    data::AbstractString,
    error_correction::Integer = 0,
    size_px::Integer = 256,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_generate_qr_code, LIB),
        Ptr{Cvoid},
        (Cstring, Int32, Int32, Ref{Int32}),
        data,
        Int32(error_correction),
        Int32(size_px),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "generate_qr_code"))
    return Barcode(h)
end

"""Generate a 1D/2D barcode from `data`. `format`/`size_px` are passthrough ints."""
function generate_barcode(data::AbstractString, format::Integer = 0, size_px::Integer = 256)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_generate_barcode, LIB),
        Ptr{Cvoid},
        (Cstring, Int32, Int32, Ref{Int32}),
        data,
        Int32(format),
        Int32(size_px),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "generate_barcode"))
    return Barcode(h)
end

"""The barcode's decoded/encoded payload string."""
function barcode_get_data(b::Barcode)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_barcode_get_data, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _barcode(b),
        code,
    )
    return _take_string(ptr, code[], "barcode_get_data")
end

"""The barcode's format as an int code."""
function barcode_get_format(b::Barcode)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_barcode_get_format, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _barcode(b),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "barcode_get_format"))
    return Int(v)
end

"""The barcode's decode confidence (0.0–1.0)."""
function barcode_get_confidence(b::Barcode)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_barcode_get_confidence, LIB),
        Float32,
        (Ptr{Cvoid}, Ref{Int32}),
        _barcode(b),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "barcode_get_confidence"))
    return Float64(v)
end

"""Render the barcode to a PNG `Vector{UInt8}`. `size_px` is advisory (passthrough)."""
function barcode_get_image_png(b::Barcode, size_px::Integer = 256)
    len = Ref{Int32}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_barcode_get_image_png, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Int32}, Ref{Int32}),
        _barcode(b),
        Int32(size_px),
        len,
        code,
    )
    ptr == C_NULL && throw(PdfOxideError(code[], "barcode_get_image_png"))
    n = len[] < 0 ? 0 : Int(len[])
    out = copy(unsafe_wrap(Array, ptr, n))
    # Raw byte buffer frees via free_bytes, not free_string.
    ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), ptr)
    return out
end

"""Render the barcode to an SVG string. `size_px` is advisory (passthrough)."""
function barcode_get_svg(b::Barcode, size_px::Integer = 256)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_barcode_get_svg, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _barcode(b),
        Int32(size_px),
        code,
    )
    return _take_string(ptr, code[], "barcode_get_svg")
end

"""Stamp a barcode onto a (0-based) page of an editor at rect `(x, y, width, height)`."""
function add_barcode_to_page(
    e::DocumentEditor,
    page::Integer,
    b::Barcode,
    x::Real,
    y::Real,
    width::Real,
    height::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_add_barcode_to_page, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Float32, Float32, Float32, Float32, Ref{Int32}),
        _editor(e),
        Int32(page),
        _barcode(b),
        Float32(x),
        Float32(y),
        Float32(width),
        Float32(height),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "add_barcode_to_page"))
    return nothing
end

# ── OCR ─────────────────────────────────────────────────────────────────────────
# An OCR engine over the native void* handle (freed via pdf_ocr_engine_free).
mutable struct OcrEngine
    handle::Ptr{Cvoid}
    function OcrEngine(h::Ptr{Cvoid})
        o = new(h)
        finalizer(close!, o)
        return o
    end
end

"""Free the native OCR engine handle now (idempotent; also runs at finalization)."""
function close!(o::OcrEngine)
    if o.handle != C_NULL
        ccall((:pdf_ocr_engine_free, LIB), Cvoid, (Ptr{Cvoid},), o.handle)
        o.handle = C_NULL
    end
    return nothing
end

_ocr(o::OcrEngine) = (o.handle == C_NULL && error("OcrEngine is closed"); o.handle)

"""Create an OCR engine from detection/recognition model + dictionary file paths."""
function ocr_engine_create(
    det_model_path::AbstractString,
    rec_model_path::AbstractString,
    dict_path::AbstractString,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_ocr_engine_create, LIB),
        Ptr{Cvoid},
        (Cstring, Cstring, Cstring, Ref{Int32}),
        det_model_path,
        rec_model_path,
        dict_path,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "ocr_engine_create"))
    return OcrEngine(h)
end

"""Whether a (0-based) page needs OCR (i.e. is scanned/hybrid)."""
function page_needs_ocr(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_ocr_page_needs_ocr, LIB),
        Bool,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "page_needs_ocr"))
    return ok
end

"""
Extract text from a (0-based) page using OCR. `engine` may be `nothing` to fall
back to native text extraction only.
"""
function ocr_extract_text(
    d::PdfDocument,
    page::Integer,
    engine::Union{Nothing,OcrEngine} = nothing,
)
    code = Ref{Int32}(0)
    eh = engine === nothing ? C_NULL : _ocr(engine)
    ptr = ccall(
        (:pdf_ocr_extract_text, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        Int32(page),
        eh,
        code,
    )
    return _take_string(ptr, code[], "ocr_extract_text")
end

# ── Render variants (reuse the RenderedImage wrapper) ───────────────────────────
"""
Render a (0-based) page with the full RenderOptions surface. Background channels
are 0.0–1.0; `transparent_background`/`render_annotations` are 0/1 flags.
"""
function render_page_with_options(
    d::PdfDocument,
    page::Integer,
    dpi::Integer,
    format::Integer,
    bg_r::Real,
    bg_g::Real,
    bg_b::Real,
    bg_a::Real,
    transparent_background::Integer,
    render_annotations::Integer,
    jpeg_quality::Integer,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page_with_options, LIB),
        Ptr{Cvoid},
        (
            Ptr{Cvoid},
            Int32,
            Int32,
            Int32,
            Float32,
            Float32,
            Float32,
            Float32,
            Int32,
            Int32,
            Int32,
            Ref{Int32},
        ),
        _doc(d),
        Int32(page),
        Int32(dpi),
        Int32(format),
        Float32(bg_r),
        Float32(bg_g),
        Float32(bg_b),
        Float32(bg_a),
        Int32(transparent_background),
        Int32(render_annotations),
        Int32(jpeg_quality),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page_with_options"))
    return RenderedImage(h)
end

"""
Render a (0-based) page with the full RenderOptions surface plus OCG layer
filtering. `excluded_layers` is a vector of OCG `/Name` strings to suppress.
"""
function render_page_with_options_ex(
    d::PdfDocument,
    page::Integer,
    dpi::Integer,
    format::Integer,
    bg_r::Real,
    bg_g::Real,
    bg_b::Real,
    bg_a::Real,
    transparent_background::Integer,
    render_annotations::Integer,
    jpeg_quality::Integer,
    excluded_layers::AbstractVector{<:AbstractString} = String[],
)
    # Marshal the Julia String list to a C array of NUL-terminated pointers.
    cstrs = [Base.cconvert(Cstring, s) for s in excluded_layers]
    ptrs = [Base.unsafe_convert(Cstring, c) for c in cstrs]
    GC.@preserve excluded_layers cstrs ptrs begin
        arg = isempty(ptrs) ? Ptr{Cstring}(C_NULL) : Base.unsafe_convert(Ptr{Cstring}, ptrs)
        code = Ref{Int32}(0)
        h = ccall(
            (:pdf_render_page_with_options_ex, LIB),
            Ptr{Cvoid},
            (
                Ptr{Cvoid},
                Int32,
                Int32,
                Int32,
                Float32,
                Float32,
                Float32,
                Float32,
                Int32,
                Int32,
                Int32,
                Ptr{Cstring},
                Csize_t,
                Ref{Int32},
            ),
            _doc(d),
            Int32(page),
            Int32(dpi),
            Int32(format),
            Float32(bg_r),
            Float32(bg_g),
            Float32(bg_b),
            Float32(bg_a),
            Int32(transparent_background),
            Int32(render_annotations),
            Int32(jpeg_quality),
            arg,
            Csize_t(length(ptrs)),
            code,
        )
        h == C_NULL && throw(PdfOxideError(code[], "render_page_with_options_ex"))
        return RenderedImage(h)
    end
end

"""Render a rectangular region of a (0-based) page (crop in PDF user-space points)."""
function render_page_region(
    d::PdfDocument,
    page::Integer,
    crop_x::Real,
    crop_y::Real,
    crop_width::Real,
    crop_height::Real,
    format::Integer = 0,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page_region, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Float32, Float32, Float32, Float32, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        Float32(crop_x),
        Float32(crop_y),
        Float32(crop_width),
        Float32(crop_height),
        Int32(format),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page_region"))
    return RenderedImage(h)
end

"""Render a (0-based) page to fit inside `w`×`h` pixels, preserving aspect ratio."""
function render_page_fit(
    d::PdfDocument,
    page::Integer,
    w::Integer,
    h::Integer,
    format::Integer = 0,
)
    code = Ref{Int32}(0)
    handle = ccall(
        (:pdf_render_page_fit, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Int32, Int32, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        Int32(w),
        Int32(h),
        Int32(format),
        code,
    )
    handle == C_NULL && throw(PdfOxideError(code[], "render_page_fit"))
    return RenderedImage(handle)
end

"""
Render a (0-based) page to a raw premultiplied RGBA8888 buffer at `dpi`. Returns
`(RenderedImage, width, height)` where the pixel bytes are in `img.data`.
"""
function render_page_raw(d::PdfDocument, page::Integer, dpi::Integer)
    ow = Ref{Int32}(0)
    oh = Ref{Int32}(0)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_render_page_raw, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Int32, Ref{Int32}, Ref{Int32}, Ref{Int32}),
        _doc(d),
        Int32(page),
        Int32(dpi),
        ow,
        oh,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "render_page_raw"))
    return (RenderedImage(h), Int(ow[]), Int(oh[]))
end

# ── Renderer (reusable config handle) ───────────────────────────────────────────
# A reusable renderer config over the native void* handle (freed via
# pdf_renderer_free).
mutable struct Renderer
    handle::Ptr{Cvoid}
    function Renderer(h::Ptr{Cvoid})
        r = new(h)
        finalizer(close!, r)
        return r
    end
end

"""Free the native renderer handle now (idempotent; also runs at finalization)."""
function close!(r::Renderer)
    if r.handle != C_NULL
        ccall((:pdf_renderer_free, LIB), Cvoid, (Ptr{Cvoid},), r.handle)
        r.handle = C_NULL
    end
    return nothing
end

"""Create a reusable renderer with `dpi`, `format`, `quality`, and anti-aliasing."""
function create_renderer(
    dpi::Integer = 150,
    format::Integer = 0,
    quality::Integer = 90,
    anti_alias::Bool = true,
)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_create_renderer, LIB),
        Ptr{Cvoid},
        (Int32, Int32, Int32, Bool, Ref{Int32}),
        Int32(dpi),
        Int32(format),
        Int32(quality),
        anti_alias,
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "create_renderer"))
    return Renderer(h)
end

"""Estimate render time (ms) for a (0-based) page of `doc`."""
function estimate_render_time(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_estimate_render_time, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "estimate_render_time"))
    return Int(v)
end

# ── Redaction (methods on DocumentEditor) ───────────────────────────────────────
"""
Queue a redaction box on a (0-based) `page`: corners `(x1, y1)`–`(x2, y2)` with
overlay colour `(r, g, b)` (DeviceRGB, 0.0–1.0). Coordinates are user-space.
"""
function redaction_add(
    e::DocumentEditor,
    page::Integer,
    x1::Real,
    y1::Real,
    x2::Real,
    y2::Real,
    r::Real,
    g::Real,
    b::Real,
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_redaction_add, LIB),
        Int32,
        (
            Ptr{Cvoid},
            Csize_t,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Ref{Int32},
        ),
        _editor(e),
        Csize_t(page),
        Float64(x1),
        Float64(y1),
        Float64(x2),
        Float64(y2),
        Float64(r),
        Float64(g),
        Float64(b),
        code,
    )
    (rc != 0 || code[] != 0) && throw(PdfOxideError(code[], "redaction_add"))
    return nothing
end

"""Number of queued redaction regions for a (0-based) `page`."""
function redaction_count(e::DocumentEditor, page::Integer)
    code = Ref{Int32}(0)
    n = ccall(
        (:pdf_redaction_count, LIB),
        Int32,
        (Ptr{Cvoid}, Csize_t, Ref{Int32}),
        _editor(e),
        Csize_t(page),
        code,
    )
    n < 0 && throw(PdfOxideError(code[], "redaction_count"))
    return Int(n)
end

"""
Destructively apply all queued redactions with overlay colour `(r, g, b)`.
`scrub_metadata` reserved for the document-scrub pass. Returns the number of
glyphs physically removed.
"""
function redaction_apply(e::DocumentEditor, scrub_metadata::Bool, r::Real, g::Real, b::Real)
    code = Ref{Int32}(0)
    n = ccall(
        (:pdf_redaction_apply, LIB),
        Int32,
        (Ptr{Cvoid}, Bool, Float64, Float64, Float64, Ref{Int32}),
        _editor(e),
        scrub_metadata,
        Float64(r),
        Float64(g),
        Float64(b),
        code,
    )
    n < 0 && throw(PdfOxideError(code[], "redaction_apply"))
    return Int(n)
end

"""Sanitize the document (strip Info/XMP/JavaScript/EmbeddedFiles); returns count removed."""
function redaction_scrub_metadata(e::DocumentEditor)
    code = Ref{Int32}(0)
    n = ccall(
        (:pdf_redaction_scrub_metadata, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _editor(e),
        code,
    )
    n < 0 && throw(PdfOxideError(code[], "redaction_scrub_metadata"))
    return Int(n)
end

# ── Constructors (return a Pdf builder handle) ──────────────────────────────────
"""Build a `Pdf` from an image file at `path`."""
function from_image(path::AbstractString)
    code = Ref{Int32}(0)
    h = ccall((:pdf_from_image, LIB), Ptr{Cvoid}, (Cstring, Ref{Int32}), path, code)
    h == C_NULL && throw(PdfOxideError(code[], "from_image"))
    return Pdf(h)
end

"""Build a `Pdf` from in-memory image bytes."""
function from_image_bytes(data::AbstractVector{UInt8})
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_from_image_bytes, LIB),
        Ptr{Cvoid},
        (Ptr{UInt8}, Int32, Ref{Int32}),
        data,
        Int32(length(data)),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "from_image_bytes"))
    return Pdf(h)
end

"""
Build a `Pdf` from HTML + CSS with a single optional embedded font. Pass
`font_bytes = nothing` (or empty) for no font.
"""
function from_html_css(
    html::AbstractString,
    css::AbstractString,
    font_bytes::Union{Nothing,AbstractVector{UInt8}} = nothing,
)
    fb = font_bytes === nothing ? UInt8[] : collect(font_bytes)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_from_html_css, LIB),
        Ptr{Cvoid},
        (Cstring, Cstring, Ptr{UInt8}, Csize_t, Ref{Int32}),
        html,
        css,
        isempty(fb) ? C_NULL : pointer(fb),
        Csize_t(length(fb)),
        code,
    )
    GC.@preserve fb begin
        h == C_NULL && throw(PdfOxideError(code[], "from_html_css"))
    end
    return Pdf(h)
end

"""
Build a `Pdf` from HTML + CSS with a multi-font cascade. `families` and `fonts`
are parallel: `families[i]` names the family carried by `fonts[i]` (its bytes).
"""
function from_html_css_with_fonts(
    html::AbstractString,
    css::AbstractString,
    families::AbstractVector{<:AbstractString},
    fonts::AbstractVector{<:AbstractVector{UInt8}},
)
    length(families) == length(fonts) ||
        throw(ArgumentError("families and fonts must be parallel (same length)"))
    n = length(families)
    # Marshal the family-name C-string array + parallel byte-pointer / length arrays.
    fam_c = [Base.cconvert(Cstring, s) for s in families]
    fam_p = [Base.unsafe_convert(Cstring, c) for c in fam_c]
    byte_vecs = [collect(f) for f in fonts]
    byte_p = Ptr{UInt8}[isempty(v) ? Ptr{UInt8}(C_NULL) : pointer(v) for v in byte_vecs]
    lens = Csize_t[Csize_t(length(v)) for v in byte_vecs]
    GC.@preserve families fam_c fam_p byte_vecs byte_p lens begin
        fam_arg = n == 0 ? Ptr{Cstring}(C_NULL) : Base.unsafe_convert(Ptr{Cstring}, fam_p)
        bp_arg = n == 0 ? Ptr{Ptr{UInt8}}(C_NULL) : pointer(byte_p)
        ln_arg = n == 0 ? Ptr{Csize_t}(C_NULL) : pointer(lens)
        code = Ref{Int32}(0)
        h = ccall(
            (:pdf_from_html_css_with_fonts, LIB),
            Ptr{Cvoid},
            (
                Cstring,
                Cstring,
                Ptr{Cstring},
                Ptr{Ptr{UInt8}},
                Ptr{Csize_t},
                Csize_t,
                Ref{Int32},
            ),
            html,
            css,
            fam_arg,
            bp_arg,
            ln_arg,
            Csize_t(n),
            code,
        )
        h == C_NULL && throw(PdfOxideError(code[], "from_html_css_with_fonts"))
        return Pdf(h)
    end
end

"""Merge the PDFs at `paths` (in order) into a single PDF `Vector{UInt8}`."""
function merge_pdfs(paths::AbstractVector{<:AbstractString})
    # Marshal the Julia String list to a C array of NUL-terminated pointers.
    path_c = [Base.cconvert(Cstring, s) for s in paths]
    path_p = [Base.unsafe_convert(Cstring, c) for c in path_c]
    GC.@preserve paths path_c path_p begin
        arg =
            isempty(path_p) ? Ptr{Cstring}(C_NULL) :
            Base.unsafe_convert(Ptr{Cstring}, path_p)
        len = Ref{Int32}(0)
        code = Ref{Int32}(0)
        ptr = ccall(
            (:pdf_merge, LIB),
            Ptr{UInt8},
            (Ptr{Cstring}, Int32, Ref{Int32}, Ref{Int32}),
            arg,
            Int32(length(path_p)),
            len,
            code,
        )
        ptr == C_NULL && throw(PdfOxideError(code[], "merge_pdfs"))
        m = len[] < 0 ? 0 : Int(len[])
        out = copy(unsafe_wrap(Array, ptr, m))
        # Raw byte buffer frees via free_bytes, not free_string.
        ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), ptr)
        return out
    end
end

# ── Page getters (on Document, 0-based page) ────────────────────────────────────
# Float32-returning page getters — one per C function, generated with @eval so
# each ccall references its C function name as a LITERAL symbol.
for (jl_fn, c_fn) in
    ((:page_get_width, :pdf_page_get_width), (:page_get_height, :pdf_page_get_height))
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer)
        code = Ref{Int32}(0)
        v = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Float32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _doc(d),
            Int32(page),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], $op))
        return Float64(v)
    end
end

"""Absolute rotation (degrees) of a (0-based) page."""
function page_get_rotation(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    v = ccall(
        (:pdf_page_get_rotation, LIB),
        Int32,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "page_get_rotation"))
    return Int(v)
end

# An opaque page-element list. Owns the native FfiElementList handle (freed via
# pdf_oxide_elements_free on close!/finalization). Per-element accessors are
# wrapped: element_count + per-index type/text/rect.
mutable struct ElementList
    handle::Ptr{Cvoid}
    function ElementList(h::Ptr{Cvoid})
        l = new(h)
        finalizer(close!, l)
        return l
    end
end

"""Free the native element-list handle now (idempotent; also runs at finalization)."""
function close!(l::ElementList)
    if l.handle != C_NULL
        ccall((:pdf_oxide_elements_free, LIB), Cvoid, (Ptr{Cvoid},), l.handle)
        l.handle = C_NULL
    end
    return nothing
end

_elements(l::ElementList) = (l.handle == C_NULL && error("ElementList is closed"); l.handle)

"""Page elements (layout regions) of a (0-based) page as an `ElementList` handle."""
function page_get_elements(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_page_get_elements, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "page_get_elements"))
    return ElementList(h)
end

"""Number of elements in an `ElementList`."""
function element_count(l::ElementList)
    n = ccall((:pdf_oxide_element_count, LIB), Int32, (Ptr{Cvoid},), _elements(l))
    n < 0 && throw(PdfOxideError(n, "element_count"))
    return Int(n)
end

# ── Timestamp (top-level fn returning bytes via out-params) ──────────────────────
"""
Add an RFC 3161 timestamp to `pdf_data` for the signature at `sig_index`, using
the TSA at `tsa_url`. Returns the timestamped PDF `Vector{UInt8}`.
"""
function add_timestamp(
    pdf_data::AbstractVector{UInt8},
    sig_index::Integer,
    tsa_url::AbstractString,
)
    out_ptr = Ref{Ptr{UInt8}}(C_NULL)
    out_len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_add_timestamp, LIB),
        Bool,
        (Ptr{UInt8}, Csize_t, Int32, Cstring, Ref{Ptr{UInt8}}, Ref{Csize_t}, Ref{Int32}),
        pdf_data,
        Csize_t(length(pdf_data)),
        Int32(sig_index),
        tsa_url,
        out_ptr,
        out_len,
        code,
    )
    (!ok || code[] != 0 || out_ptr[] == C_NULL) &&
        throw(PdfOxideError(code[], "add_timestamp"))
    return _take_bytes_uptr(out_ptr[], out_len[], code[], "add_timestamp")
end

# ── Phase-8: final C-ABI coverage ─────────────────────────────────────────────
# Everything below wraps the remaining unwrapped symbols in pdf_oxide.h, reusing
# the existing handle structs (PdfDocument/DocumentEditor/Pdf/SignatureInfo/Dss/
# Word/TextLine/Table/Image/Bbox/ElementList) and helpers (_take_string,
# _take_bytes_uptr, PdfOxideError). Every ccall uses a LITERAL symbol; @eval +
# QuoteNode is used wherever the C function name is parameterised.

# ── Office: open from DOCX/PPTX/XLSX bytes (-> PdfDocument) ─────────────────────
for (jl_fn, c_fn) in (
    (:open_from_docx_bytes, :pdf_document_open_from_docx_bytes),
    (:open_from_pptx_bytes, :pdf_document_open_from_pptx_bytes),
    (:open_from_xlsx_bytes, :pdf_document_open_from_xlsx_bytes),
)
    op = String(jl_fn)
    @eval function $jl_fn(data::AbstractVector{UInt8})
        code = Ref{Int32}(0)
        h = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{Cvoid},
            (Ptr{UInt8}, Csize_t, Ref{Int32}),
            data,
            Csize_t(length(data)),
            code,
        )
        h == C_NULL && throw(PdfOxideError(code[], $op))
        return PdfDocument(h)
    end
end

# ── Office: export the document to DOCX/PPTX/XLSX (-> owned bytes) ──────────────
for (jl_fn, c_fn) in (
    (:to_docx, :pdf_document_to_docx),
    (:to_pptx, :pdf_document_to_pptx),
    (:to_xlsx, :pdf_document_to_xlsx),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument)
        len = Ref{Csize_t}(0)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Csize_t}, Ref{Int32}),
            _doc(d),
            len,
            code,
        )
        return _take_bytes_uptr(ptr, len[], code[], $op)
    end
end

# ── In-rect text extraction (-> String) ────────────────────────────────────────
"""Extract plain text inside the rect `(x, y, w, h)` on a (0-based) page."""
function extract_text_in_rect(
    d::PdfDocument,
    page::Integer,
    x::Real,
    y::Real,
    w::Real,
    h::Real,
)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_extract_text_in_rect, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Float32, Float32, Float32, Float32, Ref{Int32}),
        _doc(d),
        Int32(page),
        Float32(x),
        Float32(y),
        Float32(w),
        Float32(h),
        code,
    )
    return _take_string(ptr, code[], "extract_text_in_rect")
end

# In-rect list openers (NULL on error -> throw); each returns the list handle.
for (jl_fn, c_fn) in (
    (:_open_words_rect, :pdf_document_extract_words_in_rect),
    (:_open_lines_rect, :pdf_document_extract_lines_in_rect),
    (:_open_tables_rect, :pdf_document_extract_tables_in_rect),
    (:_open_images_rect, :pdf_document_extract_images_in_rect),
)
    @eval function $jl_fn(
        d::PdfDocument,
        page::Integer,
        x::Real,
        y::Real,
        w::Real,
        h::Real,
        op::String,
    )
        code = Ref{Int32}(0)
        list = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Int32, Float32, Float32, Float32, Float32, Ref{Int32}),
            _doc(d),
            Int32(page),
            Float32(x),
            Float32(y),
            Float32(w),
            Float32(h),
            code,
        )
        list == C_NULL && throw(PdfOxideError(code[], op))
        return list
    end
end

# Build the per-element list reusing the same accessor C functions the whole-page
# extractors use (the in-rect openers return identical list shapes).
function _words_from_list(list::Ptr{Cvoid}, op::String)
    try
        n = ccall((:pdf_oxide_word_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Word}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            txt = _take_string(
                ccall(
                    (:pdf_oxide_word_get_text, LIB),
                    Ptr{UInt8},
                    (Ptr{Cvoid}, Int32, Ref{Int32}),
                    list,
                    Int32(i),
                    Ref{Int32}(0),
                ),
                Int32(0),
                op,
            )
            bb = _bbox_word(list, i, op)
            font = _take_string(
                ccall(
                    (:pdf_oxide_word_get_font_name, LIB),
                    Ptr{UInt8},
                    (Ptr{Cvoid}, Int32, Ref{Int32}),
                    list,
                    Int32(i),
                    Ref{Int32}(0),
                ),
                Int32(0),
                op,
            )
            scode = Ref{Int32}(0)
            fs = ccall(
                (:pdf_oxide_word_get_font_size, LIB),
                Float32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                scode,
            )
            scode[] != 0 && throw(PdfOxideError(scode[], op))
            bcode = Ref{Int32}(0)
            bold = ccall(
                (:pdf_oxide_word_is_bold, LIB),
                Bool,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                bcode,
            )
            bcode[] != 0 && throw(PdfOxideError(bcode[], op))
            out[i+1] = Word(txt, bb, font, Float64(fs), bold)
        end
        return out
    finally
        ccall((:pdf_oxide_word_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

function _lines_from_list(list::Ptr{Cvoid}, op::String)
    try
        n = ccall((:pdf_oxide_line_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{TextLine}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            txt = _take_string(
                ccall(
                    (:pdf_oxide_line_get_text, LIB),
                    Ptr{UInt8},
                    (Ptr{Cvoid}, Int32, Ref{Int32}),
                    list,
                    Int32(i),
                    Ref{Int32}(0),
                ),
                Int32(0),
                op,
            )
            bb = _bbox_line(list, i, op)
            wcode = Ref{Int32}(0)
            wc = ccall(
                (:pdf_oxide_line_get_word_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                wcode,
            )
            wcode[] != 0 && throw(PdfOxideError(wcode[], op))
            out[i+1] = TextLine(txt, bb, Int(wc))
        end
        return out
    finally
        ccall((:pdf_oxide_line_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

function _tables_from_list(list::Ptr{Cvoid}, op::String)
    try
        n = ccall((:pdf_oxide_table_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Table}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            rcode = Ref{Int32}(0)
            rows = ccall(
                (:pdf_oxide_table_get_row_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                rcode,
            )
            rcode[] != 0 && throw(PdfOxideError(rcode[], op))
            ccode = Ref{Int32}(0)
            cols = ccall(
                (:pdf_oxide_table_get_col_count, LIB),
                Int32,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                ccode,
            )
            ccode[] != 0 && throw(PdfOxideError(ccode[], op))
            hcode = Ref{Int32}(0)
            hdr = ccall(
                (:pdf_oxide_table_has_header, LIB),
                Bool,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(i),
                hcode,
            )
            hcode[] != 0 && throw(PdfOxideError(hcode[], op))
            nr = rows < 0 ? 0 : Int(rows)
            nc = cols < 0 ? 0 : Int(cols)
            cells = Matrix{String}(undef, nr, nc)
            for r = 0:(nr-1), c = 0:(nc-1)
                xcode = Ref{Int32}(0)
                cptr = ccall(
                    (:pdf_oxide_table_get_cell_text, LIB),
                    Ptr{UInt8},
                    (Ptr{Cvoid}, Int32, Int32, Int32, Ref{Int32}),
                    list,
                    Int32(i),
                    Int32(r),
                    Int32(c),
                    xcode,
                )
                cells[r+1, c+1] = _take_string(cptr, xcode[], op)
            end
            out[i+1] = Table(nr, nc, hdr, cells)
        end
        return out
    finally
        ccall((:pdf_oxide_table_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

function _images_from_list(list::Ptr{Cvoid}, op::String)
    try
        n = ccall((:pdf_oxide_image_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{Image}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            w = _i32_image_width(list, i, op)
            h = _i32_image_height(list, i, op)
            bpc = _i32_image_bpc(list, i, op)
            fmt = _str_image_format(list, i, op)
            cs = _str_image_colorspace(list, i, op)
            dlen = Ref{Int32}(0)
            dcode = Ref{Int32}(0)
            dptr = ccall(
                (:pdf_oxide_image_get_data, LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}, Ref{Int32}),
                list,
                Int32(i),
                dlen,
                dcode,
            )
            data = if dptr == C_NULL
                dcode[] != 0 && throw(PdfOxideError(dcode[], op))
                UInt8[]
            else
                m = dlen[] < 0 ? 0 : Int(dlen[])
                bytes = copy(unsafe_wrap(Array, dptr, m))
                ccall((:free_bytes, LIB), Cvoid, (Ptr{UInt8},), dptr)
                bytes
            end
            out[i+1] = Image(w, h, bpc, fmt, cs, data)
        end
        return out
    finally
        ccall((:pdf_oxide_image_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Words inside the rect `(x, y, w, h)` on a (0-based) page as `Vector{Word}`."""
extract_words_in_rect(d::PdfDocument, page::Integer, x, y, w, h) = _words_from_list(
    _open_words_rect(d, page, x, y, w, h, "extract_words_in_rect"),
    "extract_words_in_rect",
)

"""Lines inside the rect `(x, y, w, h)` on a (0-based) page as `Vector{TextLine}`."""
extract_lines_in_rect(d::PdfDocument, page::Integer, x, y, w, h) = _lines_from_list(
    _open_lines_rect(d, page, x, y, w, h, "extract_lines_in_rect"),
    "extract_lines_in_rect",
)

"""Tables inside the rect `(x, y, w, h)` on a (0-based) page as `Vector{Table}`."""
extract_tables_in_rect(d::PdfDocument, page::Integer, x, y, w, h) = _tables_from_list(
    _open_tables_rect(d, page, x, y, w, h, "extract_tables_in_rect"),
    "extract_tables_in_rect",
)

"""Images inside the rect `(x, y, w, h)` on a (0-based) page as `Vector{Image}`."""
extract_images_in_rect(d::PdfDocument, page::Integer, x, y, w, h) = _images_from_list(
    _open_images_rect(d, page, x, y, w, h, "extract_images_in_rect"),
    "extract_images_in_rect",
)

# ── Auto extraction / classification (-> String) ───────────────────────────────
"""Auto-pick the best text extraction for a (0-based) page (-> String)."""
function extract_text_auto(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_extract_text_auto, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    return _take_string(ptr, code[], "extract_text_auto")
end

"""Whole-document auto text extraction (-> String)."""
function extract_all_text(d::PdfDocument)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_extract_all_text, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    return _take_string(ptr, code[], "extract_all_text")
end

"""Auto page extraction with a JSON `options` string (-> String)."""
function extract_page_auto(d::PdfDocument, page::Integer, options::AbstractString = "{}")
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_extract_page_auto, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Cstring, Ref{Int32}),
        _doc(d),
        Int32(page),
        options,
        code,
    )
    return _take_string(ptr, code[], "extract_page_auto")
end

"""Classify a (0-based) page; returns the classifier's JSON string."""
function classify_page(d::PdfDocument, page::Integer)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_classify_page, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(page),
        code,
    )
    return _take_string(ptr, code[], "classify_page")
end

"""Classify the whole document; returns the classifier's JSON string."""
function classify_document(d::PdfDocument)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_classify_document, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    return _take_string(ptr, code[], "classify_document")
end

# ── Header / footer / artifact removal (mutating; -> count) ────────────────────
# Per-page eraser: (handle, page_index) -> i32 count.
for (jl_fn, c_fn) in (
    (:erase_header, :pdf_document_erase_header),
    (:erase_footer, :pdf_document_erase_footer),
    (:erase_artifacts, :pdf_document_erase_artifacts),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer)
        code = Ref{Int32}(0)
        n = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _doc(d),
            Int32(page),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], $op))
        return Int(n)
    end
end

# Document-wide remover: (handle, threshold) -> i32 count.
for (jl_fn, c_fn) in (
    (:remove_headers, :pdf_document_remove_headers),
    (:remove_footers, :pdf_document_remove_footers),
    (:remove_artifacts, :pdf_document_remove_artifacts),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, threshold::Real = 0.5)
        code = Ref{Int32}(0)
        n = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Float32, Ref{Int32}),
            _doc(d),
            Float32(threshold),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], $op))
        return Int(n)
    end
end

# ── Forms ─────────────────────────────────────────────────────────────────────
"""An AcroForm field: `name`, `value`, `type`, `readonly`, `required`."""
struct FormField
    name::String
    value::String
    type::String
    readonly::Bool
    required::Bool
end

# Per-field string accessor over a FfiFormFieldList handle (free_string return).
for (jl_fn, c_fn) in (
    (:_ff_name, :pdf_oxide_form_field_get_name),
    (:_ff_value, :pdf_oxide_form_field_get_value),
    (:_ff_type, :pdf_oxide_form_field_get_type),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        return _take_string(ptr, code[], op)
    end
end

# Per-field bool accessor over a FfiFormFieldList handle.
for (jl_fn, c_fn) in (
    (:_ff_readonly, :pdf_oxide_form_field_is_readonly),
    (:_ff_required, :pdf_oxide_form_field_is_required),
)
    @eval function $jl_fn(list::Ptr{Cvoid}, index::Integer, op::String)
        code = Ref{Int32}(0)
        v = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Bool,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], op))
        return v
    end
end

"""All AcroForm fields as a `Vector{FormField}` (empty when the doc has none)."""
function get_form_fields(d::PdfDocument)
    code = Ref{Int32}(0)
    list = ccall(
        (:pdf_document_get_form_fields, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    list == C_NULL && throw(PdfOxideError(code[], "get_form_fields"))
    try
        n = ccall((:pdf_oxide_form_field_count, LIB), Int32, (Ptr{Cvoid},), list)
        out = Vector{FormField}(undef, n < 0 ? 0 : Int(n))
        for i = 0:(Int(n)-1)
            out[i+1] = FormField(
                _ff_name(list, i, "get_form_fields"),
                _ff_value(list, i, "get_form_fields"),
                _ff_type(list, i, "get_form_fields"),
                _ff_readonly(list, i, "get_form_fields"),
                _ff_required(list, i, "get_form_fields"),
            )
        end
        return out
    finally
        ccall((:pdf_oxide_form_field_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# Lower-level form-field-list accessors (also exported for parity with the C ABI).
"""Field count of a freshly fetched form-field list (convenience over the doc)."""
function form_field_count(d::PdfDocument)
    code = Ref{Int32}(0)
    list = ccall(
        (:pdf_document_get_form_fields, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    list == C_NULL && throw(PdfOxideError(code[], "form_field_count"))
    try
        return Int(ccall((:pdf_oxide_form_field_count, LIB), Int32, (Ptr{Cvoid},), list))
    finally
        ccall((:pdf_oxide_form_field_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end
form_field_name(f::FormField) = f.name
form_field_value(f::FormField) = f.value
form_field_type(f::FormField) = f.type
form_field_is_readonly(f::FormField) = f.readonly
form_field_is_required(f::FormField) = f.required

"""Export form data as owned bytes in `format_type` (FDF/XFDF code)."""
function export_form_data_to_bytes(d::PdfDocument, format_type::Integer)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_export_form_data_to_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Int32, Ref{Csize_t}, Ref{Int32}),
        _doc(d),
        Int32(format_type),
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "export_form_data_to_bytes")
end

"""Import form data from a file `path`; returns the C status code."""
function import_form_data(d::PdfDocument, path::AbstractString)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_import_form_data, LIB),
        Int32,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _doc(d),
        path,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "import_form_data"))
    return Int(rc)
end

# Editor-side FDF/XFDF byte importers (mutating; -> status code).
for (jl_fn, c_fn) in (
    (:import_fdf_bytes, :pdf_editor_import_fdf_bytes),
    (:import_xfdf_bytes, :pdf_editor_import_xfdf_bytes),
)
    op = String(jl_fn)
    @eval function $jl_fn(e::DocumentEditor, data::AbstractVector{UInt8})
        code = Ref{Int32}(0)
        rc = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Int32}),
            _editor(e),
            data,
            Csize_t(length(data)),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], $op))
        return Int(rc)
    end
end

"""Import form data from `filename` into a document; returns `true` on success."""
function form_import_from_file(d::PdfDocument, filename::AbstractString)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_form_import_from_file, LIB),
        Bool,
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _doc(d),
        filename,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "form_import_from_file"))
    return ok
end

# ── Document structure / metadata (-> String / bytes / bool) ───────────────────
for (jl_fn, c_fn) in (
    (:get_outline, :pdf_document_get_outline),
    (:get_page_labels, :pdf_document_get_page_labels),
    (:get_xmp_metadata, :pdf_document_get_xmp_metadata),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            _doc(d),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""Plan a split-by-bookmarks with a JSON `options` string (-> JSON String)."""
function plan_split_by_bookmarks(d::PdfDocument, options::AbstractString = "{}")
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_plan_split_by_bookmarks, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Cstring, Ref{Int32}),
        _doc(d),
        options,
        code,
    )
    return _take_string(ptr, code[], "plan_split_by_bookmarks")
end

"""The document's original source bytes (owned; via free_bytes)."""
function get_source_bytes(d::PdfDocument)
    len = Ref{Csize_t}(0)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_document_get_source_bytes, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Csize_t}, Ref{Int32}),
        _doc(d),
        len,
        code,
    )
    return _take_bytes_uptr(ptr, len[], code[], "get_source_bytes")
end

"""Whether the document carries an XFA form (bool)."""
has_xfa(d::PdfDocument) = ccall((:pdf_document_has_xfa, LIB), Bool, (Ptr{Cvoid},), _doc(d))

"""Page count of a builder `Pdf` handle (mirrors `pdf_get_page_count`)."""
function get_page_count(p::Pdf)
    code = Ref{Int32}(0)
    n = ccall((:pdf_get_page_count, LIB), Int32, (Ptr{Cvoid}, Ref{Int32}), _pdf(p), code)
    code[] != 0 && throw(PdfOxideError(code[], "get_page_count"))
    return Int(n)
end

# ── Document-level signatures ──────────────────────────────────────────────────
"""Number of signatures in the document (-> Int)."""
function get_signature_count(d::PdfDocument)
    code = Ref{Int32}(0)
    n = ccall(
        (:pdf_document_get_signature_count, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "get_signature_count"))
    return Int(n)
end

"""The `index`-th signature as a `SignatureInfo` (owned; free via close!)."""
function get_signature(d::PdfDocument, index::Integer)
    code = Ref{Int32}(0)
    h = ccall(
        (:pdf_document_get_signature, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(index),
        code,
    )
    h == C_NULL && throw(PdfOxideError(code[], "get_signature"))
    return SignatureInfo(h)
end

"""Sign the document with `cert`, optional `reason`/`location`; -> status code."""
function sign(
    d::PdfDocument,
    cert::Certificate;
    reason::AbstractString = "",
    location::AbstractString = "",
)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_sign, LIB),
        Int32,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cstring, Cstring, Ref{Int32}),
        _doc(d),
        _cert(cert),
        reason,
        location,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "sign"))
    return Int(rc)
end

"""Verify all signatures in the document; -> status code."""
function verify_all_signatures(d::PdfDocument)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_verify_all_signatures, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "verify_all_signatures"))
    return Int(rc)
end

"""Whether the document carries a document-level timestamp; -> status code."""
function has_timestamp(d::PdfDocument)
    code = Ref{Int32}(0)
    rc = ccall(
        (:pdf_document_has_timestamp, LIB),
        Int32,
        (Ptr{Cvoid}, Ref{Int32}),
        _doc(d),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "has_timestamp"))
    return Int(rc)
end

"""Convert the document to PDF/A in place at `level`; returns `true` on success."""
function document_convert_to_pdf_a(d::PdfDocument, level::Integer)
    code = Ref{Int32}(0)
    ok = ccall(
        (:pdf_convert_to_pdf_a, LIB),
        Bool,
        (Ptr{Cvoid}, Int32, Ref{Int32}),
        _doc(d),
        Int32(level),
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "document_convert_to_pdf_a"))
    return ok
end

# ── Annotation extras (operate on the page annotation-list handle + index) ──────
"""32-bit packed RGBA color of the `index`-th annotation on a (0-based) page."""
function annotation_get_color(d::PdfDocument, page::Integer, index::Integer)
    list = _open_annotations(d, page, "annotation_get_color")
    try
        code = Ref{Int32}(0)
        v = ccall(
            (:pdf_oxide_annotation_get_color, LIB),
            UInt32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], "annotation_get_color"))
        return v
    finally
        ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# Int64 date accessors over the page annotation-list handle.
for (jl_fn, c_fn) in (
    (:annotation_creation_date, :pdf_oxide_annotation_get_creation_date),
    (:annotation_modification_date, :pdf_oxide_annotation_get_modification_date),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer, index::Integer)
        list = _open_annotations(d, page, $op)
        try
            code = Ref{Int32}(0)
            v = ccall(
                ($(QuoteNode(c_fn)), LIB),
                Int64,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(index),
                code,
            )
            code[] != 0 && throw(PdfOxideError(code[], $op))
            return Int(v)
        finally
            ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
        end
    end
end

# Bool flag accessors over the page annotation-list handle.
for (jl_fn, c_fn) in (
    (:annotation_is_hidden, :pdf_oxide_annotation_is_hidden),
    (:annotation_is_marked_deleted, :pdf_oxide_annotation_is_marked_deleted),
    (:annotation_is_printable, :pdf_oxide_annotation_is_printable),
    (:annotation_is_read_only, :pdf_oxide_annotation_is_read_only),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer, index::Integer)
        list = _open_annotations(d, page, $op)
        try
            code = Ref{Int32}(0)
            v = ccall(
                ($(QuoteNode(c_fn)), LIB),
                Bool,
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(index),
                code,
            )
            code[] != 0 && throw(PdfOxideError(code[], $op))
            return v
        finally
            ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
        end
    end
end

"""Quad-point count of the `index`-th highlight annotation on a (0-based) page."""
function highlight_quad_points_count(d::PdfDocument, page::Integer, index::Integer)
    list = _open_annotations(d, page, "highlight_quad_points_count")
    try
        code = Ref{Int32}(0)
        n = ccall(
            (:pdf_oxide_highlight_annotation_get_quad_points_count, LIB),
            Int32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], "highlight_quad_points_count"))
        return Int(n)
    finally
        ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""The `quad_index`-th quad of the `index`-th highlight annotation as 8 floats."""
function highlight_quad_point(
    d::PdfDocument,
    page::Integer,
    index::Integer,
    quad_index::Integer,
)
    list = _open_annotations(d, page, "highlight_quad_point")
    try
        x1 = Ref{Float32}(0);
        y1 = Ref{Float32}(0)
        x2 = Ref{Float32}(0);
        y2 = Ref{Float32}(0)
        x3 = Ref{Float32}(0);
        y3 = Ref{Float32}(0)
        x4 = Ref{Float32}(0);
        y4 = Ref{Float32}(0)
        code = Ref{Int32}(0)
        ccall(
            (:pdf_oxide_highlight_annotation_get_quad_point, LIB),
            Cvoid,
            (
                Ptr{Cvoid},
                Int32,
                Int32,
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Float32},
                Ref{Int32},
            ),
            list,
            Int32(index),
            Int32(quad_index),
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], "highlight_quad_point"))
        return (
            Float64(x1[]),
            Float64(y1[]),
            Float64(x2[]),
            Float64(y2[]),
            Float64(x3[]),
            Float64(y3[]),
            Float64(x4[]),
            Float64(y4[]),
        )
    finally
        ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# String annotation accessors over the page annotation-list handle.
for (jl_fn, c_fn) in (
    (:link_annotation_uri, :pdf_oxide_link_annotation_get_uri),
    (:text_annotation_icon_name, :pdf_oxide_text_annotation_get_icon_name),
)
    op = String(jl_fn)
    @eval function $jl_fn(d::PdfDocument, page::Integer, index::Integer)
        list = _open_annotations(d, page, $op)
        try
            code = Ref{Int32}(0)
            ptr = ccall(
                ($(QuoteNode(c_fn)), LIB),
                Ptr{UInt8},
                (Ptr{Cvoid}, Int32, Ref{Int32}),
                list,
                Int32(index),
                code,
            )
            return _take_string(ptr, code[], $op)
        finally
            ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
        end
    end
end

"""All annotations on a (0-based) page serialised to a JSON string."""
function annotations_to_json(d::PdfDocument, page::Integer)
    list = _open_annotations(d, page, "annotations_to_json")
    try
        code = Ref{Int32}(0)
        ptr = ccall(
            (:pdf_oxide_annotations_to_json, LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            list,
            code,
        )
        return _take_string(ptr, code[], "annotations_to_json")
    finally
        ccall((:pdf_oxide_annotation_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# ── Element / JSON accessors (operate on an ElementList handle) ─────────────────
# String element accessors over an ElementList handle.
for (jl_fn, c_fn) in (
    (:element_type, :pdf_oxide_element_get_type),
    (:element_text, :pdf_oxide_element_get_text),
)
    op = String(jl_fn)
    @eval function $jl_fn(l::ElementList, index::Integer)
        code = Ref{Int32}(0)
        ptr = ccall(
            ($(QuoteNode(c_fn)), LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            _elements(l),
            Int32(index),
            code,
        )
        return _take_string(ptr, code[], $op)
    end
end

"""The `index`-th element's bounding box as a `Bbox`."""
function element_rect(l::ElementList, index::Integer)
    x = Ref{Float32}(0);
    y = Ref{Float32}(0)
    w = Ref{Float32}(0);
    h = Ref{Float32}(0)
    code = Ref{Int32}(0)
    ccall(
        (:pdf_oxide_element_get_rect, LIB),
        Cvoid,
        (
            Ptr{Cvoid},
            Int32,
            Ref{Float32},
            Ref{Float32},
            Ref{Float32},
            Ref{Float32},
            Ref{Int32},
        ),
        _elements(l),
        Int32(index),
        x,
        y,
        w,
        h,
        code,
    )
    code[] != 0 && throw(PdfOxideError(code[], "element_rect"))
    return Bbox(Float64(x[]), Float64(y[]), Float64(w[]), Float64(h[]))
end

"""All page elements serialised to a JSON string."""
function elements_to_json(l::ElementList)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_oxide_elements_to_json, LIB),
        Ptr{UInt8},
        (Ptr{Cvoid}, Ref{Int32}),
        _elements(l),
        code,
    )
    return _take_string(ptr, code[], "elements_to_json")
end

"""Embedded fonts on a (0-based) page serialised to a JSON string."""
function fonts_to_json(d::PdfDocument, page::Integer)
    list = _open_fonts(d, page, "fonts_to_json")
    try
        code = Ref{Int32}(0)
        ptr = ccall(
            (:pdf_oxide_fonts_to_json, LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            list,
            code,
        )
        return _take_string(ptr, code[], "fonts_to_json")
    finally
        ccall((:pdf_oxide_font_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Font size of the `index`-th embedded font on a (0-based) page."""
function font_size(d::PdfDocument, page::Integer, index::Integer)
    list = _open_fonts(d, page, "font_size")
    try
        code = Ref{Int32}(0)
        v = ccall(
            (:pdf_oxide_font_get_size, LIB),
            Float32,
            (Ptr{Cvoid}, Int32, Ref{Int32}),
            list,
            Int32(index),
            code,
        )
        code[] != 0 && throw(PdfOxideError(code[], "font_size"))
        return Float64(v)
    finally
        ccall((:pdf_oxide_font_list_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

"""Serialise the results of a whole-document search for `term` to a JSON string."""
function search_results_to_json(d::PdfDocument, term::AbstractString, caseSensitive::Bool)
    code = Ref{Int32}(0)
    list = ccall(
        (:pdf_document_search_all, LIB),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cstring, Bool, Ref{Int32}),
        _doc(d),
        term,
        caseSensitive,
        code,
    )
    list == C_NULL && throw(PdfOxideError(code[], "search_results_to_json"))
    try
        jcode = Ref{Int32}(0)
        ptr = ccall(
            (:pdf_oxide_search_results_to_json, LIB),
            Ptr{UInt8},
            (Ptr{Cvoid}, Ref{Int32}),
            list,
            jcode,
        )
        return _take_string(ptr, jcode[], "search_results_to_json")
    finally
        ccall((:pdf_oxide_search_result_free, LIB), Cvoid, (Ptr{Cvoid},), list)
    end
end

# ── Crypto / FIPS ──────────────────────────────────────────────────────────────
# Parameterless string getters (free_string return, no error out-param).
for (jl_fn, c_fn) in (
    (:crypto_active_provider, :pdf_oxide_crypto_active_provider),
    (:crypto_cbom, :pdf_oxide_crypto_cbom),
    (:crypto_inventory, :pdf_oxide_crypto_inventory),
    (:crypto_policy, :pdf_oxide_crypto_policy),
    (:model_manifest, :pdf_oxide_model_manifest),
)
    op = String(jl_fn)
    @eval function $jl_fn()
        ptr = ccall(($(QuoteNode(c_fn)), LIB), Ptr{UInt8}, ())
        return _take_string(ptr, Int32(0), $op)
    end
end

# Parameterless i32 getters (no error out-param).
for (jl_fn, c_fn) in (
    (:crypto_fips_available, :pdf_oxide_crypto_fips_available),
    (:crypto_use_fips, :pdf_oxide_crypto_use_fips),
    (:prefetch_available, :pdf_oxide_prefetch_available),
)
    @eval $jl_fn() = Int(ccall(($(QuoteNode(c_fn)), LIB), Int32, ()))
end

"""Set the active crypto policy from `spec`; -> status code."""
crypto_set_policy(spec::AbstractString) =
    Int(ccall((:pdf_oxide_crypto_set_policy, LIB), Int32, (Cstring,), spec))

# ── Models / config ────────────────────────────────────────────────────────────
"""Prefetch OCR/layout models for the comma-separated `languages_csv`; -> String."""
function prefetch_models(languages_csv::AbstractString)
    code = Ref{Int32}(0)
    ptr = ccall(
        (:pdf_oxide_prefetch_models, LIB),
        Ptr{UInt8},
        (Cstring, Ref{Int32}),
        languages_csv,
        code,
    )
    return _take_string(ptr, code[], "prefetch_models")
end

"""Set the per-stream content-op limit; returns the previous limit."""
set_max_ops_per_stream(limit::Integer) =
    Int(ccall((:pdf_oxide_set_max_ops_per_stream, LIB), Int64, (Int64,), Int64(limit)))

"""Toggle preservation of unmapped glyphs; returns the previous setting."""
set_preserve_unmapped_glyphs(preserve::Integer) = Int(
    ccall((:pdf_oxide_set_preserve_unmapped_glyphs, LIB), Int32, (Int32,), Int32(preserve)),
)

end # module
