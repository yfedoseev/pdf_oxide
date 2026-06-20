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

end # module
