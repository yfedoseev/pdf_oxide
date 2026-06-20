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
)
    @eval $jl_fn(p::PdfPage) = $doc_fn(p.doc, p.index)
end

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

end # module
