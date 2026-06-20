# pdf_oxide — idiomatic Julia bindings over the C ABI via ccall.
#
# Loads the native cdylib (libpdf_oxide) at runtime; handles are wrapped in
# mutable structs with finalizers; C strings/buffers are copied into Julia and
# freed via free_string; non-success C-ABI error codes throw PdfOxideError.
#
# API surface mirrors the other language bindings; coverage is asserted by
# test/runtests.jl (one test per public method).
module PdfOxide

export PdfDocument, Pdf, PdfOxideError, PdfVersion
export open_document, open_from_bytes, open_with_password
export page_count, version, is_encrypted, has_structure_tree
export extract_text,
    to_plain_text, to_markdown, to_html, to_markdown_all, extract_structured_json
export from_markdown, from_html, from_text, save, to_bytes, close!

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
