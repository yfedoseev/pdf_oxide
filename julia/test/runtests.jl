# One @testset item per public function — mirrors the api_coverage convention
# used by every pdf_oxide binding. Self-contained: builds its own PDF.
using PdfOxide
using Test
using Aqua

sample_pdf() =
    to_bytes(from_markdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n"))

# Package-quality checks (stale deps, [compat] coverage, undefined exports,
# project-file consistency). persistent_tasks disabled — this is an FFI shim
# that loads a native lib, not relevant to that probe.
@testset "Aqua quality" begin
    Aqua.test_all(PdfOxide; persistent_tasks = false)
end

@testset "PdfOxide api coverage" begin
    # ── Pdf builder ───────────────────────────────────────────────────────────
    @test length(to_bytes(from_markdown("# md\n\nbody\n"))) > 100
    @test length(to_bytes(from_html("<h1>h</h1><p>b</p>"))) > 100
    @test length(to_bytes(from_text("plain text body"))) > 100
    let tmp = tempname() * ".pdf"
        save(from_markdown("# f\n\nx\n"), tmp)
        @test isfile(tmp)
        rm(tmp; force = true)
    end

    # ── Document open paths ───────────────────────────────────────────────────
    doc = open_from_bytes(sample_pdf())          # open_from_bytes
    @test page_count(doc) >= 1                    # page_count
    let tmp = tempname() * ".pdf"
        save(from_markdown("# f\n\nx\n"), tmp)
        d2 = open_document(tmp)                    # open_document
        @test page_count(d2) >= 1
        rm(tmp; force = true)
    end

    # ── Document inspection + extraction ──────────────────────────────────────
    @test version(doc).major >= 1                     # version
    @test is_encrypted(doc) == false              # is_encrypted
    has_structure_tree(doc)                        # has_structure_tree (smoke)
    @test occursin("Alpha", extract_text(doc, 0)) # extract_text
    @test !isempty(to_plain_text(doc, 0))         # to_plain_text
    @test !isempty(to_markdown(doc, 0))           # to_markdown
    @test occursin("<", to_html(doc, 0))          # to_html
    @test !isempty(to_markdown_all(doc))          # to_markdown_all
    @test !isempty(extract_structured_json(doc, 0)) # extract_structured_json
    @test occursin("<", to_html_all(doc))         # to_html_all
    @test !isempty(to_plain_text_all(doc))        # to_plain_text_all
    @test authenticate(doc, "") isa Bool          # authenticate (bool, no throw)

    # ── Element extraction ────────────────────────────────────────────────────
    let words = extract_words(doc, 0)             # extract_words
        @test !isempty(words)
        @test !isempty(words[1].text)
        @test words[1].bbox isa Bbox
        @test words[1].bbox.width >= 0
        @test words[1].font_size >= 0
        @test words[1].bold isa Bool
    end
    let chars = extract_chars(doc, 0)             # extract_chars
        @test !isempty(chars)
        @test chars[1].character isa UInt32
        @test chars[1].bbox isa Bbox
    end
    let lines = extract_text_lines(doc, 0)        # extract_text_lines
        @test !isempty(lines)
        @test !isempty(lines[1].text)
        @test lines[1].word_count >= 0
        @test lines[1].bbox isa Bbox
    end
    let tables = extract_tables(doc, 0)           # extract_tables
        @test tables isa Vector{Table}            # may be empty — just returns w/o error
        for t in tables
            @test t.row_count >= 0
            @test t.col_count >= 0
            @test t.has_header isa Bool
            if t.row_count > 0 && t.col_count > 0
                @test cell(t, 0, 0) isa String
            end
        end
    end

    # ── Phase-2 extraction ────────────────────────────────────────────────────
    let fonts = embedded_fonts(doc, 0)            # embedded_fonts
        @test fonts isa Vector{Font}              # may be empty — just call succeeds
        for f in fonts
            @test f.name isa String
            @test f.type isa String
            @test f.encoding isa String
            @test f.embedded isa Bool
            @test f.subset isa Bool
        end
    end
    let images = embedded_images(doc, 0)          # embedded_images
        @test images isa Vector{Image}            # may be empty — just call succeeds
        for im in images
            @test im.width >= 0
            @test im.height >= 0
            @test im.bitsPerComponent >= 0
            @test im.format isa String
            @test im.colorspace isa String
            @test im.data isa Vector{UInt8}
        end
    end
    let annots = page_annotations(doc, 0)         # page_annotations
        @test annots isa Vector{Annotation}       # may be empty — just call succeeds
        for a in annots
            @test a.type isa String
            @test a.subtype isa String
            @test a.content isa String
            @test a.author isa String
            @test a.rect isa Bbox
            @test a.borderWidth >= 0
        end
    end
    let paths = extract_paths(doc, 0)             # extract_paths
        @test paths isa Vector{Path}              # may be empty — just call succeeds
        for pa in paths
            @test pa.bbox isa Bbox
            @test pa.strokeWidth >= 0
            @test pa.hasStroke isa Bool
            @test pa.hasFill isa Bool
            @test pa.operationCount >= 0
        end
    end
    let hits = search(doc, 0, "Alpha", false)     # search
        @test !isempty(hits)
        @test occursin("Alpha", hits[1].text)
        @test hits[1].page >= 0
        @test hits[1].bbox isa Bbox
    end
    let hits = search_all(doc, "Alpha", false)    # search_all
        @test !isempty(hits)
        @test occursin("Alpha", hits[1].text)
        @test hits[1].page >= 0
        @test hits[1].bbox isa Bbox
    end

    # ── Page model ────────────────────────────────────────────────────────────
    let pg = page(doc, 0)                          # page
        @test occursin("Alpha", text(pg))         # Page.text
        @test !isempty(markdown(pg))              # Page.markdown
        @test occursin("<", html(pg))            # Page.html
        @test !isempty(plain_text(pg))           # Page.plain_text
        @test embedded_fonts(pg) isa Vector{Font}        # Page.embedded_fonts
        @test embedded_images(pg) isa Vector{Image}      # Page.embedded_images
        @test page_annotations(pg) isa Vector{Annotation} # Page.page_annotations
        @test extract_paths(pg) isa Vector{Path}         # Page.extract_paths
        @test !isempty(search(pg, "Alpha", false))       # Page.search
    end

    # ── Error path ────────────────────────────────────────────────────────────
    @test_throws PdfOxideError open_document("/nonexistent/nope.pdf")
end
