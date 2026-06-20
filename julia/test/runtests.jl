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

    # ── Page model ────────────────────────────────────────────────────────────
    let pg = page(doc, 0)                          # page
        @test occursin("Alpha", text(pg))         # Page.text
        @test !isempty(markdown(pg))              # Page.markdown
        @test occursin("<", html(pg))            # Page.html
        @test !isempty(plain_text(pg))           # Page.plain_text
    end

    # ── Error path ────────────────────────────────────────────────────────────
    @test_throws PdfOxideError open_document("/nonexistent/nope.pdf")
end
