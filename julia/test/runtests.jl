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

    # ── Phase-3 rendering ─────────────────────────────────────────────────────
    let img = render_page(doc, 0)                  # render_page (PNG default)
        @test img isa RenderedImage
        @test img.width > 0
        @test img.height > 0
        @test !isempty(img.data)
        let tmp = tempname() * ".png"
            save(img, tmp)                         # RenderedImage save
            @test isfile(tmp)
            rm(tmp; force = true)
        end
    end
    @test render_page_zoom(doc, 0, 2.0f0) isa RenderedImage     # render_page_zoom
    @test render_page_thumbnail(doc, 0, 128) isa RenderedImage  # render_page_thumbnail
    @test renderPage(doc, 0) isa RenderedImage                  # camelCase alias

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
        @test render_page(pg) isa RenderedImage           # Page.render_page
    end

    # ── DocumentEditor ────────────────────────────────────────────────────────
    let ed = open_editor_from_bytes(sample_pdf())   # open_editor_from_bytes
        @test page_count(ed) >= 1                    # pageCount
        @test version(ed).major >= 1                 # version
        @test is_modified(ed) isa Bool               # isModified (bool)
        rotate_all_pages(ed, 90)                      # rotateAllPages
        @test get_page_rotation(ed, 0) == 90          # getPageRotation
        set_producer(ed, "x")                         # setProducer
        @test get_producer(ed) == "x"                 # getProducer
        @test !isempty(save_to_bytes(ed))             # saveToBytes
        close!(ed)                                     # close
    end

    # ── Error path ────────────────────────────────────────────────────────────
    @test_throws PdfOxideError open_document("/nonexistent/nope.pdf")
    @test_throws PdfOxideError open_editor("/nonexistent/nope.pdf")
end

@testset "PdfOxide builder api coverage" begin
    # DocumentBuilder.create -> page(595,842) -> font -> heading -> paragraph ->
    # (free page builder) -> build() -> reopen -> assert page count + content.
    b = DocumentBuilder()                              # DocumentBuilder()
    set_title(b, "Builder Title")                      # set_title
    pg = page(b, 595.0f0, 842.0f0)                     # DocumentBuilder.page
    font(pg, "Helvetica", 12.0f0)                      # PageBuilder.font
    heading(pg, 1, "Title")                            # PageBuilder.heading
    paragraph(pg, "Hello world from the builder.")     # PageBuilder.paragraph
    done(pg)                                            # PageBuilder.done (consumes page)
    bytes = build(b)                                    # DocumentBuilder.build
    @test !isempty(bytes)
    @test length(bytes) > 100
    close!(b)                                           # DocumentBuilder.close

    let doc = open_from_bytes(bytes)
        @test page_count(doc) >= 1
        txt = extract_text(doc, 0)
        @test occursin("Hello", txt) || occursin("Title", txt)
        close!(doc)
    end

    # letter_page is the standard-page alternative; smoke it through build too.
    let b2 = DocumentBuilder()
        lp = letter_page(b2)                            # DocumentBuilder.letter_page
        font(lp, "Helvetica", 12.0f0)
        paragraph(lp, "Letter page body.")
        done(lp)
        @test !isempty(build(b2))
        close!(b2)
    end

    # EmbeddedFont path: only the standard-font route is asserted here (no font
    # file is bundled). The from_bytes loader must reject non-font bytes — assert
    # it raises rather than requiring a real TTF/OTF asset.
    @test_throws PdfOxideError embedded_font_from_bytes(UInt8[0x00, 0x01, 0x02])
    @test_throws PdfOxideError embedded_font_from_file("/nonexistent/font.ttf")
end

@testset "PdfOxide phase-6 signatures/PKI/validation coverage" begin
    # ── Logging round-trip ────────────────────────────────────────────────────
    set_log_level(2)                               # set_log_level
    @test get_log_level() == 2                     # get_log_level
    set_log_level(1)
    @test get_log_level() == 1

    # ── Validation (fully exercisable on a real document) ─────────────────────
    doc = open_from_bytes(sample_pdf())

    let ra = validate_pdf_a(doc, 0)                 # validate_pdf_a
        @test is_compliant(ra) isa Bool            # is_compliant (PDF/A)
        @test errors(ra) isa Vector{String}        # errors (PDF/A)
        @test warnings(ra) isa Vector{String}      # warnings (PDF/A)
        @test pdf_a_error_count(ra) >= 0           # pdf_a_error_count
        @test pdf_a_warning_count(ra) >= 0         # pdf_a_warning_count
        @test length(errors(ra)) == pdf_a_error_count(ra)
        close!(ra)
    end
    @test validatePdfA(doc, 0) isa PdfAResults     # validatePdfA alias

    let ru = validate_pdf_ua(doc, 0)               # validate_pdf_ua
        @test is_accessible(ru) isa Bool           # is_accessible
        @test errors(ru) isa Vector{String}        # errors (PDF/UA)
        @test warnings(ru) isa Vector{String}      # warnings (PDF/UA)
        @test pdf_ua_error_count(ru) >= 0          # pdf_ua_error_count
        @test pdf_ua_warning_count(ru) >= 0        # pdf_ua_warning_count
        let st = ua_stats(ru)                       # ua_stats
            @test st.structure >= 0
            @test st.images >= 0
            @test st.tables >= 0
            @test st.forms >= 0
            @test st.annotations >= 0
            @test st.pages >= 0
        end
        close!(ru)
    end
    @test validatePdfUa(doc, 0) isa UaResults      # validatePdfUa alias

    let rx = validate_pdf_x(doc, 0)                 # validate_pdf_x
        @test is_compliant(rx) isa Bool            # is_compliant (PDF/X)
        @test errors(rx) isa Vector{String}        # errors (PDF/X)
        @test warnings(rx) isa Vector{String}      # warnings (PDF/X — empty)
        @test pdf_x_error_count(rx) >= 0           # pdf_x_error_count
        close!(rx)
    end
    @test validatePdfX(doc, 0) isa PdfXResults     # validatePdfX alias

    # ── DSS: a plain document has no /DSS → nothing (not an error) ─────────────
    @test document_get_dss(doc) === nothing        # document_get_dss

    # ── Certificate / signing: no real PKCS12 cert nor network available, so
    #    assert each wrapper is reached and raises the binding error type. ─────
    @test_throws PdfOxideError certificate_load_from_bytes(UInt8[0x00, 0x01, 0x02], "")
    @test_throws PdfOxideError certificate_load_from_pem("not-a-pem", "not-a-key")

    # ── Timestamp: parsing junk DER must raise; exercises timestamp_parse. ─────
    @test_throws PdfOxideError timestamp_parse(UInt8[0x00, 0x01, 0x02, 0x03])

    # ── TSA client: creation may succeed (no I/O); the request paths do I/O,
    #    so assert they either return a Timestamp or raise the binding error. ──
    let made = nothing
        try
            made = tsa_client_create("http://127.0.0.1:0/tsa"; timeout = 1)
        catch e
            @test e isa PdfOxideError
        end
        if made isa TsaClient
            # NB: the throwing call must be OUTSIDE @test — `@test f()` captures the
            # throw as a test-error and the surrounding catch never fires.
            try
                r = tsa_request_timestamp(made, UInt8[0x01, 0x02])
                @test r isa Timestamp
            catch e
                @test e isa PdfOxideError                # tsa_request_timestamp
            end
            try
                r = tsa_request_timestamp_hash(made, zeros(UInt8, 32), 0)
                @test r isa Timestamp
            catch e
                @test e isa PdfOxideError                # tsa_request_timestamp_hash
            end
            close!(made)
        end
    end

    # ── Signing top-level wrappers need a real cert handle; build one via the
    #    PEM loader inside a try so the whole signing surface is exercised even
    #    when no key material is present. ───────────────────────────────────────
    let pdfbytes = sample_pdf(), cert = nothing
        try
            cert = certificate_load_from_pem("not-a-pem", "not-a-key")
        catch e
            @test e isa PdfOxideError
        end
        if cert isa Certificate
            # Certificate accessors (only reachable with a valid handle).
            @test certificate_get_subject(cert) isa String
            @test certificate_get_issuer(cert) isa String
            @test certificate_get_serial(cert) isa String
            @test certificate_get_validity(cert) isa Tuple
            @test certificate_is_valid(cert) isa Bool
            try
                @test sign_bytes(pdfbytes, cert, "r", "l") isa Vector{UInt8}
            catch e
                @test e isa PdfOxideError                # sign_bytes
            end
            try
                @test sign_bytes_pades(pdfbytes, cert, 0, nothing, "r", "l") isa
                      Vector{UInt8}
            catch e
                @test e isa PdfOxideError                # sign_bytes_pades
            end
            try
                @test sign_bytes_pades_opts(pdfbytes, cert, 0, nothing, "r", "l") isa
                      Vector{UInt8}
            catch e
                @test e isa PdfOxideError                # sign_bytes_pades_opts
            end
            close!(cert)
        else
            # Even without a cert, ensure the signing entry points are defined and
            # raise on a closed/invalid certificate handle (closed-handle guard).
            badcert = Certificate(Ptr{Cvoid}(0))
            @test_throws ErrorException sign_bytes(pdfbytes, badcert, "r", "l")
            @test_throws ErrorException sign_bytes_pades(
                pdfbytes,
                badcert,
                0,
                nothing,
                "r",
                "l",
            )
            @test_throws ErrorException sign_bytes_pades_opts(
                pdfbytes,
                badcert,
                0,
                nothing,
                "r",
                "l",
            )
        end
    end

    # ── SignatureInfo / Timestamp accessor wrappers: no signed document is
    #    available, so drive them against a closed handle to confirm each is
    #    defined and guarded (closed-handle guard raises ErrorException). ───────
    let s = SignatureInfo(Ptr{Cvoid}(0)), t = Timestamp(Ptr{Cvoid}(0))
        @test_throws ErrorException signature_get_signer_name(s)
        @test_throws ErrorException signature_get_signing_reason(s)
        @test_throws ErrorException signature_get_signing_location(s)
        @test_throws ErrorException signature_get_signing_time(s)
        @test_throws ErrorException signature_get_certificate(s)
        @test_throws ErrorException signature_get_pades_level(s)
        @test_throws ErrorException signature_has_timestamp(s)
        @test_throws ErrorException signature_get_timestamp(s)
        @test_throws ErrorException signature_add_timestamp(s, t)
        @test_throws ErrorException signature_verify(s)
        @test_throws ErrorException signature_verify_detached(s, sample_pdf())

        @test_throws ErrorException timestamp_get_token(t)
        @test_throws ErrorException timestamp_get_message_imprint(t)
        @test_throws ErrorException timestamp_get_time(t)
        @test_throws ErrorException timestamp_get_serial(t)
        @test_throws ErrorException timestamp_get_tsa_name(t)
        @test_throws ErrorException timestamp_get_policy_oid(t)
        @test_throws ErrorException timestamp_get_hash_algorithm(t)
        @test_throws ErrorException timestamp_verify(t)
    end

    # ── DSS accessor wrappers against a closed handle (closed-handle guard). ───
    let d = Dss(Ptr{Cvoid}(0))
        @test_throws ErrorException dss_cert_count(d)
        @test_throws ErrorException dss_crl_count(d)
        @test_throws ErrorException dss_ocsp_count(d)
        @test_throws ErrorException dss_vri_count(d)
        @test_throws ErrorException dss_get_cert(d, 0)
        @test_throws ErrorException dss_get_crl(d, 0)
        @test_throws ErrorException dss_get_ocsp(d, 0)
    end

    close!(doc)
end
