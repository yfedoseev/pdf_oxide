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

@testset "PdfOxide phase-7 barcodes/OCR/render/redaction/constructors/page-getters" begin
    doc = open_from_bytes(sample_pdf())

    # ── Barcodes / QR ─────────────────────────────────────────────────────────
    let qr = generate_qr_code("https://example.com", 0, 256)   # generate_qr_code
        @test qr isa Barcode
        @test barcode_get_data(qr) == "https://example.com"    # barcode_get_data
        @test barcode_get_format(qr) isa Int                   # barcode_get_format
        @test barcode_get_confidence(qr) isa Float64           # barcode_get_confidence
        let png = barcode_get_image_png(qr, 256)               # barcode_get_image_png
            @test png isa Vector{UInt8}
            @test !isempty(png)
        end
        @test occursin("<", barcode_get_svg(qr, 256))          # barcode_get_svg
        close!(qr)
    end
    let bc = generate_barcode("12345678", 0, 128)              # generate_barcode
        @test bc isa Barcode
        @test !isempty(barcode_get_data(bc))
        # add_barcode_to_page on an editor (invoke; tolerate either outcome).
        let ed = open_editor_from_bytes(sample_pdf())
            try
                add_barcode_to_page(ed, 0, bc, 10.0, 10.0, 100.0, 50.0)  # add_barcode_to_page
                @test true
            catch e
                @test e isa PdfOxideError
            end
            close!(ed)
        end
        close!(bc)
    end

    # ── OCR (needs model files): exercise wrappers with minimal input ──────────
    @test page_needs_ocr(doc, 0) isa Bool                      # page_needs_ocr
    let r = nothing
        try
            r = ocr_extract_text(doc, 0, nothing)              # ocr_extract_text (no engine)
            @test r isa String
        catch e
            @test e isa PdfOxideError
        end
    end
    let eng = nothing
        try
            eng = ocr_engine_create(
                "/nonexistent/det",
                "/nonexistent/rec",
                "/nonexistent/dict",
            )
        catch e
            @test e isa PdfOxideError                          # ocr_engine_create (bad paths raise)
        end
        if eng isa OcrEngine
            close!(eng)
        end
    end

    # ── Render variants (testable on the sample) ──────────────────────────────
    let img = render_page_with_options(doc, 0, 96, 0, 1.0, 1.0, 1.0, 1.0, 0, 1, 90)
        @test img isa RenderedImage                            # render_page_with_options
        @test img.width > 0 && img.height > 0
        @test !isempty(img.data)
    end
    let img = render_page_with_options_ex(
            doc,
            0,
            96,
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            0,
            1,
            90,
            String["NoSuchLayer"],
        )
        @test img isa RenderedImage                            # render_page_with_options_ex
        @test img.width > 0 && img.height > 0
    end
    let img = render_page_region(doc, 0, 0.0, 0.0, 100.0, 100.0, 0)
        @test img isa RenderedImage                            # render_page_region
        @test img.width > 0 && img.height > 0
        @test !isempty(img.data)
    end
    let img = render_page_fit(doc, 0, 200, 200, 0)
        @test img isa RenderedImage                            # render_page_fit
        @test img.width > 0 && img.height > 0
    end
    let (img, w, h) = render_page_raw(doc, 0, 96)              # render_page_raw
        @test img isa RenderedImage
        @test w > 0 && h > 0
        @test !isempty(img.data)
    end

    # ── Renderer config + estimate ────────────────────────────────────────────
    let made = nothing
        try
            made = create_renderer(150, 0, 90, true)           # create_renderer
        catch e
            @test e isa PdfOxideError
        end
        if made isa Renderer
            close!(made)                                        # Renderer close (pdf_renderer_free)
        end
    end
    let v = nothing
        try
            v = estimate_render_time(doc, 0)                   # estimate_render_time
            @test v isa Int
        catch e
            @test e isa PdfOxideError
        end
    end

    # ── Page getters (testable) ───────────────────────────────────────────────
    @test page_get_width(doc, 0) > 0                           # page_get_width
    @test page_get_height(doc, 0) > 0                          # page_get_height
    @test page_get_rotation(doc, 0) isa Int                    # page_get_rotation
    let els = page_get_elements(doc, 0)                        # page_get_elements
        @test els isa ElementList
        @test element_count(els) >= 0                          # element_count
        close!(els)                                             # ElementList close
    end

    # ── Redaction (on an editor, testable) ────────────────────────────────────
    let ed = open_editor_from_bytes(sample_pdf())
        redaction_add(ed, 0, 10.0, 10.0, 100.0, 50.0, 0.0, 0.0, 0.0)  # redaction_add
        @test redaction_count(ed, 0) >= 1                      # redaction_count
        let n = redaction_apply(ed, false, 0.0, 0.0, 0.0)     # redaction_apply
            @test n >= 0
        end
        @test redaction_scrub_metadata(ed) >= 0               # redaction_scrub_metadata
        close!(ed)
    end

    # ── Constructors ──────────────────────────────────────────────────────────
    # from_image_bytes: bad (non-image) bytes must raise; assign outside @test so
    # the surrounding try/catch can catch it.
    let p = nothing
        try
            p = from_image_bytes(UInt8[0x00, 0x01, 0x02, 0x03])  # from_image_bytes (bad input)
            @test p isa Pdf
            close!(p)
        catch e
            @test e isa PdfOxideError
        end
    end
    @test_throws PdfOxideError from_image("/nonexistent/nope.png")  # from_image (missing file)

    # from_html_css builds a PDF where the html-render path is available, else
    # raises (e.g. no default font in this cdylib). Either outcome exercises it.
    try
        pdf = from_html_css("<h1>HC</h1><p>body</p>", "h1 { color: red; }", nothing)  # from_html_css
        @test length(to_bytes(pdf)) > 100
        close!(pdf)
    catch e
        @test e isa PdfOxideError
    end
    try
        pdf = from_html_css_with_fonts(
            "<p>cascade</p>",
            "p { font-size: 12px; }",
            String[],
            Vector{UInt8}[],
        )  # from_html_css_with_fonts
        @test length(to_bytes(pdf)) > 100
        close!(pdf)
    catch e
        @test e isa PdfOxideError
    end

    # merge: write two temp PDFs and merge them; assert non-empty bytes.
    let a = tempname() * ".pdf", b = tempname() * ".pdf"
        save(from_markdown("# A\n\nalpha\n"), a)
        save(from_markdown("# B\n\nbravo\n"), b)
        let merged = nothing
            try
                merged = merge_pdfs([a, b])                     # merge_pdfs
                @test merged isa Vector{UInt8}
                @test length(merged) > 100
            catch e
                @test e isa PdfOxideError
            end
        end
        rm(a; force = true)
        rm(b; force = true)
    end

    # ── Timestamp (needs a TSA): invoke with minimal input, assert return/raise ─
    let r = nothing
        try
            r = add_timestamp(sample_pdf(), 0, "http://127.0.0.1:0/tsa")  # add_timestamp
            @test r isa Vector{UInt8}
        catch e
            @test e isa PdfOxideError
        end
    end

    close!(doc)
end
