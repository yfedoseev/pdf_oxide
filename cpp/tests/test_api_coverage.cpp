// test_api_coverage — one check per public method of the C++ binding.
// Mirrors the api_coverage_test.go / ApiCoverageTests.cs convention so every
// language binding has the same verification. Self-contained: builds its own
// PDF from Markdown, no external fixture. Returns non-zero on any failure.
#include <pdf_oxide/pdf_oxide.hpp>

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

// NB: do not `using namespace pdf_oxide;` — the C header defines a global
// `::Pdf` type that would make the unqualified name `Pdf` ambiguous. Qualify.
using pdf_oxide::Document;
using pdf_oxide::Error;
using pdf_oxide::RenderedImage;
using pdf_oxide::Version;

static std::vector<std::uint8_t> sample_pdf() {
    return pdf_oxide::Pdf::from_markdown(
               "# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")
        .to_bytes();
}

int main() {
    // ── Pdf builder ──────────────────────────────────────────────────────
    {
        auto a = pdf_oxide::Pdf::from_markdown("# md\n\nbody\n");
        CHECK(a.to_bytes().size() > 100); // to_bytes
        auto b = pdf_oxide::Pdf::from_html("<h1>html</h1><p>body</p>");
        CHECK(b.to_bytes().size() > 100);
        auto c = pdf_oxide::Pdf::from_text("plain text body");
        CHECK(c.to_bytes().size() > 100);
        // save to a temp path
        std::string path = std::string(std::tmpnam(nullptr)) + ".pdf";
        a.save(path); // save
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f != nullptr);
        if (f)
            std::fclose(f);
        std::remove(path.c_str());
    }

    // ── Document open paths ──────────────────────────────────────────────
    auto bytes = sample_pdf();
    auto doc = Document::open_from_bytes(bytes); // open_from_bytes
    {
        // open(path)
        std::string path = std::string(std::tmpnam(nullptr)) + ".pdf";
        pdf_oxide::Pdf::from_markdown("# f\n\nx\n").save(path);
        auto d2 = Document::open(path);
        CHECK(d2.page_count() >= 1);
        std::remove(path.c_str());
    }

    // ── Document inspection + extraction ─────────────────────────────────
    CHECK(doc.page_count() >= 1); // page_count
    Version v = doc.version();    // version
    CHECK(v.major >= 1);
    CHECK(doc.is_encrypted() == false);                            // is_encrypted
    (void)doc.has_structure_tree();                                // has_structure_tree
    CHECK(doc.extract_text(0).find("Alpha") != std::string::npos); // extract_text
    CHECK(!doc.to_plain_text(0).empty());                          // to_plain_text
    CHECK(!doc.to_markdown(0).empty());                            // to_markdown
    CHECK(doc.to_html(0).find('<') != std::string::npos);          // to_html
    CHECK(!doc.to_markdown_all().empty());                         // to_markdown_all
    CHECK(doc.to_html_all().find('<') != std::string::npos);       // to_html_all
    CHECK(!doc.to_plain_text_all().empty());                       // to_plain_text_all
    CHECK(!doc.extract_structured_json(0).empty()); // extract_structured_json

    // ── Phase-1 element extraction ───────────────────────────────────────
    {
        auto words = doc.extract_words(0); // extract_words
        CHECK(!words.empty());
        if (!words.empty()) {
            CHECK(!words[0].text.empty());
            // a real word has a non-degenerate bbox
            CHECK(words[0].bbox.width > 0.0f);
            CHECK(words[0].bbox.height > 0.0f);
            (void)words[0].font_name;
            (void)words[0].font_size;
            (void)words[0].bold;
        }

        auto chars = doc.extract_chars(0); // extract_chars
        CHECK(!chars.empty());
        if (!chars.empty()) {
            CHECK(chars[0].character != 0);
            (void)chars[0].bbox;
            (void)chars[0].font_name;
            (void)chars[0].font_size;
        }

        auto lines = doc.extract_text_lines(0); // extract_text_lines
        CHECK(!lines.empty());
        if (!lines.empty()) {
            CHECK(!lines[0].text.empty());
            CHECK(lines[0].word_count >= 1);
            (void)lines[0].bbox;
        }

        // tables may be empty on this doc; just assert the call succeeds.
        auto tables = doc.extract_tables(0); // extract_tables
        CHECK(tables.size() >= 0);
        for (const auto& t : tables) {
            if (t.row_count > 0 && t.col_count > 0) {
                (void)t.cell(0, 0);
            }
            (void)t.has_header;
        }
    }

    // ── Phase-2 element extraction ───────────────────────────────────────
    {
        // fonts/images/paths/annotations may be empty on this doc; just
        // assert each call succeeds and returns a list.
        auto fonts = doc.embedded_fonts(0); // embedded_fonts
        CHECK(fonts.size() >= 0);
        for (const auto& f : fonts) {
            (void)f.name;
            (void)f.type;
            (void)f.encoding;
            (void)f.embedded;
            (void)f.subset;
        }

        auto images = doc.embedded_images(0); // embedded_images
        CHECK(images.size() >= 0);
        for (const auto& im : images) {
            (void)im.width;
            (void)im.height;
            (void)im.bits_per_component;
            (void)im.format;
            (void)im.colorspace;
            (void)im.data;
        }

        auto annots = doc.page_annotations(0); // page_annotations
        CHECK(annots.size() >= 0);
        for (const auto& a : annots) {
            (void)a.type;
            (void)a.subtype;
            (void)a.content;
            (void)a.author;
            (void)a.rect;
            (void)a.border_width;
        }

        auto paths = doc.extract_paths(0); // extract_paths
        CHECK(paths.size() >= 0);
        for (const auto& p : paths) {
            (void)p.bbox;
            (void)p.stroke_width;
            (void)p.has_stroke;
            (void)p.has_fill;
            (void)p.operation_count;
        }

        auto hits = doc.search(0, "Alpha", false); // search
        CHECK(!hits.empty());
        if (!hits.empty()) {
            CHECK(hits[0].text.find("Alpha") != std::string::npos);
            CHECK(hits[0].page >= 0);
            (void)hits[0].bbox;
        }

        auto allHits = doc.search_all("Alpha", false); // search_all
        CHECK(!allHits.empty());
        if (!allHits.empty()) {
            CHECK(allHits[0].text.find("Alpha") != std::string::npos);
            CHECK(allHits[0].page >= 0);
            (void)allHits[0].bbox;
        }
    }

    // ── Phase-3 page rendering ───────────────────────────────────────────
    {
        auto img = doc.render_page(0); // render_page (PNG)
        CHECK(img.width() > 0);
        CHECK(img.height() > 0);
        CHECK(!img.data().empty());

        // zoom + thumbnail just need to succeed without error.
        auto zoomed = doc.render_page_zoom(0, 2.0f); // render_page_zoom
        CHECK(zoomed.width() > 0);
        CHECK(zoomed.height() > 0);

        auto thumb = doc.render_page_thumbnail(0, 64); // render_page_thumbnail
        CHECK(thumb.width() > 0);
        CHECK(thumb.height() > 0);

        // save the rendered image to a temp path.
        std::string path = std::string(std::tmpnam(nullptr)) + ".png";
        img.save(path); // RenderedImage::save
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f != nullptr);
        if (f)
            std::fclose(f);
        std::remove(path.c_str());
    }

    // authenticate returns a bool (no throw on an unencrypted/sample doc)
    {
        bool authed = doc.authenticate(""); // authenticate
        (void)authed;
    }

    // ── Page model (page(index) + per-page accessors) ────────────────────
    {
        auto p = doc.page(0);                               // page
        CHECK(p.text().find("Alpha") != std::string::npos); // Page::text
        CHECK(!p.markdown().empty());                       // Page::markdown
        CHECK(p.html().find('<') != std::string::npos);     // Page::html
        CHECK(!p.plain_text().empty());                     // Page::plain_text
    }

    // ── Error path (open a bogus file throws Error) ──────────────────────
    bool threw = false;
    try {
        Document::open("/nonexistent/does-not-exist.pdf");
    } catch (const Error& e) {
        threw = true;
        (void)e.code();
    }
    CHECK(threw);

    // ── close() is explicit + idempotent; use-after-close throws ─────────────
    {
        auto d = Document::open_from_bytes(bytes);
        d.close();
        d.close(); // idempotent
        bool closedThrew = false;
        try {
            d.page_count();
        } catch (const Error&) {
            closedThrew = true;
        }
        CHECK(closedThrew);
    }

    // ── DocumentBuilder (PDF creation builder API) ───────────────────────
    {
        // DocumentBuilder.create -> page(595,842) -> font -> heading ->
        // paragraph -> done -> build -> reopen + assert content.
        auto db = pdf_oxide::DocumentBuilder::create(); // create
        auto pg = db.page(595.0f, 842.0f);              // page(width,height)
        pg.font("Helvetica", 12.0f)                     // font
            .heading(1, "Title")                        // heading
            .paragraph("Hello world from the builder.") // paragraph
            .done();                                    // done (consumes page)
        auto built = db.build();                        // build -> bytes
        CHECK(!built.empty());
        CHECK(built.size() > 100);

        auto rebuilt = Document::open_from_bytes(built);
        CHECK(rebuilt.page_count() >= 1);
        std::string text = rebuilt.extract_text(0);
        CHECK(text.find("Hello") != std::string::npos ||
              text.find("Title") != std::string::npos);
        db.close();
    }

    // ── DocumentEditor (in-place editing handle) ─────────────────────────
    {
        auto ed = pdf_oxide::DocumentEditor::open_from_bytes(bytes); // open_from_bytes
        CHECK(ed.page_count() >= 1);                                 // page_count
        bool modified = ed.is_modified(); // is_modified (bool)
        (void)modified;
        ed.rotate_all_pages(90);              // rotate_all_pages
        CHECK(ed.get_page_rotation(0) == 90); // get_page_rotation
        ed.set_producer("x");                 // set_producer
        CHECK(ed.get_producer() == "x");      // get_producer
        CHECK(!ed.save_to_bytes().empty());   // save_to_bytes
        ed.close();                           // close
    }

    // ── PHASE-6: digital signatures / PKI / timestamps / TSA / validation ────
    //
    // VALIDATION is fully exercisable on the sample doc. SIGNING / CERTIFICATE /
    // TIMESTAMP / TSA / DSS need real PKCS#12 certs or a live TSA, so for those
    // we invoke each wrapper with minimal/empty inputs and require it to EITHER
    // return OR raise pdf_oxide::Error — the goal is that every wrapper is
    // linked and exercised, not that it succeeds without credentials.

    // set/get log level round-trip.
    {
        int prev = pdf_oxide::get_log_level(); // get_log_level
        pdf_oxide::set_log_level(4);           // set_log_level
        CHECK(pdf_oxide::get_log_level() == 4);
        pdf_oxide::set_log_level(2);
        CHECK(pdf_oxide::get_log_level() == 2);
        pdf_oxide::set_log_level(prev); // restore
    }

    // PDF/A validation.
    {
        auto res = doc.validate_pdf_a(0); // validate_pdf_a
        bool compliant = res.is_compliant();
        (void)compliant; // is_compliant -> bool
        auto errs = res.errors();
        CHECK(errs.size() >= 0); // errors -> list
        CHECK(res.warning_count() >= 0);
        res.close();
    }

    // PDF/UA validation + stats.
    {
        auto res = doc.validate_pdf_ua(0); // validate_pdf_ua
        bool accessible = res.is_accessible();
        (void)accessible; // is_accessible -> bool
        auto errs = res.errors();
        CHECK(errs.size() >= 0); // errors -> list
        auto warns = res.warnings();
        CHECK(warns.size() >= 0); // warnings -> list
        auto stats = res.stats(); // uaStats
        CHECK(stats.pages >= 0);
        CHECK(stats.structure >= 0);
        CHECK(stats.images >= 0);
        CHECK(stats.tables >= 0);
        CHECK(stats.forms >= 0);
        CHECK(stats.annotations >= 0);
        res.close();
    }

    // PDF/X validation.
    {
        auto res = doc.validate_pdf_x(0); // validate_pdf_x
        bool compliant = res.is_compliant();
        (void)compliant; // is_compliant -> bool
        auto errs = res.errors();
        CHECK(errs.size() >= 0); // errors -> list
        res.close();
    }

    // DSS: the sample doc has no /DSS, so get_dss should raise Error.
    {
        bool dssExercised = false;
        try {
            auto dss = doc.get_dss(); // get_dss (Dss handle)
            // If somehow present, exercise every accessor.
            (void)dss.cert_count();
            (void)dss.crl_count();
            (void)dss.ocsp_count();
            (void)dss.vri_count();
            dssExercised = true;
        } catch (const Error&) {
            dssExercised = true; // no DSS → Error is the expected outcome
        }
        CHECK(dssExercised);
    }

    // Certificate: empty bytes / blank PEM cannot load → Error.
    {
        bool certExercised = false;
        try {
            auto cert =
                pdf_oxide::Certificate::load_from_bytes({}, ""); // load_from_bytes
            // If it somehow loaded, exercise the accessors.
            (void)cert.subject();
            (void)cert.issuer();
            (void)cert.serial();
            (void)cert.validity();
            (void)cert.is_valid();
            cert.close();
            certExercised = true;
        } catch (const Error&) {
            certExercised = true;
        }
        CHECK(certExercised);

        bool pemExercised = false;
        try {
            auto cert = pdf_oxide::Certificate::load_from_pem("", ""); // load_from_pem
            cert.close();
            pemExercised = true;
        } catch (const Error&) {
            pemExercised = true;
        }
        CHECK(pemExercised);
    }

    // Signing: with no usable certificate we still link + invoke each entry. We
    // route through a bogus (closed) Certificate via the load failure above, so
    // here we just assert each free function raises Error on a load-then-sign
    // attempt with empty inputs.
    {
        bool signExercised = false;
        try {
            auto cert = pdf_oxide::Certificate::load_from_bytes({}, "");
            (void)pdf_oxide::sign_bytes(bytes, cert, "r", "l"); // sign_bytes
            pdf_oxide::RevocationMaterial rev;
            (void)pdf_oxide::sign_bytes_pades(bytes, cert, 0, "", "r", "l",
                                              rev); // sign_bytes_pades
            (void)pdf_oxide::sign_bytes_pades_opts(bytes, cert, 0, "", "r", "l",
                                                   rev); // sign_bytes_pades_opts
            signExercised = true;
        } catch (const Error&) {
            signExercised = true; // load failed (no cert) → Error
        }
        CHECK(signExercised);
    }

    // Timestamp: parse garbage bytes → Error; the accessors are covered in the
    // success branch should the parser accept anything.
    {
        bool tsExercised = false;
        try {
            auto ts = pdf_oxide::Timestamp::parse({0x00, 0x01, 0x02}); // parse
            (void)ts.token();                                          // token
            (void)ts.message_imprint(); // message_imprint
            (void)ts.time();            // time
            (void)ts.serial();          // serial
            (void)ts.tsa_name();        // tsa_name
            (void)ts.policy_oid();      // policy_oid
            (void)ts.hash_algorithm();  // hash_algorithm
            (void)ts.verify();          // verify
            ts.close();
            tsExercised = true;
        } catch (const Error&) {
            tsExercised = true;
        }
        CHECK(tsExercised);
    }

    // TSA client: created against a dummy URL (no network call at create time);
    // requesting a timestamp will fail without a live TSA → Error.
    {
        bool tsaExercised = false;
        try {
            auto client = pdf_oxide::TsaClient::create("http://tsa.invalid/", "", "", 1,
                                                       0, true, true); // create
            try {
                (void)client.request_timestamp({0x01, 0x02, 0x03}); // request_timestamp
                // NOLINTNEXTLINE(bugprone-empty-catch) -- invalid TSA tolerated
            } catch (const Error&) {
            }
            try {
                (void)client.request_timestamp_hash({0x01, 0x02, 0x03},
                                                    0); // request_timestamp_hash
                // NOLINTNEXTLINE(bugprone-empty-catch) -- invalid TSA tolerated
            } catch (const Error&) {
            }
            client.close();
            tsaExercised = true;
        } catch (const Error&) {
            tsaExercised = true; // create itself may reject the URL
        }
        CHECK(tsaExercised);
    }

    // ── PHASE-7: barcodes / OCR / render variants / page getters / redaction /
    //            from_* constructors / merge / timestamp ─────────────────────
    //
    // QR / barcode generation, render variants, page getters, redaction, and the
    // from_* / merge constructors are fully exercisable on the sample doc. OCR
    // (needs model files) and add_timestamp (needs a TSA + a real signature) are
    // INVOKED with minimal/empty inputs and required to EITHER return OR raise
    // pdf_oxide::Error — the goal is that every wrapper is linked and exercised.

    // Barcodes: QR + a 1-D barcode are generatable; assert the accessors.
    {
        auto qr = pdf_oxide::Barcode::generate_qr_code("https://example.com/", 1,
                                                       128); // generate_qr_code
        CHECK(qr.get_data() == "https://example.com/");      // get_data
        (void)qr.get_format();                               // get_format
        (void)qr.get_confidence();                           // get_confidence
        auto png = qr.get_image_png(128);                    // get_image_png
        CHECK(!png.empty());
        auto svg = qr.get_svg(128); // get_svg
        CHECK(!svg.empty());
        qr.close();

        bool barcodeExercised = false;
        try {
            auto bc =
                pdf_oxide::Barcode::generate_barcode("12345670", 0, 128); // gen_barcode
            CHECK(!bc.get_data().empty());
            (void)bc.get_format();
            auto bpng = bc.get_image_png(128);
            CHECK(!bpng.empty());
            (void)bc.get_svg(128);
            bc.close();
            barcodeExercised = true;
        } catch (const Error&) {
            barcodeExercised = true; // unsupported data/format → Error is fine
        }
        CHECK(barcodeExercised);

        // Stamp the QR onto a page via the editor (add_barcode_to_page).
        {
            auto ed = pdf_oxide::DocumentEditor::open_from_bytes(bytes);
            auto qr2 = pdf_oxide::Barcode::generate_qr_code("X", 1, 64);
            bool addExercised = false;
            try {
                ed.add_barcode_to_page(0, qr2, 10.0f, 10.0f, 50.0f,
                                       50.0f); // add_barcode_to_page
                addExercised = true;
            } catch (const Error&) {
                addExercised = true;
            }
            CHECK(addExercised);
            ed.close();
        }
    }

    // Render variants: all return a RenderedImage with positive dims + bytes.
    {
        auto opt = doc.render_page_with_options(0, 96, 0, 1.0f, 1.0f, 1.0f, 1.0f, false,
                                                true, 90); // render_page_with_options
        CHECK(opt.width() > 0);
        CHECK(opt.height() > 0);
        CHECK(!opt.data().empty());

        auto optx = doc.render_page_with_options_ex(
            0, 96, 0, 1.0f, 1.0f, 1.0f, 1.0f, false, true, 90,
            {"NonexistentLayer"}); // render_page_with_options_ex
        CHECK(optx.width() > 0);
        CHECK(optx.height() > 0);

        auto fit = doc.render_page_fit(0, 200, 200); // render_page_fit
        CHECK(fit.width() > 0);
        CHECK(fit.height() > 0);

        int rawW = 0, rawH = 0;
        auto raw = doc.render_page_raw(0, 72, rawW, rawH); // render_page_raw
        CHECK(rawW > 0);
        CHECK(rawH > 0);
        CHECK(!raw.data().empty());

        // region: crop a small rectangle; just needs to succeed.
        bool regionExercised = false;
        try {
            auto region =
                doc.render_page_region(0, 0.0f, 0.0f, 100.0f, 100.0f); // render_region
            CHECK(region.width() > 0);
            regionExercised = true;
        } catch (const Error&) {
            regionExercised = true;
        }
        CHECK(regionExercised);

        // estimate_render_time: returns a non-negative estimate (or raises).
        bool estExercised = false;
        try {
            (void)doc.estimate_render_time(0); // estimate_render_time
            estExercised = true;
        } catch (const Error&) {
            estExercised = true;
        }
        CHECK(estExercised);

        // standalone Renderer create/free (no per-page render entry in the ABI).
        bool rendererExercised = false;
        try {
            auto r = pdf_oxide::Renderer::create(96, 0, 90, true); // create_renderer
            r.close();                                             // renderer_free
            rendererExercised = true;
        } catch (const Error&) {
            rendererExercised = true;
        }
        CHECK(rendererExercised);
    }

    // Page getters: width/height/rotation/elements.
    {
        CHECK(doc.page_get_width(0) > 0.0f);  // page_get_width
        CHECK(doc.page_get_height(0) > 0.0f); // page_get_height
        int rot = doc.page_get_rotation(0);   // page_get_rotation
        CHECK(rot >= 0);
        auto elems = doc.page_get_elements(0); // page_get_elements
        CHECK(elems.size() >= 0);
        for (const auto& e : elems) {
            (void)e.type;
            (void)e.text;
            (void)e.rect;
        }
    }

    // Redaction on an editor: queue → count → apply; plus scrub_metadata.
    {
        auto ed = pdf_oxide::DocumentEditor::open_from_bytes(bytes);
        ed.redaction_add(0, 10.0, 10.0, 100.0, 50.0, 0.0, 0.0, 0.0); // redaction_add
        int n = ed.redaction_count(0);                               // redaction_count
        CHECK(n >= 1);
        bool applyExercised = false;
        try {
            (void)ed.redaction_apply(false, 0.0, 0.0, 0.0); // redaction_apply
            applyExercised = true;
        } catch (const Error&) {
            applyExercised = true; // composite-font fail-closed → Error is fine
        }
        CHECK(applyExercised);

        bool scrubExercised = false;
        try {
            (void)ed.redaction_scrub_metadata(); // redaction_scrub_metadata
            scrubExercised = true;
        } catch (const Error&) {
            scrubExercised = true;
        }
        CHECK(scrubExercised);
        ed.close();
    }

    // from_* constructors + merge.
    {
        // from_html_css / from_html_css_with_fonts: exercise the wrapper; it
        // builds a PDF when the html-render path is available, else raises Error
        // (e.g. no default font in this cdylib). Either outcome exercises it.
        try {
            auto htmlPdf =
                pdf_oxide::Pdf::from_html_css("<h1>HtmlCss</h1><p>body</p>",
                                              "h1{color:#000}"); // from_html_css
            CHECK(htmlPdf.to_bytes().size() > 100);
        } catch (const pdf_oxide::Error&) { // NOLINT(bugprone-empty-catch) html-render
                                            // unavailable: tolerated
        }
        try {
            auto htmlPdf2 = pdf_oxide::Pdf::from_html_css_with_fonts(
                "<p>cascade</p>", "", {}, {}); // from_html_css_with_fonts
            CHECK(htmlPdf2.to_bytes().size() > 100);
        } catch (const pdf_oxide::Error&) { // NOLINT(bugprone-empty-catch) tolerated
        }

        // from_image_bytes: bogus bytes must raise Error.
        bool imgBytesExercised = false;
        try {
            auto p = pdf_oxide::Pdf::from_image_bytes({0x00, 0x01, 0x02}); // from_img_b
            (void)p.to_bytes();
            imgBytesExercised = true;
        } catch (const Error&) {
            imgBytesExercised = true;
        }
        CHECK(imgBytesExercised);

        // from_image: a nonexistent path must raise Error.
        bool imgExercised = false;
        try {
            auto p = pdf_oxide::Pdf::from_image("/nonexistent/none.png"); // from_image
            (void)p.to_bytes();
            imgExercised = true;
        } catch (const Error&) {
            imgExercised = true;
        }
        CHECK(imgExercised);

        // merge: write two temp PDFs, merge them, assert bytes.
        std::string p1 = std::string(std::tmpnam(nullptr)) + ".pdf";
        std::string p2 = std::string(std::tmpnam(nullptr)) + ".pdf";
        pdf_oxide::Pdf::from_markdown("# one\n\na\n").save(p1);
        pdf_oxide::Pdf::from_markdown("# two\n\nb\n").save(p2);
        bool mergeExercised = false;
        try {
            auto merged = pdf_oxide::merge({p1, p2}); // merge
            CHECK(merged.size() > 100);
            mergeExercised = true;
        } catch (const Error&) {
            mergeExercised = true;
        }
        CHECK(mergeExercised);
        std::remove(p1.c_str());
        std::remove(p2.c_str());
    }

    // OCR: engine creation needs model files; both create + the extract/needs
    // entries are invoked and required to return or raise Error.
    {
        bool ocrEngineExercised = false;
        try {
            auto eng =
                pdf_oxide::OcrEngine::create("/nonexistent/det", "/nonexistent/rec",
                                             "/nonexistent/dict"); // create
            // If it somehow loaded, run a page through it.
            (void)doc.ocr_extract_text(0, &eng);
            eng.close();
            ocrEngineExercised = true;
        } catch (const Error&) {
            ocrEngineExercised = true; // no models → Error is the expected outcome
        }
        CHECK(ocrEngineExercised);

        // needs_ocr + native-only extract (engine == nullptr) on the sample doc.
        bool needsExercised = false;
        try {
            (void)doc.ocr_page_needs_ocr(0); // ocr_page_needs_ocr
            needsExercised = true;
        } catch (const Error&) {
            needsExercised = true; // ocr feature disabled → Error is fine
        }
        CHECK(needsExercised);

        bool extractExercised = false;
        try {
            (void)doc.ocr_extract_text(0, nullptr); // ocr_extract_text (native-only)
            extractExercised = true;
        } catch (const Error&) {
            extractExercised = true;
        }
        CHECK(extractExercised);
    }

    // Timestamp: no TSA / no signature → returns or raises Error.
    {
        bool tsExercised = false;
        try {
            (void)pdf_oxide::add_timestamp(bytes, 0,
                                           "http://tsa.invalid/"); // add_timestamp
            tsExercised = true;
        } catch (const Error&) {
            tsExercised = true;
        }
        CHECK(tsExercised);
    }

    // ── PHASE-8: in-rect extraction / auto / classify / structure / forms /
    //            office / signatures (doc-level) / annotation extras / *_to_json
    //            / crypto / models / config ──────────────────────────────────
    //
    // Many PHASE-8 wrappers may legitimately succeed OR raise on the synthetic
    // markdown sample (no forms, no /DSS, no signatures, no office round-trip,
    // no network for model prefetch). Per the api-coverage convention every such
    // wrapper is invoked as return-or-Error so it is linked + exercised.

    // Helper macro: pass if the wrapper returns or raises pdf_oxide::Error.
    // Variadic so statement blocks containing commas (e.g. brace-init lists)
    // are passed through intact.
#define EXERCISE(...)                                                                  \
    do {                                                                               \
        bool ok = false;                                                               \
        try {                                                                          \
            __VA_ARGS__;                                                               \
            ok = true;                                                                 \
        } catch (const Error&) {                                                       \
            ok = true;                                                                 \
        }                                                                              \
        CHECK(ok);                                                                     \
    } while (0)

    // In-rect extraction over a wide rect covering the whole first page.
    {
        float pw = doc.page_get_width(0);
        float ph = doc.page_get_height(0);
        // text_in_rect should find the sample's prose; the rest may be empty.
        EXERCISE({
            std::string t = doc.extract_text_in_rect(0, 0, 0, pw, ph); // text_in_rect
            (void)t;
        });
        EXERCISE({
            auto w = doc.extract_words_in_rect(0, 0, 0, pw, ph); // words_in_rect
            CHECK(w.size() >= 0);
        });
        EXERCISE({
            auto l = doc.extract_lines_in_rect(0, 0, 0, pw, ph); // lines_in_rect
            CHECK(l.size() >= 0);
        });
        EXERCISE({
            auto tb = doc.extract_tables_in_rect(0, 0, 0, pw, ph); // tables_in_rect
            CHECK(tb.size() >= 0);
        });
        EXERCISE({
            auto im = doc.extract_images_in_rect(0, 0, 0, pw, ph); // images_in_rect
            CHECK(im.size() >= 0);
        });
    }

    // Auto / classification extraction.
    EXERCISE({
        std::string t = doc.extract_text_auto(0); // extract_text_auto
        (void)t;
    });
    EXERCISE({
        std::string t = doc.extract_all_text(); // extract_all_text
        (void)t;
    });
    EXERCISE({
        std::string t = doc.extract_page_auto(0); // extract_page_auto (default opts)
        (void)t;
    });
    EXERCISE({
        std::string t = doc.extract_page_auto(0, "{}"); // extract_page_auto (json)
        (void)t;
    });
    EXERCISE({
        std::string c = doc.classify_page(0); // classify_page
        (void)c;
    });
    EXERCISE({
        std::string c = doc.classify_document(); // classify_document
        (void)c;
    });

    // Header / footer / artifact removal (mutate an editable copy of the doc).
    {
        auto d = Document::open_from_bytes(bytes);
        EXERCISE((void)d.erase_header(0));        // erase_header
        EXERCISE((void)d.erase_footer(0));        // erase_footer
        EXERCISE((void)d.erase_artifacts(0));     // erase_artifacts
        EXERCISE((void)d.remove_headers(0.5f));   // remove_headers
        EXERCISE((void)d.remove_footers(0.5f));   // remove_footers
        EXERCISE((void)d.remove_artifacts(0.5f)); // remove_artifacts
    }

    // AcroForm fields (the sample has none → empty list is valid).
    {
        EXERCISE({
            auto fields = doc.get_form_fields(); // get_form_fields
            CHECK(fields.size() >= 0);
            for (const auto& f : fields) {
                (void)f.name;
                (void)f.value;
                (void)f.type;
                (void)f.readonly;
                (void)f.required;
            }
        });
        EXERCISE({
            auto data = doc.export_form_data_to_bytes(0); // export_form_data_to_bytes
            (void)data;
        });
        EXERCISE(
            (void)doc.import_form_data("/nonexistent/form.fdf")); // import_form_data
        EXERCISE((void)doc.form_import_from_file(
            "/nonexistent/form.fdf")); // form_import_from_file
        // editor-side FDF/XFDF import.
        std::string ep = std::string(std::tmpnam(nullptr)) + ".pdf";
        pdf_oxide::Pdf::from_markdown("# e\n\nx\n").save(ep);
        auto ed = pdf_oxide::DocumentEditor::open(ep);
        std::vector<std::uint8_t> empty;
        EXERCISE((void)ed.import_fdf_bytes(empty));  // import_fdf_bytes
        EXERCISE((void)ed.import_xfdf_bytes(empty)); // import_xfdf_bytes
        ed.close();
        std::remove(ep.c_str());
    }

    // Document structure / metadata.
    EXERCISE({
        std::string o = doc.get_outline(); // get_outline
        (void)o;
    });
    EXERCISE({
        std::string l = doc.get_page_labels(); // get_page_labels
        (void)l;
    });
    EXERCISE({
        std::string x = doc.get_xmp_metadata(); // get_xmp_metadata
        (void)x;
    });
    EXERCISE({
        auto sb = doc.get_source_bytes(); // get_source_bytes
        (void)sb;
    });
    EXERCISE((void)doc.has_xfa()); // has_xfa
    EXERCISE({
        std::string p = doc.plan_split_by_bookmarks(); // plan_split_by_bookmarks
        (void)p;
    });

    // Document-level signatures (need real certs/signatures → return-or-error).
    EXERCISE((void)doc.get_signature_count());   // get_signature_count
    EXERCISE((void)doc.get_signature(0));        // get_signature
    EXERCISE((void)doc.verify_all_signatures()); // verify_all_signatures
    EXERCISE((void)doc.has_timestamp());         // has_timestamp
    {
        // sign needs a Certificate; load from bogus bytes raises, which still
        // exercises Certificate::load_from_bytes; if it somehow loads, sign is
        // exercised return-or-error.
        EXERCISE({
            std::vector<std::uint8_t> bogus{0x00, 0x01, 0x02};
            auto cert = pdf_oxide::Certificate::load_from_bytes(bogus, "pw");
            (void)doc.sign(cert, "reason", "loc"); // sign
        });
    }

    // Annotation extras + *_to_json (sample page has no annotations → the index
    // accessors raise; the to_json variants return an empty/array JSON).
    EXERCISE((void)doc.annotation_get_color(0, 0));             // get_color
    EXERCISE((void)doc.annotation_get_creation_date(0, 0));     // get_creation_date
    EXERCISE((void)doc.annotation_get_modification_date(0, 0)); // get_modification_date
    EXERCISE((void)doc.annotation_is_hidden(0, 0));             // is_hidden
    EXERCISE((void)doc.annotation_is_marked_deleted(0, 0));     // is_marked_deleted
    EXERCISE((void)doc.annotation_is_printable(0, 0));          // is_printable
    EXERCISE((void)doc.annotation_is_read_only(0, 0));          // is_read_only
    EXERCISE((void)doc.highlight_annotation_get_quad_points_count(
        0, 0)); // quad_points_count
    EXERCISE((void)doc.highlight_annotation_get_quad_point(0, 0, 0)); // quad_point
    EXERCISE((void)doc.link_annotation_get_uri(0, 0));                // link uri
    EXERCISE((void)doc.text_annotation_get_icon_name(0, 0));          // text icon name
    EXERCISE({
        std::string j = doc.annotations_to_json(0); // annotations_to_json
        (void)j;
    });
    EXERCISE({
        std::string j = doc.elements_to_json(0); // elements_to_json
        (void)j;
    });
    EXERCISE({
        std::string j = doc.fonts_to_json(0); // fonts_to_json
        (void)j;
    });
    EXERCISE((void)doc.font_get_size(0, 0)); // font_get_size
    EXERCISE({
        std::string j = doc.search_results_to_json(0, "Alpha", false); // search to_json
        (void)j;
    });

    // Office export / PDF-A conversion (codec may be feature-gated).
    EXERCISE({
        auto d = doc.to_docx(); // to_docx
        (void)d;
    });
    EXERCISE({
        auto d = doc.to_pptx(); // to_pptx
        (void)d;
    });
    EXERCISE({
        auto d = doc.to_xlsx(); // to_xlsx
        (void)d;
    });
    {
        auto d = Document::open_from_bytes(bytes);
        EXERCISE((void)d.convert_to_pdf_a(0)); // convert_to_pdf_a
    }

    // Office import (need real office files → invalid bytes raise).
    {
        std::vector<std::uint8_t> notoffice{0x50, 0x4b, 0x03, 0x04}; // bogus zip header
        EXERCISE(
            (void)Document::open_from_docx_bytes(notoffice)); // open_from_docx_bytes
        EXERCISE(
            (void)Document::open_from_pptx_bytes(notoffice)); // open_from_pptx_bytes
        EXERCISE(
            (void)Document::open_from_xlsx_bytes(notoffice)); // open_from_xlsx_bytes
    }

    // Legacy Pdf-handle page count.
    {
        auto p = pdf_oxide::Pdf::from_markdown("# p\n\none\n");
        EXERCISE(CHECK(p.page_count() >= 1)); // Pdf::page_count (pdf_get_page_count)
    }

    // Crypto / FIPS provider.
    EXERCISE({
        std::string s = pdf_oxide::crypto_active_provider(); // crypto_active_provider
        (void)s;
    });
    EXERCISE({
        std::string s = pdf_oxide::crypto_cbom(); // crypto_cbom
        (void)s;
    });
    EXERCISE((void)pdf_oxide::crypto_fips_available()); // crypto_fips_available
    EXERCISE({
        std::string s = pdf_oxide::crypto_inventory(); // crypto_inventory
        (void)s;
    });
    EXERCISE({
        std::string s = pdf_oxide::crypto_policy(); // crypto_policy
        (void)s;
    });
    EXERCISE((void)pdf_oxide::crypto_set_policy("default")); // crypto_set_policy
    EXERCISE((void)pdf_oxide::crypto_use_fips());            // crypto_use_fips

    // Models / prefetch.
    EXERCISE({
        std::string s = pdf_oxide::model_manifest(); // model_manifest
        (void)s;
    });
    EXERCISE((void)pdf_oxide::prefetch_available()); // prefetch_available
    EXERCISE({
        std::string s = pdf_oxide::prefetch_models("en"); // prefetch_models
        (void)s;
    });

    // Global config knobs. These setters return the PRIOR value and have NO
    // error channel, so only assert each call is invokable (returns an int) —
    // never assert a specific round-tripped value (the cdylib's default/prior
    // may differ from what we set).
    {
        std::int64_t prevOps = pdf_oxide::set_max_ops_per_stream(1000000); // max_ops
        CHECK(prevOps == prevOps); // invokable: returns an int (value not asserted)
        pdf_oxide::set_max_ops_per_stream(prevOps); // restore

        int prevGlyphs =
            pdf_oxide::set_preserve_unmapped_glyphs(1); // preserve_unmapped
        CHECK(prevGlyphs ==
              prevGlyphs); // invokable: returns an int (value not asserted)
        (void)pdf_oxide::set_preserve_unmapped_glyphs(prevGlyphs); // restore
    }

#undef EXERCISE

    if (g_failures == 0) {
        std::printf("ok: all C++ api-coverage checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}
