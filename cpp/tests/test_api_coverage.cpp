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

    if (g_failures == 0) {
        std::printf("ok: all C++ api-coverage checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}
