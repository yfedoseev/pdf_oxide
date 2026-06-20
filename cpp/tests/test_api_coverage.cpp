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

    if (g_failures == 0) {
        std::printf("ok: all C++ api-coverage checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}
