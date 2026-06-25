// pdf_oxide — idiomatic C++17 RAII bindings over the C ABI.
//
// Header-only: every method is a thin inline wrapper around the C functions in
// <pdf_oxide_c/pdf_oxide.h>. Handles are owned (move-only); C strings/buffers
// returned by the core are copied into std::string and freed via free_string().
//
// API surface mirrors the other language bindings (Go/C#/Ruby). Coverage is
// asserted by tests/test_api_coverage.cpp (one test per public method).
#ifndef PDF_OXIDE_HPP
#define PDF_OXIDE_HPP

extern "C" {
#include <pdf_oxide_c/pdf_oxide.h>
}

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pdf_oxide {

/// Thrown on any non-success C-ABI error code.
class Error : public std::runtime_error {
  public:
    explicit Error(int32_t code, const std::string& op)
        : std::runtime_error("pdf_oxide: " + op + " failed (error code " +
                             std::to_string(code) + ")"),
          code_(code) {}
    int32_t code() const noexcept { return code_; }

  private:
    int32_t code_;
};

namespace detail {

/// Take ownership of a C string return, copy to std::string, free it.
/// Throws Error(code, op) when `s` is null (the C ABI's failure signal).
inline std::string take_string(char* s, int32_t code, const char* op) {
    if (s == nullptr) {
        throw Error(code, op);
    }
    std::string out(s);
    free_string(s);
    return out;
}

inline std::vector<std::uint8_t> take_bytes(std::uint8_t* p, std::size_t len,
                                            int32_t code, const char* op) {
    if (p == nullptr) {
        throw Error(code, op);
    }
    std::vector<std::uint8_t> out(p, p + len);
    // Raw byte buffers MUST be freed with free_bytes, not free_string
    // (free_string does strlen on a non-NUL-terminated buffer → overflow).
    free_bytes(p);
    return out;
}

} // namespace detail

/// PDF version (e.g. 1.7).
struct Version {
    std::uint8_t major;
    std::uint8_t minor;
};

/// An axis-aligned bounding box in page coordinates.
struct Bbox {
    float x;
    float y;
    float width;
    float height;
};

/// A single extracted glyph/character.
struct Char {
    std::uint32_t character; // Unicode codepoint
    Bbox bbox;
    std::string font_name;
    float font_size;
};

/// A single extracted word.
struct Word {
    std::string text;
    Bbox bbox;
    std::string font_name;
    float font_size;
    bool bold;
};

/// A single extracted text line.
struct TextLine {
    std::string text;
    Bbox bbox;
    int word_count;
};

/// A single extracted table. `cell(row, col)` returns the cell text.
struct Table {
    int row_count;
    int col_count;
    bool has_header;
    std::vector<std::string> cells; // row-major, row_count * col_count

    /// Text of the cell at (row, col).
    const std::string& cell(int row, int col) const {
        return cells.at(static_cast<std::size_t>(row) * col_count + col);
    }
};

/// A single embedded font.
struct Font {
    std::string name;
    std::string type;
    std::string encoding;
    bool embedded;
    bool subset;
};

/// A single embedded image.
struct Image {
    int width;
    int height;
    int bits_per_component;
    std::string format;
    std::string colorspace;
    std::vector<std::uint8_t> data;
};

/// A single page annotation.
struct Annotation {
    std::string type;
    std::string subtype;
    std::string content;
    std::string author;
    Bbox rect;
    float border_width;
};

/// A single vector graphics path.
struct Path {
    Bbox bbox;
    float stroke_width;
    bool has_stroke;
    bool has_fill;
    int operation_count;
};

/// A single search hit.
struct SearchResult {
    std::string text;
    int page;
    Bbox bbox;
};

/// A single laid-out page element (PHASE-7 page getters).
struct Element {
    std::string type;
    std::string text;
    Bbox rect;
};

/// A single interactive AcroForm field (PHASE-8 forms).
struct FormField {
    std::string name;
    std::string value;
    std::string type;
    bool readonly;
    bool required;
};

/// One QuadPoints quadrilateral of a highlight/markup annotation (8 floats,
/// four (x,y) corners in page user-space).
struct QuadPoint {
    float x1, y1, x2, y2, x3, y3, x4, y4;
};

/// A rendered page image. Move-only; owns the native FfiRenderedImage handle and
/// frees it on destruction. Width/height/data are read eagerly on construction;
/// save(path) delegates to the still-live native handle.
class RenderedImage {
  public:
    /// Image width in pixels.
    int width() const noexcept { return width_; }
    /// Image height in pixels.
    int height() const noexcept { return height_; }
    /// Encoded image bytes (e.g. PNG/JPEG, per the requested format).
    const std::vector<std::uint8_t>& data() const noexcept { return data_; }

    /// Write the encoded image to `path` via the native handle.
    void save(const std::string& path) const {
        int32_t code = 0;
        if (pdf_save_rendered_image(ptr(), path.c_str(), &code) != 0) {
            throw Error(code, "RenderedImage::save");
        }
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    friend class Document;
    /// Take ownership of an FfiRenderedImage, eagerly read width/height/data
    /// (copying the byte buffer + freeing it with free_bytes), keep the handle
    /// live for save(). Frees the handle on any failure before rethrowing.
    explicit RenderedImage(FfiRenderedImage* h) : handle_(h) {
        try {
            int32_t code = 0;
            width_ = pdf_get_rendered_image_width(ptr(), &code);
            if (width_ < 0) {
                throw Error(code, "RenderedImage::width");
            }
            code = 0;
            height_ = pdf_get_rendered_image_height(ptr(), &code);
            if (height_ < 0) {
                throw Error(code, "RenderedImage::height");
            }
            code = 0;
            int32_t data_len = 0;
            std::uint8_t* p = pdf_get_rendered_image_data(ptr(), &data_len, &code);
            data_ = detail::take_bytes(
                p, static_cast<std::size_t>(data_len < 0 ? 0 : data_len), code,
                "RenderedImage::data");
        } catch (...) {
            handle_.reset();
            throw;
        }
    }
    struct Deleter {
        void operator()(FfiRenderedImage* h) const noexcept {
            if (h)
                pdf_rendered_image_free(h);
        }
    };
    FfiRenderedImage* ptr() const {
        if (!handle_)
            throw Error(0, "RenderedImage is closed");
        return handle_.get();
    }
    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint8_t> data_;
    std::unique_ptr<FfiRenderedImage, Deleter> handle_;
};

/// An opened PDF for extraction/inspection. Move-only; frees on destruction.
class Document {
  public:
    /// Open a PDF from a filesystem path.
    static Document open(const std::string& path) {
        int32_t code = 0;
        PdfDocument* h = pdf_document_open(path.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open");
        }
        return Document(h);
    }

    /// Open a PDF from in-memory bytes.
    static Document open_from_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        PdfDocument* h = pdf_document_open_from_bytes(data.data(), data.size(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open_from_bytes");
        }
        return Document(h);
    }

    /// Open a password-protected PDF.
    static Document open_with_password(const std::string& path,
                                       const std::string& password) {
        int32_t code = 0;
        PdfDocument* h =
            pdf_document_open_with_password(path.c_str(), password.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open_with_password");
        }
        return Document(h);
    }

    /// Number of pages.
    int page_count() const {
        int32_t code = 0;
        int32_t n = pdf_document_get_page_count(ptr(), &code);
        if (n < 0) {
            throw Error(code, "Document::page_count");
        }
        return n;
    }

    /// PDF version.
    Version version() const {
        Version v{0, 0};
        pdf_document_get_version(ptr(), &v.major, &v.minor);
        return v;
    }

    /// True if the document is encrypted.
    bool is_encrypted() const { return pdf_document_is_encrypted(ptr()); }

    /// True if the document carries a logical structure tree (tagged PDF).
    bool has_structure_tree() const { return pdf_document_has_structure_tree(ptr()); }

    /// Extract reading-order text for one page (0-based).
    std::string extract_text(int page_index) const {
        int32_t code = 0;
        return detail::take_string(pdf_document_extract_text(ptr(), page_index, &code),
                                   code, "Document::extract_text");
    }

    /// Plain text for one page.
    std::string to_plain_text(int page_index) const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_plain_text(ptr(), page_index, &code),
                                   code, "Document::to_plain_text");
    }

    /// Markdown for one page.
    std::string to_markdown(int page_index) const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_markdown(ptr(), page_index, &code),
                                   code, "Document::to_markdown");
    }

    /// HTML for one page.
    std::string to_html(int page_index) const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_html(ptr(), page_index, &code), code,
                                   "Document::to_html");
    }

    /// Markdown for the whole document.
    std::string to_markdown_all() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_markdown_all(ptr(), &code), code,
                                   "Document::to_markdown_all");
    }

    /// HTML for the whole document.
    std::string to_html_all() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_html_all(ptr(), &code), code,
                                   "Document::to_html_all");
    }

    /// Plain text for the whole document.
    std::string to_plain_text_all() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_to_plain_text_all(ptr(), &code), code,
                                   "Document::to_plain_text_all");
    }

    /// Authenticate against an encrypted document with `password`.
    /// Returns true on success, false for a wrong password (no error). Throws
    /// Error only when the C ABI signals a real failure via the error code.
    bool authenticate(const std::string& password) const {
        int32_t code = 0;
        bool ok = pdf_document_authenticate(ptr(), password.c_str(), &code);
        if (!ok && code != 0) {
            throw Error(code, "Document::authenticate");
        }
        return ok;
    }

    /// A lightweight, 0-based page view bound to this Document. The returned
    /// Page must not outlive the Document it was obtained from.
    class Page;
    Page page(int index) const;

    // ── PHASE-6 validation (defined out-of-line after the result types) ──────
    /// Validate PDF/A conformance at `level` (0=A1b 1=A1a 2=A2b 3=A2a 4=A2u
    /// 5=A3b 6=A3a 7=A3u).
    class PdfAResults validate_pdf_a(int level) const;
    /// Validate PDF/UA accessibility at `level`.
    class UaResults validate_pdf_ua(int level) const;
    /// Validate PDF/X conformance at `level`.
    class PdfXResults validate_pdf_x(int level) const;
    /// Read the document's DSS (Document Security Store), if present.
    class Dss get_dss() const;

    /// Structured content as a JSON string.
    std::string extract_structured_json(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_structured_to_json(ptr(), page_index, &code), code,
            "Document::extract_structured_json");
    }

    /// Extract individual characters for one page (0-based).
    std::vector<Char> extract_chars(int page_index) const {
        int32_t code = 0;
        FfiCharList* list = pdf_document_extract_chars(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_chars");
        }
        std::vector<Char> out;
        int32_t n = pdf_oxide_char_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Char c;
                code = 0;
                c.character = pdf_oxide_char_get_char(list, i, &code);
                Bbox b{0, 0, 0, 0};
                pdf_oxide_char_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                c.bbox = b;
                c.font_name =
                    detail::take_string(pdf_oxide_char_get_font_name(list, i, &code),
                                        code, "Document::extract_chars");
                c.font_size = pdf_oxide_char_get_font_size(list, i, &code);
                out.push_back(std::move(c));
            }
        } catch (...) {
            pdf_oxide_char_list_free(list);
            throw;
        }
        pdf_oxide_char_list_free(list);
        return out;
    }

    /// Extract words for one page (0-based).
    std::vector<Word> extract_words(int page_index) const {
        int32_t code = 0;
        FfiWordList* list = pdf_document_extract_words(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_words");
        }
        std::vector<Word> out;
        int32_t n = pdf_oxide_word_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Word w;
                code = 0;
                w.text = detail::take_string(pdf_oxide_word_get_text(list, i, &code),
                                             code, "Document::extract_words");
                Bbox b{0, 0, 0, 0};
                pdf_oxide_word_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                w.bbox = b;
                w.font_name =
                    detail::take_string(pdf_oxide_word_get_font_name(list, i, &code),
                                        code, "Document::extract_words");
                w.font_size = pdf_oxide_word_get_font_size(list, i, &code);
                w.bold = pdf_oxide_word_is_bold(list, i, &code);
                out.push_back(std::move(w));
            }
        } catch (...) {
            pdf_oxide_word_list_free(list);
            throw;
        }
        pdf_oxide_word_list_free(list);
        return out;
    }

    /// Extract text lines for one page (0-based).
    std::vector<TextLine> extract_text_lines(int page_index) const {
        int32_t code = 0;
        FfiTextLineList* list =
            pdf_document_extract_text_lines(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_text_lines");
        }
        std::vector<TextLine> out;
        int32_t n = pdf_oxide_line_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                TextLine l;
                code = 0;
                l.text = detail::take_string(pdf_oxide_line_get_text(list, i, &code),
                                             code, "Document::extract_text_lines");
                Bbox b{0, 0, 0, 0};
                pdf_oxide_line_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                l.bbox = b;
                l.word_count = pdf_oxide_line_get_word_count(list, i, &code);
                out.push_back(std::move(l));
            }
        } catch (...) {
            pdf_oxide_line_list_free(list);
            throw;
        }
        pdf_oxide_line_list_free(list);
        return out;
    }

    /// Extract tables for one page (0-based).
    std::vector<Table> extract_tables(int page_index) const {
        int32_t code = 0;
        FfiTableList* list = pdf_document_extract_tables(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_tables");
        }
        std::vector<Table> out;
        int32_t n = pdf_oxide_table_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Table t;
                code = 0;
                t.row_count = pdf_oxide_table_get_row_count(list, i, &code);
                t.col_count = pdf_oxide_table_get_col_count(list, i, &code);
                t.has_header = pdf_oxide_table_has_header(list, i, &code);
                int32_t rows = t.row_count < 0 ? 0 : t.row_count;
                int32_t cols = t.col_count < 0 ? 0 : t.col_count;
                t.cells.reserve(static_cast<std::size_t>(rows) * cols);
                for (int32_t r = 0; r < rows; ++r) {
                    for (int32_t c = 0; c < cols; ++c) {
                        code = 0;
                        t.cells.push_back(detail::take_string(
                            pdf_oxide_table_get_cell_text(list, i, r, c, &code), code,
                            "Document::extract_tables"));
                    }
                }
                out.push_back(std::move(t));
            }
        } catch (...) {
            pdf_oxide_table_list_free(list);
            throw;
        }
        pdf_oxide_table_list_free(list);
        return out;
    }

    /// Extract embedded fonts for one page (0-based).
    std::vector<Font> embedded_fonts(int page_index) const {
        int32_t code = 0;
        FfiFontList* list = pdf_document_get_embedded_fonts(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::embedded_fonts");
        }
        std::vector<Font> out;
        int32_t n = pdf_oxide_font_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Font f;
                code = 0;
                f.name = detail::take_string(pdf_oxide_font_get_name(list, i, &code),
                                             code, "Document::embedded_fonts");
                f.type = detail::take_string(pdf_oxide_font_get_type(list, i, &code),
                                             code, "Document::embedded_fonts");
                f.encoding =
                    detail::take_string(pdf_oxide_font_get_encoding(list, i, &code),
                                        code, "Document::embedded_fonts");
                f.embedded = pdf_oxide_font_is_embedded(list, i, &code) != 0;
                f.subset = pdf_oxide_font_is_subset(list, i, &code) != 0;
                out.push_back(std::move(f));
            }
        } catch (...) {
            pdf_oxide_font_list_free(list);
            throw;
        }
        pdf_oxide_font_list_free(list);
        return out;
    }

    /// Extract embedded images for one page (0-based).
    std::vector<Image> embedded_images(int page_index) const {
        int32_t code = 0;
        FfiImageList* list = pdf_document_get_embedded_images(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::embedded_images");
        }
        std::vector<Image> out;
        int32_t n = pdf_oxide_image_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Image img;
                code = 0;
                img.width = pdf_oxide_image_get_width(list, i, &code);
                img.height = pdf_oxide_image_get_height(list, i, &code);
                img.bits_per_component =
                    pdf_oxide_image_get_bits_per_component(list, i, &code);
                img.format =
                    detail::take_string(pdf_oxide_image_get_format(list, i, &code),
                                        code, "Document::embedded_images");
                img.colorspace =
                    detail::take_string(pdf_oxide_image_get_colorspace(list, i, &code),
                                        code, "Document::embedded_images");
                int32_t data_len = 0;
                std::uint8_t* p = pdf_oxide_image_get_data(list, i, &data_len, &code);
                img.data = detail::take_bytes(
                    p, static_cast<std::size_t>(data_len < 0 ? 0 : data_len), code,
                    "Document::embedded_images");
                out.push_back(std::move(img));
            }
        } catch (...) {
            pdf_oxide_image_list_free(list);
            throw;
        }
        pdf_oxide_image_list_free(list);
        return out;
    }

    /// Extract annotations for one page (0-based).
    std::vector<Annotation> page_annotations(int page_index) const {
        int32_t code = 0;
        FfiAnnotationList* list =
            pdf_document_get_page_annotations(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::page_annotations");
        }
        std::vector<Annotation> out;
        int32_t n = pdf_oxide_annotation_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Annotation a;
                code = 0;
                a.type =
                    detail::take_string(pdf_oxide_annotation_get_type(list, i, &code),
                                        code, "Document::page_annotations");
                a.subtype = detail::take_string(
                    pdf_oxide_annotation_get_subtype(list, i, &code), code,
                    "Document::page_annotations");
                a.content = detail::take_string(
                    pdf_oxide_annotation_get_content(list, i, &code), code,
                    "Document::page_annotations");
                a.author =
                    detail::take_string(pdf_oxide_annotation_get_author(list, i, &code),
                                        code, "Document::page_annotations");
                Bbox b{0, 0, 0, 0};
                pdf_oxide_annotation_get_rect(list, i, &b.x, &b.y, &b.width, &b.height,
                                              &code);
                a.rect = b;
                a.border_width = pdf_oxide_annotation_get_border_width(list, i, &code);
                out.push_back(std::move(a));
            }
        } catch (...) {
            pdf_oxide_annotation_list_free(list);
            throw;
        }
        pdf_oxide_annotation_list_free(list);
        return out;
    }

    /// Extract vector graphics paths for one page (0-based).
    std::vector<Path> extract_paths(int page_index) const {
        int32_t code = 0;
        FfiPathList* list = pdf_document_extract_paths(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_paths");
        }
        std::vector<Path> out;
        int32_t n = pdf_oxide_path_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Path p;
                code = 0;
                Bbox b{0, 0, 0, 0};
                pdf_oxide_path_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                p.bbox = b;
                p.stroke_width = pdf_oxide_path_get_stroke_width(list, i, &code);
                p.has_stroke = pdf_oxide_path_has_stroke(list, i, &code);
                p.has_fill = pdf_oxide_path_has_fill(list, i, &code);
                p.operation_count = pdf_oxide_path_get_operation_count(list, i, &code);
                out.push_back(p);
            }
        } catch (...) {
            pdf_oxide_path_list_free(list);
            throw;
        }
        pdf_oxide_path_list_free(list);
        return out;
    }

    /// Search a single page (0-based) for `term`.
    std::vector<SearchResult> search(int page_index, const std::string& term,
                                     bool case_sensitive) const {
        int32_t code = 0;
        FfiSearchResults* list = pdf_document_search_page(
            ptr(), page_index, term.c_str(), case_sensitive, &code);
        if (list == nullptr) {
            throw Error(code, "Document::search");
        }
        return collect_search_results(list, "Document::search");
    }

    /// Search the whole document for `term`.
    std::vector<SearchResult> search_all(const std::string& term,
                                         bool case_sensitive) const {
        int32_t code = 0;
        FfiSearchResults* list =
            pdf_document_search_all(ptr(), term.c_str(), case_sensitive, &code);
        if (list == nullptr) {
            throw Error(code, "Document::search_all");
        }
        return collect_search_results(list, "Document::search_all");
    }

    /// Render a page (0-based) to an image. `format` is 0=PNG (default), 1=JPEG.
    RenderedImage render_page(int page_index, int format = 0) const {
        int32_t code = 0;
        FfiRenderedImage* h = pdf_render_page(ptr(), page_index, format, &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page");
        }
        return RenderedImage(h);
    }

    /// Render a page (0-based) at a zoom factor. `format` is 0=PNG (default).
    RenderedImage render_page_zoom(int page_index, float zoom, int format = 0) const {
        int32_t code = 0;
        FfiRenderedImage* h =
            pdf_render_page_zoom(ptr(), page_index, zoom, format, &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page_zoom");
        }
        return RenderedImage(h);
    }

    /// Render a page (0-based) as a thumbnail fitting within `size` px.
    /// `format` is 0=PNG (default).
    RenderedImage render_page_thumbnail(int page_index, int size,
                                        int format = 0) const {
        int32_t code = 0;
        FfiRenderedImage* h =
            pdf_render_page_thumbnail(ptr(), page_index, size, format, &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page_thumbnail");
        }
        return RenderedImage(h);
    }

    // ── PHASE-7 render variants (all return a RenderedImage) ────────────────

    /// Render a page (0-based) with the full RenderOptions surface. `dpi` is the
    /// rasterization DPI, `format` 0=PNG 1=JPEG. Background channels are 0..1;
    /// set `transparent_background` true to drop the fill. `render_annotations`
    /// bakes annotations; `jpeg_quality` 0..100 applies to JPEG output.
    RenderedImage render_page_with_options(int page_index, int dpi, int format,
                                           float bg_r, float bg_g, float bg_b,
                                           float bg_a, bool transparent_background,
                                           bool render_annotations,
                                           int jpeg_quality) const {
        int32_t code = 0;
        FfiRenderedImage* h = pdf_render_page_with_options(
            ptr(), page_index, dpi, format, bg_r, bg_g, bg_b, bg_a,
            transparent_background ? 1 : 0, render_annotations ? 1 : 0, jpeg_quality,
            &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page_with_options");
        }
        return RenderedImage(h);
    }

    /// As render_page_with_options, plus OCG layer filtering: each name in
    /// `excluded_layers` is the `/Name` of an Optional Content Group to suppress.
    RenderedImage render_page_with_options_ex(
        int page_index, int dpi, int format, float bg_r, float bg_g, float bg_b,
        float bg_a, bool transparent_background, bool render_annotations,
        int jpeg_quality, const std::vector<std::string>& excluded_layers) const {
        std::vector<const char*> layers;
        layers.reserve(excluded_layers.size());
        for (const auto& s : excluded_layers) {
            layers.push_back(s.c_str());
        }
        int32_t code = 0;
        FfiRenderedImage* h = pdf_render_page_with_options_ex(
            ptr(), page_index, dpi, format, bg_r, bg_g, bg_b, bg_a,
            transparent_background ? 1 : 0, render_annotations ? 1 : 0, jpeg_quality,
            layers.empty() ? nullptr : layers.data(), layers.size(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page_with_options_ex");
        }
        return RenderedImage(h);
    }

    /// Render a rectangular region of a page (crop_* in PDF user-space points,
    /// origin bottom-left). `format` 0=PNG 1=JPEG.
    RenderedImage render_page_region(int page_index, float crop_x, float crop_y,
                                     float crop_width, float crop_height,
                                     int format = 0) const {
        int32_t code = 0;
        FfiRenderedImage* h = pdf_render_page_region(
            ptr(), page_index, crop_x, crop_y, crop_width, crop_height, format, &code);
        if (h == nullptr) {
            throw Error(code, "Document::render_page_region");
        }
        return RenderedImage(h);
    }

    /// Render a page to fit inside `w`×`h` pixels, preserving aspect ratio.
    RenderedImage render_page_fit(int page_index, int w, int h, int format = 0) const {
        int32_t code = 0;
        FfiRenderedImage* img =
            pdf_render_page_fit(ptr(), page_index, w, h, format, &code);
        if (img == nullptr) {
            throw Error(code, "Document::render_page_fit");
        }
        return RenderedImage(img);
    }

    /// Render a page to a raw premultiplied RGBA8888 pixel buffer. The output
    /// width/height are written into `out_width`/`out_height`; the pixel bytes
    /// are available via the returned RenderedImage's data().
    RenderedImage render_page_raw(int page_index, int dpi, int& out_width,
                                  int& out_height) const {
        int32_t code = 0;
        int32_t w = 0, h = 0;
        FfiRenderedImage* img =
            pdf_render_page_raw(ptr(), page_index, dpi, &w, &h, &code);
        if (img == nullptr) {
            throw Error(code, "Document::render_page_raw");
        }
        out_width = w;
        out_height = h;
        return RenderedImage(img);
    }

    /// Estimate the time (ms) to render a page. Returns the raw estimate.
    int estimate_render_time(int page_index) const {
        int32_t code = 0;
        int32_t t = pdf_estimate_render_time(ptr(), page_index, &code);
        if (t < 0 || code != 0) {
            throw Error(code, "Document::estimate_render_time");
        }
        return t;
    }

    // ── PHASE-7 page getters (0-based page) ─────────────────────────────────

    /// Page width in points.
    float page_get_width(int page_index) const {
        int32_t code = 0;
        float w = pdf_page_get_width(ptr(), page_index, &code);
        if (code != 0) {
            throw Error(code, "Document::page_get_width");
        }
        return w;
    }

    /// Page height in points.
    float page_get_height(int page_index) const {
        int32_t code = 0;
        float h = pdf_page_get_height(ptr(), page_index, &code);
        if (code != 0) {
            throw Error(code, "Document::page_get_height");
        }
        return h;
    }

    /// Page rotation in degrees.
    int page_get_rotation(int page_index) const {
        int32_t code = 0;
        int32_t r = pdf_page_get_rotation(ptr(), page_index, &code);
        if (r < 0 || code != 0) {
            throw Error(code, "Document::page_get_rotation");
        }
        return r;
    }

    /// Laid-out elements for one page (0-based).
    std::vector<Element> page_get_elements(int page_index) const {
        int32_t code = 0;
        FfiElementList* list = pdf_page_get_elements(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::page_get_elements");
        }
        std::vector<Element> out;
        int32_t n = pdf_oxide_element_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Element e;
                code = 0;
                e.type = detail::take_string(pdf_oxide_element_get_type(list, i, &code),
                                             code, "Document::page_get_elements");
                code = 0;
                e.text = detail::take_string(pdf_oxide_element_get_text(list, i, &code),
                                             code, "Document::page_get_elements");
                Bbox b{0, 0, 0, 0};
                pdf_oxide_element_get_rect(list, i, &b.x, &b.y, &b.width, &b.height,
                                           &code);
                e.rect = b;
                out.push_back(std::move(e));
            }
        } catch (...) {
            pdf_oxide_elements_free(list);
            throw;
        }
        pdf_oxide_elements_free(list);
        return out;
    }

    // ── PHASE-7 OCR (engine declared after Document) ────────────────────────

    /// True if the page is scanned/hybrid and benefits from OCR.
    bool ocr_page_needs_ocr(int page_index) const {
        int32_t code = 0;
        bool needs = pdf_ocr_page_needs_ocr(ptr(), page_index, &code);
        if (!needs && code != 0) {
            throw Error(code, "Document::ocr_page_needs_ocr");
        }
        return needs;
    }

    /// Extract text for a page via OCR. `engine` may be nullptr (native text
    /// extraction only). Declared inline after OcrEngine to borrow its handle.
    std::string ocr_extract_text(int page_index, const class OcrEngine* engine) const;

    // ── PHASE-8: in-rect extraction ──────────────────────────────────────────

    /// Reading-order text inside the rectangle (x,y,w,h) on `page_index`.
    std::string extract_text_in_rect(int page_index, float x, float y, float w,
                                     float h) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_text_in_rect(ptr(), page_index, x, y, w, h, &code),
            code, "Document::extract_text_in_rect");
    }

    /// Words inside the rectangle (x,y,w,h) on `page_index`.
    std::vector<Word> extract_words_in_rect(int page_index, float x, float y, float w,
                                            float h) const {
        int32_t code = 0;
        FfiWordList* list =
            pdf_document_extract_words_in_rect(ptr(), page_index, x, y, w, h, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_words_in_rect");
        }
        return collect_words(list, "Document::extract_words_in_rect");
    }

    /// Text lines inside the rectangle (x,y,w,h) on `page_index`.
    std::vector<TextLine> extract_lines_in_rect(int page_index, float x, float y,
                                                float w, float h) const {
        int32_t code = 0;
        FfiTextLineList* list =
            pdf_document_extract_lines_in_rect(ptr(), page_index, x, y, w, h, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_lines_in_rect");
        }
        return collect_lines(list, "Document::extract_lines_in_rect");
    }

    /// Tables inside the rectangle (x,y,w,h) on `page_index`.
    std::vector<Table> extract_tables_in_rect(int page_index, float x, float y, float w,
                                              float h) const {
        int32_t code = 0;
        FfiTableList* list =
            pdf_document_extract_tables_in_rect(ptr(), page_index, x, y, w, h, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_tables_in_rect");
        }
        return collect_tables(list, "Document::extract_tables_in_rect");
    }

    /// Images inside the rectangle (x,y,w,h) on `page_index`.
    std::vector<Image> extract_images_in_rect(int page_index, float x, float y, float w,
                                              float h) const {
        int32_t code = 0;
        FfiImageList* list =
            pdf_document_extract_images_in_rect(ptr(), page_index, x, y, w, h, &code);
        if (list == nullptr) {
            throw Error(code, "Document::extract_images_in_rect");
        }
        return collect_images(list, "Document::extract_images_in_rect");
    }

    // ── PHASE-8: auto / classification extraction ────────────────────────────

    /// Auto-mode text (native + image OCR superset) for one page (0-based).
    std::string extract_text_auto(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_text_auto(ptr(), page_index, &code), code,
            "Document::extract_text_auto");
    }

    /// Reading-order text for the whole document.
    std::string extract_all_text() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_extract_all_text(ptr(), &code), code,
                                   "Document::extract_all_text");
    }

    /// Auto-mode extraction for one page with JSON options (empty = defaults).
    std::string extract_page_auto(int page_index,
                                  const std::string& options_json = "") const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_page_auto(
                ptr(), page_index,
                options_json.empty() ? nullptr : options_json.c_str(), &code),
            code, "Document::extract_page_auto");
    }

    /// Classify a single page (JSON description of detected content type).
    std::string classify_page(int page_index) const {
        int32_t code = 0;
        return detail::take_string(pdf_document_classify_page(ptr(), page_index, &code),
                                   code, "Document::classify_page");
    }

    /// Classify the whole document (JSON description).
    std::string classify_document() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_classify_document(ptr(), &code), code,
                                   "Document::classify_document");
    }

    // ── PHASE-8: header / footer / artifact removal ──────────────────────────

    /// Erase the detected header region on `page_index`. Returns the C status.
    int erase_header(int page_index) const {
        int32_t code = 0;
        int32_t r = pdf_document_erase_header(ptr(), page_index, &code);
        if (code != 0) {
            throw Error(code, "Document::erase_header");
        }
        return r;
    }
    /// Erase the detected footer region on `page_index`. Returns the C status.
    int erase_footer(int page_index) const {
        int32_t code = 0;
        int32_t r = pdf_document_erase_footer(ptr(), page_index, &code);
        if (code != 0) {
            throw Error(code, "Document::erase_footer");
        }
        return r;
    }
    /// Erase detected artifacts on `page_index`. Returns the C status.
    int erase_artifacts(int page_index) const {
        int32_t code = 0;
        int32_t r = pdf_document_erase_artifacts(ptr(), page_index, &code);
        if (code != 0) {
            throw Error(code, "Document::erase_artifacts");
        }
        return r;
    }
    /// Remove repeated headers across the document at `threshold` frequency.
    int remove_headers(float threshold) const {
        int32_t code = 0;
        int32_t r = pdf_document_remove_headers(ptr(), threshold, &code);
        if (code != 0) {
            throw Error(code, "Document::remove_headers");
        }
        return r;
    }
    /// Remove repeated footers across the document at `threshold` frequency.
    int remove_footers(float threshold) const {
        int32_t code = 0;
        int32_t r = pdf_document_remove_footers(ptr(), threshold, &code);
        if (code != 0) {
            throw Error(code, "Document::remove_footers");
        }
        return r;
    }
    /// Remove repeated artifacts across the document at `threshold` frequency.
    int remove_artifacts(float threshold) const {
        int32_t code = 0;
        int32_t r = pdf_document_remove_artifacts(ptr(), threshold, &code);
        if (code != 0) {
            throw Error(code, "Document::remove_artifacts");
        }
        return r;
    }

    // ── PHASE-8: AcroForm fields ─────────────────────────────────────────────

    /// All interactive form fields in the document (empty list when none).
    std::vector<FormField> get_form_fields() const {
        int32_t code = 0;
        FfiFormFieldList* list = pdf_document_get_form_fields(ptr(), &code);
        if (list == nullptr) {
            throw Error(code, "Document::get_form_fields");
        }
        std::vector<FormField> out;
        int32_t n = pdf_oxide_form_field_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                FormField fld;
                code = 0;
                fld.name =
                    detail::take_string(pdf_oxide_form_field_get_name(list, i, &code),
                                        code, "Document::get_form_fields");
                fld.value =
                    detail::take_string(pdf_oxide_form_field_get_value(list, i, &code),
                                        code, "Document::get_form_fields");
                fld.type =
                    detail::take_string(pdf_oxide_form_field_get_type(list, i, &code),
                                        code, "Document::get_form_fields");
                fld.readonly = pdf_oxide_form_field_is_readonly(list, i, &code);
                fld.required = pdf_oxide_form_field_is_required(list, i, &code);
                out.push_back(std::move(fld));
            }
        } catch (...) {
            pdf_oxide_form_field_list_free(list);
            throw;
        }
        pdf_oxide_form_field_list_free(list);
        return out;
    }

    /// Export the current AcroForm field values. `format_type` selects the wire
    /// format (e.g. 0=FDF, 1=XFDF, 2=JSON per the core).
    std::vector<std::uint8_t> export_form_data_to_bytes(int format_type) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p =
            pdf_document_export_form_data_to_bytes(ptr(), format_type, &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Document::export_form_data_to_bytes");
    }

    /// Import AcroForm field values from the file at `data_path`. Returns status.
    int import_form_data(const std::string& data_path) const {
        int32_t code = 0;
        int32_t r = pdf_document_import_form_data(ptr(), data_path.c_str(), &code);
        if (code != 0) {
            throw Error(code, "Document::import_form_data");
        }
        return r;
    }

    /// Import AcroForm field values from the file at `filename` (alternate entry
    /// point). Returns true on success.
    bool form_import_from_file(const std::string& filename) const {
        int32_t code = 0;
        bool ok = pdf_form_import_from_file(ptr(), filename.c_str(), &code);
        if (!ok && code != 0) {
            throw Error(code, "Document::form_import_from_file");
        }
        return ok;
    }

    // ── PHASE-8: document structure / metadata ───────────────────────────────

    /// The document outline (bookmarks) as JSON.
    std::string get_outline() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_get_outline(ptr(), &code), code,
                                   "Document::get_outline");
    }
    /// Page labels (e.g. roman/decimal numbering ranges) as JSON.
    std::string get_page_labels() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_get_page_labels(ptr(), &code), code,
                                   "Document::get_page_labels");
    }
    /// The raw XMP metadata packet (XML string).
    std::string get_xmp_metadata() const {
        int32_t code = 0;
        return detail::take_string(pdf_document_get_xmp_metadata(ptr(), &code), code,
                                   "Document::get_xmp_metadata");
    }
    /// The original source bytes the document was opened from.
    std::vector<std::uint8_t> get_source_bytes() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_get_source_bytes(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Document::get_source_bytes");
    }
    /// True if the document carries an XFA form.
    bool has_xfa() const { return pdf_document_has_xfa(ptr()); }

    /// Plan a split of the document by top-level bookmarks. `options_json` may be
    /// empty for defaults. Returns the plan as JSON.
    std::string plan_split_by_bookmarks(const std::string& options_json = "") const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_plan_split_by_bookmarks(
                ptr(), options_json.empty() ? nullptr : options_json.c_str(), &code),
            code, "Document::plan_split_by_bookmarks");
    }

    // ── PHASE-8: signatures (document-level) ─────────────────────────────────

    /// Sign the document in place with `cert`. Returns the C status. Defined
    /// out-of-line below (needs the Certificate class).
    int sign(const class Certificate& cert, const std::string& reason = "",
             const std::string& location = "") const;

    /// Number of signatures present in the document.
    int get_signature_count() const {
        int32_t code = 0;
        int32_t n = pdf_document_get_signature_count(ptr(), &code);
        if (n < 0 || code != 0) {
            throw Error(code, "Document::get_signature_count");
        }
        return n;
    }

    /// The `index`-th signature as a SignatureInfo (takes ownership of the
    /// returned handle). Defined out-of-line below (needs SignatureInfo).
    class SignatureInfo get_signature(int index) const;

    /// Verify every signature in the document. Returns the C status code.
    int verify_all_signatures() const {
        int32_t code = 0;
        int32_t r = pdf_document_verify_all_signatures(ptr(), &code);
        if (code != 0) {
            throw Error(code, "Document::verify_all_signatures");
        }
        return r;
    }

    /// True (1) if the document carries a signature timestamp.
    int has_timestamp() const {
        int32_t code = 0;
        int32_t r = pdf_document_has_timestamp(ptr(), &code);
        if (code != 0) {
            throw Error(code, "Document::has_timestamp");
        }
        return r;
    }

    // ── PHASE-8: annotation extras / to-JSON accessors ───────────────────────
    //
    // These fetch the page's FfiAnnotationList, query the new per-annotation
    // accessors, then free it (mirroring page_annotations). Each returns the
    // value for annotation `ann_index` on `page_index`.

    /// 0xAARRGGBB color of annotation `ann_index` on `page_index`.
    std::uint32_t annotation_get_color(int page_index, int ann_index) const {
        return with_annotations<std::uint32_t>(
            page_index, "Document::annotation_get_color",
            [&](FfiAnnotationList* l, int32_t* c) {
                return pdf_oxide_annotation_get_color(l, ann_index, c);
            });
    }
    /// Creation date (epoch seconds) of annotation `ann_index`.
    std::int64_t annotation_get_creation_date(int page_index, int ann_index) const {
        return with_annotations<std::int64_t>(
            page_index, "Document::annotation_get_creation_date",
            [&](FfiAnnotationList* l, int32_t* c) {
                return pdf_oxide_annotation_get_creation_date(l, ann_index, c);
            });
    }
    /// Modification date (epoch seconds) of annotation `ann_index`.
    std::int64_t annotation_get_modification_date(int page_index, int ann_index) const {
        return with_annotations<std::int64_t>(
            page_index, "Document::annotation_get_modification_date",
            [&](FfiAnnotationList* l, int32_t* c) {
                return pdf_oxide_annotation_get_modification_date(l, ann_index, c);
            });
    }
    /// True if annotation `ann_index` is hidden.
    bool annotation_is_hidden(int page_index, int ann_index) const {
        return with_annotations<bool>(page_index, "Document::annotation_is_hidden",
                                      [&](FfiAnnotationList* l, int32_t* c) {
                                          return pdf_oxide_annotation_is_hidden(
                                              l, ann_index, c);
                                      });
    }
    /// True if annotation `ann_index` is marked deleted.
    bool annotation_is_marked_deleted(int page_index, int ann_index) const {
        return with_annotations<bool>(
            page_index, "Document::annotation_is_marked_deleted",
            [&](FfiAnnotationList* l, int32_t* c) {
                return pdf_oxide_annotation_is_marked_deleted(l, ann_index, c);
            });
    }
    /// True if annotation `ann_index` is printable.
    bool annotation_is_printable(int page_index, int ann_index) const {
        return with_annotations<bool>(page_index, "Document::annotation_is_printable",
                                      [&](FfiAnnotationList* l, int32_t* c) {
                                          return pdf_oxide_annotation_is_printable(
                                              l, ann_index, c);
                                      });
    }
    /// True if annotation `ann_index` is read-only.
    bool annotation_is_read_only(int page_index, int ann_index) const {
        return with_annotations<bool>(page_index, "Document::annotation_is_read_only",
                                      [&](FfiAnnotationList* l, int32_t* c) {
                                          return pdf_oxide_annotation_is_read_only(
                                              l, ann_index, c);
                                      });
    }
    /// QuadPoints count of highlight/markup annotation `ann_index`.
    int highlight_annotation_get_quad_points_count(int page_index,
                                                   int ann_index) const {
        return with_annotations<int>(
            page_index, "Document::highlight_annotation_get_quad_points_count",
            [&](FfiAnnotationList* l, int32_t* c) {
                return pdf_oxide_highlight_annotation_get_quad_points_count(
                    l, ann_index, c);
            });
    }
    /// The `quad_index`-th QuadPoint of highlight annotation `ann_index`.
    QuadPoint highlight_annotation_get_quad_point(int page_index, int ann_index,
                                                  int quad_index) const {
        return with_annotations<QuadPoint>(
            page_index, "Document::highlight_annotation_get_quad_point",
            [&](FfiAnnotationList* l, int32_t* c) {
                QuadPoint q{0, 0, 0, 0, 0, 0, 0, 0};
                pdf_oxide_highlight_annotation_get_quad_point(
                    l, ann_index, quad_index, &q.x1, &q.y1, &q.x2, &q.y2, &q.x3, &q.y3,
                    &q.x4, &q.y4, c);
                return q;
            });
    }
    /// The destination URI of link annotation `ann_index`.
    std::string link_annotation_get_uri(int page_index, int ann_index) const {
        int32_t code = 0;
        FfiAnnotationList* list =
            pdf_document_get_page_annotations(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::link_annotation_get_uri");
        }
        try {
            std::string out = detail::take_string(
                pdf_oxide_link_annotation_get_uri(list, ann_index, &code), code,
                "Document::link_annotation_get_uri");
            pdf_oxide_annotation_list_free(list);
            return out;
        } catch (...) {
            pdf_oxide_annotation_list_free(list);
            throw;
        }
    }
    /// The icon name of text/note annotation `ann_index`.
    std::string text_annotation_get_icon_name(int page_index, int ann_index) const {
        int32_t code = 0;
        FfiAnnotationList* list =
            pdf_document_get_page_annotations(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::text_annotation_get_icon_name");
        }
        try {
            std::string out = detail::take_string(
                pdf_oxide_text_annotation_get_icon_name(list, ann_index, &code), code,
                "Document::text_annotation_get_icon_name");
            pdf_oxide_annotation_list_free(list);
            return out;
        } catch (...) {
            pdf_oxide_annotation_list_free(list);
            throw;
        }
    }
    /// All annotations on `page_index` serialized as JSON.
    std::string annotations_to_json(int page_index) const {
        int32_t code = 0;
        FfiAnnotationList* list =
            pdf_document_get_page_annotations(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::annotations_to_json");
        }
        try {
            std::string out =
                detail::take_string(pdf_oxide_annotations_to_json(list, &code), code,
                                    "Document::annotations_to_json");
            pdf_oxide_annotation_list_free(list);
            return out;
        } catch (...) {
            pdf_oxide_annotation_list_free(list);
            throw;
        }
    }
    /// Laid-out page elements on `page_index` serialized as JSON.
    std::string elements_to_json(int page_index) const {
        int32_t code = 0;
        FfiElementList* list = pdf_page_get_elements(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::elements_to_json");
        }
        try {
            std::string out =
                detail::take_string(pdf_oxide_elements_to_json(list, &code), code,
                                    "Document::elements_to_json");
            pdf_oxide_elements_free(list);
            return out;
        } catch (...) {
            pdf_oxide_elements_free(list);
            throw;
        }
    }
    /// Embedded fonts on `page_index` serialized as JSON.
    std::string fonts_to_json(int page_index) const {
        int32_t code = 0;
        FfiFontList* list = pdf_document_get_embedded_fonts(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::fonts_to_json");
        }
        try {
            std::string out = detail::take_string(pdf_oxide_fonts_to_json(list, &code),
                                                  code, "Document::fonts_to_json");
            pdf_oxide_font_list_free(list);
            return out;
        } catch (...) {
            pdf_oxide_font_list_free(list);
            throw;
        }
    }
    /// The font size of embedded font `font_index` on `page_index`.
    float font_get_size(int page_index, int font_index) const {
        int32_t code = 0;
        FfiFontList* list = pdf_document_get_embedded_fonts(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, "Document::font_get_size");
        }
        float sz = pdf_oxide_font_get_size(list, font_index, &code);
        int32_t saved = code;
        pdf_oxide_font_list_free(list);
        if (saved != 0) {
            throw Error(saved, "Document::font_get_size");
        }
        return sz;
    }
    /// Search results for `term` on `page_index` serialized as JSON.
    std::string search_results_to_json(int page_index, const std::string& term,
                                       bool case_sensitive) const {
        int32_t code = 0;
        FfiSearchResults* list = pdf_document_search_page(
            ptr(), page_index, term.c_str(), case_sensitive, &code);
        if (list == nullptr) {
            throw Error(code, "Document::search_results_to_json");
        }
        try {
            std::string out =
                detail::take_string(pdf_oxide_search_results_to_json(list, &code), code,
                                    "Document::search_results_to_json");
            pdf_oxide_search_result_free(list);
            return out;
        } catch (...) {
            pdf_oxide_search_result_free(list);
            throw;
        }
    }

    // ── PHASE-8: office export / PDF-A conversion ────────────────────────────

    /// Convert the document to a DOCX (Office Open XML) byte buffer.
    std::vector<std::uint8_t> to_docx() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_to_docx(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Document::to_docx");
    }
    /// Convert the document to a PPTX byte buffer.
    std::vector<std::uint8_t> to_pptx() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_to_pptx(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Document::to_pptx");
    }
    /// Convert the document to an XLSX byte buffer.
    std::vector<std::uint8_t> to_xlsx() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_to_xlsx(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Document::to_xlsx");
    }

    /// Convert the document to PDF/A in place at `level`. Returns true on
    /// success.
    bool convert_to_pdf_a(int level) const {
        int32_t code = 0;
        bool ok = pdf_convert_to_pdf_a(ptr(), level, &code);
        if (!ok && code != 0) {
            throw Error(code, "Document::convert_to_pdf_a");
        }
        return ok;
    }

    // ── PHASE-8: office import (static constructors) ─────────────────────────

    /// Open a Document from DOCX bytes.
    static Document open_from_docx_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        PdfDocument* h =
            pdf_document_open_from_docx_bytes(data.data(), data.size(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open_from_docx_bytes");
        }
        return Document(h);
    }
    /// Open a Document from PPTX bytes.
    static Document open_from_pptx_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        PdfDocument* h =
            pdf_document_open_from_pptx_bytes(data.data(), data.size(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open_from_pptx_bytes");
        }
        return Document(h);
    }
    /// Open a Document from XLSX bytes.
    static Document open_from_xlsx_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        PdfDocument* h =
            pdf_document_open_from_xlsx_bytes(data.data(), data.size(), &code);
        if (h == nullptr) {
            throw Error(code, "Document::open_from_xlsx_bytes");
        }
        return Document(h);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit;
    /// this is the explicit close for API symmetry with the other bindings.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(PdfDocument* h) const noexcept {
            if (h)
                pdf_document_free(h);
        }
    };
    explicit Document(PdfDocument* h) : handle_(h) {}
    /// Marshal an FfiSearchResults handle into SearchResult values, then free it
    /// with pdf_oxide_search_result_free (NB: not a *_list_free).
    static std::vector<SearchResult> collect_search_results(FfiSearchResults* list,
                                                            const char* op) {
        std::vector<SearchResult> out;
        int32_t code = 0;
        int32_t n = pdf_oxide_search_result_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                SearchResult r;
                code = 0;
                r.text = detail::take_string(
                    pdf_oxide_search_result_get_text(list, i, &code), code, op);
                r.page = pdf_oxide_search_result_get_page(list, i, &code);
                Bbox b{0, 0, 0, 0};
                pdf_oxide_search_result_get_bbox(list, i, &b.x, &b.y, &b.width,
                                                 &b.height, &code);
                r.bbox = b;
                out.push_back(std::move(r));
            }
        } catch (...) {
            pdf_oxide_search_result_free(list);
            throw;
        }
        pdf_oxide_search_result_free(list);
        return out;
    }

    // ── PHASE-8 shared list marshallers (consume + free the passed handle) ────
    // Mirror the inline marshalling of extract_words/lines/tables/images so the
    // in-rect variants reuse the exact same Word/TextLine/Table/Image shapes.
    static std::vector<Word> collect_words(FfiWordList* list, const char* op) {
        std::vector<Word> out;
        int32_t code = 0;
        int32_t n = pdf_oxide_word_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Word w;
                code = 0;
                w.text = detail::take_string(pdf_oxide_word_get_text(list, i, &code),
                                             code, op);
                Bbox b{0, 0, 0, 0};
                pdf_oxide_word_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                w.bbox = b;
                w.font_name = detail::take_string(
                    pdf_oxide_word_get_font_name(list, i, &code), code, op);
                w.font_size = pdf_oxide_word_get_font_size(list, i, &code);
                w.bold = pdf_oxide_word_is_bold(list, i, &code);
                out.push_back(std::move(w));
            }
        } catch (...) {
            pdf_oxide_word_list_free(list);
            throw;
        }
        pdf_oxide_word_list_free(list);
        return out;
    }
    static std::vector<TextLine> collect_lines(FfiTextLineList* list, const char* op) {
        std::vector<TextLine> out;
        int32_t code = 0;
        int32_t n = pdf_oxide_line_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                TextLine l;
                code = 0;
                l.text = detail::take_string(pdf_oxide_line_get_text(list, i, &code),
                                             code, op);
                Bbox b{0, 0, 0, 0};
                pdf_oxide_line_get_bbox(list, i, &b.x, &b.y, &b.width, &b.height,
                                        &code);
                l.bbox = b;
                l.word_count = pdf_oxide_line_get_word_count(list, i, &code);
                out.push_back(std::move(l));
            }
        } catch (...) {
            pdf_oxide_line_list_free(list);
            throw;
        }
        pdf_oxide_line_list_free(list);
        return out;
    }
    static std::vector<Table> collect_tables(FfiTableList* list, const char* op) {
        std::vector<Table> out;
        int32_t code = 0;
        int32_t n = pdf_oxide_table_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Table t;
                code = 0;
                t.row_count = pdf_oxide_table_get_row_count(list, i, &code);
                t.col_count = pdf_oxide_table_get_col_count(list, i, &code);
                t.has_header = pdf_oxide_table_has_header(list, i, &code);
                t.cells.reserve(
                    static_cast<std::size_t>(t.row_count < 0 ? 0 : t.row_count) *
                    static_cast<std::size_t>(t.col_count < 0 ? 0 : t.col_count));
                for (int r = 0; r < t.row_count; ++r) {
                    for (int c = 0; c < t.col_count; ++c) {
                        t.cells.push_back(detail::take_string(
                            pdf_oxide_table_get_cell_text(list, i, r, c, &code), code,
                            op));
                    }
                }
                out.push_back(std::move(t));
            }
        } catch (...) {
            pdf_oxide_table_list_free(list);
            throw;
        }
        pdf_oxide_table_list_free(list);
        return out;
    }
    static std::vector<Image> collect_images(FfiImageList* list, const char* op) {
        std::vector<Image> out;
        int32_t code = 0;
        int32_t n = pdf_oxide_image_count(list);
        out.reserve(n < 0 ? 0 : static_cast<std::size_t>(n));
        try {
            for (int32_t i = 0; i < n; ++i) {
                Image img;
                code = 0;
                img.width = pdf_oxide_image_get_width(list, i, &code);
                img.height = pdf_oxide_image_get_height(list, i, &code);
                img.bits_per_component =
                    pdf_oxide_image_get_bits_per_component(list, i, &code);
                img.format = detail::take_string(
                    pdf_oxide_image_get_format(list, i, &code), code, op);
                img.colorspace = detail::take_string(
                    pdf_oxide_image_get_colorspace(list, i, &code), code, op);
                int32_t data_len = 0;
                std::uint8_t* p = pdf_oxide_image_get_data(list, i, &data_len, &code);
                img.data = detail::take_bytes(
                    p, static_cast<std::size_t>(data_len < 0 ? 0 : data_len), code, op);
                out.push_back(std::move(img));
            }
        } catch (...) {
            pdf_oxide_image_list_free(list);
            throw;
        }
        pdf_oxide_image_list_free(list);
        return out;
    }

    /// Fetch the page's FfiAnnotationList, run `fn(list, &code)`, free the list,
    /// and rethrow as Error on a non-zero code. Used by the annotation-extras
    /// accessors so their list lifetime/cleanup is identical everywhere.
    template <typename T, typename Fn>
    T with_annotations(int page_index, const char* op, Fn&& fn) const {
        int32_t code = 0;
        FfiAnnotationList* list =
            pdf_document_get_page_annotations(ptr(), page_index, &code);
        if (list == nullptr) {
            throw Error(code, op);
        }
        code = 0;
        T value;
        try {
            value = fn(list, &code);
        } catch (...) {
            pdf_oxide_annotation_list_free(list);
            throw;
        }
        int32_t saved = code;
        pdf_oxide_annotation_list_free(list);
        if (saved != 0) {
            throw Error(saved, op);
        }
        return value;
    }

    PdfDocument* ptr() const {
        if (!handle_)
            throw Error(0, "Document is closed");
        return handle_.get();
    }
    std::unique_ptr<PdfDocument, Deleter> handle_;
};

/// A 0-based page view bound to a Document. Holds a non-owning reference to the
/// Document, which MUST outlive the Page. Each accessor delegates to the
/// corresponding per-page Document method with the stored index.
class Document::Page {
  public:
    /// Reading-order text for this page.
    std::string text() const { return doc_->extract_text(index_); }
    /// Markdown for this page.
    std::string markdown() const { return doc_->to_markdown(index_); }
    /// HTML for this page.
    std::string html() const { return doc_->to_html(index_); }
    /// Plain text for this page.
    std::string plain_text() const { return doc_->to_plain_text(index_); }

    /// 0-based page index.
    int index() const noexcept { return index_; }

  private:
    friend class Document;
    Page(const Document* doc, int index) : doc_(doc), index_(index) {}
    const Document* doc_;
    int index_;
};

inline Document::Page Document::page(int index) const {
    return Document::Page(this, index);
}

/// A PDF produced by a builder (from markdown/html/text). Move-only.
class Pdf {
  public:
    /// Build a PDF from Markdown.
    static Pdf from_markdown(const std::string& markdown) {
        int32_t code = 0;
        ::Pdf* h = pdf_from_markdown(markdown.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_markdown");
        }
        return Pdf(h);
    }

    /// Build a PDF from HTML.
    static Pdf from_html(const std::string& html) {
        int32_t code = 0;
        ::Pdf* h = pdf_from_html(html.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_html");
        }
        return Pdf(h);
    }

    /// Build a PDF from plain text.
    static Pdf from_text(const std::string& text) {
        int32_t code = 0;
        ::Pdf* h = pdf_from_text(text.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_text");
        }
        return Pdf(h);
    }

    // ── PHASE-7 constructors ─────────────────────────────────────────────

    /// Build a single-page PDF wrapping the image at `path`.
    static Pdf from_image(const std::string& path) {
        int32_t code = 0;
        ::Pdf* h = pdf_from_image(path.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_image");
        }
        return Pdf(h);
    }

    /// Build a single-page PDF wrapping in-memory image `data`.
    static Pdf from_image_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        ::Pdf* h =
            pdf_from_image_bytes(data.data(), static_cast<int32_t>(data.size()), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_image_bytes");
        }
        return Pdf(h);
    }

    /// Build a PDF from HTML + CSS with one optional embedded font (empty
    /// `font_bytes` for none).
    static Pdf from_html_css(const std::string& html, const std::string& css,
                             const std::vector<std::uint8_t>& font_bytes = {}) {
        int32_t code = 0;
        ::Pdf* h = pdf_from_html_css(
            html.c_str(), css.c_str(), font_bytes.empty() ? nullptr : font_bytes.data(),
            static_cast<std::uintptr_t>(font_bytes.size()), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_html_css");
        }
        return Pdf(h);
    }

    /// Build a PDF from HTML + CSS with a multi-font cascade. `families` and
    /// `fonts` are parallel arrays of the same length.
    static Pdf
    from_html_css_with_fonts(const std::string& html, const std::string& css,
                             const std::vector<std::string>& families,
                             const std::vector<std::vector<std::uint8_t>>& fonts) {
        std::vector<const char*> fam_ptrs;
        std::vector<const std::uint8_t*> font_ptrs;
        std::vector<std::uintptr_t> font_lens;
        fam_ptrs.reserve(families.size());
        font_ptrs.reserve(fonts.size());
        font_lens.reserve(fonts.size());
        for (const auto& f : families) {
            fam_ptrs.push_back(f.c_str());
        }
        for (const auto& b : fonts) {
            font_ptrs.push_back(b.data());
            font_lens.push_back(static_cast<std::uintptr_t>(b.size()));
        }
        int32_t code = 0;
        ::Pdf* h = pdf_from_html_css_with_fonts(
            html.c_str(), css.c_str(), fam_ptrs.empty() ? nullptr : fam_ptrs.data(),
            font_ptrs.empty() ? nullptr : font_ptrs.data(),
            font_lens.empty() ? nullptr : font_lens.data(),
            static_cast<std::uintptr_t>(fam_ptrs.size()), &code);
        if (h == nullptr) {
            throw Error(code, "Pdf::from_html_css_with_fonts");
        }
        return Pdf(h);
    }

    /// Write the PDF to a path.
    void save(const std::string& path) const {
        int32_t code = 0;
        if (pdf_save(ptr(), path.c_str(), &code) != 0) {
            throw Error(code, "Pdf::save");
        }
    }

    /// Serialize the PDF to bytes.
    std::vector<std::uint8_t> to_bytes() const {
        int32_t code = 0;
        int32_t len = 0;
        std::uint8_t* p = pdf_save_to_bytes(ptr(), &len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(len < 0 ? 0 : len), code,
                                  "Pdf::to_bytes");
    }

    /// Number of pages in the built PDF (legacy Pdf-handle accessor).
    int page_count() const {
        int32_t code = 0;
        int32_t n = pdf_get_page_count(ptr(), &code);
        if (n < 0 || code != 0) {
            throw Error(code, "Pdf::page_count");
        }
        return n;
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(::Pdf* h) const noexcept {
            if (h)
                pdf_free(h);
        }
    };
    explicit Pdf(::Pdf* h) : handle_(h) {}
    ::Pdf* ptr() const {
        if (!handle_)
            throw Error(0, "Pdf is closed");
        return handle_.get();
    }
    std::unique_ptr<::Pdf, Deleter> handle_;
};

/// A single rectangle to erase, in page user-space coordinates.
struct EraseRect {
    double x;
    double y;
    double width;
    double height;
};

/// An open PDF for in-place editing (rotate/crop/redact/flatten/merge/save).
/// Move-only; owns the native DocumentEditor handle and frees it on destruction.
/// int32 status returns are treated as 0 = success; a non-zero status (or a set
/// error_code) raises Error. The is_* query functions are exposed as bool
/// (1 = true).
class DocumentEditor {
  public:
    /// Open a PDF for editing from a filesystem path.
    static DocumentEditor open(const std::string& path) {
        int32_t code = 0;
        ::DocumentEditor* h = document_editor_open(path.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "DocumentEditor::open");
        }
        return DocumentEditor(h);
    }

    /// Open a PDF for editing from in-memory bytes.
    static DocumentEditor open_from_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        ::DocumentEditor* h =
            document_editor_open_from_bytes(data.data(), data.size(), &code);
        if (h == nullptr) {
            throw Error(code, "DocumentEditor::open_from_bytes");
        }
        return DocumentEditor(h);
    }

    /// Number of pages.
    int page_count() const {
        int32_t code = 0;
        int32_t n = document_editor_get_page_count(ptr(), &code);
        if (n < 0) {
            throw Error(code, "DocumentEditor::page_count");
        }
        return n;
    }

    /// PDF version.
    Version version() const {
        Version v{0, 0};
        document_editor_get_version(ptr(), &v.major, &v.minor);
        return v;
    }

    /// True if the editor has pending modifications.
    bool is_modified() const { return document_editor_is_modified(ptr()); }

    /// The source path the editor was opened from (empty if from bytes).
    std::string get_source_path() const {
        int32_t code = 0;
        return detail::take_string(document_editor_get_source_path(ptr(), &code), code,
                                   "DocumentEditor::get_source_path");
    }

    /// Producer (`/Info.Producer`).
    std::string get_producer() const {
        int32_t code = 0;
        return detail::take_string(document_editor_get_producer(ptr(), &code), code,
                                   "DocumentEditor::get_producer");
    }

    /// Set the producer (`/Info.Producer`).
    void set_producer(const std::string& value) {
        int32_t code = 0;
        if (document_editor_set_producer(ptr(), value.c_str(), &code) != 0) {
            throw Error(code, "DocumentEditor::set_producer");
        }
    }

    /// Creation date (`/Info.CreationDate`, raw PDF date string).
    std::string get_creation_date() const {
        int32_t code = 0;
        return detail::take_string(document_editor_get_creation_date(ptr(), &code),
                                   code, "DocumentEditor::get_creation_date");
    }

    /// Set the creation date (raw PDF date string, e.g. `D:20260421120000Z`).
    void set_creation_date(const std::string& date_str) {
        int32_t code = 0;
        if (document_editor_set_creation_date(ptr(), date_str.c_str(), &code) != 0) {
            throw Error(code, "DocumentEditor::set_creation_date");
        }
    }

    /// Delete the page at `page_index` (0-based).
    void delete_page(int page_index) {
        int32_t code = 0;
        if (document_editor_delete_page(ptr(), page_index, &code) != 0) {
            throw Error(code, "DocumentEditor::delete_page");
        }
    }

    /// Move the page at `from` (0-based) to `to`.
    void move_page(int from, int to) {
        int32_t code = 0;
        if (document_editor_move_page(ptr(), from, to, &code) != 0) {
            throw Error(code, "DocumentEditor::move_page");
        }
    }

    /// Rotate one page by `degrees` (additive, not absolute).
    void rotate_page_by(int page_index, int degrees) {
        int32_t code = 0;
        if (document_editor_rotate_page_by(
                ptr(), static_cast<std::uintptr_t>(page_index), degrees, &code) != 0) {
            throw Error(code, "DocumentEditor::rotate_page_by");
        }
    }

    /// Rotate all pages by `degrees` (additive).
    void rotate_all_pages(int degrees) {
        int32_t code = 0;
        if (document_editor_rotate_all_pages(ptr(), degrees, &code) != 0) {
            throw Error(code, "DocumentEditor::rotate_all_pages");
        }
    }

    /// Set the absolute rotation (degrees) of one page.
    void set_page_rotation(int page_index, int degrees) {
        int32_t code = 0;
        if (document_editor_set_page_rotation(ptr(), page_index, degrees, &code) != 0) {
            throw Error(code, "DocumentEditor::set_page_rotation");
        }
    }

    /// Get the absolute rotation (degrees) of one page.
    int get_page_rotation(int page_index) const {
        int32_t code = 0;
        int32_t deg = document_editor_get_page_rotation(ptr(), page_index, &code);
        if (deg < 0 || code != 0) {
            throw Error(code, "DocumentEditor::get_page_rotation");
        }
        return deg;
    }

    /// Crop margins (in points) off every page.
    void crop_margins(float left, float right, float top, float bottom) {
        int32_t code = 0;
        if (document_editor_crop_margins(ptr(), left, right, top, bottom, &code) != 0) {
            throw Error(code, "DocumentEditor::crop_margins");
        }
    }

    /// Get the CropBox of a page (0,0,0,0 if unset).
    Bbox get_page_crop_box(int page_index) const {
        int32_t code = 0;
        double x = 0, y = 0, w = 0, h = 0;
        if (document_editor_get_page_crop_box(ptr(),
                                              static_cast<std::uintptr_t>(page_index),
                                              &x, &y, &w, &h, &code) != 0) {
            throw Error(code, "DocumentEditor::get_page_crop_box");
        }
        return Bbox{static_cast<float>(x), static_cast<float>(y), static_cast<float>(w),
                    static_cast<float>(h)};
    }

    /// Set the CropBox of a page.
    void set_page_crop_box(int page_index, double x, double y, double w, double h) {
        int32_t code = 0;
        if (document_editor_set_page_crop_box(ptr(),
                                              static_cast<std::uintptr_t>(page_index),
                                              x, y, w, h, &code) != 0) {
            throw Error(code, "DocumentEditor::set_page_crop_box");
        }
    }

    /// Get the MediaBox of a page.
    Bbox get_page_media_box(int page_index) const {
        int32_t code = 0;
        double x = 0, y = 0, w = 0, h = 0;
        if (document_editor_get_page_media_box(ptr(),
                                               static_cast<std::uintptr_t>(page_index),
                                               &x, &y, &w, &h, &code) != 0) {
            throw Error(code, "DocumentEditor::get_page_media_box");
        }
        return Bbox{static_cast<float>(x), static_cast<float>(y), static_cast<float>(w),
                    static_cast<float>(h)};
    }

    /// Set the MediaBox of a page.
    void set_page_media_box(int page_index, double x, double y, double w, double h) {
        int32_t code = 0;
        if (document_editor_set_page_media_box(ptr(),
                                               static_cast<std::uintptr_t>(page_index),
                                               x, y, w, h, &code) != 0) {
            throw Error(code, "DocumentEditor::set_page_media_box");
        }
    }

    /// Apply (burn in) redactions on a single page (0-based).
    void apply_page_redactions(int page_index) {
        int32_t code = 0;
        if (document_editor_apply_page_redactions(
                ptr(), static_cast<std::uintptr_t>(page_index), &code) != 0) {
            throw Error(code, "DocumentEditor::apply_page_redactions");
        }
    }

    /// Apply all pending redactions across the document.
    void apply_all_redactions() {
        int32_t code = 0;
        if (document_editor_apply_all_redactions(ptr(), &code) != 0) {
            throw Error(code, "DocumentEditor::apply_all_redactions");
        }
    }

    /// Erase a single rectangular region on a page (page user-space).
    void erase_region(int page_index, float x, float y, float w, float h) {
        int32_t code = 0;
        if (document_editor_erase_region(ptr(), page_index, x, y, w, h, &code) != 0) {
            throw Error(code, "DocumentEditor::erase_region");
        }
    }

    /// Erase multiple rectangular regions on a page (page user-space).
    void erase_regions(int page_index, const std::vector<EraseRect>& rects) {
        int32_t code = 0;
        std::vector<double> flat;
        flat.reserve(rects.size() * 4);
        for (const auto& r : rects) {
            flat.push_back(r.x);
            flat.push_back(r.y);
            flat.push_back(r.width);
            flat.push_back(r.height);
        }
        if (document_editor_erase_regions(ptr(),
                                          static_cast<std::uintptr_t>(page_index),
                                          flat.data(), rects.size(), &code) != 0) {
            throw Error(code, "DocumentEditor::erase_regions");
        }
    }

    /// Clear all pending erase-region entries for a page.
    void clear_erase_regions(int page_index) {
        int32_t code = 0;
        if (document_editor_clear_erase_regions(
                ptr(), static_cast<std::uintptr_t>(page_index), &code) != 0) {
            throw Error(code, "DocumentEditor::clear_erase_regions");
        }
    }

    /// True if the page is marked for redaction.
    bool is_page_marked_for_redaction(int page_index) const {
        int32_t r = document_editor_is_page_marked_for_redaction(
            ptr(), static_cast<std::uintptr_t>(page_index));
        if (r < 0) {
            throw Error(r, "DocumentEditor::is_page_marked_for_redaction");
        }
        return r == 1;
    }

    /// Remove the redaction mark from a page.
    void unmark_page_for_redaction(int page_index) {
        int32_t code = 0;
        if (document_editor_unmark_page_for_redaction(
                ptr(), static_cast<std::uintptr_t>(page_index), &code) != 0) {
            throw Error(code, "DocumentEditor::unmark_page_for_redaction");
        }
    }

    /// Flatten all forms in the document (bake values into page content).
    void flatten_forms() {
        int32_t code = 0;
        if (document_editor_flatten_forms(ptr(), &code) != 0) {
            throw Error(code, "DocumentEditor::flatten_forms");
        }
    }

    /// Flatten forms on a specific page (0-based).
    void flatten_forms_on_page(int page_index) {
        int32_t code = 0;
        if (document_editor_flatten_forms_on_page(ptr(), page_index, &code) != 0) {
            throw Error(code, "DocumentEditor::flatten_forms_on_page");
        }
    }

    /// Flatten annotations on a single page (0-based).
    void flatten_annotations(int page_index) {
        int32_t code = 0;
        if (document_editor_flatten_annotations(ptr(), page_index, &code) != 0) {
            throw Error(code, "DocumentEditor::flatten_annotations");
        }
    }

    /// Flatten all annotations across the document.
    void flatten_all_annotations() {
        int32_t code = 0;
        if (document_editor_flatten_all_annotations(ptr(), &code) != 0) {
            throw Error(code, "DocumentEditor::flatten_all_annotations");
        }
    }

    /// Number of warnings collected during the last form-flattening save.
    int flatten_warnings_count() const {
        int32_t n = document_editor_flatten_warnings_count(ptr());
        return n < 0 ? 0 : n;
    }

    /// The `index`-th flatten warning message.
    std::string flatten_warning(int index) const {
        int32_t code = 0;
        return detail::take_string(document_editor_flatten_warning(ptr(), index, &code),
                                   code, "DocumentEditor::flatten_warning");
    }

    /// True if the page is marked for annotation-flatten.
    bool is_page_marked_for_flatten(int page_index) const {
        int32_t r = document_editor_is_page_marked_for_flatten(
            ptr(), static_cast<std::uintptr_t>(page_index));
        if (r < 0) {
            throw Error(r, "DocumentEditor::is_page_marked_for_flatten");
        }
        return r == 1;
    }

    /// Remove the flatten mark from a page.
    void unmark_page_for_flatten(int page_index) {
        int32_t code = 0;
        if (document_editor_unmark_page_for_flatten(
                ptr(), static_cast<std::uintptr_t>(page_index), &code) != 0) {
            throw Error(code, "DocumentEditor::unmark_page_for_flatten");
        }
    }

    /// Set a form field value (UTF-8) by field name.
    void set_form_field_value(const std::string& name, const std::string& value) {
        int32_t code = 0;
        if (document_editor_set_form_field_value(ptr(), name.c_str(), value.c_str(),
                                                 &code) != 0) {
            throw Error(code, "DocumentEditor::set_form_field_value");
        }
    }

    /// Merge pages from a source PDF on disk into this document.
    void merge_from(const std::string& source_path) {
        int32_t code = 0;
        if (document_editor_merge_from(ptr(), source_path.c_str(), &code) != 0) {
            throw Error(code, "DocumentEditor::merge_from");
        }
    }

    /// Merge pages from an in-memory PDF into this document.
    void merge_from_bytes(const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        if (document_editor_merge_from_bytes(ptr(), data.data(), data.size(), &code) !=
            0) {
            throw Error(code, "DocumentEditor::merge_from_bytes");
        }
    }

    /// Convert the document to PDF/A in-place.
    /// level: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u.
    void convert_to_pdf_a(int level) {
        int32_t code = 0;
        if (document_editor_convert_to_pdf_a(ptr(), level, &code) != 0) {
            throw Error(code, "DocumentEditor::convert_to_pdf_a");
        }
    }

    /// Embed a file attachment into the document.
    void embed_file(const std::string& name, const std::vector<std::uint8_t>& data) {
        int32_t code = 0;
        if (document_editor_embed_file(ptr(), name.c_str(), data.data(), data.size(),
                                       &code) != 0) {
            throw Error(code, "DocumentEditor::embed_file");
        }
    }

    /// Extract a subset of pages (0-based indices) to a new in-memory PDF.
    std::vector<std::uint8_t>
    extract_pages_to_bytes(const std::vector<int32_t>& pages) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = document_editor_extract_pages_to_bytes(
            ptr(), pages.data(), pages.size(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentEditor::extract_pages_to_bytes");
    }

    /// Save the edited document to a path.
    void save(const std::string& path) const {
        int32_t code = 0;
        if (document_editor_save(ptr(), path.c_str(), &code) != 0) {
            throw Error(code, "DocumentEditor::save");
        }
    }

    /// Save the edited document to bytes.
    std::vector<std::uint8_t> save_to_bytes() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = document_editor_save_to_bytes(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentEditor::save_to_bytes");
    }

    /// Save to bytes with compression / garbage-collect / linearize options.
    std::vector<std::uint8_t> save_to_bytes_with_options(bool compress,
                                                         bool garbage_collect,
                                                         bool linearize) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = document_editor_save_to_bytes_with_options(
            ptr(), compress, garbage_collect, linearize, &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentEditor::save_to_bytes_with_options");
    }

    /// Save the edited document AES-256 encrypted to a path.
    void save_encrypted(const std::string& path, const std::string& user_password,
                        const std::string& owner_password) const {
        int32_t code = 0;
        if (document_editor_save_encrypted(ptr(), path.c_str(), user_password.c_str(),
                                           owner_password.c_str(), &code) != 0) {
            throw Error(code, "DocumentEditor::save_encrypted");
        }
    }

    /// Save the edited document AES-256 encrypted to bytes.
    std::vector<std::uint8_t>
    save_encrypted_to_bytes(const std::string& user_password,
                            const std::string& owner_password) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = document_editor_save_encrypted_to_bytes(
            ptr(), user_password.c_str(), owner_password.c_str(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentEditor::save_encrypted_to_bytes");
    }

    // ── PHASE-7 redaction (geometric, ISO 32000-1 §12.5.6.23) ───────────────

    /// Queue a redaction region on `page` (0-based). Coordinates and the overlay
    /// fill colour (r,g,b in 0..1) are page user-space / DeviceRGB.
    void redaction_add(int page, double x1, double y1, double x2, double y2, double r,
                       double g, double b) {
        int32_t code = 0;
        if (pdf_redaction_add(ptr(), static_cast<std::uintptr_t>(page), x1, y1, x2, y2,
                              r, g, b, &code) != 0) {
            throw Error(code, "DocumentEditor::redaction_add");
        }
    }

    /// Number of queued redaction regions for `page` (0-based).
    int redaction_count(int page) const {
        int32_t code = 0;
        int32_t n =
            pdf_redaction_count(ptr(), static_cast<std::uintptr_t>(page), &code);
        if (n < 0) {
            throw Error(code, "DocumentEditor::redaction_count");
        }
        return n;
    }

    /// Destructively apply all queued redactions. `scrub_metadata` also runs the
    /// document-scrub pass; (r,g,b) is the overlay fill colour (0..1). Returns
    /// the number of glyphs physically removed.
    int redaction_apply(bool scrub_metadata, double r, double g, double b) {
        int32_t code = 0;
        int32_t removed = pdf_redaction_apply(ptr(), scrub_metadata, r, g, b, &code);
        if (removed < 0) {
            throw Error(code, "DocumentEditor::redaction_apply");
        }
        return removed;
    }

    /// Sanitize the document without geometric redaction (strip /Info, XMP,
    /// document JavaScript, embedded files). Returns the number of constructs
    /// removed.
    int redaction_scrub_metadata() {
        int32_t code = 0;
        int32_t removed = pdf_redaction_scrub_metadata(ptr(), &code);
        if (removed < 0) {
            throw Error(code, "DocumentEditor::redaction_scrub_metadata");
        }
        return removed;
    }

    /// Stamp a generated barcode/QR onto `page` (0-based) at (x,y,width,height)
    /// in page user-space points. Declared inline after Barcode is defined.
    void add_barcode_to_page(int page_index, const class Barcode& barcode, float x,
                             float y, float width, float height);

    /// Import AcroForm field values from FDF `data` bytes. Returns the C status.
    int import_fdf_bytes(const std::vector<std::uint8_t>& data) const {
        int32_t code = 0;
        int32_t r = pdf_editor_import_fdf_bytes(
            ptr(), data.empty() ? nullptr : data.data(),
            static_cast<std::uintptr_t>(data.size()), &code);
        if (code != 0) {
            throw Error(code, "DocumentEditor::import_fdf_bytes");
        }
        return r;
    }

    /// Import AcroForm field values from XFDF `data` bytes. Returns the C status.
    int import_xfdf_bytes(const std::vector<std::uint8_t>& data) const {
        int32_t code = 0;
        int32_t r = pdf_editor_import_xfdf_bytes(
            ptr(), data.empty() ? nullptr : data.data(),
            static_cast<std::uintptr_t>(data.size()), &code);
        if (code != 0) {
            throw Error(code, "DocumentEditor::import_xfdf_bytes");
        }
        return r;
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(::DocumentEditor* h) const noexcept {
            if (h)
                document_editor_free(h);
        }
    };
    explicit DocumentEditor(::DocumentEditor* h) : handle_(h) {}
    ::DocumentEditor* ptr() const {
        if (!handle_)
            throw Error(0, "DocumentEditor is closed");
        return handle_.get();
    }
    std::unique_ptr<::DocumentEditor, Deleter> handle_;
};

// ── PDF CREATION builder API ─────────────────────────────────────────────────
//
// Three owned native handles mirroring the existing pattern (string-take +
// free_string, byte-take + free_bytes, closed-handle guard, RAII free on
// destruction):
//   EmbeddedFont    — a loaded TTF/OTF font (pdf_embedded_font_*). Consumed by a
//                     successful DocumentBuilder::register_embedded_font, after
//                     which the wrapper nulls its handle so it is NOT freed twice.
//   PageBuilder     — a page under construction (pdf_page_builder_*). Each op is
//                     fluent (returns *this). done() commits + consumes the
//                     handle; close()/dtor drop it via pdf_page_builder_free.
//   DocumentBuilder — the top-level builder (pdf_document_builder_*). Spawns
//                     PageBuilders, registers fonts, and builds/saves bytes.

/// A loaded TTF/OTF font for embedding. Move-only; owns the native EmbeddedFont
/// handle and frees it on destruction — UNLESS a successful
/// DocumentBuilder::register_embedded_font has consumed it (the wrapper handle
/// is nulled then, so the builder's ownership is not double-freed).
class EmbeddedFont {
  public:
    /// Load a TTF/OTF font from a filesystem path.
    static EmbeddedFont from_file(const std::string& path) {
        int32_t code = 0;
        ::EmbeddedFont* h = pdf_embedded_font_from_file(path.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "EmbeddedFont::from_file");
        }
        return EmbeddedFont(h);
    }

    /// Load a font from a byte buffer. `name` may be empty to use the
    /// PostScript name from the font face.
    static EmbeddedFont from_bytes(const std::vector<std::uint8_t>& data,
                                   const std::string& name = "") {
        int32_t code = 0;
        ::EmbeddedFont* h = pdf_embedded_font_from_bytes(
            data.data(), data.size(), name.empty() ? nullptr : name.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "EmbeddedFont::from_bytes");
        }
        return EmbeddedFont(h);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    friend class DocumentBuilder;
    struct Deleter {
        void operator()(::EmbeddedFont* h) const noexcept {
            if (h)
                pdf_embedded_font_free(h);
        }
    };
    explicit EmbeddedFont(::EmbeddedFont* h) : handle_(h) {}
    ::EmbeddedFont* ptr() const {
        if (!handle_)
            throw Error(0, "EmbeddedFont is closed");
        return handle_.get();
    }
    /// Relinquish ownership of the native handle to the caller (the builder).
    /// After this the wrapper is empty and will not free the handle.
    ::EmbeddedFont* release() { return handle_.release(); }
    std::unique_ptr<::EmbeddedFont, Deleter> handle_;
};

/// A page under construction. Move-only; owns the native FfiPageBuilder handle.
/// Every layout op is fluent (returns *this). done() commits the buffered ops to
/// the parent DocumentBuilder and consumes the handle; after done() the wrapper
/// is empty. close()/dtor drop an uncommitted handle via pdf_page_builder_free.
class PageBuilder {
  public:
    // ── text + layout ────────────────────────────────────────────────────
    PageBuilder& font(const std::string& name, float size) {
        return op1(pdf_page_builder_font(ptr(), name.c_str(), size, &c()), "font");
    }
    PageBuilder& at(float x, float y) {
        return op1(pdf_page_builder_at(ptr(), x, y, &c()), "at");
    }
    PageBuilder& text(const std::string& t) {
        return op1(pdf_page_builder_text(ptr(), t.c_str(), &c()), "text");
    }
    PageBuilder& heading(int level, const std::string& t) {
        return op1(pdf_page_builder_heading(ptr(), static_cast<std::uint8_t>(level),
                                            t.c_str(), &c()),
                   "heading");
    }
    PageBuilder& paragraph(const std::string& t) {
        return op1(pdf_page_builder_paragraph(ptr(), t.c_str(), &c()), "paragraph");
    }
    PageBuilder& space(float points) {
        return op1(pdf_page_builder_space(ptr(), points, &c()), "space");
    }
    PageBuilder& horizontal_rule() {
        return op1(pdf_page_builder_horizontal_rule(ptr(), &c()), "horizontal_rule");
    }
    PageBuilder& columns(std::uint32_t column_count, float gap_pt,
                         const std::string& t) {
        return op1(
            pdf_page_builder_columns(ptr(), column_count, gap_pt, t.c_str(), &c()),
            "columns");
    }
    PageBuilder& footnote(const std::string& ref_mark, const std::string& note_text) {
        return op1(
            pdf_page_builder_footnote(ptr(), ref_mark.c_str(), note_text.c_str(), &c()),
            "footnote");
    }
    PageBuilder& inline_text(const std::string& t) {
        return op1(pdf_page_builder_inline(ptr(), t.c_str(), &c()), "inline_text");
    }
    PageBuilder& inline_bold(const std::string& t) {
        return op1(pdf_page_builder_inline_bold(ptr(), t.c_str(), &c()), "inline_bold");
    }
    PageBuilder& inline_italic(const std::string& t) {
        return op1(pdf_page_builder_inline_italic(ptr(), t.c_str(), &c()),
                   "inline_italic");
    }
    PageBuilder& inline_color(float r, float g, float b, const std::string& t) {
        return op1(pdf_page_builder_inline_color(ptr(), r, g, b, t.c_str(), &c()),
                   "inline_color");
    }
    PageBuilder& newline() {
        return op1(pdf_page_builder_newline(ptr(), &c()), "newline");
    }

    // ── links ────────────────────────────────────────────────────────────
    PageBuilder& link_url(const std::string& url) {
        return op1(pdf_page_builder_link_url(ptr(), url.c_str(), &c()), "link_url");
    }
    PageBuilder& link_page(int page_index) {
        return op1(pdf_page_builder_link_page(
                       ptr(), static_cast<std::uintptr_t>(page_index), &c()),
                   "link_page");
    }
    PageBuilder& link_named(const std::string& destination) {
        return op1(pdf_page_builder_link_named(ptr(), destination.c_str(), &c()),
                   "link_named");
    }
    PageBuilder& link_javascript(const std::string& script) {
        return op1(pdf_page_builder_link_javascript(ptr(), script.c_str(), &c()),
                   "link_javascript");
    }

    // ── page + field JS actions ──────────────────────────────────────────
    PageBuilder& on_open(const std::string& script) {
        return op1(pdf_page_builder_on_open(ptr(), script.c_str(), &c()), "on_open");
    }
    PageBuilder& on_close(const std::string& script) {
        return op1(pdf_page_builder_on_close(ptr(), script.c_str(), &c()), "on_close");
    }
    PageBuilder& field_keystroke(const std::string& script) {
        return op1(pdf_page_builder_field_keystroke(ptr(), script.c_str(), &c()),
                   "field_keystroke");
    }
    PageBuilder& field_format(const std::string& script) {
        return op1(pdf_page_builder_field_format(ptr(), script.c_str(), &c()),
                   "field_format");
    }
    PageBuilder& field_validate(const std::string& script) {
        return op1(pdf_page_builder_field_validate(ptr(), script.c_str(), &c()),
                   "field_validate");
    }
    PageBuilder& field_calculate(const std::string& script) {
        return op1(pdf_page_builder_field_calculate(ptr(), script.c_str(), &c()),
                   "field_calculate");
    }

    // ── text-mark annotations (RGB 0..1) ─────────────────────────────────
    PageBuilder& highlight(float r, float g, float b) {
        return op1(pdf_page_builder_highlight(ptr(), r, g, b, &c()), "highlight");
    }
    PageBuilder& underline(float r, float g, float b) {
        return op1(pdf_page_builder_underline(ptr(), r, g, b, &c()), "underline");
    }
    PageBuilder& strikeout(float r, float g, float b) {
        return op1(pdf_page_builder_strikeout(ptr(), r, g, b, &c()), "strikeout");
    }
    PageBuilder& squiggly(float r, float g, float b) {
        return op1(pdf_page_builder_squiggly(ptr(), r, g, b, &c()), "squiggly");
    }

    // ── note / watermark / stamp annotations ─────────────────────────────
    PageBuilder& sticky_note(const std::string& t) {
        return op1(pdf_page_builder_sticky_note(ptr(), t.c_str(), &c()), "sticky_note");
    }
    PageBuilder& sticky_note_at(float x, float y, const std::string& t) {
        return op1(pdf_page_builder_sticky_note_at(ptr(), x, y, t.c_str(), &c()),
                   "sticky_note_at");
    }
    PageBuilder& watermark(const std::string& t) {
        return op1(pdf_page_builder_watermark(ptr(), t.c_str(), &c()), "watermark");
    }
    PageBuilder& watermark_confidential() {
        return op1(pdf_page_builder_watermark_confidential(ptr(), &c()),
                   "watermark_confidential");
    }
    PageBuilder& watermark_draft() {
        return op1(pdf_page_builder_watermark_draft(ptr(), &c()), "watermark_draft");
    }
    PageBuilder& stamp(const std::string& type_name) {
        return op1(pdf_page_builder_stamp(ptr(), type_name.c_str(), &c()), "stamp");
    }
    PageBuilder& freetext(float x, float y, float w, float h, const std::string& t) {
        return op1(pdf_page_builder_freetext(ptr(), x, y, w, h, t.c_str(), &c()),
                   "freetext");
    }

    // ── form fields ──────────────────────────────────────────────────────
    PageBuilder& text_field(const std::string& name, float x, float y, float w, float h,
                            const std::string& default_value = "") {
        return op1(pdf_page_builder_text_field(
                       ptr(), name.c_str(), x, y, w, h,
                       default_value.empty() ? nullptr : default_value.c_str(), &c()),
                   "text_field");
    }
    PageBuilder& checkbox(const std::string& name, float x, float y, float w, float h,
                          bool checked) {
        return op1(pdf_page_builder_checkbox(ptr(), name.c_str(), x, y, w, h,
                                             checked ? 1 : 0, &c()),
                   "checkbox");
    }
    PageBuilder& combo_box(const std::string& name, float x, float y, float w, float h,
                           const std::vector<std::string>& options,
                           const std::string& selected = "") {
        std::vector<const char*> opts;
        opts.reserve(options.size());
        for (const auto& o : options) {
            opts.push_back(o.c_str());
        }
        return op1(pdf_page_builder_combo_box(
                       ptr(), name.c_str(), x, y, w, h, opts.data(), opts.size(),
                       selected.empty() ? nullptr : selected.c_str(), &c()),
                   "combo_box");
    }
    PageBuilder& radio_group(const std::string& name,
                             const std::vector<std::string>& values,
                             const std::vector<float>& xs, const std::vector<float>& ys,
                             const std::vector<float>& ws, const std::vector<float>& hs,
                             const std::string& selected = "") {
        std::vector<const char*> vals;
        vals.reserve(values.size());
        for (const auto& v : values) {
            vals.push_back(v.c_str());
        }
        return op1(pdf_page_builder_radio_group(
                       ptr(), name.c_str(), vals.data(), xs.data(), ys.data(),
                       ws.data(), hs.data(), values.size(),
                       selected.empty() ? nullptr : selected.c_str(), &c()),
                   "radio_group");
    }
    PageBuilder& push_button(const std::string& name, float x, float y, float w,
                             float h, const std::string& caption) {
        return op1(pdf_page_builder_push_button(ptr(), name.c_str(), x, y, w, h,
                                                caption.c_str(), &c()),
                   "push_button");
    }
    PageBuilder& signature_field(const std::string& name, float x, float y, float w,
                                 float h) {
        return op1(
            pdf_page_builder_signature_field(ptr(), name.c_str(), x, y, w, h, &c()),
            "signature_field");
    }

    // ── barcodes ─────────────────────────────────────────────────────────
    PageBuilder& barcode_1d(int barcode_type, const std::string& data, float x, float y,
                            float w, float h) {
        return op1(pdf_page_builder_barcode_1d(ptr(), barcode_type, data.c_str(), x, y,
                                               w, h, &c()),
                   "barcode_1d");
    }
    PageBuilder& barcode_qr(const std::string& data, float x, float y, float size) {
        return op1(pdf_page_builder_barcode_qr(ptr(), data.c_str(), x, y, size, &c()),
                   "barcode_qr");
    }

    // ── images ───────────────────────────────────────────────────────────
    PageBuilder& image(const std::vector<std::uint8_t>& bytes, float x, float y,
                       float w, float h) {
        return op1(
            pdf_page_builder_image(ptr(), bytes.data(), bytes.size(), x, y, w, h, &c()),
            "image");
    }
    PageBuilder& image_with_alt(const std::vector<std::uint8_t>& bytes, float x,
                                float y, float w, float h,
                                const std::string& alt_text) {
        return op1(pdf_page_builder_image_with_alt(ptr(), bytes.data(), bytes.size(), x,
                                                   y, w, h, alt_text.c_str(), &c()),
                   "image_with_alt");
    }
    PageBuilder& image_artifact(const std::vector<std::uint8_t>& bytes, float x,
                                float y, float w, float h) {
        return op1(pdf_page_builder_image_artifact(ptr(), bytes.data(), bytes.size(), x,
                                                   y, w, h, &c()),
                   "image_artifact");
    }

    // ── vector graphics ──────────────────────────────────────────────────
    PageBuilder& rect(float x, float y, float w, float h) {
        return op1(pdf_page_builder_rect(ptr(), x, y, w, h, &c()), "rect");
    }
    PageBuilder& filled_rect(float x, float y, float w, float h, float r, float g,
                             float b) {
        return op1(pdf_page_builder_filled_rect(ptr(), x, y, w, h, r, g, b, &c()),
                   "filled_rect");
    }
    PageBuilder& line(float x1, float y1, float x2, float y2) {
        return op1(pdf_page_builder_line(ptr(), x1, y1, x2, y2, &c()), "line");
    }
    PageBuilder& stroke_rect(float x, float y, float w, float h, float width, float r,
                             float g, float b) {
        return op1(
            pdf_page_builder_stroke_rect(ptr(), x, y, w, h, width, r, g, b, &c()),
            "stroke_rect");
    }
    PageBuilder& stroke_line(float x1, float y1, float x2, float y2, float width,
                             float r, float g, float b) {
        return op1(
            pdf_page_builder_stroke_line(ptr(), x1, y1, x2, y2, width, r, g, b, &c()),
            "stroke_line");
    }
    PageBuilder& stroke_rect_dashed(float x, float y, float w, float h, float width,
                                    float r, float g, float b,
                                    const std::vector<float>& dash_array, float phase) {
        return op1(pdf_page_builder_stroke_rect_dashed(
                       ptr(), x, y, w, h, width, r, g, b,
                       dash_array.empty() ? nullptr : dash_array.data(),
                       dash_array.size(), phase, &c()),
                   "stroke_rect_dashed");
    }
    PageBuilder& stroke_line_dashed(float x1, float y1, float x2, float y2, float width,
                                    float r, float g, float b,
                                    const std::vector<float>& dash_array, float phase) {
        return op1(pdf_page_builder_stroke_line_dashed(
                       ptr(), x1, y1, x2, y2, width, r, g, b,
                       dash_array.empty() ? nullptr : dash_array.data(),
                       dash_array.size(), phase, &c()),
                   "stroke_line_dashed");
    }
    PageBuilder& text_in_rect(float x, float y, float w, float h, const std::string& t,
                              int align) {
        return op1(
            pdf_page_builder_text_in_rect(ptr(), x, y, w, h, t.c_str(), align, &c()),
            "text_in_rect");
    }
    PageBuilder& new_page_same_size() {
        return op1(pdf_page_builder_new_page_same_size(ptr(), &c()),
                   "new_page_same_size");
    }

    // ── buffered table ───────────────────────────────────────────────────
    /// Buffer a table. `widths`/`aligns` are length n_columns; `cells` is
    /// row-major (n_rows × n_columns). `has_header` promotes the first row.
    PageBuilder& table(std::size_t n_columns, const std::vector<float>& widths,
                       const std::vector<int32_t>& aligns, std::size_t n_rows,
                       const std::vector<std::string>& cells, bool has_header) {
        std::vector<const char*> cell_ptrs;
        cell_ptrs.reserve(cells.size());
        for (const auto& s : cells) {
            cell_ptrs.push_back(s.c_str());
        }
        return op1(pdf_page_builder_table(ptr(), n_columns, widths.data(),
                                          aligns.data(), n_rows, cell_ptrs.data(),
                                          has_header ? 1 : 0, &c()),
                   "table");
    }

    // ── streaming table ──────────────────────────────────────────────────
    PageBuilder& streaming_table_begin(std::size_t n_columns,
                                       const std::vector<std::string>& headers,
                                       const std::vector<float>& widths,
                                       const std::vector<int32_t>& aligns,
                                       bool repeat_header) {
        std::vector<const char*> hdrs;
        hdrs.reserve(headers.size());
        for (const auto& s : headers) {
            hdrs.push_back(s.c_str());
        }
        return op1(pdf_page_builder_streaming_table_begin(ptr(), n_columns, hdrs.data(),
                                                          widths.data(), aligns.data(),
                                                          repeat_header ? 1 : 0, &c()),
                   "streaming_table_begin");
    }
    PageBuilder& streaming_table_begin_v2(
        std::size_t n_columns, const std::vector<std::string>& headers,
        const std::vector<float>& widths, const std::vector<int32_t>& aligns,
        bool repeat_header, int mode, std::size_t sample_rows, float min_col_width_pt,
        float max_col_width_pt, std::size_t max_rowspan) {
        std::vector<const char*> hdrs;
        hdrs.reserve(headers.size());
        for (const auto& s : headers) {
            hdrs.push_back(s.c_str());
        }
        return op1(pdf_page_builder_streaming_table_begin_v2(
                       ptr(), n_columns, hdrs.data(), widths.data(), aligns.data(),
                       repeat_header ? 1 : 0, mode, sample_rows, min_col_width_pt,
                       max_col_width_pt, max_rowspan, &c()),
                   "streaming_table_begin_v2");
    }
    PageBuilder& streaming_table_set_batch_size(std::size_t batch_size) {
        return op1(
            pdf_page_builder_streaming_table_set_batch_size(ptr(), batch_size, &c()),
            "streaming_table_set_batch_size");
    }
    std::size_t streaming_table_pending_row_count() {
        return static_cast<std::size_t>(
            pdf_page_builder_streaming_table_pending_row_count(ptr()));
    }
    std::size_t streaming_table_batch_count() {
        return static_cast<std::size_t>(
            pdf_page_builder_streaming_table_batch_count(ptr()));
    }
    PageBuilder& streaming_table_flush() {
        return op1(pdf_page_builder_streaming_table_flush(ptr(), &c()),
                   "streaming_table_flush");
    }
    PageBuilder& streaming_table_push_row(const std::vector<std::string>& cells) {
        std::vector<const char*> cell_ptrs;
        cell_ptrs.reserve(cells.size());
        for (const auto& s : cells) {
            cell_ptrs.push_back(s.c_str());
        }
        return op1(pdf_page_builder_streaming_table_push_row(ptr(), cell_ptrs.size(),
                                                             cell_ptrs.data(), &c()),
                   "streaming_table_push_row");
    }
    PageBuilder&
    streaming_table_push_row_v2(const std::vector<std::string>& cells,
                                const std::vector<std::uintptr_t>& rowspans) {
        std::vector<const char*> cell_ptrs;
        cell_ptrs.reserve(cells.size());
        for (const auto& s : cells) {
            cell_ptrs.push_back(s.c_str());
        }
        return op1(pdf_page_builder_streaming_table_push_row_v2(
                       ptr(), cell_ptrs.size(), cell_ptrs.data(),
                       rowspans.empty() ? nullptr : rowspans.data(), &c()),
                   "streaming_table_push_row_v2");
    }
    PageBuilder& streaming_table_finish() {
        return op1(pdf_page_builder_streaming_table_finish(ptr(), &c()),
                   "streaming_table_finish");
    }

    /// Commit this page's buffered ops to its parent builder. **Consumes** the
    /// handle — after a successful call the wrapper is empty (no further ops).
    void done() {
        int32_t code = 0;
        if (pdf_page_builder_done(ptr(), &code) != 0) {
            throw Error(code, "PageBuilder::done");
        }
        // The C side consumed the handle; release so the dtor does not free it.
        auto* released = handle_.release();
        static_cast<void>(released);
    }

    /// Drop an uncommitted page handle now (idempotent). RAII also frees at
    /// scope exit via pdf_page_builder_free.
    void close() { handle_.reset(); }

  private:
    friend class DocumentBuilder;
    struct Deleter {
        void operator()(::FfiPageBuilder* h) const noexcept {
            if (h)
                pdf_page_builder_free(h);
        }
    };
    explicit PageBuilder(::FfiPageBuilder* h) : handle_(h) {}
    ::FfiPageBuilder* ptr() const {
        if (!handle_)
            throw Error(0, "PageBuilder is closed");
        return handle_.get();
    }
    /// Per-call error_code scratch.
    int32_t& c() {
        code_ = 0;
        return code_;
    }
    /// Raise on a non-zero status; otherwise return *this for fluent chaining.
    PageBuilder& op1(int32_t status, const char* op) {
        if (status != 0) {
            throw Error(code_, op);
        }
        return *this;
    }
    int32_t code_ = 0;
    std::unique_ptr<::FfiPageBuilder, Deleter> handle_;
};

/// The top-level PDF builder. Move-only; owns the native FfiDocumentBuilder
/// handle and frees it on destruction. Spawns PageBuilders, registers fonts, and
/// builds/saves the document to bytes or disk.
class DocumentBuilder {
  public:
    /// Create a new, empty document builder.
    static DocumentBuilder create() {
        int32_t code = 0;
        ::FfiDocumentBuilder* h = pdf_document_builder_create(&code);
        if (h == nullptr) {
            throw Error(code, "DocumentBuilder::create");
        }
        return DocumentBuilder(h);
    }

    // ── metadata ─────────────────────────────────────────────────────────
    DocumentBuilder& set_title(const std::string& title) {
        return op1(pdf_document_builder_set_title(ptr(), title.c_str(), &c()),
                   "set_title");
    }
    DocumentBuilder& set_author(const std::string& author) {
        return op1(pdf_document_builder_set_author(ptr(), author.c_str(), &c()),
                   "set_author");
    }
    DocumentBuilder& set_subject(const std::string& subject) {
        return op1(pdf_document_builder_set_subject(ptr(), subject.c_str(), &c()),
                   "set_subject");
    }
    DocumentBuilder& set_keywords(const std::string& keywords) {
        return op1(pdf_document_builder_set_keywords(ptr(), keywords.c_str(), &c()),
                   "set_keywords");
    }
    DocumentBuilder& set_creator(const std::string& creator) {
        return op1(pdf_document_builder_set_creator(ptr(), creator.c_str(), &c()),
                   "set_creator");
    }
    DocumentBuilder& on_open(const std::string& script) {
        return op1(pdf_document_builder_on_open(ptr(), script.c_str(), &c()),
                   "on_open");
    }
    DocumentBuilder& tagged_pdf_ua1() {
        return op1(pdf_document_builder_tagged_pdf_ua1(ptr(), &c()), "tagged_pdf_ua1");
    }
    DocumentBuilder& language(const std::string& lang) {
        return op1(pdf_document_builder_language(ptr(), lang.c_str(), &c()),
                   "language");
    }
    DocumentBuilder& role_map(const std::string& custom, const std::string& standard) {
        return op1(pdf_document_builder_role_map(ptr(), custom.c_str(),
                                                 standard.c_str(), &c()),
                   "role_map");
    }

    /// Register a TTF/OTF font under `name`. On SUCCESS this **consumes** the
    /// EmbeddedFont (its native handle is released to the builder so it will not
    /// be freed twice). On error the font is left intact for retry/free.
    DocumentBuilder& register_embedded_font(const std::string& name,
                                            EmbeddedFont& font) {
        int32_t code = 0;
        int32_t status = pdf_document_builder_register_embedded_font(
            ptr(), name.c_str(), font.ptr(), &code);
        if (status != 0) {
            throw Error(code, "register_embedded_font");
        }
        // The builder took ownership; relinquish so EmbeddedFont's dtor is a no-op.
        (void)font.release();
        return *this;
    }

    // ── page factories ───────────────────────────────────────────────────
    /// Start an A4 page. Only one page may be open per builder at a time.
    PageBuilder a4_page() {
        int32_t code = 0;
        ::FfiPageBuilder* h = pdf_document_builder_a4_page(ptr(), &code);
        if (h == nullptr) {
            throw Error(code, "DocumentBuilder::a4_page");
        }
        return PageBuilder(h);
    }
    /// Start a US Letter page.
    PageBuilder letter_page() {
        int32_t code = 0;
        ::FfiPageBuilder* h = pdf_document_builder_letter_page(ptr(), &code);
        if (h == nullptr) {
            throw Error(code, "DocumentBuilder::letter_page");
        }
        return PageBuilder(h);
    }
    /// Start a page with custom dimensions in PDF points (72 pt = 1 inch).
    PageBuilder page(float width, float height) {
        int32_t code = 0;
        ::FfiPageBuilder* h = pdf_document_builder_page(ptr(), width, height, &code);
        if (h == nullptr) {
            throw Error(code, "DocumentBuilder::page");
        }
        return PageBuilder(h);
    }

    // ── output ───────────────────────────────────────────────────────────
    /// Build the PDF and return the bytes. Consumes the builder STATE (the
    /// wrapper handle stays alive and is freed by the dtor).
    std::vector<std::uint8_t> build() {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_builder_build(ptr(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentBuilder::build");
    }
    /// Build and save the PDF to `path`.
    void save(const std::string& path) {
        int32_t code = 0;
        if (pdf_document_builder_save(ptr(), path.c_str(), &code) != 0) {
            throw Error(code, "DocumentBuilder::save");
        }
    }
    /// Build and save AES-256 encrypted to `path`.
    void save_encrypted(const std::string& path, const std::string& user_password,
                        const std::string& owner_password) {
        int32_t code = 0;
        if (pdf_document_builder_save_encrypted(ptr(), path.c_str(),
                                                user_password.c_str(),
                                                owner_password.c_str(), &code) != 0) {
            throw Error(code, "DocumentBuilder::save_encrypted");
        }
    }
    /// Build encrypted bytes (AES-256).
    std::vector<std::uint8_t> to_bytes_encrypted(const std::string& user_password,
                                                 const std::string& owner_password) {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_document_builder_to_bytes_encrypted(
            ptr(), user_password.c_str(), owner_password.c_str(), &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "DocumentBuilder::to_bytes_encrypted");
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(::FfiDocumentBuilder* h) const noexcept {
            if (h)
                pdf_document_builder_free(h);
        }
    };
    explicit DocumentBuilder(::FfiDocumentBuilder* h) : handle_(h) {}
    ::FfiDocumentBuilder* ptr() const {
        if (!handle_)
            throw Error(0, "DocumentBuilder is closed");
        return handle_.get();
    }
    int32_t& c() {
        code_ = 0;
        return code_;
    }
    DocumentBuilder& op1(int32_t status, const char* op) {
        if (status != 0) {
            throw Error(code_, op);
        }
        return *this;
    }
    int32_t code_ = 0;
    std::unique_ptr<::FfiDocumentBuilder, Deleter> handle_;
};

// ── PHASE-6 digital signatures / PKI / timestamps / TSA / validation ─────────
//
// Mirrors the established pattern: owned opaque handles wrapped move-only and
// freed via their pdf_*_free on close()/dtor; string returns copied + freed
// with free_string (detail::take_string); owned byte buffers copied + freed
// with free_bytes (detail::take_bytes); a closed-handle guard on every ptr().
//
// NB: the C ABI for this phase keys failure off a NULL handle / NULL char* /
// negative or sentinel int return with *error_code set, exactly like earlier
// phases. The bool-returning accessors raise only when *error_code != 0.

/// PAdES B-LT revocation material for signBytesPades — three parallel sets of
/// DER blobs (certificates, CRLs, OCSP responses). Empty sets are fine.
struct RevocationMaterial {
    std::vector<std::vector<std::uint8_t>> certs;
    std::vector<std::vector<std::uint8_t>> crls;
    std::vector<std::vector<std::uint8_t>> ocsps;
};

namespace detail {

/// Build the (ptr-array, len-array) pair the C ABI expects for one set of byte
/// blobs. The returned vectors' storage backs the pointers, so both MUST stay
/// alive for the duration of the C call.
struct ByteArrayArray {
    std::vector<const std::uint8_t*> ptrs;
    std::vector<std::uintptr_t> lens;
    explicit ByteArrayArray(const std::vector<std::vector<std::uint8_t>>& blobs) {
        ptrs.reserve(blobs.size());
        lens.reserve(blobs.size());
        for (const auto& b : blobs) {
            ptrs.push_back(b.data());
            lens.push_back(static_cast<std::uintptr_t>(b.size()));
        }
    }
    const std::uint8_t* const* data() const { return ptrs.data(); }
    const std::uintptr_t* lengths() const { return lens.data(); }
    std::uintptr_t count() const { return static_cast<std::uintptr_t>(ptrs.size()); }
};

} // namespace detail

/// A loaded signing certificate / credential pair (opaque Certificate handle).
/// Move-only; frees via pdf_certificate_free on close()/dtor.
class Certificate {
  public:
    /// Load a PKCS#12 certificate (+ key) from DER/PFX bytes, optionally
    /// unlocking it with `password`.
    static Certificate load_from_bytes(const std::vector<std::uint8_t>& data,
                                       const std::string& password = "") {
        int32_t code = 0;
        void* h = pdf_certificate_load_from_bytes(
            data.data(), static_cast<int32_t>(data.size()),
            password.empty() ? nullptr : password.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Certificate::load_from_bytes");
        }
        return Certificate(h);
    }

    /// Load signing credentials from PEM-encoded certificate + private key.
    static Certificate load_from_pem(const std::string& cert_pem,
                                     const std::string& key_pem) {
        int32_t code = 0;
        void* h =
            pdf_certificate_load_from_pem(cert_pem.c_str(), key_pem.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "Certificate::load_from_pem");
        }
        return Certificate(h);
    }

    /// Certificate subject (distinguished name).
    std::string subject() const {
        int32_t code = 0;
        return detail::take_string(pdf_certificate_get_subject(ptr(), &code), code,
                                   "Certificate::subject");
    }

    /// Certificate issuer (distinguished name).
    std::string issuer() const {
        int32_t code = 0;
        return detail::take_string(pdf_certificate_get_issuer(ptr(), &code), code,
                                   "Certificate::issuer");
    }

    /// Certificate serial number (decimal/hex string).
    std::string serial() const {
        int32_t code = 0;
        return detail::take_string(pdf_certificate_get_serial(ptr(), &code), code,
                                   "Certificate::serial");
    }

    /// Validity window as Unix epoch seconds (not_before, not_after).
    std::pair<std::int64_t, std::int64_t> validity() const {
        int32_t code = 0;
        std::int64_t not_before = 0, not_after = 0;
        pdf_certificate_get_validity(ptr(), &not_before, &not_after, &code);
        if (code != 0) {
            throw Error(code, "Certificate::validity");
        }
        return {not_before, not_after};
    }

    /// True if the certificate is currently within its validity window.
    bool is_valid() const {
        int32_t code = 0;
        int32_t r = pdf_certificate_is_valid(ptr(), &code);
        if (r < 0) {
            throw Error(code, "Certificate::is_valid");
        }
        return r == 1;
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

    /// Borrow the raw native handle (non-owning) for the signing free functions.
    /// The Certificate retains ownership.
    const void* handle_get() const { return ptr(); }

  private:
    friend class SignatureInfo;
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_certificate_free(h);
        }
    };
    explicit Certificate(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "Certificate is closed");
        return handle_.get();
    }
    std::unique_ptr<void, Deleter> handle_;
};

/// A parsed RFC 3161 timestamp token (opaque Timestamp handle). Move-only;
/// frees via pdf_timestamp_free on close()/dtor.
class Timestamp {
  public:
    /// Parse a DER-encoded TimeStampToken (or bare TSTInfo).
    static Timestamp parse(const std::vector<std::uint8_t>& bytes) {
        int32_t code = 0;
        void* h = pdf_timestamp_parse(bytes.data(),
                                      static_cast<std::uintptr_t>(bytes.size()), &code);
        if (h == nullptr) {
            throw Error(code, "Timestamp::parse");
        }
        return Timestamp(h);
    }

    /// The raw DER token bytes. The C ABI returns a borrowed const buffer that
    /// is owned by the handle, so we COPY it and do NOT free it.
    std::vector<std::uint8_t> token() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        const std::uint8_t* p = pdf_timestamp_get_token(ptr(), &out_len, &code);
        if (p == nullptr) {
            throw Error(code, "Timestamp::token");
        }
        return std::vector<std::uint8_t>(p, p + out_len);
    }

    /// The message imprint (hashed value). Borrowed const buffer → COPY, no free.
    std::vector<std::uint8_t> message_imprint() const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        const std::uint8_t* p =
            pdf_timestamp_get_message_imprint(ptr(), &out_len, &code);
        if (p == nullptr) {
            throw Error(code, "Timestamp::message_imprint");
        }
        return std::vector<std::uint8_t>(p, p + out_len);
    }

    /// Timestamp time as Unix epoch seconds.
    std::int64_t time() const {
        int32_t code = 0;
        std::int64_t t = pdf_timestamp_get_time(ptr(), &code);
        if (code != 0) {
            throw Error(code, "Timestamp::time");
        }
        return t;
    }

    /// Timestamp serial number.
    std::string serial() const {
        int32_t code = 0;
        return detail::take_string(pdf_timestamp_get_serial(ptr(), &code), code,
                                   "Timestamp::serial");
    }

    /// Issuing TSA name.
    std::string tsa_name() const {
        int32_t code = 0;
        return detail::take_string(pdf_timestamp_get_tsa_name(ptr(), &code), code,
                                   "Timestamp::tsa_name");
    }

    /// TSA policy OID.
    std::string policy_oid() const {
        int32_t code = 0;
        return detail::take_string(pdf_timestamp_get_policy_oid(ptr(), &code), code,
                                   "Timestamp::policy_oid");
    }

    /// Hash algorithm code used for the message imprint.
    int hash_algorithm() const {
        int32_t code = 0;
        int32_t a = pdf_timestamp_get_hash_algorithm(ptr(), &code);
        if (a < 0) {
            throw Error(code, "Timestamp::hash_algorithm");
        }
        return a;
    }

    /// Verify the timestamp token's internal consistency.
    bool verify() const {
        int32_t code = 0;
        bool ok = pdf_timestamp_verify(ptr(), &code);
        if (!ok && code != 0) {
            throw Error(code, "Timestamp::verify");
        }
        return ok;
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    friend class SignatureInfo;
    friend class TsaClient;
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_timestamp_free(h);
        }
    };
    explicit Timestamp(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "Timestamp is closed");
        return handle_.get();
    }
    void* raw() const { return ptr(); }
    std::unique_ptr<void, Deleter> handle_;
};

/// A parsed PDF signature (FfiSignatureInfo handle). Move-only; frees via
/// pdf_signature_free on close()/dtor.
class SignatureInfo {
  public:
    /// Wrap an FfiSignatureInfo handle obtained from the C ABI (takes ownership).
    static SignatureInfo from_handle(FfiSignatureInfo* h) {
        if (h == nullptr) {
            throw Error(0, "SignatureInfo::from_handle");
        }
        return SignatureInfo(h);
    }

    /// Signer common name.
    std::string signer_name() const {
        int32_t code = 0;
        return detail::take_string(pdf_signature_get_signer_name(ptr(), &code), code,
                                   "SignatureInfo::signer_name");
    }

    /// Stated signing reason.
    std::string signing_reason() const {
        int32_t code = 0;
        return detail::take_string(pdf_signature_get_signing_reason(ptr(), &code), code,
                                   "SignatureInfo::signing_reason");
    }

    /// Stated signing location.
    std::string signing_location() const {
        int32_t code = 0;
        return detail::take_string(pdf_signature_get_signing_location(ptr(), &code),
                                   code, "SignatureInfo::signing_location");
    }

    /// Signing time as Unix epoch seconds.
    std::int64_t signing_time() const {
        int32_t code = 0;
        std::int64_t t = pdf_signature_get_signing_time(ptr(), &code);
        if (code != 0) {
            throw Error(code, "SignatureInfo::signing_time");
        }
        return t;
    }

    /// The signer's certificate.
    Certificate certificate() const {
        int32_t code = 0;
        void* h = pdf_signature_get_certificate(ptr(), &code);
        if (h == nullptr) {
            throw Error(code, "SignatureInfo::certificate");
        }
        return Certificate(h);
    }

    /// Classified PAdES level (B-B/B-T; B-LT needs the DSS).
    int pades_level() const {
        int32_t code = 0;
        int32_t lvl = pdf_signature_get_pades_level(ptr(), &code);
        if (lvl < 0) {
            throw Error(code, "SignatureInfo::pades_level");
        }
        return lvl;
    }

    /// True if the signature carries a timestamp.
    bool has_timestamp() const {
        int32_t code = 0;
        bool r = pdf_signature_has_timestamp(ptr(), &code);
        if (!r && code != 0) {
            throw Error(code, "SignatureInfo::has_timestamp");
        }
        return r;
    }

    /// The signature's timestamp.
    Timestamp timestamp() const {
        int32_t code = 0;
        void* h = pdf_signature_get_timestamp(ptr(), &code);
        if (h == nullptr) {
            throw Error(code, "SignatureInfo::timestamp");
        }
        return Timestamp(h);
    }

    /// Attach a timestamp to this signature. Returns true on success.
    bool add_timestamp(const Timestamp& ts) {
        int32_t code = 0;
        bool ok = pdf_signature_add_timestamp(ptr(), ts.raw(), &code);
        if (!ok && code != 0) {
            throw Error(code, "SignatureInfo::add_timestamp");
        }
        return ok;
    }

    /// Run the signer-attributes crypto check. Returns the raw tri-state code
    /// (1 = valid, 0 = invalid, -1 = unknown/unsupported).
    int verify() const {
        int32_t code = 0;
        return pdf_signature_verify(ptr(), &code);
    }

    /// End-to-end detached verification against the full PDF bytes. Returns the
    /// raw tri-state code (1 = valid, 0 = invalid, -1 = unknown/unsupported).
    int verify_detached(const std::vector<std::uint8_t>& pdf_data) const {
        int32_t code = 0;
        return pdf_signature_verify_detached(
            ptr(), pdf_data.data(), static_cast<std::uintptr_t>(pdf_data.size()),
            &code);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(FfiSignatureInfo* h) const noexcept {
            if (h)
                pdf_signature_free(h);
        }
    };
    explicit SignatureInfo(FfiSignatureInfo* h) : handle_(h) {}
    FfiSignatureInfo* ptr() const {
        if (!handle_)
            throw Error(0, "SignatureInfo is closed");
        return handle_.get();
    }
    std::unique_ptr<FfiSignatureInfo, Deleter> handle_;
};

/// An RFC 3161 TSA client (opaque handle). Move-only; frees via
/// pdf_tsa_client_free on close()/dtor.
class TsaClient {
  public:
    /// Create a TSA client for `url`. `username`/`password` may be empty for
    /// anonymous TSAs.
    static TsaClient create(const std::string& url, const std::string& username = "",
                            const std::string& password = "", int timeout = 30,
                            int hash_algo = 0, bool use_nonce = true,
                            bool cert_req = true) {
        int32_t code = 0;
        void* h = pdf_tsa_client_create(url.c_str(),
                                        username.empty() ? nullptr : username.c_str(),
                                        password.empty() ? nullptr : password.c_str(),
                                        timeout, hash_algo, use_nonce, cert_req, &code);
        if (h == nullptr) {
            throw Error(code, "TsaClient::create");
        }
        return TsaClient(h);
    }

    /// Request a timestamp over `data` (the TSA hashes it).
    Timestamp request_timestamp(const std::vector<std::uint8_t>& data) const {
        int32_t code = 0;
        void* h = pdf_tsa_request_timestamp(
            ptr(), data.data(), static_cast<std::uintptr_t>(data.size()), &code);
        if (h == nullptr) {
            throw Error(code, "TsaClient::request_timestamp");
        }
        return Timestamp(h);
    }

    /// Request a timestamp over a precomputed `hash` (with `hash_algo`).
    Timestamp request_timestamp_hash(const std::vector<std::uint8_t>& hash,
                                     int hash_algo) const {
        int32_t code = 0;
        void* h = pdf_tsa_request_timestamp_hash(
            ptr(), hash.data(), static_cast<std::uintptr_t>(hash.size()), hash_algo,
            &code);
        if (h == nullptr) {
            throw Error(code, "TsaClient::request_timestamp_hash");
        }
        return Timestamp(h);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_tsa_client_free(h);
        }
    };
    explicit TsaClient(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "TsaClient is closed");
        return handle_.get();
    }
    std::unique_ptr<void, Deleter> handle_;
};

/// A Document Security Store (opaque DSS handle). Move-only; frees via
/// pdf_dss_free on close()/dtor.
class Dss {
  public:
    /// Wrap a DSS handle obtained from the C ABI (takes ownership).
    static Dss from_handle(void* h) {
        if (h == nullptr) {
            throw Error(0, "Dss::from_handle");
        }
        return Dss(h);
    }

    /// Number of certificates in the DSS.
    int cert_count() const {
        int32_t n = pdf_dss_cert_count(ptr());
        return n < 0 ? 0 : n;
    }
    /// Number of CRLs in the DSS.
    int crl_count() const {
        int32_t n = pdf_dss_crl_count(ptr());
        return n < 0 ? 0 : n;
    }
    /// Number of OCSP responses in the DSS.
    int ocsp_count() const {
        int32_t n = pdf_dss_ocsp_count(ptr());
        return n < 0 ? 0 : n;
    }
    /// Number of VRI (validation-related info) entries in the DSS.
    int vri_count() const {
        int32_t n = pdf_dss_vri_count(ptr());
        return n < 0 ? 0 : n;
    }

    /// DER bytes of the `index`-th certificate.
    std::vector<std::uint8_t> get_cert(int index) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_dss_get_cert(ptr(), index, &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Dss::get_cert");
    }
    /// DER bytes of the `index`-th CRL.
    std::vector<std::uint8_t> get_crl(int index) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_dss_get_crl(ptr(), index, &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Dss::get_crl");
    }
    /// DER bytes of the `index`-th OCSP response.
    std::vector<std::uint8_t> get_ocsp(int index) const {
        int32_t code = 0;
        std::uintptr_t out_len = 0;
        std::uint8_t* p = pdf_dss_get_ocsp(ptr(), index, &out_len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                                  "Dss::get_ocsp");
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_dss_free(h);
        }
    };
    explicit Dss(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "Dss is closed");
        return handle_.get();
    }
    std::unique_ptr<void, Deleter> handle_;
};

/// PDF/A conformance result (FfiPdfAResults handle). Move-only; frees via
/// pdf_pdf_a_results_free on close()/dtor.
class PdfAResults {
  public:
    static PdfAResults from_handle(FfiPdfAResults* h) {
        if (h == nullptr) {
            throw Error(0, "PdfAResults::from_handle");
        }
        return PdfAResults(h);
    }
    /// True if the document is PDF/A compliant at the requested level.
    bool is_compliant() const {
        int32_t code = 0;
        bool r = pdf_pdf_a_is_compliant(ptr(), &code);
        if (!r && code != 0) {
            throw Error(code, "PdfAResults::is_compliant");
        }
        return r;
    }
    /// Conformance error messages.
    std::vector<std::string> errors() const {
        std::vector<std::string> out;
        int32_t n = pdf_pdf_a_error_count(ptr());
        for (int32_t i = 0; i < n; ++i) {
            int32_t code = 0;
            out.push_back(detail::take_string(pdf_pdf_a_get_error(ptr(), i, &code),
                                              code, "PdfAResults::errors"));
        }
        return out;
    }
    /// Number of conformance warnings.
    int warning_count() const {
        int32_t n = pdf_pdf_a_warning_count(ptr());
        return n < 0 ? 0 : n;
    }

    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(FfiPdfAResults* h) const noexcept {
            if (h)
                pdf_pdf_a_results_free(h);
        }
    };
    explicit PdfAResults(FfiPdfAResults* h) : handle_(h) {}
    FfiPdfAResults* ptr() const {
        if (!handle_)
            throw Error(0, "PdfAResults is closed");
        return handle_.get();
    }
    std::unique_ptr<FfiPdfAResults, Deleter> handle_;
};

/// PDF/UA accessibility statistics returned by UaResults::stats().
struct UaStats {
    int structure;
    int images;
    int tables;
    int forms;
    int annotations;
    int pages;
};

/// PDF/UA accessibility result (FfiUaResults handle). Move-only; frees via
/// pdf_pdf_ua_results_free on close()/dtor.
class UaResults {
  public:
    static UaResults from_handle(FfiUaResults* h) {
        if (h == nullptr) {
            throw Error(0, "UaResults::from_handle");
        }
        return UaResults(h);
    }
    /// True if the document is accessible at the requested level.
    bool is_accessible() const {
        int32_t code = 0;
        bool r = pdf_pdf_ua_is_accessible(ptr(), &code);
        if (!r && code != 0) {
            throw Error(code, "UaResults::is_accessible");
        }
        return r;
    }
    /// Accessibility error messages.
    std::vector<std::string> errors() const {
        std::vector<std::string> out;
        int32_t n = pdf_pdf_ua_error_count(ptr());
        for (int32_t i = 0; i < n; ++i) {
            int32_t code = 0;
            out.push_back(detail::take_string(pdf_pdf_ua_get_error(ptr(), i, &code),
                                              code, "UaResults::errors"));
        }
        return out;
    }
    /// Accessibility warning messages.
    std::vector<std::string> warnings() const {
        std::vector<std::string> out;
        int32_t n = pdf_pdf_ua_warning_count(ptr());
        for (int32_t i = 0; i < n; ++i) {
            int32_t code = 0;
            out.push_back(detail::take_string(pdf_pdf_ua_get_warning(ptr(), i, &code),
                                              code, "UaResults::warnings"));
        }
        return out;
    }
    /// Accessibility element counts.
    UaStats stats() const {
        int32_t code = 0;
        int32_t s = 0, im = 0, t = 0, f = 0, a = 0, p = 0;
        if (!pdf_pdf_ua_get_stats(ptr(), &s, &im, &t, &f, &a, &p, &code)) {
            throw Error(code, "UaResults::stats");
        }
        return UaStats{s, im, t, f, a, p};
    }

    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(FfiUaResults* h) const noexcept {
            if (h)
                pdf_pdf_ua_results_free(h);
        }
    };
    explicit UaResults(FfiUaResults* h) : handle_(h) {}
    FfiUaResults* ptr() const {
        if (!handle_)
            throw Error(0, "UaResults is closed");
        return handle_.get();
    }
    std::unique_ptr<FfiUaResults, Deleter> handle_;
};

/// PDF/X conformance result (FfiPdfXResults handle). Move-only; frees via
/// pdf_pdf_x_results_free on close()/dtor.
class PdfXResults {
  public:
    static PdfXResults from_handle(FfiPdfXResults* h) {
        if (h == nullptr) {
            throw Error(0, "PdfXResults::from_handle");
        }
        return PdfXResults(h);
    }
    /// True if the document is PDF/X compliant at the requested level.
    bool is_compliant() const {
        int32_t code = 0;
        bool r = pdf_pdf_x_is_compliant(ptr(), &code);
        if (!r && code != 0) {
            throw Error(code, "PdfXResults::is_compliant");
        }
        return r;
    }
    /// Conformance error messages.
    std::vector<std::string> errors() const {
        std::vector<std::string> out;
        int32_t n = pdf_pdf_x_error_count(ptr());
        for (int32_t i = 0; i < n; ++i) {
            int32_t code = 0;
            out.push_back(detail::take_string(pdf_pdf_x_get_error(ptr(), i, &code),
                                              code, "PdfXResults::errors"));
        }
        return out;
    }

    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(FfiPdfXResults* h) const noexcept {
            if (h)
                pdf_pdf_x_results_free(h);
        }
    };
    explicit PdfXResults(FfiPdfXResults* h) : handle_(h) {}
    FfiPdfXResults* ptr() const {
        if (!handle_)
            throw Error(0, "PdfXResults is closed");
        return handle_.get();
    }
    std::unique_ptr<FfiPdfXResults, Deleter> handle_;
};

// ── Document-level PHASE-6 validation (out-of-line; need the result types) ───

inline PdfAResults Document::validate_pdf_a(int level) const {
    int32_t code = 0;
    FfiPdfAResults* h =
        pdf_validate_pdf_a_level(ptr(), static_cast<int32_t>(level), &code);
    if (h == nullptr) {
        throw Error(code, "Document::validate_pdf_a");
    }
    return PdfAResults::from_handle(h);
}

inline UaResults Document::validate_pdf_ua(int level) const {
    int32_t code = 0;
    FfiUaResults* h = pdf_validate_pdf_ua(ptr(), static_cast<int32_t>(level), &code);
    if (h == nullptr) {
        throw Error(code, "Document::validate_pdf_ua");
    }
    return UaResults::from_handle(h);
}

inline PdfXResults Document::validate_pdf_x(int level) const {
    int32_t code = 0;
    FfiPdfXResults* h =
        pdf_validate_pdf_x_level(ptr(), static_cast<int32_t>(level), &code);
    if (h == nullptr) {
        throw Error(code, "Document::validate_pdf_x");
    }
    return PdfXResults::from_handle(h);
}

inline Dss Document::get_dss() const {
    int32_t code = 0;
    void* h = pdf_document_get_dss(ptr(), &code);
    if (h == nullptr) {
        throw Error(code, "Document::get_dss");
    }
    return Dss::from_handle(h);
}

inline int Document::sign(const Certificate& cert, const std::string& reason,
                          const std::string& location) const {
    int32_t code = 0;
    int32_t r = pdf_document_sign(ptr(), cert.handle_get(),
                                  reason.empty() ? nullptr : reason.c_str(),
                                  location.empty() ? nullptr : location.c_str(), &code);
    if (code != 0) {
        throw Error(code, "Document::sign");
    }
    return r;
}

inline SignatureInfo Document::get_signature(int index) const {
    int32_t code = 0;
    void* h = pdf_document_get_signature(ptr(), index, &code);
    if (h == nullptr) {
        throw Error(code, "Document::get_signature");
    }
    return SignatureInfo::from_handle(static_cast<FfiSignatureInfo*>(h));
}

// ── Top-level PHASE-6 free functions ─────────────────────────────────────────

/// Set the global library log level (0=Off 1=Error 2=Warn 3=Info 4=Debug
/// 5=Trace).
inline void set_log_level(int level) {
    pdf_oxide_set_log_level(static_cast<int32_t>(level));
}

/// Get the current global library log level.
inline int get_log_level() { return static_cast<int>(pdf_oxide_get_log_level()); }

/// Sign raw PDF bytes with `cert`, returning the signed PDF.
inline std::vector<std::uint8_t> sign_bytes(const std::vector<std::uint8_t>& pdf_data,
                                            const Certificate& cert,
                                            const std::string& reason = "",
                                            const std::string& location = "") {
    int32_t code = 0;
    std::uintptr_t out_len = 0;
    std::uint8_t* p =
        pdf_sign_bytes(pdf_data.data(), static_cast<std::uintptr_t>(pdf_data.size()),
                       cert.handle_get(), reason.empty() ? nullptr : reason.c_str(),
                       location.empty() ? nullptr : location.c_str(), &out_len, &code);
    return detail::take_bytes(p, static_cast<std::size_t>(out_len), code, "sign_bytes");
}

/// Sign raw PDF bytes at a PAdES baseline `level` (0=B-B 1=B-T 2=B-LT). `tsa_url`
/// is required for level >= 1; `revocation` carries B-LT material.
inline std::vector<std::uint8_t>
sign_bytes_pades(const std::vector<std::uint8_t>& pdf_data, const Certificate& cert,
                 int level, const std::string& tsa_url = "",
                 const std::string& reason = "", const std::string& location = "",
                 const RevocationMaterial& revocation = {}) {
    detail::ByteArrayArray certs(revocation.certs);
    detail::ByteArrayArray crls(revocation.crls);
    detail::ByteArrayArray ocsps(revocation.ocsps);
    int32_t code = 0;
    std::uintptr_t out_len = 0;
    std::uint8_t* p = pdf_sign_bytes_pades(
        pdf_data.data(), static_cast<std::uintptr_t>(pdf_data.size()),
        cert.handle_get(), static_cast<int32_t>(level),
        tsa_url.empty() ? nullptr : tsa_url.c_str(),
        reason.empty() ? nullptr : reason.c_str(),
        location.empty() ? nullptr : location.c_str(), certs.data(), certs.lengths(),
        certs.count(), crls.data(), crls.lengths(), crls.count(), ocsps.data(),
        ocsps.lengths(), ocsps.count(), &out_len, &code);
    return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                              "sign_bytes_pades");
}

/// Struct-options variant of sign_bytes_pades — builds the PadesSignOptionsC and
/// delegates. Functionally identical to sign_bytes_pades.
inline std::vector<std::uint8_t> sign_bytes_pades_opts(
    const std::vector<std::uint8_t>& pdf_data, const Certificate& cert, int level,
    const std::string& tsa_url = "", const std::string& reason = "",
    const std::string& location = "", const RevocationMaterial& revocation = {}) {
    detail::ByteArrayArray certs(revocation.certs);
    detail::ByteArrayArray crls(revocation.crls);
    detail::ByteArrayArray ocsps(revocation.ocsps);
    PadesSignOptionsC options{};
    options.certificate_handle = cert.handle_get();
    options.certs = certs.data();
    options.cert_lens = certs.lengths();
    options.n_certs = certs.count();
    options.crls = crls.data();
    options.crl_lens = crls.lengths();
    options.n_crls = crls.count();
    options.ocsps = ocsps.data();
    options.ocsp_lens = ocsps.lengths();
    options.n_ocsps = ocsps.count();
    options.tsa_url = tsa_url.empty() ? nullptr : tsa_url.c_str();
    options.reason = reason.empty() ? nullptr : reason.c_str();
    options.location = location.empty() ? nullptr : location.c_str();
    options.level = static_cast<int32_t>(level);
    int32_t code = 0;
    std::uintptr_t out_len = 0;
    std::uint8_t* p = pdf_sign_bytes_pades_opts(
        pdf_data.data(), static_cast<std::uintptr_t>(pdf_data.size()), &options,
        &out_len, &code);
    return detail::take_bytes(p, static_cast<std::size_t>(out_len), code,
                              "sign_bytes_pades_opts");
}

// ── PHASE-7 barcodes / OCR / standalone renderer / merge / timestamp ─────────
//
// Same established pattern: owned opaque handles wrapped move-only and freed via
// their pdf_*_free on close()/dtor; string returns copied + freed with
// free_string (detail::take_string); owned byte buffers copied + freed with
// free_bytes (detail::take_bytes); a closed-handle guard on every ptr(). The C
// ABI keys failure off a NULL handle / NULL char* / negative-or-sentinel int
// return with *error_code set, exactly like earlier phases.

/// A generated 1-D barcode or 2-D QR code (opaque FfiBarcodeImage handle).
/// Move-only; frees via pdf_barcode_free on close()/dtor.
class Barcode {
  public:
    /// Generate a QR code from `data`. `error_correction` selects the EC level
    /// (0=L 1=M 2=Q 3=H); `size_px` is the requested raster size in pixels.
    static Barcode generate_qr_code(const std::string& data, int error_correction = 1,
                                    int size_px = 256) {
        int32_t code = 0;
        FfiBarcodeImage* h =
            pdf_generate_qr_code(data.c_str(), error_correction, size_px, &code);
        if (h == nullptr) {
            throw Error(code, "Barcode::generate_qr_code");
        }
        return Barcode(h);
    }

    /// Generate a barcode from `data`. `format` selects the symbology; `size_px`
    /// is the requested raster size in pixels.
    static Barcode generate_barcode(const std::string& data, int format,
                                    int size_px = 256) {
        int32_t code = 0;
        FfiBarcodeImage* h = pdf_generate_barcode(data.c_str(), format, size_px, &code);
        if (h == nullptr) {
            throw Error(code, "Barcode::generate_barcode");
        }
        return Barcode(h);
    }

    /// The encoded data string carried by the barcode.
    std::string get_data() const {
        int32_t code = 0;
        return detail::take_string(pdf_barcode_get_data(ptr(), &code), code,
                                   "Barcode::get_data");
    }

    /// The barcode format/symbology code.
    int get_format() const {
        int32_t code = 0;
        int32_t f = pdf_barcode_get_format(ptr(), &code);
        if (f < 0 || code != 0) {
            throw Error(code, "Barcode::get_format");
        }
        return f;
    }

    /// Decode confidence (0..1; meaningful only for decoded barcodes).
    float get_confidence() const {
        int32_t code = 0;
        float c = pdf_barcode_get_confidence(ptr(), &code);
        if (code != 0) {
            throw Error(code, "Barcode::get_confidence");
        }
        return c;
    }

    /// Render the barcode to PNG bytes at `size_px`.
    std::vector<std::uint8_t> get_image_png(int size_px = 256) const {
        int32_t code = 0;
        int32_t out_len = 0;
        std::uint8_t* p = pdf_barcode_get_image_png(ptr(), size_px, &out_len, &code);
        return detail::take_bytes(p,
                                  static_cast<std::size_t>(out_len < 0 ? 0 : out_len),
                                  code, "Barcode::get_image_png");
    }

    /// Render the barcode to an SVG string at `size_px`.
    std::string get_svg(int size_px = 256) const {
        int32_t code = 0;
        return detail::take_string(pdf_barcode_get_svg(ptr(), size_px, &code), code,
                                   "Barcode::get_svg");
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    friend class DocumentEditor;
    struct Deleter {
        void operator()(FfiBarcodeImage* h) const noexcept {
            if (h)
                pdf_barcode_free(h);
        }
    };
    explicit Barcode(FfiBarcodeImage* h) : handle_(h) {}
    FfiBarcodeImage* ptr() const {
        if (!handle_)
            throw Error(0, "Barcode is closed");
        return handle_.get();
    }
    /// Borrow the raw native handle (non-owning) for add_barcode_to_page.
    const FfiBarcodeImage* raw() const { return ptr(); }
    std::unique_ptr<FfiBarcodeImage, Deleter> handle_;
};

/// An OCR engine loaded from detection/recognition models + a dictionary
/// (opaque handle). Move-only; frees via pdf_ocr_engine_free on close()/dtor.
class OcrEngine {
  public:
    /// Create an OCR engine from model/dictionary file paths.
    static OcrEngine create(const std::string& det_model_path,
                            const std::string& rec_model_path,
                            const std::string& dict_path) {
        int32_t code = 0;
        void* h = pdf_ocr_engine_create(det_model_path.c_str(), rec_model_path.c_str(),
                                        dict_path.c_str(), &code);
        if (h == nullptr) {
            throw Error(code, "OcrEngine::create");
        }
        return OcrEngine(h);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

    /// Borrow the raw native handle (non-owning) for Document::ocr_extract_text.
    const void* handle_get() const { return ptr(); }

  private:
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_ocr_engine_free(h);
        }
    };
    explicit OcrEngine(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "OcrEngine is closed");
        return handle_.get();
    }
    std::unique_ptr<void, Deleter> handle_;
};

/// A standalone reusable renderer configuration (opaque handle). Move-only;
/// frees via pdf_renderer_free on close()/dtor.
class Renderer {
  public:
    /// Create a renderer with a `dpi`, output `format` (0=PNG 1=JPEG), JPEG
    /// `quality` (0..100), and anti-aliasing toggle.
    static Renderer create(int dpi, int format, int quality, bool anti_alias) {
        int32_t code = 0;
        void* h = pdf_create_renderer(dpi, format, quality, anti_alias, &code);
        if (h == nullptr) {
            throw Error(code, "Renderer::create");
        }
        return Renderer(h);
    }

    /// Free the native handle now (idempotent). RAII also frees at scope exit.
    void close() { handle_.reset(); }

  private:
    struct Deleter {
        void operator()(void* h) const noexcept {
            if (h)
                pdf_renderer_free(h);
        }
    };
    explicit Renderer(void* h) : handle_(h) {}
    void* ptr() const {
        if (!handle_)
            throw Error(0, "Renderer is closed");
        return handle_.get();
    }
    std::unique_ptr<void, Deleter> handle_;
};

// ── PHASE-7 out-of-line definitions (need the classes above) ─────────────────

inline std::string Document::ocr_extract_text(int page_index,
                                              const OcrEngine* engine) const {
    int32_t code = 0;
    const void* eng = engine != nullptr ? engine->handle_get() : nullptr;
    return detail::take_string(pdf_ocr_extract_text(ptr(), page_index, eng, &code),
                               code, "Document::ocr_extract_text");
}

inline void DocumentEditor::add_barcode_to_page(int page_index, const Barcode& barcode,
                                                float x, float y, float width,
                                                float height) {
    int32_t code = 0;
    if (pdf_add_barcode_to_page(ptr(), page_index, barcode.raw(), x, y, width, height,
                                &code) != 0) {
        throw Error(code, "DocumentEditor::add_barcode_to_page");
    }
}

// ── PHASE-7 top-level free functions ─────────────────────────────────────────

/// Merge multiple PDFs (by filesystem path, in order) into one, returning the
/// merged PDF bytes.
inline std::vector<std::uint8_t> merge(const std::vector<std::string>& paths) {
    std::vector<const char*> path_ptrs;
    path_ptrs.reserve(paths.size());
    for (const auto& p : paths) {
        path_ptrs.push_back(p.c_str());
    }
    int32_t code = 0;
    int32_t data_len = 0;
    std::uint8_t* p =
        pdf_merge(path_ptrs.empty() ? nullptr : path_ptrs.data(),
                  static_cast<int32_t>(path_ptrs.size()), &data_len, &code);
    return detail::take_bytes(p, static_cast<std::size_t>(data_len < 0 ? 0 : data_len),
                              code, "merge");
}

/// Add a document timestamp (RFC 3161) over the signature at `sig_index` using
/// the TSA at `tsa_url`, returning the timestamped PDF bytes.
inline std::vector<std::uint8_t>
add_timestamp(const std::vector<std::uint8_t>& pdf_data, int sig_index,
              const std::string& tsa_url) {
    std::uint8_t* out_data = nullptr;
    std::uintptr_t out_len = 0;
    int32_t code = 0;
    bool ok =
        pdf_add_timestamp(pdf_data.data(), static_cast<std::uintptr_t>(pdf_data.size()),
                          sig_index, tsa_url.c_str(), &out_data, &out_len, &code);
    if (!ok) {
        throw Error(code, "add_timestamp");
    }
    return detail::take_bytes(out_data, static_cast<std::size_t>(out_len), code,
                              "add_timestamp");
}

// ── PHASE-8 top-level: crypto / FIPS provider ────────────────────────────────

/// The name of the active crypto provider.
inline std::string crypto_active_provider() {
    return detail::take_string(pdf_oxide_crypto_active_provider(), 0,
                               "crypto_active_provider");
}
/// The crypto bill-of-materials (CBOM) as JSON.
inline std::string crypto_cbom() {
    return detail::take_string(pdf_oxide_crypto_cbom(), 0, "crypto_cbom");
}
/// 1 if a FIPS-validated provider is available, 0 otherwise.
inline int crypto_fips_available() { return pdf_oxide_crypto_fips_available(); }
/// The crypto algorithm inventory as JSON.
inline std::string crypto_inventory() {
    return detail::take_string(pdf_oxide_crypto_inventory(), 0, "crypto_inventory");
}
/// The active crypto policy as a string.
inline std::string crypto_policy() {
    return detail::take_string(pdf_oxide_crypto_policy(), 0, "crypto_policy");
}
/// Set the crypto policy from `spec`. Returns the C status code.
inline int crypto_set_policy(const std::string& spec) {
    return pdf_oxide_crypto_set_policy(spec.c_str());
}
/// Switch to the FIPS crypto provider. Returns the C status code.
inline int crypto_use_fips() { return pdf_oxide_crypto_use_fips(); }

// ── PHASE-8 top-level: models / prefetch ─────────────────────────────────────

/// The bundled/available model manifest as JSON.
inline std::string model_manifest() {
    return detail::take_string(pdf_oxide_model_manifest(), 0, "model_manifest");
}
/// 1 if model prefetch is available (network/feature enabled), 0 otherwise.
inline int prefetch_available() { return pdf_oxide_prefetch_available(); }
/// Prefetch OCR/layout models for the comma-separated `languages_csv`. Returns a
/// JSON status. May legitimately error without network/models.
inline std::string prefetch_models(const std::string& languages_csv) {
    int32_t code = 0;
    return detail::take_string(pdf_oxide_prefetch_models(languages_csv.c_str(), &code),
                               code, "prefetch_models");
}

// ── PHASE-8 top-level: global config knobs ───────────────────────────────────

/// Set the global max content-stream ops limit; returns the previous value.
inline std::int64_t set_max_ops_per_stream(std::int64_t limit) {
    return pdf_oxide_set_max_ops_per_stream(limit);
}
/// Toggle preservation of unmapped glyphs (1=on 0=off); returns the C status.
inline int set_preserve_unmapped_glyphs(int preserve) {
    return pdf_oxide_set_preserve_unmapped_glyphs(preserve);
}

} // namespace pdf_oxide

#endif // PDF_OXIDE_HPP
