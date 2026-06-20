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
                out.push_back(std::move(p));
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

} // namespace pdf_oxide

#endif // PDF_OXIDE_HPP
