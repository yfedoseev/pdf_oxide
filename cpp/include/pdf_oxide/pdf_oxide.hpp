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

} // namespace pdf_oxide

#endif // PDF_OXIDE_HPP
