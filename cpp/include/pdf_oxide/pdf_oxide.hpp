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
    // Byte buffers from the core are freed via free_string (same allocator).
    free_string(reinterpret_cast<char*>(p));
    return out;
}

} // namespace detail

/// PDF version (e.g. 1.7).
struct Version {
    std::uint8_t major;
    std::uint8_t minor;
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
        PdfDocument* h =
            pdf_document_open_from_bytes(data.data(), data.size(), &code);
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
        int32_t n = pdf_document_get_page_count(handle_.get(), &code);
        if (n < 0) {
            throw Error(code, "Document::page_count");
        }
        return n;
    }

    /// PDF version.
    Version version() const {
        Version v{0, 0};
        pdf_document_get_version(handle_.get(), &v.major, &v.minor);
        return v;
    }

    /// True if the document is encrypted.
    bool is_encrypted() const {
        return pdf_document_is_encrypted(handle_.get());
    }

    /// True if the document carries a logical structure tree (tagged PDF).
    bool has_structure_tree() const {
        return pdf_document_has_structure_tree(handle_.get());
    }

    /// Extract reading-order text for one page (0-based).
    std::string extract_text(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_text(handle_.get(), page_index, &code), code,
            "Document::extract_text");
    }

    /// Plain text for one page.
    std::string to_plain_text(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_to_plain_text(handle_.get(), page_index, &code), code,
            "Document::to_plain_text");
    }

    /// Markdown for one page.
    std::string to_markdown(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_to_markdown(handle_.get(), page_index, &code), code,
            "Document::to_markdown");
    }

    /// HTML for one page.
    std::string to_html(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_to_html(handle_.get(), page_index, &code), code,
            "Document::to_html");
    }

    /// Markdown for the whole document.
    std::string to_markdown_all() const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_to_markdown_all(handle_.get(), &code), code,
            "Document::to_markdown_all");
    }

    /// Structured content as a JSON string.
    std::string extract_structured_json(int page_index) const {
        int32_t code = 0;
        return detail::take_string(
            pdf_document_extract_structured_to_json(handle_.get(), page_index,
                                                    &code),
            code, "Document::extract_structured_json");
    }

private:
    struct Deleter {
        void operator()(PdfDocument* h) const noexcept {
            if (h) pdf_document_free(h);
        }
    };
    explicit Document(PdfDocument* h) : handle_(h) {}
    std::unique_ptr<PdfDocument, Deleter> handle_;
};

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
        if (pdf_save(handle_.get(), path.c_str(), &code) != 0) {
            throw Error(code, "Pdf::save");
        }
    }

    /// Serialize the PDF to bytes.
    std::vector<std::uint8_t> save_to_bytes() const {
        int32_t code = 0;
        int32_t len = 0;
        std::uint8_t* p = pdf_save_to_bytes(handle_.get(), &len, &code);
        return detail::take_bytes(p, static_cast<std::size_t>(len < 0 ? 0 : len),
                                  code, "Pdf::save_to_bytes");
    }

private:
    struct Deleter {
        void operator()(::Pdf* h) const noexcept {
            if (h) pdf_free(h);
        }
    };
    explicit Pdf(::Pdf* h) : handle_(h) {}
    std::unique_ptr<::Pdf, Deleter> handle_;
};

} // namespace pdf_oxide

#endif // PDF_OXIDE_HPP
