// 01_basic_extraction — build a PDF from Markdown, then extract it back.
//
// Run in CI as a smoke example (no external PDF fixture needed).
#include <pdf_oxide/pdf_oxide.hpp>

#include <cstdio>
#include <iostream>

int main() {
    try {
        // Build a small PDF from Markdown.
        auto pdf = pdf_oxide::Pdf::from_markdown(
            "# Hello pdf_oxide\n\nThis is a **C++** binding smoke example.\n");

        // Serialize and re-open it.
        std::vector<std::uint8_t> bytes = pdf.to_bytes();
        auto doc = pdf_oxide::Document::open_from_bytes(bytes);

        std::cout << "pages:   " << doc.page_count() << "\n";
        auto v = doc.version();
        std::cout << "version: " << static_cast<int>(v.major) << "."
                  << static_cast<int>(v.minor) << "\n";
        std::cout << "--- text (page 0) ---\n" << doc.extract_text(0) << "\n";
        std::cout << "--- markdown (all) ---\n" << doc.to_markdown_all() << "\n";
        return 0;
    } catch (const pdf_oxide::Error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
