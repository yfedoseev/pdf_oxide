// 02_html_extraction — build a PDF from Markdown, open it, render HTML.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "HTML OK" on success.
#include <pdf_oxide/pdf_oxide.hpp>

#include <cstdio>
#include <iostream>

int main() {
    try {
        auto pdf = pdf_oxide::Pdf::from_markdown(
            "# Hello pdf_oxide\n\nThis is a **C++** regression example.\n");
        auto doc = pdf_oxide::Document::open_from_bytes(pdf.to_bytes());

        std::string html = doc.to_html_all();
        std::cout << "--- html (all) ---\n" << html << "\n";

        if (html.find('<') == std::string::npos) {
            std::cerr << "assertion failed: html does not contain '<'\n";
            return 1;
        }
        if (html.find("pdf_oxide") == std::string::npos) {
            std::cerr << "assertion failed: html does not contain 'pdf_oxide'\n";
            return 1;
        }

        std::cout << "HTML OK\n";
        return 0;
    } catch (const pdf_oxide::Error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
