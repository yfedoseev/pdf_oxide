// 03_words_geometry — build a PDF from Markdown, open it, extract word geometry.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "WORDS OK" on success.
#include <pdf_oxide/pdf_oxide.hpp>

#include <cstdio>
#include <iostream>

int main() {
    try {
        auto pdf = pdf_oxide::Pdf::from_markdown(
            "# Hello pdf_oxide\n\nThis is a **C++** regression example.\n");
        auto doc = pdf_oxide::Document::open_from_bytes(pdf.to_bytes());

        std::vector<pdf_oxide::Word> words = doc.extract_words(0);
        std::cout << "word count: " << words.size() << "\n";

        if (words.empty()) {
            std::cerr << "assertion failed: no words extracted\n";
            return 1;
        }

        const pdf_oxide::Word& first = words[0];
        std::cout << "first word: \"" << first.text << "\"  bbox=(" << first.bbox.x
                  << ", " << first.bbox.y << ", " << first.bbox.width << ", "
                  << first.bbox.height << ")\n";

        if (first.text != "Hello") {
            std::cerr << "assertion failed: first word is not \"Hello\"\n";
            return 1;
        }
        if (!(first.bbox.width > 0.0f && first.bbox.height > 0.0f)) {
            std::cerr << "assertion failed: first word has no bbox\n";
            return 1;
        }

        std::cout << "WORDS OK\n";
        return 0;
    } catch (const pdf_oxide::Error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
