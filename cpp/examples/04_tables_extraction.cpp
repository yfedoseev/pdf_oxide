// 04_tables_extraction — build a PDF from Markdown with a table, extract tables.
//
// Shared regression scenario (mirrored across language bindings). The synthetic
// doc may yield zero tables; the contract is only that the call succeeds and
// returns a list (count >= 0). Exits non-zero on error; prints "TABLES OK".
#include <pdf_oxide/pdf_oxide.hpp>

#include <cstdio>
#include <iostream>

int main() {
    try {
        auto pdf = pdf_oxide::Pdf::from_markdown(
            "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta "
            "| 2 |\n");
        auto doc = pdf_oxide::Document::open_from_bytes(pdf.to_bytes());

        std::vector<pdf_oxide::Table> tables = doc.extract_tables(0);
        std::cout << "table count: " << tables.size() << "\n";

        for (std::size_t t = 0; t < tables.size(); ++t) {
            const pdf_oxide::Table& tbl = tables[t];
            std::cout << "table " << t << ": " << tbl.row_count << "x" << tbl.col_count
                      << "\n";
            for (int r = 0; r < tbl.row_count; ++r) {
                for (int c = 0; c < tbl.col_count; ++c) {
                    std::cout << "  cell(" << r << "," << c << ")=\"" << tbl.cell(r, c)
                              << "\"\n";
                }
            }
        }

        // extract_tables threw nothing → the call returned a valid list.
        std::cout << "TABLES OK\n";
        return 0;
    } catch (const pdf_oxide::Error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
