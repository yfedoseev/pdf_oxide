# pdf_oxide — C++ bindings

Idiomatic, header-only C++17 RAII bindings over the pdf_oxide C ABI
(`include/pdf_oxide_c/pdf_oxide.h`). Handles are move-only and freed
automatically; C strings/buffers are copied into `std::string` /
`std::vector<uint8_t>` and freed for you; C-ABI error codes are thrown as
`pdf_oxide::Error`.

## Build

The binding links the **default-feature cdylib** (not the Python wheel). Build it
once from the repo root, then build the C++ targets:

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. configure + build the C++ examples and tests
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DPDF_OXIDE_LIB_DIR="$PWD/target/release"
cmake --build cpp/build -j

# 3. run the tests (includes the api-coverage test)
ctest --test-dir cpp/build --output-on-failure
```

CMake inputs:

| variable | default | meaning |
|---|---|---|
| `PDF_OXIDE_INCLUDE_DIR` | `../include` | dir containing `pdf_oxide_c/pdf_oxide.h` |
| `PDF_OXIDE_LIB_DIR` | `../target-wheel/release` | dir containing `libpdf_oxide.{so,dylib}` |

## Use

```cpp
#include <pdf_oxide/pdf_oxide.hpp>

int main() {
    // Build a PDF from Markdown, then read it back.
    auto pdf  = pdf_oxide::Pdf::from_markdown("# Hello\n\nbody\n");
    auto doc  = pdf_oxide::Document::open_from_bytes(pdf.to_bytes());

    int pages = doc.page_count();
    std::string text = doc.extract_text(0);
    std::string md   = doc.to_markdown_all();
}
```

> Note: the C header declares a global `Pdf` type, so do **not**
> `using namespace pdf_oxide;` — qualify names (`pdf_oxide::Pdf`,
> `pdf_oxide::Document`) or bring them in with targeted `using` declarations.

## Layout

```
cpp/
  include/pdf_oxide/pdf_oxide.hpp   header-only RAII wrapper
  examples/                         runnable examples (asserted in CI)
  tests/                            ctest suite incl. test_api_coverage.cpp
  CMakeLists.txt
```

## Verification (CI — same set as every binding)

`.github/workflows/cpp.yml` on Linux + macOS: build cdylib → CMake build →
`ctest` (unit + **api-coverage**) → run example with an output assertion →
clang-format check.
