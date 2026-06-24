# pdf_oxide — C++ bindings

Idiomatic, header-only C++17 RAII bindings over the pdf_oxide C ABI
(`include/pdf_oxide_c/pdf_oxide.h`). Handles are move-only and freed
automatically; C strings/buffers are copied into `std::string` /
`std::vector<uint8_t>` and freed for you; C-ABI error codes are thrown as
`pdf_oxide::Error`.

## Install

Distributed via vcpkg (port `pdf-oxide`) and Conan. As a header-only library it
exposes the `pdf_oxide_cpp` CMake interface target:

```bash
# vcpkg
vcpkg install pdf-oxide
```

```cmake
# CMakeLists.txt
find_package(pdf_oxide_cpp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE pdf_oxide_cpp)
```

```ini
# Conan — conanfile.txt
[requires]
pdf_oxide_cpp/0.3.69
```

The wrapper links the native `libpdf_oxide` cdylib (built from the Rust source);
to build directly against a local checkout instead, see "Build" below.

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

## Install / consume (CMake `find_package`)

The wrapper ships CMake install/export rules, so after `cmake --install` a
downstream project can locate it with `find_package`:

```cmake
find_package(pdf_oxide_cpp 0.3.69 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE pdf_oxide::pdf_oxide_cpp)
```

`pdf_oxide_cpp` is header-only, so the package contains only headers + the
CMake config. The prebuilt native `libpdf_oxide.{so,dylib,dll}` is **not**
bundled — make it available to your linker (add its dir to the link search
path, or consume it via Conan below).

## Conan / vcpkg

A Conan 2.x recipe (`cpp/conanfile.py`) packages the header-only wrapper:

```bash
# from the cpp/ directory
conan create . --build=missing
```

Consume it from a downstream `conanfile.txt`:

```ini
[requires]
pdf_oxide_cpp/0.3.69

[generators]
CMakeDeps
CMakeToolchain
```

then in your `CMakeLists.txt`:

```cmake
find_package(pdf_oxide_cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE pdf_oxide::pdf_oxide_cpp)
```

The recipe declares `pdf_oxide` as a system lib (you get `-lpdf_oxide`), but it
does **not** build or ship the native library — provide the prebuilt
`libpdf_oxide` on the linker search path.

> The **vcpkg** port lives upstream in `microsoft/vcpkg` and is submitted /
> updated manually; it is not maintained in this repo.

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
