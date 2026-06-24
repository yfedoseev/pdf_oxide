# pdf_oxide — Objective-C bindings

Idiomatic Objective-C bindings over the pdf_oxide C ABI (the cbindgen header is
directly C-callable from ObjC). `NSObject` wrappers (`POXDocument`, `POXPdf`)
own the C handles and free them in `-dealloc` (ARC); returned C strings/buffers
are copied into `NSString`/`NSData` and freed via `free_string`; non-success
C-ABI error codes surface as `NSError` (`POXErrorDomain`).

## Install (CocoaPods, macOS)

CocoaPods [Trunk goes read-only on 2026-12-02](https://blog.cocoapods.org/CocoaPods-Specs-Repo/),
so this pod is **not** distributed through the central Trunk index. It is a
binary pod published as a GitHub release asset — reference its podspec directly
from your `Podfile` (no Trunk account required, works indefinitely):

```ruby
# Podfile
pod 'PdfOxide', :podspec =>
  'https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.69/PdfOxide.podspec'
```

The pod vendors a prebuilt `PdfOxide.xcframework` (the Rust native static lib),
so consumers do not build Rust. See `PUBLISHING.md` for how the release assembles
and uploads the asset.

## Build & test (macOS)

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. build + run (clang, ARC)
cd objc
make build PDF_OXIDE_LIB_DIR="$PWD/../target/release"
DYLD_LIBRARY_PATH="$PWD/../target/release" ./test_api_coverage
DYLD_LIBRARY_PATH="$PWD/../target/release" ./basic_extraction
```

## Use

```objc
#import "POXPdfOxide.h"

NSError *err = nil;
POXPdf *pdf = [POXPdf fromMarkdown:@"# Hello\n\nbody\n" error:&err];
POXDocument *doc = [POXDocument openData:[pdf toBytesWithError:&err] error:&err];

NSInteger pages = [doc pageCountError:&err];
NSString *text = [doc extractText:0 error:&err];
NSString *md   = [doc toMarkdownAllError:&err];
```

## Layout

```
objc/
  include/POXPdfOxide.h    public interface (POXDocument, POXPdf)
  src/POXPdfOxide.m        implementation over the C ABI
  examples/basic_extraction.m  runnable example (asserted in CI)
  tests/test_api_coverage.m    one check per method (exit-code test)
  Makefile
```

## Verification (CI — same set as every binding)

`.github/workflows/objc.yml` on macOS: build cdylib → `make build` (clang/ARC) →
run `test_api_coverage` (api-coverage) → run example with an output assertion.
