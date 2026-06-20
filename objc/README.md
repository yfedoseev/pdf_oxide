# pdf_oxide — Objective-C bindings

Idiomatic Objective-C bindings over the pdf_oxide C ABI (the cbindgen header is
directly C-callable from ObjC). `NSObject` wrappers (`POXDocument`, `POXPdf`)
own the C handles and free them in `-dealloc` (ARC); returned C strings/buffers
are copied into `NSString`/`NSData` and freed via `free_string`; non-success
C-ABI error codes surface as `NSError` (`POXErrorDomain`).

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
POXDocument *doc = [POXDocument openData:[pdf saveToBytesError:&err] error:&err];

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
