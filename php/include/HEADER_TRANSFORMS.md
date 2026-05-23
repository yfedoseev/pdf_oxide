# HEADER_TRANSFORMS

How `php/include/pdf_oxide.h` is derived from the canonical
`include/pdf_oxide_c/pdf_oxide.h` (cbindgen-generated) for
`FFI::cdef()` compatibility.

## Source vs target

- **Source of truth**: `include/pdf_oxide_c/pdf_oxide.h`
  (cbindgen-generated, 3937 lines, 418 `pub extern "C"`
  symbols at v0.3.55).
- **PHP-FFI bundled**: `php/include/pdf_oxide.h`
  (preprocessed for PHP's restrictive C parser).

## PHP FFI C-parser limitations

PHP FFI's parser is **not** a full C99 parser. It cannot
handle:

1. `extern "C" { ... }` blocks (a C++ construct).
2. Any preprocessor directive (`#if`, `#ifdef`, `#define`,
   `#include`, `#endif`, include guards).
3. Block comments and line comments are tolerated but
   stripping reduces parse cost.
4. `_Generic`, `_Static_assert`, and other modern C extras.
5. Compound literals.

PHP FFI **does** handle:

- `typedef`, `struct`, `enum`, `union` declarations
- function-pointer typedefs
- bare top-level function declarations (`ret_t name(args);`)
- `const`, `volatile` qualifiers
- `[N]` fixed-size arrays
- pointers including pointer-to-pointer

## Transforms applied

The preprocessing script (`php/scripts/preprocess_header.py`; re-run any time
the source header changes) performs these passes:

| # | Transform | Why |
|---|-----------|-----|
| 1 | Strip block comments `/* … */` | Reduce parse cost; PHP doesn't need doc strings. |
| 2 | Strip line comments `// …` | Same. |
| 3 | Strip ALL lines beginning with `#` (with optional whitespace) | PHP FFI parser rejects every preprocessor directive. This removes `#include`, `#define`, `#ifndef`, `#endif`, `#if !defined(PDF_OXIDE_TARGET_WASM32)`, `#ifdef __cplusplus`, etc. |
| 4 | Strip `extern "C" {` literally | PHP FFI rejects the C++ linkage block. The matching close-brace is removed in a separate brace-balance pass. |
| 5 | Brace-balance pass: drop any top-level orphan `}` that no longer has a matching open | After step 4, the `}` that closed `extern "C" {` is orphaned. |
| 6 | Collapse runs of blank lines to at most one | Cosmetic. |

## Conditional-compile blocks

The canonical header contains `#if
!defined(PDF_OXIDE_TARGET_WASM32) … #endif` guards around
~14 functions (logging, WASM-incompatible APIs). Stripping
all `#` lines effectively **keeps the body unconditionally**
— correct for PHP, which runs on host platforms only (never
on WASM). The few inline-by-default helper macros (e.g.
`DEFAULT_MAX_DECOMPRESSED_BYTES`, `READ_ONLY`, `REQUIRED`)
are also stripped; PHP wrappers using these constants must
duplicate them in PHP-land (`PdfOxide\Enums\FormFieldFlags`,
etc.).

## Function-count check

At v0.3.55:

- `pub extern "C"` in Rust source: **438** symbols.
- Functions in canonical C header: **418** (cbindgen drops
  some `#[cfg]`-gated ones and a few private helpers).
- Functions in the bundled PHP header after transform:
  **418** (no symbols are dropped by our transforms — only
  preprocessor scaffolding is stripped).

## Validation

```bash
php -r '
  $h = file_get_contents("php/include/pdf_oxide.h");
  $lib = realpath("target/release/libpdf_oxide.so");
  $ffi = FFI::cdef($h, $lib);
  echo "OK\n";
'
```

When this prints `OK` and exits 0, every symbol both **parses**
under PHP FFI **and** resolves against the shared object.

## Regenerating

When the Rust FFI surface changes (any new `pub extern "C"`
function in `src/ffi*.rs`):

1. `make c-header` (regenerates
   `include/pdf_oxide_c/pdf_oxide.h` via cbindgen).
2. Re-run `python3 php/scripts/preprocess_header.py`.
3. Validate with the `php -r` snippet above.
4. Update FunctionBindings.php to add wrappers for new
   symbols (if any high-level binding is desired).

If a new construct breaks PHP FFI parsing (unlikely with
cbindgen's conservative output, but possible if someone
adds `_Generic` or similar), extend
`php/scripts/preprocess_header.py` with a new transform
pass and document it here.
