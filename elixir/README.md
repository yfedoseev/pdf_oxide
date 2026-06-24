# pdf_oxide — Elixir bindings

Idiomatic Elixir bindings for pdf_oxide via a NIF over the C ABI. CPU-bound
extraction runs on **dirty CPU schedulers** (`ERL_NIF_DIRTY_JOB_CPU_BOUND`) so
it never blocks the BEAM scheduler. Document/Pdf handles are NIF resources freed
by the GC; functions return `{:ok, value}` / `{:error, code}`. Page indices are
0-based.

## Install

Published to [Hex.pm](https://hex.pm/) as `pdf_oxide`:

```elixir
# mix.exs
def deps do
  [
    {:pdf_oxide, "~> 0.3.69"}
  ]
end
```

The NIF is built from source on install via `elixir_make`, linking the
default-feature `libpdf_oxide` cdylib — see "Build & test" for building it.

## Build & test

The NIF links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. compile the NIF + test (elixir_make builds c_src/pdf_oxide_nif.c)
cd elixir
mix deps.get
mix compile
LD_LIBRARY_PATH="$PWD/../target/release" mix test
LD_LIBRARY_PATH="$PWD/../target/release" mix run examples/basic_extraction.exs
```

## Use

```elixir
{:ok, pdf}  = PdfOxide.from_markdown("# Hello\n\nbody\n")
{:ok, data} = PdfOxide.to_bytes(pdf)
{:ok, doc}  = PdfOxide.open_bytes(data)

{:ok, n}    = PdfOxide.page_count(doc)
{:ok, text} = PdfOxide.extract_text(doc, 0)
{:ok, md}   = PdfOxide.to_markdown_all(doc)
```

## Layout

```
elixir/
  lib/pdf_oxide.ex            idiomatic API (PdfOxide, Document, Pdf, Error)
  lib/pdf_oxide/native.ex     NIF loader
  c_src/pdf_oxide_nif.c       dirty-CPU NIF over the C ABI
  Makefile                    builds priv/pdf_oxide_nif.so (via elixir_make)
  examples/basic_extraction.exs  runnable example (asserted in CI)
  test/pdf_oxide_test.exs     ExUnit api-coverage (one test per function)
  mix.exs
```

## Verification (CI — same set as every binding)

`.github/workflows/elixir.yml` on Linux + macOS: build cdylib → OTP/Elixir →
`mix compile` (builds the NIF) → `mix test` (api-coverage) → run example with an
output assertion.
