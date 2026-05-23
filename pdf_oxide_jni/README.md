# pdf_oxide_jni — JNI shim for the Java binding

The native shim that backs the `fyi.oxide:pdf-oxide` Maven Central
artifact. Loaded by `fyi.oxide.pdf.internal.NativeLoader` at JVM
start-up via `System.load(...)` from a temp-extracted resource.

This crate is **not** published to crates.io — the consumable
artifact is the Maven jar, which bundles the compiled `cdylib`
for five native architectures (linux x86_64/aarch64, macOS
x86_64/aarch64, windows x86_64).

## Build

```bash
# Default (text/markdown/auto-extractor signals, no OCR or signatures)
cargo build -p pdf_oxide_jni --release

# Production fat-jar build (all features ON, matches v0.3.52 ocr-enabled
# prebuilts per #520)
cargo build -p pdf_oxide_jni --release --features full
```

The compiled artifact goes to `target/release/libpdf_oxide_jni.so`
(linux) / `libpdf_oxide_jni.dylib` (macOS) / `pdf_oxide_jni.dll`
(windows). The Maven build (`java/pom.xml` via
`questdb/rust-maven-plugin`) copies the per-arch artifact into
`java/src/main/resources/fyi/oxide/pdf/native/{OS}/{ARCH}/`.

## Plan and contracts

The v0.3.53 release plan, including the FFI contract, panic-barrier
invariants, exception taxonomy, native-loader contract, and parity
matrix lives at:

- `docs/releases/plans/v0.3.53/README.md` — index
- `docs/releases/plans/v0.3.53/00-common-foundation.md` — contracts
  (**read first** before touching any module here)
- `docs/releases/plans/v0.3.53/api-design.md` — the public Java
  surface this crate must support
- `docs/releases/plans/v0.3.53/feature-NNN-java-binding.md` —
  implementation tasks T1–T22

## License

MIT OR Apache-2.0 (same as pdf_oxide core).
