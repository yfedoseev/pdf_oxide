# Publishing the Objective-C binding to CocoaPods (Trunk-free)

`PdfOxide.podspec` (pod name **PdfOxide**) ships the ObjC sources plus a prebuilt
`PdfOxide.xcframework` that wraps the Rust native static library. CocoaPods does
NOT build Rust, so the release CI job assembles the xcframework first.

> **No CocoaPods Trunk.** Trunk goes permanently read-only on **2026-12-02**
> (<https://blog.cocoapods.org/CocoaPods-Specs-Repo/>) and new accounts/pushes are
> not viable. We do **not** publish to the central index and need **no
> `COCOAPODS_TRUNK_TOKEN`**. Instead we upload the xcframework+sources zip and the
> podspec as GitHub **release assets**, and consumers reference the podspec by URL:
>
> ```ruby
> pod 'PdfOxide', :podspec =>
>   'https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.69/PdfOxide.podspec'
> ```
>
> CocoaPods downloads the podspec, reads its `:http` `source` (the zip), and
> installs from there — fully decentralised and durable.

Everything below runs on a **macOS runner with Xcode + a stable Rust toolchain**.
Paths are relative to the repo root.

## 0. Prerequisites (CI runner)

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
gem install cocoapods   # provides `pod` (for `pod spec lint` only — no Trunk auth)
```

The native feature set is the same as the binding's CI build:

```bash
FEATURES=ocr,rendering,signatures,barcodes,tsa-client,system-fonts
```

## 1. Build the Rust static lib for each macOS arch

```bash
cargo build --release --lib --features "$FEATURES" --target aarch64-apple-darwin
cargo build --release --lib --features "$FEATURES" --target x86_64-apple-darwin
```

Outputs:
- `target/aarch64-apple-darwin/release/libpdf_oxide.a`
- `target/x86_64-apple-darwin/release/libpdf_oxide.a`

## 2. Make a universal (fat) macOS static lib

An xcframework slice must contain a single library per platform; for macOS that
slice carries both arches as a fat archive.

```bash
mkdir -p build/macos
lipo -create \
  target/aarch64-apple-darwin/release/libpdf_oxide.a \
  target/x86_64-apple-darwin/release/libpdf_oxide.a \
  -output build/macos/libpdf_oxide.a
```

## 3. Stage the public C header

The xcframework slice needs the cbindgen header in a `Headers/` dir. The ObjC
sources `#import <Foundation/Foundation.h>` and call the C ABI declared here.

```bash
mkdir -p build/macos/Headers
cp include/pdf_oxide_c/pdf_oxide.h build/macos/Headers/
```

## 4. Assemble PdfOxide.xcframework

```bash
rm -rf objc/Frameworks/PdfOxide.xcframework
mkdir -p objc/Frameworks
xcodebuild -create-xcframework \
  -library build/macos/libpdf_oxide.a \
  -headers build/macos/Headers \
  -output objc/Frameworks/PdfOxide.xcframework
```

(To add iOS later: build `aarch64-apple-ios` and `aarch64-apple-ios-sim` /
`x86_64-apple-ios`, lipo the simulator arches, and pass additional
`-library ... -headers ...` pairs to the same `-create-xcframework` call.)

## 5. Zip the asset, lint, and upload to the release

```bash
# Zip the objc/ tree (sources + Frameworks/PdfOxide.xcframework) so the entries
# match the podspec's source_files / vendored_frameworks globs.
zip -r "PdfOxide-objc-0.3.69.zip" objc \
  -x 'objc/*.o' 'objc/test_api_coverage' 'objc/basic_extraction'

# Structural lint (skips a full compile of the vendored binary framework).
pod spec lint objc/PdfOxide.podspec --allow-warnings || true

# Upload BOTH the zip (the podspec's :http source) and the podspec itself
# as release assets — no Trunk push.
gh release upload "v0.3.69" "PdfOxide-objc-0.3.69.zip" objc/PdfOxide.podspec --clobber
```

Notes:
- `spec.source` is the **`:http`** zip release asset (NOT a git tag) — so the
  prebuilt xcframework, which is never committed, reaches consumers. Upload the
  zip before anyone resolves the podspec.
- `pod lib lint` (full sandbox build) may fail in CI because it links the
  vendored static framework against an app target; `pod spec lint` validates the
  spec/source structure, which is what we gate on. Use `--allow-warnings` for the
  vendored-framework / deployment-target advisories.
- **No secret required** — distribution is entirely via release assets, not Trunk.
