# CocoaPods spec for the pdf_oxide Objective-C bindings.
#
# The native library (libpdf_oxide) is a Rust cdylib/staticlib. It is NOT built
# by CocoaPods — the release job (see objc/PUBLISHING.md) compiles it for the
# Apple targets and assembles `objc/Frameworks/PdfOxide.xcframework`, which this
# spec then vendors. The xcframework + sources are uploaded as a GitHub release
# asset zip (the :http source below); this podspec is uploaded alongside it.
# Distribution is Trunk-FREE (CocoaPods Trunk goes read-only 2026-12-02):
# consumers reference this podspec by URL —
#   pod 'PdfOxide', :podspec => '<release>/v#{version}/PdfOxide.podspec'
#
# VERSION: hardcoded to the canonical workspace version below. scripts/sync_version.py
# must be taught to keep this `spec.version` in lock-step with Cargo.toml (no sync
# rule is added here — that wiring is intentionally left to the release tooling).
Pod::Spec.new do |spec|
  spec.name         = 'PdfOxide'
  spec.version      = '0.3.69'
  spec.summary      = 'Idiomatic Objective-C bindings for pdf_oxide, a fast pure-Rust PDF toolkit.'
  spec.description  = <<-DESC
    Objective-C bindings over the pdf_oxide C ABI. NSObject wrappers (POXDocument,
    POXPdf, POXDocumentBuilder, …) own the native handles and free them under ARC;
    C strings/buffers are copied into NSString/NSData; non-success C-ABI error codes
    surface as NSError. Covers text/markdown/HTML extraction, element and table
    extraction, page rendering, OCR, form handling, redaction, barcodes, digital
    signatures (PAdES), timestamps, and PDF/A·UA·X conformance checks.
  DESC
  spec.homepage     = 'https://github.com/yfedoseev/pdf_oxide'
  spec.license      = { :type => 'MIT', :file => 'objc/LICENSE' }
  spec.authors      = { 'pdf_oxide contributors' => 'yfedoseev@gmail.com' }

  # Binary pod: the source is a release-asset zip whose layout mirrors the
  # repo's objc/ dir AND carries the prebuilt objc/Frameworks/PdfOxide.xcframework
  # (the xcframework is a build artifact, never committed to git — a :git source
  # would leave consumers' `pod install` without the native lib). The publish-
  # cocoapods job in release.yml assembles + uploads this asset, then pushes
  # the spec. source_files / vendored_frameworks paths resolve inside the zip.
  spec.source       = { :http => "https://github.com/yfedoseev/pdf_oxide/releases/download/v#{spec.version}/PdfOxide-objc-#{spec.version}.zip" }

  # macOS-only: the binding's feature set (system-fonts, TSA network client,
  # signatures, OCR) is validated on macOS and the CI matrix is macOS-only.
  # iOS support is possible future work once the static lib is built for the
  # iOS device/simulator targets and added as extra slices of the xcframework.
  spec.platform     = :osx, '10.15'
  spec.osx.deployment_target = '10.15'

  spec.source_files        = 'objc/include/**/*.h', 'objc/src/**/*.m'
  spec.public_header_files  = 'objc/include/**/*.h'

  # The native library, shipped as a prebuilt xcframework (assembled by the
  # release job from the Rust static libs — see objc/PUBLISHING.md).
  spec.vendored_frameworks = 'objc/Frameworks/PdfOxide.xcframework'

  spec.frameworks   = 'Foundation'
  spec.requires_arc = true
end
