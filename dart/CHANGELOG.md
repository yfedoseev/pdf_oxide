# Changelog

## 0.3.69

- Initial release of the Dart/Flutter bindings for pdf_oxide over the C ABI via
  `dart:ffi`: PDF text, Markdown and HTML extraction, page rendering, element and
  table extraction, document building, and more. Native handles are freed via
  `NativeFinalizer`; C-ABI errors surface as `PdfOxideError`.
