# frozen_string_literal: true

# Ruby bindings for pdf_oxide — high-performance PDF processing.
#
# Idiomatic 9-class API mirroring the Java binding's shape at
# `fyi.oxide.pdf.*`.  All native calls route through the FFI layer
# at `PdfOxide::FFI::Bindings`; UTF-8 marshalling is via
# `PdfOxide::FFI::StringMarshaller`.
#
# Public surface:
#   - {PdfOxide::PdfDocument}     — read-only entry point.
#   - {PdfOxide::PdfPage}         — per-page view.
#   - {PdfOxide::Pdf}             — create + transform (markdown/html/text → PDF).
#   - {PdfOxide::DocumentEditor}  — write-side: form-fill, redaction, save.
#   - {PdfOxide::AutoExtractor}   — typed-reason auto-extraction (#519).
#   - {PdfOxide::MarkdownConverter} — PDF → Markdown / HTML.
#   - {PdfOxide::PdfValidator}    — PDF/A · PDF/UA compliance.
#   - {PdfOxide::PdfSigner}       — PAdES B/T/LT/LTA signing.
#   - {PdfOxide::PdfPolicy}       — process-global crypto-governance.

require 'ffi'

require_relative 'pdf_oxide/version'
require_relative 'pdf_oxide/errors'
require_relative 'pdf_oxide/ffi/library'
require_relative 'pdf_oxide/ffi/bindings'
require_relative 'pdf_oxide/ffi/string_marshaller'

module PdfOxide
  # Convenience constants reaching into the FFI sub-module.  Keeps
  # downstream callers free of the `PdfOxide::FFI::` prefix when
  # accessing the binding layer; matches the Java binding's flat shape.
  Bindings         = FFI::Bindings
  StringMarshaller = FFI::StringMarshaller
end

require_relative 'pdf_oxide/pdf_page'
require_relative 'pdf_oxide/markdown_converter'
require_relative 'pdf_oxide/auto_extractor'
require_relative 'pdf_oxide/pdf_document'
require_relative 'pdf_oxide/pdf'
require_relative 'pdf_oxide/document_editor'
require_relative 'pdf_oxide/pdf_signer'
require_relative 'pdf_oxide/pdf_validator'
require_relative 'pdf_oxide/pdf_policy'

module PdfOxide
  class << self
    # Open a PDF for reading.
    # @return [PdfDocument]
    def open(source, password: nil, &block)
      PdfDocument.open(source, password: password, &block)
    end

    # @return [String] library version.
    def version
      VERSION
    end
  end
end
