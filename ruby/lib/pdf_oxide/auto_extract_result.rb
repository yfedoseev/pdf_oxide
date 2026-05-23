# frozen_string_literal: true

module PdfOxide
  # Result of an {AutoExtractor} call.  Carries the extracted text
  # plus the typed {ExtractReason} explaining the quality / fallback
  # state, plus an optional decoded JSON envelope from the FFI
  # boundary.  Frozen value object — field names match Python's
  # `AutoExtractResult` dataclass and PHP's `AutoExtractResult` for
  # cross-binding documentation parity.
  class AutoExtractResult
    attr_reader :text, :reason, :kind, :confidence, :classification

    def initialize(text:, reason:, kind:, confidence: 0.0, classification: nil)
      @text           = text.to_s
      @reason         = reason
      @kind           = kind
      @confidence     = confidence.to_f
      @classification = classification
      freeze
    end

    # Whether the extraction was clean.  Mirrors PHP's
    # `AutoExtractResult::isOk()`.
    def ok?
      ExtractReason.ok?(@reason)
    end

    # Whether the OCR-unavailable graceful-fallback path engaged.
    # The text is still recoverable from {#text}; the reason names
    # why the result is degraded.
    def ocr_fallback?
      ExtractReason.ocr_fallback?(@reason)
    end

    def to_h
      {
        text:           @text,
        reason:         @reason,
        kind:           @kind,
        confidence:     @confidence,
        classification: @classification
      }
    end
  end
end
