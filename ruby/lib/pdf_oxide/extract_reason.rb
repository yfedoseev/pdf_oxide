# frozen_string_literal: true

module PdfOxide
  # Typed reason explaining why an auto-extraction or classification
  # is in a particular state (v0.3.51 #519, "tell me why").
  #
  # Mirrors {PdfOxide::Enums::ExtractReason} in the PHP binding and
  # `fyi.oxide.pdf.auto.ExtractReason` in Java.  The string values are
  # the canonical snake_case tokens emitted by the Rust serde
  # `ReasonCode` enum at the FFI JSON boundary; renaming them would
  # break cross-binding parity.
  #
  # Anything other than {OK} / {NATIVE_TEXT_HIGH_CONFIDENCE} indicates
  # a degraded result whose cause is named by the constant.
  module ExtractReason
    OK                              = :ok
    NATIVE_TEXT_HIGH_CONFIDENCE     = :native_text_high_confidence
    NO_TEXT_LAYER_PRESENT           = :no_text_layer_present
    TEXT_LAYER_BELOW_THRESHOLD      = :text_layer_below_threshold
    GLYPH_MAPPING_MISSING           = :glyph_mapping_missing
    ENCRYPTED_NO_EXTRACT_PERMISSION = :encrypted_no_extract_permission
    IMAGE_TABLE_RECONSTRUCTED       = :image_table_reconstructed
    IMAGE_TABLE_NO_STRUCTURE        = :image_table_no_structure
    CHART_NOT_TRANSCRIBED           = :chart_not_transcribed
    OCR_REQUESTED_BUT_UNAVAILABLE   = :ocr_requested_but_unavailable
    OCR_LOW_CONFIDENCE_FALLBACK     = :ocr_low_confidence_fallback
    EMPTY                           = :empty

    ALL = [
      OK,
      NATIVE_TEXT_HIGH_CONFIDENCE,
      NO_TEXT_LAYER_PRESENT,
      TEXT_LAYER_BELOW_THRESHOLD,
      GLYPH_MAPPING_MISSING,
      ENCRYPTED_NO_EXTRACT_PERMISSION,
      IMAGE_TABLE_RECONSTRUCTED,
      IMAGE_TABLE_NO_STRUCTURE,
      CHART_NOT_TRANSCRIBED,
      OCR_REQUESTED_BUT_UNAVAILABLE,
      OCR_LOW_CONFIDENCE_FALLBACK,
      EMPTY
    ].freeze

    # Parse a Rust snake_case wire token into a reason Symbol.
    # Tolerant of unknown values — returns {OK} since the Rust
    # ReasonCode enum is `#[non_exhaustive]` and may grow new variants
    # in a future minor.
    def self.from_wire(wire)
      return OK if wire.nil?

      sym = wire.is_a?(Symbol) ? wire : wire.to_s.to_sym
      ALL.include?(sym) ? sym : OK
    end

    # @return [Boolean] whether the reason represents a clean extraction.
    def self.ok?(reason)
      [OK, NATIVE_TEXT_HIGH_CONFIDENCE].include?(reason)
    end

    # @return [Boolean] whether the OCR-unavailable graceful-fallback path
    #   was engaged (extraction is NOT a security op per
    #   `feedback_extraction_graceful_fallback`; the binding returns the
    #   native text layer and surfaces the reason instead of throwing).
    def self.ocr_fallback?(reason)
      [OCR_REQUESTED_BUT_UNAVAILABLE, OCR_LOW_CONFIDENCE_FALLBACK].include?(reason)
    end
  end

  # Auto-classifier's per-page kind, mirroring Rust's
  # `pdf_oxide::extractors::auto::PageKind`.  Wire tokens are the
  # snake_case strings emitted by serde at the FFI JSON boundary.
  module PageKind
    TEXT_LAYER = :text_layer
    SCANNED    = :scanned
    IMAGE_TEXT = :image_text
    MIXED      = :mixed
    EMPTY      = :empty

    ALL = [TEXT_LAYER, SCANNED, IMAGE_TEXT, MIXED, EMPTY].freeze

    def self.from_wire(wire)
      return MIXED if wire.nil?

      sym = wire.is_a?(Symbol) ? wire : wire.to_s.to_sym
      ALL.include?(sym) ? sym : MIXED
    end
  end
end
