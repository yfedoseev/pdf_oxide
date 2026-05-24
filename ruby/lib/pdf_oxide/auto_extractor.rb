# frozen_string_literal: true

require 'json'

module PdfOxide
  # v0.3.51 #519 — auto-extraction with typed reasons.
  #
  # Mirrors `fyi.oxide.pdf.AutoExtractor`.  Given a {PdfDocument},
  # returns recoverable text (native or OCR), per-page or
  # whole-document, with a typed reason naming any degraded outcome.
  # When OCR is needed but unavailable, returns the native text layer
  # with `:ocr_requested_but_unavailable` instead of raising —
  # extraction is **not** a security operation (per
  # `feedback_extraction_graceful_fallback`).
  #
  # @example
  #   doc = PdfOxide::PdfDocument.open('sample.pdf')
  #   ax  = PdfOxide::AutoExtractor.new(doc)
  #   result = ax.extract_page(0)
  #   puts result[:text]
  #   warn "degraded: #{result[:reason]}" unless ax.ok?(result[:reason])
  class AutoExtractor
    # Typed reasons mirror the Rust serde-emitted snake_case tokens
    # at the FFI JSON boundary.  Renaming would break cross-binding
    # parity with PHP / Python / Java.
    REASONS = %i[
      ok
      native_text_high_confidence
      no_text_layer_present
      text_layer_below_threshold
      glyph_mapping_missing
      encrypted_no_extract_permission
      image_table_reconstructed
      image_table_no_structure
      chart_not_transcribed
      ocr_requested_but_unavailable
      ocr_low_confidence_fallback
      empty
    ].freeze

    # Per-page kinds from the auto-classifier (Rust's `PageKind` enum).
    PAGE_KINDS = %i[text_layer scanned image_text mixed empty].freeze

    # @return [PdfDocument]
    attr_reader :document

    def initialize(document)
      raise ::PdfOxide::ArgumentError, 'document cannot be nil' if document.nil?
      raise ::PdfOxide::StateError, 'document has been closed' if document.respond_to?(:closed?) && document.closed?

      @document = document
    end

    # Cheap per-page classifier — no OCR, no rasterisation.
    # @return [Hash] { reason:, kind:, confidence:, classification: }
    def classify_page(page_index)
      json = call_json('classify_page') do |err|
        Bindings.pdf_document_classify_page(@document.handle, page_index, err)
      end
      build_classification(json)
    end

    # Whole-document classifier.
    # @return [Hash] decoded JSON envelope.
    def classify_document
      call_json('classify_document') do |err|
        Bindings.pdf_document_classify_document(@document.handle, err)
      end
    end

    # Extract a page's text via the v0.3.51 auto-router (text-vs-OCR
    # decision with graceful native fallback).  Surfaces a typed
    # reason describing the quality.
    # @return [Hash] { text:, reason:, kind:, confidence:, classification: }
    def extract_text(page_index)
      text = call_text('extract_text_auto') do |err|
        Bindings.pdf_document_extract_text_auto(@document.handle, page_index, err)
      end
      cls = begin
        classify_page(page_index)
      rescue StandardError
        { reason: :ok, kind: :mixed, confidence: 0.0 }
      end
      # Graceful fallback: if classifier wants OCR and the build can't
      # supply it, surface OCR_REQUESTED_BUT_UNAVAILABLE regardless of
      # native-side state.
      cls[:reason] = :ocr_requested_but_unavailable if cls[:kind] == :scanned && !self.class.prefetch_available?
      cls.merge(text: text)
    end

    # Rich per-page extraction — returns the full PageExtraction
    # JSON envelope (text + per-region bbox + reason + confidence)
    # merged into a Hash.
    # @param page_index [Integer]
    # @param options [Hash, nil] auto-extract options serialised to JSON.
    def extract_page(page_index, options: nil)
      options_json = options.nil? ? nil : JSON.generate(options)
      json = call_json('extract_page_auto') do |err|
        Bindings.pdf_document_extract_page_auto(@document.handle, page_index, options_json, err)
      end
      cls = build_classification(json)
      cls.merge(text: json['text'] || '', classification: json)
    end

    # @return [Boolean] true when the reason represents a clean extract.
    def ok?(reason)
      %i[ok native_text_high_confidence].include?(reason)
    end

    # @return [Boolean] true when the OCR-unavailable graceful-fallback
    #   path engaged.
    def ocr_fallback?(reason)
      %i[ocr_requested_but_unavailable ocr_low_confidence_fallback].include?(reason)
    end

    # @return [Boolean] whether the build supports OCR provisioning
    #   (i.e. the `ocr` feature is compiled in).
    def self.prefetch_available?
      Bindings.pdf_oxide_prefetch_available != 0
    end

    private

    def call_json(operation, &block)
      raw = call_text(operation, &block)
      return {} if raw.nil? || raw.empty?

      JSON.parse(raw)
    rescue JSON::ParserError
      {}
    end

    def call_text(operation)
      err = ::FFI::MemoryPointer.new(:int32)
      ptr = yield(err)
      code = err.read_int32
      raise InternalError, "#{operation} failed (#{code})" if code != 0
      return '' if ptr.nil? || ptr.null?

      StringMarshaller.from_c_string(ptr) || ''
    end

    def build_classification(json)
      json = {} unless json.is_a?(Hash)
      reason = (json['reason'] || 'ok').to_sym
      reason = :ok unless REASONS.include?(reason)
      kind = (json['kind'] || 'mixed').to_sym
      kind = :mixed unless PAGE_KINDS.include?(kind)
      {
        reason: reason,
        kind: kind,
        confidence: (json['confidence'] || 0.0).to_f,
        classification: json
      }
    end
  end
end
