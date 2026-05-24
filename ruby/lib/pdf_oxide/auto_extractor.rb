# frozen_string_literal: true

require 'json'

module PdfOxide
  # v0.3.51 #519 — auto-extraction with typed reasons.
  #
  # Given a {PdfOxide::Document}, returns recoverable text per page (or
  # whole document) with a typed {ExtractReason} naming any degraded
  # outcome.  When OCR is needed but the build cannot provide it,
  # returns the native text layer with
  # {ExtractReason::OCR_REQUESTED_BUT_UNAVAILABLE} instead of raising —
  # extraction is **not** a security op (per
  # `feedback_extraction_graceful_fallback`).
  #
  # API surface mirrors:
  # - PHP `\PdfOxide\AutoExtractor`,
  # - Python `pdf_oxide.AutoExtractor`,
  # - Java  `fyi.oxide.pdf.AutoExtractor`.
  #
  # @example
  #   doc = PdfOxide::Document.open('sample.pdf')
  #   ax  = PdfOxide::AutoExtractor.new(doc)
  #   result = ax.extract_text(0)
  #   puts result.text
  #   warn "degraded: #{result.reason}" unless result.ok?
  class AutoExtractor
    attr_reader :document

    # @param document [PdfOxide::Document] open PDF document.
    def initialize(document)
      raise ::PdfOxide::ArgumentError, 'document cannot be nil' if document.nil?
      raise ::PdfOxide::StateError, 'document has been closed' if document.closed?

      @document = document
    end

    # Cheap per-page classification (no OCR, no rasterisation).
    # Returns an {AutoExtractResult} populated from the JSON envelope.
    def classify_page(page_index)
      json = call_json('classify_page', page_index) do |err|
        FFI::Bindings.pdf_document_classify_page(@document.handle, page_index, err)
      end
      build_result_from_classification(json, text: '')
    end

    # Whole-document classification — per-page kinds + a
    # `pages_needing_ocr` array.  Returned as a Hash (decoded JSON).
    def classify_document
      json = call_json('classify_document') do |err|
        FFI::Bindings.pdf_document_classify_document(@document.handle, err)
      end
      json.is_a?(Hash) ? json : {}
    end

    # One-shot text-only auto extraction (text-vs-OCR routing with
    # graceful native fallback).  Returns an {AutoExtractResult}; the
    # page-level reason is derived from a follow-up classification
    # call (cheap; no second OCR pass) — same approach as PHP/Python.
    def extract_text(page_index)
      text = call_text('extract_text_auto', page_index) do |err|
        FFI::Bindings.pdf_document_extract_text_auto(@document.handle, page_index, err)
      end

      reason         = ExtractReason::OK
      kind           = PageKind::MIXED
      classification = nil
      confidence     = 0.0
      begin
        cls = classify_page(page_index)
        reason         = cls.reason
        kind           = cls.kind
        confidence     = cls.confidence
        classification = cls.classification
      rescue StandardError
        # Classification is best-effort; never let it mask extraction.
      end

      # Graceful-fallback hook: if the classifier wants OCR and the
      # build can't provision models, surface that as the reason
      # regardless of whether the native side already downgraded.
      if kind == PageKind::SCANNED && !self.class.prefetch_available?
        reason = ExtractReason::OCR_REQUESTED_BUT_UNAVAILABLE
      end

      AutoExtractResult.new(
        text: text,
        reason: reason,
        kind: kind,
        confidence: confidence,
        classification: classification
      )
    end

    # Rich per-page extraction — returns the full JSON `PageExtraction`
    # envelope (text + per-region bbox + reason + confidence + ocr_used)
    # wrapped in {AutoExtractResult}.
    #
    # @param page_index [Integer]
    # @param options    [Hash, nil] auto-extract options serialized to JSON
    #                                and passed through to the Rust side.
    def extract_page(page_index, options: nil)
      options_json = options.nil? ? nil : JSON.generate(options)
      json = call_json('extract_page_auto', page_index) do |err|
        FFI::Bindings.pdf_document_extract_page_auto(
          @document.handle, page_index, options_json, err
        )
      end

      text = json.is_a?(Hash) ? (json['text'] || '') : ''
      build_result_from_classification(json, text: text)
    end

    # ----------------------------------------------------------------
    # Models subsystem (v0.3.51 #519 provisioning trio)
    # ----------------------------------------------------------------

    # Whether the build supports OCR provisioning (i.e. the `ocr`
    # feature is compiled in AND a model cache appears reachable).
    # Used by AutoExtractor's graceful-fallback decision.
    def self.prefetch_available?
      FFI::Bindings.pdf_oxide_prefetch_available != 0
    end

    private

    # Wrapper for a C call that returns a malloc'd char* JSON blob.
    # Returns a parsed Hash (or [] if JSON is an array, or {} on parse
    # failure).
    def call_json(operation, *args, &block)
      json_str = call_text(operation, *args, &block)
      return {} if json_str.nil? || json_str.empty?

      JSON.parse(json_str)
    rescue JSON::ParserError
      {}
    end

    # Wrapper for a C call that returns a malloc'd char* string.
    # Frees the returned pointer via the standard StringMarshaller.
    def call_text(operation, *_)
      raise ::PdfOxide::ArgumentError, 'block required' unless block_given?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      ptr       = yield(error_ptr)
      error_code = error_ptr.read_int32

      if error_code != 0
        raise FFI::ErrorHandler.create_error(error_code, operation)
      end
      return '' if ptr.nil? || ptr.null?

      FFI::StringMarshaller.from_c_string(ptr)
    end

    def build_result_from_classification(json, text:)
      json = {} unless json.is_a?(Hash)
      AutoExtractResult.new(
        text: text,
        reason: ExtractReason.from_wire(json['reason']),
        kind: PageKind.from_wire(json['kind']),
        confidence: (json['confidence'] || 0.0).to_f,
        classification: json
      )
    end
  end
end
