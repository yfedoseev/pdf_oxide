# frozen_string_literal: true

require 'json'

module PdfOxide
  # Models subsystem (v0.3.51 #519 provisioning trio).  Wraps the
  # build-time OCR model fetch / manifest / availability triplet.
  # Mirrors Python's `pdf_oxide.Models`, Java's `Models` namespace,
  # and PHP's `AutoExtractor` provisioning methods.
  module Models
    module_function

    # Provision OCR models for the given languages (e.g. ['eng', 'rus']).
    # Returns the cache directory path on success.  When the build is
    # compiled without the `ocr` feature, the cache dir is still
    # created but no fetch happens — query {available?} to tell.
    #
    # Per `feedback_extraction_graceful_fallback`: this NEVER raises a
    # bare "OCR unavailable" error.  An empty/nil return is the
    # legitimate "no-fetch" signal.
    #
    # @param languages [Array<String>, String] one or more BCP-47 / ISO
    #   language tags.  CSV-joined for the C call.
    # @return [String] cache directory path (may be empty for no-op builds).
    def prefetch(languages)
      csv = Array(languages).join(',')

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      ptr = FFI::Bindings.pdf_oxide_prefetch_models(csv, error_ptr)
      error_code = error_ptr.read_int32

      if error_code != 0
        raise FFI::ErrorHandler.create_error(
          error_code, 'pdf_oxide_prefetch_models', languages: csv
        )
      end

      return '' if ptr.nil? || ptr.null?

      FFI::StringMarshaller.from_c_string(ptr) || ''
    end

    # Returns the model manifest as a decoded Hash, or {} when the
    # build lacks OCR support.  No error path — empty means
    # unsupported.
    def manifest
      ptr = FFI::Bindings.pdf_oxide_model_manifest
      return {} if ptr.nil? || ptr.null?

      str = FFI::StringMarshaller.from_c_string(ptr) || ''
      return {} if str.empty?

      JSON.parse(str)
    rescue JSON::ParserError
      {}
    end

    # @return [Boolean] whether the build can fetch OCR models.
    #   `false` ⇒ {prefetch} still creates the cache dir (so callers
    #   can stage offline models) but never reaches out to the network.
    def available?
      FFI::Bindings.pdf_oxide_prefetch_available != 0
    end
  end
end
