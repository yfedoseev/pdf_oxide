# frozen_string_literal: true

module PdfOxide
  # PDF/A · PDF/X · PDF/UA compliance validation (v0.3.50).
  #
  # Mirrors `fyi.oxide.pdf.PdfValidator`.  Stateless façade.
  #
  # @example
  #   PdfOxide::PdfDocument.open('compliant.pdf') do |doc|
  #     puts PdfOxide::PdfValidator.pdf_a?(doc, level: :a1b)
  #   end
  module PdfValidator
    module_function

    # PDF/A level → cdylib wire-format integer.
    #
    # Matches `src/ffi.rs:1225` (`0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b
    # 6=A3a 7=A3u`). Every binding (Java, Ruby, PHP, C#, Go) sends the
    # SAME integer for the same PDF/A level — the "B before A"
    # intra-level order is the cdylib's contract, not a Ruby choice.
    PDF_A_LEVELS = { a1b: 0, a1a: 1, a2b: 2, a2a: 3, a2u: 4, a3b: 5, a3a: 6, a3u: 7 }.freeze

    # PDF/UA level → cdylib wire-format integer.
    #
    # Matches `src/ffi.rs:5538` (`level == 2 → UA-2, else UA-1`).
    # 1-indexed, not 0-indexed; mirrors the C# `PdfUaLevel` enum.
    PDF_UA_LEVELS = { ua1: 1, ua2: 2 }.freeze

    # @return [Boolean] PDF/A compliance for `level`.
    def pdf_a?(doc, level: :a1b)
      raise ::PdfOxide::ArgumentError, 'doc cannot be nil' if doc.nil?

      ordinal = PDF_A_LEVELS.fetch(level) do
        raise ::PdfOxide::ArgumentError, "unknown PDF/A level: #{level.inspect}"
      end
      # If the native symbol is absent (older cdylib), surface a clean
      # "unavailable" verdict instead of reading an uninitialised err
      # buffer and raising a spurious ComplianceError.
      return false unless Bindings.respond_to?(:pdf_validate_pdf_a_level)

      err = ::FFI::MemoryPointer.new(:int32)
      result_ptr = Bindings.pdf_validate_pdf_a_level(doc.handle, ordinal, err)
      code = err.read_int32
      raise ComplianceError, "pdf_validate_pdf_a_level failed (#{code})" if code != 0

      compliance_verdict(result_ptr, :pdf_pdf_a_is_compliant, :pdf_pdf_a_results_free)
    rescue ::FFI::NotFoundError
      false
    end

    # @return [Boolean] PDF/UA compliance for `level`.
    def pdf_ua?(doc, level: :ua1)
      raise ::PdfOxide::ArgumentError, 'doc cannot be nil' if doc.nil?

      ordinal = PDF_UA_LEVELS.fetch(level) do
        raise ::PdfOxide::ArgumentError, "unknown PDF/UA level: #{level.inspect}"
      end
      err = ::FFI::MemoryPointer.new(:int32)
      result_ptr = Bindings.pdf_validate_pdf_ua(doc.handle, ordinal, err)
      code = err.read_int32
      raise ComplianceError, "pdf_validate_pdf_ua failed (#{code})" if code != 0

      compliance_verdict(result_ptr, :pdf_pdf_ua_is_accessible, :pdf_pdf_ua_results_free)
    rescue ::FFI::NotFoundError
      false
    end

    # @return [Hash] simplified PDF/A validation result: { compliant:, violations: }.
    def validate_pdf_a(doc, level: :a1b)
      { compliant: pdf_a?(doc, level: level), violations: [] }
    end

    # @return [Hash] simplified PDF/UA validation result.
    def validate_pdf_ua(doc, level: :ua1)
      { compliant: pdf_ua?(doc, level: level), violations: [] }
    end

    # The accessor symbols (pdf_pdf_a_is_compliant, pdf_pdf_x_is_compliant,
    # pdf_pdf_ua_is_accessible) all take (results, int32_t *error_code).
    # Pre-v0.3.55 Ruby bound them with just (results) — register garbage
    # was used as the err pointer and the cdylib wrote through it,
    # producing the same flaky segfault class as the search-result
    # accessors (#547). Both args are passed here so the new 2-arg
    # binding is honoured.
    def self.compliance_verdict(result_ptr, accessor_sym, free_sym)
      return false if result_ptr.nil? || result_ptr.null?

      err = ::FFI::MemoryPointer.new(:int32)
      begin
        Bindings.send(accessor_sym, result_ptr, err)
      ensure
        Bindings.send(free_sym, result_ptr) if Bindings.respond_to?(free_sym)
      end
    end
    private_class_method :compliance_verdict
  end
end
