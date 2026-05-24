# frozen_string_literal: true

module PdfOxide
  # v0.3.50 #235 PAdES signature + v0.3.51 5-arg shim — sign PDF bytes
  # with PAdES B-B / B-T / B-LT / B-LTA.
  #
  # The legacy entry `pdf_sign_bytes_pades` takes 18 scalar args which
  # some FFI binders can't register (notably purego on SysV/AMD64).
  # v0.3.51 added a 5-arg shim `pdf_sign_bytes_pades_opts` that takes
  # a packed `PadesSignOptionsC` struct pointer.  Every binding —
  # including Ruby — uses the shim as the canonical entry; the legacy
  # 18-arg form is still available but not exercised here.
  #
  # Per `feedback_extraction_graceful_fallback`: signing is a
  # **security operation** — it fails-closed on any non-zero return.
  module PadesSigner
    # Packed C struct mirroring `PadesSignOptionsC`.  Field order +
    # types MUST match the C header exactly — `#[repr(C)]` on the
    # Rust side guarantees layout stability across platforms.
    class PadesSignOptions < ::FFI::Struct
      layout(
        :certificate_handle, :pointer,
        :certs,              :pointer,
        :cert_lens,          :pointer,
        :n_certs,            :size_t,
        :crls,               :pointer,
        :crl_lens,           :pointer,
        :n_crls,             :size_t,
        :ocsps,              :pointer,
        :ocsp_lens,          :pointer,
        :n_ocsps,            :size_t,
        :tsa_url,            :pointer,
        :reason,             :pointer,
        :location,           :pointer,
        :level,              :int32
      )
    end

    LEVELS = {
      b: FFI::Bindings::PADES_LEVEL_B,
      t: FFI::Bindings::PADES_LEVEL_T,
      lt: FFI::Bindings::PADES_LEVEL_LT,
      lta: FFI::Bindings::PADES_LEVEL_LTA
    }.freeze

    module_function

    # Sign a PDF (bytes) at the requested PAdES level.
    #
    # The `certificate_handle` is an opaque pointer obtained from
    # the credentials API (e.g. `pdf_credentials_from_pkcs12` or
    # `pdf_credentials_get_certificate`).  Loading credentials is
    # out of scope for this module — provide an already-resolved
    # handle.
    #
    # @param pdf_bytes [String] raw PDF (BINARY).
    # @param certificate_handle [FFI::Pointer] opaque cert handle.
    # @param level [Symbol] :b, :t, :lt, or :lta.
    # @param tsa_url [String, nil] RFC 3161 TSA URL (required for ≥ :t).
    # @param reason [String, nil]
    # @param location [String, nil]
    # @return [String] BINARY-encoded signed PDF bytes.
    def sign_pades(pdf_bytes:, certificate_handle:, level:, tsa_url: nil, reason: nil, location: nil)
      raise ::PdfOxide::ArgumentError, 'pdf_bytes cannot be empty' if pdf_bytes.nil? || pdf_bytes.empty?
      raise ::PdfOxide::ArgumentError, 'certificate_handle required' if certificate_handle.nil? || certificate_handle.null?

      level_code = LEVELS.fetch(level) do
        raise ::PdfOxide::ArgumentError, "level must be one of #{LEVELS.keys.inspect}, got #{level.inspect}"
      end

      binary = pdf_bytes.dup.force_encoding(Encoding::BINARY)
      pdf_buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
      pdf_buf.write_bytes(binary, 0, binary.bytesize)

      # Hold Ruby string buffers in locals so the GC doesn't free
      # them while the C call is in flight.
      tsa_buf      = string_ptr(tsa_url)
      reason_buf   = string_ptr(reason)
      location_buf = string_ptr(location)

      opts = PadesSignOptions.new
      opts[:certificate_handle] = certificate_handle
      opts[:certs]              = ::FFI::Pointer::NULL
      opts[:cert_lens]          = ::FFI::Pointer::NULL
      opts[:n_certs]            = 0
      opts[:crls]               = ::FFI::Pointer::NULL
      opts[:crl_lens]           = ::FFI::Pointer::NULL
      opts[:n_crls]             = 0
      opts[:ocsps]              = ::FFI::Pointer::NULL
      opts[:ocsp_lens]          = ::FFI::Pointer::NULL
      opts[:n_ocsps]            = 0
      opts[:tsa_url]            = tsa_buf      || ::FFI::Pointer::NULL
      opts[:reason]             = reason_buf   || ::FFI::Pointer::NULL
      opts[:location]           = location_buf || ::FFI::Pointer::NULL
      opts[:level]              = level_code

      out_len_ptr = ::FFI::MemoryPointer.new(:size_t)
      error_ptr   = ::FFI::MemoryPointer.new(:int32)

      out_ptr = FFI::Bindings.pdf_sign_bytes_pades_opts(
        pdf_buf, binary.bytesize, opts.to_ptr, out_len_ptr, error_ptr
      )
      error_code = error_ptr.read_int32

      if error_code != 0
        raise FFI::ErrorHandler.create_error(
          error_code, 'pdf_sign_bytes_pades_opts', level: level
        )
      end
      if out_ptr.nil? || out_ptr.null?
        raise ::PdfOxide::SignatureError,
              'pdf_sign_bytes_pades_opts returned null (security op; failing closed)'
      end

      len = out_len_ptr.read(:size_t)
      signed = out_ptr.read_string(len)
      FFI::Bindings.free_bytes(out_ptr)
      signed.force_encoding(Encoding::BINARY)
    end

    # Inspect the PAdES level of an existing signature handle.
    # Returns the integer ordinal from `PadesLevel`, or -1 on error.
    def pades_level(signature_handle)
      raise ::PdfOxide::ArgumentError, 'signature_handle required' if signature_handle.nil? || signature_handle.null?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      ordinal = FFI::Bindings.pdf_signature_get_pades_level(signature_handle, error_ptr)
      error_code = error_ptr.read_int32
      raise FFI::ErrorHandler.create_error(error_code, 'pdf_signature_get_pades_level') if error_code != 0

      ordinal
    end

    # Whether a document carries a document-scoped /DocTimeStamp
    # (PAdES B-T or above).
    def document_has_timestamp?(document_handle)
      raise ::PdfOxide::ArgumentError, 'document_handle required' if document_handle.nil? || document_handle.null?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      r = FFI::Bindings.pdf_document_has_timestamp(document_handle, error_ptr)
      error_code = error_ptr.read_int32
      raise FFI::ErrorHandler.create_error(error_code, 'pdf_document_has_timestamp') if error_code != 0

      r != 0
    end

    class << self
      private

      def string_ptr(str)
        return nil if str.nil?

        s = str.to_s.encode('UTF-8')
        ::FFI::MemoryPointer.from_string(s)
      end
    end
  end
end
