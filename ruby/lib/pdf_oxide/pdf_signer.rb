# frozen_string_literal: true

module PdfOxide
  # PAdES B-B / B-T / B-LT / B-LTA digital-signature signer
  # (v0.3.50 #235 + v0.3.51 5-arg shim).
  #
  # Mirrors `fyi.oxide.pdf.PdfSigner`.  Routes every sign through the
  # 5-arg shim `pdf_sign_bytes_pades_opts` (the 18-arg legacy entry
  # exists but isn't exercised here — purego on SysV/AMD64 can't
  # register it).
  #
  # Per `feedback_extraction_graceful_fallback`: signing is a
  # **security operation** — every non-zero return fails closed.
  class PdfSigner
    # PAdES baseline level codes (mirrors Java's `SignatureLevel` enum).
    LEVELS = { b: 0, t: 1, lt: 2, lta: 3 }.freeze

    # Packed C struct mirroring `PadesSignOptionsC`.  Field order +
    # types MUST match the C header exactly — `#[repr(C)]` on the Rust
    # side guarantees layout stability across platforms.
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

    # @param certificate_handle [FFI::Pointer] PKCS#12 or PEM-loaded
    #   credentials handle (opaque pointer from the credentials API).
    def initialize(certificate_handle)
      raise ::PdfOxide::ArgumentError, 'certificate_handle required' if certificate_handle.nil? || certificate_handle.null?

      @certificate_handle = certificate_handle
    end

    # Sign a PDF (bytes) at the requested PAdES level.
    # @param pdf [String] raw PDF (BINARY).
    # @param level [Symbol] :b, :t, :lt, or :lta.
    # @param tsa_url [String, nil] RFC 3161 TSA URL (required for ≥ :t).
    # @param reason [String, nil]
    # @param location [String, nil]
    # @return [String] BINARY-encoded signed PDF bytes.
    def sign(pdf, level:, tsa_url: nil, reason: nil, location: nil)
      raise ::PdfOxide::ArgumentError, 'pdf cannot be empty' if pdf.nil? || pdf.empty?

      level_code = LEVELS.fetch(level) do
        raise ::PdfOxide::ArgumentError, "level must be one of #{LEVELS.keys.inspect}, got #{level.inspect}"
      end
      if level != :b && (tsa_url.nil? || tsa_url.empty?)
        raise ::PdfOxide::ArgumentError, "PAdES #{level} requires tsa_url"
      end

      self.class.sign_with_handle(
        pdf,
        certificate_handle: @certificate_handle,
        level_code: level_code,
        tsa_url: tsa_url,
        reason: reason,
        location: location
      )
    end

    # Static convenience — sign without constructing a Signer instance.
    # @return [String]
    def self.sign(pdf:, certificate_handle:, level:, tsa_url: nil, reason: nil, location: nil)
      new(certificate_handle).sign(pdf, level: level, tsa_url: tsa_url, reason: reason, location: location)
    end

    # @return [Integer, nil] the PAdES level of an existing signature
    #   handle, or nil if no signatures.
    def self.pades_level(signature_handle)
      raise ::PdfOxide::ArgumentError, 'signature_handle required' if signature_handle.nil? || signature_handle.null?

      err = ::FFI::MemoryPointer.new(:int32)
      ordinal = Bindings.pdf_signature_get_pades_level(signature_handle, err)
      code = err.read_int32
      raise SignatureError, "pdf_signature_get_pades_level failed (#{code})" if code != 0

      ordinal
    end

    # @return [Boolean] whether the doc carries a document-scoped /DocTimeStamp.
    def self.document_has_timestamp?(document_handle)
      raise ::PdfOxide::ArgumentError, 'document_handle required' if document_handle.nil? || document_handle.null?

      err = ::FFI::MemoryPointer.new(:int32)
      r = Bindings.pdf_document_has_timestamp(document_handle, err)
      code = err.read_int32
      raise SignatureError, "pdf_document_has_timestamp failed (#{code})" if code != 0

      r != 0
    end

    # @api private — packs PadesSignOptionsC and invokes the 5-arg shim.
    def self.sign_with_handle(pdf, certificate_handle:, level_code:, tsa_url:, reason:, location:)
      binary = pdf.dup.force_encoding(Encoding::BINARY)
      pdf_buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
      pdf_buf.write_bytes(binary, 0, binary.bytesize)

      # Hold Ruby string buffers in locals so GC doesn't free them while
      # the C call is in flight.
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

      out_len = ::FFI::MemoryPointer.new(:size_t)
      err     = ::FFI::MemoryPointer.new(:int32)
      out_ptr = Bindings.pdf_sign_bytes_pades_opts(pdf_buf, binary.bytesize, opts.to_ptr, out_len, err)
      code = err.read_int32

      raise SignatureError, "pdf_sign_bytes_pades_opts failed (#{code}); security op fails closed" if code != 0
      raise SignatureError, 'pdf_sign_bytes_pades_opts returned null; security op fails closed' if out_ptr.nil? || out_ptr.null?

      len = out_len.read(:size_t)
      signed = out_ptr.read_string(len)
      Bindings.free_bytes(out_ptr) if Bindings.respond_to?(:free_bytes)
      signed.force_encoding(Encoding::BINARY)
    end

    def self.string_ptr(str)
      return nil if str.nil?

      ::FFI::MemoryPointer.from_string(str.to_s.encode('UTF-8'))
    end
    private_class_method :string_ptr
  end
end
