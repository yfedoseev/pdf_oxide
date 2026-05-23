# frozen_string_literal: true

module PdfOxide
  # v0.3.50 #231 — destructive redaction.
  #
  # Wraps the `pdf_redaction_*` + `document_editor_*` C-ABI pair so a
  # caller can mark rectangular regions on a PDF, apply the
  # redactions destructively (true content removal + opaque overlay,
  # ISO 32000-1 §12.5.6.23), and obtain the redacted bytes.
  #
  # Per `feedback_extraction_graceful_fallback`: redaction is a
  # **security operation** — it fails-closed on any non-zero return
  # code rather than silently degrading.
  #
  # @example Black-rectangle a page region and save.
  #   PdfOxide::RedactionManager.open('source.pdf') do |r|
  #     r.add(page: 0, rect: [100, 200, 300, 250])
  #     r.apply!
  #     r.save('redacted.pdf')
  #   end
  class RedactionManager
    attr_reader :handle

    # Open a redaction session over a PDF on disk.
    # @param path [String]
    # @yield [self] block-form auto-closes the session
    # @return [RedactionManager]
    def self.open(path, &block)
      session = new(path: path)
      if block_given?
        begin
          yield session
        ensure
          session.close
        end
      else
        session
      end
    end

    # Open a redaction session over PDF bytes.
    # @param bytes [String] raw PDF bytes (BINARY-encoded)
    # @yield [self]
    def self.open_bytes(bytes, &block)
      session = new(bytes: bytes)
      if block_given?
        begin
          yield session
        ensure
          session.close
        end
      else
        session
      end
    end

    def initialize(path: nil, bytes: nil)
      raise ::PdfOxide::ArgumentError,
            'must supply path: or bytes:' if path.nil? && bytes.nil?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      @handle =
        if path
          raise ::PdfOxide::FileNotFoundError, "file not found: #{path}" unless File.exist?(path)

          FFI::Bindings.document_editor_open(File.absolute_path(path), error_ptr)
        else
          binary = bytes.dup.force_encoding(Encoding::BINARY)
          buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
          buf.write_bytes(binary, 0, binary.bytesize)
          FFI::Bindings.document_editor_open_from_bytes(buf, binary.bytesize, error_ptr)
        end

      error_code = error_ptr.read_int32
      if error_code != 0 || @handle.nil? || @handle.null?
        raise FFI::ErrorHandler.create_error(error_code, 'document_editor_open')
      end

      @closed = false
      @pending = 0
      @applied = false
      # Use a mutable tracker so an explicit `close` can defuse the
      # finalizer (`document_editor_free` is not idempotent against
      # the same pointer twice — double-free corruption ensues).
      @tracker = [@handle]
      ObjectSpace.define_finalizer(self, self.class.finalizer(@tracker))
    end

    # Add a redaction rectangle to a page.  Coordinates are PDF
    # points (origin bottom-left), pre-rotation.
    #
    # @param page [Integer] zero-based page index.
    # @param rect [Array<Numeric>] `[x1, y1, x2, y2]` — opposite corners.
    # @param color [Array<Numeric>] `[r, g, b]` overlay color, each 0.0–1.0.
    # @return [self]
    def add(page:, rect:, color: [0.0, 0.0, 0.0])
      check_open!
      raise ::PdfOxide::ArgumentError, 'rect must have 4 numeric values' unless rect.respond_to?(:length) && rect.length == 4

      x1, y1, x2, y2 = rect.map(&:to_f)
      r, g, b = color.map(&:to_f)
      error_ptr = ::FFI::MemoryPointer.new(:int32)

      rc = FFI::Bindings.pdf_redaction_add(
        @handle, Integer(page),
        x1, y1, x2, y2,
        r, g, b,
        error_ptr
      )
      check_security_op!(rc, error_ptr.read_int32, 'pdf_redaction_add', page: page)
      @pending += 1
      self
    end

    # Number of pending redactions queued on a specific page.
    def count_for(page)
      check_open!
      error_ptr = ::FFI::MemoryPointer.new(:int32)
      n = FFI::Bindings.pdf_redaction_count(@handle, Integer(page), error_ptr)
      check_security_op!(0, error_ptr.read_int32, 'pdf_redaction_count', page: page)
      n
    end

    # Apply ALL queued redactions destructively.
    #
    # @param scrub_metadata [Boolean] also strip /Info, XMP, JS.
    # @param fill_color [Array<Numeric>] overlay `[r, g, b]` in 0.0–1.0.
    # @return [self]
    def apply!(scrub_metadata: false, fill_color: [0.0, 0.0, 0.0])
      check_open!
      r, g, b = fill_color.map(&:to_f)
      error_ptr = ::FFI::MemoryPointer.new(:int32)
      rc = FFI::Bindings.pdf_redaction_apply(@handle, scrub_metadata, r, g, b, error_ptr)
      check_security_op!(rc, error_ptr.read_int32, 'pdf_redaction_apply')

      if scrub_metadata
        # The Rust apply call only redacts content; metadata is a
        # separate destructive op.  Both must succeed for the redaction
        # to be considered complete.
        error_ptr2 = ::FFI::MemoryPointer.new(:int32)
        rc2 = FFI::Bindings.pdf_redaction_scrub_metadata(@handle, error_ptr2)
        check_security_op!(rc2, error_ptr2.read_int32, 'pdf_redaction_scrub_metadata')
      end
      @applied = true
      self
    end

    # Persist the redacted PDF to disk.
    # @param path [String]
    # @return [String] absolute path written
    def save(path)
      check_open!
      check_applied!
      raise ::PdfOxide::ArgumentError, 'path cannot be nil/empty' if path.nil? || path.empty?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      rc = FFI::Bindings.document_editor_save(@handle, File.absolute_path(path), error_ptr)
      check_security_op!(rc, error_ptr.read_int32, 'document_editor_save', path: path)
      File.absolute_path(path)
    end

    # Return the redacted PDF as bytes.
    # @return [String] BINARY-encoded PDF bytes
    def to_bytes
      check_open!
      check_applied!

      len_ptr   = ::FFI::MemoryPointer.new(:size_t)
      error_ptr = ::FFI::MemoryPointer.new(:int32)
      buf_ptr   = FFI::Bindings.document_editor_save_to_bytes(@handle, len_ptr, error_ptr)
      check_security_op!(0, error_ptr.read_int32, 'document_editor_save_to_bytes')
      raise ::PdfOxide::RedactionError, 'editor returned null buffer' if buf_ptr.nil? || buf_ptr.null?

      len = len_ptr.read(:size_t)
      bytes = buf_ptr.read_string(len)
      FFI::Bindings.free_bytes(buf_ptr)
      bytes.force_encoding(Encoding::BINARY)
    end

    # Release the underlying DocumentEditor handle.
    def close
      return if @closed
      if @handle && !@handle.null?
        FFI::Bindings.document_editor_free(@handle)
      end
      # Defuse the finalizer so the GC pass doesn't double-free.
      @tracker[0] = nil
      @closed = true
      @handle = nil
    end

    def closed?
      @closed
    end

    def self.finalizer(tracker)
      proc do
        handle = tracker[0]
        if handle && !handle.null?
          FFI::Bindings.document_editor_free(handle)
          tracker[0] = nil
        end
      end
    end

    private

    def check_open!
      raise ::PdfOxide::StateError, 'redaction session closed' if @closed || @handle.nil?
    end

    def check_applied!
      raise ::PdfOxide::StateError,
            'no redactions applied; call apply! before save/to_bytes' unless @applied
    end

    # Fail-closed: any non-zero rc OR non-zero error_code raises.
    def check_security_op!(rc, error_code, operation, **context)
      if error_code != 0
        raise FFI::ErrorHandler.create_error(error_code, operation, **context)
      end
      if rc < 0
        raise ::PdfOxide::RedactionError.new(
          "#{operation} returned #{rc} (security operation; failing closed)"
        )
      end
    end
  end
end
