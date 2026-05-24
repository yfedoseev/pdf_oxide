# frozen_string_literal: true

module PdfOxide
  # Write-side counterpart to {PdfDocument}: form-fill, destructive
  # redaction (v0.3.50 #231), metadata scrubbing, and incremental save.
  #
  # Mirrors `fyi.oxide.pdf.DocumentEditor`.  Lifecycle: holds a native
  # `DocumentEditor*` handle; **must** be closed via {#close} or a
  # block-form factory.  Close is idempotent.
  #
  # Per `feedback_extraction_graceful_fallback`: destructive redaction
  # is a **security operation** — every non-zero return code raises
  # rather than silently degrading.
  #
  # @example destructive redaction (block-form auto-closes).
  #   PdfOxide::DocumentEditor.open('source.pdf') do |ed|
  #     ed.add_redaction(page: 0, rect: [100, 200, 300, 250])
  #     ed.apply_redactions!
  #     ed.save_to('redacted.pdf')
  #   end
  class DocumentEditor
    # Open an editor session over a PDF on disk (or in-memory bytes).
    # @param source [String] file path or raw PDF bytes.
    # @yield [DocumentEditor]
    # @return [DocumentEditor, Object]
    def self.open(source, &block)
      ed = new(source)
      return ed unless block_given?

      begin
        yield ed
      ensure
        ed.close
      end
    end

    def initialize(source)
      raise ::PdfOxide::ArgumentError, 'source cannot be nil' if source.nil?

      err = ::FFI::MemoryPointer.new(:int32)
      @handle =
        if source.is_a?(String) && File.exist?(source)
          Bindings.document_editor_open(File.absolute_path(source), err)
        elsif source.is_a?(String) && !source.empty?
          binary = source.dup.force_encoding(Encoding::BINARY)
          buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
          buf.write_bytes(binary, 0, binary.bytesize)
          Bindings.document_editor_open_from_bytes(buf, binary.bytesize, err)
        else
          raise FileNotFoundError, "file not found: #{source}"
        end

      code = err.read_int32
      raise IoError, "document_editor_open failed (#{code})" if code != 0
      raise IoError, 'document_editor_open returned null' if @handle.nil? || @handle.null?

      @closed  = false
      @applied = false
      @tracker = [@handle]
      ObjectSpace.define_finalizer(self, self.class.finalizer(@tracker))
    end

    # @api private
    attr_reader :handle

    # ─────────────── destructive redaction (#231) ───────────────

    # Queue a redaction rectangle for the given page.  The redaction
    # is not applied until {#apply_redactions!} runs.
    # @param page [Integer] 0-based page index.
    # @param rect [Array<Numeric>] `[x1, y1, x2, y2]` in PDF user-space.
    # @param color [Array<Numeric>] `[r, g, b]` overlay color (0.0–1.0).
    # @return [self] (fluent chaining).
    def add_redaction(page:, rect:, color: [0.0, 0.0, 0.0])
      check_open!
      raise ::PdfOxide::ArgumentError, 'rect must have 4 numeric values' unless rect.respond_to?(:length) && rect.length == 4

      x1, y1, x2, y2 = rect.map(&:to_f)
      r, g, b = color.map(&:to_f)
      err = ::FFI::MemoryPointer.new(:int32)
      rc = Bindings.pdf_redaction_add(@handle, Integer(page), x1, y1, x2, y2, r, g, b, err)
      fail_closed!(rc, err.read_int32, 'pdf_redaction_add')
      self
    end

    # Total redactions queued for the page.
    # @param page [Integer]
    # @return [Integer]
    def redaction_count(page)
      check_open!
      err = ::FFI::MemoryPointer.new(:int32)
      n = Bindings.pdf_redaction_count(@handle, Integer(page), err)
      fail_closed!(0, err.read_int32, 'pdf_redaction_count')
      n
    end

    # Apply all queued redactions destructively.
    # @param scrub_metadata [Boolean] also strip /Info, XMP, JS.
    # @param fill_color [Array<Numeric>] overlay `[r, g, b]`.
    # @return [self]
    def apply_redactions!(scrub_metadata: false, fill_color: [0.0, 0.0, 0.0])
      check_open!
      r, g, b = fill_color.map(&:to_f)
      err = ::FFI::MemoryPointer.new(:int32)
      rc = Bindings.pdf_redaction_apply(@handle, scrub_metadata, r, g, b, err)
      fail_closed!(rc, err.read_int32, 'pdf_redaction_apply')

      if scrub_metadata
        err2 = ::FFI::MemoryPointer.new(:int32)
        rc2 = Bindings.pdf_redaction_scrub_metadata(@handle, err2)
        fail_closed!(rc2, err2.read_int32, 'pdf_redaction_scrub_metadata')
      end
      @applied = true
      self
    end

    # Metadata scrubbing without redaction regions.
    # @return [self]
    def scrub_metadata
      check_open!
      err = ::FFI::MemoryPointer.new(:int32)
      rc = Bindings.pdf_redaction_scrub_metadata(@handle, err)
      fail_closed!(rc, err.read_int32, 'pdf_redaction_scrub_metadata')
      @applied = true
      self
    end

    # ─────────────── form-fill ───────────────

    # Set an AcroForm text field.
    # @param name [String] dot-separated full field name.
    # @param value [String, Boolean] new value (Boolean = checkbox/radio).
    # @return [self]
    def set_form_field(name, value)
      check_open!
      raise ::PdfOxide::ArgumentError, 'name cannot be nil' if name.nil?

      err = ::FFI::MemoryPointer.new(:int32)
      ok = if [true, false].include?(value)
             Bindings.pdf_form_field_set_value_by_name_boolean(@handle, name, value, err)
           else
             Bindings.pdf_form_field_set_value_by_name_string(@handle, name, value.to_s, err)
           end
      code = err.read_int32
      raise InternalError, "set_form_field failed (#{code})" if code != 0
      raise InternalError, 'set_form_field rejected by cdylib (field missing?)' unless ok

      self
    rescue ::FFI::NotFoundError
      # phantom in this cdylib build — leave the field-write a no-op
      # and surface a clear error rather than crashing.
      raise UnsupportedFeatureError, 'form-fill not supported by this cdylib build'
    end

    # ─────────────── save ───────────────

    # Save the edited PDF to the given path.
    # @return [String] absolute path written.
    def save_to(path)
      check_open!
      raise ::PdfOxide::ArgumentError, 'path cannot be empty' if path.nil? || path.empty?

      check_applied! if @needs_apply
      err = ::FFI::MemoryPointer.new(:int32)
      rc = Bindings.document_editor_save(@handle, File.absolute_path(path), err)
      fail_closed!(rc, err.read_int32, 'document_editor_save')
      File.absolute_path(path)
    end

    # @return [String] BINARY-encoded PDF bytes.
    def to_bytes
      check_open!
      len_ptr = ::FFI::MemoryPointer.new(:size_t)
      err     = ::FFI::MemoryPointer.new(:int32)
      buf     = Bindings.document_editor_save_to_bytes(@handle, len_ptr, err)
      fail_closed!(0, err.read_int32, 'document_editor_save_to_bytes')
      raise InternalError, 'document_editor_save_to_bytes returned null' if buf.nil? || buf.null?

      len = len_ptr.read(:size_t)
      bytes = buf.read_string(len)
      Bindings.free_bytes(buf) if Bindings.respond_to?(:free_bytes)
      bytes.force_encoding(Encoding::BINARY)
    end

    # ─────────────── lifecycle ───────────────

    # Idempotent close.
    def close
      return if @closed

      h = @handle
      @handle = nil
      @closed = true
      @tracker[0] = nil if @tracker
      Bindings.document_editor_free(h) if h && !h.null?
    end

    def closed?
      @closed
    end

    # @api private
    def self.finalizer(tracker)
      proc do
        h = tracker[0]
        if h && !h.null?
          Bindings.document_editor_free(h)
          tracker[0] = nil
        end
      end
    end

    private

    def check_open!
      raise InvalidStateError, 'DocumentEditor has been closed' if @closed || @handle.nil?
    end

    def check_applied!
      return if @applied

      raise StateError, 'no redactions applied; call apply_redactions! before save'
    end

    # Security-op fail-closed contract: any non-zero rc OR error_code raises.
    def fail_closed!(rc, error_code, operation)
      if error_code != 0
        raise RedactionError, "#{operation} failed (error code #{error_code}); security op fails closed"
      end
      return unless rc.is_a?(Integer) && rc.negative?

      raise RedactionError, "#{operation} returned #{rc}; security op fails closed"
    end
  end
end
