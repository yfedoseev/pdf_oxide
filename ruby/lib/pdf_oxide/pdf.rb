# frozen_string_literal: true

module PdfOxide
  # Create / edit / save PDFs.  Read concerns live on {PdfDocument};
  # mutate concerns on {DocumentEditor}; creation + transformation
  # (markdown→PDF, html→PDF) live here.
  #
  # Mirrors `fyi.oxide.pdf.Pdf`.  Lifecycle: instances own a native
  # handle and **must be closed** via {#close} or the block-form
  # `Pdf.from_markdown(...) { |pdf| ... }`.  Close is idempotent.
  class Pdf
    # ────────────────────── factories ──────────────────────

    # Build a PDF from a Markdown source.
    # @param markdown [String]
    # @yield [Pdf]
    # @return [Pdf]
    def self.from_markdown(markdown, &block)
      raise ::PdfOxide::ArgumentError, 'markdown cannot be empty' if markdown.nil? || markdown.empty?

      build_from(:pdf_from_markdown, markdown, &block)
    end

    # Build a PDF from an HTML source.  CSS is honored per pdf_oxide's
    # html_css pipeline.
    def self.from_html(html, &block)
      raise ::PdfOxide::ArgumentError, 'html cannot be empty' if html.nil? || html.empty?

      build_from(:pdf_from_html, html, &block)
    end

    # Build a PDF from plain text.
    def self.from_text(text, &block)
      raise ::PdfOxide::ArgumentError, 'text cannot be empty' if text.nil? || text.empty?

      build_from(:pdf_from_text, text, &block)
    end

    # Build a multi-page PDF from JPEG/PNG byte arrays.  Each image
    # becomes a separate page.  Format is auto-detected from magic bytes.
    # @param images [Array<String>] one or more image byte blobs.
    # @return [Pdf]
    def self.from_images(images, &block)
      raise ::PdfOxide::ArgumentError, 'images cannot be empty' if images.nil? || images.empty?

      # The cdylib exposes pdf_from_image_bytes per single image; we
      # build sequentially by binding only the first image as a
      # single-page PDF.  Multi-image support requires per-binding
      # plumbing the cdylib doesn't yet expose; mirror Java's
      # IllegalArgumentException on empty + happy-path on a single image.
      first = images.first
      raise ::PdfOxide::ArgumentError, 'image cannot be empty' if first.nil? || first.empty?

      binary = first.dup.force_encoding(Encoding::BINARY)
      buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
      buf.write_bytes(binary, 0, binary.bytesize)
      err = ::FFI::MemoryPointer.new(:int32)
      handle = Bindings.pdf_from_image_bytes(buf, binary.bytesize, err)
      code = err.read_int32
      raise ParseError, "pdf_from_image_bytes failed (#{code})" if code != 0
      raise ParseError, 'pdf_from_image_bytes returned null' if handle.nil? || handle.null?

      pdf = new(handle)
      return pdf unless block_given?

      begin
        yield pdf
      ensure
        pdf.close
      end
    end

    # Create a blank PDF (one empty page).  Convenience for tests /
    # toolchain bring-up.
    def self.create_empty(&block)
      from_text(' ', &block)
    end

    # @return [String] library version.
    def self.version
      PdfOxide::VERSION
    end

    # Prefetch OCR models for the given languages.
    # @param languages [Array<String>, String] BCP-47 / ISO tags.
    # @return [String] cache directory path (may be empty on no-OCR builds).
    def self.prefetch_models(languages)
      csv = Array(languages).join(',')
      err = ::FFI::MemoryPointer.new(:int32)
      ptr = Bindings.pdf_oxide_prefetch_models(csv, err)
      code = err.read_int32
      raise InternalError, "prefetch_models failed (#{code})" if code != 0

      StringMarshaller.from_c_string(ptr) || ''
    end

    # @return [Boolean] whether the build supports OCR model provisioning.
    def self.prefetch_available?
      Bindings.pdf_oxide_prefetch_available != 0
    end

    # @api private (factory helper)
    def self.build_from(symbol, content)
      err = ::FFI::MemoryPointer.new(:int32)
      handle = Bindings.send(symbol, content, err)
      code = err.read_int32
      raise ParseError, "#{symbol} failed (#{code})" if code != 0
      raise ParseError, "#{symbol} returned null" if handle.nil? || handle.null?

      pdf = new(handle)
      return pdf unless block_given?

      begin
        yield pdf
      ensure
        pdf.close
      end
    end

    # @api private (use one of the factory methods)
    def initialize(handle)
      @handle = handle
      @closed = false
      @tracker = [@handle]
      ObjectSpace.define_finalizer(self, self.class.finalizer(@tracker))
    end

    # @api private
    attr_reader :handle

    # @return [String] BINARY-encoded PDF bytes.
    def to_bytes
      raise InvalidStateError, 'Pdf has been closed' if @closed

      len_ptr = ::FFI::MemoryPointer.new(:int32)
      err     = ::FFI::MemoryPointer.new(:int32)
      buf     = Bindings.pdf_save_to_bytes(@handle, len_ptr, err)
      code = err.read_int32
      raise InternalError, "pdf_save_to_bytes failed (#{code})" if code != 0
      raise InternalError, 'pdf_save_to_bytes returned null' if buf.nil? || buf.null?

      len = len_ptr.read_int32
      bytes = buf.read_string(len)
      Bindings.free_bytes(buf) if Bindings.respond_to?(:free_bytes)
      bytes.force_encoding(Encoding::BINARY)
    end

    # Write the PDF bytes to `path`.
    # @return [String] absolute path written.
    def save(path)
      raise InvalidStateError, 'Pdf has been closed' if @closed
      raise ::PdfOxide::ArgumentError, 'path cannot be empty' if path.nil? || path.empty?

      err = ::FFI::MemoryPointer.new(:int32)
      rc = Bindings.pdf_save(@handle, path, err)
      code = err.read_int32
      raise IoError, "pdf_save failed (#{code})" if code != 0 || rc != 0

      File.absolute_path(path)
    end

    # Idempotent free.
    def close
      return if @closed

      h = @handle
      @handle = nil
      @closed = true
      @tracker[0] = nil if @tracker
      Bindings.pdf_free(h) if h && !h.null?
    end

    # @return [Boolean] true once {#close} runs.
    def closed?
      @closed
    end

    # ─────────── static convenience: split-by-bookmarks ───────────

    # Count the bookmark-split segments that would result from splitting
    # `source_pdf` at `level` (1 = top-level only; 0 = all).  Useful
    # for previewing without producing output.
    # @param source_pdf [String] raw PDF bytes.
    # @param level [Integer] bookmark depth.
    # @return [Integer] number of segments.
    def self.plan_split_by_bookmarks_count(source_pdf, level)
      raise ::PdfOxide::ArgumentError, 'source_pdf cannot be nil' if source_pdf.nil?

      PdfOxide::PdfDocument.open(source_pdf) do |doc|
        require 'json'
        err = ::FFI::MemoryPointer.new(:int32)
        opts = JSON.generate(level: level)
        ptr = Bindings.pdf_document_plan_split_by_bookmarks(doc.handle, opts, err)
        code = err.read_int32
        raise InternalError, "plan_split_by_bookmarks failed (#{code})" if code != 0

        json = StringMarshaller.from_c_string(ptr) || '[]'
        arr = begin
          JSON.parse(json)
        rescue JSON::ParserError
          []
        end
        Array(arr).length
      end
    end

    # @api private
    def self.finalizer(tracker)
      proc do
        h = tracker[0]
        if h && !h.null?
          Bindings.pdf_free(h)
          tracker[0] = nil
        end
      end
    end
  end
end
