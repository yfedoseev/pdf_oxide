# frozen_string_literal: true

module PdfOxide
  # The primary read-only entry point to a PDF.
  #
  # Mirrors `fyi.oxide.pdf.PdfDocument`.  Lifecycle: a PdfDocument owns
  # native memory and **must be closed** when no longer in use.  The
  # idiomatic Ruby pattern is the block form `PdfDocument.open(path) do |doc| ... end`
  # which closes automatically; for parity with the Java `AutoCloseable`
  # contract, an explicit `#close` is also supported and is idempotent
  # (a second call is a no-op, not a crash).
  #
  # A `Finalizer` backstop frees leaked handles on GC; callers must
  # not rely on it for timely cleanup.
  #
  # @example block form (recommended)
  #   PdfOxide::PdfDocument.open('invoice.pdf') do |doc|
  #     puts doc.extract_text(0)
  #   end
  #
  # @example explicit close
  #   doc = PdfOxide::PdfDocument.open('invoice.pdf')
  #   begin
  #     puts doc.extract_text(0)
  #   ensure
  #     doc.close
  #   end
  class PdfDocument
    # @return [String] absolute path the document was opened from
    #   (or a synthetic `<in-memory>` token for byte-opened docs).
    attr_reader :path

    # Open a PDF from disk or in-memory bytes.
    #
    # @param source [String] either a filesystem path or raw PDF bytes
    #   (auto-detected via `%PDF-` magic on BINARY-encoded input).
    # @param password [String, nil] optional password for encrypted PDFs.
    # @yield [PdfDocument] block form auto-closes on return.
    # @return [PdfDocument, Object] the document, or the block's return value.
    # @raise [FileNotFoundError] path doesn't exist.
    # @raise [ParseError] malformed PDF.
    # @raise [EncryptedError] wrong password / authentication failed.
    def self.open(source, password: nil, &block)
      doc = new(source, password: password)
      return doc unless block_given?

      begin
        yield doc
      ensure
        doc.close
      end
    end

    # One-shot: open + extract page text + close.
    # @param source [String] path or bytes (see #open).
    # @param page [Integer] 0-based page index (default 0).
    # @return [String] extracted text.
    def self.extract_text(source, page: 0)
      # rubocop:disable Security/Open — PdfDocument.open opens a PDF, not a process.
      open(source) { |d| d.extract_text(page) }
      # rubocop:enable Security/Open
    end

    # Open a PDF.  See {.open} for the block-form factory.
    def initialize(source, password: nil)
      raise ::PdfOxide::ArgumentError, 'source cannot be nil' if source.nil?

      @path, @handle = open_native(source)
      @closed = false
      # Mutable tracker lets an explicit `#close` defuse the finalizer
      # so the GC pass doesn't double-free.
      @tracker = [@handle]
      ObjectSpace.define_finalizer(self, self.class.finalizer(@tracker))

      authenticate(password) if password
    end

    # @return [FFI::Pointer] raw handle for sibling classes
    #   (MarkdownConverter, AutoExtractor, PdfValidator, PdfSigner)
    #   that need to pass the pointer to their own FFI calls.
    # @raise [InvalidStateError] document has been closed.
    def handle
      raise InvalidStateError, 'PdfDocument has been closed' if @closed || @handle.nil?

      @handle
    end

    # Authenticate against this document's encryption.
    # @param password [String]
    # @return [Boolean] true on success / unencrypted; false on wrong password.
    def authenticate(password)
      raise ::PdfOxide::ArgumentError, 'password cannot be nil' if password.nil?
      return true unless encrypted?

      # v0.3.55 cdylib doesn't expose a stable 3-arg unlock entry;
      # the legacy `pdf_document_unlock_with_password` is a phantom
      # (REMOVED) and `pdf_document_authenticate` only has the
      # 8-pointer placeholder shape.  Return false on encrypted docs
      # rather than crash — Java's PdfDocument#authenticate has the
      # same fail-closed contract.
      false
    end

    # @return [Integer] number of pages.
    def page_count
      err = ::FFI::MemoryPointer.new(:int32)
      n = Bindings.pdf_document_get_page_count(handle, err)
      raise_for_code(err.read_int32, 'page_count')
      n
    end

    # @return [String] PDF version string (e.g. "1.7").
    def pdf_version
      maj = ::FFI::MemoryPointer.new(:uint8)
      min = ::FFI::MemoryPointer.new(:uint8)
      Bindings.pdf_document_get_version(handle, maj, min)
      "#{maj.read_uint8}.#{min.read_uint8}"
    rescue ::FFI::NotFoundError
      'unknown'
    end

    # @return [Boolean] whether this PDF carries an encryption dictionary.
    def encrypted?
      # bool pdf_document_is_encrypted(const PdfDocument *handle) — no err arg.
      # The cdylib silently swallowed the extra err pointer pre-v0.3.55, so
      # encryption-detection failures were never surfaced.
      Bindings.pdf_document_is_encrypted(handle)
    end

    # Extract plain text from a single page.
    # @param page_index [Integer] 0-based page index.
    # @return [String] extracted text (empty for pages with no text layer).
    def extract_text(page_index)
      validate_page_index(page_index)
      err = ::FFI::MemoryPointer.new(:int32)
      ptr = Bindings.pdf_document_extract_text(handle, page_index, err)
      raise_for_code(err.read_int32, 'extract_text')
      StringMarshaller.from_c_string(ptr) || ''
    end

    # Extract a structured representation of a single page (#536).
    # Returns the parsed `StructuredPage` JSON as a Hash:
    # `{ "page_index", "page_width", "page_height",
    #    "regions" => [ { "kind", "text", "bbox", "spans", "column_index" } ] }`.
    # @param page [Integer] 0-based page index.
    # @return [Hash] parsed structured page.
    def extract_structured(page)
      validate_page_index(page)
      err = ::FFI::MemoryPointer.new(:int32)
      ptr = Bindings.pdf_document_extract_structured_to_json(handle, page, err)
      raise_for_code(err.read_int32, 'extract_structured')
      json = StringMarshaller.from_c_string(ptr) || ''

      require 'json'
      JSON.parse(json)
    end

    # Auto-routed extraction for a single page (v0.3.51 #517).
    # Returns native text where present, OCR'd text for scanned regions
    # when the `ocr` feature is available, and gracefully falls back to
    # native + empty/partial text when OCR is not available — never
    # raises an "OCR unavailable" error on this path.
    # @param page_index [Integer] 0-based.
    # @return [String] extracted text.
    def extract_text_auto(page_index)
      validate_page_index(page_index)
      err = ::FFI::MemoryPointer.new(:int32)
      ptr = Bindings.pdf_document_extract_text_auto(handle, page_index, err)
      raise_for_code(err.read_int32, 'extract_text_auto')
      StringMarshaller.from_c_string(ptr) || ''
    end

    # Convert one page to Markdown.
    # @param page_index [Integer]
    # @return [String] Markdown.
    def to_markdown(page_index = nil)
      page_index.nil? ? MarkdownConverter.to_markdown(self) : MarkdownConverter.to_markdown(self, page_index)
    end

    # Convert one page to HTML.
    # @param page_index [Integer]
    # @return [String] HTML.
    def to_html(page_index = nil)
      page_index.nil? ? MarkdownConverter.to_html(self) : MarkdownConverter.to_html(self, page_index)
    end

    # Search this document.
    # @param query [String] literal text (or regex when `regex: true`).
    # @param case_sensitive [Boolean]
    # @param regex [Boolean] interpret query as a regex.
    # @return [Array<Hash>] each match has keys :page, :text, :bbox
    #   (where :bbox is a Hash with :x, :y, :width, :height).
    def search(query, case_sensitive: false, regex: false)
      raise ::PdfOxide::ArgumentError, 'query cannot be nil' if query.nil?
      raise UnsupportedFeatureError, 'regex search not supported by this cdylib build' \
        if regex && !Bindings.respond_to?(:pdf_document_search_regex)

      err = ::FFI::MemoryPointer.new(:int32)
      query_utf8 = StringMarshaller.to_utf8(query)
      results = if regex
                  Bindings.pdf_document_search_regex(handle, query_utf8, case_sensitive, err)
                else
                  Bindings.pdf_document_search_all(handle, query_utf8, case_sensitive, err)
                end
      raise_for_code(err.read_int32, 'search')
      parse_search_results(results)
    end

    # @return [Array<Hash>] AcroForm fields as an array of `{name:, value:, type:, page:}`
    #   hashes.  v0.3.55 limitation: per-field `page` is -1 because
    #   pdf_oxide's form extractor doesn't yet surface per-field page
    #   placement; field is identified by `name`.  When the cdylib
    #   build lacks the form-extract accessor, returns `[]` rather
    #   than raising — the simple-PDF case is "no form fields".
    def form_fields
      return [] unless Bindings.respond_to?(:pdf_document_get_form_fields)

      err = ::FFI::MemoryPointer.new(:int32)
      ptr = begin
        Bindings.pdf_document_get_form_fields(handle, err)
      rescue ::ArgumentError
        # Phantom 8-pointer skeleton — graceful empty.
        return []
      end
      raise_for_code(err.read_int32, 'form_fields')
      return [] if ptr.nil? || ptr.null?

      json = StringMarshaller.from_c_string(ptr) || ''
      return [] if json.empty?

      require 'json'
      arr = JSON.parse(json)
      Array(arr).map do |f|
        {
          name: f['name'],
          value: f['value'],
          type: f['type'],
          page: f.fetch('page', -1)
        }
      end
    rescue JSON::ParserError
      []
    end

    # Render a single page to PNG bytes at the supplied DPI.
    # @param page_index [Integer]
    # @param dpi [Integer] resolution (default 150).
    # @return [String] PNG-encoded image bytes (BINARY).
    def render(page_index, dpi: 150)
      validate_page_index(page_index)
      err = ::FFI::MemoryPointer.new(:int32)
      img_ptr = Bindings.pdf_render_page_zoom(handle, page_index, dpi.to_f / 72.0, 0, err)
      raise_for_code(err.read_int32, 'render')
      raise InternalError, 'render returned null' if img_ptr.nil? || img_ptr.null?

      # Read length + bytes via rendered image helpers.  The cdylib
      # exposes `pdf_oxide_rendered_image_*` accessors; the simpler
      # path is the byte-buffer accessor introduced for v0.3.5x.
      bytes = read_rendered_image_bytes(img_ptr)
      Bindings.pdf_rendered_image_free(img_ptr) if Bindings.respond_to?(:pdf_rendered_image_free)
      bytes.b # binary-encoded copy (never mutates; read_* may return a frozen empty string)
    end

    # Render a single page with the full RenderOptions surface plus
    # Optional-Content-Group (OCG) layer filtering.
    #
    # @param page_index [Integer]
    # @param dpi [Integer] resolution (default 150).
    # @param format [Integer] 0 = PNG, 1 = JPEG.
    # @param background [Array(Float,Float,Float,Float)] RGBA, each 0.0..1.0.
    # @param transparent [Boolean] drop the background fill entirely.
    # @param render_annotations [Boolean] paint annotation appearances.
    # @param jpeg_quality [Integer] 1..100 (only used when format == 1).
    # @param excluded_layers [Array<String>] OCG `/Name`s to suppress.
    # @return [String] encoded image bytes (BINARY).
    def render_with_layers(page_index, dpi: 150, format: 0,
                           background: [1.0, 1.0, 1.0, 1.0], transparent: false,
                           render_annotations: true, jpeg_quality: 90,
                           excluded_layers: [])
      validate_page_index(page_index)
      bg_r, bg_g, bg_b, bg_a = background
      names = Array(excluded_layers).map(&:to_s)

      # Build a NULL-terminated-string array (char *const *).
      names_ptr = ::FFI::Pointer::NULL
      unless names.empty?
        str_ptrs = names.map { |n| ::FFI::MemoryPointer.from_string(n) }
        names_ptr = ::FFI::MemoryPointer.new(:pointer, str_ptrs.length)
        names_ptr.write_array_of_pointer(str_ptrs)
      end

      err = ::FFI::MemoryPointer.new(:int32)
      img_ptr = Bindings.pdf_render_page_with_options_ex(
        handle, page_index, dpi, format,
        bg_r.to_f, bg_g.to_f, bg_b.to_f, bg_a.to_f,
        transparent ? 1 : 0, render_annotations ? 1 : 0, jpeg_quality,
        names_ptr, names.length, err
      )
      raise_for_code(err.read_int32, 'render_with_layers')
      raise InternalError, 'render_with_layers returned null' if img_ptr.nil? || img_ptr.null?

      bytes = read_rendered_image_bytes(img_ptr)
      Bindings.pdf_rendered_image_free(img_ptr) if Bindings.respond_to?(:pdf_rendered_image_free)
      bytes.b # binary-encoded copy (never mutates; read_* may return a frozen empty string)
    end

    # @return [PdfPage] a lightweight view of the page at `index`.
    #   The page borrows from this document; using it after the doc
    #   closes raises `InvalidStateError`.
    def page(index)
      validate_page_index(index)
      PdfPage.new(self, index)
    end

    # @return [Array<PdfPage>] every page in the document (eager).
    def pages
      n = page_count
      Array.new(n) { |i| PdfPage.new(self, i) }
    end

    # Convenience accessor: get the configured {AutoExtractor} for this doc.
    # @return [AutoExtractor]
    def auto_extractor
      @auto_extractor ||= AutoExtractor.new(self)
    end

    # Free the native handle.  Idempotent — calling more than once is a
    # no-op, not a crash.  Safe to call from an ensure block.
    def close
      return if @closed

      h = @handle
      @handle = nil
      @closed = true
      # Defuse the finalizer (was @tracker[0] == @handle).
      @tracker[0] = nil if @tracker
      Bindings.pdf_document_free(h) if h && !h.null?
    end

    # @return [Boolean] true if {#close} has not been called.
    def open?
      !@closed
    end

    # @return [Boolean] true after {#close}.
    def closed?
      @closed
    end

    # Finalizer for GC cleanup.  The mutable tracker lets explicit
    # `#close` zero out the handle so a follow-up GC pass doesn't
    # double-free (the cdylib's `pdf_document_free` is not idempotent
    # on the same pointer).
    # @api private
    def self.finalizer(tracker)
      proc do
        handle = tracker[0]
        if handle && !handle.null?
          Bindings.pdf_document_free(handle)
          tracker[0] = nil
        end
      end
    end

    private

    def open_native(source)
      err = ::FFI::MemoryPointer.new(:int32)
      handle, path =
        # NB: detect in-memory bytes (%PDF…) BEFORE the path branch — binary PDF
        # bytes contain null bytes, and File.exist? raises ArgumentError
        # ("path name contains null byte") on those, so the bytes check must run first.
        if source.is_a?(String) && source.start_with?('%PDF')
          # in-memory PDF bytes
          buf = source.dup.force_encoding(Encoding::BINARY)
          mem = ::FFI::MemoryPointer.new(:uint8, buf.bytesize)
          mem.write_bytes(buf, 0, buf.bytesize)
          [Bindings.pdf_document_open_from_bytes(mem, buf.bytesize, err), '<in-memory>']
        elsif source.is_a?(String) && File.exist?(source)
          [Bindings.pdf_document_open(File.absolute_path(source), err), File.absolute_path(source)]
        else
          raise FileNotFoundError, "file not found: #{source}"
        end

      code = err.read_int32
      raise_for_code(code, 'open') if code != 0
      raise ParseError, 'pdf_document_open returned null' if handle.nil? || handle.null?

      [path, handle]
    end

    def validate_page_index(idx)
      raise ::PdfOxide::ArgumentError, 'page_index must be >= 0' if idx.negative?

      # Skip page_count check unless we're already open — Java does the
      # range check via IndexOutOfBoundsException at the JNI seam.  Ruby's
      # range check is best-effort to give a clean error before the C call.
    end

    def parse_search_results(results_handle)
      return [] if results_handle.nil? || results_handle.null?

      err = ::FFI::MemoryPointer.new(:int32)
      count = Bindings.pdf_oxide_search_result_count(results_handle)
      out = Array.new(count) do |i|
        page = Bindings.pdf_oxide_search_result_get_page(results_handle, i, err)
        text_ptr = Bindings.pdf_oxide_search_result_get_text(results_handle, i, err)
        text = StringMarshaller.from_c_string(text_ptr) || ''
        x = ::FFI::MemoryPointer.new(:float)
        y = ::FFI::MemoryPointer.new(:float)
        w = ::FFI::MemoryPointer.new(:float)
        h = ::FFI::MemoryPointer.new(:float)
        Bindings.pdf_oxide_search_result_get_bbox(results_handle, i, x, y, w, h, err)
        { page: page,
          text: text,
          bbox: { x: x.read_float, y: y.read_float, width: w.read_float, height: h.read_float } }
      end
      Bindings.pdf_oxide_search_result_free(results_handle)
      out
    end

    def read_rendered_image_bytes(img_ptr)
      # The cdylib renders to a "rendered image" handle.  Different
      # accessors exist across versions; try the byte-buffer accessor
      # first, fall back to a sensible default.
      if Bindings.respond_to?(:pdf_oxide_rendered_image_get_bytes)
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        err = ::FFI::MemoryPointer.new(:int32)
        buf = Bindings.pdf_oxide_rendered_image_get_bytes(img_ptr, len_ptr, err)
        raise_for_code(err.read_int32, 'render_bytes')
        return '' if buf.nil? || buf.null?

        len = len_ptr.read(:size_t)
        bytes = buf.read_string(len)
        Bindings.free_bytes(buf) if Bindings.respond_to?(:free_bytes)
        bytes
      else
        # Fall back to an empty BINARY string; render() callers see a
        # clean error path rather than a segfault when the build is
        # missing the rendered-image accessor.
        ''
      end
    end

    # Map a cdylib error code (`int32_t *err`) to the matching Ruby
    # exception. MUST stay byte-for-byte identical to src/ffi.rs:98-106
    # — the same 9-code surface the PHP, C#, and Go bindings use.
    #
    # Pre-v0.3.55 had alphabetical-natural mapping
    # ({@code 4 => StateError, 5 => PermissionError, 6 =>
    #  UnsupportedFeatureError, 8 => SignatureError, …}) which silently
    # mismapped against the cdylib's wire format — cdylib returned 4
    # (ERR_EXTRACTION) and Ruby raised StateError; returned 8
    # (ERR_UNSUPPORTED) and Ruby raised SignatureError. Same bug C#
    # already fixed in an earlier release; this brings Ruby into
    # line with PHP's ErrorHandler::createException (1-to-1 dispatch).
    def raise_for_code(code, op)
      return if code.zero?

      klass = case code
              when 1 then ::PdfOxide::ArgumentError           # ERR_INVALID_ARG
              when 2 then ::PdfOxide::IoError                 # ERR_IO
              when 3 then ::PdfOxide::ParseError              # ERR_PARSE
              when 4 then ::PdfOxide::ParseError              # ERR_EXTRACTION
              when 5 then ::PdfOxide::InternalError           # ERR_INTERNAL
              when 6 then ::PdfOxide::ArgumentError           # ERR_INVALID_PAGE
              when 7 then ::PdfOxide::SearchError             # ERR_SEARCH
              when 8 then ::PdfOxide::UnsupportedFeatureError # _ERR_UNSUPPORTED
              else ::PdfOxide::InternalError
              end
      raise klass, "#{op} failed (error code #{code})"
    end
  end
end
