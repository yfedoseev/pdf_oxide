# frozen_string_literal: true

module PdfOxide
  # v0.3.48 #159 — Office-to-PDF in-memory converter.
  #
  # Opens a DOCX/PPTX/XLSX byte blob as a PdfDocument handle (the
  # Office file becomes a PDF on the way in).  Mirrors PHP's
  # `OfficeConverter` and Java's `OfficeConverter` shape.
  #
  # @example
  #   bytes = File.binread('report.docx')
  #   doc   = PdfOxide::OfficeConverter.from_docx(bytes)
  #   puts doc.page_count
  module OfficeConverter
    module_function

    # Convert DOCX bytes to a PDF Document.
    # @param bytes [String] raw DOCX file bytes.
    # @return [PdfOxide::Document]
    def from_docx(bytes)
      from_bytes(bytes, :docx)
    end

    # Convert PPTX bytes to a PDF Document.
    # @param bytes [String] raw PPTX file bytes.
    # @return [PdfOxide::Document]
    def from_pptx(bytes)
      from_bytes(bytes, :pptx)
    end

    # Convert XLSX bytes to a PDF Document.
    # @param bytes [String] raw XLSX file bytes.
    # @return [PdfOxide::Document]
    def from_xlsx(bytes)
      from_bytes(bytes, :xlsx)
    end

    # Convert a file on disk by inferring format from extension.
    # @param path [String]
    # @return [PdfOxide::Document]
    def from_file(path)
      raise ::PdfOxide::FileNotFoundError, "file not found: #{path}" unless File.exist?(path)

      ext = File.extname(path).downcase.delete('.')
      bytes = File.binread(path)
      case ext
      when 'docx' then from_docx(bytes)
      when 'pptx' then from_pptx(bytes)
      when 'xlsx' then from_xlsx(bytes)
      else
        raise ::PdfOxide::ArgumentError,
              "unsupported office extension: .#{ext} (want .docx/.pptx/.xlsx)"
      end
    end

    class << self
      private

      def from_bytes(bytes, kind)
        if bytes.nil? || bytes.empty?
          raise ::PdfOxide::ArgumentError, 'bytes cannot be nil/empty'
        end

        binary = bytes.dup.force_encoding(Encoding::BINARY)
        buf = ::FFI::MemoryPointer.new(:uint8, binary.bytesize)
        buf.write_bytes(binary, 0, binary.bytesize)

        error_ptr = ::FFI::MemoryPointer.new(:int32)
        sym = case kind
              when :docx then :pdf_document_open_from_docx_bytes
              when :pptx then :pdf_document_open_from_pptx_bytes
              when :xlsx then :pdf_document_open_from_xlsx_bytes
              else raise ::PdfOxide::ArgumentError, "unknown kind: #{kind}"
              end

        handle = FFI::Bindings.send(sym, buf, binary.bytesize, error_ptr)
        error_code = error_ptr.read_int32

        if error_code != 0 || handle.nil? || handle.null?
          raise FFI::ErrorHandler.create_error(
            error_code, "office_converter_from_#{kind}",
            byte_count: binary.bytesize
          )
        end

        # Wrap the raw handle in a Document instance without going
        # through Document#initialize (which expects a file path).
        wrap_handle(handle, kind: kind)
      end

      def wrap_handle(handle, kind:)
        doc = ::PdfOxide::Document.allocate
        doc.instance_variable_set(:@handle, handle)
        doc.instance_variable_set(:@path, "<#{kind}-in-memory>")
        doc.instance_variable_set(:@closed, false)
        doc.instance_variable_set(:@managers, {})
        # Same tracker indirection as Document#initialize so that an
        # explicit `close` defuses the finalizer (prevents
        # double-free on the cdylib's `pdf_document_free`).
        tracker = [handle]
        doc.instance_variable_set(:@tracker, tracker)
        ObjectSpace.define_finalizer(doc, ::PdfOxide::Document.finalizer(tracker))
        doc
      end
    end
  end
end
