# frozen_string_literal: true

module PdfOxide
  # Main interface for reading and analyzing PDF documents
  class Document
    attr_reader :handle, :path

    # Open a PDF document
    # @param path [String] Path to PDF file
    # @yield [doc] Block to execute with document (auto-closes)
    # @return [Document, Object] Document instance or block result
    def initialize(path)
      raise ArgumentError, "Path cannot be nil" if path.nil?
      raise FileNotFoundError, "File not found: #{path}" unless File.exist?(path)

      @path = File.absolute_path(path)
      @closed = false
      @managers = {}

      # Open document via FFI
      @handle = FFI::ErrorHandler.with_error_check('open_document', path: @path) do |error_ptr|
        FFI::Bindings.pdf_document_open(@path, error_ptr)
      end

      # Register cleanup finalizer
      ObjectSpace.define_finalizer(self, self.class.finalizer(@handle))
    end

    # Open a PDF document with optional block
    # @param path [String] Path to PDF file
    # @yield [doc] Block to execute with document
    # @return [Document, Object] Document instance or block result
    def self.open(path)
      doc = new(path)
      if block_given?
        begin
          yield doc
        ensure
          doc.close
        end
      else
        doc
      end
    end

    # Get number of pages
    # @return [Integer] Page count
    def page_count
      check_open!
      FFI::ErrorHandler.with_error_check('page_count') do |error_ptr|
        FFI::Bindings.pdf_document_get_page_count(@handle, error_ptr)
      end
    end

    # Get PDF version
    # @return [String] Version string
    def version
      check_open!
      FFI::StringMarshaller.from_c_string(
        FFI::ErrorHandler.with_error_check('get_version') do |error_ptr|
          FFI::Bindings.pdf_document_get_version(@handle, error_ptr)
        end
      )
    end

    # Check if document is encrypted
    # @return [Boolean]
    def encrypted?
      check_open!
      FFI::ErrorHandler.with_error_check('is_encrypted') do |error_ptr|
        FFI::Bindings.pdf_document_is_encrypted(@handle, error_ptr)
      end
    end

    # Unlock document with password
    # @param password [String] Document password
    # @return [Boolean] Success
    def unlock(password)
      check_open!
      FFI::Bindings.pdf_document_unlock_with_password(
        @handle,
        FFI::StringMarshaller.to_utf8(password),
        FFI::MemoryPointer.new(:int32)
      )
    end

    # Get file size
    # @return [Integer] File size in bytes
    def file_size
      check_open!
      FFI::ErrorHandler.with_error_check('file_size') do |error_ptr|
        FFI::Bindings.pdf_document_get_file_size(@handle, error_ptr)
      end
    end

    # ============================================================
    # Manager Access (Lazy Initialization)
    # ============================================================

    # Access search manager
    # @return [Managers::Search]
    def search
      @managers[:search] ||= Managers::Search.new(self)
    end

    # Access rendering manager
    # @return [Managers::Rendering]
    def rendering
      @managers[:rendering] ||= Managers::Rendering.new(self)
    end

    # Access annotation manager
    # @return [Managers::Annotation]
    def annotations
      @managers[:annotations] ||= Managers::Annotation.new(self)
    end

    # Access form manager
    # @return [Managers::Form]
    def forms
      @managers[:forms] ||= Managers::Form.new(self)
    end

    # Access page manager
    # @return [Managers::Page]
    def pages
      @managers[:pages] ||= Managers::Page.new(self)
    end

    # Access metadata manager
    # @return [Managers::Metadata]
    def metadata
      @managers[:metadata] ||= Managers::Metadata.new(self)
    end

    # Access outline manager
    # @return [Managers::Outline]
    def outline
      @managers[:outline] ||= Managers::Outline.new(self)
    end

    # Access layer manager
    # @return [Managers::Layer]
    def layers
      @managers[:layers] ||= Managers::Layer.new(self)
    end

    # Access cache manager
    # @return [Managers::Cache]
    def cache
      @managers[:cache] ||= Managers::Cache.new(self)
    end

    # Access extraction manager
    # @return [Managers::Extraction]
    def extraction
      @managers[:extraction] ||= Managers::Extraction.new(self)
    end

    # Access OCR manager
    # @return [Managers::Ocr]
    def ocr
      @managers[:ocr] ||= Managers::Ocr.new(self)
    end

    # Access compliance manager
    # @return [Managers::Compliance]
    def compliance
      @managers[:compliance] ||= Managers::Compliance.new(self)
    end

    # Access signature manager
    # @return [Managers::Signature]
    def signatures
      @managers[:signatures] ||= Managers::Signature.new(self)
    end

    # Access barcode manager
    # @return [Managers::Barcode]
    def barcodes
      @managers[:barcodes] ||= Managers::Barcode.new(self)
    end

    # Access analysis manager
    # @return [Managers::Analysis]
    def analysis
      @managers[:analysis] ||= Managers::Analysis.new(self)
    end

    # Access XFA manager
    # @return [Managers::Xfa]
    def xfa
      @managers[:xfa] ||= Managers::Xfa.new(self)
    end

    # ============================================================
    # Resource Management
    # ============================================================

    # Close document and free resources
    # @return [void]
    def close
      return if @closed
      FFI::Bindings.pdf_document_free(@handle) unless @handle.nil?
      @closed = true
      @handle = nil
    end

    # Check if document is closed
    # @return [Boolean]
    def closed?
      @closed
    end

    # Finalizer for GC cleanup
    # @param handle [FFI::Pointer] Document handle
    # @return [Proc]
    def self.finalizer(handle)
      proc { FFI::Bindings.pdf_document_free(handle) unless handle.nil? || handle.null? }
    end

    private

    def check_open!
      raise StateError, 'Document has been closed' if @closed
    end
  end
end
