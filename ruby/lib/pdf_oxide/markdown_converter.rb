# frozen_string_literal: true

module PdfOxide
  # Static converters from a {PdfDocument} to Markdown or HTML.
  #
  # Mirrors `fyi.oxide.pdf.MarkdownConverter`.  Stateless — every
  # method takes the document handle as an argument.  Per-page and
  # whole-document variants are offered for both Markdown and HTML.
  module MarkdownConverter
    module_function

    # Convert a page (or the whole document) to Markdown.
    # @param doc [PdfDocument]
    # @param page_index [Integer, nil] when nil, converts the whole doc.
    # @return [String] Markdown.
    def to_markdown(doc, page_index = nil)
      raise ::PdfOxide::ArgumentError, 'doc cannot be nil' if doc.nil?

      err = ::FFI::MemoryPointer.new(:int32)
      ptr =
        if page_index.nil?
          Bindings.pdf_document_to_markdown_all(doc.handle, err)
        else
          Bindings.pdf_document_to_markdown(doc.handle, page_index, err)
        end
      code = err.read_int32
      raise InternalError, "to_markdown failed (#{code})" if code != 0

      StringMarshaller.from_c_string(ptr) || ''
    end

    # Convert a page (or the whole document) to HTML.
    # @param doc [PdfDocument]
    # @param page_index [Integer, nil] when nil, converts the whole doc.
    # @return [String] HTML.
    def to_html(doc, page_index = nil)
      raise ::PdfOxide::ArgumentError, 'doc cannot be nil' if doc.nil?

      err = ::FFI::MemoryPointer.new(:int32)
      ptr =
        if page_index.nil?
          Bindings.pdf_document_to_html_all(doc.handle, err)
        else
          Bindings.pdf_document_to_html(doc.handle, page_index, err)
        end
      code = err.read_int32
      raise InternalError, "to_html failed (#{code})" if code != 0

      StringMarshaller.from_c_string(ptr) || ''
    end
  end
end
