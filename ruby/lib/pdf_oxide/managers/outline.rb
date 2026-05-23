# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for PDF outline/bookmark operations
    # Provides access to document bookmarks and table of contents
    class Outline < Base
      # Check if document has outlines
      # @return [Boolean] Whether document has outlines
      def has_outlines?
        check_document!
        FFI::Bindings.pdf_document_has_outlines(@document.handle)
      end

      # Get count of outlines
      # @return [Integer] Number of outline entries
      def outline_count
        check_document!
        return 0 unless has_outlines?

        with_error_check('outline_count') do |error_ptr|
          FFI::Bindings.pdf_document_get_outline_count(@document.handle, error_ptr)
        end
      end

      # Get outline title at index
      # @param index [Integer] Outline index (0-indexed)
      # @return [String, nil] Outline title
      def get_outline_title(index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Outline index must be >= 0' if index < 0
        raise ::PdfOxide::ArgumentError, "Outline index #{index} exceeds outline count" if index >= outline_count

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_outline_title', index: index) do |error_ptr|
            FFI::Bindings.pdf_document_get_outline_title(@document.handle, index, error_ptr)
          end
        )
      end

      # Get destination page for outline at index
      # @param index [Integer] Outline index (0-indexed)
      # @return [Integer] Destination page index
      def get_outline_dest_page(index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Outline index must be >= 0' if index < 0
        raise ::PdfOxide::ArgumentError, "Outline index #{index} exceeds outline count" if index >= outline_count

        with_error_check('get_outline_dest_page', index: index) do |error_ptr|
          FFI::Bindings.pdf_document_get_outline_dest_page(@document.handle, index, error_ptr)
        end
      end

      # Get all outlines
      # @return [Array<Types::Outline>] Array of all outline entries
      def get_all
        check_document!
        return [] unless has_outlines?

        count = outline_count
        count.times.map do |i|
          Types::Outline.new(
            title: get_outline_title(i),
            dest_page: get_outline_dest_page(i),
            level: 0
          )
        end
      end

      # Convert outlines to array of hashes
      # @return [Array<Hash>] Outlines as array of hashes
      def to_array
        get_all.map(&:to_h)
      end

      # Build table of contents
      # @return [String] Formatted table of contents
      def build_toc
        check_document!
        return '' unless has_outlines?

        toc = "Table of Contents\n"
        toc += "=" * 40 + "\n\n"

        get_all.each do |outline|
          indent = '  ' * outline.level
          toc += "#{indent}#{outline.title} (page #{outline.dest_page + 1})\n"
        end

        toc
      end
    end
  end
end
