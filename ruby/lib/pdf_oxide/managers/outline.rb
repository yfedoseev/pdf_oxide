# frozen_string_literal: true

require_relative 'base'
require 'json'

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

      # v0.3.50: plan a split-by-bookmarks operation.  Returns the
      # decoded JSON plan from `pdf_document_plan_split_by_bookmarks`
      # — typically an array of `{start_page, end_page, title}`
      # records.  The caller then executes the plan (no destructive
      # variant ships in v0.3.55; this is the "preview" surface).
      #
      # @param options [Hash, nil] e.g. `{level: 1}` (top-level only).
      # @return [Array<Hash>, Hash] decoded JSON plan (empty Array if
      #   the document has no bookmarks).
      def plan_split_by_bookmarks(options = nil)
        check_document!
        # The C ABI requires a JSON-shaped string; "{}" is the
        # accepted "defaults" sentinel.  nil yields an Invalid
        # argument error from the Rust side.
        options_json = options.nil? ? '{}' : JSON.generate(options)

        error_ptr = ::FFI::MemoryPointer.new(:int32)
        ptr = FFI::Bindings.pdf_document_plan_split_by_bookmarks(
          @document.handle, options_json, error_ptr
        )
        error_code = error_ptr.read_int32

        if error_code != 0
          raise FFI::ErrorHandler.create_error(
            error_code, 'pdf_document_plan_split_by_bookmarks'
          )
        end
        return [] if ptr.nil? || ptr.null?

        str = FFI::StringMarshaller.from_c_string(ptr)
        return [] if str.nil? || str.empty?

        begin
          decoded = JSON.parse(str)
          decoded.is_a?(Array) || decoded.is_a?(Hash) ? decoded : []
        rescue JSON::ParserError
          []
        end
      end
    end
  end
end
