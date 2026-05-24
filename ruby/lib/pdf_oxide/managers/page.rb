# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for page manipulation operations
    # Provides methods to insert, delete, rotate, and manipulate PDF pages
    class Page < Base
      # Get page count
      # @return [Integer] Total number of pages
      def count
        check_document!
        @document.page_count
      end

      # Get page info
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Types::PageInfo] Page information
      def get_info(page_index)
        check_document!
        validate_page_index!(page_index)

        width = get_page_width(page_index)
        height = get_page_height(page_index)
        rotation = get_page_rotation(page_index)

        Types::PageInfo.new(
          index: page_index,
          width: width,
          height: height,
          rotation: rotation,
          page_count: count
        )
      end

      # Get page dimensions
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Types::PageDimensions] Page dimensions
      def get_dimensions(page_index)
        check_document!
        validate_page_index!(page_index)

        width = get_page_width(page_index)
        height = get_page_height(page_index)

        Types::PageDimensions.new(width: width, height: height, unit: 'pt')
      end

      # Get page width
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Page width in points
      def get_page_width(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_page_width', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_width(@document.handle, page_index, error_ptr)
        end
      end

      # Get page height
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Page height in points
      def get_page_height(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_page_height', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_height(@document.handle, page_index, error_ptr)
        end
      end

      # Get aspect ratio
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Width/Height ratio
      def get_aspect_ratio(page_index)
        check_document!
        validate_page_index!(page_index)

        height = get_page_height(page_index)
        return 0.0 if height.zero?

        get_page_width(page_index).to_f / height
      end

      # Check if page exists
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Boolean] Whether page exists
      def exists(page_index)
        check_document!
        page_index >= 0 && page_index < count
      end

      # Insert blank page at index
      # @param page_index [Integer] Page index where to insert
      # @return [Boolean] Whether operation succeeded
      def insert_page(page_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Page index must be >= 0' if page_index.negative?
        raise ::PdfOxide::ArgumentError, "Page index #{page_index} exceeds page count" if page_index > count

        with_error_check('insert_page', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_insert_page(@document.handle, page_index, error_ptr)
        end
        true
      end

      # Delete page at index
      # @param page_index [Integer] Page index to delete
      # @return [Boolean] Whether operation succeeded
      def delete_page(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('delete_page', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_delete_page(@document.handle, page_index, error_ptr)
        end
        true
      end

      # Duplicate page
      # @param page_index [Integer] Page index to duplicate
      # @param insert_after [Integer, nil] Index to insert after (default: after original)
      # @return [Boolean] Whether operation succeeded
      def duplicate_page(page_index, insert_after = nil)
        check_document!
        validate_page_index!(page_index)

        insert_after = page_index if insert_after.nil?

        with_error_check('duplicate_page', page: page_index, insert_after: insert_after) do |error_ptr|
          FFI::Bindings.pdf_document_duplicate_page(@document.handle, page_index, insert_after, error_ptr)
        end
        true
      end

      # Move page from one index to another
      # @param from_index [Integer] Source page index
      # @param to_index [Integer] Destination page index
      # @return [Boolean] Whether operation succeeded
      def move_page(from_index, to_index)
        check_document!
        validate_page_index!(from_index)
        validate_page_index!(to_index)

        with_error_check('move_page', from: from_index, to: to_index) do |error_ptr|
          FFI::Bindings.pdf_document_move_page(@document.handle, from_index, to_index, error_ptr)
        end
        true
      end

      # Extract pages as new document
      # @param start_page [Integer] Start page index
      # @param end_page [Integer] End page index (inclusive)
      # @return [String] Path to extracted PDF
      def extract_pages(start_page, end_page)
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        raise ::PdfOxide::ArgumentError, 'Start page must be <= end page' if start_page > end_page

        output_path = "extracted_pages_#{start_page}_#{end_page}.pdf"

        with_error_check('extract_pages', start: start_page, end: end_page) do |error_ptr|
          FFI::Bindings.pdf_document_extract_pages(
            @document.handle,
            FFI::StringMarshaller.to_utf8(output_path),
            error_ptr
          )
        end

        output_path
      end

      # Merge pages from another document
      # @param other_document [Document] Document to merge pages from
      # @param start_page [Integer] Start page index in other document
      # @param end_page [Integer] End page index in other document
      # @return [Boolean] Whether operation succeeded
      def merge_pages(other_document, start_page, end_page)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Other document cannot be nil' if other_document.nil?
        raise ::PdfOxide::ArgumentError, 'Start page must be <= end page' if start_page > end_page

        with_error_check('merge_pages', start: start_page, end: end_page) do |error_ptr|
          FFI::Bindings.pdf_document_merge_pages(
            @document.handle,
            other_document.handle,
            start_page,
            end_page,
            error_ptr
          )
        end
        true
      end

      # Rotate page
      # @param page_index [Integer] Page index
      # @param degrees [Integer] Rotation in degrees (0, 90, 180, 270)
      # @return [Boolean] Whether operation succeeded
      def rotate_page(page_index, degrees)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Rotation must be 0, 90, 180, or 270' unless [0, 90, 180, 270].include?(degrees)

        with_error_check('rotate_page', page: page_index, degrees: degrees) do |error_ptr|
          FFI::Bindings.pdf_document_set_page_rotation(@document.handle, page_index, degrees, error_ptr)
        end
        true
      end

      # Get page rotation
      # @param page_index [Integer] Page index
      # @return [Integer] Rotation in degrees (0, 90, 180, 270)
      def get_page_rotation(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_page_rotation', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_rotation(@document.handle, page_index, error_ptr)
        end
      end

      # Get media box (page boundary)
      # @param page_index [Integer] Page index
      # @return [Types::BoundingBox] Media box coordinates
      def get_media_box(page_index)
        check_document!
        validate_page_index!(page_index)

        bbox_ptr = ::FFI::MemoryPointer.new(:float, 4)
        with_error_check('get_media_box', page: page_index) do |_error_ptr|
          FFI::Bindings.pdf_document_get_media_box(@document.handle, page_index, bbox_ptr)
        end

        x, y, width, height = bbox_ptr.read_array_of_float(4)
        Types::BoundingBox.new(x: x, y: y, width: width, height: height)
      end

      # Get crop box (visible area)
      # @param page_index [Integer] Page index
      # @return [Types::BoundingBox] Crop box coordinates
      def get_crop_box(page_index)
        check_document!
        validate_page_index!(page_index)

        bbox_ptr = ::FFI::MemoryPointer.new(:float, 4)
        with_error_check('get_crop_box', page: page_index) do |_error_ptr|
          FFI::Bindings.pdf_document_get_crop_box(@document.handle, page_index, bbox_ptr)
        end

        x, y, width, height = bbox_ptr.read_array_of_float(4)
        Types::BoundingBox.new(x: x, y: y, width: width, height: height)
      end

      # Set crop box
      # @param page_index [Integer] Page index
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @return [Boolean] Whether operation succeeded
      def set_crop_box(page_index, x, y, width, height)
        check_document!
        validate_page_index!(page_index)

        with_error_check('set_crop_box', page: page_index, box: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_set_crop_box(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            error_ptr
          )
        end
        true
      end
    end
  end
end
