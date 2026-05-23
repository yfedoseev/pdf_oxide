# frozen_string_literal: true

module PdfOxide
  module Managers
    # Base class for all PDF managers with common functionality
    class Base
      attr_reader :document

      # Initialize manager
      # @param document [Document] PDF document instance
      def initialize(document)
        raise ::PdfOxide::ArgumentError, 'Document cannot be nil' if document.nil?
        @document = document
        @closed = false
      end

      # Clear all caches
      # @return [void]
      def clear_cache
        return if @closed
        check_document!
        FFI::Bindings.pdf_cache_clear(@document.handle)
      end

      # Invalidate cache for specific page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [void]
      def invalidate_page(page_index)
        return if @closed
        check_document!
        validate_page_index!(page_index)
        FFI::Bindings.pdf_cache_invalidate_page(@document.handle, page_index)
      end

      # Get cache statistics
      # @return [Hash] Cache statistics
      def cache_statistics
        return {} if @closed
        check_document!
        # Implementation would parse FFI struct
        {}
      end

      protected

      # Validate page index
      # @param page_index [Integer] Page index to validate
      # @raise [ArgumentError] If index invalid
      def validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Page index must be >= 0' if page_index < 0
        if page_index >= @document.page_count
          raise ::PdfOxide::ArgumentError, "Page index #{page_index} exceeds page count (#{@document.page_count})"
        end
      end

      # Check that document is still open
      # @raise [StateError] If document closed
      def check_document!
        if @document.nil? || @document.closed?
          raise ::PdfOxide::StateError, 'Document has been closed'
        end
      end

      # Convert error code to exception and raise
      # @param error_code [Integer] FFI error code
      # @param operation [String] Operation name for context
      # @raise [PdfOxide::Error] Appropriate error type
      def handle_error(error_code, operation = nil, **context)
        FFI::ErrorHandler.check(error_code, operation, **context)
      end

      # Execute operation with error checking
      # @param operation [String] Operation name
      # @yield [error_ptr] Block to execute with error pointer
      # @return [Object] Result from block
      def with_error_check(operation = nil, **context)
        FFI::ErrorHandler.with_error_check(operation, **context) { |error_ptr| yield(error_ptr) }
      end

      # Close manager resources
      # @return [void]
      def close
        @closed = true
      end

      # Check if manager is closed
      # @return [Boolean]
      def closed?
        @closed
      end
    end
  end
end
