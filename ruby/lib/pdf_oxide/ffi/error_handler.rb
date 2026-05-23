# frozen_string_literal: true

module PdfOxide
  module FFI
    # Handles error checking and conversion for FFI calls
    module ErrorHandler
      # Map error codes to exception classes
      ERROR_MAP = {
        1 => ::PdfOxide::ArgumentError,
        2 => ::PdfOxide::IoError,
        3 => ::PdfOxide::ParseError,
        4 => ::PdfOxide::ResourceError,
        5 => ::PdfOxide::PermissionError,
        6 => ::PdfOxide::UnsupportedFeatureError,
        7 => ::PdfOxide::InternalError,
        8 => ::PdfOxide::SignatureError,
        9 => ::PdfOxide::RedactionError,
        10 => ::PdfOxide::ComplianceError,
        11 => ::PdfOxide::AccessibilityError,
        12 => ::PdfOxide::OptimizationError
      }.freeze

      # Check error code and raise if needed
      # @param error_code [Integer] FFI error code
      # @param operation [String] Operation name for context
      # @param context [Hash] Additional context information
      # @raise [PdfOxide::Error] If error code indicates failure
      def self.check(error_code, operation = nil, **context)
        return if error_code.zero?

        error_class = ERROR_MAP.fetch(error_code, ::PdfOxide::Error)
        error = error_class.new("Operation failed with code #{error_code}")
        error.with_context(operation, **context) if operation
        raise error
      end

      # Execute FFI call with automatic error checking
      # @param operation [String] Operation name
      # @param context [Hash] Context information
      # @yield [error_ptr] Block receiving error pointer
      # @return [Object] Result from block
      # @raise [PdfOxide::Error] If operation fails
      def self.with_error_check(operation = nil, **context)
        raise LocalJumpError, 'Block required' unless block_given?

        error_ptr = ::FFI::MemoryPointer.new(:int32)
        result = yield(error_ptr)
        error_code = error_ptr.read_int32

        check(error_code, operation, **context)
        result
      end

      # Execute FFI call returning boolean result with error checking
      # @param operation [String] Operation name
      # @context [Hash] Context information
      # @yield [error_ptr] Block receiving error pointer
      # @return [Boolean] Result from block
      # @raise [PdfOxide::Error] If operation fails
      def self.with_bool_check(operation = nil, **context)
        with_error_check(operation, **context) { |error_ptr| yield(error_ptr) }
      end

      # Execute FFI call returning integer with error checking
      # @param operation [String] Operation name
      # @param context [Hash] Context information
      # @yield [error_ptr] Block receiving error pointer
      # @return [Integer] Result from block
      # @raise [PdfOxide::Error] If operation fails
      def self.with_int_check(operation = nil, **context)
        with_error_check(operation, **context) { |error_ptr| yield(error_ptr) }
      end

      # Get error message for error code
      # @param error_code [Integer] FFI error code
      # @return [String] Error message
      def self.error_message(error_code)
        case error_code
        when 1
          'Invalid argument'
        when 2
          'I/O error'
        when 3
          'Parse error'
        when 4
          'Resource error'
        when 5
          'Permission denied'
        when 6
          'Feature not supported'
        when 7
          'Internal library error'
        when 8
          'Signature error'
        when 9
          'Redaction error'
        when 10
          'Compliance error'
        when 11
          'Accessibility error'
        when 12
          'Optimization error'
        else
          "Unknown error (code: #{error_code})"
        end
      end

      # Convert error code to exception
      # @param error_code [Integer] FFI error code
      # @param operation [String] Operation name
      # @param context [Hash] Context information
      # @return [PdfOxide::Error] Exception instance
      def self.create_error(error_code, operation = nil, **context)
        error_class = ERROR_MAP.fetch(error_code, ::PdfOxide::Error)
        error_message = error_message(error_code)
        error = error_class.new(error_message)
        error.with_context(operation, **context) if operation
        error
      end
    end
  end
end
