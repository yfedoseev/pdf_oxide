# frozen_string_literal: true

module PdfOxide
  # Base error class for all PDF Oxide exceptions
  class Error < StandardError
    attr_reader :code, :details

    # Initialize error with code and details
    # @param code [String] Error code
    # @param message [String] Human-readable error message
    # @param details [Hash] Additional error details
    def initialize(code = nil, message = nil, details = {})
      @code = code
      @details = details.dup
      @details[:timestamp] = Time.now.iso8601

      super(message || error_message)
    end

    # Add context information to error
    # @param operation [String] Operation that failed
    # @param context [Hash] Additional context
    # @return [self]
    def with_context(operation, **context)
      @details[:operation] = operation
      @details[:context] = context
      self
    end

    # @return [Hash] Error as hash representation
    def to_h
      {
        class: self.class.name,
        code: @code,
        message: message,
        details: @details
      }
    end

    private

    def error_message
      return message if @code.nil?
      "[#{@code}] #{message}"
    end
  end

  # Raised when platform is not supported
  class UnsupportedPlatformError < Error
    def initialize(message = 'Unsupported platform')
      super('PLATFORM_ERROR', message)
    end
  end

  # Parsing and structure errors
  class ParseError < Error
    def initialize(message = 'Failed to parse PDF', details = {})
      super('PARSE_ERROR', message, details)
    end
  end

  class InvalidStructureError < ParseError
    def initialize(message = 'Invalid PDF structure', details = {})
      super(message, details)
    end
  end

  class CorruptedPdfError < ParseError
    def initialize(message = 'PDF is corrupted', details = {})
      super(message, details)
    end
  end

  class InvalidVersionError < ParseError
    def initialize(message = 'Unsupported PDF version', details = {})
      super(message, details)
    end
  end

  # I/O and file-related errors
  class IoError < Error
    def initialize(message = 'I/O operation failed', details = {})
      super('IO_ERROR', message, details)
    end
  end

  class FileNotFoundError < IoError
    def initialize(message = 'File not found', details = {})
      super(message, details)
    end
  end

  class FileAccessError < IoError
    def initialize(message = 'Cannot access file', details = {})
      super(message, details)
    end
  end

  class WriterError < IoError
    def initialize(message = 'Failed to write PDF', details = {})
      super(message, details)
    end
  end

  # Encryption and security errors
  class EncryptionError < Error
    def initialize(message = 'Encryption operation failed', details = {})
      super('ENCRYPTION_ERROR', message, details)
    end
  end

  # Raised when a permission-protected operation (encryption, signing,
  # redaction, owner-password) is denied.  Sibling of EncryptionError so
  # callers can distinguish "wrong-password decryption" from "lacked
  # permission to sign/redact".
  class PermissionError < Error
    def initialize(message = 'Permission denied', details = {})
      super('PERMISSION_DENIED', message, details)
    end
  end

  # Encoding/decoding failures encountered while marshalling UTF-8 between
  # Ruby and the C string interface.
  class EncodingError < Error
    def initialize(message = 'String encoding failure', details = {})
      super('ENCODING_ERROR', message, details)
    end
  end

  # Raised when a C buffer-fill helper reports that the caller-supplied
  # buffer was too small (the C side returns the required size; the
  # wrapper should resize and retry, or surface this error to the caller).
  class BufferOverflowError < Error
    def initialize(message = 'Output buffer too small', details = {})
      super('BUFFER_OVERFLOW', message, details)
    end
  end

  # Raised when the OCR subsystem fails (engine unavailable, model missing,
  # GPU device error, decode timeout).
  class OcrError < Error
    def initialize(message = 'OCR operation failed', details = {})
      super('OCR_ERROR', message, details)
    end
  end

  # Feature support errors
  class UnsupportedFeatureError < Error
    def initialize(message = 'Feature not supported', details = {})
      super('UNSUPPORTED_FEATURE', message, details)
    end
  end

  # Resource management errors
  class ResourceError < Error
    def initialize(message = 'Resource error', details = {})
      super('RESOURCE_ERROR', message, details)
    end
  end

  class StateError < ResourceError
    def initialize(message = 'Invalid state', details = {})
      super(message, details)
    end
  end

  # Internal library errors
  class InternalError < Error
    def initialize(message = 'Internal library error', details = {})
      super('INTERNAL_ERROR', message, details)
    end
  end

  # Argument validation errors
  class ArgumentError < Error
    def initialize(message = 'Invalid argument', details = {})
      super('ARGUMENT_ERROR', message, details)
    end
  end

  # Digital signature errors
  class SignatureError < Error
    def initialize(message = 'Signature operation failed', details = {})
      super('SIGNATURE_ERROR', message, details)
    end
  end

  # Redaction errors
  class RedactionError < Error
    def initialize(message = 'Redaction operation failed', details = {})
      super('REDACTION_ERROR', message, details)
    end
  end

  # Compliance errors
  class ComplianceError < Error
    def initialize(message = 'Compliance operation failed', details = {})
      super('COMPLIANCE_ERROR', message, details)
    end
  end

  # Accessibility errors
  class AccessibilityError < Error
    def initialize(message = 'Accessibility operation failed', details = {})
      super('ACCESSIBILITY_ERROR', message, details)
    end
  end

  # Optimization errors
  class OptimizationError < Error
    def initialize(message = 'Optimization operation failed', details = {})
      super('OPTIMIZATION_ERROR', message, details)
    end
  end
end
