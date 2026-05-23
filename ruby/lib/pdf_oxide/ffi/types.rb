# frozen_string_literal: true

require 'ffi'

module PdfOxide
  module FFI
    # FFI type definitions and enum mappings
    module Types
      # Error codes enum
      ERROR_CODES = {
        success: 0,
        invalid_arg: 1,
        io_error: 2,
        parse_error: 3,
        not_found: 4,
        permission_denied: 5,
        unsupported: 6,
        internal: 7
      }.freeze

      # Image format enum
      IMAGE_FORMATS = {
        png: 0,
        jpeg: 1,
        webp: 2
      }.freeze

      # Color space enum
      COLOR_SPACES = {
        srgb: 0,
        device_rgb: 1,
        linear_rgb: 2
      }.freeze

      # PDF/A levels
      PDF_A_LEVELS = {
        level_1b: 0,
        level_1a: 1,
        level_2b: 2,
        level_2a: 3,
        level_2u: 4,
        level_3b: 5,
        level_3a: 6,
        level_3u: 7
      }.freeze

      # PDF/X levels
      PDF_X_LEVELS = {
        level_1a_2001: 0,
        level_1a_2003: 1,
        level_3_2003: 2,
        level_4: 3,
        level_5: 4,
        level_6: 5
      }.freeze

      # Barcode formats
      BARCODE_FORMATS = {
        qr_code: 0,
        ean13: 1,
        ean8: 2,
        upc_a: 3,
        upc_e: 4,
        code128: 5,
        code39: 6,
        codabar: 7,
        itf: 8
      }.freeze

      # QR error correction levels
      QR_ERROR_CORRECTION_LEVELS = {
        level_l: 0,
        level_m: 1,
        level_q: 2,
        level_h: 3
      }.freeze

      # Page complexity levels
      PAGE_COMPLEXITY = {
        simple: 0,
        moderate: 1,
        complex: 2,
        very_complex: 3
      }.freeze

      # Content types
      CONTENT_TYPES = {
        text_only: 0,
        text_images: 1,
        tables: 2,
        mixed_layout: 3,
        scanned: 4,
        form: 5,
        vector_graphics: 6
      }.freeze

      # XFA form types
      XFA_FORM_TYPES = {
        static: 0,
        dynamic: 1
      }.freeze

      # XFA field types
      XFA_FIELD_TYPES = {
        text: 0,
        checkbox: 1,
        radio: 2,
        dropdown: 3,
        button: 4,
        signature: 5,
        image: 6,
        datetime: 7,
        numeric: 8,
        password: 9
      }.freeze

      # Signature algorithms
      SIGNATURE_ALGORITHMS = {
        rsa: 0,
        ecdsa: 1
      }.freeze

      # Digest algorithms
      DIGEST_ALGORITHMS = {
        sha1: 0,
        sha256: 1,
        sha384: 2,
        sha512: 3
      }.freeze

      # Convert enum value to string key
      # @param enum_hash [Hash] Enum mapping
      # @param value [Integer] Enum value
      # @return [Symbol] Enum key
      def self.enum_to_key(enum_hash, value)
        enum_hash.invert.fetch(value) { raise "Unknown enum value: #{value}" }
      end

      # Convert enum key to integer value
      # @param enum_hash [Hash] Enum mapping
      # @param key [Symbol, String] Enum key
      # @return [Integer] Enum value
      def self.enum_to_value(enum_hash, key)
        key_sym = key.is_a?(Symbol) ? key : key.to_sym
        enum_hash.fetch(key_sym) { raise "Unknown enum key: #{key}" }
      end

      # Convert error code to exception class.
      # Delegates to the canonical map in {ErrorHandler::ERROR_MAP} so that
      # all 12 codes (argument/io/parse/resource/permission/unsupported/
      # internal/signature/redaction/compliance/accessibility/optimization)
      # resolve to the same exception class everywhere in the gem.
      # @param error_code [Integer] FFI error code
      # @return [Class] Exception class
      def self.error_to_exception(error_code)
        require_relative 'error_handler'
        ErrorHandler::ERROR_MAP.fetch(error_code, ::PdfOxide::Error)
      end
    end
  end
end
