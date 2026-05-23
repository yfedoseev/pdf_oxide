# frozen_string_literal: true

module PdfOxide
  module FFI
    # Handles UTF-8 string conversion between Ruby and C
    module StringMarshaller
      # Convert Ruby string to C-compatible UTF-8 string
      # @param ruby_string [String, nil] Ruby string to convert
      # @return [String, nil] UTF-8 encoded string
      def self.to_utf8(ruby_string)
        return nil if ruby_string.nil?

        ruby_string.to_s.encode('UTF-8', invalid: :replace, undef: :replace)
      rescue Encoding::Error => e
        raise ::PdfOxide::Error.new('ENCODING_ERROR', "Failed to encode string to UTF-8: #{e.message}")
      end

      # Convert C string pointer to Ruby UTF-8 string
      # @param c_string_ptr [FFI::Pointer] Pointer to C string
      # @param free_after [Boolean] Whether to free the pointer after reading
      # @return [String, nil] UTF-8 Ruby string
      def self.from_c_string(c_string_ptr, free_after: true)
        return nil if c_string_ptr.nil? || c_string_ptr.null?

        begin
          ruby_string = c_string_ptr.read_string.force_encoding('UTF-8')
          ruby_string
        rescue Encoding::Error => e
          raise ::PdfOxide::Error.new('DECODING_ERROR', "Failed to decode C string: #{e.message}")
        ensure
          free_c_string(c_string_ptr) if free_after && !c_string_ptr.null?
        end
      end

      # Read string from pointer without freeing
      # @param c_string_ptr [FFI::Pointer] Pointer to C string
      # @return [String, nil] UTF-8 Ruby string
      def self.read_c_string(c_string_ptr)
        from_c_string(c_string_ptr, free_after: false)
      end

      # Free C string allocated by Rust.
      # The cdylib exports `pdf_free` and `free_string`; both accept a pointer
      # returned by the C-ABI string helpers.  Prefer `pdf_free` (the general
      # heap-deallocator) when available; fall back to `free_string`.
      # @param c_string_ptr [FFI::Pointer] Pointer to C string
      def self.free_c_string(c_string_ptr)
        return if c_string_ptr.nil? || c_string_ptr.null?

        if Bindings.respond_to?(:pdf_free)
          Bindings.pdf_free(c_string_ptr)
        elsif Bindings.respond_to?(:free_string)
          Bindings.free_string(c_string_ptr)
        end
      end

      # Validate UTF-8 string
      # @param string [String] String to validate
      # @return [Boolean] Whether string is valid UTF-8
      def self.valid_utf8?(string)
        string.to_s.valid_encoding?
      rescue StandardError
        false
      end

      # Convert array of C strings to Ruby strings
      # @param ptr_array [Array<FFI::Pointer>] Array of C string pointers
      # @param free_after [Boolean] Whether to free pointers after reading
      # @return [Array<String>] Array of Ruby strings
      def self.from_c_string_array(ptr_array, free_after: true)
        ptr_array.map { |ptr| from_c_string(ptr, free_after: free_after) }
      end

      # Sanitize string for C interop
      # @param string [String] String to sanitize
      # @return [String] Sanitized string safe for C
      def self.sanitize(string)
        string = string.to_s
        string.encode('UTF-8', invalid: :replace, undef: :replace)
      rescue Encoding::Error
        string.force_encoding('UTF-8')
      end
    end
  end
end
