# frozen_string_literal: true

module PdfOxide
  module FFI
    # UTF-8 string round-tripping between Ruby and the C ABI.
    #
    # The cdylib's `*char` returns are heap-allocated by Rust and must
    # be released via `free_string`; passing them to `pdf_free` (the
    # handle deallocator) corrupts the heap.  StringMarshaller hides
    # the distinction from callers.
    module StringMarshaller
      # Encode a Ruby string as UTF-8 for the C ABI.  Returns nil on
      # nil input so callers can pass `nil` through unchanged.
      # @param ruby_string [String, nil]
      # @return [String, nil]
      def self.to_utf8(ruby_string)
        return nil if ruby_string.nil?

        ruby_string.to_s.encode('UTF-8', invalid: :replace, undef: :replace)
      end

      # Read a C string pointer and free the underlying buffer.
      # @param ptr [FFI::Pointer]
      # @param free_after [Boolean] free with `free_string` after reading.
      # @return [String, nil] UTF-8 Ruby string, or nil if the pointer was null.
      def self.from_c_string(ptr, free_after: true)
        return nil if ptr.nil? || ptr.null?

        begin
          ptr.read_string.force_encoding('UTF-8')
        ensure
          free_c_string(ptr) if free_after && !ptr.null?
        end
      end

      # Free a `*char` returned by the cdylib.  Safe on null.
      def self.free_c_string(ptr)
        return if ptr.nil? || ptr.null?
        return unless Bindings.respond_to?(:free_string)

        Bindings.free_string(ptr)
      end
    end
  end
end
