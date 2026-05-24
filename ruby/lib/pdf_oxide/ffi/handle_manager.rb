# frozen_string_literal: true

module PdfOxide
  module FFI
    # Manages lifetime and tracking of FFI handles
    class HandleManager
      def initialize
        @handles = {}
        @mutex = Mutex.new
      end

      # Register a handle with automatic cleanup
      # @param handle [FFI::Pointer] Native handle
      # @param cleanup_block [Proc] Block to call on cleanup
      # @return [FFI::Pointer] The handle
      def register(handle, &cleanup_block)
        return handle if handle.nil? || handle.null?

        @mutex.synchronize do
          handle_id = handle.address
          @handles[handle_id] = cleanup_block if cleanup_block
        end

        # Register finalizer for automatic cleanup
        ObjectSpace.define_finalizer(handle, finalizer_for(handle))

        handle
      end

      # Unregister a handle and call cleanup
      # @param handle [FFI::Pointer] Native handle
      # @return [void]
      def unregister(handle)
        return if handle.nil? || handle.null?

        @mutex.synchronize do
          handle_id = handle.address
          cleanup_block = @handles.delete(handle_id)
          cleanup_block&.call
        end
      end

      # Check if handle is still valid
      # @param handle [FFI::Pointer] Native handle
      # @return [Boolean] Whether handle is valid
      def valid?(handle)
        return false if handle.nil? || handle.null?

        @mutex.synchronize { @handles.key?(handle.address) }
      end

      # Get cleanup block for a handle
      # @param handle [FFI::Pointer] Native handle
      # @return [Proc, nil] Cleanup block or nil
      def cleanup_for(handle)
        return nil if handle.nil? || handle.null?

        @mutex.synchronize { @handles[handle.address] }
      end

      private

      def finalizer_for(handle)
        handle_address = handle.address
        proc do
          @mutex.synchronize do
            cleanup_block = @handles.delete(handle_address)
            cleanup_block&.call(handle)
          end
        end
      end
    end

    # Global handle manager instance
    @handle_manager = HandleManager.new

    # Get global handle manager
    # @return [HandleManager] Global handle manager
    def self.handle_manager
      @handle_manager
    end

    # Register a handle
    # @param handle [FFI::Pointer] Native handle
    # @param cleanup_block [Proc] Block to call on cleanup
    # @return [FFI::Pointer] The handle
    def self.register_handle(handle, &cleanup_block)
      handle_manager.register(handle, &cleanup_block)
    end

    # Unregister a handle
    # @param handle [FFI::Pointer] Native handle
    # @return [void]
    def self.unregister_handle(handle)
      handle_manager.unregister(handle)
    end
  end
end
