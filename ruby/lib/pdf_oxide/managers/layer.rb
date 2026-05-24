# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for PDF layer/OCG (Optional Content Group) operations
    # Provides access to and manipulation of document layers
    class Layer < Base
      # Check if document has layers
      # @return [Boolean] Whether document has layers
      def has_layers?
        check_document!
        FFI::Bindings.pdf_document_has_layers(@document.handle)
      end

      # Get count of layers
      # @return [Integer] Number of layers
      def layer_count
        check_document!
        return 0 unless has_layers?

        with_error_check('layer_count') do |error_ptr|
          FFI::Bindings.pdf_document_get_layer_count(@document.handle, error_ptr)
        end
      end

      # Get layer name at index
      # @param index [Integer] Layer index (0-indexed)
      # @return [String, nil] Layer name
      def get_layer_name(index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Layer index must be >= 0' if index.negative?
        raise ::PdfOxide::ArgumentError, "Layer index #{index} exceeds layer count" if index >= layer_count

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_layer_name', index: index) do |error_ptr|
            FFI::Bindings.pdf_document_get_layer_name(@document.handle, index, error_ptr)
          end
        )
      end

      # Get all layers
      # @return [Array<Types::Layer>] Array of all layers
      def get_all
        check_document!
        return [] unless has_layers?

        count = layer_count
        count.times.map do |i|
          Types::Layer.new(
            name: get_layer_name(i),
            index: i,
            visible: true,
            printable: true
          )
        end
      end

      # Get layer by index
      # @param index [Integer] Layer index
      # @return [Types::Layer] Layer at index
      def get_layer(index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Layer index must be >= 0' if index.negative?
        raise ::PdfOxide::ArgumentError, "Layer index #{index} exceeds layer count" if index >= layer_count

        Types::Layer.new(
          name: get_layer_name(index),
          index: index,
          visible: true,
          printable: true
        )
      end

      # Get layer by name
      # @param name [String] Layer name
      # @return [Types::Layer, nil] Layer with given name
      def get_layer_by_name(name)
        check_document!
        layer = get_all.find { |l| l.name == name }
        layer || raise(::PdfOxide::NotFoundError, "Layer '#{name}' not found")
      end

      # Convert layers to array of hashes
      # @return [Array<Hash>] Layers as array of hashes
      def to_array
        get_all.map(&:to_h)
      end

      # Get layer names
      # @return [Array<String>] Array of layer names
      def layer_names
        check_document!
        get_all.map(&:name)
      end

      # Get layers info
      # @return [Hash] Layer information
      def info
        check_document!
        {
          has_layers: has_layers?,
          layer_count: layer_count,
          layers: to_array
        }
      end

      # List all layers as formatted string
      # @return [String] Formatted layer list
      def list
        check_document!
        return "No layers found\n" unless has_layers?

        output = "Document Layers\n"
        output += "#{'=' * 40}\n\n"

        get_all.each do |layer|
          vis = layer.visible ? '✓' : '✗'
          pri = layer.printable ? '✓' : '✗'
          output += "[#{layer.index}] #{layer.name} (visible: #{vis}, printable: #{pri})\n"
        end

        output
      end
    end
  end
end
