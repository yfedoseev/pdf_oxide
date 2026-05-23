# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a PDF layer (optional content group)
    class Layer
      attr_reader :name, :index, :visible, :printable

      # Initialize layer
      # @param name [String] Layer name
      # @param index [Integer] Layer index
      # @param visible [Boolean] Whether layer is visible
      # @param printable [Boolean] Whether layer is printable
      def initialize(name:, index:, visible: true, printable: true)
        @name = name
        @index = index
        @visible = visible
        @printable = printable
      end

      # Convert to hash
      # @return [Hash] Layer as hash
      def to_h
        {
          name: @name,
          index: @index,
          visible: @visible,
          printable: @printable
        }
      end

      # String representation
      # @return [String] Layer as string
      def to_s
        "Layer(name='#{@name}', index=#{@index}, visible=#{@visible})"
      end

      # Inspect representation
      # @return [String] Layer inspection
      def inspect
        to_s
      end

      # Equality comparison
      # @param other [Object] Other object to compare
      # @return [Boolean] Whether objects are equal
      def ==(other)
        other.is_a?(Layer) &&
          @name == other.name &&
          @index == other.index &&
          @visible == other.visible &&
          @printable == other.printable
      end

      # Hash code
      # @return [Integer] Hash code
      def hash
        [@name, @index, @visible, @printable].hash
      end
    end
  end
end
