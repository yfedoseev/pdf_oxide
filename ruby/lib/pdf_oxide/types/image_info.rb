# frozen_string_literal: true

module PdfOxide
  module Types
    # Information about an image in a PDF
    class ImageInfo
      attr_reader :name, :width, :height, :color_space, :bits_per_component, :compression

      # Initialize image info
      # @param name [String] Image name
      # @param width [Integer] Image width
      # @param height [Integer] Image height
      # @param color_space [String] Color space
      # @param bits_per_component [Integer] Bits per component
      # @param compression [String] Compression method
      def initialize(
        name:,
        width:,
        height:,
        color_space: nil,
        bits_per_component: 8,
        compression: nil
      )
        @name = name.to_s
        @width = width.to_i
        @height = height.to_i
        @color_space = color_space
        @bits_per_component = bits_per_component.to_i
        @compression = compression
      end

      # Get image dimensions as string
      # @return [String] Dimensions string
      def dimensions
        "#{@width}x#{@height}"
      end

      # Calculate total pixel count
      # @return [Integer] Pixel count
      def pixel_count
        @width * @height
      end

      # Estimate file size in bytes (rough estimate)
      # @return [Integer] Estimated size
      def estimated_size_bytes
        (pixel_count * @bits_per_component) / 8
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          name: @name,
          width: @width,
          height: @height,
          color_space: @color_space,
          bits_per_component: @bits_per_component,
          compression: @compression,
          dimensions: dimensions
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "ImageInfo(#{@name}, #{dimensions})"
      end

      # Check equality
      # @param other [ImageInfo] Other image info
      # @return [Boolean] Whether equal
      def ==(other)
        return false unless other.is_a?(ImageInfo)

        @name == other.name && @width == other.width && @height == other.height
      end

      alias eql? ==
    end
  end
end
