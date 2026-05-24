# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a bounding box with coordinates and dimensions
    class BoundingBox
      attr_reader :x, :y, :width, :height

      # Initialize bounding box
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param width [Float] Box width
      # @param height [Float] Box height
      def initialize(x:, y:, width:, height:)
        @x = x.to_f
        @y = y.to_f
        @width = width.to_f
        @height = height.to_f
      end

      # Get right coordinate
      # @return [Float] Right coordinate
      def right
        @x + @width
      end

      # Get bottom coordinate
      # @return [Float] Bottom coordinate
      def bottom
        @y + @height
      end

      # Get area of bounding box
      # @return [Float] Area in square units
      def area
        @width * @height
      end

      # Check if point is inside bounding box
      # @param px [Float] Point X coordinate
      # @param py [Float] Point Y coordinate
      # @return [Boolean] Whether point is inside
      def contains_point?(px, py)
        px >= @x && px <= right && py >= @y && py <= bottom
      end

      # Check if bounding box overlaps with another
      # @param other [BoundingBox] Other bounding box
      # @return [Boolean] Whether boxes overlap
      def overlaps_with?(other)
        @x < other.right && right > other.x &&
          @y < other.bottom && bottom > other.y
      end

      # Expand bounding box by margins
      # @param margin [Float] Margin to expand
      # @return [BoundingBox] New expanded bounding box
      def expand(margin)
        BoundingBox.new(
          x: @x - margin,
          y: @y - margin,
          width: @width + 2 * margin,
          height: @height + 2 * margin
        )
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        { x: @x, y: @y, width: @width, height: @height }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "BoundingBox(#{@x}, #{@y}, #{@width}x#{@height})"
      end

      # Check equality
      # @param other [BoundingBox] Other bounding box
      # @return [Boolean] Whether equal
      def ==(other)
        return false unless other.is_a?(BoundingBox)

        @x == other.x && @y == other.y && @width == other.width && @height == other.height
      end

      alias eql? ==
    end
  end
end
