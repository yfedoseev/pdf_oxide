# frozen_string_literal: true

module PdfOxide
  module Types
    # Information about a PDF page
    class PageInfo
      attr_reader :index, :width, :height, :rotation, :page_count

      # Initialize page info
      # @param index [Integer] Page index
      # @param width [Float] Page width
      # @param height [Float] Page height
      # @param rotation [Integer] Page rotation (0, 90, 180, 270)
      # @param page_count [Integer] Total page count
      def initialize(index:, width:, height:, rotation: 0, page_count: 0)
        @index = index
        @width = width
        @height = height
        @rotation = rotation
        @page_count = page_count
      end

      # Get aspect ratio
      # @return [Float] Width/Height ratio
      def aspect_ratio
        @height.zero? ? 0.0 : (@width.to_f / @height)
      end

      # Convert to hash
      # @return [Hash] Page info as hash
      def to_h
        {
          index: @index,
          width: @width,
          height: @height,
          rotation: @rotation,
          page_count: @page_count,
          aspect_ratio: aspect_ratio
        }
      end

      # String representation
      # @return [String] Page info as string
      def to_s
        "PageInfo(index=#{@index}, width=#{@width}, height=#{@height}, rotation=#{@rotation})"
      end

      # Inspect representation
      # @return [String] Page info inspection
      def inspect
        to_s
      end
    end
  end
end
