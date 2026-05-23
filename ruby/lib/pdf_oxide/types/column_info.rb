# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents information about a detected column in a page
    class ColumnInfo
      attr_reader :index, :x, :y, :width, :height, :page

      def initialize(index: 0, x: 0, y: 0, width: 0, height: 0, page: 0)
        @index = index
        @x = x
        @y = y
        @width = width
        @height = height
        @page = page
      end

      def to_h
        {
          index: @index,
          x: @x,
          y: @y,
          width: @width,
          height: @height,
          page: @page
        }
      end

      def to_s
        "ColumnInfo(index=#{@index}, page=#{@page}, x=#{@x}, y=#{@y}, width=#{@width}, height=#{@height})"
      end

      def inspect
        to_s
      end

      def bbox
        Types::BoundingBox.new(x: @x, y: @y, width: @width, height: @height)
      end

      def area
        @width * @height
      end

      def ==(other)
        other.is_a?(ColumnInfo) && index == other.index && page == other.page
      end

      def hash
        [index, page].hash
      end
    end
  end
end
