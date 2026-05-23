# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents information about a detected table in a page
    class TableInfo
      attr_reader :index, :x, :y, :width, :height, :rows, :columns, :page

      def initialize(index: 0, x: 0, y: 0, width: 0, height: 0, rows: 0, columns: 0, page: 0)
        @index = index
        @x = x
        @y = y
        @width = width
        @height = height
        @rows = rows
        @columns = columns
        @page = page
      end

      def to_h
        {
          index: @index,
          x: @x,
          y: @y,
          width: @width,
          height: @height,
          rows: @rows,
          columns: @columns,
          page: @page
        }
      end

      def to_s
        "TableInfo(index=#{@index}, page=#{@page}, size=#{@rows}x#{@columns})"
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

      def cell_count
        @rows * @columns
      end

      def average_cell_width
        @columns > 0 ? @width / @columns : 0
      end

      def average_cell_height
        @rows > 0 ? @height / @rows : 0
      end

      def ==(other)
        other.is_a?(TableInfo) && index == other.index && page == other.page
      end

      def hash
        [index, page].hash
      end
    end
  end
end
