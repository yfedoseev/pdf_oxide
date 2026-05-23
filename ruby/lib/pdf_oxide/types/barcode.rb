# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a barcode in a PDF (including QR codes)
    class Barcode
      attr_reader :data, :format, :page, :x, :y, :width, :height

      def initialize(data:, format: 'code128', page: 0, x: 0, y: 0, width: 0, height: 0)
        @data = data
        @format = format
        @page = page
        @x = x
        @y = y
        @width = width
        @height = height
      end

      def to_h
        {
          data: @data,
          format: @format,
          page: @page,
          x: @x,
          y: @y,
          width: @width,
          height: @height
        }
      end

      def to_s
        "Barcode(format=#{@format}, data=#{@data[0..20]}..., page=#{@page})"
      end

      def inspect
        to_s
      end

      def qr_code?
        @format.to_s.downcase == 'qr'
      end

      def bbox
        Types::BoundingBox.new(x: @x, y: @y, width: @width, height: @height)
      end

      def ==(other)
        other.is_a?(Barcode) && data == other.data && format == other.format && page == other.page
      end

      def hash
        [data, format, page].hash
      end
    end
  end
end
