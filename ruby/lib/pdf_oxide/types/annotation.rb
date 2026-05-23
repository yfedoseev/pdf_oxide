# frozen_string_literal: true

require_relative 'bounding_box'

module PdfOxide
  module Types
    # Represents an annotation on a PDF page
    class Annotation
      attr_reader :type, :page, :bbox, :text, :color, :author, :created_at, :subject

      # Annotation types
      TYPES = {
        text: 0,
        highlight: 1,
        underline: 2,
        strikeout: 3,
        square: 4,
        circle: 5,
        ink: 6,
        line: 7,
        polygon: 8,
        polyline: 9,
        stamp: 10,
        file_attachment: 11,
        sound: 12,
        movie: 13
      }.freeze

      # Initialize annotation
      # @param type [Integer, Symbol] Annotation type
      # @param page [Integer] Page number
      # @param bbox [BoundingBox, Hash] Bounding box
      # @param text [String] Annotation text
      # @param color [Integer] RGB color as integer
      # @param author [String] Author name
      # @param created_at [Time] Creation time
      # @param subject [String] Subject line
      def initialize(
        type:,
        page:,
        bbox:,
        text: nil,
        color: 0xFFFF00,
        author: nil,
        created_at: nil,
        subject: nil
      )
        @type = normalize_type(type)
        @page = page.to_i
        @bbox = bbox.is_a?(BoundingBox) ? bbox : BoundingBox.new(**bbox)
        @text = text
        @color = color.is_a?(Integer) ? color : color.to_i(16)
        @author = author
        @created_at = created_at || Time.now
        @subject = subject
      end

      # Get annotation type name
      # @return [Symbol] Type name
      def type_name
        TYPES.invert[@type]
      end

      # Get RGB color components
      # @return [Hash] RGB components
      def color_rgb
        {
          r: (@color >> 16) & 0xFF,
          g: (@color >> 8) & 0xFF,
          b: @color & 0xFF
        }
      end

      # Get color as hex string
      # @return [String] Hex color string
      def color_hex
        format('#%06X', @color)
      end

      # Check if highlight annotation
      # @return [Boolean]
      def highlight?
        @type == TYPES[:highlight]
      end

      # Check if text annotation
      # @return [Boolean]
      def text?
        @type == TYPES[:text]
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          type: type_name,
          page: @page,
          bbox: @bbox.to_h,
          text: @text,
          color: color_hex,
          author: @author,
          created_at: @created_at.iso8601,
          subject: @subject
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "Annotation(#{type_name}, page=#{@page}, text='#{text_preview}')"
      end

      private

      def normalize_type(type)
        if type.is_a?(Symbol)
          TYPES.fetch(type) { raise ArgumentError, "Unknown type: #{type}" }
        else
          type.to_i
        end
      end

      def text_preview
        return '' if @text.nil?
        return @text if @text.length <= 30
        "#{@text[0...27]}..."
      end
    end
  end
end
