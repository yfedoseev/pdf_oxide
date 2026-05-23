# frozen_string_literal: true

require_relative 'bounding_box'

module PdfOxide
  module Types
    # Represents a text search result with location
    class SearchResult
      attr_reader :page, :text, :bbox, :context

      # Initialize search result
      # @param page [Integer] Page number (0-indexed)
      # @param text [String] Matched text
      # @param bbox [BoundingBox] Bounding box of match
      # @param context [String] Context around match
      def initialize(page:, text:, bbox:, context: nil)
        @page = page.to_i
        @text = text.to_s
        @bbox = bbox.is_a?(BoundingBox) ? bbox : BoundingBox.new(**bbox)
        @context = context
      end

      # Get one-indexed page number (more user-friendly)
      # @return [Integer] One-indexed page number
      def page_number
        @page + 1
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          page: @page,
          page_number: page_number,
          text: @text,
          bbox: @bbox.to_h,
          context: @context
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "SearchResult(page=#{page_number}, text='#{text_preview}')"
      end

      # Check equality
      # @param other [SearchResult] Other search result
      # @return [Boolean] Whether equal
      def ==(other)
        return false unless other.is_a?(SearchResult)
        @page == other.page && @text == other.text && @bbox == other.bbox
      end

      alias eql? ==

      private

      def text_preview
        return @text if @text.length <= 50
        "#{@text[0...47]}..."
      end
    end
  end
end
