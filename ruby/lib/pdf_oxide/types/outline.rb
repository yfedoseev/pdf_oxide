# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a PDF outline entry (bookmark)
    class Outline
      attr_reader :title, :dest_page, :level

      # Initialize outline
      # @param title [String] Outline title
      # @param dest_page [Integer] Destination page index
      # @param level [Integer] Hierarchy level (default 0)
      def initialize(title:, dest_page:, level: 0)
        @title = title
        @dest_page = dest_page
        @level = level
      end

      # Convert to hash
      # @return [Hash] Outline as hash
      def to_h
        { title: @title, dest_page: @dest_page, level: @level }
      end

      # String representation
      # @return [String] Outline as string
      def to_s
        "Outline(title='#{@title}', dest_page=#{@dest_page}, level=#{@level})"
      end

      # Inspect representation
      # @return [String] Outline inspection
      def inspect
        to_s
      end

      # Equality comparison
      # @param other [Object] Other object to compare
      # @return [Boolean] Whether objects are equal
      def ==(other)
        other.is_a?(Outline) &&
          @title == other.title &&
          @dest_page == other.dest_page &&
          @level == other.level
      end

      # Hash code
      # @return [Integer] Hash code
      def hash
        [@title, @dest_page, @level].hash
      end
    end
  end
end
