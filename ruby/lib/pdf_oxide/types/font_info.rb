# frozen_string_literal: true

module PdfOxide
  module Types
    # Information about a font in a PDF
    class FontInfo
      attr_reader :name, :family, :size, :embedded, :encoding, :subtype

      # Initialize font info
      # @param name [String] Font name
      # @param family [String] Font family
      # @param size [Float] Font size
      # @param embedded [Boolean] Whether font is embedded
      # @param encoding [String] Font encoding
      # @param subtype [String] Font subtype
      def initialize(name:, family:, size:, embedded: false, encoding: nil, subtype: nil)
        @name = name.to_s
        @family = family.to_s
        @size = size.to_f
        @embedded = embedded
        @encoding = encoding
        @subtype = subtype
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          name: @name,
          family: @family,
          size: @size,
          embedded: @embedded,
          encoding: @encoding,
          subtype: @subtype
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "FontInfo(#{@name}, #{@family}, #{@size}pt)"
      end

      # Check equality
      # @param other [FontInfo] Other font info
      # @return [Boolean] Whether equal
      def ==(other)
        return false unless other.is_a?(FontInfo)
        @name == other.name && @family == other.family && @size == other.size
      end

      alias eql? ==
    end
  end
end
