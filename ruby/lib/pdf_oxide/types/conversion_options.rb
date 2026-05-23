# frozen_string_literal: true

module PdfOxide
  module Types
    # Options for PDF conversion operations
    class ConversionOptions
      attr_accessor :preserve_layout, :detect_headings, :extract_tables, :detect_columns, :ocr_enabled

      # Initialize conversion options
      # @param preserve_layout [Boolean] Preserve page layout
      # @param detect_headings [Boolean] Detect headings in markdown
      # @param extract_tables [Boolean] Extract table structure
      # @param detect_columns [Boolean] Detect multi-column layout
      # @param ocr_enabled [Boolean] Use OCR for scanned documents
      def initialize(
        preserve_layout: true,
        detect_headings: true,
        extract_tables: true,
        detect_columns: true,
        ocr_enabled: false
      )
        @preserve_layout = preserve_layout
        @detect_headings = detect_headings
        @extract_tables = extract_tables
        @detect_columns = detect_columns
        @ocr_enabled = ocr_enabled
      end

      # Create preset for markdown conversion
      # @return [ConversionOptions] Configured instance
      def self.markdown
        new(preserve_layout: true, detect_headings: true, extract_tables: true)
      end

      # Create preset for HTML conversion
      # @return [ConversionOptions] Configured instance
      def self.html
        new(preserve_layout: true, detect_columns: true)
      end

      # Create preset for text extraction
      # @return [ConversionOptions] Configured instance
      def self.text
        new(preserve_layout: false, detect_headings: false, extract_tables: false)
      end

      # Create preset for OCR processing
      # @return [ConversionOptions] Configured instance
      def self.ocr
        new(preserve_layout: true, ocr_enabled: true, detect_headings: true)
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          preserve_layout: @preserve_layout,
          detect_headings: @detect_headings,
          extract_tables: @extract_tables,
          detect_columns: @detect_columns,
          ocr_enabled: @ocr_enabled
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        flags = []
        flags << 'preserve-layout' if @preserve_layout
        flags << 'detect-headings' if @detect_headings
        flags << 'extract-tables' if @extract_tables
        flags << 'detect-columns' if @detect_columns
        flags << 'ocr' if @ocr_enabled
        "ConversionOptions(#{flags.join(', ')})"
      end
    end
  end
end
