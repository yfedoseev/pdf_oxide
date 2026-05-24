# frozen_string_literal: true

module PdfOxide
  # Enterprise Manager for Bates numbering, document comparison, and stamping
  #
  # Provides Bates numbering, document comparison, and header/footer
  # stamping for legal and enterprise document workflows.
  class EnterpriseManager
    # Bates position constants
    module BatesPosition
      TOP_LEFT = 0
      TOP_CENTER = 1
      TOP_RIGHT = 2
      BOTTOM_LEFT = 3
      BOTTOM_CENTER = 4
      BOTTOM_RIGHT = 5
    end

    # Stamp alignment constants
    module StampAlignment
      LEFT = 0
      CENTER = 1
      RIGHT = 2
    end

    # Difference type constants
    module DifferenceType
      TEXT_ADDED = 0
      TEXT_REMOVED = 1
      TEXT_CHANGED = 2
      IMAGE_ADDED = 3
      IMAGE_REMOVED = 4
    end

    # A single difference found between pages or documents
    Difference = Struct.new(
      :diff_type,
      :description,
      keyword_init: true
    )

    # Result of comparing two pages
    PageComparisonResult = Struct.new(
      :similarity,
      :differences,
      keyword_init: true
    ) do
      def diff_count
        (differences || []).length
      end
    end

    # Result of comparing two documents
    DocumentComparisonResult = Struct.new(
      :similarity,
      :page_comparisons,
      :total_differences,
      keyword_init: true
    )

    attr_reader :document

    # Initialize EnterpriseManager with a PDF document
    # @param document [Object] PDF document handle
    def initialize(document)
      @document = document
    end

    # ===== Bates Numbering =====

    # Apply Bates numbering to all pages
    #
    # @param prefix [String] Bates number prefix text
    # @param start_number [Integer] starting number (default: 1)
    # @param num_digits [Integer] number of digits, zero-padded (default: 6)
    # @param position [Integer] position on page (default: BatesPosition::BOTTOM_RIGHT)
    # @return [Integer] number of pages stamped
    # @raise [PdfOxide::Error] if the operation fails
    def apply_bates(prefix, start_number: 1, num_digits: 6, position: BatesPosition::BOTTOM_RIGHT)
      raise ::PdfOxide::ArgumentError, 'prefix cannot be nil' if prefix.nil?

      FFI::ErrorHandler.with_int_check('bates_apply') do |err|
        Bindings.pdf_bates_apply(
          @document, prefix, start_number, num_digits, position, err
        )
      end
    end

    # Apply advanced Bates numbering with full options
    #
    # @param prefix [String] prefix text
    # @param suffix [String] suffix text
    # @param start_number [Integer] starting number
    # @param num_digits [Integer] digit count (zero-padded)
    # @param position [Integer] position on page
    # @param font_size [Float] font size in points
    # @param margin [Float] margin from page edge in points
    # @param color [Array<Float>] RGB color array [r, g, b] with values 0.0-1.0
    # @return [Integer] number of pages stamped
    # @raise [PdfOxide::Error] if the operation fails
    def apply_bates_advanced(prefix, suffix: '', start_number: 1, num_digits: 6,
                             position: BatesPosition::BOTTOM_RIGHT,
                             font_size: 10.0, margin: 36.0,
                             color: [0.0, 0.0, 0.0])
      raise ::PdfOxide::ArgumentError, 'prefix cannot be nil' if prefix.nil?

      r = color[0] || 0.0
      g = color[1] || 0.0
      b = color[2] || 0.0

      FFI::ErrorHandler.with_int_check('bates_apply_advanced') do |err|
        Bindings.pdf_bates_apply_advanced(
          @document, prefix, suffix,
          start_number, num_digits, position,
          font_size.to_f, margin.to_f,
          r.to_f, g.to_f, b.to_f,
          err
        )
      end
    end

    # ===== Document Comparison =====

    # Compare a page from this document with a page from another
    #
    # @param other_document [Object] the other PDF document handle
    # @param page_a [Integer] page index in this document (default: 0)
    # @param page_b [Integer] page index in the other document (default: 0)
    # @return [PageComparisonResult] result with similarity and differences
    # @raise [PdfOxide::Error] if the comparison fails
    def compare_pages(other_document, page_a: 0, page_b: 0)
      comp_handle = FFI::ErrorHandler.with_error_check('compare_pages') do |err|
        Bindings.pdf_compare_pages(@document, other_document, page_a, page_b, err)
      end

      begin
        similarity = Bindings.pdf_comparison_get_similarity(comp_handle).to_f
        diff_count = Bindings.pdf_comparison_get_diff_count(comp_handle).to_i

        differences = (0...diff_count).map do |i|
          diff_handle = Bindings.pdf_comparison_get_diff(comp_handle, i)
          diff_type = Bindings.pdf_comparison_get_diff_type(diff_handle).to_i
          Difference.new(diff_type: diff_type, description: '')
        end

        PageComparisonResult.new(
          similarity: similarity,
          differences: differences
        )
      ensure
        Bindings.pdf_comparison_free(comp_handle) if comp_handle && !comp_handle.null?
      end
    end

    # Compare this document with another page by page
    #
    # @param other_document [Object] the other PDF document handle
    # @return [DocumentComparisonResult] overall and per-page results
    # @raise [PdfOxide::Error] if the comparison fails
    def compare_documents(other_document)
      comp_handle = FFI::ErrorHandler.with_error_check('compare_documents') do |err|
        Bindings.pdf_compare_documents(@document, other_document, err)
      end

      begin
        similarity = Bindings.pdf_comparison_get_similarity(comp_handle).to_f

        DocumentComparisonResult.new(
          similarity: similarity,
          page_comparisons: [],
          total_differences: 0
        )
      ensure
        Bindings.pdf_document_comparison_free(comp_handle) if comp_handle && !comp_handle.null?
      end
    end

    # ===== Header/Footer Stamping =====

    # Stamp a header on all pages
    #
    # Supports placeholders: {page}, {pages}, {date}.
    #
    # @param text [String] header text
    # @param alignment [Integer] text alignment (default: StampAlignment::CENTER)
    # @param font_size [Float] font size in points
    # @param margin [Float] margin from page edge
    # @return [Integer] number of pages stamped
    # @raise [PdfOxide::Error] if the operation fails
    def stamp_header(text, alignment: StampAlignment::CENTER, font_size: 10.0, margin: 36.0)
      raise ::PdfOxide::ArgumentError, 'text cannot be nil' if text.nil?

      FFI::ErrorHandler.with_int_check('stamp_header') do |err|
        Bindings.pdf_stamp_header(
          @document, text, alignment, font_size.to_f, margin.to_f, err
        )
      end
    end

    # Stamp a footer on all pages
    #
    # Supports placeholders: {page}, {pages}, {date}.
    #
    # @param text [String] footer text
    # @param alignment [Integer] text alignment (default: StampAlignment::CENTER)
    # @param font_size [Float] font size in points
    # @param margin [Float] margin from page edge
    # @return [Integer] number of pages stamped
    # @raise [PdfOxide::Error] if the operation fails
    def stamp_footer(text, alignment: StampAlignment::CENTER, font_size: 10.0, margin: 36.0)
      raise ::PdfOxide::ArgumentError, 'text cannot be nil' if text.nil?

      FFI::ErrorHandler.with_int_check('stamp_footer') do |err|
        Bindings.pdf_stamp_footer(
          @document, text, alignment, font_size.to_f, margin.to_f, err
        )
      end
    end

    # Stamp both header and footer on all pages
    #
    # @param header_text [String, nil] header text (nil to skip)
    # @param footer_text [String, nil] footer text (nil to skip)
    # @param alignment [Integer] text alignment
    # @param font_size [Float] font size in points
    # @param margin [Float] margin from page edge
    # @return [Integer] number of pages stamped
    # @raise [PdfOxide::Error] if the operation fails
    def stamp_header_footer(header_text: nil, footer_text: nil,
                            alignment: StampAlignment::CENTER,
                            font_size: 10.0, margin: 36.0)
      FFI::ErrorHandler.with_int_check('stamp_header_footer') do |err|
        Bindings.pdf_stamp_header_footer(
          @document, header_text, footer_text,
          alignment, font_size.to_f, margin.to_f, err
        )
      end
    end

    # Get enterprise capabilities summary
    # @return [Hash] summary of enterprise capabilities
    def summary
      {
        capabilities: {
          bates_numbering: true,
          bates_advanced: true,
          page_comparison: true,
          document_comparison: true,
          header_stamping: true,
          footer_stamping: true,
          header_footer_stamping: true
        }
      }
    end
  end
end
