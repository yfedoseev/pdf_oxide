# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for page analysis and content detection operations
    # Provides methods to analyze page complexity, detect content types, tables, and columns
    class Analysis < Base
      # COMPLEXITY LEVEL CONSTANTS
      COMPLEXITY_SIMPLE = 0
      COMPLEXITY_MEDIUM = 1
      COMPLEXITY_COMPLEX = 2
      COMPLEXITY_VERY_COMPLEX = 3

      COMPLEXITY_LEVELS = {
        simple: COMPLEXITY_SIMPLE,
        medium: COMPLEXITY_MEDIUM,
        complex: COMPLEXITY_COMPLEX,
        very_complex: COMPLEXITY_VERY_COMPLEX
      }.freeze

      COMPLEXITY_NAMES = COMPLEXITY_LEVELS.invert.freeze

      # CONTENT TYPE CONSTANTS
      CONTENT_TYPE_TEXT = 0
      CONTENT_TYPE_IMAGE = 1
      CONTENT_TYPE_MIXED = 2
      CONTENT_TYPE_FORM = 3

      CONTENT_TYPES = {
        text: CONTENT_TYPE_TEXT,
        image: CONTENT_TYPE_IMAGE,
        mixed: CONTENT_TYPE_MIXED,
        form: CONTENT_TYPE_FORM
      }.freeze

      CONTENT_TYPE_NAMES = CONTENT_TYPES.invert.freeze

      # Analyze a page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Hash] Analysis results
      def analyze_page(page_index)
        check_document!
        validate_page_index!(page_index)

        result_handle = with_error_check('analyze_page', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_analyze_page(@document.handle, page_index, error_ptr)
        end

        parse_analysis_result(result_handle)
      end

      # Detect columns on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Detected columns with bounding boxes
      def detect_columns(page_index)
        check_document!
        validate_page_index!(page_index)

        columns_handle = with_error_check('detect_columns', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_detect_columns(@document.handle, page_index, error_ptr)
        end

        parse_columns_result(columns_handle)
      end

      # Detect tables on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Detected tables with bounding boxes and cell information
      def detect_tables(page_index)
        check_document!
        validate_page_index!(page_index)

        tables_handle = with_error_check('detect_tables', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_detect_tables(@document.handle, page_index, error_ptr)
        end

        parse_tables_result(tables_handle)
      end

      # Get page complexity level
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Symbol] Complexity level (:simple, :medium, :complex, :very_complex)
      def get_page_complexity(page_index)
        check_document!
        validate_page_index!(page_index)

        complexity_int = with_error_check('get_page_complexity', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_complexity(@document.handle, page_index, error_ptr)
        end

        COMPLEXITY_NAMES[complexity_int] || :simple
      end

      # Get complexity score
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Complexity score (0.0 to 1.0)
      def get_complexity_score(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_complexity_score', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_complexity_score(@document.handle, page_index, error_ptr)
        end
      end

      # Detect content type on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Symbol] Content type (:text, :image, :mixed, :form)
      def detect_content_type(page_index)
        check_document!
        validate_page_index!(page_index)

        content_type_int = with_error_check('detect_content_type', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_content_type(@document.handle, page_index, error_ptr)
        end

        CONTENT_TYPE_NAMES[content_type_int] || :mixed
      end

      # Get text density on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Text density (0.0 to 1.0)
      def get_text_density(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_text_density', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_text_density(@document.handle, page_index, error_ptr)
        end
      end

      # Get image density on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Image density (0.0 to 1.0)
      def get_image_density(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('get_image_density', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_image_density(@document.handle, page_index, error_ptr)
        end
      end

      # Extract layout analysis
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Hash] Layout information (blocks, coordinates, etc.)
      def extract_layout_analysis(page_index)
        check_document!
        validate_page_index!(page_index)

        layout_handle = with_error_check('extract_layout_analysis', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_extract_layout(@document.handle, page_index, error_ptr)
        end

        parse_layout_result(layout_handle)
      end

      # Analyze all pages
      # @return [Array<Hash>] Analysis results for all pages
      def analyze_all_pages
        check_document!
        (0...@document.page_count).map { |i| analyze_page(i) }
      end

      # Get analysis statistics for entire document
      # @return [Hash] Document-wide statistics
      def document_statistics
        check_document!
        all_analyses = analyze_all_pages

        total_complexity = all_analyses.sum { |a| a[:complexity_score] || 0 }
        avg_complexity = @document.page_count > 0 ? total_complexity / @document.page_count : 0

        content_types = all_analyses.map { |a| a[:content_type] }.tally
        complexity_levels = all_analyses.map { |a| a[:complexity_level] }.tally

        {
          total_pages: @document.page_count,
          average_complexity: avg_complexity,
          complexity_levels: complexity_levels,
          content_types: content_types,
          has_tables: all_analyses.any? { |a| (a[:table_count] || 0) > 0 },
          has_columns: all_analyses.any? { |a| (a[:column_count] || 0) > 0 }
        }
      end

      private

      def parse_analysis_result(handle)
        return { complexity_score: 0, complexity_level: :simple, content_type: :mixed } if handle.nil? || handle.null?

        begin
          {
            complexity_score: FFI::Bindings.pdf_analysis_get_complexity_score(handle),
            complexity_level: COMPLEXITY_NAMES[FFI::Bindings.pdf_analysis_get_complexity(handle)] || :simple,
            content_type: CONTENT_TYPE_NAMES[FFI::Bindings.pdf_analysis_get_content_type(handle)] || :mixed,
            text_density: FFI::Bindings.pdf_analysis_get_text_density(handle),
            image_density: FFI::Bindings.pdf_analysis_get_image_density(handle),
            column_count: FFI::Bindings.pdf_analysis_get_column_count(handle),
            table_count: FFI::Bindings.pdf_analysis_get_table_count(handle)
          }
        ensure
          FFI::Bindings.pdf_analysis_result_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_columns_result(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_column_count(handle)

          columns = count.times.map do |i|
            bbox_ptr = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_oxide_column_get_bbox(handle, i, bbox_ptr)
            x, y, width, height = bbox_ptr.read_array_of_float(4)

            {
              index: i,
              x: x,
              y: y,
              width: width,
              height: height,
              bbox: Types::BoundingBox.new(x: x, y: y, width: width, height: height)
            }
          end

          columns
        ensure
          FFI::Bindings.pdf_oxide_column_list_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_tables_result(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_table_count(handle)

          tables = count.times.map do |i|
            bbox_ptr = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_oxide_table_get_bbox(handle, i, bbox_ptr)
            x, y, width, height = bbox_ptr.read_array_of_float(4)

            row_count = FFI::Bindings.pdf_oxide_table_get_row_count(handle, i)
            col_count = FFI::Bindings.pdf_oxide_table_get_col_count(handle, i)

            {
              index: i,
              x: x,
              y: y,
              width: width,
              height: height,
              rows: row_count,
              columns: col_count,
              bbox: Types::BoundingBox.new(x: x, y: y, width: width, height: height)
            }
          end

          tables
        ensure
          FFI::Bindings.pdf_oxide_table_list_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_layout_result(handle)
        return {} if handle.nil? || handle.null?

        begin
          {
            block_count: FFI::Bindings.pdf_analysis_get_block_count(handle),
            text_blocks: FFI::Bindings.pdf_analysis_get_text_block_count(handle),
            image_blocks: FFI::Bindings.pdf_analysis_get_image_block_count(handle),
            layout_type: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_analysis_get_layout_type(handle)
            ) || 'unknown'
          }
        ensure
          FFI::Bindings.pdf_analysis_result_free(handle) unless handle.nil? || handle.null?
        end
      end
    end
  end
end
