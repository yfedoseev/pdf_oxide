# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for OCR (Optical Character Recognition) operations
    # Provides methods to perform OCR on scanned documents and extract text
    class Ocr < Base
      attr_reader :engine_handle

      # Initialize OCR manager
      # @param document [Document] PDF document
      def initialize(document)
        super(document)
        @engine_handle = nil
      end

      # Check if OCR engine is available
      # @return [Boolean] Whether OCR engine is available
      def available?
        check_document!
        begin
          ensure_engine_initialized
          true
        rescue StandardError
          false
        end
      end

      # Initialize OCR engine with default configuration
      # @return [Boolean] Whether initialization succeeded
      def initialize_engine
        check_document!
        return true if @engine_handle && !@engine_handle.null?

        @engine_handle = with_error_check('initialize_ocr_engine') do |error_ptr|
          FFI::Bindings.pdf_ocr_engine_create(nil, error_ptr)
        end

        true
      end

      # Initialize OCR engine with custom configuration
      # @param config [Types::OcrConfig] OCR configuration
      # @return [Boolean] Whether initialization succeeded
      def initialize_engine_with_config(config)
        check_document!
        raise ::PdfOxide::ArgumentError, 'config must be an OcrConfig' unless config.is_a?(Types::OcrConfig)

        return true if @engine_handle && !@engine_handle.null?

        # Create config handle
        config_ptr = with_error_check('create_ocr_config') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_create(error_ptr)
        end

        # Apply configuration settings
        apply_ocr_config_settings(config_ptr, config)

        # Create engine with config
        @engine_handle = with_error_check('initialize_ocr_engine_with_config') do |error_ptr|
          FFI::Bindings.pdf_ocr_engine_create_with_config(config_ptr, error_ptr)
        end

        # Free config handle (config is copied into engine)
        FFI::Bindings.pdf_ocr_config_free(config_ptr) unless config_ptr.nil?

        true
      end

      # Get OCR engine version
      # @return [String] Engine version string
      def engine_version
        check_document!
        ensure_engine_initialized

        FFI::StringMarshaller.from_c_string(
          with_error_check('engine_version') do |error_ptr|
            FFI::Bindings.pdf_ocr_engine_get_version(@engine_handle, error_ptr)
          end
        ) || 'unknown'
      end

      # Check if page needs OCR
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Boolean] Whether page needs OCR
      def page_needs_ocr?(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        with_error_check('page_needs_ocr', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_page_needs_ocr(
            @document.handle,
            page_index,
            error_ptr
          )
        end
      end

      # Perform OCR on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Extracted text
      def ocr_page(page_index, options = {})
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        text = FFI::StringMarshaller.from_c_string(
          with_error_check('ocr_page', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_ocr_extract_text(
              @document.handle,
              page_index,
              @engine_handle,
              false,
              error_ptr
            )
          end
        )

        text || ''
      end

      # Perform OCR on all pages
      # @return [String] Combined text from all pages
      def ocr_document
        check_document!
        ensure_engine_initialized

        text_parts = (0...@document.page_count).map { |i| ocr_page(i) }
        text_parts.join("\n\n")
      end

      # Apply OCR to page (modify PDF)
      # @param page_index [Integer] Page index (0-indexed)
      # @param options [Hash] OCR options
      # @return [Boolean] Whether operation succeeded
      def apply_ocr_to_page(page_index, options = {})
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        with_error_check('apply_ocr_to_page', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_apply_ocr(
            @document.handle,
            @engine_handle,
            error_ptr
          )
        end
        true
      end

      # Apply OCR to entire document
      # @param options [Hash] OCR options
      # @return [Boolean] Whether operation succeeded
      def apply_ocr_to_document(options = {})
        check_document!
        ensure_engine_initialized

        (0...@document.page_count).each do |i|
          apply_ocr_to_page(i, options)
        end

        true
      end

      # Detect if page is scanned
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Boolean] Whether page is scanned
      def page_is_scanned?(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        page_needs_ocr?(page_index)
      end

      # Get OCR confidence for page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Confidence score (0.0 to 1.0)
      def get_ocr_confidence(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        result_handle = with_error_check('get_ocr_confidence', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_recognize_page(
            @document.handle,
            page_index,
            @engine_handle,
            error_ptr
          )
        end

        parse_ocr_confidence(result_handle)
      end

      # Get OCR statistics
      # @return [Hash] OCR statistics
      def ocr_statistics
        check_document!
        ensure_engine_initialized

        {
          engine_version: engine_version,
          pages_to_ocr: (0...@document.page_count).count { |i| page_needs_ocr?(i) },
          total_pages: @document.page_count
        }
      end

      # Detect language in page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Language code
      def detect_language(page_index)
        check_document!
        validate_page_index!(page_index)
        # Note: Language detection would require additional FFI support
        'en' # Default to English
      end

      # Detect OCR needs for page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Boolean] Whether OCR is needed
      def detect_ocr_needs(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        with_error_check('detect_ocr_needs', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_detect_page(@document.handle, page_index, error_ptr)
        end
      end

      # Extract OCR spans from page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] OCR spans with coordinates
      def extract_ocr_spans(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        spans_handle = with_error_check('extract_ocr_spans', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_extract_spans(@document.handle, page_index, error_ptr)
        end

        parse_ocr_spans(spans_handle)
      end

      # Perform batch OCR on multiple pages
      # @param pages [Array<Integer>] Page indices
      # @return [Hash] Batch results
      def batch_ocr(pages)
        check_document!
        ensure_engine_initialized

        pages.each { |p| validate_page_index!(p) }

        results = {}
        pages.each do |page_index|
          results[page_index] = ocr_page(page_index)
        end

        results
      end

      # Extract OCR results for pages
      # @param start_page [Integer] Start page (0-indexed)
      # @param end_page [Integer] End page (0-indexed)
      # @return [Array<Hash>] OCR results for range
      def extract_ocr_pages(start_page, end_page)
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        ensure_engine_initialized

        results_handle = with_error_check('extract_ocr_pages', start: start_page, end: end_page) do |error_ptr|
          FFI::Bindings.pdf_ocr_extract_pages(@document.handle, start_page, end_page, error_ptr)
        end

        parse_ocr_batch_results(results_handle)
      end

      # Get OCR engine status
      # @return [String] Engine status
      def engine_status
        check_document!
        ensure_engine_initialized

        status_ptr = with_error_check('engine_status') do |error_ptr|
          FFI::Bindings.pdf_ocr_engine_get_status(@engine_handle, error_ptr)
        end

        FFI::StringMarshaller.from_c_string(status_ptr) || 'unknown'
      end

      # Get OCR result confidence score
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Float] Confidence score (0.0 to 1.0)
      def ocr_result_confidence(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        result_ptr = with_error_check('ocr_result_confidence', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_recognize_page(
            @document.handle,
            page_index,
            @engine_handle,
            error_ptr
          )
        end

        with_error_check('get_result_confidence') do |error_ptr|
          FFI::Bindings.pdf_oxide_ocr_result_confidence(result_ptr, error_ptr)
        end
      end

      # Get OCR span bounding box
      # @param span_data [Hash] Span data
      # @return [Hash] Bounding box coordinates
      def get_ocr_span_bbox(span_data)
        return {} if span_data.nil? || span_data[:handle].nil?

        bbox = ::FFI::MemoryPointer.new(:float, 4)

        with_error_check('get_ocr_span_bbox') do |error_ptr|
          FFI::Bindings.pdf_ocr_span_get_bbox(span_data[:handle], bbox, error_ptr)
        end

        {
          x: bbox[0].read_float,
          y: bbox[1].read_float,
          width: bbox[2].read_float,
          height: bbox[3].read_float
        }
      end

      # Get character confidence in OCR span
      # @param span_data [Hash] Span data
      # @param char_index [Integer] Character index
      # @return [Float] Confidence score
      def get_ocr_span_character_confidence(span_data, char_index)
        return 0.0 if span_data.nil? || span_data[:handle].nil?

        with_error_check('get_ocr_span_char_confidence', char: char_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_span_get_char_confidence(span_data[:handle], char_index, error_ptr)
        end
      end

      # Extract OCR text with aggregated statistics from page range (FFI-wired)
      # @param start_page [Integer] Starting page index
      # @param end_page [Integer] Ending page index
      # @param skip_non_scanned [Boolean] Skip pages without selectable text
      # @return [Hash] Aggregated OCR statistics
      def extract_page_range(start_page, end_page, skip_non_scanned = true)
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        ensure_engine_initialized

        total_spans = 0
        confidence_sum = 0.0
        skipped_pages = 0

        # Process each page in range
        (start_page..end_page).each do |page_idx|
          # Check if page needs OCR if skip_non_scanned is enabled
          if skip_non_scanned
            unless page_needs_ocr?(page_idx)
              skipped_pages += 1
              next
            end
          end

          # Extract text and confidence for page
          begin
            # Get spans from page
            spans = extract_ocr_spans(page_idx)
            total_spans += spans.length > 0 ? spans.length : 1

            # Accumulate confidence
            confidence = get_ocr_confidence(page_idx)
            confidence_sum += confidence
          rescue StandardError
            # Continue with next page on error
            next
          end
        end

        # Calculate average confidence
        processed_pages = (end_page - start_page + 1) - skipped_pages
        avg_confidence = processed_pages > 0 ? confidence_sum / processed_pages : 0.0

        {
          start_page: start_page,
          end_page: end_page,
          total_pages: end_page - start_page + 1,
          total_spans: total_spans,
          average_confidence: avg_confidence,
          skipped_pages: skipped_pages
        }
      end

      # Phase 4: OCR Enhancement Methods

      # Get character-level confidence scores for OCR span
      # @param span_handle [Pointer] OCR span handle
      # @return [Array<Hash>] Character confidences with text
      def get_character_confidences(span_handle)
        return [] if span_handle.nil? || span_handle.null?

        # First get the text from the span
        text_ptr = with_error_check('get_span_text') do |error_ptr|
          FFI::Bindings.pdf_ocr_results_get_text(span_handle, error_ptr)
        end

        text = FFI::StringMarshaller.from_c(text_ptr) || ''

        # Extract character-level confidence for each character
        text.chars.each_with_index.map do |char, idx|
          confidence = with_error_check('get_char_confidence', char_idx: idx) do |error_ptr|
            FFI::Bindings.pdf_ocr_span_get_char_confidence(span_handle, idx, error_ptr)
          end

          {
            character: char,
            index: idx,
            confidence: confidence.to_f
          }
        end
      end

      # Extract text span with confidence information
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Text spans with confidence
      def extract_text_spans(page_index)
        check_document!
        validate_page_index!(page_index)
        ensure_engine_initialized

        spans_handle = with_error_check('extract_text_spans', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_ocr_extract_spans(@document.handle, page_index, error_ptr)
        end

        parse_text_spans_with_confidence(spans_handle)
      end

      # Extract OCR results with full detail (text, confidence, positions)
      # @param start_page [Integer] Start page (0-indexed)
      # @param end_page [Integer] End page (0-indexed)
      # @param skip_non_scanned [Boolean] Skip pages without selectable text
      # @return [Array<Hash>] Detailed OCR results for page range
      def extract_pages_detailed(start_page, end_page, skip_non_scanned = true)
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        ensure_engine_initialized

        results = []

        (start_page..end_page).each do |page_idx|
          # Check if page needs OCR if skip_non_scanned is enabled
          if skip_non_scanned
            unless page_needs_ocr?(page_idx)
              next
            end
          end

          begin
            # Extract text spans with confidence
            spans = extract_text_spans(page_idx)
            text = ocr_page(page_idx)
            confidence = ocr_result_confidence(page_idx)

            results << {
              page: page_idx,
              text: text,
              confidence: confidence,
              spans: spans
            }
          rescue StandardError => e
            # Log error and continue with next page
            $stderr.puts "OCR error on page #{page_idx}: #{e.message}"
            next
          end
        end

        results
      end

      # Check if GPU acceleration is available
      # @return [Boolean] Whether GPU is available
      def gpu_available?
        check_document!
        ensure_engine_initialized

        with_error_check('gpu_available') do |error_ptr|
          FFI::Bindings.pdf_ocr_gpu_available(error_ptr)
        end
      end

      # Get number of available GPU devices
      # @return [Integer] Number of GPU devices
      def gpu_device_count
        check_document!
        ensure_engine_initialized

        with_error_check('gpu_device_count') do |error_ptr|
          FFI::Bindings.pdf_ocr_gpu_device_count(error_ptr)
        end
      end

      # Get OCR engine capabilities
      # @return [Hash] Engine capabilities information
      def engine_capabilities
        check_document!
        ensure_engine_initialized

        {
          version: engine_version,
          status: engine_status,
          gpu_available: gpu_available?,
          gpu_device_count: gpu_device_count,
          supported_languages: get_supported_languages
        }
      end

      # Get list of supported OCR languages
      # @return [Array<String>] Supported language codes
      def get_supported_languages
        # Default supported languages (would be extended by FFI in full implementation)
        %w[en es fr de it pt ru ja ko zh ar hi]
      end

      # Perform batch OCR with detailed results and progress tracking
      # @param pages [Array<Integer>] Page indices to process
      # @param options [Hash] Options (skip_non_scanned, etc.)
      # @yield [page, result] Progress callback
      # @return [Array<Hash>] Batch OCR results
      def batch_ocr_detailed(pages, options = {})
        check_document!
        ensure_engine_initialized

        pages.each { |p| validate_page_index!(p) }

        skip_non_scanned = options.fetch(:skip_non_scanned, true)
        results = []

        pages.each do |page_index|
          # Skip non-scanned pages if option enabled
          if skip_non_scanned
            unless page_needs_ocr?(page_index)
              yield(page_index, nil) if block_given?
              next
            end
          end

          begin
            text = ocr_page(page_index)
            confidence = ocr_result_confidence(page_index)
            spans = extract_text_spans(page_index)

            result = {
              page: page_index,
              text: text,
              confidence: confidence,
              span_count: spans.length,
              spans: spans
            }

            results << result
            yield(page_index, result) if block_given?
          rescue StandardError => e
            # Log error and continue
            $stderr.puts "Batch OCR error on page #{page_index}: #{e.message}"
            yield(page_index, nil) if block_given?
            next
          end
        end

        results
      end

      # Create OCR summary for document
      # @return [Hash] Document OCR summary
      def document_ocr_summary
        check_document!
        ensure_engine_initialized

        total_pages = @document.page_count
        pages_with_ocr_needed = (0...total_pages).count { |i| page_needs_ocr?(i) }
        pages_without_ocr = total_pages - pages_with_ocr_needed

        {
          total_pages: total_pages,
          pages_needing_ocr: pages_with_ocr_needed,
          pages_with_text: pages_without_ocr,
          engine_version: engine_version,
          gpu_available: gpu_available?,
          estimated_processing_time_seconds: estimate_ocr_time
        }
      end

      # Estimate OCR processing time for document
      # @return [Integer] Estimated time in seconds
      def estimate_ocr_time
        check_document!

        pages_to_process = (0...@document.page_count).count { |i| page_needs_ocr?(i) }

        # Heuristic: ~100ms per page base, 50ms per page if GPU available
        base_time_ms = pages_to_process * 100
        gpu_discount = gpu_available? ? pages_to_process * 50 : 0
        estimated_ms = base_time_ms - gpu_discount

        (estimated_ms / 1000.0).ceil
      end

      # Release OCR engine resources
      # @return [void]
      def release_engine
        return if @engine_handle.nil? || @engine_handle.null?

        FFI::Bindings.pdf_ocr_engine_free(@engine_handle)
        @engine_handle = nil
      end

      private

      def ensure_engine_initialized
        initialize_engine if @engine_handle.nil? || @engine_handle.null?
      end

      def parse_ocr_confidence(handle)
        return 0.0 if handle.nil? || handle.null?

        begin
          FFI::Bindings.pdf_ocr_results_average_confidence(handle)
        ensure
          FFI::Bindings.pdf_ocr_results_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_ocr_spans(spans_handle)
        return [] if spans_handle.nil? || spans_handle.null?

        begin
          count = FFI::Bindings.pdf_ocr_results_count(spans_handle)

          spans = count.times.map do |i|
            text_ptr = FFI::Bindings.pdf_ocr_results_get_text(spans_handle, i)
            bbox = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_ocr_span_get_bbox(spans_handle, bbox)

            {
              index: i,
              text: FFI::StringMarshaller.read_c_string(text_ptr) || '',
              x: bbox[0].read_float,
              y: bbox[1].read_float,
              width: bbox[2].read_float,
              height: bbox[3].read_float
            }
          end

          spans
        ensure
          FFI::Bindings.pdf_ocr_results_free(spans_handle) unless spans_handle.nil? || spans_handle.null?
        end
      end

      def parse_ocr_batch_results(results_handle)
        return [] if results_handle.nil? || results_handle.null?

        begin
          count = FFI::Bindings.pdf_ocr_results_count(results_handle)

          results = count.times.map do |i|
            page = FFI::Bindings.pdf_ocr_batch_results_get_page(results_handle, i)
            text_ptr = FFI::Bindings.pdf_ocr_results_get_text(results_handle, i)

            {
              page: page,
              text: FFI::StringMarshaller.read_c_string(text_ptr) || '',
              index: i
            }
          end

          results
        ensure
          FFI::Bindings.pdf_ocr_results_free(results_handle) unless results_handle.nil? || results_handle.null?
        end
      end

      def apply_ocr_config_settings(config_ptr, config)
        return if config_ptr.nil? || config_ptr.null?

        # Apply detection threshold
        with_error_check('set_detection_threshold') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_set_detection_threshold(
            config_ptr,
            config.detection_threshold.to_f,
            error_ptr
          )
        end

        # Apply recognition threshold
        with_error_check('set_recognition_threshold') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_set_recognition_threshold(
            config_ptr,
            config.recognition_threshold.to_f,
            error_ptr
          )
        end

        # Apply max side length
        with_error_check('set_max_side_len') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_set_max_side_len(
            config_ptr,
            config.max_side_len.to_i,
            error_ptr
          )
        end

        # Apply GPU settings
        with_error_check('set_use_gpu') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_set_use_gpu(
            config_ptr,
            config.use_gpu,
            error_ptr
          )
        end

        # Apply GPU device ID
        with_error_check('set_gpu_device_id') do |error_ptr|
          FFI::Bindings.pdf_ocr_config_set_gpu_device_id(
            config_ptr,
            config.gpu_device_id.to_i,
            error_ptr
          )
        end
      end

      def parse_text_spans_with_confidence(spans_handle)
        return [] if spans_handle.nil? || spans_handle.null?

        begin
          count = FFI::Bindings.pdf_ocr_results_count(spans_handle)

          spans = count.times.map do |i|
            text_ptr = FFI::Bindings.pdf_ocr_results_get_text(spans_handle, i)
            text = FFI::StringMarshaller.from_c(text_ptr) || ''

            # Get bounding box
            bbox = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_ocr_span_get_bbox(spans_handle, bbox)

            # Get span handle for character confidence
            span_ptr = FFI::Bindings.pdf_ocr_results_get_span(spans_handle, i)

            # Get average confidence for this span
            avg_confidence = 0.0
            if span_ptr && !span_ptr.null?
              avg_confidence = text.empty? ? 0.0 : text.length.times.map do |char_idx|
                FFI::Bindings.pdf_ocr_span_get_char_confidence(span_ptr, char_idx)
              end.sum / text.length.to_f
            end

            {
              index: i,
              text: text,
              confidence: avg_confidence,
              x: bbox[0].read_float,
              y: bbox[1].read_float,
              width: bbox[2].read_float,
              height: bbox[3].read_float
            }
          end

          spans
        ensure
          FFI::Bindings.pdf_ocr_results_free(spans_handle) unless spans_handle.nil? || spans_handle.null?
        end
      end
    end
  end
end
