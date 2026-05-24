# frozen_string_literal: true

require 'json'
require_relative 'base'

module PdfOxide
  module Managers
    # Manager for barcode operations
    # Provides methods to generate and add barcodes and QR codes to PDFs
    class Barcode < Base
      # BARCODE FORMAT CONSTANTS
      BARCODE_FORMAT_CODE128 = 0
      BARCODE_FORMAT_CODE39 = 1
      BARCODE_FORMAT_CODE93 = 2
      BARCODE_FORMAT_EAN13 = 3
      BARCODE_FORMAT_EAN8 = 4
      BARCODE_FORMAT_UPCA = 5
      BARCODE_FORMAT_UPCE = 6
      BARCODE_FORMAT_ITF = 7

      BARCODE_FORMATS = {
        code128: BARCODE_FORMAT_CODE128,
        code39: BARCODE_FORMAT_CODE39,
        code93: BARCODE_FORMAT_CODE93,
        ean13: BARCODE_FORMAT_EAN13,
        ean8: BARCODE_FORMAT_EAN8,
        upca: BARCODE_FORMAT_UPCA,
        upce: BARCODE_FORMAT_UPCE,
        itf: BARCODE_FORMAT_ITF
      }.freeze

      FORMAT_NAMES = BARCODE_FORMATS.invert.freeze

      # Add QR code to page
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param size [Float] QR code size
      # @param data [String] Data to encode
      # @param options [Hash] Additional options (error_correction, etc.)
      # @return [Boolean] Whether operation succeeded
      def add_qr_code(page_index, x, y, size, data, options = {})
        check_document!
        validate_page_index!(page_index)

        data_utf8 = FFI::StringMarshaller.to_utf8(data)
        error_correction = options.fetch(:error_correction, 0)

        with_error_check('add_qr_code', page: page_index, size: size) do |error_ptr|
          FFI::Bindings.pdf_document_add_qr_code(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            size.to_f,
            data_utf8,
            error_correction,
            error_ptr
          )
        end
        true
      end

      # Add barcode to page
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width of barcode
      # @param height [Float] Height of barcode
      # @param data [String] Data to encode
      # @param format [Symbol, Integer] Barcode format
      # @param options [Hash] Additional options
      # @return [Boolean] Whether operation succeeded
      def add_barcode(page_index, x, y, width, height, data, format = :code128, _options = {})
        check_document!
        validate_page_index!(page_index)

        data_utf8 = FFI::StringMarshaller.to_utf8(data)
        format_int = format.is_a?(Symbol) ? BARCODE_FORMATS.fetch(format, BARCODE_FORMAT_CODE128) : format

        with_error_check('add_barcode', page: page_index, format: format) do |error_ptr|
          FFI::Bindings.pdf_document_add_barcode(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            data_utf8,
            format_int,
            error_ptr
          )
        end
        true
      end

      # Generate QR code as PNG image
      # @param data [String] Data to encode
      # @param size [Float] QR code size
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_qr_code(data, size = 200, options = {})
        data_utf8 = FFI::StringMarshaller.to_utf8(data)
        error_correction = options.fetch(:error_correction, 0)

        result_ptr = with_error_check('generate_qr_code', size: size) do |error_ptr|
          FFI::Bindings.pdf_generate_qr_code(
            data_utf8,
            size.to_i,
            error_correction,
            error_ptr
          )
        end

        read_image_data(result_ptr)
      end

      # Generate barcode as PNG image
      # @param data [String] Data to encode
      # @param format [Symbol, Integer] Barcode format
      # @param width [Float] Width of barcode
      # @param height [Float] Height of barcode
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_barcode(data, format = :code128, width = 200, height = 100, _options = {})
        data_utf8 = FFI::StringMarshaller.to_utf8(data)
        format_int = format.is_a?(Symbol) ? BARCODE_FORMATS.fetch(format, BARCODE_FORMAT_CODE128) : format

        result_ptr = with_error_check('generate_barcode', format: format, width: width, height: height) do |error_ptr|
          FFI::Bindings.pdf_generate_barcode(
            data_utf8,
            format_int,
            width.to_i,
            height.to_i,
            error_ptr
          )
        end

        read_image_data(result_ptr)
      end

      # Extract barcodes from page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Detected barcodes with data and format
      def extract_barcodes(page_index)
        check_document!
        validate_page_index!(page_index)

        barcodes_handle = with_error_check('extract_barcodes', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_extract_barcodes(@document.handle, page_index, error_ptr)
        end

        parse_barcode_list(barcodes_handle)
      end

      # Get all barcodes in document
      # @return [Array<Hash>] All barcodes with page numbers
      def get_all_barcodes
        check_document!
        all_barcodes = []
        (0...@document.page_count).each do |page_idx|
          barcodes = extract_barcodes(page_idx)
          barcodes.each do |barcode|
            barcode[:page] = page_idx
            all_barcodes << barcode
          end
        end
        all_barcodes
      end

      # Get barcode statistics
      # @return [Hash] Barcode statistics
      def barcode_statistics
        check_document!
        all_barcodes = get_all_barcodes

        format_counts = {}
        all_barcodes.each do |barcode|
          format_name = FORMAT_NAMES[barcode[:format]] || 'unknown'
          format_counts[format_name] ||= 0
          format_counts[format_name] += 1
        end

        {
          total_barcodes: all_barcodes.count,
          by_format: format_counts,
          by_page: all_barcodes.group_by { |b| b[:page] }.transform_values(&:count)
        }
      end

      # Phase 3: Barcode Completion Methods

      # Generate EAN-13 barcode
      # @param data [String] EAN-13 data (13 digits)
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_ean13(data, _options = {})
        validate_ean_data!(data, 13)
        data_utf8 = FFI::StringMarshaller.to_utf8(data)

        result_ptr = with_error_check('generate_ean13', data: data) do |error_ptr|
          FFI::Bindings.pdf_generate_ean13(data_utf8, error_ptr)
        end

        read_image_data(result_ptr)
      end

      # Generate EAN-8 barcode
      # @param data [String] EAN-8 data (8 digits)
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_ean8(data, _options = {})
        validate_ean_data!(data, 8)
        data_utf8 = FFI::StringMarshaller.to_utf8(data)

        result_ptr = with_error_check('generate_ean8', data: data) do |error_ptr|
          FFI::Bindings.pdf_generate_ean8(data_utf8, error_ptr)
        end

        read_image_data(result_ptr)
      end

      # Generate UPC-A barcode
      # @param data [String] UPC-A data (12 digits)
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_upc_a(data, _options = {})
        validate_upc_data!(data)
        data_utf8 = FFI::StringMarshaller.to_utf8(data)

        result_ptr = with_error_check('generate_upc_a', data: data) do |error_ptr|
          FFI::Bindings.pdf_generate_upc_a(data_utf8, error_ptr)
        end

        read_image_data(result_ptr)
      end

      # Generate Code128 barcode
      # @param data [String] Code128 data
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_code128(data, _options = {})
        raise ::PdfOxide::ArgumentError, 'Data cannot be empty' if data.nil? || data.empty?

        data_utf8 = FFI::StringMarshaller.to_utf8(data)

        result_ptr = with_error_check('generate_code128', data: data) do |error_ptr|
          FFI::Bindings.pdf_generate_code128(data_utf8, error_ptr)
        end

        read_image_data(result_ptr)
      end

      # Generate Code39 barcode
      # @param data [String] Code39 data
      # @param options [Hash] Additional options
      # @return [String] PNG image data
      def generate_code39(data, _options = {})
        raise ::PdfOxide::ArgumentError, 'Data cannot be empty' if data.nil? || data.empty?

        data_utf8 = FFI::StringMarshaller.to_utf8(data)

        result_ptr = with_error_check('generate_code39', data: data) do |error_ptr|
          FFI::Bindings.pdf_generate_code39(data_utf8, error_ptr)
        end

        read_image_data(result_ptr)
      end

      # Convert barcode to Base64 string
      # @param barcode_handle [Pointer] Barcode handle from generation function
      # @param format [Integer] Image format (0=PNG, 1=JPEG, etc.)
      # @return [String] Base64-encoded image
      def barcode_to_base64(barcode_handle, format = 0)
        raise ::PdfOxide::ArgumentError, 'Invalid barcode handle' if barcode_handle.nil? || barcode_handle.null?

        with_error_check('barcode_to_base64') do |error_ptr|
          FFI::Bindings.pdf_barcode_get_image_base64(barcode_handle, format, error_ptr)
        end
      end

      # Add barcode to page with automatic fitting
      # @param page_index [Integer] Page index (0-indexed)
      # @param barcode_handle [Pointer] Barcode handle
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param max_width [Float] Maximum width
      # @param max_height [Float] Maximum height
      # @return [Boolean] Whether operation succeeded
      def add_barcode_fit(page_index, barcode_handle, x, y, max_width, max_height)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Invalid barcode handle' if barcode_handle.nil? || barcode_handle.null?

        with_error_check('add_barcode_fit', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_add_barcode_to_page_fit(
            @document.handle,
            page_index,
            barcode_handle,
            x.to_f,
            y.to_f,
            max_width.to_f,
            max_height.to_f,
            error_ptr
          )
        end
        true
      end

      # Add QR code with label to page
      # @param page_index [Integer] Page index (0-indexed)
      # @param data [String] QR code data
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param size [Float] QR code size
      # @param label [String] Label text below QR code
      # @param options [Hash] Additional options
      # @return [Boolean] Whether operation succeeded
      def add_qr_with_label(page_index, data, x, y, size, label = '', _options = {})
        check_document!
        validate_page_index!(page_index)

        data_utf8 = FFI::StringMarshaller.to_utf8(data)
        label_utf8 = FFI::StringMarshaller.to_utf8(label)

        with_error_check('add_qr_with_label', page: page_index, data: data, label: label) do |error_ptr|
          FFI::Bindings.pdf_add_qr_code_with_label(
            @document.handle,
            page_index,
            data_utf8,
            x.to_f,
            y.to_f,
            size.to_f,
            label_utf8,
            error_ptr
          )
        end
        true
      end

      # Detect barcodes on page with full information
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Detected barcodes with data, format, position, and confidence
      def detect_barcodes(page_index)
        check_document!
        validate_page_index!(page_index)

        barcodes_handle = with_error_check('detect_barcodes', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_detect_barcodes_on_page(@document.handle, page_index, error_ptr)
        end

        parse_barcode_detections(barcodes_handle)
      end

      # Detect barcodes across all pages
      # @return [Array<Hash>] All detected barcodes with page numbers
      def detect_all_barcodes
        check_document!
        all_barcodes = []

        (0...@document.page_count).each do |page_idx|
          barcodes = detect_barcodes(page_idx)
          barcodes.each do |barcode|
            barcode[:page] = page_idx
            all_barcodes << barcode
          end
        end

        all_barcodes
      end

      # Get barcode detection statistics with confidence data
      # @return [Hash] Detection statistics
      def barcode_detection_stats
        check_document!
        all_barcodes = detect_all_barcodes

        format_counts = {}
        confidence_by_format = {}

        all_barcodes.each do |barcode|
          format_name = FORMAT_NAMES[barcode[:format]] || 'unknown'

          format_counts[format_name] ||= 0
          format_counts[format_name] += 1

          confidence_by_format[format_name] ||= []
          confidence_by_format[format_name] << barcode[:confidence]
        end

        confidence_stats = confidence_by_format.transform_values do |confidences|
          {
            count: confidences.length,
            min: confidences.min,
            max: confidences.max,
            avg: confidences.sum.to_f / confidences.length
          }
        end

        {
          total_detections: all_barcodes.count,
          by_format: format_counts,
          confidence_stats: confidence_stats,
          by_page: all_barcodes.group_by { |b| b[:page] }.transform_values(&:count)
        }
      end

      # Export barcodes to JSON format
      # @param include_pages [Boolean] Include page information
      # @return [String] JSON string
      def export_barcodes_json(include_pages = true)
        check_document!
        barcodes = include_pages ? detect_all_barcodes : get_all_barcodes

        barcodes.map do |barcode|
          {
            data: barcode[:data],
            format: barcode[:format_name] || FORMAT_NAMES[barcode[:format]],
            confidence: barcode[:confidence],
            bounds: barcode[:bounds],
            page: barcode[:page]
          }
        end.to_json
      end

      private

      def validate_ean_data!(data, expected_length)
        raise ::PdfOxide::ArgumentError, "EAN data must be #{expected_length} digits" if data.nil? || data.empty?
        raise ::PdfOxide::ArgumentError, "EAN data must be #{expected_length} digits" unless data.length == expected_length
        raise ::PdfOxide::ArgumentError, 'EAN data must contain only digits' unless data.match?(/^\d+$/)
      end

      def validate_upc_data!(data)
        raise ::PdfOxide::ArgumentError, 'UPC-A data must be 12 digits' if data.nil? || data.length != 12
        raise ::PdfOxide::ArgumentError, 'UPC-A data must contain only digits' unless data.match?(/^\d+$/)
      end

      def read_image_data(result_ptr)
        return nil if result_ptr.nil?

        # Get size and read PNG data
        size_ptr = ::FFI::MemoryPointer.new(:int32)
        data_ptr = FFI::Bindings.pdf_barcode_get_image_png(result_ptr, size_ptr)
        size = size_ptr.read_int32

        image_data = data_ptr.read_bytes(size) if size.positive? && !data_ptr.null?

        FFI::Bindings.pdf_barcode_free(result_ptr) unless result_ptr.nil?
        image_data
      end

      def parse_barcode_list(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_barcode_count(handle)

          barcodes = count.times.map do |i|
            {
              data: FFI::StringMarshaller.read_c_string(
                FFI::Bindings.pdf_oxide_barcode_get_data(handle, i)
              ),
              format: FFI::Bindings.pdf_oxide_barcode_get_format(handle, i),
              format_name: FORMAT_NAMES[FFI::Bindings.pdf_oxide_barcode_get_format(handle, i)] || 'unknown'
            }
          end

          barcodes
        ensure
          FFI::Bindings.pdf_oxide_barcode_list_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_barcode_detections(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_barcode_count(handle)

          barcodes = count.times.map do |i|
            barcode_ptr = FFI::Bindings.pdf_oxide_barcode_get_data(handle, i)

            # Extract bounds
            x_ptr = ::FFI::MemoryPointer.new(:float)
            y_ptr = ::FFI::MemoryPointer.new(:float)
            w_ptr = ::FFI::MemoryPointer.new(:float)
            h_ptr = ::FFI::MemoryPointer.new(:float)

            FFI::Bindings.pdf_barcode_get_bounds(barcode_ptr, x_ptr, y_ptr, w_ptr, h_ptr)

            bounds = {
              x: x_ptr.read_float,
              y: y_ptr.read_float,
              width: w_ptr.read_float,
              height: h_ptr.read_float
            }

            # Extract confidence
            confidence = FFI::Bindings.pdf_barcode_get_confidence(barcode_ptr)

            {
              data: FFI::StringMarshaller.from_c(FFI::Bindings.pdf_barcode_get_data(barcode_ptr)),
              format: FFI::Bindings.pdf_barcode_get_format(barcode_ptr),
              format_name: FORMAT_NAMES[FFI::Bindings.pdf_barcode_get_format(barcode_ptr)] || 'unknown',
              confidence: confidence,
              bounds: bounds
            }
          end

          barcodes
        ensure
          FFI::Bindings.pdf_oxide_barcode_list_free(handle) unless handle.nil? || handle.null?
        end
      end
    end
  end
end
