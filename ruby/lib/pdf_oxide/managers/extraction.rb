# frozen_string_literal: true

require_relative 'base'
require 'fileutils'

module PdfOxide
  module Managers
    # Manager for content extraction operations
    # Provides methods to extract text, fonts, images, and other content from PDF pages
    class Extraction < Base
      # Extract text from page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Extracted text
      def extract_text(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_text', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_extract_text(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Extract text from all pages
      # @return [String] Combined text from all pages
      def extract_text_all
        check_document!
        (0...@document.page_count).map { |i| extract_text(i) }.join("\n\n")
      end

      # Extract page as Markdown
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Page content as Markdown
      def extract_to_markdown(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_to_markdown', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_to_markdown(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Extract all pages as Markdown
      # @return [String] All pages as Markdown
      def extract_to_markdown_all
        check_document!
        (0...@document.page_count).map { |i| extract_to_markdown(i) }.join("\n\n---\n\n")
      end

      # Extract page as HTML
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Page content as HTML
      def extract_to_html(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_to_html', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_to_html(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Extract all pages as HTML
      # @return [String] All pages as HTML
      def extract_to_html_all
        check_document!
        html_parts = (0...@document.page_count).map { |i| extract_to_html(i) }
        "<html><body>#{html_parts.join}</body></html>"
      end

      # Get embedded fonts on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Types::FontInfo>] Array of font information
      def get_embedded_fonts(page_index)
        check_document!
        validate_page_index!(page_index)

        fonts_handle = with_error_check('get_embedded_fonts', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_embedded_fonts(@document.handle, page_index, error_ptr)
        end

        parse_font_list(fonts_handle)
      end

      # Get embedded images on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Types::ImageInfo>] Array of image information
      def get_embedded_images(page_index)
        check_document!
        validate_page_index!(page_index)

        images_handle = with_error_check('get_embedded_images', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_embedded_images(@document.handle, page_index, error_ptr)
        end

        parse_image_list(images_handle)
      end

      # Extract image to file
      # @param page_index [Integer] Page index (0-indexed)
      # @param image_index [Integer] Image index on page
      # @param output_path [String] Path to save image
      # @return [Boolean] Whether extraction succeeded
      def extract_image(page_index, image_index, output_path)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Image index must be >= 0' if image_index.negative?
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('extract_image', page: page_index, image: image_index, path: output_path) do |error_ptr|
          FFI::Bindings.pdf_document_extract_image(
            @document.handle,
            page_index,
            image_index,
            output_path_utf8,
            error_ptr
          )
        end
        true
      end

      # Extract all images from page
      # @param page_index [Integer] Page index (0-indexed)
      # @param output_dir [String] Directory to save images
      # @return [Integer] Number of images extracted
      def extract_all_images(page_index, output_dir)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Output directory cannot be empty' if output_dir.nil? || output_dir.empty?

        # Create directory if it doesn't exist
        FileUtils.mkdir_p(output_dir)

        output_dir_utf8 = FFI::StringMarshaller.to_utf8(output_dir)

        with_error_check('extract_all_images', page: page_index, dir: output_dir) do |error_ptr|
          FFI::Bindings.pdf_document_extract_all_images(
            @document.handle,
            page_index,
            output_dir_utf8,
            error_ptr
          )
        end
      end

      # Get text with bounding boxes
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Text with coordinate information
      def get_text_with_bbox(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_text_with_bbox', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_extract_with_bbox(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Get text statistics for page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Hash] Text statistics
      def get_text_statistics(page_index)
        check_document!
        validate_page_index!(page_index)

        stats_ptr = with_error_check('get_text_statistics', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_text_statistics(@document.handle, page_index, error_ptr)
        end

        parse_text_statistics(stats_ptr)
      end

      # Get page resources (fonts, images, etc.)
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Hash] Resource information
      def get_page_resources(page_index)
        check_document!
        validate_page_index!(page_index)

        {
          fonts: get_embedded_fonts(page_index),
          images: get_embedded_images(page_index),
          font_count: get_embedded_fonts(page_index).count,
          image_count: get_embedded_images(page_index).count
        }
      end

      # Get font usage information
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Hash] Font usage statistics
      def get_font_usage(page_index)
        check_document!
        validate_page_index!(page_index)

        fonts = get_embedded_fonts(page_index)
        {
          total_fonts: fonts.count,
          embedded_fonts: fonts.count(&:embedded),
          fonts: fonts.map { |f| { name: f.name, family: f.family, embedded: f.embedded } }
        }
      end

      # Get unique characters on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<String>] Unique characters
      def get_unique_characters(page_index)
        check_document!
        validate_page_index!(page_index)

        text = extract_text(page_index)
        text.each_char.uniq.sort
      end

      # Extract plain text from page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Plain text without formatting
      def extract_plain_text(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_plain_text', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_to_plain_text(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Extract all pages as plain text
      # @return [String] All pages as plain text
      def extract_plain_text_all
        check_document!
        (0...@document.page_count).map { |i| extract_plain_text(i) }.join("\n\n")
      end

      # Extract embedded files from document
      # @return [Array<Hash>] List of embedded files with metadata
      def extract_embedded_files
        check_document!

        files_handle = with_error_check('extract_embedded_files') do |error_ptr|
          FFI::Bindings.pdf_document_extract_embedded_files(@document.handle, error_ptr)
        end

        parse_embedded_files(files_handle)
      end

      # Extract links from page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Links with coordinates and targets
      def extract_links(page_index)
        check_document!
        validate_page_index!(page_index)

        links_handle = with_error_check('extract_links', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_extract_links(@document.handle, page_index, error_ptr)
        end

        parse_links(links_handle)
      end

      # Extract page with layout information
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Text with layout preserved
      def extract_with_layout(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_with_layout', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_extract_with_layout(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Extract specific font from page
      # @param page_index [Integer] Page index (0-indexed)
      # @param font_index [Integer] Font index
      # @param output_path [String] Path to save font file
      # @return [Boolean] Whether extraction succeeded
      def extract_font(page_index, font_index, output_path)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Font index must be >= 0' if font_index.negative?
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('extract_font', page: page_index, font: font_index) do |error_ptr|
          FFI::Bindings.pdf_document_extract_font(
            @document.handle,
            page_index,
            font_index,
            output_path_utf8,
            error_ptr
          )
        end
        true
      end

      # Get all fonts used in document
      # @return [Hash] Complete font usage statistics
      def get_font_usage_all
        check_document!

        font_usage_ptr = with_error_check('get_font_usage_all') do |error_ptr|
          FFI::Bindings.pdf_document_get_font_usage(@document.handle, error_ptr)
        end

        parse_font_usage(font_usage_ptr)
      end

      # Count text occurrences in document
      # @param text [String] Text to search for
      # @return [Integer] Number of occurrences
      def count_text_occurrences(text)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Text cannot be empty' if text.nil? || text.empty?

        text_utf8 = FFI::StringMarshaller.to_utf8(text)

        with_error_check('count_text_occurrences', text: text) do |error_ptr|
          FFI::Bindings.pdf_document_count_text_occurrences(@document.handle, text_utf8, error_ptr)
        end
      end

      # Get page label
      # @param page_index [Integer] Page index (0-indexed)
      # @return [String] Page label (e.g., "i", "1", "A")
      def get_page_label(page_index)
        check_document!
        validate_page_index!(page_index)

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_page_label', page: page_index) do |error_ptr|
            FFI::Bindings.pdf_document_get_page_label(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # Get text with exact coordinates
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Hash>] Text segments with coordinates
      def get_text_with_coordinates(page_index)
        check_document!
        validate_page_index!(page_index)

        text_with_coords = with_error_check('get_text_with_coordinates', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_text_with_coordinates(@document.handle, page_index, error_ptr)
        end

        parse_text_with_coordinates(text_with_coords)
      end

      # Get unique characters in entire document
      # @return [Array<String>] All unique characters used
      def get_unique_characters_all
        check_document!

        chars_ptr = with_error_check('get_unique_characters_all') do |error_ptr|
          FFI::Bindings.pdf_document_get_unique_characters(@document.handle, error_ptr)
        end

        parse_unique_characters(chars_ptr)
      end

      # Search annotations in document
      # @param query [String] Search query
      # @return [Array<Hash>] Found annotations
      def search_annotations(query)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Query cannot be empty' if query.nil? || query.empty?

        query_utf8 = FFI::StringMarshaller.to_utf8(query)

        results_handle = with_error_check('search_annotations', query: query) do |error_ptr|
          FFI::Bindings.pdf_document_search_annotations(@document.handle, query_utf8, error_ptr)
        end

        parse_search_results(results_handle)
      end

      private

      def parse_font_list(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_font_count(handle)

          fonts = count.times.map do |i|
            name_ptr = FFI::Bindings.pdf_oxide_font_get_name(handle, i)
            family_ptr = FFI::Bindings.pdf_oxide_font_get_family(handle, i)

            Types::FontInfo.new(
              name: FFI::StringMarshaller.read_c_string(name_ptr),
              family: FFI::StringMarshaller.read_c_string(family_ptr),
              size: FFI::Bindings.pdf_oxide_font_get_size(handle, i),
              embedded: FFI::Bindings.pdf_oxide_font_is_embedded(handle, i)
            )
          end

          fonts
        ensure
          FFI::Bindings.pdf_oxide_font_list_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_image_list(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_image_count(handle)

          images = count.times.map do |i|
            name_ptr = FFI::Bindings.pdf_oxide_image_get_name(handle, i)

            Types::ImageInfo.new(
              name: FFI::StringMarshaller.read_c_string(name_ptr),
              width: FFI::Bindings.pdf_oxide_image_get_width(handle, i),
              height: FFI::Bindings.pdf_oxide_image_get_height(handle, i),
              color_space: FFI::Bindings.pdf_oxide_image_get_color_space(handle, i),
              bits_per_component: FFI::Bindings.pdf_oxide_image_get_bits_per_component(handle, i)
            )
          end

          images
        ensure
          FFI::Bindings.pdf_oxide_image_list_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_text_statistics(stats_ptr)
        return {} if stats_ptr.nil? || stats_ptr.null?

        {
          character_count: 0,
          word_count: 0,
          line_count: 0,
          paragraph_count: 0
        }
      end

      def parse_embedded_files(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_embedded_file_count(handle)

          files = count.times.map do |i|
            name_ptr = FFI::Bindings.pdf_oxide_embedded_file_get_name(handle, i)
            size = FFI::Bindings.pdf_oxide_embedded_file_get_size(handle, i)

            {
              name: FFI::StringMarshaller.read_c_string(name_ptr),
              size: size,
              index: i
            }
          end

          files
        ensure
          FFI::Bindings.pdf_oxide_embedded_file_list_free(handle)
        end
      end

      def parse_links(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_link_count(handle)

          links = count.times.map do |i|
            url_ptr = FFI::Bindings.pdf_oxide_link_get_url(handle, i)
            bbox = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_oxide_link_get_bbox(handle, i, bbox)

            {
              url: FFI::StringMarshaller.read_c_string(url_ptr),
              x: bbox[0].read_float,
              y: bbox[1].read_float,
              width: bbox[2].read_float,
              height: bbox[3].read_float
            }
          end

          links
        ensure
          FFI::Bindings.pdf_oxide_link_list_free(handle)
        end
      end

      def parse_font_usage(font_usage_ptr)
        return {} if font_usage_ptr.nil? || font_usage_ptr.null?

        begin
          count = FFI::Bindings.pdf_oxide_font_usage_count(font_usage_ptr)

          fonts = count.times.map do |i|
            name_ptr = FFI::Bindings.pdf_oxide_font_usage_get_name(font_usage_ptr, i)

            {
              name: FFI::StringMarshaller.read_c_string(name_ptr),
              pages: FFI::Bindings.pdf_oxide_font_usage_get_page_count(font_usage_ptr, i),
              embedded: FFI::Bindings.pdf_oxide_font_usage_is_embedded(font_usage_ptr, i)
            }
          end

          { total_fonts: count, fonts: fonts }
        ensure
          FFI::Bindings.pdf_oxide_font_usage_free(font_usage_ptr)
        end
      end

      def parse_text_with_coordinates(text_with_coords)
        return [] if text_with_coords.nil? || text_with_coords.empty?

        # Parse text with coordinate data
        segments = []
        text_with_coords.split("\n").each do |line|
          # Expected format: "text|x|y|width|height"
          parts = line.split('|')
          next unless parts.length >= 5

          segments << {
            text: parts[0],
            x: parts[1].to_f,
            y: parts[2].to_f,
            width: parts[3].to_f,
            height: parts[4].to_f
          }
        end

        segments
      end

      def parse_unique_characters(chars_ptr)
        return [] if chars_ptr.nil? || chars_ptr.empty?

        chars_ptr.each_char.uniq.sort
      end

      def parse_search_results(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_search_result_count(handle)

          results = count.times.map do |i|
            page = FFI::Bindings.pdf_oxide_search_result_get_page(handle, i)
            text_ptr = FFI::Bindings.pdf_oxide_search_result_get_text(handle, i)
            bbox = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_oxide_search_result_get_bbox(handle, i, bbox)

            {
              page: page,
              text: FFI::StringMarshaller.read_c_string(text_ptr),
              x: bbox[0].read_float,
              y: bbox[1].read_float,
              width: bbox[2].read_float,
              height: bbox[3].read_float
            }
          end

          results
        ensure
          FFI::Bindings.pdf_oxide_search_result_free(handle)
        end
      end
    end
  end
end
