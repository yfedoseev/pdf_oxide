# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for text search operations
    class Search < Base
      # Search for text on a specific page
      # @param query [String] Text to search for
      # @param page_index [Integer] Page index (0-indexed)
      # @param options [Hash] Search options
      # @return [Array<Types::SearchResult>] Search results
      def search_page(query, page_index, options = {})
        check_document!
        validate_page_index!(page_index)

        case_sensitive = options.fetch(:case_sensitive, false)
        query_utf8 = FFI::StringMarshaller.to_utf8(query)

        results_handle = with_error_check('search_page', query: query, page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_search_page(
            @document.handle,
            query_utf8,
            page_index,
            case_sensitive,
            error_ptr
          )
        end

        parse_search_results(results_handle)
      end

      # Search for text across all pages
      # @param query [String] Text to search for
      # @param options [Hash] Search options
      # @return [Array<Types::SearchResult>] Search results
      def search_all(query, options = {})
        check_document!

        case_sensitive = options.fetch(:case_sensitive, false)
        query_utf8 = FFI::StringMarshaller.to_utf8(query)

        results_handle = with_error_check('search_all', query: query) do |error_ptr|
          FFI::Bindings.pdf_document_search_all(
            @document.handle,
            query_utf8,
            case_sensitive,
            error_ptr
          )
        end

        parse_search_results(results_handle)
      end

      # Search using regex pattern
      # @param pattern [String] Regex pattern
      # @param options [Hash] Search options
      # @return [Array<Types::SearchResult>] Search results
      def search_regex(pattern, options = {})
        check_document!

        case_sensitive = options.fetch(:case_sensitive, false)
        pattern_utf8 = FFI::StringMarshaller.to_utf8(pattern)

        results_handle = with_error_check('search_regex', pattern: pattern) do |error_ptr|
          FFI::Bindings.pdf_document_search_regex(
            @document.handle,
            pattern_utf8,
            case_sensitive,
            error_ptr
          )
        end

        parse_search_results(results_handle)
      rescue StandardError => e
        raise ::PdfOxide::ParseError, "Invalid regex pattern: #{e.message}"
      end

      # Search within specific area
      # @param query [String] Text to search for
      # @param page_index [Integer] Page index
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param width [Float] Area width
      # @param height [Float] Area height
      # @return [Array<Types::SearchResult>] Search results
      def search_in_area(query, page_index, x, y, width, height)
        check_document!
        validate_page_index!(page_index)

        query_utf8 = FFI::StringMarshaller.to_utf8(query)

        results_handle = with_error_check('search_in_area', query: query, page: page_index,
                                                            area: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_search_in_area(
            @document.handle,
            query_utf8,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            error_ptr
          )
        end

        parse_search_results(results_handle)
      end

      # Replace text in document
      # @param old_text [String] Text to find
      # @param new_text [String] Replacement text
      # @param options [Hash] Replace options
      # @return [Boolean] Whether replacement occurred
      def replace_text(old_text, new_text, options = {})
        check_document!

        case_sensitive = options.fetch(:case_sensitive, false)
        old_utf8 = FFI::StringMarshaller.to_utf8(old_text)
        new_utf8 = FFI::StringMarshaller.to_utf8(new_text)

        with_error_check('replace_text', old: old_text, new: new_text) do |error_ptr|
          FFI::Bindings.pdf_document_replace_text(
            @document.handle,
            old_utf8,
            new_utf8,
            case_sensitive,
            error_ptr
          )
        end
      end

      # Get count of search results
      # @param query [String] Text to search for
      # @return [Integer] Number of matches
      def search_count(query, options = {})
        search_all(query, options).count
      end

      private

      def parse_search_results(results_handle)
        return [] if results_handle.nil? || results_handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_search_result_count(results_handle)

          results = count.times.map do |i|
            page = FFI::Bindings.pdf_oxide_search_result_get_page(results_handle, i)
            text_ptr = FFI::Bindings.pdf_oxide_search_result_get_text(results_handle, i)
            text = FFI::StringMarshaller.read_c_string(text_ptr, free_after: false)

            # Estimate bbox - this would need actual FFI implementation
            bbox = Types::BoundingBox.new(x: 0, y: 0, width: 100, height: 20)

            Types::SearchResult.new(page: page, text: text, bbox: bbox)
          end

          results
        ensure
          FFI::Bindings.pdf_oxide_search_result_free(results_handle) unless results_handle.nil? || results_handle.null?
        end
      end
    end
  end
end
