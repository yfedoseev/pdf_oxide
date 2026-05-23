# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for extraction strategies
    # Provides different text extraction algorithms optimized for various layouts
    class ExtractionStrategy < Base
      # Available strategies
      STRATEGY_SIMPLE = 'simple'
      STRATEGY_LAYOUT_AWARE = 'layout'
      STRATEGY_TABLE_AWARE = 'table'
      STRATEGY_PRESERVE_FORMATTING = 'preserve'
      STRATEGY_OCR_ENHANCED = 'ocr'

      # Create extraction strategy
      # @param strategy_type [String] Strategy name
      # @return [Hash] Strategy handle
      def create_strategy(strategy_type = STRATEGY_SIMPLE)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Strategy type cannot be empty' if strategy_type.nil? || strategy_type.empty?

        strategy_utf8 = FFI::StringMarshaller.to_utf8(strategy_type)

        strategy_ptr = with_error_check('create_strategy', strategy: strategy_type) do |error_ptr|
          FFI::Bindings.pdf_create_extraction_strategy(@document.handle, strategy_utf8, error_ptr)
        end

        {
          handle: strategy_ptr,
          type: strategy_type,
          created_at: Time.now.to_i
        }
      end

      # Get strategy description
      # @param strategy [Hash] Strategy handle
      # @return [String] Description of the strategy
      def get_strategy_description(strategy)
        raise ::PdfOxide::ArgumentError, 'Invalid strategy' if strategy.nil? || strategy[:handle].nil?

        desc_ptr = with_error_check('get_strategy_description') do |error_ptr|
          FFI::Bindings.pdf_strategy_get_description(strategy[:handle], error_ptr)
        end

        FFI::StringMarshaller.from_c_string(desc_ptr) || ''
      end

      # Check if strategy recommends OCR
      # @param strategy [Hash] Strategy handle
      # @return [Boolean] Whether OCR is recommended
      def strategy_recommends_ocr?(strategy)
        raise ::PdfOxide::ArgumentError, 'Invalid strategy' if strategy.nil? || strategy[:handle].nil?

        with_error_check('strategy_recommends_ocr') do |error_ptr|
          FFI::Bindings.pdf_strategy_recommends_ocr(strategy[:handle], error_ptr)
        end
      end

      # Get available strategies
      # @return [Array<Hash>] List of available extraction strategies
      def get_available_strategies
        [
          { type: STRATEGY_SIMPLE, description: 'Basic text extraction' },
          { type: STRATEGY_LAYOUT_AWARE, description: 'Preserves layout structure' },
          { type: STRATEGY_TABLE_AWARE, description: 'Optimized for table detection' },
          { type: STRATEGY_PRESERVE_FORMATTING, description: 'Maintains formatting' },
          { type: STRATEGY_OCR_ENHANCED, description: 'Enhanced with OCR' }
        ]
      end

      # Use strategy to extract text from page
      # @param page_index [Integer] Page index (0-indexed)
      # @param strategy [Hash] Strategy handle
      # @return [String] Extracted text
      def extract_with_strategy(page_index, strategy)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Invalid strategy' if strategy.nil? || strategy[:handle].nil?

        # Extract using the strategy (would call Rust function)
        FFI::StringMarshaller.from_c_string(
          with_error_check('extract_with_strategy', page: page_index, strategy: strategy[:type]) do |error_ptr|
            FFI::Bindings.pdf_document_extract_text(@document.handle, page_index, error_ptr)
          end
        ) || ''
      end

      # List all strategy details
      # @return [Hash] Complete strategy information
      def strategy_statistics
        {
          available_strategies: get_available_strategies.length,
          strategies: get_available_strategies,
          timestamp: Time.now.to_i
        }
      end

      # Free strategy resources
      # @param strategy [Hash] Strategy handle
      # @return [Boolean] Whether cleanup succeeded
      def free_strategy(strategy)
        return false if strategy.nil? || strategy[:handle].nil?

        with_error_check('free_strategy') do |error_ptr|
          FFI::Bindings.pdf_strategy_free(strategy[:handle], error_ptr)
        end
      end
    end
  end
end
