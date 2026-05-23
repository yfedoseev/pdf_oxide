# frozen_string_literal: true

module PdfOxide
  # Optimization Manager for PDF file size reduction operations
  #
  # Provides font subsetting, image downsampling, content deduplication,
  # and combined optimization for reducing PDF file size.
  class OptimizationManager
    # Full optimization result record
    OptimizationResult = Struct.new(
      :bytes_saved,
      :details,
      keyword_init: true
    )

    attr_reader :document

    # Initialize OptimizationManager with a PDF document
    # @param document [Object] PDF document handle
    def initialize(document)
      @document = document
    end

    # Subset fonts to remove unused glyphs
    # @return [Integer] estimated bytes saved
    # @raise [PdfOxide::OptimizationError] if the operation fails
    def subset_fonts
      FFI::ErrorHandler.with_int_check('optimize_subset_fonts') do |err|
        Bindings.pdf_optimize_subset_fonts(@document, err)
      end
    end

    # Downsample images to reduce file size
    # @param target_dpi [Integer] target DPI for downsampling (default: 150)
    # @param quality [Integer] JPEG quality 1-100 (default: 85)
    # @return [Integer] estimated bytes saved
    # @raise [PdfOxide::OptimizationError] if the operation fails
    def downsample_images(target_dpi: 150, quality: 85)
      FFI::ErrorHandler.with_int_check('optimize_downsample_images') do |err|
        Bindings.pdf_optimize_downsample_images(@document, target_dpi, quality, err)
      end
    end

    # Deduplicate identical content streams and objects
    # @return [Integer] estimated bytes saved
    # @raise [PdfOxide::OptimizationError] if the operation fails
    def deduplicate
      FFI::ErrorHandler.with_int_check('optimize_deduplicate') do |err|
        Bindings.pdf_optimize_deduplicate(@document, err)
      end
    end

    # Run full optimization pipeline (fonts + images + dedup)
    # @param target_dpi [Integer] target DPI for image downsampling (default: 150)
    # @param quality [Integer] JPEG quality 1-100 (default: 85)
    # @return [OptimizationResult] result with total bytes saved
    # @raise [PdfOxide::OptimizationError] if the operation fails
    def optimize_full(target_dpi: 150, quality: 85)
      result_handle = FFI::ErrorHandler.with_error_check('optimize_full') do |err|
        Bindings.pdf_optimize_full(@document, target_dpi, quality, err)
      end

      begin
        bytes_saved = Bindings.pdf_optimization_result_bytes_saved(result_handle).to_i

        OptimizationResult.new(
          bytes_saved: bytes_saved,
          details: ''
        )
      ensure
        Bindings.pdf_optimization_result_free(result_handle) if result_handle && !result_handle.null?
      end
    end

    # Get optimization capabilities summary
    # @return [Hash] summary of optimization capabilities
    def summary
      {
        capabilities: {
          subset_fonts: true,
          downsample_images: true,
          deduplicate: true,
          full_optimization: true
        }
      }
    end
  end
end
