# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for page rendering operations
    class Rendering < Base
      # Render page to file
      # @param page_index [Integer] Page index (0-indexed)
      # @param output_path [String] Output file path
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [Boolean] Success status
      def render_page_to_file(page_index, output_path, options = {})
        check_document!
        validate_page_index!(page_index)

        render_opts = normalize_render_options(options)
        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('render_page_to_file', page: page_index, output: output_path) do |error_ptr|
          # Format: 0=png, 1=jpeg, 2=webp
          format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

          FFI::Bindings.pdf_render_page_to_file(
            @document.handle,
            page_index,
            output_path_utf8,
            format_int,
            error_ptr
          )
        end
      end

      # Render page to bytes
      # @param page_index [Integer] Page index
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [String] Image bytes
      def render_page_to_bytes(page_index, options = {})
        check_document!
        validate_page_index!(page_index)

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        image_handle = with_error_check('render_page_to_bytes', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_render_page_to_bytes(
            @document.handle,
            page_index,
            format_int,
            error_ptr
          )
        end

        extract_image_bytes(image_handle)
      end

      # Render page range to files
      # @param start_page [Integer] Start page index
      # @param end_page [Integer] End page index
      # @param output_dir [String] Output directory
      # @param options [Hash] Rendering options
      # @return [Integer] Number of pages rendered
      def render_page_range(start_page, end_page, output_dir, options = {})
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        raise ::PdfOxide::ArgumentError, 'start_page must be <= end_page' if start_page > end_page

        render_opts = normalize_render_options(options)
        output_dir_utf8 = FFI::StringMarshaller.to_utf8(output_dir)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        with_error_check('render_page_range', start: start_page, end: end_page) do |error_ptr|
          FFI::Bindings.pdf_render_page_range(
            @document.handle,
            start_page,
            end_page,
            output_dir_utf8,
            format_int,
            error_ptr
          )
        end
      end

      # Render all pages
      # @param output_dir [String] Output directory
      # @param options [Hash] Rendering options
      # @return [Integer] Number of pages rendered
      def render_all(output_dir, options = {})
        render_page_range(0, @document.page_count - 1, output_dir, options)
      end

      # Render page thumbnail
      # @param page_index [Integer] Page index
      # @param max_size [Integer] Maximum thumbnail size
      # @param options [Hash] Rendering options
      # @return [String] Thumbnail image bytes
      def render_thumbnail(page_index, max_size = 100, _options = {})
        check_document!
        validate_page_index!(page_index)

        image_handle = with_error_check('render_thumbnail', page: page_index, size: max_size) do |error_ptr|
          FFI::Bindings.pdf_render_page_thumbnail(
            @document.handle,
            page_index,
            max_size.to_i,
            error_ptr
          )
        end

        extract_image_bytes(image_handle)
      end

      # Get page dimensions
      # @param page_index [Integer] Page index
      # @return [Types::PageDimensions] Page dimensions
      def page_dimensions(page_index)
        check_document!
        validate_page_index!(page_index)

        width = with_error_check('get_page_width', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_width(@document.handle, page_index, error_ptr)
        end

        height = with_error_check('get_page_height', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_page_height(@document.handle, page_index, error_ptr)
        end

        Types::PageDimensions.new(width: width, height: height, unit: 'pt')
      end

      # Render with zoom
      # @param page_index [Integer] Page index
      # @param zoom_level [Float] Zoom level
      # @param options [Hash] Rendering options
      # @return [String] Image bytes
      def render_with_zoom(page_index, zoom_level, options = {})
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Zoom level must be positive' if zoom_level <= 0

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        image_handle = with_error_check('render_with_zoom', page: page_index, zoom: zoom_level) do |error_ptr|
          FFI::Bindings.pdf_render_page_zoom(
            @document.handle,
            page_index,
            zoom_level.to_f,
            format_int,
            error_ptr
          )
        end

        extract_image_bytes(image_handle)
      end

      # Render specific region of a page
      # @param page_index [Integer] Page index (0-indexed)
      # @param region [Hash] Region with :x, :y, :width, :height keys
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [String] Image bytes
      def render_region(page_index, region, options = {})
        check_document!
        validate_page_index!(page_index)
        validate_region!(region)

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        image_handle = with_error_check('render_region', page: page_index, region: region) do |error_ptr|
          FFI::Bindings.pdf_render_page_region(
            @document.handle,
            page_index,
            region[:x].to_f,
            region[:y].to_f,
            region[:width].to_f,
            region[:height].to_f,
            format_int,
            error_ptr
          )
        end

        extract_image_bytes(image_handle)
      end

      # Render page fitted to maximum dimensions
      # @param page_index [Integer] Page index (0-indexed)
      # @param max_width [Integer] Maximum width in pixels
      # @param max_height [Integer] Maximum height in pixels
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [String] Image bytes
      def render_fit(page_index, max_width, max_height, options = {})
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'max_width must be positive' if max_width.to_i <= 0
        raise ::PdfOxide::ArgumentError, 'max_height must be positive' if max_height.to_i <= 0

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        image_handle = with_error_check('render_fit', page: page_index, max_width: max_width, max_height: max_height) do |error_ptr|
          FFI::Bindings.pdf_render_page_fit(
            @document.handle,
            page_index,
            max_width.to_i,
            max_height.to_i,
            format_int,
            error_ptr
          )
        end

        extract_image_bytes(image_handle)
      end

      # Render page and return as Base64-encoded string
      # @param page_index [Integer] Page index (0-indexed)
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [String] Base64-encoded image string
      def render_to_base64(page_index, options = {})
        check_document!
        validate_page_index!(page_index)

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        with_error_check('render_to_base64', page: page_index) do |error_ptr|
          base64_ptr = FFI::Bindings.pdf_render_page_to_base64(
            @document.handle,
            page_index,
            format_int,
            error_ptr
          )
          return FFI::StringMarshaller.from_c(base64_ptr)
        end
      end

      # Render region and return as Base64-encoded string
      # @param page_index [Integer] Page index (0-indexed)
      # @param region [Hash] Region with :x, :y, :width, :height keys
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [String] Base64-encoded image string
      def render_region_to_base64(page_index, region, options = {})
        check_document!
        validate_page_index!(page_index)
        validate_region!(region)

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        with_error_check('render_region_to_base64', page: page_index, region: region) do |error_ptr|
          base64_ptr = FFI::Bindings.pdf_rendered_image_to_base64(
            @document.handle,
            page_index,
            region[:x].to_f,
            region[:y].to_f,
            region[:width].to_f,
            region[:height].to_f,
            format_int,
            error_ptr
          )
          return FFI::StringMarshaller.from_c(base64_ptr)
        end
      end

      # Estimate rendering time for a page
      # @param page_index [Integer] Page index (0-indexed)
      # @param options [Hash, Types::RenderOptions] Rendering options
      # @return [Integer] Estimated time in milliseconds
      def estimate_render_time(page_index, options = {})
        check_document!
        validate_page_index!(page_index)

        render_opts = normalize_render_options(options)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, render_opts.format)

        with_error_check('estimate_render_time', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_estimate_render_time(
            @document.handle,
            page_index,
            format_int,
            render_opts.dpi.to_i,
            error_ptr
          )
        end
      end

      # Get renderer statistics
      # @return [Hash] Statistics with :pages_rendered, :total_time_ms, :avg_time_per_page_ms
      def renderer_statistics
        check_document!

        with_error_check('get_renderer_statistics') do |error_ptr|
          stats_ptr = FFI::Bindings.pdf_renderer_get_statistics(@document.handle, error_ptr)
          parse_renderer_statistics(stats_ptr)
        end
      end

      # Reset renderer statistics
      # @return [Boolean] Success status
      def reset_renderer_statistics
        check_document!

        with_error_check('reset_renderer_statistics') do |error_ptr|
          FFI::Bindings.pdf_renderer_reset_statistics(@document.handle, error_ptr)
        end
      end

      # Convert image bytes to a different format
      # @param image_bytes [String] Image data
      # @param from_format [Symbol] Source format (:png, :jpeg, :webp)
      # @param to_format [Symbol] Target format (:png, :jpeg, :webp)
      # @return [String] Converted image bytes
      def convert_image_format(image_bytes, from_format, to_format)
        raise ::PdfOxide::ArgumentError, 'image_bytes cannot be nil' if image_bytes.nil?
        raise ::PdfOxide::ArgumentError, 'from_format is required' if from_format.nil?
        raise ::PdfOxide::ArgumentError, 'to_format is required' if to_format.nil?

        from_fmt_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, from_format)
        to_fmt_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, to_format)

        with_error_check('convert_image_format', from: from_format, to: to_format) do |error_ptr|
          converted_handle = FFI::Bindings.pdf_rendered_image_convert(
            image_bytes,
            image_bytes.bytesize,
            from_fmt_int,
            to_fmt_int,
            error_ptr
          )
          extract_image_bytes(converted_handle)
        end
      end

      # Convert image bytes to Base64
      # @param image_bytes [String] Image data
      # @param format [Symbol] Image format (:png, :jpeg, :webp)
      # @return [String] Base64-encoded image
      def image_to_base64(image_bytes, format)
        raise ::PdfOxide::ArgumentError, 'image_bytes cannot be nil' if image_bytes.nil?
        raise ::PdfOxide::ArgumentError, 'format is required' if format.nil?

        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, format)

        with_error_check('image_to_base64', format: format) do |error_ptr|
          base64_ptr = FFI::Bindings.pdf_rendered_image_to_base64(
            image_bytes,
            image_bytes.bytesize,
            format_int,
            error_ptr
          )
          return FFI::StringMarshaller.from_c(base64_ptr)
        end
      end

      # Get MIME type for image format
      # @param format [Symbol] Image format (:png, :jpeg, :webp)
      # @return [String] MIME type (e.g., 'image/png')
      def mime_type_for(format)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, format)
        mime_ptr = FFI::Bindings.pdf_image_format_mime_type(format_int)
        FFI::StringMarshaller.from_c(mime_ptr)
      end

      # Get file extension for image format
      # @param format [Symbol] Image format (:png, :jpeg, :webp)
      # @return [String] File extension (e.g., '.png')
      def extension_for(format)
        format_int = FFI::Types.enum_to_value(FFI::Types::IMAGE_FORMATS, format)
        ext_ptr = FFI::Bindings.pdf_image_format_extension(format_int)
        FFI::StringMarshaller.from_c(ext_ptr)
      end

      # Create a page renderer with custom options
      # @param options [Hash] Renderer options including :dpi, :format, :quality, :background_color
      # @return [Hash] Renderer handle and metadata
      def create_page_renderer(options = {})
        check_document!

        default_opts = { dpi: 150, format: 'png', quality: 85, background_color: '#FFFFFF' }
        opts = default_opts.merge(options)

        renderer_ptr = with_error_check('create_page_renderer', options: opts) do |error_ptr|
          FFI::Bindings.pdf_page_renderer_create(@document.handle, error_ptr)
        end

        {
          handle: renderer_ptr,
          options: opts,
          created_at: Time.now.to_i
        }
      end

      # Set renderer options
      # @param renderer [Hash] Renderer handle from create_page_renderer
      # @param options [Hash] Options to set
      # @return [Boolean] Whether options were set
      def set_renderer_options(renderer, options = {})
        check_document!
        raise ::PdfOxide::ArgumentError, 'Invalid renderer' if renderer.nil? || renderer[:handle].nil?

        with_error_check('set_renderer_options', options: options) do |error_ptr|
          FFI::Bindings.pdf_page_renderer_set_options(@document.handle, renderer[:handle], options, error_ptr)
        end
      end

      # Render page with custom renderer
      # @param page_index [Integer] Page index (0-indexed)
      # @param renderer [Hash] Renderer from create_page_renderer
      # @return [Hash] Rendered image information
      def render_with_renderer(page_index, renderer)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Invalid renderer' if renderer.nil? || renderer[:handle].nil?

        image_ptr = with_error_check('render_with_renderer', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_render_page_to_bytes(@document.handle, page_index, renderer[:options][:dpi] || 150, error_ptr)
        end

        {
          data: image_ptr,
          page: page_index,
          format: renderer[:options][:format],
          dpi: renderer[:options][:dpi],
          timestamp: Time.now.to_i
        }
      end

      # Get rendered image dimensions
      # @param image_data [Hash] Image data from render functions
      # @return [Hash] Width and height
      def get_image_dimensions(image_data)
        raise ::PdfOxide::ArgumentError, 'Invalid image data' if image_data.nil? || image_data[:data].nil?

        width = with_error_check('get_image_width') do |error_ptr|
          FFI::Bindings.pdf_rendered_image_width(image_data[:data], error_ptr)
        end

        height = with_error_check('get_image_height') do |error_ptr|
          FFI::Bindings.pdf_rendered_image_height(image_data[:data], error_ptr)
        end

        { width: width, height: height }
      end

      # Save rendered image to file
      # @param image_data [Hash] Image data from render functions
      # @param output_path [String] File path to save to
      # @return [Boolean] Whether save succeeded
      def save_rendered_image(image_data, output_path)
        raise ::PdfOxide::ArgumentError, 'Invalid image data' if image_data.nil? || image_data[:data].nil?
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('save_rendered_image', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_rendered_image_save(image_data[:data], output_path_utf8, error_ptr)
        end
      end

      private

      def normalize_render_options(options)
        options.is_a?(Types::RenderOptions) ? options : Types::RenderOptions.new(**options)
      end

      def validate_region!(region)
        raise ::PdfOxide::ArgumentError, 'region must be a Hash' unless region.is_a?(Hash)
        raise ::PdfOxide::ArgumentError, 'region must have :x key' unless region.key?(:x)
        raise ::PdfOxide::ArgumentError, 'region must have :y key' unless region.key?(:y)
        raise ::PdfOxide::ArgumentError, 'region must have :width key' unless region.key?(:width)
        raise ::PdfOxide::ArgumentError, 'region must have :height key' unless region.key?(:height)
        raise ::PdfOxide::ArgumentError, 'region :width must be positive' if region[:width].to_f <= 0
        raise ::PdfOxide::ArgumentError, 'region :height must be positive' if region[:height].to_f <= 0
      end

      def extract_image_bytes(image_handle)
        return nil if image_handle.nil? || image_handle.null?

        begin
          size = FFI::Bindings.pdf_rendered_image_size(image_handle)
          data_ptr = FFI::Bindings.pdf_rendered_image_data(image_handle)
          data_ptr.read_bytes(size)
        ensure
          FFI::Bindings.pdf_rendered_image_free(image_handle)
        end
      end

      def parse_renderer_statistics(stats_ptr)
        return {} if stats_ptr.nil? || stats_ptr.null?

        {
          pages_rendered: FFI::Bindings.pdf_renderer_statistics_pages_rendered(stats_ptr),
          total_time_ms: FFI::Bindings.pdf_renderer_statistics_total_time(stats_ptr),
          avg_time_per_page_ms: FFI::Bindings.pdf_renderer_statistics_avg_time(stats_ptr)
        }
      end
    end
  end
end
