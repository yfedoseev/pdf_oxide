# frozen_string_literal: true

module PdfOxide
  module Types
    # Options for page rendering
    class RenderOptions
      attr_accessor :dpi, :format, :quality, :anti_alias, :background_color,
                    :color_space, :apply_color_profile, :max_width, :max_height

      # Predefined quality presets
      PRESETS = {
        draft: { dpi: 72, quality: 60, format: :png },
        normal: { dpi: 150, quality: 80, format: :png },
        high: { dpi: 300, quality: 95, format: :png },
        print: { dpi: 600, quality: 95, format: :png }
      }.freeze

      # Initialize render options
      # @param dpi [Integer] Resolution in dots per inch
      # @param format [Symbol] Output format (:png, :jpeg, :webp)
      # @param quality [Integer] Quality (0-100)
      # @param anti_alias [Boolean] Enable anti-aliasing
      # @param background_color [Integer] Background color as hex (0xFFFFFF)
      # @param color_space [Symbol] Color space (:srgb, :device_rgb, :linear_rgb)
      # @param apply_color_profile [Boolean] Apply color profile
      # @param max_width [Integer, nil] Maximum width in pixels
      # @param max_height [Integer, nil] Maximum height in pixels
      def initialize(
        dpi: 150,
        format: :png,
        quality: 80,
        anti_alias: true,
        background_color: 0xFFFFFF,
        color_space: :srgb,
        apply_color_profile: true,
        max_width: nil,
        max_height: nil
      )
        @dpi = dpi.to_i
        @format = format.is_a?(Symbol) ? format : format.to_sym
        @quality = quality.to_i
        @anti_alias = anti_alias
        @background_color = background_color.is_a?(Integer) ? background_color : background_color.to_i(16)
        @color_space = color_space.is_a?(Symbol) ? color_space : color_space.to_sym
        @apply_color_profile = apply_color_profile
        @max_width = max_width&.to_i
        @max_height = max_height&.to_i

        validate!
      end

      # Create preset render options
      # @param preset [Symbol] Preset name
      # @return [RenderOptions] Configured options
      def self.preset(preset_name)
        preset_name_sym = preset_name.is_a?(Symbol) ? preset_name : preset_name.to_sym
        preset_config = PRESETS.fetch(preset_name_sym) do
          raise ArgumentError, "Unknown preset: #{preset_name}"
        end
        new(**preset_config)
      end

      # Convenient class methods for presets
      def self.draft
        preset(:draft)
      end

      def self.normal
        preset(:normal)
      end

      def self.high
        preset(:high)
      end

      def self.print
        preset(:print)
      end

      # Set quality for JPEG rendering
      # @param quality [Integer] JPEG quality (0-100)
      # @return [self]
      def jpeg_quality(quality)
        @quality = quality.to_i
        self
      end

      # Set maximum dimensions
      # @param width [Integer] Max width
      # @param height [Integer] Max height
      # @return [self]
      def max_dimensions(width, height)
        @max_width = width.to_i
        @max_height = height.to_i
        self
      end

      # Convert to hash for FFI
      # @return [Hash] Hash representation
      def to_h
        {
          dpi: @dpi,
          format: @format,
          quality: @quality,
          anti_alias: @anti_alias,
          background_color: @background_color,
          color_space: @color_space,
          apply_color_profile: @apply_color_profile,
          max_width: @max_width,
          max_height: @max_height
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "RenderOptions(#{@dpi}dpi, #{@format}, quality=#{@quality})"
      end

      private

      def validate!
        raise ArgumentError, "DPI must be positive: #{@dpi}" if @dpi <= 0
        raise ArgumentError, "Quality must be 0-100: #{@quality}" if @quality < 0 || @quality > 100
        raise ArgumentError, "Invalid format: #{@format}" unless %i[png jpeg webp].include?(@format)
      end
    end
  end
end
