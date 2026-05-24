# frozen_string_literal: true

module PdfOxide
  module Types
    # OCR configuration builder for customizing OCR engine behavior
    # Supports detection thresholds, recognition settings, GPU acceleration, and language configuration
    class OcrConfig
      # Default configuration values
      DEFAULTS = {
        detection_threshold: 0.5,
        recognition_threshold: 0.5,
        max_side_len: 960,
        use_gpu: false,
        gpu_device_id: 0,
        languages: ['en'],
        page_processing_mode: :full # :full, :text_only, :image_only
      }.freeze

      attr_reader :detection_threshold, :recognition_threshold, :max_side_len, :use_gpu,
                  :gpu_device_id, :languages, :page_processing_mode

      def initialize(detection_threshold: DEFAULTS[:detection_threshold],
                     recognition_threshold: DEFAULTS[:recognition_threshold],
                     max_side_len: DEFAULTS[:max_side_len],
                     use_gpu: DEFAULTS[:use_gpu],
                     gpu_device_id: DEFAULTS[:gpu_device_id],
                     languages: DEFAULTS[:languages],
                     page_processing_mode: DEFAULTS[:page_processing_mode])
        @detection_threshold = validate_threshold!(detection_threshold, 'detection_threshold')
        @recognition_threshold = validate_threshold!(recognition_threshold, 'recognition_threshold')
        @max_side_len = validate_positive_integer!(max_side_len, 'max_side_len')
        @use_gpu = use_gpu
        @gpu_device_id = validate_non_negative_integer!(gpu_device_id, 'gpu_device_id')
        @languages = Array(languages).map(&:to_s).freeze
        @page_processing_mode = validate_processing_mode!(page_processing_mode)
      end

      # Create builder-friendly config with detection threshold
      # @param threshold [Float] Detection threshold (0.0-1.0)
      # @return [OcrConfig] Updated config
      def with_detection_threshold(threshold)
        OcrConfig.new(
          detection_threshold: threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Set recognition threshold
      # @param threshold [Float] Recognition threshold (0.0-1.0)
      # @return [OcrConfig] Updated config
      def with_recognition_threshold(threshold)
        OcrConfig.new(
          detection_threshold: @detection_threshold,
          recognition_threshold: threshold,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Set both detection and recognition thresholds
      # @param detection [Float] Detection threshold (0.0-1.0)
      # @param recognition [Float] Recognition threshold (0.0-1.0)
      # @return [OcrConfig] Updated config
      def with_thresholds(detection, recognition)
        OcrConfig.new(
          detection_threshold: detection,
          recognition_threshold: recognition,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Set maximum side length
      # @param length [Integer] Maximum side length in pixels
      # @return [OcrConfig] Updated config
      def with_max_side_len(length)
        OcrConfig.new(
          detection_threshold: @detection_threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: length,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Enable or disable GPU acceleration
      # @param enabled [Boolean] Whether to use GPU
      # @param device_id [Integer] GPU device ID (default: 0)
      # @return [OcrConfig] Updated config
      def with_gpu(enabled = true, device_id = 0)
        OcrConfig.new(
          detection_threshold: @detection_threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: @max_side_len,
          use_gpu: enabled,
          gpu_device_id: device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Set OCR languages
      # @param langs [Array<String>, String] Language codes (e.g., 'en', 'es', 'fr')
      # @return [OcrConfig] Updated config
      def with_languages(langs)
        OcrConfig.new(
          detection_threshold: @detection_threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: Array(langs),
          page_processing_mode: @page_processing_mode
        )
      end

      # Builder: Set page processing mode
      # @param mode [Symbol] Processing mode (:full, :text_only, :image_only)
      # @return [OcrConfig] Updated config
      def with_processing_mode(mode)
        OcrConfig.new(
          detection_threshold: @detection_threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: mode
        )
      end

      # Create preset configuration for balanced accuracy/speed
      # @return [OcrConfig] Balanced config
      def self.balanced
        new(
          detection_threshold: 0.5,
          recognition_threshold: 0.5,
          max_side_len: 960,
          use_gpu: false
        )
      end

      # Create preset configuration for high accuracy (slower)
      # @return [OcrConfig] High accuracy config
      def self.high_accuracy
        new(
          detection_threshold: 0.7,
          recognition_threshold: 0.7,
          max_side_len: 1280,
          use_gpu: true
        )
      end

      # Create preset configuration for high speed (lower accuracy)
      # @return [OcrConfig] High speed config
      def self.fast
        new(
          detection_threshold: 0.3,
          recognition_threshold: 0.3,
          max_side_len: 640,
          use_gpu: false
        )
      end

      # Create preset configuration for low resource usage
      # @return [OcrConfig] Low resource config
      def self.low_resource
        new(
          detection_threshold: 0.4,
          recognition_threshold: 0.4,
          max_side_len: 512,
          use_gpu: false
        )
      end

      # Convert configuration to hash
      # @return [Hash] Configuration as hash
      def to_h
        {
          detection_threshold: @detection_threshold,
          recognition_threshold: @recognition_threshold,
          max_side_len: @max_side_len,
          use_gpu: @use_gpu,
          gpu_device_id: @gpu_device_id,
          languages: @languages,
          page_processing_mode: @page_processing_mode
        }
      end

      # Convert configuration to string representation
      # @return [String] String representation
      def to_s
        "OcrConfig(detection=#{@detection_threshold}, recognition=#{@recognition_threshold}, " \
        "gpu=#{@use_gpu}, languages=#{@languages.join(',')})"
      end

      # Compare configurations for equality
      # @param other [OcrConfig] Configuration to compare
      # @return [Boolean] Whether configurations are equal
      def ==(other)
        other.is_a?(OcrConfig) &&
          detection_threshold == other.detection_threshold &&
          recognition_threshold == other.recognition_threshold &&
          max_side_len == other.max_side_len &&
          use_gpu == other.use_gpu &&
          gpu_device_id == other.gpu_device_id &&
          languages == other.languages &&
          page_processing_mode == other.page_processing_mode
      end

      # Hash code for use in collections
      # @return [Integer] Hash code
      def hash
        [detection_threshold, recognition_threshold, max_side_len, use_gpu, gpu_device_id, languages, page_processing_mode].hash
      end

      private

      def validate_threshold!(value, name)
        value = value.to_f
        raise ArgumentError, "#{name} must be between 0.0 and 1.0, got #{value}" unless value >= 0.0 && value <= 1.0

        value
      end

      def validate_positive_integer!(value, name)
        value = value.to_i
        raise ArgumentError, "#{name} must be positive, got #{value}" unless value.positive?

        value
      end

      def validate_non_negative_integer!(value, name)
        value = value.to_i
        raise ArgumentError, "#{name} must be non-negative, got #{value}" unless value >= 0

        value
      end

      def validate_processing_mode!(mode)
        mode_sym = mode.is_a?(Symbol) ? mode : mode.to_sym
        valid_modes = %i[full text_only image_only]
        raise ArgumentError, "Invalid processing mode: #{mode_sym}" unless valid_modes.include?(mode_sym)

        mode_sym
      end
    end
  end
end
