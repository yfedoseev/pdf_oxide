# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents page dimensions with unit conversion
    class PageDimensions
      attr_reader :width, :height, :unit

      # Standard paper sizes
      PAPER_SIZES = {
        letter: { width: 8.5, height: 11.0, unit: 'in' },
        a4: { width: 210, height: 297, unit: 'mm' },
        a3: { width: 297, height: 420, unit: 'mm' },
        legal: { width: 8.5, height: 14.0, unit: 'in' }
      }.freeze

      # Initialize page dimensions
      # @param width [Float] Page width
      # @param height [Float] Page height
      # @param unit [String] Unit of measurement (pt, mm, in, cm)
      def initialize(width:, height:, unit: 'pt')
        @width = width.to_f
        @height = height.to_f
        @unit = unit.to_s.downcase
        validate_unit!
      end

      # Create dimensions from paper size name
      # @param name [Symbol, String] Paper size name
      # @return [PageDimensions] Dimensions object
      def self.from_paper_size(name)
        name_sym = name.is_a?(Symbol) ? name : name.to_s.downcase.to_sym
        size_info = PAPER_SIZES.fetch(name_sym) { raise ArgumentError, "Unknown paper size: #{name}" }
        new(**size_info)
      end

      # Convert to different unit
      # @param target_unit [String] Target unit
      # @return [PageDimensions] New dimensions in target unit
      def convert_to(target_unit)
        target_unit = target_unit.to_s.downcase
        validate_unit_conversion!(target_unit)

        # Convert to points first (base unit)
        width_pt = value_to_points(@width, @unit)
        height_pt = value_to_points(@height, @unit)

        # Then convert to target
        new_width = from_points(width_pt, target_unit)
        new_height = from_points(height_pt, target_unit)

        PageDimensions.new(width: new_width, height: new_height, unit: target_unit)
      end

      # Convert to inches
      # @return [PageDimensions] Dimensions in inches
      def to_inches
        convert_to('in')
      end

      # Convert to millimeters
      # @return [PageDimensions] Dimensions in millimeters
      def to_millimeters
        convert_to('mm')
      end

      # Convert to centimeters
      # @return [PageDimensions] Dimensions in centimeters
      def to_centimeters
        convert_to('cm')
      end

      # Convert to points
      # @return [PageDimensions] Dimensions in points
      def to_points
        convert_to('pt')
      end

      # Get aspect ratio (width / height)
      # @return [Float] Aspect ratio
      def aspect_ratio
        @width / @height
      end

      # Check if landscape orientation
      # @return [Boolean] Whether landscape
      def landscape?
        @width > @height
      end

      # Check if portrait orientation
      # @return [Boolean] Whether portrait
      def portrait?
        @width < @height
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        { width: @width, height: @height, unit: @unit }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "#{@width}x#{@height}#{@unit}"
      end

      # Check equality
      # @param other [PageDimensions] Other dimensions
      # @return [Boolean] Whether equal
      def ==(other)
        return false unless other.is_a?(PageDimensions)

        # Compare after converting both to points
        to_points.width == other.to_points.width &&
          to_points.height == other.to_points.height
      end

      alias eql? ==

      private

      def validate_unit!
        valid_units = %w[pt mm in cm]
        raise ArgumentError, "Invalid unit: #{@unit}" unless valid_units.include?(@unit)
      end

      def validate_unit_conversion!(target_unit)
        valid_units = %w[pt mm in cm]
        raise ArgumentError, "Invalid target unit: #{target_unit}" unless valid_units.include?(target_unit)
      end

      # NB: distinct name from the public no-arg #to_points to avoid a
      # Ruby method redefinition shadowing the public converter.
      def value_to_points(value, from_unit)
        case from_unit
        when 'pt'
          value
        when 'mm'
          value * 2.834645669 # 72 / 25.4
        when 'in'
          value * 72
        when 'cm'
          value * 28.346456693 # (72 / 2.54)
        else
          raise ArgumentError, "Unknown unit: #{from_unit}"
        end
      end

      def from_points(points, to_unit)
        case to_unit
        when 'pt'
          points
        when 'mm'
          points / 2.834645669
        when 'in'
          points / 72
        when 'cm'
          points / 28.346456693
        else
          raise ArgumentError, "Unknown unit: #{to_unit}"
        end
      end
    end
  end
end
