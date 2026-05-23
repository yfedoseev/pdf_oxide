# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a form field in a PDF
    class FormField
      attr_reader :name, :value, :type, :flags, :options

      # Field types
      TYPES = {
        text: 0,
        checkbox: 1,
        radio_button: 2,
        dropdown: 3,
        list_box: 4,
        signature: 5,
        button: 6
      }.freeze

      # Field flags
      FLAGS = {
        read_only: 1,
        required: 2,
        no_export: 4,
        multi_line: 0x1000,
        password: 0x2000,
        no_spellcheck: 0x4000,
        combo_box: 0x20000,
        edit: 0x40000,
        sort: 0x80000,
        radio_never_together: 0x4000000,
        multi_select: 0x200000
      }.freeze

      # Initialize form field
      # @param name [String] Field name
      # @param value [String] Field value
      # @param type [Integer, Symbol] Field type
      # @param flags [Integer] Field flags
      # @param options [Array<String>] Options for dropdown/radio
      def initialize(name:, value: nil, type: :text, flags: 0, options: [])
        @name = name.to_s
        @value = value
        @type = normalize_type(type)
        @flags = flags.to_i
        @options = options || []
      end

      # Get field type name
      # @return [Symbol] Type name
      def type_name
        TYPES.invert[@type]
      end

      # Check if read-only
      # @return [Boolean]
      def read_only?
        (@flags & FLAGS[:read_only]) != 0
      end

      # Check if required
      # @return [Boolean]
      def required?
        (@flags & FLAGS[:required]) != 0
      end

      # Check if multi-line text
      # @return [Boolean]
      def multi_line?
        (@flags & FLAGS[:multi_line]) != 0
      end

      # Check if password field
      # @return [Boolean]
      def password?
        (@flags & FLAGS[:password]) != 0
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          name: @name,
          value: @value,
          type: type_name,
          flags: @flags,
          options: @options,
          read_only: read_only?,
          required: required?
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        "FormField(#{@name}, type=#{type_name}, value='#{@value}')"
      end

      private

      def normalize_type(type)
        if type.is_a?(Symbol)
          TYPES.fetch(type) { raise ArgumentError, "Unknown type: #{type}" }
        else
          type.to_i
        end
      end
    end
  end
end
