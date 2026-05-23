# frozen_string_literal: true

require_relative 'base'
require 'json'

module PdfOxide
  module Managers
    # Manager for PDF form field operations
    # Provides methods to enumerate, read, write, and manipulate form fields
    class Form < Base
      # FORM FIELD TYPES
      FIELD_TYPE_UNKNOWN = 0
      FIELD_TYPE_PUSHBUTTON = 1
      FIELD_TYPE_CHECKBOX = 2
      FIELD_TYPE_RADIOBUTTON = 3
      FIELD_TYPE_TEXT = 4
      FIELD_TYPE_COMBO = 5
      FIELD_TYPE_LISTBOX = 6
      FIELD_TYPE_SIGNATURE = 7

      FIELD_TYPES = {
        pushbutton: FIELD_TYPE_PUSHBUTTON,
        checkbox: FIELD_TYPE_CHECKBOX,
        radiobutton: FIELD_TYPE_RADIOBUTTON,
        text: FIELD_TYPE_TEXT,
        combo: FIELD_TYPE_COMBO,
        listbox: FIELD_TYPE_LISTBOX,
        signature: FIELD_TYPE_SIGNATURE
      }.freeze

      TYPE_NAMES = FIELD_TYPES.invert.freeze

      # Get count of form fields
      # @return [Integer] Number of form fields
      def form_field_count
        check_document!
        with_error_check('form_field_count') do |error_ptr|
          FFI::Bindings.pdf_document_get_form_field_count(@document.handle, error_ptr)
        end
      end

      # Check if document has AcroForm fields
      # @return [Boolean] Whether document has AcroForm fields
      def has_acro_forms?
        check_document!
        FFI::Bindings.pdf_document_has_acro_forms(@document.handle)
      end

      # Check if document has XFA form
      # @return [Boolean] Whether document has XFA form
      def has_xfa_forms?
        check_document!
        with_error_check('has_xfa') do |error_ptr|
          FFI::Bindings.pdf_document_has_xfa(@document.handle, error_ptr)
        end
      end

      # Get all form field names
      # @return [Array<String>] Array of field names
      def form_field_names
        check_document!
        names_handle = with_error_check('form_field_names') do |error_ptr|
          FFI::Bindings.pdf_document_get_form_field_names(@document.handle, error_ptr)
        end

        parse_form_field_names(names_handle)
      end

      # Get form field value
      # @param field_name [String] Name of form field
      # @return [String, nil] Field value
      def get_form_field_value(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_form_field_value', field: field_name) do |error_ptr|
            FFI::Bindings.pdf_document_get_form_field_value(@document.handle, field_name_utf8, error_ptr)
          end
        )
      end

      # Set form field value
      # @param field_name [String] Name of form field
      # @param value [String] New field value
      # @return [Boolean] Whether operation succeeded
      def set_form_field_value(field_name, value)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)
        value_utf8 = FFI::StringMarshaller.to_utf8(value.to_s)

        with_error_check('set_form_field_value', field: field_name, value: value) do |error_ptr|
          FFI::Bindings.pdf_document_set_form_field_value(
            @document.handle,
            field_name_utf8,
            value_utf8,
            error_ptr
          )
        end
        true
      end

      # Get form field type
      # @param field_name [String] Name of form field
      # @return [Symbol, Integer] Field type
      def get_form_field_type(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)

        type_int = with_error_check('get_form_field_type', field: field_name) do |error_ptr|
          FFI::Bindings.pdf_document_get_form_field_type(@document.handle, field_name_utf8, error_ptr)
        end

        TYPE_NAMES[type_int] || type_int
      end

      # Get form field flags
      # @param field_name [String] Name of form field
      # @return [Integer] Field flags
      def get_form_field_flags(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)

        with_error_check('get_form_field_flags', field: field_name) do |error_ptr|
          FFI::Bindings.pdf_document_get_form_field_flags(@document.handle, field_name_utf8, error_ptr)
        end
      end

      # Get all form fields
      # @return [Array<Types::FormField>] Array of form field information
      def get_all_form_fields
        check_document!
        names = form_field_names

        names.map do |name|
          Types::FormField.new(
            name: name,
            type: get_form_field_type(name),
            value: get_form_field_value(name),
            flags: get_form_field_flags(name)
          )
        end
      end

      # Reset form fields to default values
      # @param field_names [Array<String>, nil] Specific fields to reset or nil for all
      # @return [Boolean] Whether operation succeeded
      def reset_form_fields(field_names = nil)
        check_document!

        with_error_check('reset_form_fields') do |error_ptr|
          # API resets all fields, so we'll call it directly
          FFI::Bindings.pdf_document_reset_form_fields(@document.handle, error_ptr)
        end
        true
      end

      # Flatten form fields (make them read-only)
      # @return [Boolean] Whether operation succeeded
      def flatten_forms
        check_document!

        with_error_check('flatten_forms') do |error_ptr|
          FFI::Bindings.pdf_document_flatten_forms(@document.handle, error_ptr)
        end
        true
      end

      # Get all form field values as hash
      # @return [Hash] Field names mapped to values
      def get_all_values
        check_document!
        fields_hash = {}

        form_field_names.each do |name|
          fields_hash[name] = get_form_field_value(name)
        end

        fields_hash
      end

      # Set multiple form field values
      # @param values [Hash] Field names mapped to new values
      # @return [Boolean] Whether all operations succeeded
      def set_all_values(values)
        check_document!

        values.each do |field_name, value|
          set_form_field_value(field_name, value)
        end

        true
      end

      # Get form information
      # @return [Hash] Form information
      def form_info
        check_document!
        {
          has_acro_forms: has_acro_forms?,
          has_xfa_forms: has_xfa_forms?,
          field_count: form_field_count,
          fields: get_all_form_fields.map(&:to_h)
        }
      end

      # Export form data to JSON
      # @param output_path [String, nil] Path to save JSON or nil to return string
      # @return [String] JSON string
      def export_form_data(output_path = nil)
        check_document!
        form_data = get_all_values.to_json

        if output_path
          File.write(output_path, form_data)
          output_path
        else
          form_data
        end
      end

      # Import form data from JSON
      # @param input_path_or_json [String] File path or JSON string
      # @return [Boolean] Whether import succeeded
      def import_form_data(input_path_or_json)
        check_document!

        json_data = if File.exist?(input_path_or_json)
                      JSON.parse(File.read(input_path_or_json))
                    else
                      JSON.parse(input_path_or_json)
                    end

        set_all_values(json_data)
        true
      rescue JSON::ParserError => e
        raise ::PdfOxide::ParseError.new("Invalid JSON format: #{e.message}")
      end

      # Export form data to FDF (Forms Data Format) file
      #
      # FDF is a standard format for exchanging form data between PDF applications.
      # This method exports only the form field data, not the entire PDF structure.
      #
      # @param output_path [String] Path to save the FDF file
      # @return [Boolean] Whether export succeeded
      # @raise [ArgumentError] If output path is invalid
      # @example
      #   manager.export_to_fdf('/path/to/form_data.fdf')
      def export_to_fdf(output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be nil or empty' if output_path.nil? || output_path.empty?

        output_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('export_to_fdf', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_form_export_to_fdf(@document.handle, output_utf8, error_ptr)
        end
      end

      # Export form data to XFDF (XML Forms Data Format) file
      #
      # XFDF is an XML-based format for form data, annotations, and other PDF data.
      # It's a more modern alternative to FDF with better readability.
      #
      # @param output_path [String] Path to save the XFDF file
      # @return [Boolean] Whether export succeeded
      # @raise [ArgumentError] If output path is invalid
      # @example
      #   manager.export_to_xfdf('/path/to/form_data.xfdf')
      def export_to_xfdf(output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be nil or empty' if output_path.nil? || output_path.empty?

        output_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('export_to_xfdf', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_form_export_to_xfdf(@document.handle, output_utf8, error_ptr)
        end
      end

      # Import form data from FDF or XFDF file
      #
      # Automatically detects the file format (FDF or XFDF) and imports
      # the form field values into the document.
      #
      # @param input_path [String] Path to the FDF or XFDF file
      # @return [Boolean] Whether import succeeded
      # @raise [ArgumentError] If input path is invalid
      # @raise [FileNotFoundError] If file does not exist
      # @example
      #   manager.import_from_file('/path/to/form_data.xfdf')
      def import_from_file(input_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Input path cannot be nil or empty' if input_path.nil? || input_path.empty?
        raise ::PdfOxide::FileNotFoundError, "File not found: #{input_path}" unless File.exist?(input_path)

        input_utf8 = FFI::StringMarshaller.to_utf8(input_path)

        with_error_check('import_from_file', path: input_path) do |error_ptr|
          FFI::Bindings.pdf_form_import_from_file(@document.handle, input_utf8, error_ptr)
        end
      end

      # Reset all form fields to their default values
      #
      # This operation cannot be undone. All fields will be reset to the
      # values specified in the PDF's default value (DV) entries.
      #
      # @return [Boolean] Whether reset succeeded
      # @example
      #   manager.reset_all_fields
      def reset_all_fields
        check_document!

        with_error_check('reset_all_fields') do |error_ptr|
          FFI::Bindings.pdf_form_reset_all_fields(@document.handle, error_ptr)
        end
      end

      # Find form field by name
      #
      # Searches for a field by its fully-qualified name and returns its index.
      #
      # @param field_name [String] Name of the field to find
      # @return [Integer] Field index, or -1 if not found
      # @example
      #   index = manager.find_field_by_name('Address.City')
      def find_field_by_name(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be nil or empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)

        with_error_check('find_field_by_name', field: field_name) do |error_ptr|
          FFI::Bindings.pdf_form_field_find_by_name(@document.handle, field_name_utf8, error_ptr)
        end
      end

      # Set field value by name (type-aware)
      #
      # Sets a form field value using the appropriate type based on the value.
      # Strings are set as text, booleans are set directly.
      #
      # @param field_name [String] Name of the field
      # @param value [String, Boolean] Value to set
      # @return [Boolean] Whether operation succeeded
      # @example
      #   manager.set_field_value_by_name('Name', 'John Doe')
      #   manager.set_field_value_by_name('Checkbox1', true)
      def set_field_value_by_name(field_name, value)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be nil or empty' if field_name.nil? || field_name.empty?

        field_name_utf8 = FFI::StringMarshaller.to_utf8(field_name)

        if value.is_a?(TrueClass) || value.is_a?(FalseClass)
          with_error_check('set_field_value_by_name', field: field_name, value: value) do |error_ptr|
            FFI::Bindings.pdf_form_field_set_value_by_name_boolean(@document.handle, field_name_utf8, value, error_ptr)
          end
        else
          value_utf8 = FFI::StringMarshaller.to_utf8(value.to_s)
          with_error_check('set_field_value_by_name', field: field_name, value: value) do |error_ptr|
            FFI::Bindings.pdf_form_field_set_value_by_name_string(@document.handle, field_name_utf8, value_utf8, error_ptr)
          end
        end
      end

      # Get form field values by type
      # @param field_type [Symbol, Integer] Type of fields
      # @return [Hash] Fields of specified type with their values
      def get_fields_by_type(field_type)
        check_document!

        type_int = field_type.is_a?(Symbol) ? FIELD_TYPES.fetch(field_type) : field_type

        result = {}
        get_all_form_fields.each do |field|
          result[field.name] = get_form_field_value(field.name) if field.type == type_int
        end

        result
      end

      # Filter form fields
      # @param criteria [Hash] Filter criteria
      # @return [Array<Types::FormField>] Filtered fields
      def filter_form_fields(criteria = {})
        check_document!

        fields = get_all_form_fields

        fields.select do |field|
          (criteria[:name].nil? || field.name.include?(criteria[:name])) &&
            (criteria[:type].nil? || field.type == criteria[:type])
        end
      end

      # Get form statistics
      # @return [Hash] Form statistics
      def form_statistics
        check_document!
        fields = get_all_form_fields

        stats = {
          total_fields: fields.count,
          by_type: {},
          empty_fields: 0
        }

        fields.each do |field|
          type_name = TYPE_NAMES[field.type] || 'unknown'
          stats[:by_type][type_name] ||= 0
          stats[:by_type][type_name] += 1

          stats[:empty_fields] += 1 if field.value.nil? || field.value.empty?
        end

        stats
      end

      private

      def parse_form_field_names(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_form_field_count(handle)

          names = count.times.map do |i|
            name_ptr = FFI::Bindings.pdf_oxide_form_field_get_name(handle, i)
            FFI::StringMarshaller.read_c_string(name_ptr)
          end

          names
        ensure
          FFI::Bindings.pdf_oxide_form_field_list_free(handle) unless handle.nil? || handle.null?
        end
      end
    end
  end
end
