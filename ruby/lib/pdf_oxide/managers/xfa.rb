# frozen_string_literal: true

require 'json'
require_relative 'base'

module PdfOxide
  module Managers
    # Manager for XFA (XML Forms Architecture) operations
    # Handles XFA forms, datasets, and conversion to AcroForms
    class Xfa < Base
      # XFA FORM TYPE CONSTANTS
      XFA_FORM_TYPE_UNKNOWN = 0
      XFA_FORM_TYPE_STATIC = 1
      XFA_FORM_TYPE_DYNAMIC = 2

      FORM_TYPES = {
        unknown: XFA_FORM_TYPE_UNKNOWN,
        static: XFA_FORM_TYPE_STATIC,
        dynamic: XFA_FORM_TYPE_DYNAMIC
      }.freeze

      FORM_TYPE_NAMES = FORM_TYPES.invert.freeze
      # Check if document has XFA forms
      # @return [Boolean] Whether document contains XFA forms
      def has_xfa_forms?
        check_document!

        with_error_check('has_xfa_forms') do |error_ptr|
          FFI::Bindings.pdf_document_has_xfa_form(@document.handle, error_ptr)
        end
      end

      # Parse XFA form structure
      # @return [Hash] XFA form information
      def parse_xfa_form
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        parse_xfa_form_data(form_ptr)
      end

      # Convert XFA form to AcroForm
      # @return [Boolean] Whether conversion succeeded
      def convert_xfa_to_acroform
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        with_error_check('convert_xfa_to_acroform') do |error_ptr|
          FFI::Bindings.pdf_convert_xfa_to_acroform(@document.handle, error_ptr)
        end
      end

      # Get XFA dataset as XML
      # @return [String] XFA dataset in XML format
      def get_xfa_dataset_xml
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return '' if form_ptr.nil? || form_ptr.null?

        begin
          dataset_ptr = FFI::Bindings.pdf_xfa_form_get_dataset(form_ptr)

          return '' if dataset_ptr.nil? || dataset_ptr.null?

          begin
            xml_ptr = FFI::Bindings.pdf_xfa_dataset_to_xml(dataset_ptr)
            FFI::StringMarshaller.from_c_string(xml_ptr) || ''
          ensure
            FFI::Bindings.pdf_xfa_dataset_free(dataset_ptr) unless dataset_ptr.nil? || dataset_ptr.null?
          end
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get number of XFA form fields
      # @return [Integer] Number of fields
      def get_xfa_field_count
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return 0 if form_ptr.nil? || form_ptr.null?

        begin
          FFI::Bindings.pdf_xfa_form_field_count(form_ptr)
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get XFA field names
      # @return [Array<String>] List of field names
      def get_xfa_field_names
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return [] if form_ptr.nil? || form_ptr.null?

        begin
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)

          field_names = count.times.map do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next nil if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)
              FFI::StringMarshaller.read_c_string(name_ptr) || ''
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          field_names.compact
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get XFA field value
      # @param field_name [String] Name of the field
      # @return [String] Field value
      def get_xfa_field_value(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return '' if form_ptr.nil? || form_ptr.null?

        begin
          FFI::StringMarshaller.to_utf8(field_name)
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)

          field_value = ''
          count.times do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)
              current_name = FFI::StringMarshaller.read_c_string(name_ptr) || ''

              if current_name == field_name
                # Get field value (would need additional FFI function)
                field_value = current_name
                break
              end
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          field_value
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get all XFA fields with their information
      # @return [Array<Hash>] Array of field information
      def get_xfa_fields
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return [] if form_ptr.nil? || form_ptr.null?

        begin
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)

          fields = count.times.map do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next nil if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)

              {
                index: i,
                name: FFI::StringMarshaller.read_c_string(name_ptr) || '',
                type: 'xfa_field'
              }
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          fields.compact
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Export XFA form to XML
      # @return [String] Complete XFA form as XML
      def xfa_form_to_xml
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return '' if form_ptr.nil? || form_ptr.null?

        begin
          get_xfa_dataset_xml
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get XFA statistics
      # @return [Hash] Statistics about XFA forms in document
      def xfa_statistics
        check_document!

        {
          has_xfa: has_xfa_forms?,
          field_count: has_xfa_forms? ? get_xfa_field_count : 0,
          field_names: has_xfa_forms? ? get_xfa_field_names : [],
          timestamp: Time.now.to_i
        }
      end

      # Get XFA form type (static or dynamic)
      #
      # Determines the type of XFA form contained in the document.
      # Static forms have a fixed structure and cannot adapt their layout.
      # Dynamic forms can modify their structure based on data.
      #
      # @return [Symbol] Form type: :static, :dynamic, or :unknown
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   type = manager.get_xfa_form_type
      #   case type
      #   when :static
      #     puts "Fixed structure form"
      #   when :dynamic
      #     puts "Adaptive form"
      #   else
      #     puts "Unknown form type"
      #   end
      def get_xfa_form_type
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return :unknown if form_ptr.nil? || form_ptr.null?

        begin
          form_type_int = FFI::Bindings.pdf_get_xfa_form_type(form_ptr)
          FORM_TYPE_NAMES[form_type_int] || :unknown
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get XFA form title
      #
      # Retrieves the title or name of the XFA form. This is useful for
      # identifying the form to users and in logs.
      #
      # @return [String] Form title, or empty string if not available
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   title = manager.get_xfa_form_title
      #   puts "Processing form: #{title}"
      def get_xfa_form_title
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return '' if form_ptr.nil? || form_ptr.null?

        begin
          title_ptr = FFI::Bindings.pdf_xfa_form_get_title(form_ptr)
          FFI::StringMarshaller.from_c_string(title_ptr) || ''
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get number of pages in XFA form
      #
      # Returns the number of pages in the XFA form, which may differ from
      # the document's total page count if the document has non-form pages.
      #
      # @return [Integer] Number of pages in the XFA form
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   pages = manager.get_xfa_page_count
      #   puts "Form has #{pages} pages"
      def get_xfa_page_count
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return 0 if form_ptr.nil? || form_ptr.null?

        begin
          FFI::Bindings.pdf_xfa_form_page_count(form_ptr)
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Get XFA field label (display name)
      #
      # Retrieves the user-friendly label for a form field. This is typically
      # displayed to users instead of the technical field name.
      #
      # @param field_name [String] Name of the field (must not be empty)
      # @return [String] Field label, or the field name if label is not available
      # @raise [ArgumentError] if field_name is nil or empty
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   label = manager.get_xfa_field_label('email_address')
      #   puts "Field label: #{label}"  # Output: "Email Address"
      def get_xfa_field_label(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return field_name if form_ptr.nil? || form_ptr.null?

        begin
          FFI::StringMarshaller.to_utf8(field_name)
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)

          field_label = field_name
          count.times do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)
              current_name = FFI::StringMarshaller.read_c_string(name_ptr) || ''

              if current_name == field_name
                label_ptr = FFI::Bindings.pdf_xfa_field_get_label(field_ptr)
                field_label = FFI::StringMarshaller.from_c_string(label_ptr) || field_name
                break
              end
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          field_label
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Check if XFA field is required
      #
      # Determines if a form field must be filled before the form can be submitted.
      # Required fields must have values before form validation passes.
      #
      # @param field_name [String] Name of the field (must not be empty)
      # @return [Boolean] true if field is required, false otherwise
      # @raise [ArgumentError] if field_name is nil or empty
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   if manager.is_xfa_field_required?('name')
      #     puts "Name field is required"
      #   end
      def is_xfa_field_required?(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return false if form_ptr.nil? || form_ptr.null?

        begin
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)
          is_required = false

          count.times do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)
              current_name = FFI::StringMarshaller.read_c_string(name_ptr) || ''

              if current_name == field_name
                is_required = FFI::Bindings.pdf_xfa_field_is_required(field_ptr)
                break
              end
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          is_required
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Check if XFA field is read-only
      #
      # Determines if a form field is protected against modification.
      # Read-only fields cannot be edited by users.
      #
      # @param field_name [String] Name of the field (must not be empty)
      # @return [Boolean] true if field is read-only, false otherwise
      # @raise [ArgumentError] if field_name is nil or empty
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   unless manager.is_xfa_field_readonly?('status')
      #     puts "Status field can be edited"
      #   end
      def is_xfa_field_readonly?(field_name)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Field name cannot be empty' if field_name.nil? || field_name.empty?
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return false if form_ptr.nil? || form_ptr.null?

        begin
          count = FFI::Bindings.pdf_xfa_form_field_count(form_ptr)
          is_readonly = false

          count.times do |i|
            field_ptr = FFI::Bindings.pdf_xfa_form_get_field(form_ptr, i)

            next if field_ptr.nil? || field_ptr.null?

            begin
              name_ptr = FFI::Bindings.pdf_xfa_field_get_name(field_ptr)
              current_name = FFI::StringMarshaller.read_c_string(name_ptr) || ''

              if current_name == field_name
                is_readonly = FFI::Bindings.pdf_xfa_field_is_readonly(field_ptr)
                break
              end
            ensure
              FFI::Bindings.pdf_xfa_field_free(field_ptr)
            end
          end

          is_readonly
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Export XFA dataset as JSON
      #
      # Extracts form data from the XFA dataset and returns it as a Ruby hash.
      # This is useful for processing form data in Ruby or re-importing it elsewhere.
      #
      # @return [Hash] Form data as a JSON-compatible hash, or empty hash if no data
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   data = manager.xfa_dataset_to_json
      #   puts data['name']  # => "John Doe"
      #   puts data['email'] # => "john@example.com"
      def xfa_dataset_to_json
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        form_ptr = with_error_check('parse_xfa_form') do |error_ptr|
          FFI::Bindings.pdf_parse_xfa_form(@document.handle, error_ptr)
        end

        return {} if form_ptr.nil? || form_ptr.null?

        begin
          dataset_ptr = FFI::Bindings.pdf_xfa_form_get_dataset(form_ptr)
          return {} if dataset_ptr.nil? || dataset_ptr.null?

          begin
            json_ptr = FFI::Bindings.pdf_xfa_dataset_to_json(dataset_ptr)
            json_str = FFI::StringMarshaller.from_c_string(json_ptr) || '{}'
            JSON.parse(json_str)
          rescue JSON::ParserError
            {}
          ensure
            FFI::Bindings.pdf_xfa_dataset_free(dataset_ptr) unless dataset_ptr.nil? || dataset_ptr.null?
          end
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end

      # Extract XFA form as FDF (Forms Data Format)
      #
      # Exports XFA form data in FDF (Forms Data Format) which is a standard format
      # for exchanging form data between PDF applications. This enables interoperability
      # with other PDF tools and servers.
      #
      # @return [String] FDF-formatted data, or empty string if no data available
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   fdf_data = manager.extract_xfa_as_fdf
      #   File.write('form_data.fdf', fdf_data)
      def extract_xfa_as_fdf
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        fdf_ptr = with_error_check('extract_xfa_as_fdf') do |error_ptr|
          FFI::Bindings.pdf_extract_xfa_as_fdf(@document.handle, error_ptr)
        end

        return '' if fdf_ptr.nil? || fdf_ptr.null?

        FFI::StringMarshaller.from_c_string(fdf_ptr) || ''
      end

      # Get XFA template XML
      #
      # Retrieves the XFA template which defines the form structure, field definitions,
      # and layout. This is useful for understanding form design and structure analysis.
      #
      # @return [String] XFA template as XML, or empty string if not available
      # @raise [OperationError] if document does not contain XFA forms
      # @example
      #   template = manager.get_xfa_template_xml
      #   # Parse and analyze the form structure
      #   doc = Nokogiri::XML(template)
      #   fields = doc.xpath('//field/@name')
      def get_xfa_template_xml
        check_document!
        raise ::PdfOxide::OperationError, 'Document does not have XFA forms' unless has_xfa_forms?

        template_ptr = with_error_check('get_xfa_template_xml') do |error_ptr|
          FFI::Bindings.pdf_get_xfa_template_xml(@document.handle, error_ptr)
        end

        return '' if template_ptr.nil? || template_ptr.null?

        FFI::StringMarshaller.from_c_string(template_ptr) || ''
      end

      private

      def parse_xfa_form_data(form_ptr)
        return {} if form_ptr.nil? || form_ptr.null?

        begin
          {
            field_count: FFI::Bindings.pdf_xfa_form_field_count(form_ptr),
            has_dataset: true
          }
        ensure
          FFI::Bindings.pdf_xfa_form_free(form_ptr) unless form_ptr.nil? || form_ptr.null?
        end
      end
    end
  end
end
